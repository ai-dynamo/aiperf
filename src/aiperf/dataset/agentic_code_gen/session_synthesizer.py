# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Core session synthesis engine.

Generates multi-turn Agentic Code sessions using a state machine that models
context growth, mixture delays, and probabilistic resets.
"""

from __future__ import annotations

import math
import uuid

import numpy as np

from aiperf.dataset.agentic_code_gen.distributions import (
    sample_lognormal,
    sample_mixture_delay,
)
from aiperf.dataset.agentic_code_gen.models import (
    LognormalParams,
    SessionDistributionConfig,
    SessionEndReason,
    SynthesizedSession,
    SynthesizedTurn,
)
from aiperf.dataset.agentic_code_gen.prefix_model import PrefixAllocator

OUTPUT_MIN = 30


class SessionSynthesizer:
    """Synthesizes multi-turn sessions from distribution config.

    State machine per session:
        START -> derive initial_context (L1 + L1.5 + sampled L2) -> Turn 0
        TURN_LOOP:
            1. Sample delay (mixture: agentic 70% / human 30%)
            2. Sample new_tokens (lognormal)
            3. input_length = prev_input + prev_output + new_tokens
            4. Check RESET:
               a. input_length >= max_prompt_tokens -> forced retire
               b. P(reset) based on context scaling -> if triggered, end session
            5. Sample output_length (lognormal)
            6. Generate hash_ids (prefix_model)
            7. -> TURN_LOOP
    """

    def __init__(self, config: SessionDistributionConfig, seed: int = 42) -> None:
        self._config = config
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._allocator = PrefixAllocator(config.cache, config.block_size)
        self._session_counter = 0

        # Pre-compute Zipf weights for group assignment
        ng = config.cache.layer1_5_groups.num_groups
        weights = np.array(
            [
                1.0 / (k**config.cache.layer1_5_groups.zipf_alpha)
                for k in range(1, ng + 1)
            ]
        )
        self._group_weights = weights / weights.sum()

        # Fixed prefix size (L1 + L1.5)
        self._fixed_prefix = config.cache.layer1_tokens + config.cache.layer1_5_tokens

        # Output floor: respect config max if it's below the default minimum
        gen_max = config.generation_length.max
        self._output_min = (
            min(OUTPUT_MIN, int(gen_max)) if gen_max is not None else OUTPUT_MIN
        )

        # Pre-compute bias-corrected new_tokens params: shift mu by log(bias)
        # to compensate for right-tail truncation at context limit
        ntp = config.new_tokens_per_turn
        if ntp.bias != 1.0:
            shifted_mu = ntp.mu + math.log(ntp.bias)
            self._new_tokens_params = LognormalParams(
                mu=shifted_mu,
                sigma=ntp.sigma,
                mean=ntp.mean * ntp.bias,
                median=ntp.median * ntp.bias,
                min=ntp.min,
                max=ntp.max,
            )
        else:
            self._new_tokens_params = ntp

    @property
    def config(self) -> SessionDistributionConfig:
        return self._config

    @property
    def allocator(self) -> PrefixAllocator:
        return self._allocator

    def _next_session_index(self) -> int:
        idx = self._session_counter
        self._session_counter += 1
        return idx

    def _should_reset(self, input_length: int) -> bool:
        """Check probabilistic reset based on context utilization."""
        cfg = self._config.reset
        if cfg is None:
            return False
        ratio = input_length / self._config.max_prompt_tokens
        p = cfg.base_probability * (1.0 + (cfg.context_scaling - 1.0) * ratio)
        return bool(self._rng.random() < p)

    def _sample_group_id(self) -> int:
        return int(
            self._rng.choice(
                self._config.cache.layer1_5_groups.num_groups, p=self._group_weights
            )
        )

    def _sample_initial_context(self) -> int:
        l2_tokens = int(
            sample_lognormal(self._config.cache.layer2, self._rng, size=1)[0]
        )
        l2_tokens = max(l2_tokens, 1)
        return min(self._fixed_prefix + l2_tokens, self._config.max_prompt_tokens - 1)

    def _sample_output_length(self) -> int:
        return int(
            sample_lognormal(
                self._config.generation_length,
                self._rng,
                size=1,
                clip_min=self._output_min,
            )[0]
        )

    def _sample_delay_ms(self, prev_input: int) -> float:
        delay_ms = float(
            sample_mixture_delay(self._config.inter_turn_delay, self._rng, size=1)[0]
        )
        context_ratio = prev_input / self._config.max_prompt_tokens
        delay_ms *= max(0.2, 1.0 - 0.8 * context_ratio)
        if self._config.inter_turn_delay.max is not None:
            delay_ms = min(delay_ms, self._config.inter_turn_delay.max)
        return delay_ms

    def _sample_new_tokens(self) -> int:
        new_tokens = int(
            sample_lognormal(self._new_tokens_params, self._rng, size=1)[0]
        )
        return max(new_tokens, 1)

    def _sample_turn_target(self) -> int:
        turns_cfg = self._config.turns
        if turns_cfg is None:
            raise RuntimeError("explicit turn sampling requested without turns config")
        sampled = sample_lognormal(turns_cfg.to_lognormal(), self._rng, size=1)[0]
        target = round(sampled)
        return min(max(target, turns_cfg.min), turns_cfg.max)

    def _synthesize_explicit_turn_session(self) -> SynthesizedSession:
        turns_cfg = self._config.turns
        if turns_cfg is None:
            raise RuntimeError("explicit turn mode requested without turns config")

        target_turns = self._sample_turn_target()
        max_attempts = turns_cfg.max_session_attempts or 1
        for _ in range(max_attempts):
            session_index = self._next_session_index()
            rand_bytes = self._rng.bytes(16)
            session_id = f"sess-{uuid.UUID(bytes=rand_bytes).hex[:12]}"
            group_id = self._sample_group_id()

            initial_ctx = self._sample_initial_context()
            if initial_ctx >= self._config.max_prompt_tokens:
                continue

            output_len = self._sample_output_length()
            timestamp_ms = 0.0
            hash_ids = self._allocator.turn_hash_ids(
                session_index,
                group_id=group_id,
                input_length=initial_ctx,
                prev_session_ids=None,
            )
            turns: list[SynthesizedTurn] = [
                SynthesizedTurn(
                    turn_index=0,
                    input_length=initial_ctx,
                    output_length=output_len,
                    new_tokens=initial_ctx,
                    delay_ms=0.0,
                    timestamp_ms=timestamp_ms,
                    hash_ids=hash_ids,
                )
            ]

            prev_input = initial_ctx
            prev_output = output_len
            realized = True
            for turn_idx in range(1, target_turns):
                delay_ms = self._sample_delay_ms(prev_input)
                timestamp_ms += delay_ms

                new_tokens = self._sample_new_tokens()
                input_length = prev_input + prev_output + new_tokens
                if input_length >= self._config.max_prompt_tokens:
                    if turns_cfg.allow_truncation:
                        return SynthesizedSession(
                            session_id=session_id,
                            group_id=group_id,
                            turns=turns,
                            end_reason=SessionEndReason.FORCED_RETIRE,
                        )
                    realized = False
                    break

                output_len = self._sample_output_length()
                prev_session = self._allocator.extract_session_ids(turns[-1].hash_ids)
                hash_ids = self._allocator.turn_hash_ids(
                    session_index,
                    group_id=group_id,
                    input_length=input_length,
                    prev_session_ids=prev_session,
                )
                turns.append(
                    SynthesizedTurn(
                        turn_index=turn_idx,
                        input_length=input_length,
                        output_length=output_len,
                        new_tokens=new_tokens,
                        delay_ms=delay_ms,
                        timestamp_ms=timestamp_ms,
                        hash_ids=hash_ids,
                    )
                )
                prev_input = input_length
                prev_output = output_len

            if realized:
                return SynthesizedSession(
                    session_id=session_id,
                    group_id=group_id,
                    turns=turns,
                    end_reason=SessionEndReason.TARGET_TURN_COUNT,
                )

        raise RuntimeError(
            "Failed to synthesize explicit-turn session for "
            f"target_turns={target_turns} after {max_attempts} attempts "
            f"with max_prompt_tokens={self._config.max_prompt_tokens}"
        )

    def synthesize_session(
        self, inject_restart: bool = False
    ) -> list[SynthesizedSession]:
        """Generate a single multi-turn session, possibly split at a restart point.

        If inject_restart is True, the session splits into two:
        - Session A: turns [0..restart_at_turn-1], end_reason=RESTART_SPLIT
        - Session B: new session_id, same group_id, turn 0 inherits A's last
          accumulated tokens and hash_ids, then continues generating.

        Returns a list: [session] normally, [session_a, session_b] on restart.
        """
        if self._config.turns is not None:
            return [self._synthesize_explicit_turn_session()]

        session_index = self._next_session_index()
        rand_bytes = self._rng.bytes(16)
        session_id = f"sess-{uuid.UUID(bytes=rand_bytes).hex[:12]}"
        turns: list[SynthesizedTurn] = []

        lo, hi = self._config.restart_turn_range
        restart_at_turn = int(self._rng.integers(lo, hi)) if inject_restart else -1

        group_id = self._sample_group_id()

        # Turn 0: derive initial_context = L1 + L1.5 + sampled L2
        initial_ctx = self._sample_initial_context()
        output_len = self._sample_output_length()

        timestamp_ms = 0.0
        hash_ids = self._allocator.turn_hash_ids(
            session_index,
            group_id=group_id,
            input_length=initial_ctx,
            prev_session_ids=None,
        )

        turns.append(
            SynthesizedTurn(
                turn_index=0,
                input_length=initial_ctx,
                output_length=output_len,
                new_tokens=initial_ctx,
                delay_ms=0.0,
                timestamp_ms=timestamp_ms,
                hash_ids=hash_ids,
            )
        )

        prev_input = initial_ctx
        prev_output = output_len

        turn_idx = 1
        end_reason = SessionEndReason.FORCED_RETIRE
        while True:
            # Split into two sessions at restart turn
            if turn_idx == restart_at_turn:
                session_a = SynthesizedSession(
                    session_id=session_id,
                    group_id=group_id,
                    turns=turns,
                    end_reason=SessionEndReason.RESTART_SPLIT,
                )
                session_b = self._synthesize_continuation(
                    session_index=session_index,
                    group_id=group_id,
                    prev_input=prev_input,
                    prev_output=prev_output,
                    prev_hash_ids=turns[-1].hash_ids,
                )
                return [session_a, session_b]

            # 1. Sample delay
            delay_ms = self._sample_delay_ms(prev_input)
            timestamp_ms += delay_ms

            # 2. Sample new tokens (bias-corrected for truncation)
            new_tokens = self._sample_new_tokens()

            # 3. Compute input length
            input_length = prev_input + prev_output + new_tokens

            # 4a. Forced retire if over context limit
            if input_length >= self._config.max_prompt_tokens:
                end_reason = SessionEndReason.FORCED_RETIRE
                break

            # 4b. Probabilistic reset
            if self._should_reset(input_length):
                end_reason = SessionEndReason.PROBABILISTIC_RESET
                break

            # 5. Sample output length
            output_len = self._sample_output_length()

            # 6. Generate hash_ids (extend previous session ids)
            prev_session = self._allocator.extract_session_ids(turns[-1].hash_ids)
            hash_ids = self._allocator.turn_hash_ids(
                session_index,
                group_id=group_id,
                input_length=input_length,
                prev_session_ids=prev_session,
            )

            turns.append(
                SynthesizedTurn(
                    turn_index=turn_idx,
                    input_length=input_length,
                    output_length=output_len,
                    new_tokens=new_tokens,
                    delay_ms=delay_ms,
                    timestamp_ms=timestamp_ms,
                    hash_ids=hash_ids,
                )
            )

            prev_input = input_length
            prev_output = output_len
            turn_idx += 1

        return [
            SynthesizedSession(
                session_id=session_id,
                group_id=group_id,
                turns=turns,
                end_reason=end_reason,
            )
        ]

    def _synthesize_continuation(
        self,
        session_index: int,
        group_id: int,
        prev_input: int,
        prev_output: int,
        prev_hash_ids: list[int],
    ) -> SynthesizedSession:
        """Create Session B: a continuation after a restart split.

        Turn 0 carries all accumulated tokens from Session A's last turn
        and extends A's hash_ids to cover the full initial context.
        """
        rand_bytes = self._rng.bytes(16)
        session_id = f"sess-{uuid.UUID(bytes=rand_bytes).hex[:12]}"

        initial_input = prev_input + prev_output
        initial_input = min(initial_input, self._config.max_prompt_tokens - 1)

        output_len = self._sample_output_length()

        prev_session_ids = self._allocator.extract_session_ids(prev_hash_ids)
        hash_ids = self._allocator.turn_hash_ids(
            session_index,
            group_id=group_id,
            input_length=initial_input,
            prev_session_ids=prev_session_ids,
        )

        turns: list[SynthesizedTurn] = [
            SynthesizedTurn(
                turn_index=0,
                input_length=initial_input,
                output_length=output_len,
                new_tokens=initial_input,
                delay_ms=0.0,
                timestamp_ms=0.0,
                hash_ids=hash_ids,
            )
        ]

        prev_input_b = initial_input
        prev_output_b = output_len
        turn_idx = 1
        end_reason = SessionEndReason.FORCED_RETIRE

        while True:
            delay_ms = self._sample_delay_ms(prev_input_b)
            timestamp_ms = turns[-1].timestamp_ms + delay_ms

            new_tokens = self._sample_new_tokens()

            input_length = prev_input_b + prev_output_b + new_tokens

            if input_length >= self._config.max_prompt_tokens:
                end_reason = SessionEndReason.FORCED_RETIRE
                break

            if self._should_reset(input_length):
                end_reason = SessionEndReason.PROBABILISTIC_RESET
                break

            output_len = self._sample_output_length()

            prev_session = self._allocator.extract_session_ids(turns[-1].hash_ids)
            hash_ids = self._allocator.turn_hash_ids(
                session_index,
                group_id=group_id,
                input_length=input_length,
                prev_session_ids=prev_session,
            )

            turns.append(
                SynthesizedTurn(
                    turn_index=turn_idx,
                    input_length=input_length,
                    output_length=output_len,
                    new_tokens=new_tokens,
                    delay_ms=delay_ms,
                    timestamp_ms=timestamp_ms,
                    hash_ids=hash_ids,
                )
            )

            prev_input_b = input_length
            prev_output_b = output_len
            turn_idx += 1

        return SynthesizedSession(
            session_id=session_id,
            group_id=group_id,
            turns=turns,
            end_reason=end_reason,
            is_restart_continuation=True,
        )

    def synthesize_sessions(self, num_sessions: int) -> list[SynthesizedSession]:
        """Generate multiple sessions with optional restart splits.

        Restart probability decreases linearly from restart_initial_probability
        to 0 over the first 75% of sessions. Session B's (restart continuations)
        are scattered randomly into the back portion of the queue, starting after
        25% of primary sessions to ensure they never overlap with their Session A
        in the concurrency window.
        """
        if self._config.turns is not None:
            return [
                self.synthesize_session(inject_restart=False)[0]
                for _ in range(num_sessions)
            ]

        restart_probability = self._config.restart_initial_probability
        cutoff = 0.75
        primary: list[SynthesizedSession] = []
        deferred: list[tuple[SynthesizedSession, int]] = []
        for i in range(num_sessions):
            progress = i / max(1, num_sessions - 1)
            if progress >= cutoff:
                p_restart = 0.0
            else:
                p_restart = restart_probability * (1.0 - progress / cutoff)
            inject = float(self._rng.random()) < p_restart
            result = self.synthesize_session(inject_restart=inject)
            origin_index = len(primary)
            primary.append(result[0])
            if len(result) > 1:
                deferred.extend((session, origin_index) for session in result[1:])

        if not deferred:
            return primary

        # Scatter deferred sessions into the back portion of the queue.
        # min_offset ensures Session B never shares a concurrency window
        # with its Session A (restarts only fire in first 75%).
        min_offset = max(1, int(num_sessions * 0.25))
        for session_b, origin_index in deferred:
            low = min(origin_index + min_offset, len(primary))
            pos = (
                len(primary)
                if low >= len(primary)
                else int(self._rng.integers(low, len(primary) + 1))
            )
            primary.insert(pos, session_b)

        return primary
