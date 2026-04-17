# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the session synthesizer."""

from __future__ import annotations

import math

import numpy as np
import pytest

from aiperf.dataset.agentic_code_gen.distributions import lognormal_from_mean_median
from aiperf.dataset.agentic_code_gen.models import (
    CacheLayerConfig,
    Layer15GroupConfig,
    LognormalParams,
    MixtureDelayConfig,
    ResetConfig,
    SessionDistributionConfig,
    SessionEndReason,
    SynthesizedSession,
    TurnCountConfig,
)
from aiperf.dataset.agentic_code_gen.session_synthesizer import SessionSynthesizer


class TestSessionSynthesizer:
    def test_reproducible_with_same_seed(
        self, coding_config: SessionDistributionConfig
    ) -> None:
        s1 = SessionSynthesizer(coding_config, seed=42)
        s2 = SessionSynthesizer(coding_config, seed=42)
        session_1 = s1.synthesize_session()[0]
        session_2 = s2.synthesize_session()[0]
        assert session_1.session_id == session_2.session_id
        assert len(session_1.turns) == len(session_2.turns)
        for t1, t2 in zip(session_1.turns, session_2.turns, strict=True):
            assert t1.input_length == t2.input_length
            assert t1.output_length == t2.output_length

    def test_different_seeds_produce_different_sessions(
        self, coding_config: SessionDistributionConfig
    ) -> None:
        s1 = SessionSynthesizer(coding_config, seed=42)
        s2 = SessionSynthesizer(coding_config, seed=99)
        session_1 = s1.synthesize_session()[0]
        session_2 = s2.synthesize_session()[0]
        assert session_1.session_id != session_2.session_id

    def test_turn_indices_sequential(
        self, coding_config: SessionDistributionConfig
    ) -> None:
        synth = SessionSynthesizer(coding_config, seed=42)
        session = synth.synthesize_session()[0]
        for i, turn in enumerate(session.turns):
            assert turn.turn_index == i

    def test_input_length_grows(self, coding_config: SessionDistributionConfig) -> None:
        synth = SessionSynthesizer(coding_config, seed=42)
        session = synth.synthesize_session()[0]
        if len(session.turns) > 1:
            for i in range(1, len(session.turns)):
                assert session.turns[i].input_length > session.turns[i - 1].input_length

    def test_hash_ids_prefix_property(
        self, coding_config: SessionDistributionConfig
    ) -> None:
        synth = SessionSynthesizer(coding_config, seed=42)
        session = synth.synthesize_session()[0]
        for i in range(1, len(session.turns)):
            prev_ids = session.turns[i - 1].hash_ids
            curr_ids = session.turns[i].hash_ids
            assert curr_ids[: len(prev_ids)] == prev_ids

    def test_l1_ids_consistent_across_sessions(
        self, coding_config: SessionDistributionConfig
    ) -> None:
        synth = SessionSynthesizer(coding_config, seed=42)
        sessions = synth.synthesize_sessions(3)
        l1_blocks = synth.allocator.l1_blocks
        canonical_l1 = list(range(l1_blocks))
        for session in sessions:
            ids = session.turns[0].hash_ids
            l1_used = min(l1_blocks, len(ids))
            assert ids[:l1_used] == canonical_l1[:l1_used]

    def test_context_stays_under_max(
        self, coding_config: SessionDistributionConfig
    ) -> None:
        synth = SessionSynthesizer(coding_config, seed=42)
        session = synth.synthesize_session()[0]
        for turn in session.turns:
            assert turn.input_length < coding_config.max_prompt_tokens

    def test_output_length_clipped_at_minimum(
        self, coding_config: SessionDistributionConfig
    ) -> None:
        synth = SessionSynthesizer(coding_config, seed=42)
        sessions = synth.synthesize_sessions(10)
        for session in sessions:
            for turn in session.turns:
                assert turn.output_length >= 30

    def test_first_turn_has_zero_delay(
        self, coding_config: SessionDistributionConfig
    ) -> None:
        synth = SessionSynthesizer(coding_config, seed=42)
        session = synth.synthesize_session()[0]
        assert session.turns[0].delay_ms == 0.0

    def test_subsequent_turns_have_positive_delay(
        self, coding_config: SessionDistributionConfig
    ) -> None:
        synth = SessionSynthesizer(coding_config, seed=42)
        session = synth.synthesize_session()[0]
        for turn in session.turns[1:]:
            assert turn.delay_ms > 0

    def test_timestamps_monotonically_increase(
        self, coding_config: SessionDistributionConfig
    ) -> None:
        synth = SessionSynthesizer(coding_config, seed=42)
        session = synth.synthesize_session()[0]
        for i in range(1, len(session.turns)):
            assert session.turns[i].timestamp_ms > session.turns[i - 1].timestamp_ms

    def test_multiple_sessions_have_unique_ids(
        self, coding_config: SessionDistributionConfig
    ) -> None:
        synth = SessionSynthesizer(coding_config, seed=42)
        sessions = synth.synthesize_sessions(20)
        ids = [s.session_id for s in sessions]
        assert len(set(ids)) == len(ids)

    def test_end_reason_is_set(self, coding_config: SessionDistributionConfig) -> None:
        synth = SessionSynthesizer(coding_config, seed=42)
        sessions = synth.synthesize_sessions(20)
        for session in sessions:
            assert session.end_reason in (
                SessionEndReason.FORCED_RETIRE,
                SessionEndReason.PROBABILISTIC_RESET,
                SessionEndReason.RESTART_SPLIT,
            )


class TestSessionSynthesizerSmallConfig:
    def test_forced_retire_at_context_limit(
        self, small_config: SessionDistributionConfig
    ) -> None:
        synth = SessionSynthesizer(small_config, seed=42)
        sessions = synth.synthesize_sessions(50)
        for session in sessions:
            for turn in session.turns:
                assert turn.input_length < small_config.max_prompt_tokens

    def test_sessions_have_at_least_one_turn(
        self, small_config: SessionDistributionConfig
    ) -> None:
        synth = SessionSynthesizer(small_config, seed=42)
        sessions = synth.synthesize_sessions(50)
        for session in sessions:
            assert len(session.turns) >= 1


class TestExplicitTurnMode:
    def _overflowing_turn_mode_config(
        self, allow_truncation: bool
    ) -> SessionDistributionConfig:
        return SessionDistributionConfig(
            new_tokens_per_turn=lognormal_from_mean_median(mean=2_000, median=2_000),
            generation_length=lognormal_from_mean_median(mean=1, median=1),
            inter_turn_delay=MixtureDelayConfig(
                agentic_fraction=0.7,
                agentic_delay=lognormal_from_mean_median(mean=3_000, median=2_000),
                human_delay=lognormal_from_mean_median(mean=45_000, median=30_000),
            ),
            turns=TurnCountConfig(
                mean=3,
                median=3,
                min=3,
                max=3,
                allow_truncation=allow_truncation,
                max_session_attempts=None if allow_truncation else 2,
            ),
            max_prompt_tokens=3_500,
            block_size=64,
            cache=CacheLayerConfig(
                layer1_tokens=100,
                layer1_5_tokens=50,
                layer2=lognormal_from_mean_median(mean=1_000, median=1_000),
                layer1_5_groups=Layer15GroupConfig(num_groups=5, zipf_alpha=1.2),
            ),
        )

    def test_sessions_match_exact_target_turn_count(
        self, turns_config: SessionDistributionConfig
    ) -> None:
        synth = SessionSynthesizer(turns_config, seed=42)
        sessions = synth.synthesize_sessions(20)
        assert all(len(session.turns) == 4 for session in sessions)

    def test_sessions_end_with_target_turn_reason(
        self, turns_config: SessionDistributionConfig
    ) -> None:
        synth = SessionSynthesizer(turns_config, seed=42)
        sessions = synth.synthesize_sessions(10)
        assert all(
            session.end_reason == SessionEndReason.TARGET_TURN_COUNT
            for session in sessions
        )

    def test_turn_mode_disables_restart_splitting(
        self, turns_config: SessionDistributionConfig
    ) -> None:
        synth = SessionSynthesizer(turns_config, seed=42)
        sessions = synth.synthesize_sessions(10)
        assert all(not session.is_restart_continuation for session in sessions)

    def test_retry_exhaustion_raises_runtime_error(self) -> None:
        config = self._overflowing_turn_mode_config(allow_truncation=False)
        synth = SessionSynthesizer(config, seed=42)
        with pytest.raises(RuntimeError, match="target_turns=3"):
            synth.synthesize_session()

    def test_allow_truncation_returns_partial_session(self) -> None:
        config = self._overflowing_turn_mode_config(allow_truncation=True)
        synth = SessionSynthesizer(config, seed=42)
        session = synth.synthesize_session()[0]
        assert len(session.turns) == 2
        assert session.end_reason == SessionEndReason.FORCED_RETIRE

    @pytest.mark.parametrize("allow_truncation", [False, True])
    def test_allow_truncation_flag_controls_overflow_behavior(
        self, allow_truncation: bool
    ) -> None:
        config = self._overflowing_turn_mode_config(allow_truncation=allow_truncation)
        synth = SessionSynthesizer(config, seed=42)

        if allow_truncation:
            session = synth.synthesize_session()[0]
            assert len(session.turns) == 2
            assert session.end_reason == SessionEndReason.FORCED_RETIRE
        else:
            with pytest.raises(RuntimeError, match="target_turns=3"):
                synth.synthesize_session()


class TestTurnModeValidation:
    def test_turns_mode_rejects_reset(self) -> None:
        with pytest.raises(
            ValueError, match="turns mode cannot be combined with reset"
        ):
            SessionDistributionConfig(
                turns=TurnCountConfig(mean=4, median=4, min=4, max=4),
                reset=ResetConfig(base_probability=0.02, context_scaling=2.0),
            )

    def test_turns_mode_rejects_restart_initial_probability(self) -> None:
        with pytest.raises(
            ValueError,
            match="turns mode cannot be combined with restart_initial_probability",
        ):
            SessionDistributionConfig(
                turns=TurnCountConfig(mean=4, median=4, min=4, max=4),
                restart_initial_probability=0.1,
            )

    def test_deprecated_restart_fraction_maps_to_initial_probability(self) -> None:
        config = SessionDistributionConfig(restart_fraction=0.1)
        assert config.restart_initial_probability == 0.1
        assert "restart_fraction" not in config.model_dump()

    def test_restart_probability_alias_rejects_conflicting_values(self) -> None:
        with pytest.raises(ValueError, match="restart_fraction cannot differ"):
            SessionDistributionConfig(
                restart_fraction=0.1,
                restart_initial_probability=0.2,
            )

    def test_turns_null_keeps_default_reset_mode(self) -> None:
        config = SessionDistributionConfig(turns=None)
        assert config.turns is None
        assert config.reset is not None

    @pytest.mark.parametrize("restart_turn_range", [[0, 5], [5, 5], [6, 5]])
    def test_restart_turn_range_invalid_values_raise(
        self, restart_turn_range: list[int]
    ) -> None:
        with pytest.raises(ValueError, match="restart_turn_range"):
            SessionDistributionConfig(restart_turn_range=restart_turn_range)

    def test_turns_mode_rejects_impossible_minimum(self) -> None:
        with pytest.raises(ValueError, match="minimum turn count cannot fit"):
            SessionDistributionConfig(
                new_tokens_per_turn=lognormal_from_mean_median(mean=100, median=100),
                generation_length=lognormal_from_mean_median(mean=50, median=30),
                turns=TurnCountConfig(mean=3, median=3, min=3, max=3),
                max_prompt_tokens=400,
                block_size=64,
                cache=CacheLayerConfig(
                    layer1_tokens=100,
                    layer1_5_tokens=50,
                    layer2=LognormalParams(mean=200, median=200, min=200),
                    layer1_5_groups=Layer15GroupConfig(num_groups=5, zipf_alpha=1.2),
                ),
            )

    def test_turns_mode_allows_impossible_minimum_with_truncation(self) -> None:
        config = SessionDistributionConfig(
            new_tokens_per_turn=lognormal_from_mean_median(mean=100, median=100),
            generation_length=lognormal_from_mean_median(mean=50, median=30),
            turns=TurnCountConfig(
                mean=3,
                median=3,
                min=3,
                max=3,
                allow_truncation=True,
            ),
            max_prompt_tokens=400,
            block_size=64,
            cache=CacheLayerConfig(
                layer1_tokens=100,
                layer1_5_tokens=50,
                layer2=LognormalParams(mean=200, median=200, min=200),
                layer1_5_groups=Layer15GroupConfig(num_groups=5, zipf_alpha=1.2),
            ),
        )
        assert config.turns is not None
        assert config.turns.allow_truncation is True
        assert config.turns.max_session_attempts is None

    def test_turns_mode_rejects_attempts_with_truncation(self) -> None:
        with pytest.raises(
            ValueError,
            match="max_session_attempts cannot be set when allow_truncation is true",
        ):
            TurnCountConfig(
                mean=3,
                median=3,
                min=3,
                max=3,
                allow_truncation=True,
                max_session_attempts=2,
            )


class TestMaxIsl:
    def test_no_turn_exceeds_max_prompt_tokens(
        self, small_config: SessionDistributionConfig
    ) -> None:
        """Turn 0 initial_ctx is clipped to max_prompt_tokens."""
        synth = SessionSynthesizer(small_config, seed=42)
        sessions = synth.synthesize_sessions(200)
        for session in sessions:
            for turn in session.turns:
                assert turn.input_length <= small_config.max_prompt_tokens

    def test_max_isl_override_clips_sessions(self) -> None:
        """Simulates --max-isl by lowering max_prompt_tokens."""
        base = SessionDistributionConfig(
            new_tokens_per_turn=lognormal_from_mean_median(mean=200, median=100),
            generation_length=lognormal_from_mean_median(mean=50, median=30),
            inter_turn_delay=MixtureDelayConfig(
                agentic_fraction=0.7,
                agentic_delay=lognormal_from_mean_median(mean=3_000, median=2_000),
                human_delay=lognormal_from_mean_median(mean=45_000, median=30_000),
            ),
            reset=ResetConfig(base_probability=0.02, context_scaling=2.0),
            max_prompt_tokens=50_000,
            block_size=64,
            cache=CacheLayerConfig(
                layer1_tokens=100,
                layer1_5_tokens=50,
                layer2=lognormal_from_mean_median(mean=4_000, median=3_000),
                layer1_5_groups=Layer15GroupConfig(num_groups=5, zipf_alpha=1.2),
            ),
        )
        max_isl = 2_000
        clipped = base.__class__.model_validate(
            {**base.model_dump(), "max_prompt_tokens": max_isl}
        )
        synth = SessionSynthesizer(clipped, seed=42)
        sessions = synth.synthesize_sessions(200)
        for session in sessions:
            for turn in session.turns:
                assert turn.input_length <= max_isl


class TestInitialContextFloor:
    def test_initial_context_exceeds_layer1_tokens(
        self, small_config: SessionDistributionConfig
    ) -> None:
        """The synthesizer floors initial_ctx to layer1_tokens + 1."""
        synth = SessionSynthesizer(small_config, seed=42)
        sessions = synth.synthesize_sessions(100)
        l1_tokens = small_config.cache.layer1_tokens
        for session in sessions:
            assert session.turns[0].input_length > l1_tokens

    def test_turn0_block_count_ge_l1_blocks(
        self, small_config: SessionDistributionConfig
    ) -> None:
        """Turn 0 should have at least as many blocks as L1 requires."""
        synth = SessionSynthesizer(small_config, seed=42)
        sessions = synth.synthesize_sessions(50)
        alloc = synth.allocator
        for session in sessions:
            assert len(session.turns[0].hash_ids) >= alloc.l1_blocks


class TestRestartScattering:
    def test_restart_continuations_appear_after_origin_sessions(self) -> None:
        config = SessionDistributionConfig(
            reset=ResetConfig(base_probability=0.0, context_scaling=1.0),
            restart_initial_probability=1.0,
            restart_turn_range=[1, 2],
        )
        synth = SessionSynthesizer(config, seed=42)
        sessions = synth.synthesize_sessions(20)
        restart_origins = [
            (idx, session)
            for idx, session in enumerate(sessions)
            if session.end_reason == SessionEndReason.RESTART_SPLIT
        ]
        continuations = [
            (idx, session)
            for idx, session in enumerate(sessions)
            if session.is_restart_continuation
        ]
        assert continuations

        for continuation_idx, continuation in continuations:
            matches = [
                origin_idx
                for origin_idx, origin in restart_origins
                if origin.group_id == continuation.group_id
                and continuation.turns[0].hash_ids[: len(origin.turns[-1].hash_ids)]
                == origin.turns[-1].hash_ids
            ]
            assert matches
            assert continuation_idx > max(matches)


class TestGroupAssignment:
    def test_group_ids_within_range(
        self, coding_config: SessionDistributionConfig
    ) -> None:
        """Every session's group_id must be in [0, num_groups)."""
        synth = SessionSynthesizer(coding_config, seed=42)
        sessions = synth.synthesize_sessions(100)
        num_groups = coding_config.cache.layer1_5_groups.num_groups
        for session in sessions:
            assert 0 <= session.group_id < num_groups

    def test_zipf_distribution_is_skewed(
        self, coding_config: SessionDistributionConfig
    ) -> None:
        """Group 0 should appear more often than uniform (Zipf skew)."""
        synth = SessionSynthesizer(coding_config, seed=42)
        sessions = synth.synthesize_sessions(500)
        group_counts = np.bincount(
            [s.group_id for s in sessions],
            minlength=coding_config.cache.layer1_5_groups.num_groups,
        )
        uniform_expected = 500 / coding_config.cache.layer1_5_groups.num_groups
        assert group_counts[0] > uniform_expected * 2, (
            f"Group 0 count {group_counts[0]} not significantly above "
            f"uniform expectation {uniform_expected:.0f}"
        )

    def test_multiple_groups_used(
        self, coding_config: SessionDistributionConfig
    ) -> None:
        """With 500 sessions and 50 groups, at least 10 distinct groups should appear."""
        synth = SessionSynthesizer(coding_config, seed=42)
        sessions = synth.synthesize_sessions(500)
        distinct_groups = len({s.group_id for s in sessions})
        assert distinct_groups >= 10

    def test_same_group_shares_l15_blocks(
        self, coding_config: SessionDistributionConfig
    ) -> None:
        """Sessions in the same group must share identical L1.5 hash IDs."""
        synth = SessionSynthesizer(coding_config, seed=42)
        sessions = synth.synthesize_sessions(50)
        alloc = synth.allocator
        l1 = alloc.l1_blocks
        l15 = alloc.l15_blocks

        by_group: dict[int, list] = {}
        for s in sessions:
            by_group.setdefault(s.group_id, []).append(s)

        for group_id, group_sessions in by_group.items():
            if len(group_sessions) < 2:
                continue
            ref = group_sessions[0].turns[0].hash_ids[l1 : l1 + l15]
            for s in group_sessions[1:]:
                actual = s.turns[0].hash_ids[l1 : l1 + l15]
                assert actual == ref, (
                    f"Group {group_id}: L1.5 mismatch between "
                    f"{group_sessions[0].session_id} and {s.session_id}"
                )

    def test_different_groups_have_different_l15_blocks(
        self, coding_config: SessionDistributionConfig
    ) -> None:
        """Sessions in different groups must have different L1.5 hash IDs."""
        synth = SessionSynthesizer(coding_config, seed=42)
        sessions = synth.synthesize_sessions(50)
        alloc = synth.allocator
        l1 = alloc.l1_blocks
        l15 = alloc.l15_blocks

        by_group: dict[int, list] = {}
        for s in sessions:
            by_group.setdefault(s.group_id, []).append(s)

        group_ids = list(by_group.keys())
        if len(group_ids) >= 2:
            s_a = by_group[group_ids[0]][0]
            s_b = by_group[group_ids[1]][0]
            l15_a = s_a.turns[0].hash_ids[l1 : l1 + l15]
            l15_b = s_b.turns[0].hash_ids[l1 : l1 + l15]
            assert l15_a != l15_b


class TestDistributionFidelity:
    def test_initial_context_mean_within_tolerance(
        self, coding_config: SessionDistributionConfig
    ) -> None:
        synth = SessionSynthesizer(coding_config, seed=42)
        sessions = synth.synthesize_sessions(500)
        initial_contexts = [s.turns[0].input_length for s in sessions]
        observed_mean = np.mean(initial_contexts)
        cache = coding_config.cache
        target_mean = cache.layer1_tokens + cache.layer1_5_tokens + cache.layer2.mean
        pct_error = abs(observed_mean - target_mean) / target_mean * 100
        assert pct_error < 10, (
            f"Initial context mean {observed_mean:.0f} vs target {target_mean:.0f} ({pct_error:.1f}%)"
        )

    def test_generation_length_mean_within_tolerance(
        self, coding_config: SessionDistributionConfig
    ) -> None:
        synth = SessionSynthesizer(coding_config, seed=42)
        sessions = synth.synthesize_sessions(500)
        output_lens = [t.output_length for s in sessions for t in s.turns]
        observed_mean = np.mean(output_lens)
        target_mean = coding_config.generation_length.mean
        pct_error = abs(observed_mean - target_mean) / target_mean * 100
        assert pct_error < 15, (
            f"Generation length mean {observed_mean:.0f} vs target {target_mean:.0f} ({pct_error:.1f}%)"
        )


class TestRestartContinuation:
    """Coverage for Session B produced by a restart split.

    Regression surface: Session B turn 0 previously reused Session A's last
    hash_ids, which were sized for prev_input. Session B's turn 0 input is
    prev_input + prev_output, so the array was undersized and tripped the
    final-block-size check in generator/prompt.py with errors like
    'final hash block size: 972 must be <= 512'.
    """

    def _assert_block_size_invariant(self, turn, block_size: int, context: str) -> None:
        expected_blocks = (
            math.ceil(turn.input_length / block_size) if turn.input_length > 0 else 0
        )
        assert len(turn.hash_ids) == expected_blocks, (
            f"{context}: hash_ids count {len(turn.hash_ids)} != "
            f"ceil({turn.input_length}/{block_size}) = {expected_blocks}"
        )
        final_block_size = turn.input_length - (len(turn.hash_ids) - 1) * block_size
        assert 0 < final_block_size <= block_size, (
            f"{context}: final block size {final_block_size} violates "
            f"0 < x <= {block_size} (prompt.py sanity check)"
        )

    def test_inject_restart_produces_session_a_and_b(
        self, coding_config: SessionDistributionConfig
    ) -> None:
        synth = SessionSynthesizer(coding_config, seed=42)
        sessions = synth.synthesize_session(inject_restart=True)
        assert len(sessions) == 2
        session_a, session_b = sessions
        assert session_a.end_reason == SessionEndReason.RESTART_SPLIT
        assert not session_a.is_restart_continuation
        assert session_b.is_restart_continuation
        assert session_a.group_id == session_b.group_id

    def test_continuation_turn0_hash_ids_sized_for_initial_input(
        self, coding_config: SessionDistributionConfig
    ) -> None:
        synth = SessionSynthesizer(coding_config, seed=42)
        _, session_b = synth.synthesize_session(inject_restart=True)
        turn0 = session_b.turns[0]
        self._assert_block_size_invariant(
            turn0, coding_config.block_size, "Session B turn 0"
        )

    def test_continuation_all_turns_pass_prompt_sanity_check(
        self, coding_config: SessionDistributionConfig
    ) -> None:
        synth = SessionSynthesizer(coding_config, seed=42)
        _, session_b = synth.synthesize_session(inject_restart=True)
        for turn in session_b.turns:
            self._assert_block_size_invariant(
                turn,
                coding_config.block_size,
                f"Session B turn {turn.turn_index}",
            )

    def test_continuation_initial_input_equals_prev_input_plus_output(
        self, coding_config: SessionDistributionConfig
    ) -> None:
        synth = SessionSynthesizer(coding_config, seed=42)
        session_a, session_b = synth.synthesize_session(inject_restart=True)
        a_last = session_a.turns[-1]
        expected = min(
            a_last.input_length + a_last.output_length,
            coding_config.max_prompt_tokens,
        )
        assert session_b.turns[0].input_length == expected

    def test_continuation_preserves_session_id_continuity(
        self, coding_config: SessionDistributionConfig
    ) -> None:
        """Session B turn 0's session IDs must extend Session A's last turn
        session IDs, so the KV cache blocks from A are still referenced."""
        synth = SessionSynthesizer(coding_config, seed=42)
        session_a, session_b = synth.synthesize_session(inject_restart=True)
        alloc = synth.allocator
        a_last_session_ids = alloc.extract_session_ids(session_a.turns[-1].hash_ids)
        b_turn0_session_ids = alloc.extract_session_ids(session_b.turns[0].hash_ids)
        assert len(b_turn0_session_ids) >= len(a_last_session_ids)
        assert b_turn0_session_ids[: len(a_last_session_ids)] == a_last_session_ids

    def test_continuation_preserves_shared_prefix_across_a_and_b(
        self, coding_config: SessionDistributionConfig
    ) -> None:
        """L1 + L1.5 must be identical between Session A's last turn and B's turn 0."""
        synth = SessionSynthesizer(coding_config, seed=42)
        session_a, session_b = synth.synthesize_session(inject_restart=True)
        alloc = synth.allocator
        prefix_blocks = alloc.prefix_blocks
        a_prefix = session_a.turns[-1].hash_ids[:prefix_blocks]
        b_prefix = session_b.turns[0].hash_ids[:prefix_blocks]
        assert a_prefix == b_prefix

    @pytest.mark.parametrize("seed", [0, 1, 7, 42, 99, 2026])
    def test_continuation_block_size_invariant_across_seeds(
        self, coding_config: SessionDistributionConfig, seed: int
    ) -> None:
        """Across many seeds the restart continuation must never produce
        an undersized or oversized hash_ids array."""
        synth = SessionSynthesizer(coding_config, seed=seed)
        sessions = synth.synthesize_session(inject_restart=True)
        if len(sessions) < 2:
            pytest.skip("Restart did not split for this seed")
        _, session_b = sessions
        block_size = coding_config.block_size
        for turn in session_b.turns:
            self._assert_block_size_invariant(
                turn,
                block_size,
                f"seed={seed} turn {turn.turn_index}",
            )

    def test_bulk_restart_sessions_all_valid(
        self, coding_config: SessionDistributionConfig
    ) -> None:
        """Simulate the kv-reuse-difficult failure mode: many sessions with
        a high restart probability, every is_restart_continuation turn must
        satisfy the block size invariant."""
        config = coding_config.model_copy(update={"restart_initial_probability": 1.0})
        synth = SessionSynthesizer(config, seed=42)
        sessions = synth.synthesize_sessions(40)
        restart_sessions = [s for s in sessions if s.is_restart_continuation]
        assert restart_sessions, "Expected at least one restart continuation"
        for session in restart_sessions:
            for turn in session.turns:
                self._assert_block_size_invariant(
                    turn,
                    config.block_size,
                    f"session {session.session_id} turn {turn.turn_index}",
                )

    def test_continuation_turn0_has_zero_delay_and_timestamp(
        self, coding_config: SessionDistributionConfig
    ) -> None:
        """Session B is queued independently; its turn 0 anchors a new timeline."""
        synth = SessionSynthesizer(coding_config, seed=42)
        _, session_b = synth.synthesize_session(inject_restart=True)
        assert session_b.turns[0].delay_ms == 0.0
        assert session_b.turns[0].timestamp_ms == 0.0

    def test_continuation_has_fresh_session_id(
        self, coding_config: SessionDistributionConfig
    ) -> None:
        synth = SessionSynthesizer(coding_config, seed=42)
        session_a, session_b = synth.synthesize_session(inject_restart=True)
        assert session_a.session_id != session_b.session_id


class TestRestartPropagation:
    """Coverage for multi-depth restart chains (Session A -> B -> C -> ...)."""

    def _propagating_config(
        self,
        coding_config: SessionDistributionConfig,
        *,
        max_restart_depth: int,
        restart_depth_decay: float = 1.0,
        restart_initial_probability: float = 1.0,
    ) -> SessionDistributionConfig:
        return coding_config.model_copy(
            update={
                "max_restart_depth": max_restart_depth,
                "restart_depth_decay": restart_depth_decay,
                "restart_initial_probability": restart_initial_probability,
            }
        )

    def test_default_max_depth_is_one(
        self, coding_config: SessionDistributionConfig
    ) -> None:
        """Default config preserves single-level restart behavior."""
        assert coding_config.max_restart_depth == 1

    def test_max_depth_one_produces_at_most_two_sessions(
        self, coding_config: SessionDistributionConfig
    ) -> None:
        synth = SessionSynthesizer(coding_config, seed=42)
        result = synth.synthesize_session(inject_restart=True)
        assert len(result) == 2
        assert result[0].restart_depth == 0
        assert result[1].restart_depth == 1

    def test_max_depth_caps_chain_length(
        self, coding_config: SessionDistributionConfig
    ) -> None:
        """max_restart_depth=N => chain can have at most N+1 sessions (primary + N)."""
        for depth in (1, 2, 3, 5):
            config = self._propagating_config(coding_config, max_restart_depth=depth)
            synth = SessionSynthesizer(config, seed=42)
            chains_observed = []
            for _ in range(30):
                result = synth.synthesize_session(inject_restart=True)
                chains_observed.append(len(result))
            assert max(chains_observed) <= depth + 1, (
                f"depth={depth}: saw chain of {max(chains_observed)}"
            )

    def test_full_propagation_produces_deep_chain(
        self, coding_config: SessionDistributionConfig
    ) -> None:
        """With decay=1, probability=1, max_depth=5 -> every chain is full depth
        unless a continuation hits forced retire / probabilistic reset before
        its restart turn. At least one run in 30 attempts should reach depth 5."""
        config = self._propagating_config(
            coding_config,
            max_restart_depth=5,
            restart_depth_decay=1.0,
            restart_initial_probability=1.0,
        )
        synth = SessionSynthesizer(config, seed=42)
        max_depth_seen = 0
        for _ in range(30):
            result = synth.synthesize_session(inject_restart=True)
            max_depth_seen = max(max_depth_seen, result[-1].restart_depth)
        assert max_depth_seen == 5, f"only reached depth {max_depth_seen}"

    def test_decay_reduces_chain_length(
        self, coding_config: SessionDistributionConfig
    ) -> None:
        """Higher decay -> shorter chains on average."""
        config_full = self._propagating_config(
            coding_config,
            max_restart_depth=5,
            restart_depth_decay=1.0,
            restart_initial_probability=1.0,
        )
        config_decayed = self._propagating_config(
            coding_config,
            max_restart_depth=5,
            restart_depth_decay=0.3,
            restart_initial_probability=1.0,
        )
        synth_full = SessionSynthesizer(config_full, seed=42)
        synth_decayed = SessionSynthesizer(config_decayed, seed=42)
        total_full = sum(
            len(synth_full.synthesize_session(inject_restart=True)) for _ in range(50)
        )
        total_decayed = sum(
            len(synth_decayed.synthesize_session(inject_restart=True))
            for _ in range(50)
        )
        assert total_full > total_decayed

    def test_chain_has_monotonic_restart_depth(
        self, coding_config: SessionDistributionConfig
    ) -> None:
        config = self._propagating_config(
            coding_config,
            max_restart_depth=5,
            restart_depth_decay=1.0,
            restart_initial_probability=1.0,
        )
        synth = SessionSynthesizer(config, seed=42)
        result = synth.synthesize_session(inject_restart=True)
        depths = [s.restart_depth for s in result]
        assert depths == list(range(len(result)))

    def test_chain_preserves_session_id_continuity_at_all_depths(
        self, coding_config: SessionDistributionConfig
    ) -> None:
        """Every continuation's turn 0 session IDs must extend the previous
        session's last-turn session IDs, all the way down the chain."""
        config = self._propagating_config(
            coding_config,
            max_restart_depth=4,
            restart_depth_decay=1.0,
            restart_initial_probability=1.0,
        )
        synth = SessionSynthesizer(config, seed=42)
        result = synth.synthesize_session(inject_restart=True)
        assert len(result) >= 2
        alloc = synth.allocator
        for parent, child in zip(result, result[1:], strict=False):
            parent_session_ids = alloc.extract_session_ids(parent.turns[-1].hash_ids)
            child_session_ids = alloc.extract_session_ids(child.turns[0].hash_ids)
            assert child_session_ids[: len(parent_session_ids)] == parent_session_ids

    def test_chain_all_turns_pass_block_size_invariant(
        self, coding_config: SessionDistributionConfig
    ) -> None:
        """Every turn in every chain member must satisfy the final-block-size
        check from generator/prompt.py, regardless of depth."""
        config = self._propagating_config(
            coding_config,
            max_restart_depth=5,
            restart_depth_decay=1.0,
            restart_initial_probability=1.0,
        )
        synth = SessionSynthesizer(config, seed=42)
        block_size = config.block_size
        for _ in range(20):
            result = synth.synthesize_session(inject_restart=True)
            for session in result:
                for turn in session.turns:
                    expected = math.ceil(turn.input_length / block_size)
                    assert len(turn.hash_ids) == expected
                    final_block = (
                        turn.input_length - (len(turn.hash_ids) - 1) * block_size
                    )
                    assert 0 < final_block <= block_size

    def test_chain_session_ids_are_unique(
        self, coding_config: SessionDistributionConfig
    ) -> None:
        config = self._propagating_config(
            coding_config,
            max_restart_depth=4,
            restart_depth_decay=1.0,
            restart_initial_probability=1.0,
        )
        synth = SessionSynthesizer(config, seed=42)
        result = synth.synthesize_session(inject_restart=True)
        session_ids = [s.session_id for s in result]
        assert len(session_ids) == len(set(session_ids))

    def test_all_chain_members_share_group_id(
        self, coding_config: SessionDistributionConfig
    ) -> None:
        """A chain represents one user's sequence of sessions on the same repo,
        so every session in the chain must share the primary's group_id."""
        config = self._propagating_config(
            coding_config,
            max_restart_depth=4,
            restart_depth_decay=1.0,
            restart_initial_probability=1.0,
        )
        synth = SessionSynthesizer(config, seed=42)
        result = synth.synthesize_session(inject_restart=True)
        group_ids = {s.group_id for s in result}
        assert len(group_ids) == 1

    def test_only_chain_terminus_has_non_restart_end_reason(
        self, coding_config: SessionDistributionConfig
    ) -> None:
        """Intermediate chain members end with RESTART_SPLIT; only the last
        session has FORCED_RETIRE or PROBABILISTIC_RESET."""
        config = self._propagating_config(
            coding_config,
            max_restart_depth=5,
            restart_depth_decay=1.0,
            restart_initial_probability=1.0,
        )
        synth = SessionSynthesizer(config, seed=42)
        result = synth.synthesize_session(inject_restart=True)
        for intermediate in result[:-1]:
            assert intermediate.end_reason == SessionEndReason.RESTART_SPLIT
        assert result[-1].end_reason in (
            SessionEndReason.FORCED_RETIRE,
            SessionEndReason.PROBABILISTIC_RESET,
        )

    def test_is_restart_continuation_tracks_restart_depth(
        self, coding_config: SessionDistributionConfig
    ) -> None:
        """is_restart_continuation must be True iff restart_depth > 0, so
        downstream writer code remains correct."""
        config = self._propagating_config(
            coding_config,
            max_restart_depth=4,
            restart_depth_decay=1.0,
            restart_initial_probability=1.0,
        )
        synth = SessionSynthesizer(config, seed=42)
        result = synth.synthesize_session(inject_restart=True)
        for session in result:
            assert session.is_restart_continuation == (session.restart_depth > 0)

    def test_bulk_generation_preserves_chain_ordering(
        self, coding_config: SessionDistributionConfig
    ) -> None:
        """In the synthesize_sessions output, sessions within a single chain
        (identified by shared session_index via the allocator's session_base)
        must appear in increasing restart_depth order."""
        config = self._propagating_config(
            coding_config,
            max_restart_depth=4,
            restart_depth_decay=1.0,
            restart_initial_probability=1.0,
        )
        synth = SessionSynthesizer(config, seed=42)
        sessions = synth.synthesize_sessions(40)

        # Group chain members by the session_index implied by their L2+L3
        # session_ids (first session block ID is monotonic per session_index).
        alloc = synth.allocator
        prefix_blocks = alloc.prefix_blocks

        def first_session_block(s: SynthesizedSession) -> int | None:
            ids = s.turns[0].hash_ids
            return ids[prefix_blocks] if len(ids) > prefix_blocks else None

        chains: dict[int, list[tuple[int, int]]] = {}
        for pos, s in enumerate(sessions):
            base = first_session_block(s)
            if base is None:
                continue
            # Round to session_base (blocks are contiguous per session_index).
            session_base_key = (base - alloc.session_base(0)) // 4000
            chains.setdefault(session_base_key, []).append((pos, s.restart_depth))

        chains_with_multiple = [c for c in chains.values() if len(c) >= 2]
        assert chains_with_multiple, "Expected at least one multi-session chain"
        for chain in chains_with_multiple:
            chain.sort()
            depths_in_position_order = [depth for _, depth in chain]
            assert depths_in_position_order == sorted(depths_in_position_order), (
                f"chain depths out of order: {depths_in_position_order}"
            )
