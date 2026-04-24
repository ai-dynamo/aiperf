# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python simulation engine for KV cache pressure modeling.

This is the source-of-truth implementation for the discrete-event simulation.
The JS embedded in simulation.html mirrors this logic. Tests validate
correctness here; the JS is a rendering-equivalent copy.

Key design decisions:
- Eviction triggers on *deduplicated* token count (unique cache footprint),
  matching real prefix-caching systems where shared blocks occupy memory once.
- Block IDs from the JSONL carry their own block_size from synthesis;
  the simulation does not re-interpret block granularity.
- Cache hit rate is computed from actual block hits, not a user input.
  Each turn's prefill cost depends on how many of its hash_ids are already
  in the global block cache (hits) vs need to be fetched (misses).
"""

from __future__ import annotations

import heapq
import math
from collections import defaultdict, deque

from aiperf.dataset.agentic_code_gen.prefix_model import MAX_GROUP_BLOCKS, MAX_GROUPS
from aiperf.dataset.agentic_code_gen.reporting.simulation_models import (
    SessionState,
    SimulationConfig,
    SimulationResult,
    TimeSeriesPoint,
    TurnEvent,
    _compute_dedup_tokens,
    _register_turn_blocks,
    _rehydrate_evicted_session,
)

# Re-exports for external callers and tests.
__all__ = [
    "SessionState",
    "SimulationConfig",
    "SimulationResult",
    "TimeSeriesPoint",
    "TurnEvent",
    "_compute_dedup_tokens",
    "simulate",
]


class _Simulator:
    """Encapsulates mutable simulation state and event handlers.

    Splitting the former monolithic `simulate` function into a class lets each
    event type live in its own small method, satisfying complexity limits
    without changing behavior.
    """

    def __init__(self, sessions: list[dict], config: SimulationConfig) -> None:
        config.validate()
        self.sessions = sessions
        self.config = config

        self.l1_block_count = math.ceil(config.l1_tokens / config.block_size)
        self.session_region_base = self.l1_block_count + MAX_GROUPS * MAX_GROUP_BLOCKS
        self.total_capacity_tokens = (
            config.gpu_kv_capacity_gb * 1e9 / config.kv_bytes_per_token
        )

        self.session_states = [
            SessionState(is_restart=bool(s.get("is_restart", False))) for s in sessions
        ]
        self._precompute_cumulative_hids()

        # Event queue: (time, counter, event_type, session_idx, turn_idx)
        self.pq: list[tuple[float, int, str, int, int]] = []
        self.counter = 0
        self.prefill_free_at = 0.0
        self.next_slot = 0
        self.active_count = 0
        self.next_session = 0

        # KV cache state
        self.cached_tokens = 0
        self.alive_sessions = 0
        self.session_cache_tokens = [0.0] * len(sessions)
        self.session_group_id = [s.get("group_id", 0) for s in sessions]
        self.active_groups: dict[int, int] = {}

        # Unique blocks
        self.block_refcount: dict[int, int] = defaultdict(int)
        self.session_blocks: list[set[int]] = [set() for _ in sessions]

        # LRU eviction
        self.lru_queue: deque[tuple[int, int]] = deque()
        self.lru_generation = [0] * len(sessions)
        self.in_lru_queue: set[int] = set()
        self.session_evicted = [False] * len(sessions)
        self.eviction_count = 0
        self.miss_l15_blocks = 0
        self.miss_session_blocks = 0
        self.evicted_blocks: set[int] = set()
        self.total_evicted_blocks = 0

        # Cumulative hit/miss tracking
        self.total_hit_tokens = 0
        self.total_miss_tokens = 0

        # Aggregated time-series bookkeeping
        self.active_requests = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.max_time = 0.0
        self.time_series: list[TimeSeriesPoint] = []

    def _precompute_cumulative_hids(self) -> None:
        """Attach cumulative hash_ids to each turn (union of turns 0..N)."""
        for s in self.sessions:
            cumulative: set[int] = set()
            for turn in s["turns"]:
                cumulative.update(turn.get("hash_ids", []))
                turn["cumulative_hash_ids"] = list(cumulative)

    def _push(self, time: float, etype: str, s_idx: int, t_idx: int) -> None:
        heapq.heappush(self.pq, (time, self.counter, etype, s_idx, t_idx))
        self.counter += 1

    def _reserve_prefill(
        self, ready_time: float, duration: float
    ) -> tuple[float, float]:
        start_time = max(ready_time, self.prefill_free_at)
        end_time = start_time + duration
        self.prefill_free_at = end_time
        return start_time, end_time

    def _assign_slot(self, s_idx: int) -> None:
        if self.next_slot >= self.config.concurrency:
            raise RuntimeError("no concurrency slot available")
        self.session_states[s_idx].slot = self.next_slot
        self.next_slot += 1

    def _cached_group_counts(self) -> dict[int, int]:
        counts: dict[int, int] = defaultdict(int)
        for idx, tokens in enumerate(self.session_cache_tokens):
            if tokens > 0:
                counts[self.session_group_id[idx]] += 1
        return counts

    def _cached_session_count(self) -> int:
        return sum(1 for tokens in self.session_cache_tokens if tokens > 0)

    def _unique_cached_tokens(self) -> int:
        return _compute_dedup_tokens(
            self.cached_tokens,
            self.alive_sessions,
            self.active_groups,
            self.config.l1_tokens,
            self.config.l1_5_tokens,
            cached_sessions=self._cached_session_count(),
            cached_groups=self._cached_group_counts(),
        )

    def _evict_victim(self, victim_idx: int) -> None:
        self.cached_tokens -= self.session_cache_tokens[victim_idx]
        self.session_cache_tokens[victim_idx] = 0
        for bid in self.session_blocks[victim_idx]:
            self.block_refcount[bid] -= 1
            if self.block_refcount[bid] <= 0:
                del self.block_refcount[bid]
                self.evicted_blocks.add(bid)
                self.total_evicted_blocks += 1
        self.session_blocks[victim_idx].clear()
        self.session_evicted[victim_idx] = True
        self.eviction_count += 1

    def _evict_lru(self) -> None:
        unique_cached = self._unique_cached_tokens()
        while unique_cached > self.total_capacity_tokens and self.lru_queue:
            victim_idx, generation = self.lru_queue.popleft()
            if (
                victim_idx not in self.in_lru_queue
                or generation != self.lru_generation[victim_idx]
            ):
                continue
            self.in_lru_queue.discard(victim_idx)
            if self.session_cache_tokens[victim_idx] == 0:
                continue
            self._evict_victim(victim_idx)
            unique_cached = self._unique_cached_tokens()

    def _add_to_lru(self, s_idx: int) -> None:
        if s_idx not in self.in_lru_queue:
            self.lru_generation[s_idx] += 1
            self.lru_queue.append((s_idx, self.lru_generation[s_idx]))
            self.in_lru_queue.add(s_idx)

    def _remove_from_lru(self, s_idx: int) -> None:
        if s_idx in self.in_lru_queue:
            self.in_lru_queue.discard(s_idx)
            self.lru_generation[s_idx] += 1

    def _start_session(
        self, s_idx: int, time: float, inherit_slot: int | None = None
    ) -> None:
        self.session_states[s_idx].start_time = time
        if inherit_slot is not None:
            self.session_states[s_idx].slot = inherit_slot
        else:
            self._assign_slot(s_idx)
        self._start_turn(s_idx, 0, time)

    def _start_turn(self, s_idx: int, t_idx: int, time: float) -> None:
        turn = self.sessions[s_idx]["turns"][t_idx]
        delay = 0.0 if t_idx == 0 else turn["delay_ms"]
        turn_ready_time = time + delay
        if t_idx > 0 and delay > 0:
            self._add_to_lru(s_idx)
        self._push(turn_ready_time, "turn_ready", s_idx, t_idx)

    def _on_turn_ready(self, s_idx: int, t_idx: int, time: float) -> None:
        turn = self.sessions[s_idx]["turns"][t_idx]
        self._remove_from_lru(s_idx)

        # Cache hits come from actual block state across cumulative hash_ids.
        # Evicted blocks become misses even for prior turns of the same session.
        all_hids = turn.get("cumulative_hash_ids", [])
        hit_blocks = sum(1 for bid in all_hids if bid in self.block_refcount)
        miss_blocks = len(all_hids) - hit_blocks
        hit_tokens = hit_blocks * self.config.block_size
        miss_tokens = miss_blocks * self.config.block_size
        self.total_hit_tokens += hit_tokens
        self.total_miss_tokens += miss_tokens

        prefill_duration = (miss_tokens / self.config.prefill_tps) * 1000
        decode_duration = (turn["output_length"] / self.config.decode_tps) * 1000

        prefill_start, prefill_end = self._reserve_prefill(time, prefill_duration)
        decode_start = prefill_end
        decode_end = decode_start + decode_duration

        self._push(decode_start, "request_start", s_idx, t_idx)
        self._push(decode_end, "request_end", s_idx, t_idx)

        self.session_states[s_idx].turn_events.append(
            TurnEvent(
                turn_idx=t_idx,
                delay_start=time - (0.0 if t_idx == 0 else turn["delay_ms"]),
                turn_ready=time,
                prefill_start=prefill_start,
                decode_start=decode_start,
                decode_end=decode_end,
                input_length=turn["input_length"],
                output_length=turn["output_length"],
                hit_tokens=hit_tokens,
                miss_tokens=miss_tokens,
            )
        )

    def _on_request_start(self, s_idx: int, t_idx: int) -> None:
        self.active_requests += 1
        turn = self.sessions[s_idx]["turns"][t_idx]
        self.input_tokens += turn["cumulative_input_length"]
        self.output_tokens += turn["output_length"]

        if self.session_evicted[s_idx]:
            self.session_evicted[s_idx] = False
            self.session_cache_tokens[s_idx] = 0
            miss_l15_add, miss_session_add = _rehydrate_evicted_session(
                sessions=self.sessions,
                s_idx=s_idx,
                t_idx=t_idx,
                session_blocks=self.session_blocks[s_idx],
                evicted_blocks=self.evicted_blocks,
                block_refcount=self.block_refcount,
                l1_block_count=self.l1_block_count,
                session_region_base=self.session_region_base,
            )
            self.miss_l15_blocks += miss_l15_add
            self.miss_session_blocks += miss_session_add

        prev_cache = self.session_cache_tokens[s_idx]
        self.session_cache_tokens[s_idx] = turn["cumulative_input_length"]
        self.cached_tokens += self.session_cache_tokens[s_idx] - prev_cache
        if t_idx == 0:
            self.alive_sessions += 1
            gid = self.session_group_id[s_idx]
            self.active_groups[gid] = self.active_groups.get(gid, 0) + 1

        miss_l15_add, miss_session_add = _register_turn_blocks(
            hids=turn.get("hash_ids", []),
            session_blocks=self.session_blocks[s_idx],
            evicted_blocks=self.evicted_blocks,
            block_refcount=self.block_refcount,
            l1_block_count=self.l1_block_count,
            session_region_base=self.session_region_base,
        )
        self.miss_l15_blocks += miss_l15_add
        self.miss_session_blocks += miss_session_add

        self._evict_lru()

    def _on_request_end(self, s_idx: int, t_idx: int, time: float) -> None:
        self.active_requests -= 1
        turn = self.sessions[s_idx]["turns"][t_idx]
        self.input_tokens -= turn["cumulative_input_length"]
        self.output_tokens -= turn["output_length"]

        self.cached_tokens += turn["output_length"]
        self.session_cache_tokens[s_idx] += turn["output_length"]

        if t_idx + 1 < len(self.sessions[s_idx]["turns"]):
            self._start_turn(s_idx, t_idx + 1, time)
            return

        self.alive_sessions -= 1
        gid = self.session_group_id[s_idx]
        self.active_groups[gid] -= 1
        if self.active_groups[gid] <= 0:
            del self.active_groups[gid]
        self._add_to_lru(s_idx)

        self.session_states[s_idx].end_time = time
        freed_slot = self.session_states[s_idx].slot
        self.active_count -= 1
        if self.next_session < len(self.sessions):
            self.active_count += 1
            self._start_session(self.next_session, time, freed_slot)
            self.next_session += 1

    def _record_time_point(self, time: float) -> None:
        unique_cached = self._unique_cached_tokens()
        kv_cache_gb = unique_cached * self.config.kv_bytes_per_token / 1e9
        self.time_series.append(
            TimeSeriesPoint(
                time_s=time / 1000,
                active_requests=self.active_requests,
                input_tokens=self.input_tokens,
                output_tokens=self.output_tokens,
                queued=len(self.sessions) - self.next_session,
                active_sessions=self.active_count,
                kv_cache_gb=kv_cache_gb,
                unique_cached_tokens=unique_cached,
                alive_sessions=self.alive_sessions,
                unique_blocks=len(self.block_refcount),
                eviction_count=self.eviction_count,
                miss_l15_blocks=self.miss_l15_blocks,
                miss_session_blocks=self.miss_session_blocks,
                total_evicted_blocks=self.total_evicted_blocks,
                cumulative_hit_tokens=self.total_hit_tokens,
                cumulative_miss_tokens=self.total_miss_tokens,
            )
        )

    def _launch_initial_batch(self) -> None:
        while (
            self.next_session < len(self.sessions)
            and self.active_count < self.config.concurrency
        ):
            self.active_count += 1
            self._start_session(self.next_session, 0.0)
            self.next_session += 1

    def _dispatch_event(self, etype: str, s_idx: int, t_idx: int, time: float) -> None:
        if etype == "request_start":
            self._on_request_start(s_idx, t_idx)
        elif etype == "turn_ready":
            self._on_turn_ready(s_idx, t_idx, time)
        elif etype == "request_end":
            self._on_request_end(s_idx, t_idx, time)

    def run(self) -> SimulationResult:
        self._launch_initial_batch()
        while self.pq:
            time, _cnt, etype, s_idx, t_idx = heapq.heappop(self.pq)
            self.max_time = max(self.max_time, time)
            self._dispatch_event(etype, s_idx, t_idx, time)
            if etype in ("request_start", "request_end"):
                self._record_time_point(time)
        return self._build_result()

    def _build_result(self) -> SimulationResult:
        total_prefill_ms = 0.0
        total_decode_ms = 0.0
        total_wait_ms = 0.0
        ttft_sum = 0.0
        turn_count = 0
        for s in self.session_states:
            for evt in s.turn_events:
                total_wait_ms += evt.prefill_start - evt.turn_ready
                total_prefill_ms += evt.decode_start - evt.prefill_start
                total_decode_ms += evt.decode_end - evt.decode_start
                ttft_sum += evt.decode_start - evt.turn_ready
                turn_count += 1
        avg_ttft = ttft_sum / turn_count if turn_count > 0 else 0.0

        total_tokens = self.total_hit_tokens + self.total_miss_tokens
        cache_hit_rate = (
            self.total_hit_tokens / total_tokens if total_tokens > 0 else 0.0
        )

        return SimulationResult(
            time_series=self.time_series,
            session_states=self.session_states,
            max_time=self.max_time,
            total_prefill_ms=total_prefill_ms,
            total_decode_ms=total_decode_ms,
            total_wait_ms=total_wait_ms,
            avg_ttft=avg_ttft,
            turn_count=turn_count,
            eviction_count=self.eviction_count,
            miss_l15_blocks=self.miss_l15_blocks,
            miss_session_blocks=self.miss_session_blocks,
            total_evicted_blocks=self.total_evicted_blocks,
            cache_hit_rate=cache_hit_rate,
        )


def simulate(sessions: list[dict], config: SimulationConfig) -> SimulationResult:
    """Run discrete-event simulation over synthesized sessions.

    Cache hit rate is computed from actual block hits: for each turn, blocks
    already in the global cache (blockRefCount) are hits; the rest are misses
    that need prefill. This replaces the old flat cache_hit_rate input.
    """
    return _Simulator(sessions, config).run()
