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

import contextlib
import heapq
import math
from collections import defaultdict
from dataclasses import dataclass, field

from aiperf.dataset.agentic_code_gen.prefix_model import MAX_GROUP_BLOCKS, MAX_GROUPS


@dataclass
class SimulationConfig:
    """Parameters for running the simulation."""

    concurrency: int = field(
        default=50, metadata={"description": "Max concurrent sessions"}
    )
    prefill_workers: int = field(
        default=8, metadata={"description": "Number of prefill workers"}
    )
    dp_workers: int = field(
        default=4, metadata={"description": "Data-parallel worker count"}
    )
    per_worker_tps: int = field(
        default=4000, metadata={"description": "Prefill tokens/sec per worker"}
    )
    decode_tps: int = field(default=200, metadata={"description": "Decode tokens/sec"})
    kv_bytes_per_token: int = field(
        default=35136, metadata={"description": "KV cache bytes per token"}
    )
    gpu_kv_capacity_gb: float = field(
        default=300.0, metadata={"description": "Total KV cache capacity in GB"}
    )
    l1_tokens: int = field(
        default=32000, metadata={"description": "L1 (global system prompt) tokens"}
    )
    l1_5_tokens: int = field(
        default=20000, metadata={"description": "L1.5 (group-shared prefix) tokens"}
    )
    block_size: int = field(
        default=512,
        metadata={"description": "KV cache block size in tokens (from synthesis)"},
    )


@dataclass
class TimeSeriesPoint:
    """A single observation in the simulation time series."""

    time_s: float
    active_requests: int
    input_tokens: int
    output_tokens: int
    queued: int
    active_sessions: int
    kv_cache_gb: float
    unique_cached_tokens: int
    alive_sessions: int
    unique_blocks: int
    eviction_count: int
    miss_l15_blocks: int
    miss_session_blocks: int
    total_evicted_blocks: int
    cumulative_hit_tokens: int
    cumulative_miss_tokens: int


@dataclass
class TurnEvent:
    """Timing data for a single turn within a session."""

    turn_idx: int
    delay_start: float
    prefill_start: float
    decode_start: float
    decode_end: float
    input_length: int
    output_length: int
    hit_tokens: int
    miss_tokens: int


@dataclass
class SessionState:
    """Per-session tracking during simulation."""

    start_time: float | None = None
    end_time: float | None = None
    turn_events: list[TurnEvent] = field(default_factory=list)
    slot: int | None = None
    is_restart: bool = False


@dataclass
class SimulationResult:
    """Complete output from a simulation run."""

    time_series: list[TimeSeriesPoint]
    session_states: list[SessionState]
    max_time: float
    total_prefill_ms: float
    total_decode_ms: float
    total_wait_ms: float
    avg_ttft: float
    turn_count: int
    eviction_count: int
    miss_l15_blocks: int
    miss_session_blocks: int
    total_evicted_blocks: int
    cache_hit_rate: float


def _classify_block(bid: int, l1_block_count: int, session_region_base: int) -> str:
    """Classify a block ID into its cache layer."""
    if bid < l1_block_count:
        return "l1"
    if bid < session_region_base:
        return "l15"
    return "session"


def _compute_dedup_tokens(
    cached_tokens: int,
    alive_sessions: int,
    active_groups: dict[int, int],
    l1_tokens: int,
    l1_5_tokens: int,
) -> int:
    """Compute deduplicated unique cache footprint in tokens.

    In a real prefix-caching system, L1 blocks are stored once regardless
    of how many sessions reference them, and L1.5 blocks are stored once
    per group. This subtracts the duplicate copies.
    """
    l1_dedup = max(0, alive_sessions - 1) * l1_tokens
    l15_dedup = sum(max(0, cnt - 1) * l1_5_tokens for cnt in active_groups.values())
    return max(0, cached_tokens - l1_dedup - l15_dedup)


def simulate(sessions: list[dict], config: SimulationConfig) -> SimulationResult:
    """Run discrete-event simulation over synthesized sessions.

    Cache hit rate is computed from actual block hits: for each turn, blocks
    already in the global cache (blockRefCount) are hits; the rest are misses
    that need prefill. This replaces the old flat cache_hit_rate input.
    """
    effective_prefill_workers = config.prefill_workers * config.dp_workers
    prefill_worker_free_at = [0.0] * effective_prefill_workers

    l1_block_count = math.ceil(config.l1_tokens / config.block_size)
    session_region_base = l1_block_count + MAX_GROUPS * MAX_GROUP_BLOCKS

    session_states = [
        SessionState(is_restart=bool(s.get("is_restart", False))) for s in sessions
    ]

    # Precompute cumulative hash_ids: turn N's full block set = union of turns 0..N
    for s in sessions:
        cumulative: set[int] = set()
        for turn in s["turns"]:
            cumulative.update(turn.get("hash_ids", []))
            turn["cumulative_hash_ids"] = list(cumulative)

    # Priority queue: (time, counter, event_type, session_idx, turn_idx)
    pq: list[tuple[float, int, str, int, int]] = []
    counter = 0

    def push_event(time: float, etype: str, s_idx: int, t_idx: int) -> None:
        nonlocal counter
        heapq.heappush(pq, (time, counter, etype, s_idx, t_idx))
        counter += 1

    def acquire_prefill_worker(ready_time: float) -> tuple[int, float]:
        best_idx = 0
        for i in range(1, effective_prefill_workers):
            if prefill_worker_free_at[i] < prefill_worker_free_at[best_idx]:
                best_idx = i
        start_time = max(ready_time, prefill_worker_free_at[best_idx])
        return best_idx, start_time

    # Slot tracking for Gantt
    slot_free_at = [0.0] * config.concurrency
    next_slot = 0

    def assign_slot(s_idx: int, _start_time: float) -> None:
        nonlocal next_slot
        if next_slot < config.concurrency:
            session_states[s_idx].slot = next_slot
            next_slot += 1
            return
        best_slot = 0
        for i in range(1, config.concurrency):
            if slot_free_at[i] < slot_free_at[best_slot]:
                best_slot = i
        session_states[s_idx].slot = best_slot

    def release_slot(s_idx: int, end_time: float) -> None:
        slot = session_states[s_idx].slot
        if slot is not None:
            slot_free_at[slot] = end_time

    active_count = 0
    next_session = 0

    # KV cache tracking
    cached_tokens = 0
    alive_sessions = 0
    session_cache_tokens = [0.0] * len(sessions)
    session_group_id = [s.get("group_id", 0) for s in sessions]
    active_groups: dict[int, int] = {}

    # Unique block tracking
    block_refcount: dict[int, int] = defaultdict(int)
    session_blocks: list[set[int]] = [set() for _ in sessions]

    # LRU eviction
    lru_queue: list[int] = []
    in_lru_queue: set[int] = set()
    session_evicted = [False] * len(sessions)
    eviction_count = 0
    miss_l15_blocks = 0
    miss_session_blocks = 0
    evicted_blocks: set[int] = set()
    total_evicted_blocks = 0
    total_capacity_tokens = config.gpu_kv_capacity_gb * 1e9 / config.kv_bytes_per_token

    # Cache hit tracking (cumulative across all turns)
    total_hit_tokens = 0
    total_miss_tokens = 0

    def evict_lru() -> None:
        nonlocal cached_tokens, eviction_count, total_evicted_blocks
        unique_cached = _compute_dedup_tokens(
            cached_tokens,
            alive_sessions,
            active_groups,
            config.l1_tokens,
            config.l1_5_tokens,
        )
        while unique_cached > total_capacity_tokens and lru_queue:
            victim_idx = lru_queue.pop(0)
            in_lru_queue.discard(victim_idx)
            if session_cache_tokens[victim_idx] == 0:
                continue
            cached_tokens -= session_cache_tokens[victim_idx]
            session_cache_tokens[victim_idx] = 0
            for bid in session_blocks[victim_idx]:
                block_refcount[bid] -= 1
                if block_refcount[bid] <= 0:
                    del block_refcount[bid]
                    evicted_blocks.add(bid)
                    total_evicted_blocks += 1
            session_blocks[victim_idx].clear()
            session_evicted[victim_idx] = True
            eviction_count += 1
            unique_cached = _compute_dedup_tokens(
                cached_tokens,
                alive_sessions,
                active_groups,
                config.l1_tokens,
                config.l1_5_tokens,
            )

    def add_to_lru(s_idx: int) -> None:
        if s_idx not in in_lru_queue:
            lru_queue.append(s_idx)
            in_lru_queue.add(s_idx)

    def remove_from_lru(s_idx: int) -> None:
        if s_idx in in_lru_queue:
            with contextlib.suppress(ValueError):
                lru_queue.remove(s_idx)
            in_lru_queue.discard(s_idx)

    def start_session(s_idx: int, time: float, inherit_slot: int | None = None) -> None:
        session_states[s_idx].start_time = time
        if inherit_slot is not None:
            session_states[s_idx].slot = inherit_slot
        else:
            assign_slot(s_idx, time)
        start_turn(s_idx, 0, time)

    def start_turn(s_idx: int, t_idx: int, time: float) -> None:
        nonlocal total_hit_tokens, total_miss_tokens
        turn = sessions[s_idx]["turns"][t_idx]
        delay = 0.0 if t_idx == 0 else turn["delay_ms"]
        turn_ready_time = time + delay

        if t_idx > 0 and delay > 0:
            add_to_lru(s_idx)

        # Compute cache hits from actual block state:
        # check ALL cumulative blocks (not just incremental) against cache.
        # Blocks from prior turns of this session are already cached (hits).
        # If session was evicted during idle, those blocks are gone (misses).
        all_hids = turn.get("cumulative_hash_ids", [])
        hit_blocks = sum(1 for bid in all_hids if bid in block_refcount)
        miss_blocks = len(all_hids) - hit_blocks
        hit_tokens = hit_blocks * config.block_size
        miss_tokens = miss_blocks * config.block_size
        total_hit_tokens += hit_tokens
        total_miss_tokens += miss_tokens

        # Only miss tokens need prefill
        prefill_duration = (miss_tokens / config.per_worker_tps) * 1000
        decode_duration = (turn["output_length"] / config.decode_tps) * 1000

        worker_idx, prefill_start = acquire_prefill_worker(turn_ready_time)
        prefill_end = prefill_start + prefill_duration
        prefill_worker_free_at[worker_idx] = prefill_end

        decode_start = prefill_end
        decode_end = decode_start + decode_duration

        push_event(decode_start, "request_start", s_idx, t_idx)
        push_event(decode_end, "request_end", s_idx, t_idx)

        session_states[s_idx].turn_events.append(
            TurnEvent(
                turn_idx=t_idx,
                delay_start=time,
                prefill_start=prefill_start,
                decode_start=decode_start,
                decode_end=decode_end,
                input_length=turn["input_length"],
                output_length=turn["output_length"],
                hit_tokens=hit_tokens,
                miss_tokens=miss_tokens,
            )
        )

    # Launch initial batch
    while next_session < len(sessions) and active_count < config.concurrency:
        active_count += 1
        start_session(next_session, 0.0)
        next_session += 1

    active_requests = 0
    input_tokens = 0
    output_tokens = 0
    max_time = 0.0

    time_series_raw: list[TimeSeriesPoint] = []

    while pq:
        time, _cnt, etype, s_idx, t_idx = heapq.heappop(pq)
        max_time = max(max_time, time)

        if etype == "request_start":
            active_requests += 1
            turn = sessions[s_idx]["turns"][t_idx]
            input_tokens += turn["cumulative_input_length"]
            output_tokens += turn["output_length"]

            remove_from_lru(s_idx)

            # Re-cache evicted session's blocks
            if session_evicted[s_idx]:
                session_evicted[s_idx] = False
                session_cache_tokens[s_idx] = 0
                s_blocks = session_blocks[s_idx]
                for t in range(t_idx):
                    prev_hids = sessions[s_idx]["turns"][t].get("hash_ids", [])
                    for bid in prev_hids:
                        if bid not in s_blocks:
                            if bid in evicted_blocks:
                                layer = _classify_block(
                                    bid, l1_block_count, session_region_base
                                )
                                if layer == "l15":
                                    miss_l15_blocks += 1
                                elif layer != "l1":
                                    miss_session_blocks += 1
                                evicted_blocks.discard(bid)
                            s_blocks.add(bid)
                            block_refcount[bid] += 1

            # Cache: allocate tokens for this turn
            prev_cache = session_cache_tokens[s_idx]
            session_cache_tokens[s_idx] = turn["cumulative_input_length"]
            cached_tokens += session_cache_tokens[s_idx] - prev_cache
            if t_idx == 0:
                alive_sessions += 1
                gid = session_group_id[s_idx]
                active_groups[gid] = active_groups.get(gid, 0) + 1

            # Unique blocks: add hash_ids and detect misses
            hids = turn.get("hash_ids", [])
            s_blocks = session_blocks[s_idx]
            for bid in hids:
                if bid not in s_blocks:
                    if bid in evicted_blocks:
                        layer = _classify_block(
                            bid, l1_block_count, session_region_base
                        )
                        if layer == "l15":
                            miss_l15_blocks += 1
                        elif layer != "l1":
                            miss_session_blocks += 1
                        evicted_blocks.discard(bid)
                    s_blocks.add(bid)
                    block_refcount[bid] += 1

            evict_lru()

        elif etype == "request_end":
            active_requests -= 1
            turn = sessions[s_idx]["turns"][t_idx]
            input_tokens -= turn["cumulative_input_length"]
            output_tokens -= turn["output_length"]

            cached_tokens += turn["output_length"]
            session_cache_tokens[s_idx] += turn["output_length"]

            if t_idx + 1 < len(sessions[s_idx]["turns"]):
                start_turn(s_idx, t_idx + 1, time)
            else:
                alive_sessions -= 1
                gid = session_group_id[s_idx]
                active_groups[gid] -= 1
                if active_groups[gid] <= 0:
                    del active_groups[gid]
                add_to_lru(s_idx)

                session_states[s_idx].end_time = time
                freed_slot = session_states[s_idx].slot
                release_slot(s_idx, time)
                active_count -= 1
                if next_session < len(sessions):
                    active_count += 1
                    start_session(next_session, time, freed_slot)
                    next_session += 1

        if etype in ("request_start", "request_end"):
            unique_cached = _compute_dedup_tokens(
                cached_tokens,
                alive_sessions,
                active_groups,
                config.l1_tokens,
                config.l1_5_tokens,
            )
            kv_cache_gb = unique_cached * config.kv_bytes_per_token / 1e9

            time_series_raw.append(
                TimeSeriesPoint(
                    time_s=time / 1000,
                    active_requests=active_requests,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    queued=len(sessions) - next_session,
                    active_sessions=active_count,
                    kv_cache_gb=kv_cache_gb,
                    unique_cached_tokens=unique_cached,
                    alive_sessions=alive_sessions,
                    unique_blocks=len(block_refcount),
                    eviction_count=eviction_count,
                    miss_l15_blocks=miss_l15_blocks,
                    miss_session_blocks=miss_session_blocks,
                    total_evicted_blocks=total_evicted_blocks,
                    cumulative_hit_tokens=total_hit_tokens,
                    cumulative_miss_tokens=total_miss_tokens,
                )
            )

    # Compute aggregate stats
    total_prefill_ms = 0.0
    total_decode_ms = 0.0
    total_wait_ms = 0.0
    ttft_sum = 0.0
    turn_count = 0
    for s in session_states:
        for evt in s.turn_events:
            total_wait_ms += evt.prefill_start - evt.delay_start
            total_prefill_ms += evt.decode_start - evt.prefill_start
            total_decode_ms += evt.decode_end - evt.decode_start
            ttft_sum += evt.decode_start - evt.delay_start
            turn_count += 1
    avg_ttft = ttft_sum / turn_count if turn_count > 0 else 0.0

    total_tokens = total_hit_tokens + total_miss_tokens
    cache_hit_rate = total_hit_tokens / total_tokens if total_tokens > 0 else 0.0

    last = (
        time_series_raw[-1]
        if time_series_raw
        else TimeSeriesPoint(
            time_s=0,
            active_requests=0,
            input_tokens=0,
            output_tokens=0,
            queued=0,
            active_sessions=0,
            kv_cache_gb=0,
            unique_cached_tokens=0,
            alive_sessions=0,
            unique_blocks=0,
            eviction_count=0,
            miss_l15_blocks=0,
            miss_session_blocks=0,
            total_evicted_blocks=0,
            cumulative_hit_tokens=0,
            cumulative_miss_tokens=0,
        )
    )

    return SimulationResult(
        time_series=time_series_raw,
        session_states=session_states,
        max_time=max_time,
        total_prefill_ms=total_prefill_ms,
        total_decode_ms=total_decode_ms,
        total_wait_ms=total_wait_ms,
        avg_ttft=avg_ttft,
        turn_count=turn_count,
        eviction_count=eviction_count,
        miss_l15_blocks=last.miss_l15_blocks,
        miss_session_blocks=last.miss_session_blocks,
        total_evicted_blocks=last.total_evicted_blocks,
        cache_hit_rate=cache_hit_rate,
    )
