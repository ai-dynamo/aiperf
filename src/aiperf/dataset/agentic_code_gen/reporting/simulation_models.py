# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data models and stateless helpers for the KV cache simulation engine.

Extracted from simulation_engine.py so the engine file itself stays small and
focused on the event loop. Tests import `SimulationConfig`, `TimeSeriesPoint`,
`_compute_dedup_tokens` from the engine module, which re-exports them here.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class SimulationConfig:
    """Parameters for running the simulation."""

    concurrency: int = field(
        default=50, metadata={"description": "Max concurrent sessions"}
    )
    prefill_tps: int = field(
        default=64000, metadata={"description": "Aggregate prefill tokens/sec"}
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

    def validate(self) -> None:
        """Raise ValueError for invalid simulation parameters."""
        positive_fields = {
            "concurrency": self.concurrency,
            "prefill_tps": self.prefill_tps,
            "decode_tps": self.decode_tps,
            "kv_bytes_per_token": self.kv_bytes_per_token,
            "gpu_kv_capacity_gb": self.gpu_kv_capacity_gb,
            "block_size": self.block_size,
        }
        for name, value in positive_fields.items():
            if value <= 0:
                raise ValueError(f"{name} must be > 0")
        nonnegative_fields = {
            "l1_tokens": self.l1_tokens,
            "l1_5_tokens": self.l1_5_tokens,
        }
        for name, value in nonnegative_fields.items():
            if value < 0:
                raise ValueError(f"{name} must be >= 0")


@dataclass(slots=True)
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


@dataclass(slots=True)
class TurnEvent:
    """Timing data for a single turn within a session."""

    turn_idx: int
    delay_start: float
    turn_ready: float
    prefill_start: float
    decode_start: float
    decode_end: float
    input_length: int
    output_length: int
    hit_tokens: int
    miss_tokens: int


@dataclass(slots=True)
class SessionState:
    """Per-session tracking during simulation."""

    start_time: float | None = None
    end_time: float | None = None
    turn_events: list[TurnEvent] = field(default_factory=list)
    slot: int | None = None
    is_restart: bool = False


@dataclass(slots=True)
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
    *,
    cached_sessions: int | None = None,
    cached_groups: dict[int, int] | None = None,
) -> int:
    """Compute deduplicated unique cache footprint in tokens.

    In a real prefix-caching system, L1 blocks are stored once regardless
    of how many sessions reference them, and L1.5 blocks are stored once
    per group. This subtracts the duplicate copies.
    """
    session_count = alive_sessions if cached_sessions is None else cached_sessions
    group_counts = active_groups if cached_groups is None else cached_groups
    l1_dedup = max(0, session_count - 1) * l1_tokens
    l15_dedup = sum(max(0, cnt - 1) * l1_5_tokens for cnt in group_counts.values())
    return max(0, cached_tokens - l1_dedup - l15_dedup)


def _register_turn_blocks(
    *,
    hids: list[int],
    session_blocks: set[int],
    evicted_blocks: set[int],
    block_refcount: dict[int, int],
    l1_block_count: int,
    session_region_base: int,
) -> tuple[int, int]:
    """Add a turn's block hashes to the session set, counting evicted-block misses by layer."""
    miss_l15_delta = 0
    miss_session_delta = 0
    for bid in hids:
        if bid in session_blocks:
            continue
        if bid in evicted_blocks:
            layer = _classify_block(bid, l1_block_count, session_region_base)
            if layer == "l15":
                miss_l15_delta += 1
            elif layer != "l1":
                miss_session_delta += 1
            evicted_blocks.discard(bid)
        session_blocks.add(bid)
        block_refcount[bid] += 1
    return miss_l15_delta, miss_session_delta


def _rehydrate_evicted_session(
    *,
    sessions: list[dict],
    s_idx: int,
    t_idx: int,
    session_blocks: set[int],
    evicted_blocks: set[int],
    block_refcount: dict[int, int],
    l1_block_count: int,
    session_region_base: int,
) -> tuple[int, int]:
    """Replay all prior turns' block hashes for an evicted session, tallying layered misses."""
    miss_l15_delta = 0
    miss_session_delta = 0
    for t in range(t_idx):
        prev_hids = sessions[s_idx]["turns"][t].get("hash_ids", [])
        delta_l15, delta_session = _register_turn_blocks(
            hids=prev_hids,
            session_blocks=session_blocks,
            evicted_blocks=evicted_blocks,
            block_refcount=block_refcount,
            l1_block_count=l1_block_count,
            session_region_base=session_region_base,
        )
        miss_l15_delta += delta_l15
        miss_session_delta += delta_session
    return miss_l15_delta, miss_session_delta
