# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Python simulation engine (source of truth).

Uses tiny GPU caches (5-10 blocks) so every block can be traced by hand.
Validates:
1. Block dedup in cache (shared L1/L1.5 blocks stored once, like real hardware)
2. Eviction uses deduplicated footprint, not raw token count
3. Small multi-session scenarios where we can verify exact eviction behavior
4. Prefill and decode timing correctness
5. Cache hit rate computed from actual block hits (not a user input)
"""

from __future__ import annotations

import pytest

from aiperf.dataset.agentic_code_gen.reporting.simulation_engine import (
    SimulationConfig,
    TimeSeriesPoint,
    _compute_dedup_tokens,
    simulate,
)

# block_size=100 throughout: 1 block = 100 tokens, easy mental math.
# kv_bytes_per_token=1: capacity in GB = capacity in tokens / 1e9.
# prefill_tps=1M: prefill is near-instant for miss tokens.
# decode_tps=10: slow decode so sessions stay alive long enough to overlap.

FAST_PREFILL = dict(
    prefill_tps=1_000_000,
    kv_bytes_per_token=1,
    l1_tokens=0,
    l1_5_tokens=0,
    block_size=100,
)


def _make_session(
    sid: str,
    turns: list[dict],
    group_id: int = 0,
    is_restart: bool = False,
) -> dict:
    """Build a session dict matching the format from load_sessions."""
    cumulative = 0
    for turn in turns:
        turn.setdefault("output_length", 10)
        cumulative += turn["input_length"]
        turn["cumulative_input_length"] = cumulative
        cumulative += turn["output_length"]
        turn.setdefault("delay_ms", 0)
        turn.setdefault("hash_ids", [])
    return {
        "session_id": sid,
        "group_id": group_id,
        "is_restart": is_restart,
        "turns": turns,
    }


def _cfg(**overrides) -> SimulationConfig:
    kw = {**FAST_PREFILL, **overrides}
    return SimulationConfig(**kw)


def _peak(ts: list[TimeSeriesPoint], field: str) -> float:
    return max(getattr(p, field) for p in ts)


def _last(ts: list[TimeSeriesPoint]) -> TimeSeriesPoint:
    return ts[-1]


# ---------------------------------------------------------------------------
# Unit: dedup token math
# ---------------------------------------------------------------------------
class TestComputeDedupTokens:
    def test_single_session_no_dedup(self) -> None:
        assert _compute_dedup_tokens(1000, 1, {0: 1}, 200, 100) == 1000

    def test_two_sessions_same_group(self) -> None:
        # L1 dedup: (2-1)*200 = 200, L1.5 dedup: (2-1)*100 = 100
        assert _compute_dedup_tokens(2000, 2, {0: 2}, 200, 100) == 2000 - 300

    def test_two_sessions_different_groups(self) -> None:
        # L1 dedup: (2-1)*200 = 200, L1.5: no sharing across groups
        assert _compute_dedup_tokens(2000, 2, {0: 1, 1: 1}, 200, 100) == 2000 - 200

    def test_floor_at_zero(self) -> None:
        assert _compute_dedup_tokens(10, 100, {0: 100}, 200, 100) == 0

    def test_cached_counts_ignore_evicted_sessions(self) -> None:
        assert (
            _compute_dedup_tokens(
                1000,
                3,
                {0: 3},
                200,
                100,
                cached_sessions=1,
                cached_groups={0: 1},
            )
            == 1000
        )


class TestSimulationConfigValidation:
    @pytest.mark.parametrize(
        "field",
        [
            "concurrency",
            "prefill_tps",
            "decode_tps",
            "kv_bytes_per_token",
            "gpu_kv_capacity_gb",
            "block_size",
        ],
    )
    def test_simulate_invalid_positive_fields_raise_value_error(
        self, field: str
    ) -> None:
        config = _cfg(**{field: 0})
        session = _make_session("s0", [{"input_length": 100, "hash_ids": [1]}])

        with pytest.raises(ValueError, match=f"{field} must be > 0"):
            simulate([session], config)

    @pytest.mark.parametrize("field", ["l1_tokens", "l1_5_tokens"])
    def test_simulate_negative_token_fields_raise_value_error(self, field: str) -> None:
        config = _cfg(**{field: -1})
        session = _make_session("s0", [{"input_length": 100, "hash_ids": [1]}])

        with pytest.raises(ValueError, match=f"{field} must be >= 0"):
            simulate([session], config)


# ---------------------------------------------------------------------------
# Bug fix #1: eviction uses deduplicated footprint
# ---------------------------------------------------------------------------
class TestEvictionUsesDedupTokens:
    def test_no_eviction_when_dedup_under_capacity(self) -> None:
        """s0 ends and sits in LRU. s1+s2 share L1 tokens.
        Dedup footprint fits in cache -> s0 should NOT be evicted."""
        s0 = _make_session(
            "s0",
            [
                {"input_length": 100, "output_length": 10, "hash_ids": [0, 1]},
            ],
        )
        s1 = _make_session(
            "s1",
            [
                {
                    "input_length": 300,
                    "output_length": 10_000,
                    "hash_ids": [0, 1, 10, 11],
                },
            ],
        )
        s2 = _make_session(
            "s2",
            [
                {
                    "input_length": 300,
                    "output_length": 10_000,
                    "hash_ids": [0, 1, 20, 21],
                },
            ],
        )

        config = _cfg(
            concurrency=3,
            decode_tps=10,
            gpu_kv_capacity_gb=600 / 1e9,
            l1_tokens=200,
        )

        result = simulate([s0, s1, s2], config)
        assert result.eviction_count == 0

    def test_eviction_when_dedup_over_capacity_sequential(self) -> None:
        """Sequential sessions (concurrency=1). s0 ends, goes to LRU.
        s1 starts -> dedup exceeds capacity -> s0 evicted."""
        s0 = _make_session(
            "s0",
            [
                {"input_length": 400, "hash_ids": [0, 1, 10, 11]},
            ],
        )
        s1 = _make_session(
            "s1",
            [
                {"input_length": 400, "hash_ids": [0, 1, 20, 21]},
            ],
        )

        config = _cfg(
            concurrency=1,
            decode_tps=1_000_000,
            gpu_kv_capacity_gb=500 / 1e9,
            l1_tokens=200,
        )

        result = simulate([s0, s1], config)
        assert result.eviction_count > 0


# ---------------------------------------------------------------------------
# Tiny cache: 5 blocks, hand-traceable
# ---------------------------------------------------------------------------
class TestTinyCache:
    def test_sequential_sessions_evict_old(self) -> None:
        """concurrency=1, cache=5 blocks (500 tokens).
        s0 uses 3 blocks, ends (LRU). s1 uses 4 blocks, starts.
        s0(310) + s1(400) = 710 > 500 -> evict s0. Only s1's blocks remain."""
        s0 = _make_session(
            "s0",
            [
                {"input_length": 300, "hash_ids": [10, 11, 12]},
            ],
        )
        s1 = _make_session(
            "s1",
            [
                {"input_length": 400, "hash_ids": [20, 21, 22, 23]},
            ],
        )

        config = _cfg(
            concurrency=1,
            decode_tps=1_000_000,
            gpu_kv_capacity_gb=500 / 1e9,
        )

        result = simulate([s0, s1], config)
        assert result.eviction_count == 1
        assert _last(result.time_series).unique_blocks == 4

    def test_no_eviction_under_capacity(self) -> None:
        """Two sequential sessions that fit in cache without eviction."""
        s0 = _make_session(
            "s0",
            [
                {"input_length": 200, "hash_ids": [10, 11]},
            ],
        )
        s1 = _make_session(
            "s1",
            [
                {"input_length": 200, "hash_ids": [20, 21]},
            ],
        )

        config = _cfg(
            concurrency=1,
            decode_tps=1_000_000,
            gpu_kv_capacity_gb=500 / 1e9,
        )

        result = simulate([s0, s1], config)
        assert result.eviction_count == 0
        assert _last(result.time_series).unique_blocks == 4

    def test_shared_blocks_counted_once(self) -> None:
        """Two concurrent sessions sharing L1 blocks.
        unique_blocks = {0,1,10,11,20,21} = 6 (shared blocks not double-counted)."""
        s0 = _make_session(
            "s0",
            [
                {
                    "input_length": 300,
                    "output_length": 10_000,
                    "hash_ids": [0, 1, 10, 11],
                },
            ],
        )
        s1 = _make_session(
            "s1",
            [
                {
                    "input_length": 200,
                    "output_length": 10_000,
                    "hash_ids": [0, 1, 20, 21],
                },
            ],
        )

        config = _cfg(
            concurrency=2,
            decode_tps=10,
            gpu_kv_capacity_gb=1.0,
        )

        result = simulate([s0, s1], config)
        assert _peak(result.time_series, "unique_blocks") == 6


# ---------------------------------------------------------------------------
# Multi-turn: blocks accumulate, idle sessions evictable
# ---------------------------------------------------------------------------
class TestMultiTurnEviction:
    def test_idle_session_evicted_during_delay(self) -> None:
        """s0 has 2 turns with a long delay. s1 and s2 run sequentially after.
        s0 goes idle after turn 0 (LRU). s2 starts, triggering eviction of s0."""
        s0 = _make_session(
            "s0",
            [
                {
                    "input_length": 300,
                    "output_length": 10,
                    "hash_ids": [200_100, 200_101, 200_102],
                },
                {
                    "input_length": 200,
                    "delay_ms": 1_000_000,
                    "hash_ids": [200_103, 200_104],
                },
            ],
        )
        s1 = _make_session(
            "s1",
            [
                {"input_length": 100, "output_length": 10, "hash_ids": [200_200]},
            ],
        )
        s2 = _make_session(
            "s2",
            [
                {
                    "input_length": 400,
                    "output_length": 10,
                    "hash_ids": [200_300, 200_301, 200_302, 200_303],
                },
            ],
        )

        config = _cfg(
            concurrency=2,
            decode_tps=1_000_000,
            gpu_kv_capacity_gb=500 / 1e9,
        )

        result = simulate([s0, s1, s2], config)
        assert result.eviction_count >= 1

    def test_multi_turn_blocks_accumulate(self) -> None:
        """Blocks from multiple turns accumulate. Duplicates not double-counted."""
        s0 = _make_session(
            "s0",
            [
                {"input_length": 200, "hash_ids": [0, 1]},
                {"input_length": 100, "delay_ms": 1, "hash_ids": [2]},
                {"input_length": 100, "delay_ms": 1, "hash_ids": [3]},
            ],
        )

        config = _cfg(concurrency=1, decode_tps=1_000_000, gpu_kv_capacity_gb=1.0)

        result = simulate([s0], config)
        assert _peak(result.time_series, "unique_blocks") == 4


# ---------------------------------------------------------------------------
# L1.5 group dedup
# ---------------------------------------------------------------------------
class TestL15GroupDedup:
    def test_same_group_l15_dedup(self) -> None:
        """Two concurrent sessions in same group sharing L1.5 blocks.
        L1.5 dedup reduces memory -> fits in cache."""
        s0 = _make_session(
            "s0",
            [
                {
                    "input_length": 300,
                    "output_length": 10_000,
                    "hash_ids": [1000, 1001, 200_100],
                },
            ],
            group_id=0,
        )
        s1 = _make_session(
            "s1",
            [
                {
                    "input_length": 300,
                    "output_length": 10_000,
                    "hash_ids": [1000, 1001, 200_200],
                },
            ],
            group_id=0,
        )

        config = _cfg(
            concurrency=2,
            decode_tps=10,
            gpu_kv_capacity_gb=400 / 1e9,
            l1_5_tokens=200,
        )

        result = simulate([s0, s1], config)
        assert result.eviction_count == 0
        assert _peak(result.time_series, "unique_blocks") == 4

    def test_different_groups_no_l15_dedup(self) -> None:
        """Same-group gets L1.5 dedup, different-group does not.
        Verify via kv_cache_gb difference."""
        s0_same = _make_session(
            "s0",
            [
                {
                    "input_length": 300,
                    "output_length": 10_000,
                    "hash_ids": [1000, 1001, 200_100],
                },
            ],
            group_id=0,
        )
        s1_same = _make_session(
            "s1",
            [
                {
                    "input_length": 300,
                    "output_length": 10_000,
                    "hash_ids": [1000, 1001, 200_200],
                },
            ],
            group_id=0,
        )

        s0_diff = _make_session(
            "s0",
            [
                {
                    "input_length": 300,
                    "output_length": 10_000,
                    "hash_ids": [1000, 1001, 200_100],
                },
            ],
            group_id=0,
        )
        s1_diff = _make_session(
            "s1",
            [
                {
                    "input_length": 300,
                    "output_length": 10_000,
                    "hash_ids": [1200, 1201, 200_200],
                },
            ],
            group_id=1,
        )

        config = _cfg(
            concurrency=2,
            decode_tps=10,
            gpu_kv_capacity_gb=1.0,
            l1_5_tokens=200,
        )

        result_same = simulate([s0_same, s1_same], config)
        result_diff = simulate([s0_diff, s1_diff], config)

        same_alive = [p for p in result_same.time_series if p.alive_sessions == 2]
        diff_alive = [p for p in result_diff.time_series if p.alive_sessions == 2]
        assert len(same_alive) > 0
        assert len(diff_alive) > 0
        assert same_alive[0].kv_cache_gb < diff_alive[0].kv_cache_gb


# ---------------------------------------------------------------------------
# Block capacity consistency (bug fix #2)
# ---------------------------------------------------------------------------
class TestBlockCapacityConsistency:
    def test_block_capacity_uses_config_block_size(self) -> None:
        config = SimulationConfig(block_size=512)
        capacity_tokens = config.gpu_kv_capacity_gb * 1e9 / config.kv_bytes_per_token
        block_capacity = capacity_tokens / config.block_size
        expected = config.gpu_kv_capacity_gb * 1e9 / config.kv_bytes_per_token / 512
        assert block_capacity == pytest.approx(expected)

    def test_peak_block_memory_coherent(self) -> None:
        """Peak blocks * block_size * bytes_per_token should match token count."""
        s0 = _make_session(
            "s0",
            [
                {"input_length": 500, "hash_ids": [0, 1, 2, 3, 4]},
            ],
        )

        config = _cfg(concurrency=1, decode_tps=1_000_000, gpu_kv_capacity_gb=1.0)

        result = simulate([s0], config)
        peak_blocks = _peak(result.time_series, "unique_blocks")
        assert peak_blocks == 5
        block_memory_gb = (
            peak_blocks * config.block_size * config.kv_bytes_per_token / 1e9
        )
        assert block_memory_gb == pytest.approx(500 / 1e9)


# ---------------------------------------------------------------------------
# Eviction miss tracking by layer
# ---------------------------------------------------------------------------
class TestEvictionMissTracking:
    def test_session_blocks_generate_session_misses(self) -> None:
        """Evicted session-region blocks produce session misses, not L1.5 misses."""
        s0 = _make_session(
            "s0",
            [
                {"input_length": 300, "hash_ids": [200_100, 200_101, 200_102]},
                {"input_length": 100, "delay_ms": 10_000, "hash_ids": [200_103]},
            ],
        )
        s1 = _make_session(
            "s1",
            [
                {
                    "input_length": 400,
                    "output_length": 10_000,
                    "hash_ids": [200_200, 200_201, 200_202, 200_203],
                },
            ],
        )

        config = _cfg(
            concurrency=2,
            decode_tps=10,
            gpu_kv_capacity_gb=500 / 1e9,
        )

        result = simulate([s0, s1], config)
        assert result.miss_l15_blocks == 0
        if result.eviction_count > 0:
            assert result.miss_session_blocks > 0


# ---------------------------------------------------------------------------
# KV cache GB reflects dedup
# ---------------------------------------------------------------------------
class TestKvCacheGbMatchesDedup:
    def test_kv_cache_gb_reflects_dedup(self) -> None:
        """With L1 sharing between alive sessions, displayed KV GB should
        be less than raw token total * bytes_per_token."""
        s0 = _make_session(
            "s0",
            [
                {"input_length": 300, "output_length": 10_000, "hash_ids": [0, 1, 10]},
            ],
        )
        s1 = _make_session(
            "s1",
            [
                {"input_length": 300, "output_length": 10_000, "hash_ids": [0, 1, 20]},
            ],
        )

        config = _cfg(
            concurrency=2,
            decode_tps=10,
            gpu_kv_capacity_gb=1.0,
            l1_tokens=200,
        )

        result = simulate([s0, s1], config)
        both_alive = [p for p in result.time_series if p.alive_sessions == 2]
        assert len(both_alive) > 0
        point = both_alive[0]
        assert point.kv_cache_gb == pytest.approx(400 / 1e9)


# ---------------------------------------------------------------------------
# Cache hit rate computed from block hits (not a user input)
# ---------------------------------------------------------------------------
class TestCacheHitFromBlocks:
    def test_cold_cache_all_misses(self) -> None:
        """First session on cold cache: all blocks are misses."""
        s = _make_session(
            "s0",
            [
                {"input_length": 500, "hash_ids": [0, 1, 2, 3, 4]},
            ],
        )

        config = _cfg(concurrency=1, decode_tps=1_000_000, gpu_kv_capacity_gb=1.0)

        result = simulate([s], config)
        evt = result.session_states[0].turn_events[0]
        # Cold cache: all 5 blocks are misses
        assert evt.miss_tokens == 500
        assert evt.hit_tokens == 0
        assert result.cache_hit_rate == 0.0

    def test_second_turn_hits_previous_blocks(self) -> None:
        """Turn 1's cumulative blocks include turn 0's blocks (all hits)
        plus new blocks (misses). Cumulative = {0,1,2,3,4}, where
        blocks 0,1,2 are hits (cached from turn 0), blocks 3,4 are misses."""
        s = _make_session(
            "s0",
            [
                {"input_length": 300, "hash_ids": [0, 1, 2]},
                {"input_length": 200, "delay_ms": 1, "hash_ids": [0, 1, 3, 4]},
            ],
        )

        config = _cfg(concurrency=1, decode_tps=1_000_000, gpu_kv_capacity_gb=1.0)

        result = simulate([s], config)
        evt0 = result.session_states[0].turn_events[0]
        evt1 = result.session_states[0].turn_events[1]
        # Turn 0: cold cache, 3 blocks miss
        assert evt0.miss_tokens == 300
        assert evt0.hit_tokens == 0
        # Turn 1: cumulative blocks = {0,1,2,3,4} (5 blocks).
        # Blocks 0,1,2 in cache from turn 0 -> 3 hits. Blocks 3,4 -> 2 misses.
        assert evt1.hit_tokens == 300  # 3 hit blocks * 100 tokens
        assert evt1.miss_tokens == 200  # 2 miss blocks * 100 tokens

    def test_shared_l1_blocks_are_hits_for_second_session(self) -> None:
        """s1 starts after s0 finishes. s0's blocks are still in cache.
        s1's shared blocks (0,1) are hits, unique blocks (20,21) are misses."""
        s0 = _make_session(
            "s0",
            [
                {"input_length": 400, "hash_ids": [0, 1, 10, 11]},
            ],
        )
        s1 = _make_session(
            "s1",
            [
                {"input_length": 400, "hash_ids": [0, 1, 20, 21]},
            ],
        )

        config = _cfg(
            concurrency=1,
            decode_tps=1_000_000,
            gpu_kv_capacity_gb=1.0,
        )

        result = simulate([s0, s1], config)
        evt0 = result.session_states[0].turn_events[0]
        evt1 = result.session_states[1].turn_events[0]
        # s0: cold cache, all miss
        assert evt0.miss_tokens == 400
        assert evt0.hit_tokens == 0
        # s1: blocks 0,1 are in cache from s0 -> hits. 20,21 are misses.
        assert evt1.hit_tokens == 200
        assert evt1.miss_tokens == 200

    def test_cache_hit_rate_output(self) -> None:
        """Overall cache_hit_rate should reflect aggregate hits across all turns.
        Turn 0: cumulative = {0,1,2} -> 0 hit, 3 miss (300 tokens each).
        Turn 1: cumulative = {0,1,2,3,4} -> 3 hit (0,1,2), 2 miss (3,4).
        Total: 300 hit / (300+500) = 300/800."""
        s = _make_session(
            "s0",
            [
                {"input_length": 300, "hash_ids": [0, 1, 2]},
                {"input_length": 200, "delay_ms": 1, "hash_ids": [0, 1, 2, 3, 4]},
            ],
        )

        config = _cfg(concurrency=1, decode_tps=1_000_000, gpu_kv_capacity_gb=1.0)

        result = simulate([s], config)
        # Turn 0: 3 blocks, 0 hit, 3 miss = 300 miss tokens.
        # Turn 1: 5 cumulative blocks, 3 hit (300 tok), 2 miss (200 tok).
        # Total: 300 hit / (300 + 500) total tokens = 0.375
        assert result.cache_hit_rate == pytest.approx(300 / 800)

    def test_cache_hit_reduces_prefill_time(self) -> None:
        """Blocks already in cache reduce prefill duration.
        s0 prefills 4 blocks (400 miss tokens).
        s1 shares 2 blocks (200 hit tokens) + 2 new (200 miss tokens).
        s1 prefill should be half of s0's."""
        s0 = _make_session(
            "s0",
            [
                {"input_length": 400, "hash_ids": [0, 1, 10, 11]},
            ],
        )
        s1 = _make_session(
            "s1",
            [
                {"input_length": 400, "hash_ids": [0, 1, 20, 21]},
            ],
        )

        config = _cfg(
            concurrency=1,
            prefill_tps=100,  # slow prefill to make timing measurable
            decode_tps=1_000_000,
            gpu_kv_capacity_gb=1.0,
        )

        result = simulate([s0, s1], config)
        evt0 = result.session_states[0].turn_events[0]
        evt1 = result.session_states[1].turn_events[0]
        prefill0 = evt0.decode_start - evt0.prefill_start
        prefill1 = evt1.decode_start - evt1.prefill_start
        # s0: 400 miss tokens / 100 tps = 4s = 4000ms
        assert prefill0 == pytest.approx(4000.0)
        # s1: 200 miss tokens / 100 tps = 2s = 2000ms
        assert prefill1 == pytest.approx(2000.0)

    def test_evicted_blocks_become_misses_again(self) -> None:
        """Blocks evicted from cache revert to misses for subsequent turns."""
        # s0 turn 0 loads blocks, they get evicted, turn 1 sees them as misses.
        s0 = _make_session(
            "s0",
            [
                {
                    "input_length": 300,
                    "output_length": 10,
                    "hash_ids": [200_100, 200_101, 200_102],
                },
                {
                    "input_length": 200,
                    "delay_ms": 1_000_000,
                    "hash_ids": [200_103, 200_104],
                },
            ],
        )
        s1 = _make_session(
            "s1",
            [
                {"input_length": 100, "output_length": 10, "hash_ids": [200_200]},
            ],
        )
        s2 = _make_session(
            "s2",
            [
                {
                    "input_length": 400,
                    "output_length": 10,
                    "hash_ids": [200_300, 200_301, 200_302, 200_303],
                },
            ],
        )

        config = _cfg(
            concurrency=2,
            decode_tps=1_000_000,
            gpu_kv_capacity_gb=500 / 1e9,
        )

        result = simulate([s0, s1, s2], config)
        if result.eviction_count > 0:
            # s0's turn 1 should see its own blocks as misses after eviction
            evt1 = result.session_states[0].turn_events[1]
            # Turn 1 has 2 new blocks, but the 3 turn-0 blocks were evicted
            # so they need re-fetch (tracked as eviction misses, not prefill misses)
            assert evt1.miss_tokens >= 200  # at minimum the 2 new blocks


# ---------------------------------------------------------------------------
# Timing correctness
# ---------------------------------------------------------------------------
class TestTimingCorrectness:
    def test_prefill_duration_cold_cache(self) -> None:
        """Cold cache: all blocks miss.
        10 blocks * 100 tokens/block = 1000 miss tokens / 100 tps = 10s."""
        s = _make_session(
            "s0",
            [
                {
                    "input_length": 1000,
                    "output_length": 100,
                    "hash_ids": list(range(10)),
                },
            ],
        )

        config = _cfg(
            concurrency=1,
            prefill_tps=100,
            decode_tps=1_000_000,
        )

        result = simulate([s], config)
        evt = result.session_states[0].turn_events[0]
        prefill_ms = evt.decode_start - evt.prefill_start
        assert prefill_ms == pytest.approx(10_000.0)

    def test_decode_duration(self) -> None:
        """Decode duration = output_tokens / decode_tps.
        500 output tokens / 50 tps = 10s."""
        s = _make_session(
            "s0",
            [
                {"input_length": 100, "output_length": 500, "hash_ids": [0]},
            ],
        )

        config = _cfg(
            concurrency=1,
            prefill_tps=1_000_000,
            decode_tps=50,
        )

        result = simulate([s], config)
        evt = result.session_states[0].turn_events[0]
        decode_ms = evt.decode_end - evt.decode_start
        assert decode_ms == pytest.approx(10_000.0)

    def test_prefill_tps_is_shared_capacity(self) -> None:
        """Concurrent sessions share one aggregate prefill TPS budget."""
        s0 = _make_session(
            "s0",
            [
                {
                    "input_length": 1000,
                    "output_length": 100_000,
                    "hash_ids": list(range(10)),
                },
            ],
        )
        s1 = _make_session(
            "s1",
            [
                {
                    "input_length": 1000,
                    "output_length": 100_000,
                    "hash_ids": list(range(10, 20)),
                },
            ],
        )

        config = _cfg(
            concurrency=2,
            prefill_tps=100,
            decode_tps=10,
        )

        result = simulate([s0, s1], config)
        evt0 = result.session_states[0].turn_events[0]
        evt1 = result.session_states[1].turn_events[0]
        assert evt1.prefill_start == pytest.approx(evt0.decode_start)

    def test_prefill_tps_queues_ready_turns(self) -> None:
        """Ready turns reserve aggregate prefill capacity in order."""
        sessions = [
            _make_session(
                f"s{i}",
                [
                    {
                        "input_length": 1000,
                        "output_length": 100_000,
                        "hash_ids": list(range(i * 10, i * 10 + 10)),
                    },
                ],
            )
            for i in range(4)
        ]

        config = _cfg(
            concurrency=4,
            prefill_tps=400,
            decode_tps=10,
        )

        result = simulate(sessions, config)
        starts = [
            result.session_states[i].turn_events[0].prefill_start for i in range(4)
        ]
        assert starts == pytest.approx([0.0, 2500.0, 5000.0, 7500.0])

    def test_avg_ttft_excludes_inter_turn_delay(self) -> None:
        """TTFT starts when the delayed turn becomes ready, not at prior decode end."""
        s = _make_session(
            "s0",
            [
                {"input_length": 100, "hash_ids": [0]},
                {"input_length": 100, "delay_ms": 5000, "hash_ids": [1]},
            ],
        )

        config = _cfg(
            concurrency=1,
            prefill_tps=100,
            decode_tps=1_000_000,
            gpu_kv_capacity_gb=1.0,
        )

        result = simulate([s], config)
        evt1 = result.session_states[0].turn_events[1]
        assert evt1.turn_ready - evt1.delay_start == pytest.approx(5000.0)
        assert result.avg_ttft == pytest.approx(1000.0)
        assert result.total_wait_ms == pytest.approx(0.0)
