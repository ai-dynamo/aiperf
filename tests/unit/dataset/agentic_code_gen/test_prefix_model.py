# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the prefix model."""

from __future__ import annotations

import math

import pytest

from aiperf.dataset.agentic_code_gen.distributions import lognormal_from_mean_median
from aiperf.dataset.agentic_code_gen.models import CacheLayerConfig, Layer15GroupConfig
from aiperf.dataset.agentic_code_gen.prefix_model import (
    MAX_GROUP_BLOCKS,
    MAX_GROUPS,
    MAX_SESSION_BLOCKS,
    PrefixAllocator,
)


@pytest.fixture
def allocator() -> PrefixAllocator:
    return PrefixAllocator(
        CacheLayerConfig(
            layer1_tokens=32_000,
            layer1_5_tokens=20_000,
            layer2=lognormal_from_mean_median(mean=10_000, median=5_000),
        ),
        block_size=512,
    )


@pytest.fixture
def small_allocator() -> PrefixAllocator:
    return PrefixAllocator(
        CacheLayerConfig(
            layer1_tokens=200,
            layer1_5_tokens=100,
            layer2=lognormal_from_mean_median(mean=200, median=150),
        ),
        block_size=64,
    )


class TestPrefixAllocator:
    def test_l1_block_count(self, allocator: PrefixAllocator) -> None:
        assert allocator.l1_blocks == math.ceil(32_000 / 512)  # 63

    def test_l15_block_count(self, allocator: PrefixAllocator) -> None:
        assert allocator.l15_blocks == math.ceil(20_000 / 512)  # 40

    def test_l1_ids_identical_across_sessions(self, allocator: PrefixAllocator) -> None:
        ids_0 = allocator.turn_hash_ids(
            session_index=0, group_id=0, input_length=80_000
        )
        ids_1 = allocator.turn_hash_ids(
            session_index=1, group_id=1, input_length=80_000
        )
        l1_0 = ids_0[: allocator.l1_blocks]
        l1_1 = ids_1[: allocator.l1_blocks]
        assert l1_0 == l1_1

    def test_l15_ids_shared_within_group(self, allocator: PrefixAllocator) -> None:
        ids_0 = allocator.turn_hash_ids(
            session_index=0, group_id=3, input_length=80_000
        )
        ids_1 = allocator.turn_hash_ids(
            session_index=1, group_id=3, input_length=80_000
        )
        l1 = allocator.l1_blocks
        l15 = allocator.l15_blocks
        assert ids_0[l1 : l1 + l15] == ids_1[l1 : l1 + l15]

    def test_l15_ids_differ_across_groups(self, allocator: PrefixAllocator) -> None:
        ids_0 = allocator.turn_hash_ids(
            session_index=0, group_id=0, input_length=80_000
        )
        ids_1 = allocator.turn_hash_ids(
            session_index=1, group_id=1, input_length=80_000
        )
        l1 = allocator.l1_blocks
        l15 = allocator.l15_blocks
        assert ids_0[l1 : l1 + l15] != ids_1[l1 : l1 + l15]

    def test_session_ids_grow_with_turns(self, allocator: PrefixAllocator) -> None:
        ids_t0 = allocator.turn_hash_ids(
            session_index=0, group_id=0, input_length=80_000
        )
        sess_t0 = allocator.extract_session_ids(ids_t0)
        ids_t1 = allocator.turn_hash_ids(
            session_index=0, group_id=0, input_length=90_000, prev_session_ids=sess_t0
        )
        sess_t1 = allocator.extract_session_ids(ids_t1)
        assert len(sess_t1) > len(sess_t0)
        assert sess_t1[: len(sess_t0)] == sess_t0

    def test_turn_n_is_prefix_of_turn_n_plus_1(
        self, allocator: PrefixAllocator
    ) -> None:
        ids_t0 = allocator.turn_hash_ids(
            session_index=0, group_id=0, input_length=80_000
        )
        sess_t0 = allocator.extract_session_ids(ids_t0)
        ids_t1 = allocator.turn_hash_ids(
            session_index=0, group_id=0, input_length=100_000, prev_session_ids=sess_t0
        )
        assert ids_t1[: len(ids_t0)] == ids_t0

    def test_zero_input_length_no_blocks(self, allocator: PrefixAllocator) -> None:
        ids = allocator.turn_hash_ids(session_index=0, group_id=0, input_length=0)
        assert ids == []

    def test_hash_ids_count_equals_ceil_isl_over_block_size(
        self, allocator: PrefixAllocator
    ) -> None:
        for isl in [10_000, 37_000, 80_000, 100_000, 200_000]:
            ids = allocator.turn_hash_ids(session_index=0, group_id=0, input_length=isl)
            expected = math.ceil(isl / 512)
            assert len(ids) == expected, (
                f"ISL={isl}: got {len(ids)}, expected {expected}"
            )

    def test_small_input_only_uses_l1(self, allocator: PrefixAllocator) -> None:
        """Input smaller than L1 tokens should only produce L1 blocks."""
        ids = allocator.turn_hash_ids(session_index=0, group_id=0, input_length=10_000)
        expected = math.ceil(10_000 / 512)  # 20
        assert len(ids) == expected
        assert all(i < allocator.l1_blocks for i in ids)

    def test_session_base_spacing(self, allocator: PrefixAllocator) -> None:
        base_0 = allocator.session_base(0)
        base_1 = allocator.session_base(1)
        assert base_1 - base_0 == MAX_SESSION_BLOCKS

    def test_max_session_blocks_boundary_allowed(
        self, allocator: PrefixAllocator
    ) -> None:
        input_length = (
            allocator.l1_blocks + allocator.l15_blocks + MAX_SESSION_BLOCKS
        ) * allocator.block_size

        ids = allocator.turn_hash_ids(
            session_index=0, group_id=0, input_length=input_length
        )
        session_ids = allocator.extract_session_ids(ids)

        assert len(session_ids) == MAX_SESSION_BLOCKS
        assert session_ids[-1] == allocator.session_base(1) - 1

    def test_max_session_blocks_overflow_raises(
        self, allocator: PrefixAllocator
    ) -> None:
        input_length = (
            allocator.l1_blocks + allocator.l15_blocks + MAX_SESSION_BLOCKS + 1
        ) * allocator.block_size

        with pytest.raises(ValueError, match="MAX_SESSION_BLOCKS"):
            allocator.turn_hash_ids(
                session_index=0, group_id=0, input_length=input_length
            )

    def test_build_hash_ids_rejects_missing_session_ids(
        self, allocator: PrefixAllocator
    ) -> None:
        input_length = (
            allocator.l1_blocks + allocator.l15_blocks + 2
        ) * allocator.block_size

        with pytest.raises(ValueError, match="needs 2 session IDs"):
            allocator.build_hash_ids(
                group_id=0,
                input_length=input_length,
                session_ids=[allocator.session_base(0)],
            )

    def test_no_hash_id_collisions_across_sessions(
        self, small_allocator: PrefixAllocator
    ) -> None:
        alloc = small_allocator
        ids_s0 = set(alloc.turn_hash_ids(session_index=0, group_id=0, input_length=500))
        ids_s1 = set(alloc.turn_hash_ids(session_index=1, group_id=0, input_length=500))
        l1 = set(range(alloc.l1_blocks))
        l15_base = alloc.group_base(0)
        l15 = set(range(l15_base, l15_base + alloc.l15_blocks))
        shared = l1 | l15
        overlap = ids_s0 & ids_s1
        assert overlap == shared

    def test_group_count_exceeding_reserved_regions_raises(self) -> None:
        config = CacheLayerConfig(
            layer1_tokens=0,
            layer1_5_tokens=0,
            layer2=lognormal_from_mean_median(mean=100, median=100),
            layer1_5_groups=Layer15GroupConfig(
                num_groups=MAX_GROUPS + 1,
                zipf_alpha=1.0,
            ),
        )

        with pytest.raises(ValueError, match="MAX_GROUPS"):
            PrefixAllocator(config, block_size=64)

    def test_l15_blocks_exceeding_reserved_region_raises(self) -> None:
        config = CacheLayerConfig(
            layer1_tokens=0,
            layer1_5_tokens=(MAX_GROUP_BLOCKS + 1) * 64,
            layer2=lognormal_from_mean_median(mean=100, median=100),
        )

        with pytest.raises(ValueError, match="MAX_GROUP_BLOCKS"):
            PrefixAllocator(config, block_size=64)

    def test_turn0_no_l3(self, allocator: PrefixAllocator) -> None:
        """Turn 0 should have L1 + L1.5 + session prefix blocks (no L3)."""
        ids = allocator.turn_hash_ids(session_index=0, group_id=0, input_length=80_000)
        l1_ids = ids[: allocator.l1_blocks]
        l15_ids = ids[allocator.l1_blocks : allocator.prefix_blocks]
        session_ids = allocator.extract_session_ids(ids)

        assert l1_ids == list(range(allocator.l1_blocks))
        assert len(l15_ids) == allocator.l15_blocks
        assert len(l1_ids) + len(l15_ids) + len(session_ids) == len(ids)

        base = allocator.session_base(0)
        expected_session = list(range(base, base + len(session_ids)))
        assert session_ids == expected_session

    def test_growing_input_maintains_prefix_property(
        self, allocator: PrefixAllocator
    ) -> None:
        """Simulate 5 turns of growing input, verify prefix property throughout."""
        prev_ids = None
        prev_session = None
        isl = 80_000
        for _ in range(5):
            ids = allocator.turn_hash_ids(
                session_index=0,
                group_id=0,
                input_length=isl,
                prev_session_ids=prev_session,
            )
            assert len(ids) == math.ceil(isl / allocator.block_size)
            if prev_ids is not None:
                assert ids[: len(prev_ids)] == prev_ids
            prev_session = allocator.extract_session_ids(ids)
            prev_ids = ids
            isl += 10_000
