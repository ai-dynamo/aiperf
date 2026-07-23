# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for worker-chain classification of detected chains."""

from aiperf.dataset.loader.weka_agent_chains import (
    AgentChain,
    ChainDetectionResult,
    ChainFork,
    is_aux_chain,
    is_reduction_chain,
    worker_group_assignment,
    worker_group_members,
)
from aiperf.dataset.loader.weka_trace import _worker_suffix
from aiperf.dataset.loader.weka_trace_models import WekaNormalRequest

PARAMS = {"max_requests": 1, "isl_ratio": 0.10, "isl_floor": 16384}
MAIN_PEAK = 134_144


def _chain(*input_lengths: int) -> list[WekaNormalRequest]:
    """A worker chain's request list, one request per given input length."""
    return [
        WekaNormalRequest(
            type="n",
            t=float(i),
            model="m",
            input_length=isl,
            output_length=10,
            hash_ids=[1],
        )
        for i, isl in enumerate(input_lengths)
    ]


def test_singleton_small_fresh_context_is_aux():
    assert is_aux_chain(_chain(2944), MAIN_PEAK, **PARAMS) is True


def test_multi_request_chain_is_not_aux():
    assert is_aux_chain(_chain(2944, 2944), MAIN_PEAK, **PARAMS) is False


def test_large_context_singleton_is_not_aux():
    assert is_aux_chain(_chain(20_000), MAIN_PEAK, **PARAMS) is False


def test_floor_applies_when_main_context_small():
    assert is_aux_chain(_chain(5_000), 1_000, **PARAMS) is True


def test_relative_ratio_governs_above_floor():
    assert (
        is_aux_chain(
            _chain(15_000), 200_000, max_requests=1, isl_ratio=0.10, isl_floor=0
        )
        is True
    )
    assert (
        is_aux_chain(
            _chain(25_000), 200_000, max_requests=1, isl_ratio=0.10, isl_floor=0
        )
        is False
    )


def test_max_requests_zero_disables_classification():
    assert (
        is_aux_chain(
            _chain(100), MAIN_PEAK, max_requests=0, isl_ratio=0.10, isl_floor=16384
        )
        is False
    )


def test_threshold_is_strict_less_than():
    assert is_aux_chain(_chain(16384), MAIN_PEAK, **PARAMS) is False
    assert is_aux_chain(_chain(16383), MAIN_PEAK, **PARAMS) is True


def test_empty_chain_is_not_aux():
    assert is_aux_chain([], MAIN_PEAK, **PARAMS) is False


def test_cross_model_singleton_is_aux_regardless_of_size():
    big = _chain(200_000)
    assert is_aux_chain(big, MAIN_PEAK, main_model="m", **PARAMS) is False
    assert is_aux_chain(big, MAIN_PEAK, main_model="opus", **PARAMS) is True


def test_cross_model_arm_can_be_disabled():
    big = _chain(200_000)
    assert (
        is_aux_chain(big, MAIN_PEAK, main_model="opus", cross_model=False, **PARAMS)
        is False
    )


def test_cross_model_only_reclassifies_short_chains():
    multi = _chain(200_000, 200_000)
    assert is_aux_chain(multi, MAIN_PEAK, main_model="opus", **PARAMS) is False


def test_no_main_model_skips_cross_model_arm():
    assert is_aux_chain(_chain(200_000), MAIN_PEAK, **PARAMS) is False


RED = {"osl_max": 4000, "ratio": 20.0, "isl_floor": 16384}


def _one(isl: int, osl: int, model: str = "m") -> list[WekaNormalRequest]:
    """A single-request worker chain with explicit input/output lengths."""
    return [
        WekaNormalRequest(
            type="n",
            t=0.0,
            model=model,
            input_length=isl,
            output_length=osl,
            hash_ids=[1],
        )
    ]


def test_reduction_large_in_short_out_is_aux():
    assert is_reduction_chain(_one(41_280, 330), **RED) is True


def test_reduction_generative_output_is_not_reduction():
    assert is_reduction_chain(_one(41_280, 8_000), **RED) is False


def test_reduction_requires_large_input():
    assert is_reduction_chain(_one(5_000, 100), **RED) is False


def test_reduction_requires_high_ratio():
    assert is_reduction_chain(_one(30_000, 2_000), **RED) is False
    assert is_reduction_chain(_one(30_000, 100), **RED) is True


def test_reduction_only_single_request():
    two = _one(41_280, 330) + _one(41_280, 330)
    assert is_reduction_chain(two, **RED) is False


def test_reduction_zero_output_is_not_reduction():
    assert is_reduction_chain(_one(41_280, 0), **RED) is False


def test_reduction_osl_max_zero_disables():
    assert (
        is_reduction_chain(_one(41_280, 330), osl_max=0, ratio=20.0, isl_floor=16384)
        is False
    )


def test_reduction_osl_bound_is_strict():
    assert is_reduction_chain(_one(200_000, 4_000), **RED) is False
    assert is_reduction_chain(_one(200_000, 3_999), **RED) is True


def test_reduction_empty_chain_is_not_reduction():
    assert is_reduction_chain([], **RED) is False


def _worker(
    t: float = 0.0, dur: float = 10.0, fork_depth: int = 100, fork_outer: int = 0
) -> AgentChain:
    """A worker chain forked from shared context (``fork_depth`` blocks) at fork"""
    return AgentChain(
        requests=[
            (
                0,
                WekaNormalRequest(
                    type="n",
                    t=t,
                    api_time=dur,
                    model="m",
                    input_length=50_000,
                    output_length=200,
                    hash_ids=[1, 2, 3],
                ),
            )
        ],
        fork=ChainFork(
            parent_chain=0, fork_outer_idx=fork_outer, depth=fork_depth, fork_time=t
        ),
    )


def _result(*workers: AgentChain) -> ChainDetectionResult:
    """A detection result: chain 0 is main, the rest are workers."""
    main = AgentChain(
        requests=[
            (
                0,
                WekaNormalRequest(
                    type="n",
                    t=0.0,
                    model="m",
                    input_length=80_000,
                    output_length=200,
                    hash_ids=[99],
                ),
            )
        ]
    )
    chains = [main, *workers]
    return ChainDetectionResult(
        chains=chains,
        main_index=0,
        worker_indices=list(range(1, len(chains))),
        seams_merged=0,
        unclassified_empty_hash=0,
    )


def test_worker_group_overlapping_intervals_are_members():
    r = _result(_worker(0, 100), _worker(10, 100), _worker(20, 100))
    assert worker_group_members(r, group_min=3) == {1, 2, 3}


def test_worker_group_non_overlapping_not_grouped():
    r = _result(_worker(0, 50), _worker(100, 50), _worker(200, 50))
    assert worker_group_members(r, group_min=3) == set()


def test_worker_group_staggered_but_overlapping_still_one_group():
    r = _result(_worker(0, 300), _worker(100, 300), _worker(200, 300))
    assert worker_group_members(r, group_min=3) == {1, 2, 3}


def test_worker_group_transitive_overlap_chains_into_one_group():
    r = _result(_worker(0, 100), _worker(90, 100), _worker(180, 100))
    assert worker_group_members(r, group_min=3) == {1, 2, 3}


def test_worker_group_overlap_does_not_bridge_distinct_fork_points():
    r = _result(
        _worker(0, 100, fork_outer=5),
        _worker(1, 100, fork_outer=5),
        _worker(2, 100, fork_outer=5),
        _worker(0, 100, fork_outer=9),
        _worker(1, 100, fork_outer=9),
        _worker(2, 100, fork_outer=9),
    )
    a = worker_group_assignment(r, group_min=3)
    assert len({a[c][0] for c in range(1, 7)}) == 2
    assert {a[c][0] for c in (1, 2, 3)} != {a[c][0] for c in (4, 5, 6)}


def test_worker_group_below_min_not_members():
    r = _result(_worker(0, 100), _worker(10, 100))
    assert worker_group_members(r, group_min=3) == set()


def test_worker_group_requires_fork_depth():
    r = _result(_worker(0, 100, 0), _worker(10, 100, 0), _worker(20, 100, 0))
    assert worker_group_members(r, group_min=3) == set()


def test_worker_group_fork_none_never_groups():
    solo = _worker(0, 100)
    solo.fork = None
    r = _result(solo, _worker(10, 100), _worker(20, 100))
    assert worker_group_members(r, group_min=3) == set()


def test_worker_group_min_zero_disables():
    r = _result(_worker(0, 100), _worker(10, 100), _worker(20, 100))
    assert worker_group_members(r, group_min=0) == set()


def test_worker_group_assignment_group_and_member():
    r = _result(_worker(0, 100), _worker(1, 100), _worker(2, 100))
    assert worker_group_assignment(r, group_min=3) == {
        1: (0, 0),
        2: (0, 1),
        3: (0, 2),
    }


def test_worker_group_assignment_two_disjoint_components_two_groups():
    r = _result(
        _worker(0, 100),
        _worker(1, 100),
        _worker(2, 100),
        _worker(1000, 100),
        _worker(1001, 100),
        _worker(1002, 100),
    )
    a = worker_group_assignment(r, group_min=3)
    assert {a[c][0] for c in (1, 2, 3)} == {0}
    assert {a[c][0] for c in (4, 5, 6)} == {1}
    assert a[4] == (1, 0) and a[6] == (1, 2)


def test_worker_group_assignment_groups_ordered_by_first_start():
    r = _result(
        _worker(1000, 100),
        _worker(1001, 100),
        _worker(1002, 100),
        _worker(0, 100),
        _worker(1, 100),
        _worker(2, 100),
    )
    a = worker_group_assignment(r, group_min=3)
    assert {a[c][0] for c in (4, 5, 6)} == {0}
    assert {a[c][0] for c in (1, 2, 3)} == {1}


def test_worker_group_members_matches_assignment_keys():
    r = _result(
        _worker(0, 100), _worker(10, 100), _worker(20, 100), _worker(30, 100, 0)
    )
    assert worker_group_members(r, group_min=3) == {1, 2, 3}
    assert worker_group_members(r, group_min=3) == set(
        worker_group_assignment(r, group_min=3)
    )


def test_worker_suffix_precedence_and_shape():
    assert (
        _worker_suffix(n=2, is_aux=True, is_reduction=True, wg_coord=(1, 0))
        == "aux:002"
    )
    assert (
        _worker_suffix(n=3, is_aux=False, is_reduction=True, wg_coord=(1, 0))
        == "aux:003"
    )
    assert (
        _worker_suffix(n=5, is_aux=False, is_reduction=False, wg_coord=(1, 2))
        == "wg:001_002"
    )
    assert (
        _worker_suffix(n=7, is_aux=False, is_reduction=False, wg_coord=None) == "fa:007"
    )
