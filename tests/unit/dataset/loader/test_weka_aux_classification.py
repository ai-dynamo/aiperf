# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for worker-chain classification of detected chains.

Three predicates partition detected worker chains into agent vs auxiliary vs
parallel fan-out:

- ``is_aux_chain``: a short chain from a small fresh context, or a cross-model
  one-shot, is a tool-issued sidecar (``::fa:`` -> ``::aux:``).
- ``is_reduction_chain``: a same-model single large-input/short-output one-shot
  is a reduction sidecar (``::fa:`` -> ``::aux:``).
- ``worker_group_assignment`` / ``worker_group_members``: among workers that
  forked from shared context, those whose active intervals OVERLAP form a
  concurrent parallel fan-out, tagged ``::fa:`` -> ``::wg:{group}_{member}``.

All apply at both layers (top-level flat chains and subagent overflow). The
``_worker_suffix`` helper resolves the precedence (aux > reduction > worker-
group > solo agent) and builds the marker + index/coordinate.
"""

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

# Defaults mirroring Environment.DATASET.WEKA_AUX_* so the tests read as the
# production predicate; individual cases override what they exercise.
PARAMS = {"max_requests": 1, "isl_ratio": 0.10, "isl_floor": 16384}
MAIN_PEAK = 134_144  # representative main-chain peak context (corpus median-ish)


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
    # one request, ~3k tokens vs a 134k main context -> sidecar
    assert is_aux_chain(_chain(2944), MAIN_PEAK, **PARAMS) is True


def test_multi_request_chain_is_not_aux():
    # two requests exceeds max_requests=1 -> sustained agent, stays ::fa:
    assert is_aux_chain(_chain(2944, 2944), MAIN_PEAK, **PARAMS) is False


def test_large_context_singleton_is_not_aux():
    # 20k > max(floor 16384, 0.1 * 134144 = 13414) -> a real one-shot agent
    assert is_aux_chain(_chain(20_000), MAIN_PEAK, **PARAMS) is False


def test_floor_applies_when_main_context_small():
    # main peak tiny -> ratio term ~0, floor (16384) governs; 5k < 16384 -> aux
    assert is_aux_chain(_chain(5_000), 1_000, **PARAMS) is True


def test_relative_ratio_governs_above_floor():
    # floor disabled -> threshold = 0.10 * 200000 = 20000
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
    # the escape hatch: nothing is ever aux, every chain stays ::fa:
    assert (
        is_aux_chain(
            _chain(100), MAIN_PEAK, max_requests=0, isl_ratio=0.10, isl_floor=16384
        )
        is False
    )


def test_threshold_is_strict_less_than():
    # input length exactly at the threshold is not below it -> not aux
    assert is_aux_chain(_chain(16384), MAIN_PEAK, **PARAMS) is False
    assert is_aux_chain(_chain(16383), MAIN_PEAK, **PARAMS) is True


def test_empty_chain_is_not_aux():
    assert is_aux_chain([], MAIN_PEAK, **PARAMS) is False


def test_cross_model_singleton_is_aux_regardless_of_size():
    # a one-shot on a different model than the main agent is a tool sidecar
    # (Haiku WebFetch under an Opus agent), even with a large fetched payload
    big = _chain(200_000)  # one request, model "m", ISL far above the floor
    assert is_aux_chain(big, MAIN_PEAK, main_model="m", **PARAMS) is False
    assert is_aux_chain(big, MAIN_PEAK, main_model="opus", **PARAMS) is True


def test_cross_model_arm_can_be_disabled():
    big = _chain(200_000)
    assert (
        is_aux_chain(big, MAIN_PEAK, main_model="opus", cross_model=False, **PARAMS)
        is False
    )


def test_cross_model_only_reclassifies_short_chains():
    # a multi-request cross-model chain is a genuine cross-model agent (e.g. a
    # Haiku Explore subagent looping), not a one-shot sidecar
    multi = _chain(200_000, 200_000)  # 2 requests > max_requests=1
    assert is_aux_chain(multi, MAIN_PEAK, main_model="opus", **PARAMS) is False


def test_no_main_model_skips_cross_model_arm():
    # without a main_model reference the cross-model arm is inert (size only)
    assert is_aux_chain(_chain(200_000), MAIN_PEAK, **PARAMS) is False


# --- reduction arm (is_reduction_chain) ---

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
    # 41k in, 330 out (ratio ~125) -> a reduction sidecar
    assert is_reduction_chain(_one(41_280, 330), **RED) is True


def test_reduction_generative_output_is_not_reduction():
    # a long completion is generative agent work, not a reduction
    assert is_reduction_chain(_one(41_280, 8_000), **RED) is False


def test_reduction_requires_large_input():
    # below the floor is the size arm's territory, not a reduction
    assert is_reduction_chain(_one(5_000, 100), **RED) is False


def test_reduction_requires_high_ratio():
    # 30k / 2000 = ratio 15 < 20 -> a balanced call, not a reduction
    assert is_reduction_chain(_one(30_000, 2_000), **RED) is False
    # 30k / 100 = ratio 300 -> a reduction
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
    # output exactly at osl_max is not below it -> not a reduction
    assert is_reduction_chain(_one(200_000, 4_000), **RED) is False
    assert is_reduction_chain(_one(200_000, 3_999), **RED) is True


def test_reduction_empty_chain_is_not_reduction():
    assert is_reduction_chain([], **RED) is False


# --- worker-group membership + overlap assignment ---


def _worker(
    t: float = 0.0, dur: float = 10.0, fork_depth: int = 100, fork_outer: int = 0
) -> AgentChain:
    """A worker chain forked from shared context (``fork_depth`` blocks) at fork
    point ``(parent 0, fork_outer)`` whose single request spans the active
    interval ``[t, t + dur)`` (``api_time=dur``).

    Grouping requires BOTH the same fork point AND temporal overlap; ``hash_ids``
    are generic since block-0 is not used for grouping."""
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
    # 3 workers whose [t, t+dur) intervals all overlap -> one concurrent group
    r = _result(_worker(0, 100), _worker(10, 100), _worker(20, 100))
    assert worker_group_members(r, group_min=3) == {1, 2, 3}


def test_worker_group_non_overlapping_not_grouped():
    # staggered starts AND no overlap (each ends before the next begins) -> none
    r = _result(_worker(0, 50), _worker(100, 50), _worker(200, 50))
    assert worker_group_members(r, group_min=3) == set()


def test_worker_group_staggered_but_overlapping_still_one_group():
    # the "started 230s later but still alive together" case: starts spread 200s
    # yet every interval overlaps the next -> one concurrent swarm (the wg:017
    # scenario that motivated overlap-based grouping)
    r = _result(_worker(0, 300), _worker(100, 300), _worker(200, 300))
    assert worker_group_members(r, group_min=3) == {1, 2, 3}


def test_worker_group_transitive_overlap_chains_into_one_group():
    # A[0,100]-B[90,190] overlap, B-C[180,280] overlap, A-C do NOT -> one component
    r = _result(_worker(0, 100), _worker(90, 100), _worker(180, 100))
    assert worker_group_members(r, group_min=3) == {1, 2, 3}


def test_worker_group_overlap_does_not_bridge_distinct_fork_points():
    # two trios that fully overlap in time but forked from DIFFERENT parent
    # requests stay separate -- the fork-point scope stops overlap from bridging
    # unrelated fan-outs (the failure mode of pure interval overlap).
    r = _result(
        _worker(0, 100, fork_outer=5),
        _worker(1, 100, fork_outer=5),
        _worker(2, 100, fork_outer=5),
        _worker(0, 100, fork_outer=9),
        _worker(1, 100, fork_outer=9),
        _worker(2, 100, fork_outer=9),
    )
    a = worker_group_assignment(r, group_min=3)
    assert len({a[c][0] for c in range(1, 7)}) == 2  # two groups despite full overlap
    assert {a[c][0] for c in (1, 2, 3)} != {a[c][0] for c in (4, 5, 6)}


def test_worker_group_below_min_not_members():
    # only 2 overlap -> below group_min=3 -> none
    r = _result(_worker(0, 100), _worker(10, 100))
    assert worker_group_members(r, group_min=3) == set()


def test_worker_group_requires_fork_depth():
    # 3 overlap but depth 0 (didn't fork from shared context) -> none
    r = _result(_worker(0, 100, 0), _worker(10, 100, 0), _worker(20, 100, 0))
    assert worker_group_members(r, group_min=3) == set()


def test_worker_group_fork_none_never_groups():
    # a chain with no fork is never a member
    solo = _worker(0, 100)
    solo.fork = None
    r = _result(solo, _worker(10, 100), _worker(20, 100))  # only 2 real members
    assert worker_group_members(r, group_min=3) == set()


def test_worker_group_min_zero_disables():
    r = _result(_worker(0, 100), _worker(10, 100), _worker(20, 100))
    assert worker_group_members(r, group_min=0) == set()


# --- overlap coordinate assignment (group / member) ---


def test_worker_group_assignment_group_and_member():
    # one overlapping component -> group 0, members 0..2 in (t0, outer) order
    r = _result(_worker(0, 100), _worker(1, 100), _worker(2, 100))
    assert worker_group_assignment(r, group_min=3) == {
        1: (0, 0),
        2: (0, 1),
        3: (0, 2),
    }


def test_worker_group_assignment_two_disjoint_components_two_groups():
    # two overlap clusters separated by a quiet gap -> two distinct groups
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
    # group ordinal follows the earliest member's start, not declaration order
    r = _result(
        _worker(1000, 100),
        _worker(1001, 100),
        _worker(1002, 100),
        _worker(0, 100),
        _worker(1, 100),
        _worker(2, 100),
    )
    a = worker_group_assignment(r, group_min=3)
    # the early cluster starts first -> group 0
    assert {a[c][0] for c in (4, 5, 6)} == {0}
    assert {a[c][0] for c in (1, 2, 3)} == {1}


def test_worker_group_members_matches_assignment_keys():
    # 3 overlapping depth>0 workers form a group; the depth-0 one is excluded
    r = _result(
        _worker(0, 100), _worker(10, 100), _worker(20, 100), _worker(30, 100, 0)
    )
    assert worker_group_members(r, group_min=3) == {1, 2, 3}
    assert worker_group_members(r, group_min=3) == set(
        worker_group_assignment(r, group_min=3)
    )


# --- session-id suffix precedence (_worker_suffix) ---


def test_worker_suffix_precedence_and_shape():
    # aux wins over everything; reduction next; then worker-group coordinate.
    assert (
        _worker_suffix(n=2, is_aux=True, is_reduction=True, wg_coord=(1, 0))
        == "aux:002"
    )
    assert (
        _worker_suffix(n=3, is_aux=False, is_reduction=True, wg_coord=(1, 0))
        == "aux:003"
    )
    # worker-group: underscore-joined (group, member) value, colon stays structural
    assert (
        _worker_suffix(n=5, is_aux=False, is_reduction=False, wg_coord=(1, 2))
        == "wg:001_002"
    )
    assert (
        _worker_suffix(n=7, is_aux=False, is_reduction=False, wg_coord=None) == "fa:007"
    )
