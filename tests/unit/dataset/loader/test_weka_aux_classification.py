# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for worker-chain classification of detected chains.

Three predicates partition detected worker chains into agent vs auxiliary vs
parallel fan-out:

- ``is_aux_chain``: a short chain from a small fresh context, or a cross-model
  one-shot, is a tool-issued sidecar (``::fa:`` -> ``::aux:``).
- ``is_reduction_chain``: a same-model single large-input/short-output one-shot
  is a reduction sidecar (``::fa:`` -> ``::aux:red:``).
- ``worker_group_members``: workers sharing a spawn block with enough forked
  siblings are a parallel fan-out group (``::fa:`` -> ``::wg:``).

All apply at both layers (top-level flat chains and subagent overflow). The
``_worker_marker`` helper resolves the precedence (aux > reduction > worker-
group > solo agent).
"""

from aiperf.dataset.loader.weka_agent_chains import (
    AgentChain,
    ChainDetectionResult,
    ChainFork,
    is_aux_chain,
    is_reduction_chain,
    worker_group_members,
)
from aiperf.dataset.loader.weka_trace import _worker_marker
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


# --- worker-group membership (worker_group_members) ---


def _worker(first_block: int, fork_depth: int) -> AgentChain:
    """A worker chain whose first request opens at ``first_block`` and which
    forked from shared context at ``fork_depth`` blocks."""
    return AgentChain(
        requests=[
            (
                0,
                WekaNormalRequest(
                    type="n",
                    t=0.0,
                    model="m",
                    input_length=50_000,
                    output_length=200,
                    hash_ids=[first_block, 2, 3],
                ),
            )
        ],
        fork=ChainFork(
            parent_chain=0, fork_outer_idx=0, depth=fork_depth, fork_time=0.0
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


def test_worker_group_shared_spawn_block_with_forks():
    # 3 workers share spawn block 7 and forked from shared context -> all members
    r = _result(_worker(7, 100), _worker(7, 100), _worker(7, 100))
    assert worker_group_members(r, group_min=3) == {1, 2, 3}


def test_worker_group_below_min_not_members():
    # only 2 share the block -> below group_min=3 -> none
    r = _result(_worker(7, 100), _worker(7, 100))
    assert worker_group_members(r, group_min=3) == set()


def test_worker_group_requires_fork_depth():
    # 3 share the block but none forked from shared context (depth 0) -> none
    r = _result(_worker(7, 0), _worker(7, 0), _worker(7, 0))
    assert worker_group_members(r, group_min=3) == set()


def test_worker_group_distinct_blocks_not_grouped():
    # no shared spawn point -> no group
    r = _result(_worker(1, 100), _worker(2, 100), _worker(3, 100))
    assert worker_group_members(r, group_min=3) == set()


def test_worker_group_only_forked_members_qualify():
    # 3 share the block (group passes) but one has depth 0 -> only forked qualify
    r = _result(_worker(7, 100), _worker(7, 0), _worker(7, 100))
    assert worker_group_members(r, group_min=3) == {1, 3}


def test_worker_group_min_zero_disables():
    r = _result(_worker(7, 100), _worker(7, 100), _worker(7, 100))
    assert worker_group_members(r, group_min=0) == set()


# --- marker precedence (_worker_marker) ---


def test_worker_marker_precedence():
    # aux wins over everything; reduction over worker-group; wg over solo agent
    assert _worker_marker(is_aux=True, is_reduction=True, is_worker_group=True) == "aux"
    assert (
        _worker_marker(is_aux=False, is_reduction=True, is_worker_group=True)
        == "aux:red"
    )
    assert (
        _worker_marker(is_aux=False, is_reduction=False, is_worker_group=True) == "wg"
    )
    assert (
        _worker_marker(is_aux=False, is_reduction=False, is_worker_group=False) == "fa"
    )
