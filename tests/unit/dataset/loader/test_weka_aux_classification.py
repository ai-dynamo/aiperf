# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for auxiliary one-shot (sidecar) classification of detected chains.

``is_aux_chain`` decides whether a detected worker chain is a genuine agent or
a tool-issued sidecar: short and starting from a small, fresh context relative
to the enclosing main chain's peak. Applies to top-level flat chains
(``::fa:`` -> ``::aux:``) and subagent overflow (``:cNNN`` -> ``:auxNNN``).
"""

from aiperf.dataset.loader.weka_agent_chains import is_aux_chain
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
