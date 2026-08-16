# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for _compute_normalized_model_s (OSL-aligned timing normalization)."""

from __future__ import annotations

import math

import pytest

from aiperf.timing.strategies.agent_graph_replay import _compute_normalized_model_s


def test_empty_durations_returns_none() -> None:
    """No LLM calls → (None, 0)."""
    result, low = _compute_normalized_model_s([], [], [], [])
    assert result is None
    assert low == 0


def test_all_none_osl_returns_none() -> None:
    """All OSL missing → normalized_model_s is None; raw durations used as fallback."""
    result, low = _compute_normalized_model_s(
        [1.0, 2.0], [None, None], [None, None], [None, None]
    )
    assert result is None
    assert low == 0


def test_incomplete_parallel_metadata_returns_none() -> None:
    """An incomplete trace result cannot produce a partial normalization."""
    result, low = _compute_normalized_model_s([1.0], [], [], [])
    assert result is None
    assert low == 0


def test_perfect_match_equals_raw_sum() -> None:
    """target_osl == observed_osl → ratio 1.0, normalized = raw sum."""
    result, low = _compute_normalized_model_s(
        [1.0, 2.0], [0.2, 0.5], [100, 200], [100, 200]
    )
    assert result is not None
    assert math.isclose(result, 3.0)
    assert low == 0


def test_under_osl_rescales_only_generation_and_warns() -> None:
    """Under-length output preserves TTFT while rescaling decode time."""
    # Agent Trace Replay: TTFT=.2, generation=.8; 49 observed decode tokens become 199.
    result, low = _compute_normalized_model_s([1.0], [0.2], [200], [50])
    assert result is not None
    assert math.isclose(result, 0.2 + (0.8 / 49 * 199))
    assert low == 1


def test_over_osl_scales_down_no_warning() -> None:
    """observed > target → scale down, no warning."""
    # Agent Trace Replay: TTFT=.5, 199 observed decode tokens become 99.
    result, low = _compute_normalized_model_s([2.0], [0.5], [100], [200])
    assert result is not None
    assert math.isclose(result, 0.5 + (1.5 / 199 * 99))
    assert low == 0


def test_zero_observed_falls_back_to_raw() -> None:
    """observed=0 → raw fallback, not a warning (guard against divide-by-zero)."""
    result, low = _compute_normalized_model_s([1.5], [0.2], [100], [0])
    # 0 observed → can't normalize; falls back to raw
    assert result is None  # no valid call → None
    assert low == 0


def test_mixed_some_have_osl() -> None:
    """Some calls have OSL, others don't → partial normalization with raw fallback."""
    # call 0: target=100, observed=50 → normalized = 1.0 * 2.0 = 2.0
    # call 1: no OSL → raw = 3.0
    result, low = _compute_normalized_model_s(
        [1.0, 3.0],
        [0.2, None],
        [100, None],
        [50, None],
    )
    assert result is not None
    # has_any=True because call 0 has data; call 1 uses its raw duration.
    assert math.isclose(result, 0.2 + (0.8 / 49 * 99) + 3.0)
    assert low == 0  # observed=50, threshold=0.5*100=50; 50 < 50 is False


def test_exactly_at_threshold_no_warning() -> None:
    """observed == 50% of target → NOT a warning (strictly less-than threshold)."""
    result, low = _compute_normalized_model_s([1.0], [0.2], [100], [50])
    # 50 < 0.5 * 100 → 50 < 50 → False → no warning
    assert low == 0


def test_just_below_threshold_warns() -> None:
    """observed == 49% of target → warning."""
    result, low = _compute_normalized_model_s([1.0], [0.2], [100], [49])
    assert low == 1


@pytest.mark.parametrize(
    "durations, ttft_s, target_osl, observed_osl, expected_norm, expected_low",
    [
        ([1.0, 1.0], [0.2, 0.2], [100, 100], [100, 100], 2.0, 0),
        (
            [1.0, 1.0],
            [0.2, 0.2],
            [100, 100],
            [200, 200],
            2 * (0.2 + (0.8 / 199 * 99)),
            0,
        ),
        ([1.0, 1.0], [0.2, 0.2], [100, 100], [10, 10], 2 * (0.2 + (0.8 / 9 * 99)), 2),
        (
            [1.0, 1.0],
            [0.2, None],
            [100, None],
            [100, None],
            2.0,
            0,
        ),  # second call raw fallback
    ],
    ids=["parity", "over-osl", "severe-under-osl", "partial"],
)
def test_parametrized_cases(
    durations: list[float],
    ttft_s: list[float | None],
    target_osl: list[int | None],
    observed_osl: list[int | None],
    expected_norm: float,
    expected_low: int,
) -> None:
    result, low = _compute_normalized_model_s(
        durations, ttft_s, target_osl, observed_osl
    )
    assert result is not None
    assert math.isclose(result, expected_norm)
    assert low == expected_low
