# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Numerical contracts for the official online parity gate."""

from __future__ import annotations

import math
import runpy
from pathlib import Path

import pytest

_GATE = runpy.run_path(
    str(Path(__file__).parents[2] / "tools" / "dynamo_online_parity_gate.py")
)
_compact_json = _GATE["_compact_json"]
_distribution_gate = _GATE["_distribution_gate"]
_holm_rejections = _GATE["_holm_rejections"]
_ulp_distance = _GATE["_ulp_distance"]


def test_gate_json_rejects_non_finite_diagnostics() -> None:
    with pytest.raises(ValueError, match="Out of range float values"):
        _compact_json({"metric": math.inf})


def test_ulp_distance_accepts_one_binary64_step_only() -> None:
    for value, direction in [(1.0, math.inf), (-1.0, -math.inf)]:
        adjacent = math.nextafter(value, direction)
        second = math.nextafter(adjacent, direction)

        assert _ulp_distance(value, value) == 0
        assert _ulp_distance(value, adjacent) == 1
        assert _ulp_distance(value, second) == 2


def test_ulp_distance_handles_signed_zero_and_non_finite_values() -> None:
    assert _ulp_distance(-0.0, 0.0) == 0
    assert math.isinf(_ulp_distance(1.0, math.inf))


def test_distribution_gate_rejects_systematic_shift_hidden_by_one_overlap() -> None:
    gate = _distribution_gate(
        [110.0] * 8 + [100.0],
        [100.0] * 9,
        5.0,
    )

    assert gate["gate_method"] == "tolerance_shifted_exact_conditional_rank_sum"
    assert gate["high_regression_p_numerator"] == 10
    assert gate["permutation_count"] == 48_620
    assert gate["unadjusted_per_field_passed"] is False


def test_distribution_gate_rejects_a_stable_systematic_shift() -> None:
    gate = _distribution_gate(
        [106.0, 106.1, 105.9] * 3,
        [100.0, 100.1, 99.9] * 3,
        5.0,
    )

    assert gate["gate_method"] == "tolerance_shifted_exact_conditional_rank_sum"
    assert gate["high_regression_p_value"] < 0.025
    assert gate["unadjusted_per_field_passed"] is False


def test_distribution_gate_keeps_limit_relative_to_official_reference() -> None:
    gate = _distribution_gate(
        [105.1] * 9,
        [100.0] * 9,
        5.0,
    )

    assert gate["high_regression_p_numerator"] == 1
    assert gate["unadjusted_per_field_passed"] is False


def test_distribution_gate_treats_exact_positive_boundary_as_a_tie() -> None:
    gate = _distribution_gate(
        [105.0] * 9,
        [100.0] * 9,
        5.0,
    )

    assert gate["high_regression_p_value"] == 1.0
    assert gate["unadjusted_per_field_passed"] is True


def test_distribution_gate_keeps_negative_limit_relative_to_official_reference() -> (
    None
):
    outside = _distribution_gate(
        [94.9] * 9,
        [100.0] * 9,
        5.0,
    )
    boundary = _distribution_gate(
        [95.0] * 9,
        [100.0] * 9,
        5.0,
    )

    assert outside["low_regression_p_numerator"] == 1
    assert outside["unadjusted_per_field_passed"] is False
    assert boundary["low_regression_p_value"] == 1.0
    assert boundary["unadjusted_per_field_passed"] is True


def test_distribution_gate_rejects_low_shift_hidden_by_one_overlap() -> None:
    gate = _distribution_gate(
        [90.0] * 8 + [100.0],
        [100.0] * 9,
        5.0,
    )

    assert gate["low_regression_p_numerator"] == 10
    assert gate["permutation_count"] == 48_620
    assert gate["unadjusted_per_field_passed"] is False


def test_distribution_gate_rejects_separated_noisy_ranges() -> None:
    gate = _distribution_gate(
        [120.0, 130.0, 140.0] * 3,
        [90.0, 100.0, 110.0] * 3,
        5.0,
    )

    assert gate["gate_method"] == "tolerance_shifted_exact_conditional_rank_sum"
    assert gate["high_regression_p_value"] < 0.025
    assert gate["unadjusted_per_field_passed"] is False


def test_distribution_gate_tolerates_one_outlier_in_each_block() -> None:
    gate = _distribution_gate(
        [100.0, 100.0, 140.0] * 3,
        [100.0] * 9,
        5.0,
    )

    assert gate["block_medians_percent"] == [0.0, 0.0, 0.0]
    assert gate["unadjusted_per_field_passed"] is True


def test_distribution_gate_reports_inconclusive_block_instability_without_failing() -> (
    None
):
    gate = _distribution_gate(
        [110.0] * 3 + [90.0] * 3 + [100.0] * 3,
        [100.0] * 9,
        5.0,
    )

    assert gate["signed_block_median_percent"] == 0.0
    assert gate["median_absolute_block_median_percent"] > 5.0
    assert gate["unadjusted_per_field_passed"] is True


def test_holm_family_gate_still_rejects_hidden_shift_among_all_case_fields() -> None:
    gate = _distribution_gate(
        [110.0] * 8 + [100.0],
        [100.0] * 9,
        5.0,
    )
    hypotheses = [
        {
            "name": "report/hidden/high",
            "p_denominator": gate["permutation_count"],
            "p_numerator": gate["high_regression_p_numerator"],
        },
        *(
            {
                "name": f"report/noise-{index}/high",
                "p_denominator": gate["permutation_count"],
                "p_numerator": gate["permutation_count"],
            }
            for index in range(135)
        ),
    ]

    assert [item["name"] for item in _holm_rejections(hypotheses)] == [
        "report/hidden/high"
    ]
