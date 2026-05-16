# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for module-scope helpers in :mod:`aiperf.orchestrator.search_planner._bayesian_helpers`.

These helpers read no planner state, so direct unit tests are sufficient — no
fixtures, no skopt dependency required (constants and pure-Python coercions only).
"""

from __future__ import annotations

import pytest
from pytest import param

from aiperf.config.adaptive_search import SearchSpaceDimension, SLAFilter
from aiperf.orchestrator.search_planner._bayesian_helpers import (
    NO_DATA_SENTINEL_LOSS,
    PENALTY_WEIGHT_MULTIPLIER,
    PLATEAU_MEAN_EPSILON,
    coerce_for_kind,
    signed_violation,
)


class TestConstants:
    def test_plateau_mean_epsilon_is_small_positive(self) -> None:
        assert 0 < PLATEAU_MEAN_EPSILON < 1e-6

    def test_no_data_sentinel_loss_is_finite_and_large(self) -> None:
        assert NO_DATA_SENTINEL_LOSS == 1.0e6
        assert NO_DATA_SENTINEL_LOSS > 1e3

    def test_penalty_weight_multiplier_is_finite(self) -> None:
        assert PENALTY_WEIGHT_MULTIPLIER == 100.0
        assert PENALTY_WEIGHT_MULTIPLIER > 0


class TestCoerceForKind:
    @pytest.mark.parametrize(
        "kind, value, expected_type, expected_value",
        [
            param("int", 5, int, 5, id="int-from-python-int"),
            param("int", 5.7, int, 5, id="int-truncates-float"),
            param("int", -3.9, int, -3, id="int-truncates-toward-zero-negative"),
            param("real", 5, float, 5.0, id="real-from-python-int"),
            param("real", 1.5, float, 1.5, id="real-from-python-float"),
            param("real", 0, float, 0.0, id="real-from-zero"),
        ],
    )  # fmt: skip
    def test_coerce_python_scalars(
        self,
        kind: str,
        value: int | float,
        expected_type: type,
        expected_value: int | float,
    ) -> None:
        dim = SearchSpaceDimension(
            path="phases.profiling.concurrency", lo=1, hi=100, kind=kind
        )  # type: ignore[arg-type]
        result = coerce_for_kind(value, dim)
        assert type(result) is expected_type
        assert result == expected_value

    def test_coerce_int_from_numpy_scalar(self) -> None:
        np = pytest.importorskip("numpy")
        dim = SearchSpaceDimension(
            path="phases.profiling.concurrency", lo=1, hi=100, kind="int"
        )
        result = coerce_for_kind(np.int64(42), dim)
        assert type(result) is int
        assert result == 42

    def test_coerce_real_from_numpy_scalar(self) -> None:
        np = pytest.importorskip("numpy")
        dim = SearchSpaceDimension(
            path="phases.profiling.concurrency", lo=0.0, hi=1.0, kind="real"
        )
        result = coerce_for_kind(np.float64(0.25), dim)
        assert type(result) is float
        assert result == 0.25


class TestSignedViolation:
    @pytest.mark.parametrize(
        "op, threshold, value, expected",
        [
            param("lt", 200.0, 250.0, 50.0, id="lt-violation-positive"),
            param("lt", 200.0, 150.0, -50.0, id="lt-slack-negative"),
            param("lt", 200.0, 200.0, 0.0, id="lt-on-boundary-zero"),
            param("le", 200.0, 250.0, 50.0, id="le-violation-positive"),
            param("le", 200.0, 199.0, -1.0, id="le-slack-negative"),
            param("gt", 100.0, 50.0, 50.0, id="gt-violation-positive"),
            param("gt", 100.0, 150.0, -50.0, id="gt-slack-negative"),
            param("gt", 100.0, 100.0, 0.0, id="gt-on-boundary-zero"),
            param("ge", 100.0, 50.0, 50.0, id="ge-violation-positive"),
            param("ge", 100.0, 150.0, -50.0, id="ge-slack-negative"),
        ],
    )  # fmt: skip
    def test_sign_convention(
        self,
        op: str,
        threshold: float,
        value: float,
        expected: float,
    ) -> None:
        sla = SLAFilter(metric_tag="time_to_first_token", op=op, threshold=threshold)  # type: ignore[arg-type]
        assert signed_violation(value, sla) == pytest.approx(expected)

    def test_clamping_max_zero_isolates_violations(self) -> None:
        """Caller pattern: max(0, signed_violation(...)) — slack must clamp to 0."""
        sla = SLAFilter(metric_tag="ttft", op="lt", threshold=200.0)
        assert max(0.0, signed_violation(150.0, sla)) == 0.0
        assert max(0.0, signed_violation(250.0, sla)) == 50.0
