# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Outcome constraint feasibility plumbing tests.

Verifies that ``build_outcome_constraints_func`` reads observations from
``trial.user_attrs`` and produces Optuna-convention signed violations
(<= 0 feasible, > 0 violates).
"""

from __future__ import annotations

import pytest
from pytest import param

from aiperf.config.sweep import OutcomeConstraint
from aiperf.orchestrator.search_planner._optuna_helpers import (
    _UNMEASURABLE_VIOLATION,
    build_outcome_constraints_func,
)


@pytest.mark.parametrize(
    "op, bound, observed, expected_violation",
    [
        param("<=", 100.0, 50.0, -50.0, id="le-feasible"),
        param("<=", 100.0, 150.0, 50.0, id="le-violates"),
        param(">=", 100.0, 150.0, -50.0, id="ge-feasible"),
        param(">=", 100.0, 50.0, 50.0, id="ge-violates"),
        param("==", 100.0, 100.0, 0.0, id="eq-feasible"),
        param("==", 100.0, 99.0, 1.0, id="eq-violates"),
    ],
)  # fmt: skip
def test_outcome_constraint_signed_violation(op, bound, observed, expected_violation):
    constraints = [OutcomeConstraint(metric="m", op=op, bound=bound)]
    func = build_outcome_constraints_func(constraints)

    class _Trial:
        user_attrs = {"outcome:m": observed}

    result = list(func(_Trial()))
    assert result == [pytest.approx(expected_violation)]


def test_outcome_constraint_missing_observation_marks_violation():
    constraints = [OutcomeConstraint(metric="m", op="<=", bound=100.0)]
    func = build_outcome_constraints_func(constraints)

    class _Trial:
        user_attrs: dict = {}

    result = list(func(_Trial()))
    assert result[0] == _UNMEASURABLE_VIOLATION
    assert result[0] > 0


def test_outcome_constraint_multiple_constraints_concatenated():
    constraints = [
        OutcomeConstraint(metric="m1", op="<=", bound=100.0),
        OutcomeConstraint(metric="m2", op=">=", bound=10.0),
    ]
    func = build_outcome_constraints_func(constraints)

    class _Trial:
        user_attrs = {"outcome:m1": 50.0, "outcome:m2": 5.0}

    result = list(func(_Trial()))
    assert result == [pytest.approx(-50.0), pytest.approx(5.0)]
