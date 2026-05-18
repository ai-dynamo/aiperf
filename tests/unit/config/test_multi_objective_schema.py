# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Schema tests for multi-objective AdaptiveSearchSweep."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from pytest import param

from aiperf.common.enums import OptimizationDirection
from aiperf.config.sweep import (
    AdaptiveSearchSweep,
    Objective,
    OutcomeConstraint,
    SearchSpaceDimension,
)


def _base_kwargs(**overrides):
    base = dict(
        search_space=[
            SearchSpaceDimension(path="concurrency", lo=1, hi=100, kind="int")
        ],
        max_iterations=10,
        objectives=[
            Objective(
                metric="output_token_throughput",
                direction=OptimizationDirection.MAXIMIZE,
            )
        ],
    )
    base.update(overrides)
    return base


def test_objective_threshold_defaults_none():
    obj = Objective(metric="x", direction=OptimizationDirection.MAXIMIZE)
    assert obj.threshold is None


def test_objective_threshold_accepts_finite_float():
    obj = Objective(
        metric="x", direction=OptimizationDirection.MAXIMIZE, threshold=42.0
    )
    assert obj.threshold == 42.0


def test_objective_threshold_rejects_nan():
    with pytest.raises(ValidationError, match="finite"):
        Objective(
            metric="x", direction=OptimizationDirection.MAXIMIZE, threshold=float("nan")
        )


def test_outcome_constraint_basic():
    c = OutcomeConstraint(metric="latency_p99", op="<=", bound=200.0)
    assert (c.metric, c.op, c.bound) == ("latency_p99", "<=", 200.0)


def test_outcome_constraint_rejects_unknown_op():
    with pytest.raises(ValidationError):
        OutcomeConstraint(metric="x", op="!", bound=1.0)


def test_objectives_min_length_one():
    with pytest.raises(ValidationError, match="at least 1"):
        AdaptiveSearchSweep(**_base_kwargs(objectives=[]))


def test_objectives_default_outcome_constraints_empty():
    sweep = AdaptiveSearchSweep(**_base_kwargs())
    assert sweep.outcome_constraints == []


def test_objectives_accepts_two_for_pareto():
    sweep = AdaptiveSearchSweep(
        **_base_kwargs(
            objectives=[
                Objective(
                    metric="output_token_throughput",
                    direction=OptimizationDirection.MAXIMIZE,
                ),
                Objective(
                    metric="time_to_first_token",
                    direction=OptimizationDirection.MINIMIZE,
                ),
            ],
            optuna_sampler="botorch",
            optuna_acquisition="qlognehvi",
        )
    )
    assert len(sweep.objectives) == 2


def test_objectives_with_outcome_constraints():
    sweep = AdaptiveSearchSweep(
        **_base_kwargs(
            outcome_constraints=[
                OutcomeConstraint(metric="error_rate", op="<=", bound=0.01),
            ],
        )
    )
    assert sweep.outcome_constraints[0].metric == "error_rate"


@pytest.mark.parametrize(
    "acq, n_obj, should_pass",
    [
        param("logei", 1, True, id="single-obj-logei-ok"),
        param("qlogei", 1, True, id="single-obj-qlogei-ok"),
        param("qlognehvi", 2, True, id="multi-obj-qlognehvi-ok"),
        param("qlognehvi", 1, False, id="multi-obj-acq-with-single-obj-rejected"),
        param("qlogei", 2, False, id="single-obj-acq-with-multi-obj-rejected"),
    ],
)  # fmt: skip
def test_acquisition_objective_count_cross_validation(acq, n_obj, should_pass):
    objectives = [
        Objective(metric=f"m{i}", direction=OptimizationDirection.MAXIMIZE)
        for i in range(n_obj)
    ]
    kwargs = _base_kwargs(
        objectives=objectives,
        optuna_sampler="botorch",
        optuna_acquisition=acq,
    )
    if should_pass:
        AdaptiveSearchSweep(**kwargs)
    else:
        with pytest.raises(ValidationError, match="acquisition"):
            AdaptiveSearchSweep(**kwargs)
