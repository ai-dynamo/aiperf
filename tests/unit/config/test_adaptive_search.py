# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for AdaptiveSearchSweep and SearchSpaceDimension.

Post-redesign, the BO config lives on ``AdaptiveSearchSweep``
(``aiperf.config.sweep``) — a sweep variant — with the optimization
target nested under an ``Objective``. The leaf
``SearchSpaceDimension`` and ``SLAFilter`` types still ship from
``aiperf.config.sweep.adaptive``.
"""

from __future__ import annotations

import warnings

import pytest
from pydantic import ValidationError
from pytest import param

from aiperf.common.enums import OptimizationDirection
from aiperf.config.sweep import AdaptiveSearchSweep, Objective
from aiperf.config.sweep.adaptive import SearchSpaceDimension


def test_search_space_dimension_int():
    dim = SearchSpaceDimension(
        path="phases.profiling.concurrency", lo=1, hi=1000, kind="int"
    )
    assert dim.path == "phases.profiling.concurrency"
    assert dim.kind == "int"


def test_search_space_dimension_rejects_lo_gt_hi():
    with pytest.raises(ValidationError):
        SearchSpaceDimension(path="x", lo=10, hi=1, kind="int")


def test_adaptive_search_sweep_minimal():
    cfg = AdaptiveSearchSweep(
        search_space=[
            SearchSpaceDimension(
                path="phases.profiling.concurrency", lo=1, hi=1000, kind="int"
            ),
        ],
        objectives=[
            Objective(
                metric="output_token_throughput",
                stat="avg",
                direction=OptimizationDirection.MAXIMIZE,
            )
        ],
        max_iterations=20,
    )
    assert cfg.max_iterations == 20
    assert cfg.plateau_window == 8  # default
    assert cfg.objectives[0].metric == "output_token_throughput"


def test_adaptive_search_sweep_rejects_empty_search_space():
    with pytest.raises(ValidationError):
        AdaptiveSearchSweep(
            search_space=[],
            objectives=[
                Objective(
                    metric="x", stat="avg", direction=OptimizationDirection.MAXIMIZE
                )
            ],
            max_iterations=20,
        )


def test_adaptive_search_sweep_rejects_max_iterations_below_two():
    with pytest.raises(ValidationError):
        AdaptiveSearchSweep(
            search_space=[SearchSpaceDimension(path="x", lo=1, hi=10, kind="int")],
            objectives=[
                Objective(
                    metric="x", stat="avg", direction=OptimizationDirection.MAXIMIZE
                )
            ],
            max_iterations=1,  # below ge=2
        )


def test_adaptive_search_sweep_rejects_initial_points_at_or_above_max_iterations():
    with pytest.raises(ValidationError, match="n_initial_points"):
        AdaptiveSearchSweep(
            search_space=[SearchSpaceDimension(path="x", lo=1, hi=10, kind="int")],
            objectives=[
                Objective(
                    metric="x", stat="avg", direction=OptimizationDirection.MAXIMIZE
                )
            ],
            max_iterations=5,
            n_initial_points=5,  # not strictly less than max_iterations
        )


@pytest.mark.parametrize(
    "planner",
    [
        param("bayesian", id="bayesian"),
        param("optuna", id="optuna"),
    ],
)
def test_n_initial_points_gate_fires_for_bo_planners(planner: str):
    """The n_initial_points < max_iterations gate guards GP fitting for
    BO planners (bayesian preset and optuna expert mode). Both must continue
    to reject the bad combo."""
    with pytest.raises(ValidationError, match="n_initial_points"):
        AdaptiveSearchSweep(
            planner=planner,
            search_space=[SearchSpaceDimension(path="x", lo=1, hi=10, kind="int")],
            objectives=[
                Objective(
                    metric="x", stat="avg", direction=OptimizationDirection.MAXIMIZE
                )
            ],
            max_iterations=3,
            n_initial_points=5,
        )


@pytest.mark.parametrize(
    "planner",
    [
        param("monotonic_sla", id="monotonic_sla"),
        param("smooth_isotonic", id="smooth_isotonic"),
    ],
)
def test_n_initial_points_gate_skipped_for_non_bo_planners(planner: str):
    """monotonic_sla / smooth_isotonic drive their own probe sequences and
    do not consume n_initial_points. The schema default (n_initial_points=5)
    must not block small-budget configs (max_iterations=3) for these
    planners — otherwise users must work around an irrelevant BO knob."""
    sweep = AdaptiveSearchSweep(
        planner=planner,
        search_space=[SearchSpaceDimension(path="x", lo=1, hi=10, kind="int")],
        objectives=[
            Objective(metric="x", stat="avg", direction=OptimizationDirection.MAXIMIZE)
        ],
        max_iterations=3,
        n_initial_points=5,
    )
    assert sweep.max_iterations == 3
    assert sweep.n_initial_points == 5


@pytest.mark.parametrize(
    "planner",
    [
        param("monotonic_sla", id="monotonic_sla"),
        param("smooth_isotonic", id="smooth_isotonic"),
    ],
)
@pytest.mark.parametrize(
    "field_name,field_value",
    [
        param("improvement_patience", 3, id="improvement_patience"),
        param("plateau_window", 4, id="plateau_window"),
        param("plateau_threshold", 0.005, id="plateau_threshold"),
    ],
)
def test_three_signal_keys_warn_for_1d_planners(
    planner: str, field_name: str, field_value: float
):
    """Setting any three-signal convergence key on a 1D SLA planner
    (monotonic_sla, smooth_isotonic) must emit a UserWarning naming the
    explicitly-set fields. These planners terminate via algorithm-specific
    signals (max_iterations + *_precision_reached / *_no_failure_in_range /
    *_no_pass_in_range) and silently ignore the three-signal keys."""
    kwargs = {
        "planner": planner,
        "search_space": [SearchSpaceDimension(path="x", lo=1, hi=10, kind="int")],
        "objectives": [
            Objective(metric="x", stat="avg", direction=OptimizationDirection.MAXIMIZE)
        ],
        "max_iterations": 5,
        field_name: field_value,
    }
    with pytest.warns(UserWarning, match=field_name):
        AdaptiveSearchSweep(**kwargs)


@pytest.mark.parametrize(
    "planner",
    [
        param("monotonic_sla", id="monotonic_sla"),
        param("smooth_isotonic", id="smooth_isotonic"),
    ],
)
def test_no_warning_when_three_signal_keys_left_at_defaults(planner: str):
    """Recipes that nominate a 1D planner inherit schema defaults for the
    three-signal keys. Defaulted values must NOT trigger the warning —
    only fields the user explicitly set (i.e. in `model_fields_set`)."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sweep = AdaptiveSearchSweep(
            planner=planner,
            search_space=[SearchSpaceDimension(path="x", lo=1, hi=10, kind="int")],
            objectives=[
                Objective(
                    metric="x", stat="avg", direction=OptimizationDirection.MAXIMIZE
                )
            ],
            max_iterations=5,
        )
    assert str(sweep.planner) == planner


@pytest.mark.parametrize(
    "planner",
    [
        param("bayesian", id="bayesian"),
        param("optuna", id="optuna"),
    ],
)
def test_no_warning_for_nd_planners_with_three_signal_keys(planner: str):
    """N-D planners (bayesian, optuna) consume the three-signal keys via
    `evaluate_three_signal_convergence`. Setting them explicitly is the
    intended use; no warning should fire."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sweep = AdaptiveSearchSweep(
            planner=planner,
            search_space=[SearchSpaceDimension(path="x", lo=1, hi=10, kind="int")],
            objectives=[
                Objective(
                    metric="x", stat="avg", direction=OptimizationDirection.MAXIMIZE
                )
            ],
            max_iterations=20,
            n_initial_points=2,
            improvement_patience=3,
            plateau_window=4,
            plateau_threshold=0.005,
        )
    assert sweep.improvement_patience == 3


def test_warning_lists_only_explicitly_set_three_signal_keys():
    """The warning must enumerate exactly which fields the user set so
    they know which flags no-op. A user who set only ``improvement_patience``
    should see ``improvement_patience`` in the message but not ``plateau_window``
    or ``plateau_threshold``."""
    with pytest.warns(UserWarning) as records:
        AdaptiveSearchSweep(
            planner="monotonic_sla",
            search_space=[SearchSpaceDimension(path="x", lo=1, hi=10, kind="int")],
            objectives=[
                Objective(
                    metric="x", stat="avg", direction=OptimizationDirection.MAXIMIZE
                )
            ],
            max_iterations=5,
            improvement_patience=7,
        )
    assert len(records) == 1
    msg = str(records[0].message)
    assert "improvement_patience" in msg
    assert "plateau_window" not in msg
    assert "plateau_threshold" not in msg


# ------------------------------------------------------------------------
# Pin every numeric-bound validator on AdaptiveSearchSweep.
#
# Pre-fix only `max_iterations < 2` had explicit test coverage. Drift on
# `max_iterations > 200`, `improvement_patience < 2`, `plateau_threshold
# <= 0`, `plateau_window < 2`, or `random_seed < 0` would silently accept
# adversarial inputs that crash the BO planner downstream (NaN division,
# infinite loops, negative-seed RNG init). These tests force the bound to
# be a contract, not an accident.
# ------------------------------------------------------------------------


def _minimal_kwargs(**overrides):
    """AdaptiveSearchSweep ctor kwargs with only the required fields filled."""
    base = dict(
        search_space=[SearchSpaceDimension(path="x", lo=1, hi=10, kind="int")],
        objectives=[
            Objective(metric="x", stat="avg", direction=OptimizationDirection.MAXIMIZE)
        ],
        max_iterations=20,
    )
    base.update(overrides)
    return base


class TestAdaptiveSearchMaxIterationsBoundary:
    def test_at_lower_bound_accepted(self):
        AdaptiveSearchSweep(**_minimal_kwargs(max_iterations=2, n_initial_points=1))

    def test_at_upper_bound_accepted(self):
        AdaptiveSearchSweep(**_minimal_kwargs(max_iterations=200))

    def test_above_upper_bound_rejected(self):
        with pytest.raises(ValidationError, match="less than or equal to 200"):
            AdaptiveSearchSweep(**_minimal_kwargs(max_iterations=201))


class TestAdaptiveSearchImprovementPatienceBoundary:
    def test_at_lower_bound_accepted(self):
        AdaptiveSearchSweep(**_minimal_kwargs(improvement_patience=2))

    def test_below_lower_bound_rejected(self):
        with pytest.raises(ValidationError, match="greater than or equal to 2"):
            AdaptiveSearchSweep(**_minimal_kwargs(improvement_patience=1))

    def test_negative_rejected(self):
        with pytest.raises(ValidationError, match="greater than or equal to 2"):
            AdaptiveSearchSweep(**_minimal_kwargs(improvement_patience=-1))


class TestAdaptiveSearchPlateauThresholdBoundary:
    def test_zero_rejected(self):
        with pytest.raises(ValidationError, match="greater than 0"):
            AdaptiveSearchSweep(**_minimal_kwargs(plateau_threshold=0.0))

    def test_negative_rejected(self):
        with pytest.raises(ValidationError, match="greater than 0"):
            AdaptiveSearchSweep(**_minimal_kwargs(plateau_threshold=-0.01))

    def test_small_positive_accepted(self):
        sweep = AdaptiveSearchSweep(**_minimal_kwargs(plateau_threshold=1e-9))
        assert sweep.plateau_threshold == 1e-9


class TestAdaptiveSearchPlateauWindowBoundary:
    def test_at_lower_bound_accepted(self):
        AdaptiveSearchSweep(**_minimal_kwargs(plateau_window=2))

    def test_below_lower_bound_rejected(self):
        with pytest.raises(ValidationError, match="greater than or equal to 2"):
            AdaptiveSearchSweep(**_minimal_kwargs(plateau_window=1))


class TestAdaptiveSearchRandomSeedBoundary:
    def test_default_is_none(self):
        sweep = AdaptiveSearchSweep(**_minimal_kwargs())
        assert sweep.random_seed is None

    def test_zero_accepted(self):
        """Seed=0 is a valid user choice (falsy but not None). Must not be
        rejected by a `gt=0` drift, and must not be replaced by truthiness
        checks downstream."""
        sweep = AdaptiveSearchSweep(**_minimal_kwargs(random_seed=0))
        assert sweep.random_seed == 0

    def test_negative_rejected(self):
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            AdaptiveSearchSweep(**_minimal_kwargs(random_seed=-1))


class TestAdaptiveSearchNInitialPointsBoundary:
    def test_at_lower_bound_accepted(self):
        AdaptiveSearchSweep(**_minimal_kwargs(n_initial_points=1, max_iterations=2))

    def test_zero_rejected(self):
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            AdaptiveSearchSweep(**_minimal_kwargs(n_initial_points=0))


@pytest.mark.parametrize(
    "path",
    [
        param("rate", id="bare-rate"),
        param("phases.profiling.smoothness", id="profiling-smoothness"),
        param("phases.warmup.rate", id="warmup-rate"),
        param("phases.measured.smoothness", id="named-smoothness"),
        param("phases.warmup.rate_ramp", id="warmup-ramp"),
        param("phases.measured.rate_ramp.duration", id="nested-ramp-duration"),
    ],
)  # fmt: skip
@pytest.mark.parametrize("lo", [0, -1])
def test_search_space_field_rejects_nonpositive_bound(path: str, lo: float) -> None:
    with pytest.raises(ValidationError, match="lower bound must be > 0"):
        SearchSpaceDimension(path=path, lo=lo, hi=10)


@pytest.mark.parametrize(
    "path", ["users", "phases.warmup.users", "phases.measured.users"]
)
@pytest.mark.parametrize(
    "lo,kind,message",
    [
        param(0, "int", "lower bound must be >= 1", id="zero"),
        param(0.5, "int", "lower bound must be >= 1", id="fraction-below-one"),
        param(1, "real", "must use ':int'", id="real-kind"),
    ],
)  # fmt: skip
def test_search_space_users_constraints(
    path: str, lo: float, kind: str, message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        SearchSpaceDimension(path=path, lo=lo, hi=10, kind=kind)


@pytest.mark.parametrize(
    "path,lo,hi,kind",
    [
        param("phases.warmup.rate", 0.01, 10, "real", id="positive-rate"),
        param("phases.measured.smoothness", 0.01, 10, "real", id="positive-smoothness"),
        param("phases.warmup.rate_ramp", 0.01, 10, "real", id="positive-ramp"),
        param("phases.measured.rate_ramp.duration", 0.01, 10, "real", id="nested-ramp"),
        param("phases.warmup.users", 1, 10, "int", id="minimum-users"),
        param("phases.profiling.cancellation.rate", 0, 100, "real", id="zero-cancellation"),
        param("unrelated", -10, 0, "real", id="unrelated-negative"),
    ],
)  # fmt: skip
def test_search_space_field_accepts_valid_bounds(
    path: str, lo: float, hi: float, kind: str
) -> None:
    dimension = SearchSpaceDimension(path=path, lo=lo, hi=hi, kind=kind)
    assert (dimension.lo, dimension.hi, dimension.kind) == (lo, hi, kind)


@pytest.mark.parametrize("path", ["rate_series", "phases.warmup.rate_series"])
def test_search_space_rate_series_rejected(path: str) -> None:
    with pytest.raises(ValidationError, match="piecewise-linear"):
        SearchSpaceDimension(path=path, lo=1, hi=10)


@pytest.mark.parametrize("field", ["lo", "hi"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_search_space_field_bounds_must_be_finite(field: str, value: float) -> None:
    bounds = {"lo": 1, "hi": 10, field: value}
    with pytest.raises(ValidationError, match="must be finite"):
        SearchSpaceDimension(path="phases.warmup.rate", **bounds)


@pytest.mark.parametrize(
    "lo,hi",
    [
        param(-1, 10, id="below-zero"),
        param(0, 101, id="above-100"),
    ],
)  # fmt: skip
def test_search_space_cancellation_rate_rejects_out_of_range_bounds(
    lo: float, hi: float
) -> None:
    with pytest.raises(ValidationError, match=r"within \[0, 100\]"):
        SearchSpaceDimension(path="phases.profiling.cancellation.rate", lo=lo, hi=hi)
