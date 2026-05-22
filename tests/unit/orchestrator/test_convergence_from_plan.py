# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify each ConvergenceCriterion subclass builds correctly from a BenchmarkPlan."""

from unittest.mock import MagicMock

import pytest

from aiperf.orchestrator.convergence import (
    CIWidthConvergence,
    CVConvergence,
    DistributionConvergence,
)


@pytest.fixture
def plan():
    """Minimal BenchmarkPlan-shaped object exposing the fields each criterion reads."""
    p = MagicMock()
    p.multi_run.convergence.metric = "time_to_first_token"
    p.multi_run.convergence.stat = "avg"
    p.multi_run.convergence.threshold = 0.1
    p.multi_run.convergence.mode = "ci_width"
    p.multi_run.convergence.min_runs = 4
    p.confidence_level = 0.95
    p.export_jsonl_file = "profile_export.jsonl"
    return p


def test_ci_width_from_plan_maps_fields(plan):
    crit = CIWidthConvergence.from_plan(plan)
    assert isinstance(crit, CIWidthConvergence)
    assert crit._metric == "time_to_first_token"
    assert crit._stat == "avg"
    assert crit._threshold == 0.1
    assert crit._confidence_level == 0.95
    assert crit._min_runs == 4


def test_cv_from_plan_maps_fields(plan):
    crit = CVConvergence.from_plan(plan)
    assert isinstance(crit, CVConvergence)
    assert crit._metric == "time_to_first_token"
    assert crit._threshold == 0.1
    assert crit._stat == "avg"
    assert crit._min_runs == 4


def test_distribution_from_plan_maps_fields(plan):
    crit = DistributionConvergence.from_plan(plan)
    assert isinstance(crit, DistributionConvergence)
    assert crit._metric == "time_to_first_token"
    assert crit._p_value_threshold == 0.1
    assert crit._jsonl_filename == "profile_export.jsonl"
    assert crit._min_runs == 4


def test_distribution_from_plan_uses_default_jsonl_when_none(plan):
    plan.export_jsonl_file = None
    crit = DistributionConvergence.from_plan(plan)
    from aiperf.orchestrator.jsonl_loader import DEFAULT_JSONL_FILENAME

    assert crit._jsonl_filename == DEFAULT_JSONL_FILENAME


def test_from_plan_none_threshold_falls_through_to_class_defaults(plan):
    """When ConvergenceConfig.threshold is None, each criterion uses its own default.

    This is the contract that lets users pick a `--convergence-mode` without
    also choosing a `--convergence-threshold`: the algorithm-author's default
    applies, which differs per criterion (CI-width is looser than CV/distribution
    by design because it's a relative width, not a raw dispersion).
    """
    plan.multi_run.convergence.threshold = None

    ci = CIWidthConvergence.from_plan(plan)
    assert ci._threshold == 0.10  # CI-width class default

    cv = CVConvergence.from_plan(plan)
    assert cv._threshold == 0.05  # CV class default

    dist = DistributionConvergence.from_plan(plan)
    assert dist._p_value_threshold == 0.05  # KS p-value class default


def test_build_convergence_criterion_dispatches_via_plugin_registry(plan):
    """`_build_convergence_criterion(plan)` returns the right criterion class for each mode.

    Behavioral equivalence pin: verifies all three built-in modes resolve to
    their corresponding criterion classes through the plugin registry. If a
    third party registers a fourth criterion under `convergence_criterion`,
    its name string passed via `plan.multi_run.convergence.mode` will route the same way.
    """
    from aiperf.cli_runner._strategy import _build_convergence_criterion

    plan.multi_run.convergence.mode = "ci_width"
    assert isinstance(_build_convergence_criterion(plan), CIWidthConvergence)

    plan.multi_run.convergence.mode = "cv"
    assert isinstance(_build_convergence_criterion(plan), CVConvergence)

    plan.multi_run.convergence.mode = "distribution"
    assert isinstance(_build_convergence_criterion(plan), DistributionConvergence)


def test_build_convergence_criterion_unknown_mode_raises(plan):
    """Unknown convergence mode names must be a hard error, not a silent fallback.

    Pins the plugin-registry contract: a typo in `--convergence-mode` cannot
    accidentally route to a default criterion class. Otherwise a user setting
    `--convergence-mode ci-width` (hyphen) instead of `ci_width` would get
    the default and never know.
    """
    from aiperf.cli_runner._strategy import _build_convergence_criterion
    from aiperf.plugin.types import TypeNotFoundError

    plan.multi_run.convergence.mode = "not_a_real_mode"
    with pytest.raises(TypeNotFoundError, match="not_a_real_mode"):
        _build_convergence_criterion(plan)


def test_class_default_thresholds_are_load_bearing():
    """Each criterion's `__init__` threshold default is now user-visible.

    Because `ConvergenceConfig.threshold` defaults to `None` and `from_plan`
    only forwards `threshold=` when set, these class defaults ARE the
    user-facing default for `--convergence-mode <mode>` with no
    `--convergence-threshold` flag. Changing them silently changes the
    documented CLI defaults; this test forces the docs/help-text update to
    happen alongside the code change.
    """
    assert CIWidthConvergence(metric="ttft")._threshold == 0.10
    assert CVConvergence(metric="ttft")._threshold == 0.05
    assert DistributionConvergence(metric="ttft")._p_value_threshold == 0.05


def test_from_plan_is_abstract_on_base_class():
    """Third-party `ConvergenceCriterion` subclasses must implement `from_plan`.

    Pins the plugin contract: the CLI dispatcher calls `cls.from_plan(plan)`
    on every registered criterion, so a subclass missing it would crash at
    runtime rather than at class-definition time. This test forces the
    failure to surface earlier — at instantiation of an incomplete subclass.
    """
    from aiperf.orchestrator.convergence import ConvergenceCriterion

    class IncompleteCriterion(ConvergenceCriterion):
        def is_converged(self, results):
            return False

    with pytest.raises(TypeError, match="abstract"):
        IncompleteCriterion()  # type: ignore[abstract]
