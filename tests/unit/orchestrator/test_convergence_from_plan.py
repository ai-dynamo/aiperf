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
    p.convergence_metric = "time_to_first_token"
    p.convergence_stat = "avg"
    p.convergence_threshold = 0.1
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


def test_cv_from_plan_maps_fields(plan):
    crit = CVConvergence.from_plan(plan)
    assert isinstance(crit, CVConvergence)
    assert crit._metric == "time_to_first_token"
    assert crit._threshold == 0.1
    assert crit._stat == "avg"


def test_distribution_from_plan_maps_fields(plan):
    crit = DistributionConvergence.from_plan(plan)
    assert isinstance(crit, DistributionConvergence)
    assert crit._metric == "time_to_first_token"
    assert crit._p_value_threshold == 0.1
    assert crit._jsonl_filename == "profile_export.jsonl"


def test_distribution_from_plan_uses_default_jsonl_when_none(plan):
    plan.export_jsonl_file = None
    crit = DistributionConvergence.from_plan(plan)
    from aiperf.orchestrator.jsonl_loader import DEFAULT_JSONL_FILENAME

    assert crit._jsonl_filename == DEFAULT_JSONL_FILENAME


def test_build_convergence_criterion_dispatches_via_plugin_registry(plan):
    """`_build_convergence_criterion(plan)` returns the right criterion class for each mode.

    Behavioral equivalence pin: verifies all three built-in modes resolve to
    their corresponding criterion classes through the plugin registry. If a
    third party registers a fourth criterion under `convergence_criterion`,
    its name string passed via `plan.convergence_mode` will route the same way.
    """
    from aiperf._cli_runner_helpers import _build_convergence_criterion

    plan.convergence_mode = "ci_width"
    assert isinstance(_build_convergence_criterion(plan), CIWidthConvergence)

    plan.convergence_mode = "cv"
    assert isinstance(_build_convergence_criterion(plan), CVConvergence)

    plan.convergence_mode = "distribution"
    assert isinstance(_build_convergence_criterion(plan), DistributionConvergence)
