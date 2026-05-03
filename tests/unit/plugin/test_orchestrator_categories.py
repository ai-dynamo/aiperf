# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for orchestrator plugin categories (search_planner, convergence_criterion)."""

from aiperf.plugin.schema.schemas import (
    ConvergenceCriterionMetadata,
    SearchPlannerMetadata,
)


def test_convergence_criterion_metadata_shape():
    """ConvergenceCriterionMetadata declares required capability fields."""
    md = ConvergenceCriterionMetadata(
        min_samples=3,
        requires_confidence_level=True,
        requires_jsonl_export=False,
        metric_kinds=["continuous"],
    )
    assert md.min_samples == 3
    assert md.requires_confidence_level is True
    assert md.requires_jsonl_export is False
    assert md.metric_kinds == ["continuous"]


def test_search_planner_metadata_shape():
    """SearchPlannerMetadata declares dimension-kind support and extras."""
    md = SearchPlannerMetadata(
        supports_continuous=True,
        supports_discrete=True,
        supports_categorical=False,
        requires_initial_samples=5,
        compatible_objective_directions=["maximize", "minimize"],
        requires_extras=["bo"],
    )
    assert md.supports_continuous is True
    assert md.supports_categorical is False
    assert md.requires_initial_samples == 5
    assert md.requires_extras == ["bo"]
