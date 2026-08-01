# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for orchestrator plugin categories (search_planner, convergence_criterion)."""

from aiperf.orchestrator.convergence import (
    CIWidthConvergence,
    CVConvergence,
    DistributionConvergence,
)
from aiperf.plugin import plugins
from aiperf.plugin.enums import ConvergenceCriterionType, PluginType, SearchPlannerType
from aiperf.plugin.plugins import (
    get_convergence_criterion_metadata,
    get_search_planner_metadata,
    get_typed_metadata,
)
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
        requires_extras=["botorch"],
    )
    assert md.supports_continuous is True
    assert md.supports_categorical is False
    assert md.requires_initial_samples == 5
    assert md.requires_extras == ["botorch"]


def test_convergence_criterion_plugin_lookup_by_name():
    """All three built-in criteria are reachable via plugins.get_class."""
    assert (
        plugins.get_class(PluginType.CONVERGENCE_CRITERION, "ci_width")
        is CIWidthConvergence
    )
    assert plugins.get_class(PluginType.CONVERGENCE_CRITERION, "cv") is CVConvergence
    assert (
        plugins.get_class(PluginType.CONVERGENCE_CRITERION, "distribution")
        is DistributionConvergence
    )
    assert ConvergenceCriterionType.CI_WIDTH == "ci_width"


def test_convergence_criterion_metadata_accessible():
    """Plugin entries expose typed Pydantic metadata via the per-category helper."""
    md = get_convergence_criterion_metadata("distribution")
    assert isinstance(md, ConvergenceCriterionMetadata)
    assert md.requires_jsonl_export is True
    assert md.min_samples == 3
    assert md.metric_kinds == ["continuous"]


def test_convergence_criterion_metadata_via_generic_helper():
    """The generic `get_typed_metadata` dispatches via _CATEGORY_METADATA_CLASSES."""
    md = get_typed_metadata(PluginType.CONVERGENCE_CRITERION, "ci_width")
    assert isinstance(md, ConvergenceCriterionMetadata)
    assert md.requires_confidence_level is True
    assert md.requires_jsonl_export is False


def test_search_planner_plugin_lookup_by_name():
    """The bayesian planner is reachable via plugins.get_class without importing the heavy BO deps."""
    from aiperf.orchestrator.search_planner.bayesian import BayesianSearchPlanner

    assert (
        plugins.get_class(PluginType.SEARCH_PLANNER, "bayesian")
        is BayesianSearchPlanner
    )
    assert SearchPlannerType.BAYESIAN == "bayesian"


def test_search_planner_metadata_typed_access():
    """SearchPlannerMetadata is reachable via the per-category typed helper."""
    md = get_search_planner_metadata("bayesian")
    assert isinstance(md, SearchPlannerMetadata)
    assert md.requires_extras == []
    assert md.supports_continuous is True
    assert md.supports_discrete is True
    assert md.requires_initial_samples == 5


def test_search_planner_metadata_via_generic_helper():
    """get_typed_metadata dispatches via _CATEGORY_METADATA_CLASSES for search_planner."""
    md = get_typed_metadata(PluginType.SEARCH_PLANNER, "bayesian")
    assert isinstance(md, SearchPlannerMetadata)
    assert md.compatible_objective_directions == ["maximize", "minimize"]


def _categories_declaring_metadata_class() -> list[str]:
    """Categories whose categories.yaml entry declares a metadata_class.

    Derived from the registry rather than hardcoded so the test covers every
    declaring category automatically, including ones added later.
    """
    return sorted(
        category
        for category in plugins.list_categories()
        if (plugins.get_category_metadata(category) or {}).get("metadata_class")
    )


def test_get_typed_metadata_all_declared_categories_returns_typed_object() -> None:
    """Every category declaring metadata_class in categories.yaml gets a typed object.

    Guards against the mapping used by `get_typed_metadata` drifting behind
    categories.yaml, which silently degrades those categories to raw dicts.
    """
    declaring = _categories_declaring_metadata_class()
    assert declaring, "no categories declare metadata_class"

    untyped: list[str] = []
    for category in declaring:
        names = [e.name for e in plugins.list_entries(category)]
        if not names:
            continue
        md = get_typed_metadata(category, names[0])
        if isinstance(md, dict):
            untyped.append(f"{category}:{names[0]}")

    assert not untyped, (
        f"categories returned raw dicts instead of typed metadata: {untyped}"
    )
