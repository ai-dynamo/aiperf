# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test that `_build_search_planner` routes to MultiTierPlanner when 2+ tiers configured."""

from unittest.mock import MagicMock

from aiperf.config.sweep import AdaptiveSearchSweep, Objective
from aiperf.config.sweep.adaptive import SearchSpaceDimension, SLAFilter, SLOTier
from aiperf.plugin.enums import SearchPlannerType


def _sla_filter(threshold: float = 200.0) -> SLAFilter:
    return SLAFilter(
        metric_tag="time_to_first_token",
        stat="p95",
        op="lt",
        threshold=threshold,
    )


def _make_plan(
    *,
    sla_tiers: list[SLOTier] | None = None,
    planner: SearchPlannerType = SearchPlannerType.SMOOTH_ISOTONIC,
) -> MagicMock:
    plan = MagicMock()
    plan.sweep = AdaptiveSearchSweep(
        search_space=[
            SearchSpaceDimension(
                path="phases.profiling.concurrency", lo=1, hi=100, kind="int"
            )
        ],
        objectives=[Objective(metric="output_token_throughput", direction="maximize")],
        planner=planner,
        max_iterations=10,
        n_initial_points=2,
        sla_filters=[_sla_filter()],
        sla_tiers=sla_tiers or [],
    )
    plan.configs = [MagicMock()]
    return plan


def test_build_search_planner_returns_multi_tier_when_two_tiers():
    """When sla_tiers has 2+ entries, MultiTierPlanner is instantiated."""
    from aiperf.cli_runner._strategy import _build_search_planner
    from aiperf.orchestrator.search_planner.multi_tier_planner import MultiTierPlanner

    tiers = [
        SLOTier(label="fast", filters=[_sla_filter(300.0)]),
        SLOTier(label="standard", filters=[_sla_filter(100.0)]),
    ]
    plan = _make_plan(sla_tiers=tiers)
    planner = _build_search_planner(plan)
    assert isinstance(planner, MultiTierPlanner)


def test_build_search_planner_returns_single_tier_planner_when_no_tiers():
    """When sla_tiers is empty, the normal plugin-dispatched planner is used."""
    from aiperf.cli_runner._strategy import _build_search_planner
    from aiperf.orchestrator.search_planner.multi_tier_planner import MultiTierPlanner
    from aiperf.orchestrator.search_planner.smooth_isotonic import (
        SmoothIsotonicSLAPlanner,
    )

    plan = _make_plan(sla_tiers=[])
    planner = _build_search_planner(plan)
    assert isinstance(planner, SmoothIsotonicSLAPlanner)
    assert not isinstance(planner, MultiTierPlanner)


def test_build_search_planner_single_tier_list_uses_normal_planner():
    """A single-element sla_tiers list (< 2) does NOT activate multi-tier."""
    from aiperf.cli_runner._strategy import _build_search_planner
    from aiperf.orchestrator.search_planner.multi_tier_planner import MultiTierPlanner

    tiers = [SLOTier(label="only", filters=[_sla_filter(200.0)])]
    plan = _make_plan(sla_tiers=tiers)
    planner = _build_search_planner(plan)
    assert not isinstance(planner, MultiTierPlanner)


def test_build_search_planner_multi_tier_works_with_monotonic_planner():
    """MultiTierPlanner is instantiated regardless of the underlying planner type."""
    from aiperf.cli_runner._strategy import _build_search_planner
    from aiperf.orchestrator.search_planner.multi_tier_planner import MultiTierPlanner

    plan = MagicMock()
    tiers = [
        SLOTier(label="fast", filters=[_sla_filter(300.0)]),
        SLOTier(label="economy", filters=[_sla_filter(50.0)]),
    ]
    plan.sweep = AdaptiveSearchSweep(
        search_space=[
            SearchSpaceDimension(
                path="phases.profiling.concurrency", lo=1, hi=100, kind="int"
            )
        ],
        objectives=[Objective(metric="output_token_throughput", direction="maximize")],
        planner=SearchPlannerType.MONOTONIC_SLA,
        max_iterations=10,
        n_initial_points=2,
        sla_filters=[_sla_filter()],
        sla_tiers=tiers,
    )
    plan.configs = [MagicMock()]
    planner = _build_search_planner(plan)
    assert isinstance(planner, MultiTierPlanner)


def test_build_search_planner_warns_when_non_isotonic_style_with_tiers(caplog):
    """A non-smooth_isotonic --search-style + tiers warns that the style's search
    algorithm is not used (multi-tier runs its own bracket/bisection)."""
    import logging

    from aiperf.cli_runner._strategy import _build_search_planner

    tiers = [
        SLOTier(label="fast", filters=[_sla_filter(300.0)]),
        SLOTier(label="standard", filters=[_sla_filter(100.0)]),
    ]
    plan = _make_plan(sla_tiers=tiers, planner=SearchPlannerType.MONOTONIC_SLA)
    with caplog.at_level(logging.WARNING, logger="aiperf.cli_runner._strategy"):
        _build_search_planner(plan)

    style_warnings = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "not used" in r.getMessage()
    ]
    assert len(style_warnings) == 1, (
        f"expected one 'search algorithm not used' warning, got: "
        f"{[r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]}"
    )


def test_build_search_planner_no_warning_for_isotonic_style_with_tiers(caplog):
    """The default smooth_isotonic style + tiers does NOT warn — it's the style
    multi-tier reuses for precision/warmup."""
    import logging

    from aiperf.cli_runner._strategy import _build_search_planner

    tiers = [
        SLOTier(label="fast", filters=[_sla_filter(300.0)]),
        SLOTier(label="standard", filters=[_sla_filter(100.0)]),
    ]
    plan = _make_plan(sla_tiers=tiers, planner=SearchPlannerType.SMOOTH_ISOTONIC)
    with caplog.at_level(logging.WARNING, logger="aiperf.cli_runner._strategy"):
        _build_search_planner(plan)

    style_warnings = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "not used" in r.getMessage()
    ]
    assert not style_warnings, (
        f"did not expect a style warning, got: "
        f"{[r.getMessage() for r in style_warnings]}"
    )
