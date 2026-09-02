# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Property test: Error Handling — Infeasible and Missing Metrics.

Feature: multi-tier-slo-search, Property 14: Error Handling — Infeasible and Missing Metrics

Validates: Requirements 10.1, 10.2

For any observation with no successful trials, all SLO tiers SHALL be marked
infeasible at that concurrency. For any SLA filter referencing a metric absent
from the observation, that filter SHALL be treated as failed.
"""

from __future__ import annotations

from typing import Literal

from hypothesis import given, settings
from hypothesis import strategies as st

from aiperf.common.models.export_models import JsonMetricResult
from aiperf.config.config import BenchmarkConfig
from aiperf.config.sweep import AdaptiveSearchSweep, Objective
from aiperf.config.sweep.adaptive import SearchSpaceDimension, SLAFilter, SLOTier
from aiperf.orchestrator.models import RunResult
from aiperf.orchestrator.search_planner._sla_helpers import iteration_feasibility
from aiperf.orchestrator.search_planner.multi_tier_planner import MultiTierPlanner

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

METRIC_TAGS = ["output_token_throughput", "time_to_first_token", "inter_token_latency"]
STATS: list[Literal["avg", "p50", "p90", "p95", "p99"]] = [
    "avg",
    "p50",
    "p90",
    "p95",
    "p99",
]
OPS: list[Literal["lt", "le", "gt", "ge"]] = ["lt", "le", "gt", "ge"]


def _make_sla_filter(
    metric_tag: str,
    stat: Literal["avg", "p50", "p90", "p95", "p99"],
    op: Literal["lt", "le", "gt", "ge"],
    threshold: float,
) -> SLAFilter:
    """Construct an SLAFilter using populate_by_name."""
    return SLAFilter.model_validate(
        {"metric_tag": metric_tag, "stat": stat, "op": op, "threshold": threshold}
    )


def _sla_filter_strategy() -> st.SearchStrategy[SLAFilter]:
    """Generate a random SLAFilter."""
    return st.builds(
        _make_sla_filter,
        metric_tag=st.sampled_from(METRIC_TAGS),
        stat=st.sampled_from(STATS),
        op=st.sampled_from(OPS),
        threshold=st.floats(
            min_value=1.0, max_value=10000.0, allow_nan=False, allow_infinity=False
        ),
    )


def _make_slo_tier(label: str, filters: list[SLAFilter]) -> SLOTier:
    """Construct an SLOTier using populate_by_name."""
    return SLOTier.model_validate({"label": label, "filters": filters})


def _tier_strategy(label: str) -> st.SearchStrategy[SLOTier]:
    """Generate an SLOTier with 1-3 filters."""
    return st.builds(
        _make_slo_tier,
        label=st.just(label),
        filters=st.lists(_sla_filter_strategy(), min_size=1, max_size=3),
    )


def _tier_configs_strategy() -> st.SearchStrategy[list[SLOTier]]:
    """Generate 2-5 tiers with unique labels."""
    return st.integers(min_value=2, max_value=5).flatmap(
        lambda n: st.tuples(*[_tier_strategy(f"tier_{i}") for i in range(n)]).map(list)
    )


def _failed_run_result() -> RunResult:
    """Create a RunResult with success=False."""
    return RunResult(
        label="trial",
        success=False,
        summary_metrics={},
        error="simulated failure",
    )


def _base_config() -> BenchmarkConfig:
    """Minimal BenchmarkConfig for MultiTierPlanner instantiation."""
    return BenchmarkConfig.model_validate(
        {
            "models": ["m"],
            "endpoint": {"urls": ["http://x"], "type": "chat"},
            "datasets": [{"name": "profiling", "type": "synthetic"}],
            "phases": [
                {
                    "name": "profiling",
                    "type": "concurrency",
                    "concurrency": 1,
                    "requests": 10,
                }
            ],
        }
    )


def _sweep_cfg(
    *, lo: int = 1, hi: int = 256, max_iterations: int = 30
) -> AdaptiveSearchSweep:
    """Create an AdaptiveSearchSweep config for the planner."""
    return AdaptiveSearchSweep(
        search_space=[
            SearchSpaceDimension(
                path="phases.profiling.concurrency", lo=lo, hi=hi, kind="int"
            )
        ],
        objectives=[Objective(metric="output_token_throughput", direction="maximize")],
        max_iterations=max_iterations,
        n_initial_points=2,
        sla_filters=[
            SLAFilter(
                metric_tag="output_token_throughput",
                stat="avg",
                op="gt",
                threshold=100.0,
            )
        ],
        sla_tiers=[],
    )


def _make_planner(tiers: list[SLOTier]) -> MultiTierPlanner:
    """Instantiate a MultiTierPlanner with given tiers."""
    return MultiTierPlanner(
        base_config=_base_config(),
        cfg=_sweep_cfg(),
        tiers=tiers,
    )


# ---------------------------------------------------------------------------
# Property 14: Error Handling — Infeasible and Missing Metrics
# ---------------------------------------------------------------------------


class TestProperty14ErrorHandlingInfeasibleAndMissingMetrics:
    """Property 14: Error Handling — Infeasible and Missing Metrics.

    **Validates: Requirements 10.1, 10.2**
    """

    @given(
        tiers=_tier_configs_strategy(),
        num_failed_trials=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=100, deadline=None)
    def test_no_successful_trials_marks_all_tiers_infeasible(
        self,
        tiers: list[SLOTier],
        num_failed_trials: int,
    ) -> None:
        """When all trials fail, iteration_feasibility returns False for every tier
        and the planner marks all tiers as infeasible at that concurrency.

        **Validates: Requirements 10.1**
        """
        # Create observation with only failed trials
        failed_results = [_failed_run_result() for _ in range(num_failed_trials)]

        # Verify via iteration_feasibility directly: every tier must be infeasible
        for tier in tiers:
            feasible = iteration_feasibility(failed_results, tier.filters)
            assert feasible is False, (
                f"Tier {tier.label} should be infeasible when no trials succeed, "
                f"but iteration_feasibility returned True"
            )

    @given(
        tiers=_tier_configs_strategy(),
        num_failed_trials=st.integers(min_value=1, max_value=3),
    )
    @settings(max_examples=100, deadline=None)
    def test_no_successful_trials_updates_infeasible_min_on_planner(
        self,
        tiers: list[SLOTier],
        num_failed_trials: int,
    ) -> None:
        """The MultiTierPlanner updates infeasible_min for all tiers when
        no successful trials are returned via tell().

        **Validates: Requirements 10.1**
        """
        planner = _make_planner(tiers)

        # ask() to get the first probe
        pair = planner.ask()
        assert pair is not None
        _, variation = pair

        # tell() with only failed results
        failed_results = [
            RunResult(
                label=f"trial_{i}",
                success=False,
                summary_metrics={},
                error="simulated failure",
                variation_label=variation.label,
                variation_values=variation.values,
            )
            for i in range(num_failed_trials)
        ]
        planner.tell(variation, failed_results)

        # All brackets must have infeasible_min set (all tiers infeasible)
        probed_value = list(variation.values.values())[0]
        for bracket in planner._brackets:
            assert bracket.infeasible_min is not None, (
                f"Tier {bracket.tier.label} should have infeasible_min set "
                f"after no successful trials at concurrency={probed_value}"
            )

    @given(
        num_tiers=st.integers(min_value=2, max_value=5),
        num_filters_per_tier=st.integers(min_value=1, max_value=3),
        metric_value=st.floats(
            min_value=1.0, max_value=10000.0, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=100, deadline=None)
    def test_missing_metrics_treated_as_failed(
        self,
        num_tiers: int,
        num_filters_per_tier: int,
        metric_value: float,
    ) -> None:
        """When a successful trial is missing the metric required by an SLA filter,
        that filter is treated as failed.

        **Validates: Requirements 10.2**
        """
        # Use a metric tag that will NOT be present in the observation
        missing_metric_tag = "nonexistent_metric_xyz"

        # Build tiers that reference the missing metric
        tiers = [
            SLOTier(
                label=f"tier_{i}",
                filters=[
                    SLAFilter(
                        metric_tag=missing_metric_tag,
                        stat="avg",
                        op="gt",
                        threshold=metric_value,
                    )
                    for _ in range(num_filters_per_tier)
                ],
            )
            for i in range(num_tiers)
        ]

        # Observation with a successful trial that has different metrics
        observation = [
            RunResult(
                label="trial",
                success=True,
                summary_metrics={
                    "output_token_throughput": JsonMetricResult(
                        unit="tok/s", avg=500.0
                    ),
                },
            )
        ]

        # Every tier should be infeasible because the required metric is missing
        for tier in tiers:
            feasible = iteration_feasibility(observation, tier.filters)
            assert feasible is False, (
                f"Tier {tier.label} should be infeasible when metric "
                f"{missing_metric_tag!r} is missing from the observation"
            )

    @given(
        num_tiers=st.integers(min_value=2, max_value=4),
        present_metric_value=st.floats(
            min_value=1.0, max_value=10000.0, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=100, deadline=None)
    def test_missing_stat_treated_as_failed(
        self,
        num_tiers: int,
        present_metric_value: float,
    ) -> None:
        """When a successful trial has the metric but is missing the specific stat
        required by an SLA filter, that filter is treated as failed.

        **Validates: Requirements 10.2**
        """
        # Build tiers that require p99 stat, but observation only has avg
        tiers = [
            SLOTier(
                label=f"tier_{i}",
                filters=[
                    SLAFilter(
                        metric_tag="output_token_throughput",
                        stat="p99",
                        op="gt",
                        threshold=present_metric_value,
                    )
                ],
            )
            for i in range(num_tiers)
        ]

        # Observation has the metric but only avg stat, not p99
        observation = [
            RunResult(
                label="trial",
                success=True,
                summary_metrics={
                    "output_token_throughput": JsonMetricResult(
                        unit="tok/s", avg=present_metric_value * 2
                    ),
                },
            )
        ]

        # Every tier should be infeasible because p99 is None
        for tier in tiers:
            feasible = iteration_feasibility(observation, tier.filters)
            assert feasible is False, (
                f"Tier {tier.label} should be infeasible when stat 'p99' is "
                f"missing even though metric 'output_token_throughput' is present"
            )

    @given(
        tiers=_tier_configs_strategy(),
    )
    @settings(max_examples=100, deadline=None)
    def test_missing_metrics_planner_marks_tiers_infeasible(
        self,
        tiers: list[SLOTier],
    ) -> None:
        """The MultiTierPlanner marks tiers infeasible when tell() receives results
        with successful trials but missing the required metrics.

        **Validates: Requirements 10.2**
        """
        planner = _make_planner(tiers)

        pair = planner.ask()
        assert pair is not None
        _, variation = pair

        # Successful trial but with a metric that no tier references
        results = [
            RunResult(
                label="trial",
                success=True,
                summary_metrics={
                    "unrelated_metric": JsonMetricResult(unit="ms", avg=100.0),
                },
                variation_label=variation.label,
                variation_values=variation.values,
            )
        ]
        planner.tell(variation, results)

        # Verify iteration recorded as infeasible (no tier can pass)
        history = planner.history()
        assert len(history) == 1
        assert history[0].feasible is False, (
            "Iteration should be infeasible when required metrics are missing "
            "from all successful trials"
        )
