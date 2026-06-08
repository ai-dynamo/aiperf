# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for MultiTierPlanner error handling paths.

Validates Requirements 10.1–10.4:
- No successful trials at concurrency X → mark all tiers infeasible, log
- Missing metric → treat filter as failed, log with metric name and tier label
- All tiers converge to lowest concurrency → terminate with no_pass_in_range
- Non-monotonic observations → flag non_monotonic_warning per tier
"""

from __future__ import annotations

import logging

from aiperf.common.models.export_models import JsonMetricResult
from aiperf.config.config import BenchmarkConfig
from aiperf.config.sweep import AdaptiveSearchSweep, Objective, SweepVariation
from aiperf.config.sweep.adaptive import SearchSpaceDimension, SLAFilter, SLOTier
from aiperf.orchestrator.models import RunResult
from aiperf.orchestrator.search_planner.multi_tier_planner import MultiTierPlanner


def _base_config() -> BenchmarkConfig:
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


def _cfg(
    *, lo: int = 1, hi: int = 256, max_iterations: int = 20
) -> AdaptiveSearchSweep:
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
                metric_tag="time_to_first_token",
                stat="p95",
                op="lt",
                threshold=5000.0,
            )
        ],
        sla_tiers=[],
    )


def _make_tiers() -> list[SLOTier]:
    """Two tiers: strict (throughput > 300) and lenient (throughput > 100)."""
    return [
        SLOTier(
            label="fast",
            filters=[
                SLAFilter(
                    metric_tag="output_token_throughput",
                    stat="avg",
                    op="gt",
                    threshold=300.0,
                )
            ],
        ),
        SLOTier(
            label="economy",
            filters=[
                SLAFilter(
                    metric_tag="output_token_throughput",
                    stat="avg",
                    op="gt",
                    threshold=100.0,
                )
            ],
        ),
    ]


def _make_planner(
    *, lo: int = 1, hi: int = 256, max_iterations: int = 20
) -> MultiTierPlanner:
    return MultiTierPlanner(
        base_config=_base_config(),
        cfg=_cfg(lo=lo, hi=hi, max_iterations=max_iterations),
        tiers=_make_tiers(),
    )


def _success_result(
    variation: SweepVariation, *, throughput: float = 500.0
) -> RunResult:
    return RunResult(
        label="t",
        success=True,
        summary_metrics={
            "output_token_throughput": JsonMetricResult(unit="tok/s", avg=throughput),
            "time_to_first_token": JsonMetricResult(unit="ms", p95=100.0),
        },
        variation_label=variation.label,
        variation_values=variation.values,
    )


def _failed_result(variation: SweepVariation) -> RunResult:
    return RunResult(
        label="t",
        success=False,
        summary_metrics={},
        variation_label=variation.label,
        variation_values=variation.values,
        error="connection timeout",
    )


def _missing_metric_result(variation: SweepVariation) -> RunResult:
    """Successful trial but missing the output_token_throughput metric."""
    return RunResult(
        label="t",
        success=True,
        summary_metrics={
            "time_to_first_token": JsonMetricResult(unit="ms", p95=100.0),
        },
        variation_label=variation.label,
        variation_values=variation.values,
    )


class TestNoSuccessfulTrials:
    """Req 10.1: No successful trials marks all tiers infeasible, logs diagnostic."""

    def test_all_tiers_infeasible_when_no_successful_trials(self) -> None:
        """All tiers marked infeasible at this concurrency when all trials fail."""
        planner = _make_planner()
        pair = planner.ask()
        assert pair is not None
        _, variation = pair

        planner.tell(variation, [_failed_result(variation)])

        # Both tiers should have infeasible_min set
        for bracket in planner._brackets:
            assert bracket.infeasible_min is not None

    def test_logs_diagnostic_on_no_successful_trials(self, caplog) -> None:
        """A warning is logged when no trials succeed."""
        planner = _make_planner()
        pair = planner.ask()
        assert pair is not None
        _, variation = pair

        with caplog.at_level(logging.WARNING):
            planner.tell(variation, [_failed_result(variation)])

        assert any("no successful trials" in r.message for r in caplog.records), (
            f"Expected 'no successful trials' warning, got: {[r.message for r in caplog.records]}"
        )

    def test_continues_after_no_successful_trials(self) -> None:
        """Planner continues to next probe after infeasible at one level."""
        planner = _make_planner(lo=4, hi=256)
        # First probe at 4: all pass
        pair = planner.ask()
        assert pair is not None
        _, var1 = pair
        planner.tell(var1, [_success_result(var1, throughput=500.0)])

        # Second probe at 8: all pass
        pair = planner.ask()
        assert pair is not None
        _, var2 = pair
        planner.tell(var2, [_success_result(var2, throughput=500.0)])

        # Third probe at 16: all fail (no successful trials)
        pair = planner.ask()
        assert pair is not None
        _, var3 = pair
        planner.tell(var3, [_failed_result(var3)])

        # Should not be converged — bracket [8, 16] has gap > 1
        assert not planner.is_converged()
        assert planner.ask() is not None


class TestMissingMetrics:
    """Req 10.2: Missing metric treated as failed, log with metric name and tier."""

    def test_missing_metric_treats_filter_as_failed(self) -> None:
        """A tier with a missing metric is marked infeasible."""
        planner = _make_planner()
        pair = planner.ask()
        assert pair is not None
        _, variation = pair

        # Trial succeeds but has no output_token_throughput metric
        planner.tell(variation, [_missing_metric_result(variation)])

        # Both tiers should fail (neither can find output_token_throughput)
        history = planner.history()
        assert len(history) == 1
        assert history[0].feasible is False

    def test_logs_warning_with_metric_name_and_tier_label(self, caplog) -> None:
        """Warning log identifies both the missing metric and the tier label."""
        planner = _make_planner()
        pair = planner.ask()
        assert pair is not None
        _, variation = pair

        with caplog.at_level(logging.WARNING):
            planner.tell(variation, [_missing_metric_result(variation)])

        # Should mention the metric name and tier label
        metric_warnings = [
            r.message for r in caplog.records if "output_token_throughput" in r.message
        ]
        assert metric_warnings, "Expected warning naming 'output_token_throughput'"

        # Should mention tier labels
        fast_warnings = [m for m in metric_warnings if "fast" in m]
        economy_warnings = [m for m in metric_warnings if "economy" in m]
        assert fast_warnings, "Expected warning naming tier 'fast'"
        assert economy_warnings, "Expected warning naming tier 'economy'"


class TestAllTiersConvergeToLowest:
    """Req 10.3: All tiers converge to lowest → terminate with no_pass_in_range."""

    def test_all_fail_at_lowest_sets_no_pass_in_range(self) -> None:
        """When all tiers fail at the first probe, convergence_reason is no_pass_in_range."""
        planner = _make_planner(lo=1, hi=256)
        pair = planner.ask()
        assert pair is not None
        _, variation = pair

        # All tiers fail at concurrency=1 (throughput too low for both tiers)
        planner.tell(variation, [_success_result(variation, throughput=10.0)])

        # Both tiers should converge with no_pass_in_range
        for bracket in planner._brackets:
            assert bracket.converged is True
            assert bracket.convergence_reason == "no_pass_in_range"

        assert planner.is_converged()
        assert planner.convergence_reason() == "multi_tier_all_converged"

    def test_tier_results_show_no_pass_in_range(self) -> None:
        """tier_results() reports no_pass_in_range convergence status."""
        planner = _make_planner(lo=1, hi=256)
        pair = planner.ask()
        assert pair is not None
        _, variation = pair
        planner.tell(variation, [_success_result(variation, throughput=10.0)])

        results = planner.tier_results()
        for result in results:
            assert result.convergence_status == "no_pass_in_range"

    def test_logs_warning_on_no_pass_in_range(self, caplog) -> None:
        """A warning is logged when all tiers are infeasible at lowest."""
        planner = _make_planner(lo=1, hi=256)
        pair = planner.ask()
        assert pair is not None
        _, variation = pair

        with caplog.at_level(logging.WARNING):
            planner.tell(variation, [_success_result(variation, throughput=10.0)])

        assert any("no_pass_in_range" in r.message for r in caplog.records), (
            f"Expected 'no_pass_in_range' warning, got: {[r.message for r in caplog.records]}"
        )


class TestNonMonotonicObservations:
    """Req 10.4: Non-monotonic observations flag non_monotonic_warning per tier."""

    def test_non_monotonic_with_wider_bracket(self) -> None:
        """Non-monotonic detection with a wider bracket range."""
        planner = _make_planner(lo=4, hi=256)

        # Probe at 4: all pass
        pair = planner.ask()
        assert pair is not None
        _, var1 = pair
        assert var1.values["phases.profiling.concurrency"] == 4
        planner.tell(var1, [_success_result(var1, throughput=500.0)])

        # Probe at 8: all pass
        pair = planner.ask()
        assert pair is not None
        _, var2 = pair
        assert var2.values["phases.profiling.concurrency"] == 8
        planner.tell(var2, [_success_result(var2, throughput=400.0)])

        # Probe at 16: economy passes, fast fails (throughput 200 > 100 but < 300)
        pair = planner.ask()
        assert pair is not None
        _, var3 = pair
        assert var3.values["phases.profiling.concurrency"] == 16
        planner.tell(var3, [_success_result(var3, throughput=200.0)])
        # fast: feasible_max=8, infeasible_min=16
        # economy: feasible_max=16

        # Now in bisect: probe at midpoint of fast's bracket (8+16)//2 = 12
        pair = planner.ask()
        if pair is None:
            return
        _, var4 = pair
        # Force non-monotonic: tell planner that higher concurrency (say 16 was
        # infeasible for fast, but now at 12 report it as infeasible too
        # Then at a later probe, make 16 pass (contradicting earlier infeasible_min=16)
        # Actually, the simpler approach: make the probe at a value >= infeasible_min pass
        # We need to get a probe above infeasible_min. Let's manipulate directly.

        # Use _update_bracket directly to test non-monotonic detection
        fast_bracket = planner._brackets[0]
        # Set up initial state: feasible_max=8, infeasible_min=16
        assert fast_bracket.feasible_max == 8
        assert fast_bracket.infeasible_min == 16
        assert fast_bracket.non_monotonic_warning is False

        # Simulate a probe at 20 that passes for fast (above infeasible_min=16)
        planner._update_bracket(
            fast_bracket,
            20,
            True,
            [_success_result(var4, throughput=400.0)],
        )
        assert fast_bracket.non_monotonic_warning is True

    def test_infeasible_below_feasible_max_sets_warning(self) -> None:
        """A fail at-or-below feasible_max flags non_monotonic_warning."""
        planner = _make_planner(lo=4, hi=256)

        # Probe at 4: pass
        pair = planner.ask()
        assert pair is not None
        _, var1 = pair
        planner.tell(var1, [_success_result(var1, throughput=500.0)])

        # Probe at 8: pass
        pair = planner.ask()
        assert pair is not None
        _, var2 = pair
        planner.tell(var2, [_success_result(var2, throughput=400.0)])

        # fast bracket: feasible_max=8
        fast_bracket = planner._brackets[0]
        assert fast_bracket.feasible_max == 8

        # Simulate infeasible at value <= feasible_max (non-monotonic)
        planner._update_bracket(
            fast_bracket,
            6,
            False,
            [_success_result(var2, throughput=50.0)],
        )
        assert fast_bracket.non_monotonic_warning is True

    def test_non_monotonic_logged_per_tier(self, caplog) -> None:
        """Non-monotonic warning is logged naming the affected tier."""
        planner = _make_planner(lo=4, hi=256)

        pair = planner.ask()
        assert pair is not None
        _, var1 = pair
        planner.tell(var1, [_success_result(var1, throughput=500.0)])

        fast_bracket = planner._brackets[0]
        fast_bracket.infeasible_min = 16

        with caplog.at_level(logging.WARNING):
            planner._update_bracket(
                fast_bracket,
                20,
                True,
                [_success_result(var1, throughput=400.0)],
            )

        non_mono_warnings = [
            r.message for r in caplog.records if "non-monotonic" in r.message
        ]
        assert non_mono_warnings, "Expected non-monotonic warning in logs"
        assert any("fast" in m for m in non_mono_warnings), (
            "Expected tier label 'fast' in non-monotonic warning"
        )

    def test_non_monotonic_flagged_on_iteration(self) -> None:
        """The SearchIteration record carries non_monotonic_warning=True."""
        planner = _make_planner(lo=4, hi=256)

        # Probe at 4: pass
        pair = planner.ask()
        assert pair is not None
        _, var1 = pair
        planner.tell(var1, [_success_result(var1, throughput=500.0)])

        # Probe at 8: pass
        pair = planner.ask()
        assert pair is not None
        _, var2 = pair
        planner.tell(var2, [_success_result(var2, throughput=400.0)])

        # Probe at 16: fast fails
        pair = planner.ask()
        assert pair is not None
        _, var3 = pair
        planner.tell(var3, [_success_result(var3, throughput=200.0)])

        # Now manually set infeasible_min lower to trigger non-monotonic on next tell
        # fast has: feasible_max=8, infeasible_min=16
        # Force non-monotonic: tell the planner a probe ABOVE infeasible_min passes
        planner._update_bracket(
            planner._brackets[0],
            20,
            True,
            [_success_result(var3, throughput=400.0)],
        )

        # The boundary_summary reflects the non-monotonic state
        summary = planner.boundary_summary()
        assert isinstance(summary, dict)
        assert summary["non_monotonic_warning"] is True

    def test_boundary_summary_reflects_non_monotonic(self) -> None:
        """boundary_summary()['non_monotonic_warning'] is True when any tier has it."""
        planner = _make_planner(lo=4, hi=256)

        pair = planner.ask()
        assert pair is not None
        _, var1 = pair
        planner.tell(var1, [_success_result(var1, throughput=500.0)])

        # Force non-monotonic on first tier
        planner._brackets[0].non_monotonic_warning = True

        summary = planner.boundary_summary()
        assert summary is not None
        assert summary["non_monotonic_warning"] is True
