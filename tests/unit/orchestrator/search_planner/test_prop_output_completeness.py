# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Property test: Tier_Result Output Completeness.

Feature: multi-tier-slo-search, Property 11: Tier_Result Output Completeness

Validates: Requirements 6.1, 6.2, 6.4, 6.5

For any completed multi-tier search with N tiers, the output SHALL contain
exactly N TierResult entries, each with non-null label, bracket_lower OR
bracket_upper (at least one bound known), convergence_status, probe_count,
and filters fields.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from aiperf.common.models.export_models import JsonMetricResult
from aiperf.config.config import BenchmarkConfig
from aiperf.config.sweep import AdaptiveSearchSweep, Objective, SweepVariation
from aiperf.config.sweep.adaptive import SearchSpaceDimension, SLAFilter, SLOTier
from aiperf.orchestrator.aggregation.sweep import OptimizationDirection
from aiperf.orchestrator.models import RunResult
from aiperf.orchestrator.search_planner.multi_tier_planner import MultiTierPlanner

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_config() -> BenchmarkConfig:
    """Minimal BenchmarkConfig for the MultiTierPlanner."""
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


def _adaptive_cfg(
    *,
    lo: int = 1,
    hi: int = 1024,
    max_iterations: int = 50,
) -> AdaptiveSearchSweep:
    """Minimal AdaptiveSearchSweep for multi-tier testing."""
    return AdaptiveSearchSweep(
        planner="smooth_isotonic",
        search_space=[
            SearchSpaceDimension(
                path="phases.profiling.concurrency",
                lo=lo,
                hi=hi,
                kind="int",
            )
        ],
        objectives=[
            Objective(
                metric="output_token_throughput",
                stat="avg",
                direction=OptimizationDirection.MAXIMIZE,
            )
        ],
        max_iterations=max_iterations,
        n_initial_points=1,
        sla_filters=[],
        sla_warmup_seconds=0,
    )


def _make_tier(label: str, threshold: float) -> SLOTier:
    """Create an SLOTier with a single throughput filter."""
    return SLOTier(
        label=label,
        filters=[
            SLAFilter(
                metric_tag="output_token_throughput",
                stat="avg",
                op="gt",
                threshold=threshold,
            )
        ],
    )


def _make_result(
    variation: SweepVariation,
    *,
    throughput: float,
) -> RunResult:
    """Create a RunResult with a given throughput value."""
    return RunResult(
        label="t",
        success=True,
        summary_metrics={
            "output_token_throughput": JsonMetricResult(unit="tok/s", avg=throughput),
        },
        variation_label=variation.label,
        variation_values=variation.values,
    )


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


@st.composite
def _tier_set(draw: st.DrawFn) -> list[SLOTier]:
    """Generate 2-5 tiers with distinct labels and increasing thresholds."""
    num_tiers = draw(st.integers(min_value=2, max_value=5))
    thresholds = sorted(
        draw(
            st.lists(
                st.floats(
                    min_value=10.0,
                    max_value=500.0,
                    allow_nan=False,
                    allow_infinity=False,
                ),
                min_size=num_tiers,
                max_size=num_tiers,
                unique=True,
            )
        )
    )
    return [_make_tier(f"tier_{i}", thresholds[i]) for i in range(num_tiers)]


@st.composite
def _convergence_scenario(draw: st.DrawFn) -> dict:
    """Generate a scenario that runs to convergence or max_iterations.

    Returns dict with:
    - tiers: list[SLOTier]
    - max_iterations: int (small to keep test fast)
    - throughput_at_boundary: float (threshold between pass/fail)
    """
    tiers = draw(_tier_set())
    max_iterations = draw(st.integers(min_value=4, max_value=12))
    # Pick a throughput that causes some tiers to pass and some to fail
    # Use the median threshold as the boundary point
    median_threshold = sorted(t.filters[0].threshold for t in tiers)[len(tiers) // 2]
    throughput_at_boundary = draw(
        st.floats(
            min_value=median_threshold * 0.5,
            max_value=median_threshold * 2.0,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    return {
        "tiers": tiers,
        "max_iterations": max_iterations,
        "throughput_at_boundary": throughput_at_boundary,
    }


# ---------------------------------------------------------------------------
# Property 11: Tier_Result Output Completeness
# ---------------------------------------------------------------------------


class TestProperty11TierResultOutputCompleteness:
    """Property 11: Tier_Result Output Completeness.

    **Validates: Requirements 6.1, 6.2, 6.4, 6.5**
    """

    @given(scenario=_convergence_scenario())
    @settings(max_examples=100, deadline=None)
    def test_tier_results_count_equals_tier_count(
        self,
        scenario: dict,
    ) -> None:
        """After completion, tier_results() returns exactly N entries for N tiers.

        **Validates: Requirements 6.1**
        """
        tiers = scenario["tiers"]
        max_iterations = scenario["max_iterations"]
        throughput_boundary = scenario["throughput_at_boundary"]

        cfg = _adaptive_cfg(lo=1, hi=256, max_iterations=max_iterations)
        planner = MultiTierPlanner(_base_config(), cfg, tiers)

        # Drive planner to completion
        for _ in range(max_iterations):
            if planner.is_converged():
                break
            pair = planner.ask()
            if pair is None:
                break
            _, variation = pair
            concurrency = list(variation.values.values())[0]
            # Throughput decreases with concurrency to create a boundary
            throughput = throughput_boundary * (64.0 / max(concurrency, 1))
            results = [_make_result(variation, throughput=throughput)]
            planner.tell(variation, results)

        assert planner.is_converged() is True
        tier_results = planner.tier_results()
        assert len(tier_results) == len(tiers)

    @given(scenario=_convergence_scenario())
    @settings(max_examples=100, deadline=None)
    def test_tier_results_have_non_null_required_fields(
        self,
        scenario: dict,
    ) -> None:
        """Each TierResult has non-null label, convergence_status, probe_count, filters.

        **Validates: Requirements 6.2, 6.4**
        """
        tiers = scenario["tiers"]
        max_iterations = scenario["max_iterations"]
        throughput_boundary = scenario["throughput_at_boundary"]

        cfg = _adaptive_cfg(lo=1, hi=256, max_iterations=max_iterations)
        planner = MultiTierPlanner(_base_config(), cfg, tiers)

        for _ in range(max_iterations):
            if planner.is_converged():
                break
            pair = planner.ask()
            if pair is None:
                break
            _, variation = pair
            concurrency = list(variation.values.values())[0]
            throughput = throughput_boundary * (64.0 / max(concurrency, 1))
            results = [_make_result(variation, throughput=throughput)]
            planner.tell(variation, results)

        assert planner.is_converged() is True
        tier_results = planner.tier_results()

        for result in tier_results:
            assert result.label is not None
            assert result.convergence_status is not None
            assert result.convergence_status in {
                "converged",
                "partial",
                "no_pass_in_range",
                "no_failure_in_range",
            }
            assert result.probe_count is not None
            assert result.filters is not None

    @given(scenario=_convergence_scenario())
    @settings(max_examples=100, deadline=None)
    def test_tier_results_probe_count_non_negative(
        self,
        scenario: dict,
    ) -> None:
        """Each TierResult has probe_count >= 0.

        **Validates: Requirements 6.5**
        """
        tiers = scenario["tiers"]
        max_iterations = scenario["max_iterations"]
        throughput_boundary = scenario["throughput_at_boundary"]

        cfg = _adaptive_cfg(lo=1, hi=256, max_iterations=max_iterations)
        planner = MultiTierPlanner(_base_config(), cfg, tiers)

        for _ in range(max_iterations):
            if planner.is_converged():
                break
            pair = planner.ask()
            if pair is None:
                break
            _, variation = pair
            concurrency = list(variation.values.values())[0]
            throughput = throughput_boundary * (64.0 / max(concurrency, 1))
            results = [_make_result(variation, throughput=throughput)]
            planner.tell(variation, results)

        assert planner.is_converged() is True
        tier_results = planner.tier_results()

        for result in tier_results:
            assert result.probe_count >= 0

    @given(scenario=_convergence_scenario())
    @settings(max_examples=100, deadline=None)
    def test_tier_results_filters_non_empty(
        self,
        scenario: dict,
    ) -> None:
        """Each TierResult has a non-empty filters list.

        **Validates: Requirements 6.4**
        """
        tiers = scenario["tiers"]
        max_iterations = scenario["max_iterations"]
        throughput_boundary = scenario["throughput_at_boundary"]

        cfg = _adaptive_cfg(lo=1, hi=256, max_iterations=max_iterations)
        planner = MultiTierPlanner(_base_config(), cfg, tiers)

        for _ in range(max_iterations):
            if planner.is_converged():
                break
            pair = planner.ask()
            if pair is None:
                break
            _, variation = pair
            concurrency = list(variation.values.values())[0]
            throughput = throughput_boundary * (64.0 / max(concurrency, 1))
            results = [_make_result(variation, throughput=throughput)]
            planner.tell(variation, results)

        assert planner.is_converged() is True
        tier_results = planner.tier_results()

        for result in tier_results:
            assert len(result.filters) > 0

    @given(scenario=_convergence_scenario())
    @settings(max_examples=100, deadline=None)
    def test_tier_results_labels_match_configured_tiers(
        self,
        scenario: dict,
    ) -> None:
        """Each TierResult's label matches the corresponding tier's configured label.

        **Validates: Requirements 6.1, 6.2**
        """
        tiers = scenario["tiers"]
        max_iterations = scenario["max_iterations"]
        throughput_boundary = scenario["throughput_at_boundary"]

        cfg = _adaptive_cfg(lo=1, hi=256, max_iterations=max_iterations)
        planner = MultiTierPlanner(_base_config(), cfg, tiers)

        for _ in range(max_iterations):
            if planner.is_converged():
                break
            pair = planner.ask()
            if pair is None:
                break
            _, variation = pair
            concurrency = list(variation.values.values())[0]
            throughput = throughput_boundary * (64.0 / max(concurrency, 1))
            results = [_make_result(variation, throughput=throughput)]
            planner.tell(variation, results)

        assert planner.is_converged() is True
        tier_results = planner.tier_results()

        configured_labels = [t.label for t in tiers]
        result_labels = [r.label for r in tier_results]
        assert result_labels == configured_labels

    @given(scenario=_convergence_scenario())
    @settings(max_examples=100, deadline=None)
    def test_tier_results_have_at_least_one_bracket_bound(
        self,
        scenario: dict,
    ) -> None:
        """Each TierResult has bracket_lower OR bracket_upper (at least one bound known).

        **Validates: Requirements 6.4**
        """
        tiers = scenario["tiers"]
        max_iterations = scenario["max_iterations"]
        throughput_boundary = scenario["throughput_at_boundary"]

        cfg = _adaptive_cfg(lo=1, hi=256, max_iterations=max_iterations)
        planner = MultiTierPlanner(_base_config(), cfg, tiers)

        for _ in range(max_iterations):
            if planner.is_converged():
                break
            pair = planner.ask()
            if pair is None:
                break
            _, variation = pair
            concurrency = list(variation.values.values())[0]
            throughput = throughput_boundary * (64.0 / max(concurrency, 1))
            results = [_make_result(variation, throughput=throughput)]
            planner.tell(variation, results)

        assert planner.is_converged() is True
        tier_results = planner.tier_results()

        for result in tier_results:
            # At least one bracket bound must be known after the search completes
            assert result.bracket_lower is not None or result.bracket_upper is not None
