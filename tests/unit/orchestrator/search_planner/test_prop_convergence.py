# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Property test: Convergence and Partial Results.

Feature: multi-tier-slo-search, Property 6: Convergence and Partial Results

Validates: Requirements 3.2, 3.5

For any multi-tier search, if all tier brackets resolve to within the
configured precision, the planner SHALL terminate. If max_iterations is
exhausted first, every tier SHALL report its best-known bracket bounds and a
convergence_status of "partial".
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
from aiperf.orchestrator.search_planner.multi_tier_models import BracketState
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
def _converged_bracket_set(draw: st.DrawFn) -> list[BracketState]:
    """Generate 2-5 BracketState objects where all have converged=True."""
    count = draw(st.integers(min_value=2, max_value=5))
    brackets = []
    for i in range(count):
        feasible_max = draw(st.integers(min_value=1, max_value=500))
        tier = _make_tier(f"tier_{i}", threshold=float(100 * (i + 1)))
        b = BracketState(
            tier=tier,
            feasible_max=feasible_max,
            infeasible_min=feasible_max + 1,
            converged=True,
            convergence_reason="multi_tier_precision_reached",
            probe_count=draw(st.integers(min_value=1, max_value=20)),
        )
        brackets.append(b)
    return brackets


@st.composite
def _low_max_iterations_config(draw: st.DrawFn) -> int:
    """Generate a low max_iterations value (2-6) to force early termination."""
    return draw(st.integers(min_value=2, max_value=6))


@st.composite
def _precision_bracket_pair(draw: st.DrawFn) -> tuple[int, int]:
    """Generate feasible_max and infeasible_min where gap <= 1."""
    feasible_max = draw(st.integers(min_value=2, max_value=500))
    # gap is exactly 1 (precision reached)
    return feasible_max, feasible_max + 1


# ---------------------------------------------------------------------------
# Property 6: Convergence and Partial Results
# ---------------------------------------------------------------------------


class TestProperty6ConvergenceAndPartialResults:
    """Property 6: Convergence and Partial Results.

    **Validates: Requirements 3.2, 3.5**
    """

    @given(brackets=_converged_bracket_set())
    @settings(max_examples=100, deadline=None)
    def test_all_brackets_converged_implies_planner_terminates(
        self,
        brackets: list[BracketState],
    ) -> None:
        """When all brackets have converged=True, is_converged() returns True.

        **Validates: Requirements 3.2**
        """
        tiers = [b.tier for b in brackets]
        cfg = _adaptive_cfg(max_iterations=100)
        planner = MultiTierPlanner(_base_config(), cfg, tiers)

        # Directly inject converged bracket states
        planner._brackets = brackets

        assert planner.is_converged() is True
        assert planner.convergence_reason() == "multi_tier_all_converged"

    @given(max_iters=_low_max_iterations_config())
    @settings(max_examples=100, deadline=None)
    def test_max_iterations_exhausted_reports_partial_results(
        self,
        max_iters: int,
    ) -> None:
        """When max_iterations exhausted, non-converged tiers report 'partial'.

        **Validates: Requirements 3.5**
        """
        # Both tiers use the same threshold (gt:50). We control the throughput
        # to create a bracket boundary in the middle of [1, 512]:
        # low concurrency passes, high concurrency fails. This ensures both
        # bounds get established for both tiers, but with a wide gap that
        # cannot converge within the low iteration budget.
        tier_a = _make_tier("alpha", threshold=50.0)
        tier_b = _make_tier("beta", threshold=50.0)
        tiers = [tier_a, tier_b]

        cfg = _adaptive_cfg(lo=1, hi=512, max_iterations=max_iters)
        planner = MultiTierPlanner(_base_config(), cfg, tiers)

        # Directly inject bracket state: both tiers have wide gaps that
        # cannot converge within the iteration budget. This bypasses the
        # bracket phase entirely and tests pure max_iterations termination.
        for bracket in planner._brackets:
            bracket.feasible_max = 10
            bracket.infeasible_min = 200

        # Set planner to bisection phase so it tries to allocate probes
        planner._phase = "bisect"

        # Drive the planner through its iteration budget.
        # Return throughput that depends on concurrency to create realistic
        # bracket updates, but the gap stays too wide to converge.
        for _ in range(max_iters):
            if planner.is_converged():
                break
            pair = planner.ask()
            if pair is None:
                break
            bench_cfg, variation = pair
            concurrency = list(variation.values.values())[0]
            # Below 100: passes; above 100: fails. This keeps the gap wide.
            throughput = 100.0 if concurrency <= 100 else 20.0
            results = [_make_result(variation, throughput=throughput)]
            planner.tell(variation, results)

        assert planner.is_converged() is True
        assert planner.convergence_reason() == "max_iterations"

        # Verify tier_results reflect partial status for non-converged tiers
        tier_results = planner.tier_results()
        assert len(tier_results) == 2

        for result in tier_results:
            bracket = planner._brackets[
                next(
                    i
                    for i, b in enumerate(planner._brackets)
                    if b.tier.label == result.label
                )
            ]
            if not bracket.converged:
                assert result.convergence_status == "partial"

    @given(data=st.data())
    @settings(max_examples=100, deadline=None)
    def test_bracket_precision_reached_marks_tier_converged(
        self,
        data: st.DataObject,
    ) -> None:
        """Bracket with infeasible_min - feasible_max <= 1 converges with precision_reached.

        **Validates: Requirements 3.2**
        """
        feasible_max = data.draw(st.integers(min_value=2, max_value=500))
        tier = _make_tier("precise", threshold=100.0)

        bracket = BracketState(
            tier=tier,
            feasible_max=feasible_max,
            infeasible_min=feasible_max + 1,
            converged=False,
        )

        # Create a second tier to satisfy the 2+ tier requirement
        tier2 = _make_tier("other", threshold=200.0)
        bracket2 = BracketState(
            tier=tier2,
            feasible_max=feasible_max,
            infeasible_min=feasible_max + 1,
            converged=False,
        )

        tiers = [tier, tier2]
        cfg = _adaptive_cfg(max_iterations=100)
        planner = MultiTierPlanner(_base_config(), cfg, tiers)

        # Inject bracket state directly
        planner._brackets = [bracket, bracket2]

        # Trigger convergence check
        planner._check_bracket_convergence()

        assert bracket.converged is True
        assert bracket.convergence_reason == "multi_tier_precision_reached"
        assert bracket2.converged is True
        assert bracket2.convergence_reason == "multi_tier_precision_reached"

    @given(max_iters=_low_max_iterations_config())
    @settings(max_examples=100, deadline=None)
    def test_partial_results_include_best_known_bounds(
        self,
        max_iters: int,
    ) -> None:
        """Partial results always include bracket_lower and bracket_upper when known.

        **Validates: Requirements 3.5**
        """
        tier_a = _make_tier("alpha", threshold=50.0)
        tier_b = _make_tier("beta", threshold=500.0)
        tiers = [tier_a, tier_b]

        cfg = _adaptive_cfg(lo=1, hi=256, max_iterations=max_iters)
        planner = MultiTierPlanner(_base_config(), cfg, tiers)

        # Drive with throughput=100.0: "alpha" (gt:50) passes, "beta" (gt:500) fails
        for _ in range(max_iters):
            pair = planner.ask()
            if pair is None:
                break
            bench_cfg, variation = pair
            results = [_make_result(variation, throughput=100.0)]
            planner.tell(variation, results)

        assert planner.is_converged() is True

        tier_results = planner.tier_results()
        for result in tier_results:
            # All tier results should have non-None bracket fields when
            # any probes have been run (at least one bound known)
            bracket = planner._brackets[
                next(
                    i
                    for i, b in enumerate(planner._brackets)
                    if b.tier.label == result.label
                )
            ]
            if bracket.feasible_max is not None:
                assert result.bracket_lower is not None
            if bracket.infeasible_min is not None:
                assert result.bracket_upper is not None

    @given(
        num_tiers=st.integers(min_value=2, max_value=5),
        data=st.data(),
    )
    @settings(max_examples=100, deadline=None)
    def test_all_converged_reason_is_multi_tier_all_converged(
        self,
        num_tiers: int,
        data: st.DataObject,
    ) -> None:
        """When all brackets converge, convergence_reason is 'multi_tier_all_converged'.

        **Validates: Requirements 3.2**
        """
        tiers = [
            _make_tier(f"tier_{i}", threshold=float(50 * (i + 1)))
            for i in range(num_tiers)
        ]
        cfg = _adaptive_cfg(max_iterations=100)
        planner = MultiTierPlanner(_base_config(), cfg, tiers)

        # Set up all brackets as precision-converged
        for bracket in planner._brackets:
            fmax = data.draw(st.integers(min_value=2, max_value=500))
            bracket.feasible_max = fmax
            bracket.infeasible_min = fmax + 1
            bracket.converged = True
            bracket.convergence_reason = "multi_tier_precision_reached"

        assert planner.is_converged() is True
        assert planner.convergence_reason() == "multi_tier_all_converged"

    @given(max_iters=st.integers(min_value=2, max_value=5))
    @settings(max_examples=100, deadline=None)
    def test_convergence_terminates_ask_returns_none(
        self,
        max_iters: int,
    ) -> None:
        """After convergence, ask() returns None.

        **Validates: Requirements 3.2**
        """
        tier_a = _make_tier("t1", threshold=50.0)
        tier_b = _make_tier("t2", threshold=500.0)
        tiers = [tier_a, tier_b]

        cfg = _adaptive_cfg(lo=1, hi=128, max_iterations=max_iters)
        planner = MultiTierPlanner(_base_config(), cfg, tiers)

        # Exhaust iterations
        for _ in range(max_iters):
            pair = planner.ask()
            if pair is None:
                break
            _, variation = pair
            results = [_make_result(variation, throughput=100.0)]
            planner.tell(variation, results)

        # After convergence, ask returns None
        assert planner.is_converged() is True
        assert planner.ask() is None
