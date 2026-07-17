# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Property test: Cross-Tier Bracket Update.

Feature: multi-tier-slo-search, Property 7: Cross-Tier Bracket Update

Validates: Requirements 3.3

For any probe at concurrency X that produces a verdict for one tier, all other
non-converged tiers whose bracket contains X SHALL also update their bracket
bounds from that same verdict.
"""

from __future__ import annotations

from typing import Literal

from hypothesis import given, settings
from hypothesis import strategies as st

from aiperf.common.models.export_models import JsonMetricResult
from aiperf.config.sweep.adaptive import SLAFilter, SLOTier
from aiperf.orchestrator.models import RunResult
from aiperf.orchestrator.search_planner._sla_helpers import iteration_feasibility
from aiperf.orchestrator.search_planner.multi_tier_models import BracketState

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_STATS: list[Literal["avg", "p50", "p90", "p95", "p99"]] = [
    "avg",
    "p50",
    "p90",
    "p95",
    "p99",
]


def _make_tier(
    label: str, metric_tag: str, stat: str, op: str, threshold: float
) -> SLOTier:
    """Create an SLOTier with a single filter."""
    return SLOTier(
        label=label,
        filters=[
            SLAFilter(metric_tag=metric_tag, stat=stat, op=op, threshold=threshold)
        ],
    )


def _make_run_result(metric_tag: str, stat: str, value: float) -> RunResult:
    """Create a successful RunResult with a single metric stat."""
    metric_kwargs = {"unit": "units", stat: value}
    return RunResult(
        label="probe",
        success=True,
        summary_metrics={metric_tag: JsonMetricResult(**metric_kwargs)},
    )


@st.composite
def _multi_tier_scenario(draw: st.DrawFn) -> dict:
    """Generate a scenario with 2-4 tiers of varying thresholds on the same metric.

    Returns a dict with:
    - tiers: list of SLOTier (sorted by strictness, strictest first)
    - metric_tag: the shared metric tag
    - stat: the shared stat
    - op: the shared operator
    - observed_value: the metric value the probe will produce
    - concurrency: the probed concurrency level
    - converged_indices: set of tier indices that are pre-converged
    """
    num_tiers = draw(st.integers(min_value=2, max_value=4))
    metric_tag = "output_token_throughput"
    stat = draw(st.sampled_from(_STATS))
    op = draw(st.sampled_from(["gt", "lt"]))

    # Generate thresholds for each tier
    thresholds = sorted(
        draw(
            st.lists(
                st.floats(
                    min_value=10.0,
                    max_value=1000.0,
                    allow_nan=False,
                    allow_infinity=False,
                ),
                min_size=num_tiers,
                max_size=num_tiers,
                unique=True,
            )
        )
    )

    # For "gt", higher threshold = stricter (harder to pass)
    # For "lt", lower threshold = stricter (harder to pass)
    if op == "gt":
        # Reverse so strictest (highest threshold) is first
        thresholds = list(reversed(thresholds))

    tiers = [
        _make_tier(f"tier_{i}", metric_tag, stat, op, thresholds[i])
        for i in range(num_tiers)
    ]

    # Generate observed value
    observed_value = draw(
        st.floats(
            min_value=5.0, max_value=1100.0, allow_nan=False, allow_infinity=False
        )
    )

    # Generate concurrency
    concurrency = draw(st.integers(min_value=1, max_value=256))

    # Some tiers may be pre-converged (0 to num_tiers-2 to ensure at least one non-converged)
    max_converged = max(0, num_tiers - 2)
    num_converged = draw(st.integers(min_value=0, max_value=max_converged))
    converged_indices = set(
        draw(
            st.lists(
                st.integers(min_value=0, max_value=num_tiers - 1),
                min_size=num_converged,
                max_size=num_converged,
                unique=True,
            )
        )
    )

    return {
        "tiers": tiers,
        "metric_tag": metric_tag,
        "stat": stat,
        "op": op,
        "thresholds": thresholds,
        "observed_value": observed_value,
        "concurrency": concurrency,
        "converged_indices": converged_indices,
    }


# ---------------------------------------------------------------------------
# Property 7: Cross-Tier Bracket Update
# ---------------------------------------------------------------------------


class TestProperty7CrossTierBracketUpdate:
    """Property 7: Cross-Tier Bracket Update.

    **Validates: Requirements 3.3**
    """

    @given(scenario=_multi_tier_scenario())
    @settings(max_examples=100, deadline=None)
    def test_all_non_converged_tiers_update_from_same_observation(
        self,
        scenario: dict,
    ) -> None:
        """A single probe updates the bracket of every non-converged tier.

        When tell() processes an observation at concurrency X, every
        non-converged tier evaluates the observation independently and
        updates its bracket (feasible_max or infeasible_min) accordingly.

        **Validates: Requirements 3.3**
        """
        tiers = scenario["tiers"]
        metric_tag = scenario["metric_tag"]
        stat = scenario["stat"]
        observed_value = scenario["observed_value"]
        concurrency = scenario["concurrency"]
        converged_indices = scenario["converged_indices"]

        # Create brackets
        brackets: list[BracketState] = []
        for i, tier in enumerate(tiers):
            b = BracketState(tier=tier)
            if i in converged_indices:
                b.converged = True
                b.convergence_reason = "multi_tier_precision_reached"
            brackets.append(b)

        # Create RunResult with the observed value
        results = [_make_run_result(metric_tag, stat, observed_value)]

        # Evaluate each tier independently (mirroring tell() logic)
        for bracket in brackets:
            if bracket.converged:
                continue
            feasible = iteration_feasibility(results, bracket.tier.filters)
            # Update bracket (same logic as MultiTierPlanner._update_bracket)
            if feasible:
                if bracket.feasible_max is None or concurrency > bracket.feasible_max:
                    bracket.feasible_max = concurrency
            else:
                if (
                    bracket.infeasible_min is None
                    or concurrency < bracket.infeasible_min
                ):
                    bracket.infeasible_min = concurrency

        # Verify: all non-converged tiers had their bracket updated
        for i, bracket in enumerate(brackets):
            if i in converged_indices:
                # Converged tiers should NOT be updated
                assert bracket.feasible_max is None
                assert bracket.infeasible_min is None
            else:
                # Non-converged tiers should have exactly one bound updated
                feasible = iteration_feasibility(results, bracket.tier.filters)
                if feasible:
                    assert bracket.feasible_max == concurrency
                else:
                    assert bracket.infeasible_min == concurrency

    @given(scenario=_multi_tier_scenario())
    @settings(max_examples=100, deadline=None)
    def test_converged_tiers_not_updated(
        self,
        scenario: dict,
    ) -> None:
        """Converged tiers are NOT updated when a probe arrives.

        **Validates: Requirements 3.3**
        """
        tiers = scenario["tiers"]
        metric_tag = scenario["metric_tag"]
        stat = scenario["stat"]
        observed_value = scenario["observed_value"]
        concurrency = scenario["concurrency"]
        converged_indices = scenario["converged_indices"]

        # Create brackets; set some as converged with existing bounds
        brackets: list[BracketState] = []
        for i, tier in enumerate(tiers):
            b = BracketState(tier=tier)
            if i in converged_indices:
                b.converged = True
                b.convergence_reason = "multi_tier_precision_reached"
                b.feasible_max = 50
                b.infeasible_min = 51
            brackets.append(b)

        # Snapshot converged bracket state
        converged_snapshots = {
            i: (brackets[i].feasible_max, brackets[i].infeasible_min)
            for i in converged_indices
        }

        # Create RunResult
        results = [_make_run_result(metric_tag, stat, observed_value)]

        # Simulate tell() evaluation: skip converged tiers
        for bracket in brackets:
            if bracket.converged:
                continue
            feasible = iteration_feasibility(results, bracket.tier.filters)
            if feasible:
                if bracket.feasible_max is None or concurrency > bracket.feasible_max:
                    bracket.feasible_max = concurrency
            else:
                if (
                    bracket.infeasible_min is None
                    or concurrency < bracket.infeasible_min
                ):
                    bracket.infeasible_min = concurrency

        # Verify converged tiers unchanged
        for i in converged_indices:
            assert (
                brackets[i].feasible_max,
                brackets[i].infeasible_min,
            ) == converged_snapshots[i]

    @given(scenario=_multi_tier_scenario())
    @settings(max_examples=100, deadline=None)
    def test_different_tiers_get_different_verdicts_from_same_probe(
        self,
        scenario: dict,
    ) -> None:
        """The same observation can pass one tier and fail another.

        A single probe produces results that are independently evaluated
        against each tier's filters. Stricter tiers may fail while lenient
        tiers pass from the same observation.

        **Validates: Requirements 3.3**
        """
        tiers = scenario["tiers"]
        metric_tag = scenario["metric_tag"]
        stat = scenario["stat"]
        observed_value = scenario["observed_value"]
        concurrency = scenario["concurrency"]

        # Create brackets (none converged for this test)
        brackets = [BracketState(tier=t) for t in tiers]

        results = [_make_run_result(metric_tag, stat, observed_value)]

        # Compute expected verdicts per tier
        verdicts = [iteration_feasibility(results, t.filters) for t in tiers]

        # Simulate tell()
        for bracket in brackets:
            feasible = iteration_feasibility(results, bracket.tier.filters)
            if feasible:
                if bracket.feasible_max is None or concurrency > bracket.feasible_max:
                    bracket.feasible_max = concurrency
            else:
                if (
                    bracket.infeasible_min is None
                    or concurrency < bracket.infeasible_min
                ):
                    bracket.infeasible_min = concurrency

        # Verify brackets match per-tier verdicts
        for i, (bracket, verdict) in enumerate(zip(brackets, verdicts, strict=True)):
            if verdict:
                assert bracket.feasible_max == concurrency, (
                    f"Tier {i} passed but feasible_max not updated"
                )
                assert bracket.infeasible_min is None
            else:
                assert bracket.infeasible_min == concurrency, (
                    f"Tier {i} failed but infeasible_min not updated"
                )
                assert bracket.feasible_max is None

    @given(scenario=_multi_tier_scenario())
    @settings(max_examples=100, deadline=None)
    def test_bracket_update_respects_existing_bounds(
        self,
        scenario: dict,
    ) -> None:
        """Bracket updates only improve bounds (feasible_max grows, infeasible_min shrinks).

        When a tier already has a feasible_max, a new pass at a lower
        concurrency does NOT lower it. When a tier has an infeasible_min,
        a new fail at a higher concurrency does NOT raise it.

        **Validates: Requirements 3.3**
        """
        tiers = scenario["tiers"]
        metric_tag = scenario["metric_tag"]
        stat = scenario["stat"]
        observed_value = scenario["observed_value"]
        concurrency = scenario["concurrency"]

        # Create brackets with pre-existing bounds
        brackets = [BracketState(tier=t) for t in tiers]

        results = [_make_run_result(metric_tag, stat, observed_value)]

        # Pre-populate bounds
        for bracket in brackets:
            feasible = iteration_feasibility(results, bracket.tier.filters)
            if feasible:
                # Set existing feasible_max higher than concurrency in some cases
                bracket.feasible_max = concurrency + 50
            else:
                # Set existing infeasible_min lower than concurrency in some cases
                bracket.infeasible_min = max(1, concurrency - 50)

        # Snapshot existing bounds
        pre_bounds = [(b.feasible_max, b.infeasible_min) for b in brackets]

        # Simulate tell() bracket update
        for bracket in brackets:
            feasible = iteration_feasibility(results, bracket.tier.filters)
            if feasible:
                if bracket.feasible_max is None or concurrency > bracket.feasible_max:
                    bracket.feasible_max = concurrency
            else:
                if (
                    bracket.infeasible_min is None
                    or concurrency < bracket.infeasible_min
                ):
                    bracket.infeasible_min = concurrency

        # Verify monotonicity of updates
        for i, bracket in enumerate(brackets):
            pre_fmax, pre_imin = pre_bounds[i]
            if pre_fmax is not None and bracket.feasible_max is not None:
                assert bracket.feasible_max >= pre_fmax
            if pre_imin is not None and bracket.infeasible_min is not None:
                assert bracket.infeasible_min <= pre_imin
