# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Property test: Per-Tier Evaluation From Shared Observation.

Feature: multi-tier-slo-search, Property 3: Per-Tier Evaluation From Shared Observation

Validates: Requirements 2.1, 2.3

For any observation at concurrency X and any set of SLO tiers, evaluating the
observation against all tiers SHALL produce one independent pass/fail verdict per
tier, where a tier passes iff at least one successful trial satisfies ALL of that
tier's filters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from hypothesis import given, settings
from hypothesis import strategies as st

from aiperf.common.models.export_models import JsonMetricResult
from aiperf.config.sweep.adaptive import SLAFilter, SLOTier
from aiperf.orchestrator.models import RunResult
from aiperf.orchestrator.search_planner._sla_helpers import iteration_feasibility

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


def _metric_result(value: float, stat: str) -> JsonMetricResult:
    """Create a JsonMetricResult with a specific stat set to value."""
    kwargs: dict[str, float | str] = {"unit": "ms"}
    kwargs[stat] = value
    return JsonMetricResult(**kwargs)


def _make_sla_filter(
    metric_tag: str,
    stat: Literal["avg", "p50", "p90", "p95", "p99"],
    op: Literal["lt", "le", "gt", "ge"],
    threshold: float,
) -> SLAFilter:
    """Construct an SLAFilter from plain args."""
    return SLAFilter(metric_tag=metric_tag, stat=stat, op=op, threshold=threshold)


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
    """Construct an SLOTier from plain args."""
    return SLOTier(label=label, filters=filters)


def _tier_strategy(label: str) -> st.SearchStrategy[SLOTier]:
    """Generate an SLOTier with 1-3 filters."""
    return st.builds(
        _make_slo_tier,
        label=st.just(label),
        filters=st.lists(_sla_filter_strategy(), min_size=1, max_size=3),
    )


def _run_result_with_metrics(
    metric_values: dict[str, dict[str, float]],
    success: bool = True,
) -> RunResult:
    """Create a RunResult with specific metric values.

    metric_values maps metric_tag -> {stat: value}.
    """
    summary_metrics: dict[str, JsonMetricResult] = {}
    for tag, stat_values in metric_values.items():
        kwargs: dict[str, float | str] = {"unit": "ms"}
        for stat_name, val in stat_values.items():
            kwargs[stat_name] = val
        summary_metrics[tag] = JsonMetricResult(**kwargs)
    return RunResult(
        label="trial",
        success=success,
        summary_metrics=summary_metrics,
        artifacts_path=Path("/tmp/fake"),
    )


def _multi_tier_config_strategy() -> st.SearchStrategy[list[SLOTier]]:
    """Generate 2-5 tiers with different configurations."""
    return st.integers(min_value=2, max_value=5).flatmap(
        lambda n: st.lists(
            st.integers(min_value=0, max_value=n - 1).flatmap(
                lambda i: _tier_strategy(f"tier_{i}")
            ),
            min_size=n,
            max_size=n,
        )
    )


def _observation_strategy() -> st.SearchStrategy[list[RunResult]]:
    """Generate 1-5 RunResults with random metric values for each known metric/stat."""
    metric_values_strategy = st.fixed_dictionaries(
        {
            tag: st.fixed_dictionaries(
                {
                    stat: st.floats(
                        min_value=0.1,
                        max_value=10000.0,
                        allow_nan=False,
                        allow_infinity=False,
                    )
                    for stat in STATS
                }
            )
            for tag in METRIC_TAGS
        }
    )
    run_strategy = st.builds(
        lambda mv: _run_result_with_metrics(mv, success=True),
        mv=metric_values_strategy,
    )
    return st.lists(run_strategy, min_size=1, max_size=5)


# ---------------------------------------------------------------------------
# Property 3: Per-Tier Evaluation From Shared Observation
# ---------------------------------------------------------------------------


class TestProperty3PerTierEvaluationFromSharedObservation:
    """Property 3: Per-Tier Evaluation From Shared Observation.

    **Validates: Requirements 2.1, 2.3**
    """

    @given(
        tiers=_multi_tier_config_strategy(),
        observation=_observation_strategy(),
    )
    @settings(max_examples=100, deadline=None)
    def test_each_tier_produces_independent_verdict(
        self,
        tiers: list[SLOTier],
        observation: list[RunResult],
    ) -> None:
        """Each tier's verdict depends only on its own filters, not other tiers.

        **Validates: Requirements 2.1, 2.3**
        """
        verdicts = [iteration_feasibility(observation, tier.filters) for tier in tiers]

        # Each verdict must be a boolean
        assert all(isinstance(v, bool) for v in verdicts)
        # We get exactly one verdict per tier
        assert len(verdicts) == len(tiers)

        # Verify independence: each verdict is consistent with manual evaluation
        for tier, verdict in zip(tiers, verdicts, strict=True):
            # A tier passes iff at least one successful trial satisfies ALL filters
            expected = any(
                all(_trial_satisfies_filter(run, f) for f in tier.filters)
                for run in observation
                if run.success
            )
            assert verdict == expected, (
                f"Tier {tier.label}: expected {expected}, got {verdict}"
            )

    @given(
        metric_value=st.floats(
            min_value=50.0, max_value=500.0, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=100, deadline=None)
    def test_lenient_tier_passes_while_strict_tier_fails(
        self,
        metric_value: float,
    ) -> None:
        """A lenient tier can pass while a strict tier fails on the same observation.

        Constructs tiers where one has a lower threshold (lenient, gt) and another
        has a higher threshold (strict, gt). When metric value is between them,
        the lenient tier passes and the strict tier fails.

        **Validates: Requirements 2.1, 2.3**
        """
        # Strict tier: output_token_throughput avg > 400
        strict_tier = SLOTier(
            label="strict",
            filters=[
                SLAFilter(
                    metric_tag="output_token_throughput",
                    stat="avg",
                    op="gt",
                    threshold=400.0,
                )
            ],
        )
        # Lenient tier: output_token_throughput avg > 50
        lenient_tier = SLOTier(
            label="lenient",
            filters=[
                SLAFilter(
                    metric_tag="output_token_throughput",
                    stat="avg",
                    op="gt",
                    threshold=50.0,
                )
            ],
        )

        observation = [
            _run_result_with_metrics(
                {"output_token_throughput": {"avg": metric_value}}, success=True
            )
        ]

        strict_verdict = iteration_feasibility(observation, strict_tier.filters)
        lenient_verdict = iteration_feasibility(observation, lenient_tier.filters)

        # Lenient threshold is 50, strict threshold is 400.
        # If value > 400: both pass. If 50 < value <= 400: lenient passes, strict fails.
        # If value <= 50: both fail.
        if metric_value > 400.0:
            assert strict_verdict is True
            assert lenient_verdict is True
        elif metric_value > 50.0:
            assert strict_verdict is False
            assert lenient_verdict is True
        else:
            assert strict_verdict is False
            assert lenient_verdict is False

    @given(
        metric_value=st.floats(
            min_value=1.0, max_value=10000.0, allow_nan=False, allow_infinity=False
        ),
        num_tiers=st.integers(min_value=2, max_value=5),
    )
    @settings(max_examples=100, deadline=None)
    def test_passing_strict_implies_passing_lenient_when_ordered(
        self,
        metric_value: float,
        num_tiers: int,
    ) -> None:
        """When tiers are monotonically ordered (gt thresholds), passing a stricter
        tier implies passing all more lenient tiers.

        **Validates: Requirements 2.1, 2.3**
        """
        # Create ordered tiers: tier_0 is strictest (highest gt threshold),
        # tier_N is most lenient (lowest gt threshold).
        base_threshold = 100.0
        step = 50.0
        tiers = [
            SLOTier(
                label=f"tier_{i}",
                filters=[
                    SLAFilter(
                        metric_tag="output_token_throughput",
                        stat="avg",
                        op="gt",
                        threshold=base_threshold + (num_tiers - 1 - i) * step,
                    )
                ],
            )
            for i in range(num_tiers)
        ]
        # tiers[0] has highest threshold (strictest), tiers[-1] has lowest (most lenient)

        observation = [
            _run_result_with_metrics(
                {"output_token_throughput": {"avg": metric_value}}, success=True
            )
        ]

        verdicts = [iteration_feasibility(observation, tier.filters) for tier in tiers]

        # Monotonicity: if strict tier passes, all more-lenient tiers must also pass
        for i in range(len(verdicts)):
            if verdicts[i]:
                # All tiers with index > i (more lenient) must also pass
                for j in range(i + 1, len(verdicts)):
                    assert verdicts[j] is True, (
                        f"Tier {i} passes (threshold={tiers[i].filters[0].threshold}) "
                        f"but tier {j} fails (threshold={tiers[j].filters[0].threshold}) "
                        f"with metric_value={metric_value}"
                    )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _trial_satisfies_filter(run: RunResult, sla: SLAFilter) -> bool:
    """Local reimplementation to verify independence of iteration_feasibility."""
    if not run.success:
        return False
    metric = run.summary_metrics.get(sla.metric_tag)
    if metric is None:
        return False
    value = getattr(metric, sla.stat, None)
    if value is None:
        return False
    if sla.op == "lt":
        return value < sla.threshold
    if sla.op == "le":
        return value <= sla.threshold
    if sla.op == "gt":
        return value > sla.threshold
    return value >= sla.threshold
