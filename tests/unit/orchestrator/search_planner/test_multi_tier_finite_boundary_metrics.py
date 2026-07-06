# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression test: non-finite (NaN/Inf) stat values are scrubbed from
``TierResult.boundary_metrics`` before they reach ``search_history.json``.

Covers the ``is_finite_value`` guard in
``MultiTierPlanner._extract_boundary_metrics``. ``boundary_metrics`` is a
``dict[str, Any]`` so the ``FiniteFloat`` field validator does not guard it;
the scrub must happen in the extractor. ``JsonMetricResult`` stat fields are
plain ``float | None``, so a NaN/Inf stat is constructible and can leak.
"""

from __future__ import annotations

import math

from aiperf.common.models.export_models import JsonMetricResult
from aiperf.config.config import BenchmarkConfig
from aiperf.config.sweep import AdaptiveSearchSweep, Objective
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


def _throughput_tier(label: str, threshold: float) -> SLOTier:
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


def _cfg() -> AdaptiveSearchSweep:
    return AdaptiveSearchSweep(
        search_space=[
            SearchSpaceDimension(
                path="phases.profiling.concurrency", lo=1, hi=256, kind="int"
            )
        ],
        objectives=[Objective(metric="output_token_throughput", direction="maximize")],
        max_iterations=50,
        n_initial_points=2,
        sla_filters=[],
        sla_tiers=[],
    )


def test_boundary_metrics_drops_non_finite_stats():
    """A passing probe whose stats include NaN/Inf must not leak those values
    into any tier's ``boundary_metrics``; finite stats are preserved."""
    tiers = [_throughput_tier("fast", 300.0), _throughput_tier("standard", 100.0)]
    planner = MultiTierPlanner(base_config=_base_config(), cfg=_cfg(), tiers=tiers)

    pair = planner.ask()
    assert pair is not None
    _, variation = pair

    # avg=500 passes both tiers (sets feasible_max); p50/p99 are non-finite.
    result = RunResult(
        label="trial_0",
        success=True,
        summary_metrics={
            "output_token_throughput": JsonMetricResult(
                unit="tok/s",
                avg=500.0,
                p50=float("nan"),
                p99=float("inf"),
            ),
        },
        variation_label=variation.label,
        variation_values=variation.values,
    )
    planner.tell(variation, [result])

    tier_results = planner.tier_results()
    assert tier_results, "expected per-tier results"

    for tr in tier_results:
        boundary_metrics = tr.boundary_metrics or {}
        # Nothing non-finite may survive into the serialized output.
        for tag, stats in boundary_metrics.items():
            for stat, value in stats.items():
                assert math.isfinite(value), (
                    f"non-finite stat leaked into boundary_metrics: "
                    f"{tag}.{stat}={value}"
                )
        otp = boundary_metrics.get("output_token_throughput", {})
        # Finite stat preserved; non-finite stats dropped.
        assert otp.get("avg") == 500.0
        assert "p50" not in otp
        assert "p99" not in otp
