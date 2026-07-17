# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for multi-tier grid evaluation.

Validates that :func:`evaluate_tiers_on_grid` correctly evaluates all tiers
at every grid point and derives per-tier boundaries post-hoc.
"""

from __future__ import annotations

from aiperf.config.sweep.adaptive import SLAFilter, SLOTier
from aiperf.orchestrator.search_planner.multi_tier_grid import (
    evaluate_tiers_on_grid,
)


def _make_tier(label: str, filters: list[SLAFilter]) -> SLOTier:
    return SLOTier(label=label, filters=filters)


def _make_row(concurrency: int, metrics: dict) -> dict:
    return {
        "parameters": {"phases.profiling.concurrency": concurrency},
        "metrics": metrics,
    }


def _throughput_metrics(avg: float) -> dict:
    """Build metrics block with output_token_throughput."""
    return {"output_token_throughput": {"avg": avg}}


def _latency_metrics(p95: float) -> dict:
    """Build metrics block with time_to_first_token."""
    return {"time_to_first_token": {"p95": p95}}


def _combined_metrics(throughput_avg: float, ttft_p95: float) -> dict:
    return {
        "output_token_throughput": {"avg": throughput_avg},
        "time_to_first_token": {"p95": ttft_p95},
    }


class TestEvaluateTiersOnGrid:
    """Core functionality of multi-tier grid evaluation."""

    def test_grid_scrubs_non_finite_observed_in_binding_constraint(self) -> None:
        """A non-finite metric value at a grid point must not leak into
        ``binding_constraint['observed']`` (which is serialized to
        search_history.json). NaN comparisons fail the filter, so the value
        reaches the breach dict and must be scrubbed to None."""
        tier = _make_tier(
            "fast",
            [
                SLAFilter(
                    metric_tag="output_token_throughput",
                    stat="avg",
                    op="gt",
                    threshold=100.0,
                )
            ],
        )
        rows = [_make_row(8, _throughput_metrics(float("nan")))]
        results = evaluate_tiers_on_grid(rows, [tier], "phases.profiling.concurrency")
        assert len(results) == 1
        binding = results[0].binding_constraint
        assert binding is not None
        assert binding["observed"] is None

    def test_grid_keeps_finite_observed_in_binding_constraint(self) -> None:
        """A finite (but failing) metric value is preserved in the breach dict."""
        tier = _make_tier(
            "fast",
            [
                SLAFilter(
                    metric_tag="output_token_throughput",
                    stat="avg",
                    op="gt",
                    threshold=100.0,
                )
            ],
        )
        rows = [_make_row(8, _throughput_metrics(50.0))]
        results = evaluate_tiers_on_grid(rows, [tier], "phases.profiling.concurrency")
        binding = results[0].binding_constraint
        assert binding is not None
        assert binding["observed"] == 50.0

    def test_basic_two_tier_boundary_detection(self) -> None:
        """Two tiers with different thresholds find different boundaries."""
        strict = _make_tier(
            "fast",
            [
                SLAFilter(
                    metric_tag="output_token_throughput",
                    stat="avg",
                    op="gt",
                    threshold=300.0,
                )
            ],
        )
        lenient = _make_tier(
            "economy",
            [
                SLAFilter(
                    metric_tag="output_token_throughput",
                    stat="avg",
                    op="gt",
                    threshold=100.0,
                )
            ],
        )
        rows = [
            _make_row(8, _throughput_metrics(500.0)),
            _make_row(16, _throughput_metrics(350.0)),
            _make_row(32, _throughput_metrics(200.0)),
            _make_row(64, _throughput_metrics(80.0)),
        ]

        results = evaluate_tiers_on_grid(
            rows, [strict, lenient], "phases.profiling.concurrency"
        )

        assert len(results) == 2
        assert results[0].label == "fast"
        assert results[0].boundary_concurrency == 16
        assert results[1].label == "economy"
        assert results[1].boundary_concurrency == 32

    def test_all_pass_no_failure(self) -> None:
        """When all grid points pass, boundary is the maximum grid value."""
        tier = _make_tier(
            "lenient",
            [
                SLAFilter(
                    metric_tag="output_token_throughput",
                    stat="avg",
                    op="gt",
                    threshold=10.0,
                )
            ],
        )
        rows = [
            _make_row(8, _throughput_metrics(500.0)),
            _make_row(64, _throughput_metrics(200.0)),
            _make_row(128, _throughput_metrics(100.0)),
        ]

        results = evaluate_tiers_on_grid(rows, [tier], "phases.profiling.concurrency")

        assert results[0].boundary_concurrency == 128
        assert results[0].convergence_status == "converged"
        assert results[0].bracket_upper is None

    def test_no_pass_at_any_point(self) -> None:
        """When no grid points pass, boundary is None with no_pass_in_range status."""
        tier = _make_tier(
            "impossible",
            [
                SLAFilter(
                    metric_tag="output_token_throughput",
                    stat="avg",
                    op="gt",
                    threshold=9999.0,
                )
            ],
        )
        rows = [
            _make_row(8, _throughput_metrics(500.0)),
            _make_row(64, _throughput_metrics(200.0)),
        ]

        results = evaluate_tiers_on_grid(rows, [tier], "phases.profiling.concurrency")

        assert results[0].boundary_concurrency is None
        assert results[0].convergence_status == "no_pass_in_range"
        assert results[0].bracket_lower is None
        assert results[0].bracket_upper == 8

    def test_empty_grid_returns_partial(self) -> None:
        """Empty grid results in partial status."""
        tier = _make_tier(
            "t1",
            [
                SLAFilter(
                    metric_tag="output_token_throughput",
                    stat="avg",
                    op="gt",
                    threshold=100.0,
                )
            ],
        )

        results = evaluate_tiers_on_grid([], [tier], "phases.profiling.concurrency")

        assert results[0].boundary_concurrency is None
        assert results[0].convergence_status == "partial"
        assert results[0].probe_count == 0

    def test_binding_constraint_captured(self) -> None:
        """The first failing filter at the boundary is captured."""
        tier = _make_tier(
            "strict",
            [
                SLAFilter(
                    metric_tag="output_token_throughput",
                    stat="avg",
                    op="gt",
                    threshold=300.0,
                ),
                SLAFilter(
                    metric_tag="time_to_first_token",
                    stat="p95",
                    op="lt",
                    threshold=5000.0,
                ),
            ],
        )
        rows = [
            _make_row(8, _combined_metrics(500.0, 1000.0)),
            _make_row(16, _combined_metrics(250.0, 2000.0)),
        ]

        results = evaluate_tiers_on_grid(rows, [tier], "phases.profiling.concurrency")

        assert results[0].boundary_concurrency == 8
        assert results[0].binding_constraint is not None
        assert results[0].binding_constraint["metric_tag"] == "output_token_throughput"
        assert results[0].binding_constraint["observed"] == 250.0

    def test_missing_metric_treated_as_failure(self) -> None:
        """A missing metric in the grid row causes the tier to fail at that point."""
        tier = _make_tier(
            "t1",
            [
                SLAFilter(
                    metric_tag="nonexistent_metric",
                    stat="avg",
                    op="gt",
                    threshold=100.0,
                )
            ],
        )
        rows = [
            _make_row(8, _throughput_metrics(500.0)),
            _make_row(16, _throughput_metrics(300.0)),
        ]

        results = evaluate_tiers_on_grid(rows, [tier], "phases.profiling.concurrency")

        assert results[0].boundary_concurrency is None
        assert results[0].convergence_status == "no_pass_in_range"
        assert results[0].binding_constraint is not None
        assert results[0].binding_constraint["observed"] is None

    def test_leaf_param_key_fallback(self) -> None:
        """When rows use leaf key instead of full path, still works."""
        tier = _make_tier(
            "t1",
            [
                SLAFilter(
                    metric_tag="output_token_throughput",
                    stat="avg",
                    op="gt",
                    threshold=100.0,
                )
            ],
        )
        rows = [
            {"parameters": {"concurrency": 8}, "metrics": _throughput_metrics(500.0)},
            {"parameters": {"concurrency": 16}, "metrics": _throughput_metrics(50.0)},
        ]

        results = evaluate_tiers_on_grid(rows, [tier], "phases.profiling.concurrency")

        assert results[0].boundary_concurrency == 8

    def test_probe_count_equals_grid_points(self) -> None:
        """Each tier's probe_count reflects total grid points evaluated."""
        tier = _make_tier(
            "t1",
            [
                SLAFilter(
                    metric_tag="output_token_throughput",
                    stat="avg",
                    op="gt",
                    threshold=100.0,
                )
            ],
        )
        rows = [_make_row(i, _throughput_metrics(500.0 - i * 10)) for i in range(1, 9)]

        results = evaluate_tiers_on_grid(rows, [tier], "phases.profiling.concurrency")

        assert results[0].probe_count == 8

    def test_filters_echoed_in_result(self) -> None:
        """The tier's filters are echoed in the TierResult."""
        tier = _make_tier(
            "t1",
            [
                SLAFilter(
                    metric_tag="output_token_throughput",
                    stat="avg",
                    op="gt",
                    threshold=100.0,
                ),
                SLAFilter(
                    metric_tag="time_to_first_token",
                    stat="p95",
                    op="lt",
                    threshold=5000.0,
                ),
            ],
        )
        rows = [_make_row(8, _combined_metrics(500.0, 1000.0))]

        results = evaluate_tiers_on_grid(rows, [tier], "phases.profiling.concurrency")

        assert len(results[0].filters) == 2
        assert results[0].filters[0]["metric_tag"] == "output_token_throughput"
        assert results[0].filters[1]["metric_tag"] == "time_to_first_token"

    def test_non_monotonic_grid_finds_max_passing(self) -> None:
        """Non-monotonic results still find the max passing point."""
        tier = _make_tier(
            "t1",
            [
                SLAFilter(
                    metric_tag="output_token_throughput",
                    stat="avg",
                    op="gt",
                    threshold=100.0,
                )
            ],
        )
        # Non-monotonic: passes at 8, fails at 16, passes at 32
        rows = [
            _make_row(8, _throughput_metrics(200.0)),
            _make_row(16, _throughput_metrics(50.0)),
            _make_row(32, _throughput_metrics(150.0)),
        ]

        results = evaluate_tiers_on_grid(rows, [tier], "phases.profiling.concurrency")

        # Max passing is 32 (highest value where tier passes)
        assert results[0].boundary_concurrency == 32

    def test_three_tiers_ordered_boundaries(self) -> None:
        """Three tiers produce correctly ordered boundaries (strict < lenient)."""
        fast = _make_tier(
            "fast",
            [
                SLAFilter(
                    metric_tag="output_token_throughput",
                    stat="avg",
                    op="gt",
                    threshold=300.0,
                )
            ],
        )
        standard = _make_tier(
            "standard",
            [
                SLAFilter(
                    metric_tag="output_token_throughput",
                    stat="avg",
                    op="gt",
                    threshold=100.0,
                )
            ],
        )
        economy = _make_tier(
            "economy",
            [
                SLAFilter(
                    metric_tag="output_token_throughput",
                    stat="avg",
                    op="gt",
                    threshold=30.0,
                )
            ],
        )
        # Throughput decreases as concurrency increases
        rows = [
            _make_row(8, _throughput_metrics(500.0)),
            _make_row(16, _throughput_metrics(350.0)),
            _make_row(32, _throughput_metrics(200.0)),
            _make_row(64, _throughput_metrics(80.0)),
            _make_row(128, _throughput_metrics(25.0)),
        ]

        results = evaluate_tiers_on_grid(
            rows, [fast, standard, economy], "phases.profiling.concurrency"
        )

        assert results[0].boundary_concurrency == 16  # fast: >300 passes at 16
        assert results[1].boundary_concurrency == 32  # standard: >100 passes at 32
        assert results[2].boundary_concurrency == 64  # economy: >30 passes at 64
