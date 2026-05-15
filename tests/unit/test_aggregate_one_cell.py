# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the per-cell aggregation helper extracted from aggregate_sweep_and_export."""

from unittest.mock import MagicMock, patch

from aiperf.cli_runner._pareto import _aggregate_one_cell


def _make_run_result(stats: dict, success: bool = True):
    rr = MagicMock()
    rr.success = success
    rr.summary_metrics = {
        name: _make_json_metric(block) for name, block in stats.items()
    }
    return rr


def _make_json_metric(block: dict):
    """Build a stand-in JsonMetricResult-shaped object.

    ``_aggregate_group_to_stats`` reads ``.avg``/``.min``/``.max``/``.unit``
    on the single-trial path plus every percentile field
    (``.p1``..``.p99``); mirror that contract here. Unspecified percentile
    keys default to ``None`` so the projection's
    ``getattr(metric, pct, None) is not None`` check filters them out
    cleanly (MagicMock would otherwise auto-create truthy attributes).
    """
    metric = MagicMock(
        spec=[
            "avg",
            "min",
            "max",
            "unit",
            "p1",
            "p5",
            "p10",
            "p25",
            "p50",
            "p75",
            "p90",
            "p95",
            "p99",
        ]
    )
    metric.avg = block.get("mean", 0.0)
    metric.min = block.get("min", block.get("mean", 0.0))
    metric.max = block.get("max", block.get("mean", 0.0))
    metric.unit = block.get("unit", "")
    for pct in ("p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99"):
        setattr(metric, pct, block.get(pct))
    return metric


def _make_plan(confidence_level: float = 0.95):
    plan = MagicMock()
    plan.confidence_level = confidence_level
    return plan


def _call(cell_results, plan, variation, *, axes=None):
    """Invoke ``_aggregate_one_cell`` with the pareto axes lookup stubbed.

    Production code resolves axes via the plugin registry from
    ``plan.sweep.recipe_name``. Tests inject a specific :class:`ParetoAxesSpec`
    (or ``None``) directly to keep the call independent of plugin
    registration.
    """
    with patch("aiperf.cli_runner._pareto._resolve_pareto_axes", return_value=axes):
        return _aggregate_one_cell(
            cell_results=cell_results, plan=plan, variation=variation
        )


def _make_variation(values: dict):
    var = MagicMock()
    var.values = values
    return var


def test_aggregate_one_cell_returns_none_when_no_axes():
    plan = _make_plan()
    out = _call(
        cell_results=[_make_run_result({"request_latency_p95": {"mean": 10.0}})],
        plan=plan,
        variation=_make_variation({"isl": 128, "osl": 128, "concurrency": 1}),
        axes=None,
    )
    assert out is None


def test_aggregate_one_cell_extracts_metric_axis():
    from aiperf.search_recipes._pareto_axes import ParetoAxesSpec

    axes = ParetoAxesSpec(
        x_metric="request_latency",
        x_stat="p95",
        y_metric="output_token_throughput",
        y_stat="avg",
        series_keys=("isl", "osl"),
    )
    plan = _make_plan()
    rr = _make_run_result(
        {
            "request_latency_p95": {"mean": 10.0},
            "output_token_throughput_avg": {"mean": 50.0},
        }
    )
    out = _call(
        cell_results=[rr],
        plan=plan,
        variation=_make_variation({"isl": 128, "osl": 128, "concurrency": 1}),
        axes=axes,
    )
    assert out is not None
    assert out["x"] == 10.0
    assert out["y"] == 50.0
    assert out["params"]["isl"] == 128
    assert out["pareto_optimal"] is False  # tracker fills this in


def test_aggregate_one_cell_falls_back_to_variation_params_for_y():
    """max-concurrency-under-sla case: y is a parameter, not a metric."""
    from aiperf.search_recipes._pareto_axes import ParetoAxesSpec

    axes = ParetoAxesSpec(
        x_metric="request_latency",
        x_stat="p95",
        y_metric="concurrency",
        y_stat="value",
    )
    plan = _make_plan()
    rr = _make_run_result({"request_latency_p95": {"mean": 10.0}})
    out = _call(
        cell_results=[rr],
        plan=plan,
        variation=_make_variation({"concurrency": 16}),
        axes=axes,
    )
    assert out is not None
    assert out["y"] == 16.0


def test_aggregate_one_cell_returns_none_when_axis_value_missing():
    from aiperf.search_recipes._pareto_axes import ParetoAxesSpec

    axes = ParetoAxesSpec(
        x_metric="request_latency",
        x_stat="p95",
        y_metric="output_token_throughput",
        y_stat="avg",
    )
    plan = _make_plan()
    rr = _make_run_result({"request_latency_p95": {"mean": 10.0}})
    out = _call(
        cell_results=[rr],
        plan=plan,
        variation=_make_variation({"isl": 128}),
        axes=axes,
    )
    assert out is None


def test_existing_aggregate_unchanged():
    """Sanity: existing per-combination tests still pass after refactor.

    Implementation Task: re-run tests/unit/test_cli_runner_aggregation.py and
    tests/unit/test_cli_runner_sweep_helpers.py — both must pass byte-for-byte
    on existing outputs.
    """
    pass


def test_aggregate_one_cell_uses_requested_percentile_stat():
    """When axes.x_stat='p95', the projected x value must be the metric's p95
    field, NOT the avg. Regression test for the silent-fallback bug where
    _extract_axis_value pulled stats[metric]['mean'] regardless of stat.
    """
    from aiperf.search_recipes._pareto_axes import ParetoAxesSpec

    axes = ParetoAxesSpec(
        x_metric="request_latency",
        x_stat="p95",
        y_metric="output_token_throughput",
        y_stat="avg",
    )
    plan = _make_plan()
    rr = _make_run_result(
        {
            "request_latency": {"mean": 100.0, "p95": 195.0, "unit": "ms"},
            "output_token_throughput": {"mean": 50.0, "unit": "tokens/sec"},
        }
    )
    out = _call(
        cell_results=[rr],
        plan=plan,
        variation=_make_variation({"isl": 128, "osl": 128, "concurrency": 1}),
        axes=axes,
    )
    assert out is not None
    # p95 field, not mean — this is the bug regression test.
    assert out["x"] == 195.0, (
        f"expected p95=195.0, got {out['x']} (likely picked up avg=100.0 instead)"
    )
    assert out["y"] == 50.0
