# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the shared Pareto dominance helpers."""

import math
import warnings

from aiperf.search_recipes._pareto_dominance import (
    mark_pareto_optimal,
    quarantine_non_finite,
)


def _cell(x: float, y: float, **extra) -> dict:
    return {"x": x, "y": y, "pareto_optimal": False, **extra}


def test_mark_pareto_optimal_default_lower_x_higher_y():
    cells = [_cell(10.0, 50.0), _cell(20.0, 40.0), _cell(15.0, 55.0)]
    mark_pareto_optimal(cells)
    assert cells[0]["pareto_optimal"] is True
    assert cells[1]["pareto_optimal"] is False
    assert cells[2]["pareto_optimal"] is True


def test_mark_pareto_optimal_invertable_directions():
    # Higher x AND higher y both dominant.
    cells = [_cell(1.0, 1.0), _cell(2.0, 2.0), _cell(2.0, 1.5)]
    mark_pareto_optimal(cells, x_minimize=False, y_maximize=True)
    assert cells[0]["pareto_optimal"] is False
    assert cells[1]["pareto_optimal"] is True
    assert cells[2]["pareto_optimal"] is False


def test_quarantine_non_finite_excludes_nan_inf():
    cells = [
        _cell(10.0, 50.0),
        _cell(float("nan"), 100.0),
        _cell(20.0, float("inf")),
    ]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        finite = quarantine_non_finite(cells, x_label="latency", y_label="throughput")
    assert len(finite) == 1
    assert math.isfinite(finite[0]["x"]) and math.isfinite(finite[0]["y"])
    assert any(issubclass(item.category, UserWarning) for item in w)
    # The two excluded cells are flagged in-place as not pareto-optimal.
    assert cells[1]["pareto_optimal"] is False
    assert cells[2]["pareto_optimal"] is False


def test_quarantine_non_finite_no_warning_when_all_finite():
    cells = [_cell(1.0, 1.0), _cell(2.0, 2.0)]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        quarantine_non_finite(cells, x_label="x", y_label="y")
    assert all(not issubclass(item.category, UserWarning) for item in w)


def test_existing_pareto_sweep_export_still_works():
    """Sanity: the lifted helpers still drive the existing exporter behavior."""
    from aiperf.search_recipes._pareto_sweep_export import ParetoSweepExport

    rows = [
        {
            "parameters": {"isl": 128, "osl": 128, "concurrency": 1},
            "metrics": {
                "request_latency_p95": {"mean": 10.0},
                "output_token_throughput_avg": {"mean": 50.0},
            },
        },
        {
            "parameters": {"isl": 128, "osl": 128, "concurrency": 4},
            "metrics": {
                "request_latency_p95": {"mean": 20.0},
                "output_token_throughput_avg": {"mean": 40.0},
            },
        },
    ]
    out = ParetoSweepExport().process(
        {"per_combination_metrics": rows},
        {
            "x_metric": "request_latency",
            "x_stat": "p95",
            "y_metric": "output_token_throughput",
            "y_stat": "avg",
            "isl_key": "isl",
            "osl_key": "osl",
            "concurrency_key": "concurrency",
        },
    )
    flagged = [c for c in out["cells"] if c["pareto_optimal"]]
    assert len(flagged) == 1
    assert flagged[0]["concurrency"] == 1
