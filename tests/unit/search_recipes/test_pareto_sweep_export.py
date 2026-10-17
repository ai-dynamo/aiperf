# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest


def _params() -> dict[str, Any]:
    return {
        "x_metric": "request_latency",
        "x_stat": "p95",
        "y_metric": "output_token_throughput",
        "y_stat": "avg",
        "isl_key": "isl",
        "osl_key": "osl",
        "concurrency_key": "concurrency",
    }


def _row(
    isl: int, osl: int, conc: int, *, latency_p95: float, throughput: float
) -> dict[str, Any]:
    return {
        "parameters": {"isl": isl, "osl": osl, "concurrency": conc},
        "metrics": {
            "request_latency_p95": {"mean": latency_p95},
            "output_token_throughput_avg": {"mean": throughput},
        },
    }


def test_pareto_sweep_export_emits_one_cell_per_row() -> None:
    from aiperf.search_recipes.post_process import ParetoSweepExport

    agg = {
        "per_combination_metrics": [
            _row(128, 128, 1, latency_p95=10, throughput=50),
            _row(128, 128, 4, latency_p95=12, throughput=180),
            _row(256, 256, 1, latency_p95=15, throughput=40),
        ]
    }
    out = ParetoSweepExport().process(agg, _params())
    assert len(out["cells"]) == 3


def test_pareto_sweep_export_marks_pareto_optimal() -> None:
    """A cell is pareto-optimal iff no other cell has both lower latency AND higher throughput."""
    from aiperf.search_recipes.post_process import ParetoSweepExport

    agg = {
        "per_combination_metrics": [
            _row(
                128, 128, 1, latency_p95=10, throughput=50
            ),  # pareto-optimal (lowest latency)
            _row(
                128, 128, 4, latency_p95=12, throughput=200
            ),  # pareto-optimal (highest throughput)
            _row(256, 256, 1, latency_p95=15, throughput=30),  # dominated by row 0
        ]
    }
    out = ParetoSweepExport().process(agg, _params())
    optimal = [c for c in out["cells"] if c["pareto_optimal"]]
    assert len(optimal) == 2
    pairs = {(c["isl"], c["osl"], c["concurrency"]) for c in optimal}
    assert pairs == {(128, 128, 1), (128, 128, 4)}


def test_pareto_sweep_export_carries_axes() -> None:
    from aiperf.search_recipes.post_process import ParetoSweepExport

    agg = {
        "per_combination_metrics": [
            _row(128, 128, 1, latency_p95=10, throughput=50),
            _row(128, 128, 4, latency_p95=12, throughput=200),
        ]
    }
    out = ParetoSweepExport().process(agg, _params())
    assert out["x_metric"] == "request_latency"
    assert out["y_metric"] == "output_token_throughput"


def test_pareto_sweep_export_raises_on_empty() -> None:
    from aiperf.search_recipes.post_process import ParetoSweepExport

    with pytest.raises(ValueError, match="no rows"):
        ParetoSweepExport().process({"per_combination_metrics": []}, _params())


def test_pareto_sweep_export_reads_the_requested_stat_from_a_single_trial_row() -> None:
    """A single trial keeps each metric's stats in one block, whose mean is the
    average. The frontier has to be drawn on the p95 the recipe asked for."""
    from aiperf.search_recipes.post_process import ParetoSweepExport

    def row(conc: int, avg: float, p95: float) -> dict[str, Any]:
        return {
            "parameters": {"isl": 128, "osl": 128, "concurrency": conc},
            "metrics": {
                "request_latency": {"mean": avg, "avg": avg, "p95": p95},
                "output_token_throughput": {"mean": 100.0, "avg": 100.0},
            },
        }

    agg = {
        "per_combination_metrics": [
            row(1, avg=5.0, p95=40.0),
            row(2, avg=10.0, p95=20.0),
        ]
    }
    cells = ParetoSweepExport().process(agg, _params())["cells"]
    assert [(c["concurrency"], c["x"], c["pareto_optimal"]) for c in cells] == [
        (1, 40.0, False),
        (2, 20.0, True),
    ]
