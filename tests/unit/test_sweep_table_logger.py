# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for SweepTableLogger and its suppress predicate."""

from __future__ import annotations

import logging
import math
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from pytest import param

from aiperf.cli_runner._sweep_table import (
    _format_metric_value,
    _recompute_pareto_marks,
    _should_emit_sweep_table,
)
from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.plugin.enums import UIType
from aiperf.search_recipes._pareto_axes import ParetoAxesSpec


def _make_plan(
    n_variations: int = 3, ui_type: UIType = UIType.SIMPLE
) -> SimpleNamespace:
    """Build a minimal plan stub for predicate tests."""
    return SimpleNamespace(
        variations=[
            SimpleNamespace(values={"concurrency": i}) for i in range(n_variations)
        ],
        configs=[SimpleNamespace(ui_type=ui_type)],
    )


@pytest.mark.parametrize(
    "n_variations, ui_type, no_flag, isatty, expected",
    [
        param(3, UIType.SIMPLE, False, True, True, id="all-clear-emits"),
        param(3, UIType.SIMPLE, True, True, False, id="flag-suppresses"),
        param(3, UIType.DASHBOARD, False, True, False, id="dashboard-suppresses"),
        param(3, UIType.SIMPLE, False, False, False, id="non-tty-suppresses"),
        param(1, UIType.SIMPLE, False, True, False, id="single-cell-suppresses"),
        param(0, UIType.SIMPLE, False, True, False, id="zero-cell-suppresses"),
    ],
)  # fmt: skip
def test_should_emit_sweep_table(
    n_variations: int,
    ui_type: UIType,
    no_flag: bool,
    isatty: bool,
    expected: bool,
) -> None:
    plan = _make_plan(n_variations=n_variations, ui_type=ui_type)
    with patch("aiperf.cli_runner._sweep_table.sys.stdout") as fake_stdout:
        fake_stdout.isatty.return_value = isatty
        result = _should_emit_sweep_table(plan, no_sweep_table=no_flag)
    assert result is expected


@pytest.mark.parametrize(
    "stats, metric, stat, expected",
    [
        param({"thru": {"avg": 1402.1}}, "thru", "avg", "1402.10", id="finite-float"),
        param({"thru": {"avg": 1402}}, "thru", "avg", "1402.00", id="finite-int"),
        param({"thru": {"avg": None}}, "thru", "avg", "", id="none-empty"),
        param({"thru": {"avg": math.nan}}, "thru", "avg", "", id="nan-empty"),
        param({"thru": {"avg": math.inf}}, "thru", "avg", "", id="pos-inf-empty"),
        param({"thru": {"avg": -math.inf}}, "thru", "avg", "", id="neg-inf-empty"),
        param({}, "thru", "avg", "", id="missing-metric-empty"),
        param({"thru": {}}, "thru", "avg", "", id="missing-stat-empty"),
    ],
)  # fmt: skip
def test_format_metric_value(
    stats: dict[str, dict[str, Any]],
    metric: str,
    stat: str,
    expected: str,
) -> None:
    assert _format_metric_value(stats, metric, stat) == expected


def test_recompute_pareto_marks_minimize_x_maximize_y() -> None:
    """Lower x is better, higher y is better. Frontier = (1, 5) and (2, 6)."""
    axes = ParetoAxesSpec(
        x_metric="ttft",
        x_stat="p99",
        x_minimize=True,
        y_metric="thru",
        y_stat="avg",
        y_maximize=True,
    )
    rows = [
        {"x": 1.0, "y": 5.0, "pareto_optimal": False},  # frontier
        {"x": 3.0, "y": 4.0, "pareto_optimal": False},  # dominated by (1, 5)
        {"x": 2.0, "y": 6.0, "pareto_optimal": False},  # frontier
        {"x": 4.0, "y": 5.0, "pareto_optimal": False},  # dominated by (2, 6)
    ]
    _recompute_pareto_marks(rows, axes)
    assert [r["pareto_optimal"] for r in rows] == [True, False, True, False]


def test_recompute_pareto_marks_minimize_both() -> None:
    """Both axes minimized. Only the per-axis minima can be on the frontier."""
    axes = ParetoAxesSpec(
        x_metric="ttft",
        x_stat="p99",
        x_minimize=True,
        y_metric="latency",
        y_stat="p95",
        y_maximize=False,
    )
    rows = [
        {"x": 1.0, "y": 10.0, "pareto_optimal": False},  # frontier (best x)
        {"x": 5.0, "y": 2.0, "pareto_optimal": False},  # frontier (best y)
        {"x": 3.0, "y": 5.0, "pareto_optimal": False},  # frontier (middle trade-off)
        {"x": 4.0, "y": 8.0, "pareto_optimal": False},  # dominated by (3, 5)
    ]
    _recompute_pareto_marks(rows, axes)
    assert [r["pareto_optimal"] for r in rows] == [True, True, True, False]


def test_recompute_pareto_marks_handles_missing_values() -> None:
    """Rows with None x or y are never marked optimal."""
    axes = ParetoAxesSpec(
        x_metric="ttft",
        x_stat="p99",
        x_minimize=True,
        y_metric="thru",
        y_stat="avg",
        y_maximize=True,
    )
    rows = [
        {"x": None, "y": 5.0, "pareto_optimal": True},  # gets cleared
        {"x": 1.0, "y": 5.0, "pareto_optimal": False},  # frontier
    ]
    _recompute_pareto_marks(rows, axes)
    assert [r["pareto_optimal"] for r in rows] == [False, True]


def test_recompute_pareto_marks_empty() -> None:
    axes = ParetoAxesSpec(
        x_metric="ttft",
        x_stat="p99",
        y_metric="thru",
        y_stat="avg",
    )
    rows: list[dict[str, Any]] = []
    _recompute_pareto_marks(rows, axes)  # no error
    assert rows == []


def _make_plan_for_logger(
    *,
    variations: list[dict[str, Any]] | None = None,
) -> SimpleNamespace:
    """Plan stub with the attributes ``SweepTableLogger.__init__`` reads.

    Param names are derived from the union of per-variation ``values``
    keys (mirrors ``cli_runner._guard_against_in_process_sweep``). Tests
    that need a non-None ``pareto_axes`` patch
    ``aiperf.cli_runner._pareto._resolve_pareto_axes`` directly.
    """
    if variations is None:
        variations = [{"concurrency": 1}, {"concurrency": 2}]
    return SimpleNamespace(
        variations=[
            SimpleNamespace(values=dict(v), index=i, label=f"v{i}")
            for i, v in enumerate(variations)
        ],
        sweep=SimpleNamespace(recipe_name=None, search_recipe=None),
        confidence_level=0.95,
    )


def test_logger_init_captures_param_names_no_pareto() -> None:
    from aiperf.cli_runner._sweep_table import SweepTableLogger

    plan = _make_plan_for_logger(
        variations=[
            {"concurrency": 1, "max_tokens": 32},
            {"concurrency": 2, "max_tokens": 64},
        ],
    )
    logger = AIPerfLogger("test_sweep_table_logger")
    table_logger = SweepTableLogger(plan, logger)
    assert table_logger._param_names == ["concurrency", "max_tokens"]
    assert table_logger._pareto_axes is None
    assert table_logger._rows == []


def test_logger_init_captures_pareto_axes() -> None:
    from aiperf.cli_runner._sweep_table import SweepTableLogger

    axes = ParetoAxesSpec(
        x_metric="ttft",
        x_stat="p99",
        x_minimize=True,
        y_metric="thru",
        y_stat="avg",
        y_maximize=True,
    )
    plan = _make_plan_for_logger()
    logger = AIPerfLogger("test_sweep_table_logger")
    with patch("aiperf.cli_runner._pareto._resolve_pareto_axes", return_value=axes):
        table_logger = SweepTableLogger(plan, logger)
    assert table_logger._pareto_axes is axes


def test_build_row_finite_values_no_pareto() -> None:
    from aiperf.cli_runner._sweep_table import SweepTableLogger

    plan = _make_plan_for_logger(
        variations=[{"concurrency": 1}, {"concurrency": 2}],
    )
    table_logger = SweepTableLogger(plan, AIPerfLogger("t"))
    stats = {
        "output_token_throughput": {"avg": 1402.1},
        "time_to_first_token": {"p99": 128.6},
        "inter_token_latency": {"p99": 21.8},
        "request_latency": {"p95": 410.5},
    }
    row = table_logger._build_row(
        params={"concurrency": 32},
        stats=stats,
        trials=3,
        pareto_optimal=False,
    )
    assert row == ["32", "1402.10", "128.60", "21.80", "410.50", "3"]


def test_build_row_with_pareto_marker() -> None:
    from aiperf.cli_runner._sweep_table import SweepTableLogger

    axes = ParetoAxesSpec(
        x_metric="time_to_first_token",
        x_stat="p99",
        x_minimize=True,
        y_metric="output_token_throughput",
        y_stat="avg",
        y_maximize=True,
    )
    plan = _make_plan_for_logger()
    with patch("aiperf.cli_runner._pareto._resolve_pareto_axes", return_value=axes):
        table_logger = SweepTableLogger(plan, AIPerfLogger("t"))
    stats = {
        "output_token_throughput": {"avg": 1402.1},
        "time_to_first_token": {"p99": 128.6},
        "inter_token_latency": {"p99": 21.8},
        "request_latency": {"p95": 410.5},
    }
    row = table_logger._build_row(
        params={"concurrency": 32},
        stats=stats,
        trials=3,
        pareto_optimal=True,
    )
    assert row[-1] == "*"
    assert row[-2] == "3"  # trials column still present


def test_build_row_handles_missing_metrics() -> None:
    from aiperf.cli_runner._sweep_table import SweepTableLogger

    plan = _make_plan_for_logger(
        variations=[{"concurrency": 1}, {"concurrency": 2}],
    )
    table_logger = SweepTableLogger(plan, AIPerfLogger("t"))
    row = table_logger._build_row(
        params={"concurrency": 32},
        stats={},
        trials=1,
        pareto_optimal=False,
    )
    assert row == ["32", "", "", "", "", "1"]


def _make_run_result_stub(metric_value: float) -> SimpleNamespace:
    """Minimal stub that the patched _aggregate_cell_stats can consume."""
    return SimpleNamespace(metric_value=metric_value)


def test_call_emits_table_to_logger(caplog: pytest.LogCaptureFixture) -> None:
    """Three sequential cell_callback fires produce three info-level log records,
    each containing all prior rows in the rendered table.
    """
    from aiperf.cli_runner._sweep_table import SweepTableLogger

    plan = _make_plan_for_logger()
    logger = AIPerfLogger("aiperf.cli_runner._sweep_table.test_emit")
    table_logger = SweepTableLogger(plan, logger)

    fake_stats = {
        "output_token_throughput": {"avg": 100.0},
        "time_to_first_token": {"p99": 10.0},
        "inter_token_latency": {"p99": 1.0},
        "request_latency": {"p95": 50.0},
    }
    with (
        patch.object(
            SweepTableLogger, "_aggregate_cell_stats", return_value=fake_stats
        ),
        caplog.at_level(logging.INFO, logger=logger.logger_name),
    ):
        for c in (8, 16, 32):
            cell = {
                "params": {"concurrency": c},
                "x": None,
                "y": None,
                "pareto_optimal": False,
                "_cell_results": [_make_run_result_stub(100.0)],
            }
            variation_key = ("", (("concurrency", c),))
            table_logger(variation_key, cell)

    info_records = [r for r in caplog.records if r.levelno == logging.INFO]
    assert len(info_records) == 3
    final_block = info_records[-1].getMessage()
    assert "concurrency" in final_block
    for c in ("8", "16", "32"):
        assert c in final_block
    assert final_block.count("100.00") >= 3


def test_call_with_pareto_axes_marks_frontier() -> None:
    from aiperf.cli_runner._sweep_table import SweepTableLogger

    axes = ParetoAxesSpec(
        x_metric="time_to_first_token",
        x_stat="p99",
        x_minimize=True,
        y_metric="output_token_throughput",
        y_stat="avg",
        y_maximize=True,
    )
    plan = _make_plan_for_logger()
    logger = AIPerfLogger("aiperf.cli_runner._sweep_table.test_pareto")
    with patch("aiperf.cli_runner._pareto._resolve_pareto_axes", return_value=axes):
        table_logger = SweepTableLogger(plan, logger)

    cells = [
        {"params": {"concurrency": 8}, "ttft": 100.0, "thru": 50.0},  # dominated
        {"params": {"concurrency": 16}, "ttft": 80.0, "thru": 80.0},  # frontier
        {"params": {"concurrency": 32}, "ttft": 60.0, "thru": 60.0},  # frontier
    ]

    def _fake_stats(self: SweepTableLogger, cell_results: list[Any]) -> dict[str, Any]:
        c = cell_results[0]
        return {
            "time_to_first_token": {"p99": c.ttft},
            "output_token_throughput": {"avg": c.thru},
            "inter_token_latency": {"p99": 0.0},
            "request_latency": {"p95": 0.0},
        }

    with patch.object(SweepTableLogger, "_aggregate_cell_stats", _fake_stats):
        for cell in cells:
            run_stub = SimpleNamespace(ttft=cell["ttft"], thru=cell["thru"])
            internal_cell = {
                "params": cell["params"],
                "x": cell["ttft"],
                "y": cell["thru"],
                "pareto_optimal": False,
                "_cell_results": [run_stub],
            }
            variation_key = ("", tuple(sorted(cell["params"].items())))
            table_logger(variation_key, internal_cell)

    assert [r["pareto_optimal"] for r in table_logger._rows] == [False, True, True]
