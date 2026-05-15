# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-cell streaming sweep table emitted via the AIPerf logger.

Consumes the ``MultiRunOrchestrator.cell_callback`` extension point. On each
fire, re-renders the full accumulated table as a self-contained log block;
the most recent block is always authoritative.
"""

from __future__ import annotations

import math
import sys
from io import StringIO
from typing import TYPE_CHECKING, Any

from rich.box import SIMPLE_HEAVY
from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from aiperf.common.aiperf_logger import AIPerfLogger
    from aiperf.config.resolution.plan import BenchmarkPlan
    from aiperf.search_recipes._pareto_axes import ParetoAxesSpec


def _should_emit_sweep_table(
    plan: BenchmarkPlan,
    *,
    no_sweep_table: bool,
) -> bool:
    """Return True iff the streaming sweep table should be wired up.

    Suppresses when any of: the explicit flag is set, stdout is not a
    TTY (CI logs / pipes / redirects), the DASHBOARD UI is active (it
    handles its own rendering and a stdout-bound table block would
    corrupt it), or there are fewer than two variations to compare.
    """
    from aiperf.plugin.enums import UIType

    if no_sweep_table:
        return False
    if not sys.stdout.isatty():
        return False
    first_config = plan.configs[0] if plan.configs else None
    if first_config is not None and first_config.ui_type == UIType.DASHBOARD:
        return False
    return len(plan.variations) > 1


def _format_metric_value(
    stats: dict[str, Any],
    metric: str,
    stat: str,
    decimals: int = 2,
) -> str:
    """Format ``stats[metric][stat]`` for table display.

    Empty string for missing keys, ``None``, NaN, and +/-inf — same
    missing-value convention as ``AggregateSweepCsvExporter._format_number``
    so on-disk and on-screen metric absence agree.

    Deliberately diverges from ``_format_number`` for integer values: the
    CSV exporter renders ``1402`` as ``"1402"`` (preserving native type),
    while this formatter renders it as ``"1402.00"`` so a table column
    has uniform decimal width across rows.
    """
    metric_data = stats.get(metric)
    if not isinstance(metric_data, dict):
        return ""
    value = metric_data.get(stat)
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    if isinstance(value, (int, float)):
        return f"{value:.{decimals}f}"
    return str(value)


def _recompute_pareto_marks(
    rows: list[dict[str, Any]],
    axes: ParetoAxesSpec,
) -> None:
    """Stamp ``pareto_optimal`` on each row in-place.

    A row is on the frontier iff no other row dominates it. Row A
    dominates row B when A is no-worse-than B on both axes and strictly
    better on at least one. Rows with ``None`` x or y are never marked
    optimal — they have insufficient data to participate in dominance.
    """
    x_min = axes.x_minimize
    # Spec is asymmetric: x has x_minimize, y has y_maximize.
    # Normalize y to a "minimize" flag for the helpers below.
    y_min = not axes.y_maximize

    def _better_or_equal(a: float, b: float, minimize: bool) -> bool:
        return a <= b if minimize else a >= b

    def _strictly_better(a: float, b: float, minimize: bool) -> bool:
        return a < b if minimize else a > b

    for i, row in enumerate(rows):
        x_i, y_i = row.get("x"), row.get("y")
        if x_i is None or y_i is None:
            row["pareto_optimal"] = False
            continue
        dominated = False
        for j, other in enumerate(rows):
            if i == j:
                continue
            x_j, y_j = other.get("x"), other.get("y")
            if x_j is None or y_j is None:
                continue
            if (
                _better_or_equal(x_j, x_i, x_min)
                and _better_or_equal(y_j, y_i, y_min)
                and (
                    _strictly_better(x_j, x_i, x_min)
                    or _strictly_better(y_j, y_i, y_min)
                )
            ):
                dominated = True
                break
        row["pareto_optimal"] = not dominated


HEADLINE_METRICS: list[tuple[str, str]] = [
    ("output_token_throughput", "avg"),
    ("time_to_first_token", "p99"),
    ("inter_token_latency", "p99"),
    ("request_latency", "p95"),
]


class SweepTableLogger:
    """Per-cell streaming sweep table, emitted via the AIPerf logger.

    Designed to be used as the ``cell_callback`` argument to
    ``MultiRunOrchestrator``. On each fire, accumulates a row for the
    completed variation and emits the full accumulated table as a single
    log block. The most recent block is always authoritative; markers
    (e.g. Pareto frontier) may flip across blocks as new data arrives.

    Parameter names are derived from the union of per-variation
    ``values`` keys on ``plan.variations`` (the only place per-cell
    parameter identity actually lives — ``plan.sweep`` may be ``None``,
    a grid/zip/scenario sweep, or an adaptive-search sweep, none of
    which expose a uniform ``parameter_names`` field). Mirrors the
    derivation used by ``cli_runner._reject_in_process_sweep_under_operator``.
    """

    def __init__(
        self,
        plan: BenchmarkPlan,
        logger: AIPerfLogger,
    ) -> None:
        from aiperf.cli_runner._pareto import _resolve_pareto_axes

        self._plan = plan
        self._logger = logger
        self._param_names: list[str] = sorted(
            {
                k
                for variation in plan.variations
                if variation is not None
                for k in variation.values
            }
        )
        self._pareto_axes = _resolve_pareto_axes(plan)
        self._rows: list[dict[str, Any]] = []

    def _aggregate_cell_stats(self, cell_results: list[Any]) -> dict[str, Any]:
        """Aggregate one cell's trial results into a per-metric stats dict.

        Delegates to the canonical ``_aggregate_group_to_stats`` helper so
        the live-table view shares its source of truth with the on-disk
        sweep aggregator. Returns ``{}`` when aggregation yields ``None``
        (e.g., no usable trials).
        """
        from aiperf.cli_runner._sweep_aggregate import _aggregate_group_to_stats

        return (
            _aggregate_group_to_stats(cell_results, self._plan.confidence_level) or {}
        )

    def _build_row(
        self,
        *,
        params: dict[str, Any],
        stats: dict[str, Any],
        trials: int,
        pareto_optimal: bool,
    ) -> list[str]:
        """Build one table row as a list of formatted column strings."""
        row: list[str] = [str(params.get(name, "")) for name in self._param_names]
        for metric, stat in HEADLINE_METRICS:
            row.append(_format_metric_value(stats, metric, stat))
        row.append(str(trials))
        if self._pareto_axes is not None:
            row.append("*" if pareto_optimal else "")
        return row

    def __call__(self, variation_key: tuple, cell: dict[str, Any]) -> None:
        """Cell callback hook: append a row, recompute markers, emit table."""
        params = cell.get("params", {})
        cell_results = cell.get("_cell_results", []) or []
        stats = self._aggregate_cell_stats(cell_results)
        x = cell.get("x")
        y = cell.get("y")
        self._rows.append(
            {
                "params": dict(params),
                "x": x,
                "y": y,
                "trials": len(cell_results),
                "stats": stats,
                "pareto_optimal": False,
            }
        )
        if self._pareto_axes is not None:
            _recompute_pareto_marks(self._rows, self._pareto_axes)
        rendered = self._render()
        self._logger.info(f"\n{rendered}\n")

    def _render(self) -> str:
        """Render the current accumulated table to a string."""
        table = Table(box=SIMPLE_HEAVY, show_header=True)
        for name in self._param_names:
            table.add_column(name, justify="right")
        for metric, _stat in HEADLINE_METRICS:
            table.add_column(metric, justify="right")
        table.add_column("trials", justify="right")
        if self._pareto_axes is not None:
            table.add_column("pareto", justify="center")

        for row in self._rows:
            cells = self._build_row(
                params=row["params"],
                stats=row["stats"],
                trials=row["trials"],
                pareto_optimal=row["pareto_optimal"],
            )
            table.add_row(*cells)

        buf = StringIO()
        Console(file=buf, force_terminal=False, width=200).print(table)
        return buf.getvalue().rstrip("\n")
