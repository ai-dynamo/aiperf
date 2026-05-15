# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``ParetoSweepExport`` post-process handler for the ``pareto-sweep`` recipe.

Re-exported from :mod:`aiperf.search_recipes.post_process` -- consumers should
import :class:`ParetoSweepExport` from there to match the other built-in handlers.
"""

from __future__ import annotations

from typing import Any, ClassVar

from aiperf.search_recipes._pareto_dominance import (
    mark_pareto_optimal,
    quarantine_non_finite,
)
from aiperf.search_recipes._post_process_shared import _stat_or_raise


def _row_to_cell(
    row: dict[str, Any],
    *,
    x_flat: str,
    x_metric: str,
    y_flat: str,
    y_metric: str,
    isl_key: str,
    osl_key: str,
    conc_key: str,
) -> dict[str, Any] | None:
    """Project one ``per_combination_metrics`` row into a Pareto cell dict.

    Returns ``None`` when the row is missing any of the expected swept
    parameters or metric blocks; the caller filters these out. Splitting the
    per-row projection out of :meth:`ParetoSweepExport.process` keeps the
    method's branch count below the C901=10 complexity ceiling.
    """
    row_params = row.get("parameters") or {}
    metrics = row.get("metrics") or {}
    x_block = metrics.get(x_flat) or metrics.get(x_metric)
    y_block = metrics.get(y_flat) or metrics.get(y_metric)
    if x_block is None or y_block is None:
        return None
    if "mean" not in x_block or "mean" not in y_block:
        return None
    if (
        isl_key not in row_params
        or osl_key not in row_params
        or conc_key not in row_params
    ):
        return None
    return {
        "isl": int(row_params[isl_key]),
        "osl": int(row_params[osl_key]),
        "concurrency": int(row_params[conc_key]),
        "x": float(x_block["mean"]),
        "y": float(y_block["mean"]),
        "pareto_optimal": False,  # filled in by the dominance pass
    }


class ParetoSweepExport:
    """Emit a Pareto-frontier JSON artifact for the pareto-sweep recipe.

    Reads ``per_combination_metrics`` rows produced by ``SweepAnalyzer.compute``,
    pulls the (x, y) pair from each row, and marks each cell with a
    ``pareto_optimal`` flag. A cell is pareto-optimal iff no other cell has
    x <= this cell's x AND y >= this cell's y with strict improvement in at
    least one axis (x is "lower is better", y is "higher is better").

    Required ``params`` keys: ``x_metric``, ``x_stat``, ``y_metric``, ``y_stat``,
    ``isl_key``, ``osl_key``, ``concurrency_key`` (flat keys into
    ``row['parameters']``).

    Example output:
        {
            "x_metric": "time_to_first_token", "x_stat": "p95",
            "y_metric": "output_token_throughput", "y_stat": "avg",
            "cells": [
                {"isl": 128, "osl": 128, "concurrency": 1,
                 "x": 10.0, "y": 50.0, "pareto_optimal": True},
                ...
            ],
        }
    """

    name: ClassVar[str] = "pareto_sweep_export"
    description: ClassVar[str] = (
        "Emit a Pareto-frontier JSON artifact for the pareto-sweep recipe."
    )

    def process(
        self,
        sweep_aggregate: dict[str, Any],
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Return Pareto cells annotated with ``pareto_optimal`` flags.

        Reads axis metric/stat and parameter-key names from ``params``. Non-finite
        axis values are kept in the output but marked non-optimal before dominance is
        computed over the finite subset.
        """
        x_metric = str(params["x_metric"])
        x_stat = _stat_or_raise(params["x_stat"], handler="pareto_sweep_export")
        y_metric = str(params["y_metric"])
        y_stat = _stat_or_raise(params["y_stat"], handler="pareto_sweep_export")
        isl_key = str(params["isl_key"])
        osl_key = str(params["osl_key"])
        conc_key = str(params["concurrency_key"])

        x_flat = f"{x_metric}_{x_stat}"
        y_flat = f"{y_metric}_{y_stat}"

        rows = sweep_aggregate.get("per_combination_metrics") or []
        cells: list[dict[str, Any]] = []
        for row in rows:
            cell = _row_to_cell(
                row,
                x_flat=x_flat,
                x_metric=x_metric,
                y_flat=y_flat,
                y_metric=y_metric,
                isl_key=isl_key,
                osl_key=osl_key,
                conc_key=conc_key,
            )
            if cell is not None:
                cells.append(cell)
        if not cells:
            raise ValueError(
                "pareto_sweep_export: no rows with the expected (isl, osl, "
                f"concurrency) parameters and metrics ({x_flat}, {y_flat}); "
                "check that the recipe ran end-to-end with --streaming."
            )

        finite_cells = quarantine_non_finite(cells, x_label=x_flat, y_label=y_flat)
        mark_pareto_optimal(finite_cells)

        return {
            "x_metric": x_metric,
            "x_stat": x_stat,
            "y_metric": y_metric,
            "y_stat": y_stat,
            "cells": cells,
        }
