# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CSV exporter for sweep aggregate results."""

from __future__ import annotations

import csv
import io
from typing import Any

from aiperf.exporters.aggregate.aggregate_base_exporter import AggregateBaseExporter


class AggregateSweepCsvExporter(AggregateBaseExporter):
    """Exports sweep-level summary to CSV: one row per variation with key metrics."""

    def __init__(self, config, sweep_dict: dict[str, Any], **kwargs) -> None:
        super().__init__(config, **kwargs)
        self._sweep_dict = sweep_dict

    def get_file_name(self) -> str:
        return "profile_export_aiperf_sweep.csv"

    def _generate_content(self) -> str:
        combos = self._sweep_dict.get("per_combination_metrics", [])
        if not combos:
            return ""

        param_names, metric_names = _collect_column_names(combos)
        header = _build_header(param_names, metric_names)

        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(header)
        for combo in combos:
            writer.writerow(_build_row(combo, param_names, metric_names))

        return buf.getvalue()


def _collect_column_names(
    combos: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    param_names: list[str] = []
    metric_names: list[str] = []
    for combo in combos:
        for k in combo.get("parameters", {}):
            if k not in param_names:
                param_names.append(k)
        for k in combo.get("metrics", {}):
            if k not in metric_names:
                metric_names.append(k)
    return param_names, metric_names


def _build_header(param_names: list[str], metric_names: list[str]) -> list[str]:
    header = list(param_names)
    for m in metric_names:
        header.extend([f"{m}_mean", f"{m}_std", f"{m}_ci_low", f"{m}_ci_high"])
    return header


def _build_row(
    combo: dict[str, Any],
    param_names: list[str],
    metric_names: list[str],
) -> list[Any]:
    params = combo.get("parameters", {})
    metrics = combo.get("metrics", {})
    row: list[Any] = [params.get(p, "") for p in param_names]
    for m in metric_names:
        stats = metrics.get(m, {})
        row.append(_to_native(stats.get("mean", "")))
        row.append(_to_native(stats.get("std", "")))
        row.append(_to_native(stats.get("ci_low", "")))
        row.append(_to_native(stats.get("ci_high", "")))
    return row


def _to_native(val: Any) -> Any:
    """Convert numpy scalars to Python native types for CSV."""
    if hasattr(val, "item"):
        return val.item()
    return val
