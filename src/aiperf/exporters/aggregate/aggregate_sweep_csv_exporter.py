# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CSV exporter for sweep aggregate results.

Multi-section CSV layout (matches PR #699 schema, byte-compatible with
main's :class:`AggregateSweepCsvExporter`). Reads sweep sections directly
from the enclosing :class:`~aiperf.orchestrator.aggregation.base.AggregateResult`.
"""

from __future__ import annotations

import csv
import io
from typing import Any, Protocol

from aiperf.exporters.aggregate.aggregate_base_exporter import AggregateBaseExporter


class _CsvWriter(Protocol):
    """Structural type for ``csv.writer``: only ``writerow`` is used here."""

    def writerow(self, row: list[Any]) -> Any: ...


class AggregateSweepCsvExporter(AggregateBaseExporter):
    """Exports sweep aggregate results to a multi-section CSV file.

    Layout (blank-line separated):

    1. Per-combination metrics table -- one row per parameter combination
       with ``mean``/``std``/``min``/``max``/``cv`` columns per metric.
    2. Best configurations -- one row per objective
       (``best_throughput``, ``best_latency_p99``).
    3. Pareto optimal points -- one row per non-dominated combination.
    4. Metadata -- aggregation type, sweep parameters, combination count,
       profile-run counts (matches PR #699 schema).

    Constructor surface matches sibling aggregate exporters
    (``AggregateConfidenceCsvExporter`` etc.): just ``(config, **kwargs)``.
    Sweep sections are read from ``self._result.metadata`` (sweep_parameters,
    num_combinations, best_configurations, pareto_optimal) and from
    ``self._result.metrics`` (per_combination_metrics).

    Example:
        >>> result = AggregateResult(
        ...     aggregation_type="sweep",
        ...     num_runs=2,
        ...     num_successful_runs=2,
        ...     metadata={
        ...         "sweep_parameters": [{"name": "concurrency", "values": [10, 20]}],
        ...         "num_combinations": 2,
        ...         "best_configurations": {},
        ...         "pareto_optimal": [],
        ...     },
        ...     metrics=[
        ...         {"parameters": {"concurrency": 10},
        ...          "metrics": {"request_throughput_avg": {"mean": 100.0}}},
        ...     ],
        ... )  # doctest: +SKIP
        >>> exp = AggregateSweepCsvExporter(AggregateExporterConfig(result, dir))  # doctest: +SKIP
        >>> await exp.export()  # doctest: +SKIP
    """

    def get_file_name(self) -> str:
        """Return ``"profile_export_aiperf_sweep.csv"``."""
        return "profile_export_aiperf_sweep.csv"

    @staticmethod
    def _format_number(value: float | int | None, decimals: int = 2) -> str:
        """Format a number for CSV output.

        NaN, +inf, -inf, and None all render as empty string so non-finite
        values land as blank cells in the CSV (mirrors the JSON ``null``
        treatment via ``scrub_non_finite``). Numpy scalars are unwrapped
        via ``.item()``.
        """
        return _format_number(value, decimals)

    def _generate_content(self) -> str:
        """Generate the multi-section CSV content."""
        buf = io.StringIO()
        writer = csv.writer(buf)

        sweep_parameters = self._result.metadata.get("sweep_parameters", [])
        param_names = [p["name"] for p in sweep_parameters]
        per_combination_metrics = self._result.metrics or []

        _write_per_combination_section(writer, per_combination_metrics, param_names)

        writer.writerow([])
        _write_best_configurations_section(
            writer,
            self._result.metadata.get("best_configurations", {}),
            param_names,
        )

        writer.writerow([])
        _write_pareto_section(
            writer,
            self._result.metadata.get("pareto_optimal", []),
            param_names,
        )

        writer.writerow([])
        _write_metadata_section(
            writer, self._result.metadata, param_names, self._result
        )

        return buf.getvalue()


def _write_per_combination_section(
    writer: _CsvWriter,
    per_combination_metrics: list[dict[str, Any]],
    param_names: list[str],
) -> None:
    """Section 1: param-cols + per-metric ``mean/std/min/max/cv`` columns.

    Header is the UNION of metric keys across every combo, not just the
    first one's keys — sweeps frequently produce combos where later cells
    surface metrics absent from the first cell (e.g. accuracy metrics
    that only appear in evaluation-mode rows).
    """
    if not per_combination_metrics:
        return

    seen: set[str] = set()
    for combo_entry in per_combination_metrics:
        for k in combo_entry.get("metrics", {}).keys():
            seen.add(k)
    metric_names = sorted(seen)
    header = list(param_names)
    for metric_name in metric_names:
        header.extend(
            [
                f"{metric_name}_mean",
                f"{metric_name}_std",
                f"{metric_name}_min",
                f"{metric_name}_max",
                f"{metric_name}_cv",
            ]
        )
    writer.writerow(header)

    for combo_entry in per_combination_metrics:
        parameters = combo_entry.get("parameters", {})
        metrics = combo_entry.get("metrics", {})
        row: list[Any] = [parameters.get(name, "") for name in param_names]
        for metric_name in metric_names:
            metric_data = metrics.get(metric_name, {})
            if isinstance(metric_data, dict):
                row.extend(
                    [
                        _format_number(metric_data.get("mean")),
                        _format_number(metric_data.get("std")),
                        _format_number(metric_data.get("min")),
                        _format_number(metric_data.get("max")),
                        _format_number(metric_data.get("cv"), decimals=4),
                    ]
                )
            else:
                row.extend(["", "", "", "", ""])
        writer.writerow(row)


def _write_best_configurations_section(
    writer: _CsvWriter, best_configs: dict[str, Any], param_names: list[str]
) -> None:
    """Section 2: one row per objective with parameter values + metric/unit."""
    writer.writerow(["Best Configurations"])
    if not best_configs:
        return
    writer.writerow(["Configuration", *param_names, "Metric", "Unit"])
    for config_name, config_data in best_configs.items():
        formatted = config_name.replace("_", " ").title()
        parameters = config_data.get("parameters", {})
        row = [formatted]
        row.extend(parameters.get(name, "") for name in param_names)
        row.extend(
            [
                _format_number(config_data.get("metric")),
                config_data.get("unit", ""),
            ]
        )
        writer.writerow(row)


def _write_pareto_section(
    writer: _CsvWriter, pareto_optimal: list[dict[str, Any]], param_names: list[str]
) -> None:
    """Section 3: one row per non-dominated parameter combination."""
    writer.writerow(["Pareto Optimal Points"])
    if not pareto_optimal:
        writer.writerow(["None"])
        return
    writer.writerow(param_names)
    for combo_params in pareto_optimal:
        writer.writerow([combo_params.get(name, "") for name in param_names])


def _write_metadata_section(
    writer: _CsvWriter,
    metadata: dict[str, Any],
    param_names: list[str],
    result: Any,
) -> None:
    """Section 4: aggregation type, sweep parameters, run counts.

    Schema mirrors PR #699: includes ``Aggregation Type``,
    ``Number of Profile Runs``, and ``Number of Successful Runs`` so the
    CSV stays byte-compatible with main's :class:`AggregateSweepCsvExporter`.
    """
    writer.writerow(["Metadata"])
    writer.writerow(["Field", "Value"])
    writer.writerow(["Aggregation Type", result.aggregation_type])
    writer.writerow(["Sweep Parameters", ", ".join(param_names)])
    writer.writerow(["Number of Combinations", metadata.get("num_combinations", 0)])
    writer.writerow(["Number of Profile Runs", result.num_runs])
    writer.writerow(["Number of Successful Runs", result.num_successful_runs])


def _format_number(value: float | int | None, decimals: int = 2) -> str:
    """Format a number for CSV output; numpy scalars unwrapped via ``.item()``.

    Non-finite values (NaN, +inf, -inf) and None all render as empty string
    so CSV cells stay blank rather than carrying spec-incompatible literals.
    """
    import math

    if value is None:
        return ""
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.{decimals}f}"
    return str(value)
