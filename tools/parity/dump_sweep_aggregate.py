# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capture the byte-exact single-trial sweep-aggregate JSON + CSV.

Drives the production ``aiperf.cli_runner._sweep_aggregate`` +
``aiperf.orchestrator.aggregation.sweep`` + the aggregate sweep exporters over a
list of synthetic per-cell ``native-v2.json`` reports (one trial per cell). The
native Rust sweep aggregate (``rust/cli/src/sweep/aggregate.rs``) must reproduce
both artifacts byte-for-byte.

Only the single-trial path is exercised here: it is scipy-free (each group has
one result, so ``_json_metric_to_stats`` reads the cell summary directly with no
``ConfidenceAggregation``/``scipy.stats.t.ppf`` CI math). The multi-trial
confidence path is intentionally out of scope — its confidence intervals depend
on scipy's t-distribution inverse CDF, which is not bit-reproducible in a clean
Rust port.

Usage: python tools/parity/dump_sweep_aggregate.py <spec.json>

The spec is ``{"cells": [{"variation": {"index", "label", "values"},
"report": {<native-v2 dict>}}]}``. Emits ``{"json": <sweep json str>,
"csv": <sweep csv str>}``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import orjson


def main() -> int:
    if len(sys.argv) != 2:
        sys.stderr.write("usage: dump_sweep_aggregate.py <spec.json>\n")
        return 2

    spec = json.loads(Path(sys.argv[1]).read_text())

    from aiperf.cli_runner._sweep_aggregate import (
        _build_per_combination_stats,
        _build_sweep_aggregate_result,
        _compute_sweep_parameters,
        _group_results_by_variation,
    )
    from aiperf.exporters.aggregate import (
        AggregateExporterConfig,
        AggregateSweepCsvExporter,
        AggregateSweepJsonExporter,
    )
    from aiperf.orchestrator.aggregation.sweep import SweepAnalyzer
    from aiperf.orchestrator.models import RunResult
    from aiperf.orchestrator.native_report import project_native_summary

    confidence_level = 0.95
    results: list[RunResult] = []
    for cell in spec["cells"]:
        variation = cell["variation"]
        summary = project_native_summary(cell["report"])
        results.append(
            RunResult(
                label=variation["label"],
                success=True,
                summary_metrics=summary,
                variation_label=variation["label"],
                variation_values=variation.get("values", {}),
                variation_index=variation["index"],
                trial_index=0,
            )
        )

    groups = _group_results_by_variation(results)
    sweep_parameters = _compute_sweep_parameters(groups)
    per_combination_stats = _build_per_combination_stats(groups, confidence_level)
    sweep_dict = SweepAnalyzer.compute(per_combination_stats, sweep_parameters, sla_filters=None)
    sweep_dict.setdefault("metadata", {})
    sweep_dict["metadata"]["sweep_mode"] = "repeated"
    sweep_dict["metadata"]["confidence_level"] = confidence_level
    sweep_dict["metadata"]["num_trials_per_value"] = max(
        (len(g) for g in groups.values()), default=0
    )

    aggregate_result = _build_sweep_aggregate_result(results, sweep_dict)
    cfg = AggregateExporterConfig(result=aggregate_result, output_dir=Path("/tmp"))
    json_content = AggregateSweepJsonExporter(cfg)._generate_content()
    csv_content = AggregateSweepCsvExporter(cfg)._generate_content()

    sys.stdout.buffer.write(
        orjson.dumps(
            {"json": json_content, "csv": csv_content},
            option=orjson.OPT_INDENT_2,
        )
    )
    sys.stdout.buffer.write(b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
