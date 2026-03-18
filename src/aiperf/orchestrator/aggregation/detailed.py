# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Detailed aggregation strategy using per-request JSONL data."""

import numpy as np

from aiperf.orchestrator.aggregation.base import AggregateResult, AggregationStrategy
from aiperf.orchestrator.jsonl_loader import DEFAULT_JSONL_FILENAME, load_all_metrics
from aiperf.orchestrator.models import RunResult


class DetailedAggregation(AggregationStrategy):
    """Aggregation strategy that reads per-request JSONL data and computes true combined percentiles.

    Unlike ConfidenceAggregation which operates on run-level summary stats,
    this strategy combines all per-request metric values from the profiling
    phase into a single population per metric, producing accurate distribution
    statistics (p50, p90, p95, p99) over the full request population.
    """

    def __init__(self, jsonl_filename: str = DEFAULT_JSONL_FILENAME) -> None:
        self._jsonl_filename = jsonl_filename

    def get_aggregation_type(self) -> str:
        """Return aggregation type identifier."""
        return "detailed"

    def aggregate(self, results: list[RunResult]) -> AggregateResult:
        """Aggregate per-request JSONL data from multiple runs.

        Args:
            results: List of RunResult from orchestrator.

        Returns:
            AggregateResult with combined percentiles and per-run breakdowns.
        """
        successful = [r for r in results if r.success]
        failed = [
            {"label": r.label, "error": r.error} for r in results if not r.success
        ]

        # metric_name -> list of (label, values_array) tuples
        per_run_data: dict[str, list[tuple[str, np.ndarray]]] = {}

        for run in successful:
            if run.artifacts_path is None:
                continue
            run_metrics = load_all_metrics(run.artifacts_path, self._jsonl_filename)
            if not run_metrics:
                continue
            for metric_name, values in run_metrics.items():
                if metric_name not in per_run_data:
                    per_run_data[metric_name] = []
                per_run_data[metric_name].append((run.label, np.array(values)))

        metrics: dict[str, dict] = {}
        for metric_name, run_entries in per_run_data.items():
            combined_values = np.concatenate([v for _, v in run_entries])
            if len(combined_values) == 0:
                continue

            per_run = [
                {
                    "label": label,
                    "mean": float(np.mean(vals)),
                    "count": len(vals),
                }
                for label, vals in run_entries
            ]

            metrics[metric_name] = {
                "combined": {
                    "mean": float(np.mean(combined_values)),
                    "std": float(np.std(combined_values, ddof=1))
                    if len(combined_values) > 1
                    else 0.0,
                    "p50": float(np.percentile(combined_values, 50)),
                    "p90": float(np.percentile(combined_values, 90)),
                    "p95": float(np.percentile(combined_values, 95)),
                    "p99": float(np.percentile(combined_values, 99)),
                    "count": len(combined_values),
                },
                "per_run": per_run,
            }

        return AggregateResult(
            aggregation_type="detailed",
            num_runs=len(results),
            num_successful_runs=len(successful),
            failed_runs=failed,
            metrics=metrics,
            metadata={"run_labels": [r.label for r in successful]},
        )
