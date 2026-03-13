# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Detailed aggregation strategy using per-request JSONL data."""

import logging
from pathlib import Path

import numpy as np
import orjson

from aiperf.orchestrator.aggregation.base import AggregateResult, AggregationStrategy
from aiperf.orchestrator.models import RunResult

logger = logging.getLogger(__name__)

JSONL_FILENAME = "profile_export.jsonl"


class DetailedAggregation(AggregationStrategy):
    """Aggregation strategy that reads per-request JSONL data and computes true combined percentiles.

    Unlike ConfidenceAggregation which operates on run-level summary stats,
    this strategy combines all per-request metric values from the profiling
    phase into a single population per metric, producing accurate distribution
    statistics (p50, p90, p95, p99) over the full request population.
    """

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
            run_metrics = self._load_all_metrics(run.artifacts_path)
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

    def _load_all_metrics(self, artifacts_path: Path) -> dict[str, list[float]]:
        """Read all per-request metric values from a run's JSONL export.

        Args:
            artifacts_path: Path to the run's artifacts directory.

        Returns:
            Dict mapping metric name to list of float values.
            Empty dict if the file is missing, empty, or unreadable.
        """
        jsonl_path = artifacts_path / JSONL_FILENAME
        if not jsonl_path.exists():
            return {}

        metrics: dict[str, list[float]] = {}
        try:
            with open(jsonl_path, "rb") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = orjson.loads(line)
                    except orjson.JSONDecodeError:
                        logger.warning(
                            "Skipping malformed JSONL line in %s", jsonl_path
                        )
                        continue

                    if not isinstance(record, dict):
                        logger.warning(
                            "Skipping non-dict JSONL record in %s", jsonl_path
                        )
                        continue

                    metadata = record.get("metadata", {})
                    if not isinstance(metadata, dict):
                        continue
                    if metadata.get("benchmark_phase") != "profiling":
                        continue
                    if record.get("error") is not None:
                        continue

                    record_metrics = record.get("metrics", {})
                    if not isinstance(record_metrics, dict):
                        continue

                    for metric_name, metric_entry in record_metrics.items():
                        value = (
                            metric_entry.get("value")
                            if isinstance(metric_entry, dict)
                            else None
                        )
                        if value is None:
                            continue
                        try:
                            float_value = float(value)
                        except (ValueError, TypeError):
                            logger.warning(
                                "Skipping non-numeric metric value for %s in %s",
                                metric_name,
                                jsonl_path,
                            )
                            continue
                        if metric_name not in metrics:
                            metrics[metric_name] = []
                        metrics[metric_name].append(float_value)
        except OSError:
            logger.exception("I/O error reading %s", jsonl_path)
            return {}

        return metrics
