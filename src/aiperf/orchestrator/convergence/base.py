# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Base class for convergence criteria."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import orjson

from aiperf.orchestrator.models import RunResult

logger = logging.getLogger(__name__)

JSONL_FILENAME = "profile_export.jsonl"


class ConvergenceCriterion(ABC):
    """Abstract base for determining whether benchmark metrics have converged across runs."""

    @abstractmethod
    def is_converged(self, results: list[RunResult]) -> bool:
        """Determine whether metrics have converged across the given runs.

        Args:
            results: Results from runs executed so far.

        Returns:
            True if metrics have converged, False otherwise.
        """

    def _load_request_metrics(
        self, artifacts_path: Path, metric_name: str
    ) -> list[float]:
        """Read per-request metric values from a run's JSONL export.

        Reads profile_export.jsonl line-by-line, filters to profiling phase,
        excludes error records and records missing the target metric, and
        returns the metric values as a flat list.

        Args:
            artifacts_path: Path to the run's artifacts directory.
            metric_name: Name of the metric to extract (e.g. "time_to_first_token").

        Returns:
            List of float metric values from valid profiling-phase records.
            Empty list if the file is missing, empty, or contains no matching records.
        """
        jsonl_path = artifacts_path / JSONL_FILENAME
        if not jsonl_path.exists():
            return []

        values: list[float] = []
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
                        continue

                    metadata = record.get("metadata", {})
                    if not isinstance(metadata, dict):
                        continue
                    if metadata.get("benchmark_phase") != "profiling":
                        continue

                    if record.get("error") is not None:
                        continue

                    metrics = record.get("metrics", {})
                    if not isinstance(metrics, dict):
                        continue
                    metric_entry = metrics.get(metric_name)
                    if metric_entry is None or not isinstance(metric_entry, dict):
                        continue

                    value = metric_entry.get("value")
                    if value is None:
                        continue

                    try:
                        values.append(float(value))
                    except (ValueError, TypeError):
                        logger.warning(
                            "Skipping non-numeric value for %s in %s",
                            metric_name,
                            jsonl_path,
                        )
                        continue
        except OSError:
            logger.exception("I/O error reading %s", jsonl_path)
            return []

        return values
