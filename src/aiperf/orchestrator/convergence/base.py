# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Base class for convergence criteria."""

from abc import ABC, abstractmethod
from pathlib import Path

from aiperf.orchestrator.jsonl_loader import DEFAULT_JSONL_FILENAME, load_single_metric
from aiperf.orchestrator.models import RunResult


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
        self,
        artifacts_path: Path,
        metric_name: str,
        jsonl_filename: str = DEFAULT_JSONL_FILENAME,
    ) -> list[float]:
        """Read per-request metric values from a run's JSONL export.

        Args:
            artifacts_path: Path to the run's artifacts directory.
            metric_name: Name of the metric to extract (e.g. "time_to_first_token").
            jsonl_filename: JSONL filename within the artifacts directory.

        Returns:
            List of float metric values from valid profiling-phase records.
            Empty list if the file is missing, empty, or contains no matching records.
        """
        return load_single_metric(artifacts_path, metric_name, jsonl_filename)
