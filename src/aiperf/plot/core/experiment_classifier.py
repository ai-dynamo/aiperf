# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experiment classification and GPU-telemetry helpers for the plot DataLoader.

Split out of ``data_loader.py`` purely to keep that file within the ergonomics
file-size budget. The two concerns live together because both are pure,
aggregated-dict-driven helpers the orchestrator delegates to.
"""

from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from aiperf.plot.core.plot_specs import ExperimentClassificationConfig


class ExperimentClassifierMixin:
    """Classify runs into baselines/treatments and derive experiment groups."""

    classification_config: ExperimentClassificationConfig | None

    def _classify_experiment_type(self, run_path: Path, run_name: str) -> str:
        """Classify run as 'baseline' or 'treatment' via plot_config.yaml patterns."""
        if self.classification_config:
            for pattern in self.classification_config.baselines:
                if fnmatch(run_name, pattern) or fnmatch(str(run_path), pattern):
                    return "baseline"

            for pattern in self.classification_config.treatments:
                if fnmatch(run_name, pattern) or fnmatch(str(run_path), pattern):
                    return "treatment"

            return self.classification_config.default

        return "treatment"

    def _extract_experiment_group(self, run_path: Path, run_name: str) -> str:
        """Derive experiment-group identifier from parent dir (if matched) or run name."""
        if self.classification_config:
            parent = run_path.parent
            if parent and parent.name:
                parent_name = parent.name

                for pattern in self.classification_config.baselines:
                    if fnmatch(parent_name, pattern):
                        return parent_name

                for pattern in self.classification_config.treatments:
                    if fnmatch(parent_name, pattern):
                        return parent_name

        result = run_name if run_name else str(run_path.name)

        if not result:
            self.warning(
                f"Could not extract experiment_group from {run_path}, using full path"
            )
            result = str(run_path)

        return result


class TelemetryMixin:
    """Extract telemetry data + GPU counts from aggregated stats."""

    def extract_telemetry_data(
        self, aggregated: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Extract telemetry dict ({summary, endpoints}) from aggregated stats."""
        if not aggregated or "telemetry_data" not in aggregated:
            self.debug("No telemetry data found in aggregated statistics")
            return None

        telemetry = aggregated.get("telemetry_data")
        if not isinstance(telemetry, dict):
            self.warning("Telemetry data exists but has unexpected structure")
            return None

        if "summary" not in telemetry or "endpoints" not in telemetry:
            self.warning("Telemetry data missing expected keys (summary, endpoints)")
            return None

        self.info(
            f"Extracted telemetry data with {len(telemetry.get('endpoints', {}))} endpoints"
        )
        return telemetry

    def get_telemetry_summary(
        self, aggregated: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Telemetry summary (start_time, end_time, endpoints_*) or None if absent."""
        telemetry = self.extract_telemetry_data(aggregated)
        return telemetry.get("summary") if telemetry else None

    def calculate_gpu_count_from_telemetry(
        self, aggregated: dict[str, Any]
    ) -> int | None:
        """Sum unique GPU count across all telemetry endpoints."""
        telemetry = self.extract_telemetry_data(aggregated)
        if not telemetry:
            return None

        endpoints = telemetry.get("endpoints", {})
        if not isinstance(endpoints, dict):
            self.warning("Telemetry endpoints data has unexpected structure")
            return None

        gpu_count = 0
        for _endpoint_name, endpoint_data in endpoints.items():
            if not isinstance(endpoint_data, dict):
                continue

            gpus = endpoint_data.get("gpus", {})
            if isinstance(gpus, dict):
                gpu_count += len(gpus)

        if gpu_count == 0:
            self.debug("No GPUs found in telemetry data")
            return None

        self.info(f"Calculated GPU count from telemetry: {gpu_count}")
        return gpu_count
