# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pydantic models describing a loaded AIPerf profiling run."""

from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import Field

from aiperf.common.models import AIPerfBaseModel
from aiperf.common.models.record_models import MetricResult


class RunMetadata(AIPerfBaseModel):
    """Metadata for a single profiling run."""

    run_name: str = Field(description="Name of the run (typically directory name)")
    run_path: Path = Field(description="Path to the run directory")
    model: str | None = Field(default=None, description="Model name used in the run")
    concurrency: int | None = Field(default=None, description="Concurrency level used")
    request_count: int | None = Field(
        default=None, description="Total number of requests"
    )
    duration_seconds: float | None = Field(
        default=None, description="Duration of the run in seconds"
    )
    endpoint_type: str | None = Field(
        default=None, description="Type of endpoint (e.g., 'chat', 'completions')"
    )
    start_time: str | None = Field(
        default=None, description="ISO timestamp when the profiling run started"
    )
    end_time: str | None = Field(
        default=None, description="ISO timestamp when the profiling run ended"
    )
    was_cancelled: bool = Field(
        default=False, description="Whether the profiling run was cancelled early"
    )
    experiment_type: str = Field(
        default="treatment",
        description="Classification of run as 'baseline' or 'treatment' for visualization",
    )
    experiment_group: str = Field(
        default="",
        description="Experiment group identifier extracted from run name or path for grouping variants",
    )


class RunData(AIPerfBaseModel):
    """Complete data for a single profiling run."""

    model_config = {"arbitrary_types_allowed": True}

    metadata: RunMetadata = Field(description="Metadata for the run")
    requests: pd.DataFrame | None = Field(
        description="DataFrame containing per-request data, or None if not loaded"
    )
    aggregated: dict[str, Any] = Field(
        description="Dictionary containing aggregated statistics. The 'metrics' key "
        "contains a dict mapping metric tags to MetricResult objects"
    )
    timeslices: pd.DataFrame | None = Field(
        default=None,
        description="DataFrame containing timeslice data in tidy format with columns: "
        "[Timeslice, Metric, Unit, Stat, Value], or None if not loaded",
    )
    slice_duration: float | None = Field(
        default=None,
        description="Duration of each time slice in seconds, or None if not available",
    )
    gpu_telemetry: pd.DataFrame | None = Field(
        default=None,
        description="DataFrame containing GPU telemetry time series data, or None if not loaded",
    )
    server_metrics: pd.DataFrame | None = Field(
        default=None,
        description="DataFrame containing server metrics time series data in tidy format with columns: "
        "[timestamp_ns, endpoint_url, metric_name, metric_type, value, histogram_count, histogram_sum, "
        "labels_json, unit], or None if not loaded",
    )
    server_metrics_aggregated: dict[str, Any] = Field(
        default_factory=dict,
        description="Dictionary containing aggregated server metrics statistics by metric name. "
        "Structure: {metric_name: {endpoint_url: {labels_key: {type, stats, unit, description, timeslices}}}}",
    )

    def get_metric(self, metric_name: str) -> MetricResult | dict[str, Any] | None:
        """Get a metric from aggregated data."""
        if not self.aggregated:
            return None

        if "metrics" in self.aggregated:
            return self.aggregated["metrics"].get(metric_name)

        return self.aggregated.get(metric_name)
