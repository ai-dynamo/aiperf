# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pydantic response models for the operator results/analytics HTTP API."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from aiperf.common.finite import FiniteFloat
from aiperf.common.models import AIPerfBaseModel


class JobEntry(AIPerfBaseModel):
    """Summary of a stored benchmark job."""

    namespace: str = Field(description="Kubernetes namespace")
    job_id: str = Field(description="Job identifier")
    file_count: int = Field(ge=0, description="Number of stored result files")
    total_size_bytes: int = Field(
        ge=0, description="Total size of stored files in bytes"
    )
    model: str | None = Field(
        default=None,
        description="Model name extracted from the CR spec at run time. "
        "``None`` for jobs whose ``job_spec.json`` is missing or unreadable. "
        "Used by the UI's `?cluster=<ns> · <model>` deep-link from the "
        "job-detail similar-runs chip.",
    )
    endpoint: str | None = Field(
        default=None,
        description="Endpoint URL extracted from the CR spec at run time. "
        "``None`` when the spec file is missing or doesn't carry one.",
    )


class ResultsHistoryListResponse(AIPerfBaseModel):
    """Response for listing all jobs with stored results."""

    jobs: list[JobEntry] = Field(
        default_factory=list, description="Available benchmark results"
    )


class FileEntry(AIPerfBaseModel):
    """Metadata for a stored result file."""

    name: str = Field(description="Display filename (without .zst suffix)")
    stored_name: str = Field(description="Actual filename on disk")
    size_bytes: int = Field(ge=0, description="File size on disk in bytes")
    compressed: bool = Field(description="Whether the file is stored as zstd")
    mtime_epoch: int = Field(ge=0, description="File modification-time epoch seconds")


class FileListResponse(AIPerfBaseModel):
    """Response for listing files in a job's results directory."""

    namespace: str = Field(description="Kubernetes namespace")
    job_id: str = Field(description="Job identifier")
    ready: bool = Field(
        default=True,
        description="Whether final top-level result files are ready to download.",
    )
    summary_available: bool = Field(
        default=False,
        description="Whether this run has a JSON summary available through the "
        "profile-export alias.",
    )
    per_record_filename: str | None = Field(
        default=None,
        description="Configured per-record JSONL artifact name when stored for this run.",
    )
    server_metrics_filename: str | None = Field(
        default=None,
        description="Configured server-metrics JSON artifact name when stored for this run.",
    )
    files: list[FileEntry] = Field(
        default_factory=list, description="Available result files"
    )


class RunHistoryEntry(AIPerfBaseModel):
    """One historical run directory under ``<ns>/<name>/``."""

    epoch: str = Field(description="Epoch-seconds key of this run.")
    mtime_epoch: int = Field(ge=0, description="Directory modification-time epoch.")
    file_count: int = Field(ge=0, description="Number of files in the run dir.")
    total_size_bytes: int = Field(ge=0, description="Total bytes across all files.")
    is_latest: bool = Field(description="True when this run matches latest.txt.")


class RunHistoryListResponse(AIPerfBaseModel):
    """Response listing every run dir for one ``<ns>/<name>`` job."""

    namespace: str = Field(description="Kubernetes namespace.")
    job_id: str = Field(description="AIPerfJob name.")
    latest_epoch: str | None = Field(
        default=None,
        description="Current latest.txt target, or None if no runs exist.",
    )
    runs: list[RunHistoryEntry] = Field(
        default_factory=list,
        description="Historical runs, newest first.",
    )


class LeaderboardEntry(AIPerfBaseModel):
    """A single row in a leaderboard ranking."""

    namespace: str = Field(description="Kubernetes namespace")
    job_id: str = Field(description="Job identifier")
    epoch: str | None = Field(
        default=None,
        description="Run epoch (decimal seconds) the row was sourced from.",
    )
    value: FiniteFloat | None = Field(description="Metric value")
    unit: str | None = Field(description="Metric unit")
    start_time: str | None = Field(description="Benchmark start time (ISO)")
    end_time: str | None = Field(description="Benchmark end time (ISO)")
    model: str | None = Field(description="Model name")
    endpoint: str | None = Field(description="Endpoint URL")


class LeaderboardResponse(AIPerfBaseModel):
    """Ranked benchmark results for a metric."""

    metric: str = Field(description="Metric name")
    stat: str = Field(description="Statistic used for ranking")
    order: str = Field(description="Sort order (asc or desc)")
    entries: list[LeaderboardEntry] = Field(
        default_factory=list, description="Ranked entries"
    )


class HistoryEntry(AIPerfBaseModel):
    """A single data point in a time-series history."""

    namespace: str = Field(description="Kubernetes namespace")
    job_id: str = Field(description="Job identifier")
    epoch: str | None = Field(
        default=None,
        description="Run epoch (decimal seconds) the row was sourced from.",
    )
    value: FiniteFloat | None = Field(description="Metric value")
    unit: str | None = Field(description="Metric unit")
    start_time: str | None = Field(description="Benchmark start time (ISO)")
    model: str | None = Field(description="Model name")
    endpoint: str | None = Field(description="Endpoint URL")


class HistoryResponse(AIPerfBaseModel):
    """Metric values over time."""

    metric: str = Field(description="Metric name")
    stat: str = Field(description="Statistic tracked")
    entries: list[HistoryEntry] = Field(
        default_factory=list, description="Time-ordered entries"
    )


class ScatterEntry(AIPerfBaseModel):
    """One row in the scatter dataset — all four dashboard metrics for a single run."""

    namespace: str = Field(description="Kubernetes namespace")
    job_id: str = Field(description="Job identifier")
    epoch: str | None = Field(
        default=None,
        description="Run epoch (decimal seconds).",
    )
    model: str | None = Field(default=None, description="Model name")
    request_throughput_avg: FiniteFloat | None = Field(
        default=None, description="Average request throughput (req/s)"
    )
    request_latency_p99: FiniteFloat | None = Field(
        default=None, description="P99 request latency (ms)"
    )
    time_to_first_token_avg: FiniteFloat | None = Field(
        default=None, description="Average time to first token (ms)"
    )
    output_token_throughput_avg: FiniteFloat | None = Field(
        default=None, description="Average output token throughput (tok/s)"
    )


class ScatterResponse(AIPerfBaseModel):
    """All scatter entries for the dashboard chart."""

    entries: list[ScatterEntry] = Field(
        default_factory=list, description="Scatter entries, newest epoch first"
    )


class CompareResponse(AIPerfBaseModel):
    """Side-by-side comparison of specific jobs."""

    job_ids: list[str] = Field(description="Compared job IDs")
    metrics: list[str] = Field(description="Compared metrics")
    entries: list[dict[str, Any]] = Field(
        default_factory=list, description="Per-job metric values"
    )
    meta: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Per-job context keyed by '<namespace>/<job_id>'. "
        "Each value carries gpu_count, gpu_name, model, endpoint — used by "
        "the UI to normalize throughput per GPU and color points by "
        "accelerator (InferenceX-style correlation).",
    )
