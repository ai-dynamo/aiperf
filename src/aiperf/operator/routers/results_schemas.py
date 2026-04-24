# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pydantic response models for the operator results/analytics HTTP API."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from aiperf.common.models import AIPerfBaseModel


class JobEntry(AIPerfBaseModel):
    """Summary of a stored benchmark job."""

    namespace: str = Field(description="Kubernetes namespace")
    job_id: str = Field(description="Job identifier")
    file_count: int = Field(description="Number of stored result files")
    total_size_bytes: int = Field(description="Total size of stored files in bytes")


class ResultsHistoryListResponse(AIPerfBaseModel):
    """Response for listing all jobs with stored results."""

    jobs: list[JobEntry] = Field(
        default_factory=list, description="Available benchmark results"
    )


class FileEntry(AIPerfBaseModel):
    """Metadata for a stored result file."""

    name: str = Field(description="Display filename (without .zst suffix)")
    stored_name: str = Field(description="Actual filename on disk")
    size_bytes: int = Field(description="File size on disk in bytes")
    compressed: bool = Field(description="Whether the file is stored as zstd")


class FileListResponse(AIPerfBaseModel):
    """Response for listing files in a job's results directory."""

    namespace: str = Field(description="Kubernetes namespace")
    job_id: str = Field(description="Job identifier")
    files: list[FileEntry] = Field(
        default_factory=list, description="Available result files"
    )


class LeaderboardEntry(AIPerfBaseModel):
    """A single row in a leaderboard ranking."""

    namespace: str = Field(description="Kubernetes namespace")
    job_id: str = Field(description="Job identifier")
    value: float | None = Field(description="Metric value")
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
    value: float | None = Field(description="Metric value")
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


class CompareResponse(AIPerfBaseModel):
    """Side-by-side comparison of specific jobs."""

    job_ids: list[str] = Field(description="Compared job IDs")
    metrics: list[str] = Field(description="Compared metrics")
    entries: list[dict[str, Any]] = Field(
        default_factory=list, description="Per-job metric values"
    )
