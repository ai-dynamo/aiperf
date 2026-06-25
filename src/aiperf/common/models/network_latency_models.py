# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pydantic import Field

from aiperf.common.models.base_models import AIPerfBaseModel
from aiperf.common.models.error_models import ErrorDetails, ErrorDetailsCount

__all__ = [
    "NetworkLatencyResults",
    "NetworkLatencySample",
    "NetworkLatencyTargetSummary",
]


class NetworkLatencySample(AIPerfBaseModel):
    """Single TCP-handshake RTT probe sample against one endpoint target.

    Emitted once per probe (success or failure) by the probe collector and
    written verbatim to the per-sample JSONL artifact. A failed probe carries
    ``success=False``, ``rtt_ns=None`` and the captured ``error``.
    """

    timestamp_ns: int = Field(
        ge=0,
        description="Wall-clock timestamp (time.time_ns) when the probe was issued",
    )
    target_url: str = Field(
        description="Endpoint URL the target was derived from (credential-free)"
    )
    target_host: str = Field(description="Resolved/parsed host the probe connected to")
    target_port: int = Field(
        ge=1, le=65535, description="TCP port the probe connected to"
    )
    probe_type: str = Field(
        default="tcp_connect",
        description="Probe mechanism used; always 'tcp_connect' (raw TCP handshake)",
    )
    rtt_ns: int | None = Field(
        default=None,
        ge=0,
        description="Measured TCP-handshake round-trip time in nanoseconds, or None on failure",
    )
    success: bool = Field(
        description="Whether the TCP handshake completed within the connect timeout"
    )
    error: ErrorDetails | None = Field(
        default=None,
        description="Error details when the probe failed (timeout/refused), else None",
    )


class NetworkLatencyTargetSummary(AIPerfBaseModel):
    """Aggregate RTT statistics for a single probe target over the profiling phase."""

    target_url: str = Field(description="Endpoint URL the target was derived from")
    target_host: str = Field(description="Host the probes connected to")
    target_port: int = Field(
        ge=1, le=65535, description="TCP port the probes connected to"
    )
    count: int = Field(
        ge=0, description="Total number of probes issued for this target"
    )
    success_count: int = Field(ge=0, description="Number of successful probes")
    failure_count: int = Field(ge=0, description="Number of failed probes")
    min_ns: float | None = Field(
        default=None, ge=0, description="Minimum successful RTT in nanoseconds"
    )
    mean_ns: float | None = Field(
        default=None, ge=0, description="Mean successful RTT in nanoseconds"
    )
    median_ns: float | None = Field(
        default=None, ge=0, description="Median successful RTT in nanoseconds"
    )
    p90_ns: float | None = Field(
        default=None, ge=0, description="90th-percentile successful RTT in nanoseconds"
    )
    p99_ns: float | None = Field(
        default=None, ge=0, description="99th-percentile successful RTT in nanoseconds"
    )
    stddev_ns: float | None = Field(
        default=None,
        ge=0,
        description="Standard deviation of successful RTT in nanoseconds",
    )


class NetworkLatencyResults(AIPerfBaseModel):
    """Aggregate network-latency calibration results for a profile run.

    The aggregate ``mean_ns`` (mean over all successful samples across all
    targets) is the value delivered to the metric results processor for the
    ``network_adjusted_*`` metric injection.
    """

    benchmark_id: str | None = Field(
        default=None,
        description="Unique identifier for this benchmark run (UUID), shared across exports",
    )
    target_summaries: dict[str, NetworkLatencyTargetSummary] = Field(
        default_factory=dict,
        description="Per-target aggregate RTT statistics keyed by 'host:port'",
    )
    count: int = Field(
        default=0, ge=0, description="Total number of probes issued across all targets"
    )
    success_count: int = Field(
        default=0,
        ge=0,
        description="Total number of successful probes across all targets",
    )
    failure_count: int = Field(
        default=0, ge=0, description="Total number of failed probes across all targets"
    )
    min_ns: float | None = Field(
        default=None,
        ge=0,
        description="Minimum successful RTT in nanoseconds (all targets)",
    )
    mean_ns: float | None = Field(
        default=None,
        ge=0,
        description="Mean successful RTT in nanoseconds (all targets); delivered to the metric processor",
    )
    median_ns: float | None = Field(
        default=None,
        ge=0,
        description="Median successful RTT in nanoseconds (all targets)",
    )
    p90_ns: float | None = Field(
        default=None,
        ge=0,
        description="90th-percentile successful RTT in nanoseconds (all targets)",
    )
    p99_ns: float | None = Field(
        default=None,
        ge=0,
        description="99th-percentile successful RTT in nanoseconds (all targets)",
    )
    stddev_ns: float | None = Field(
        default=None,
        ge=0,
        description="Standard deviation of successful RTT in nanoseconds (all targets)",
    )
    error_summary: list[ErrorDetailsCount] = Field(
        default_factory=list,
        description="Unique probe error details and their counts",
    )
