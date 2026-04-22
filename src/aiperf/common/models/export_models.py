# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar

from pydantic import ConfigDict, Field

from aiperf.common.models.base_models import AIPerfBaseModel
from aiperf.common.models.error_models import ErrorDetailsCount
from aiperf.config import BenchmarkConfig

# =============================================================================
# JSON Metric Result
# =============================================================================


@dataclass(slots=True, kw_only=True)
class JsonMetricResult:
    """The result values of a single metric for JSON export.

    Slotted dataclass — shared between Pydantic parents (``JsonExportData``
    metric-result fields) via ``__pydantic_config__`` and msgspec wire
    envelopes (``TelemetryExportData.endpoints[...].gpus[...].metrics``
    carried on ``ProcessTelemetryResultMessage``).

    NOTE:
    This model mimics the GenAI-Perf JSON output. Be careful not to add or
    remove fields that are not present in the GenAI-Perf JSON output.
    """

    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    unit: str
    avg: float | None = None
    p1: float | None = None
    p5: float | None = None
    p10: float | None = None
    p25: float | None = None
    p50: float | None = None
    p75: float | None = None
    p90: float | None = None
    p95: float | None = None
    p99: float | None = None
    min: int | float | None = None
    max: int | float | None = None
    std: float | None = None


# =============================================================================
# Telemetry Export Data
# =============================================================================


@dataclass(slots=True, kw_only=True)
class TelemetrySummary:
    """Summary information for telemetry collection."""

    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    start_time: datetime
    end_time: datetime
    endpoints_configured: list[str] | None = None
    endpoints_successful: list[str] | None = None


@dataclass(slots=True, kw_only=True)
class GpuSummary:
    """Summary of GPU telemetry data."""

    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    gpu_index: int
    gpu_name: str
    gpu_uuid: str
    hostname: str | None
    metrics: dict[str, JsonMetricResult]
    namespace: str | None = None
    pod_name: str | None = None


@dataclass(slots=True, kw_only=True)
class EndpointData:
    """Data for a single endpoint."""

    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    gpus: dict[str, GpuSummary]


@dataclass(slots=True, kw_only=True)
class TelemetryExportData:
    """Telemetry data structure for JSON export."""

    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    summary: TelemetrySummary
    endpoints: dict[str, EndpointData]
    error_summary: list[ErrorDetailsCount] | None = None


# =============================================================================
# Timeslice Export Data
# =============================================================================


class TimesliceData(AIPerfBaseModel):
    """Data for a single timeslice.

    Contains metrics for one time slice with dynamic metric fields
    added via Pydantic's extra="allow" setting.
    """

    model_config = ConfigDict(extra="allow")

    timeslice_index: int


class TimesliceCollectionExportData(AIPerfBaseModel):
    """Export data for all timeslices in a single file.

    Contains an array of timeslice data objects with metadata.
    """

    timeslices: list[TimesliceData]
    input_config: BenchmarkConfig | None = None


# =============================================================================
# Main JSON Export Data
# =============================================================================


class JsonExportData(AIPerfBaseModel):
    """Summary data to be exported to a JSON file.

    NOTE:
    This model has been designed to mimic the structure of the GenAI-Perf JSON output
    as closely as possible. Be careful when modifying this model to not break the
    compatibility with the GenAI-Perf JSON output.
    """

    # NOTE: The extra="allow" setting is needed to allow additional metrics not defined in this class
    #       to be added to the export data. It is also already set in the AIPerfBaseModel,
    #       but we are setting it here to guard against base model changes.
    model_config = ConfigDict(extra="allow")

    # Increment on breaking changes to the export structure
    SCHEMA_VERSION: ClassVar[str] = "1.0"

    schema_version: str | None = Field(
        default=None,
        description="Schema version for this export format (for backward compatibility)",
    )
    aiperf_version: str | None = Field(
        default=None,
        description="AIPerf version that generated this export (for backward compatibility)",
    )
    benchmark_id: str | None = Field(
        default=None,
        description="Unique identifier for this benchmark run (for backward compatibility)",
    )
    request_throughput: JsonMetricResult | None = None
    request_latency: JsonMetricResult | None = None
    request_count: JsonMetricResult | None = None
    time_to_first_token: JsonMetricResult | None = None
    time_to_second_token: JsonMetricResult | None = None
    inter_token_latency: JsonMetricResult | None = None
    output_token_throughput: JsonMetricResult | None = None
    output_token_throughput_per_user: JsonMetricResult | None = None
    output_sequence_length: JsonMetricResult | None = None
    input_sequence_length: JsonMetricResult | None = None
    goodput: JsonMetricResult | None = None
    good_request_count: JsonMetricResult | None = None
    output_token_count: JsonMetricResult | None = None
    reasoning_token_count: JsonMetricResult | None = None
    min_request_timestamp: JsonMetricResult | None = None
    max_response_timestamp: JsonMetricResult | None = None
    inter_chunk_latency: JsonMetricResult | None = None
    total_output_tokens: JsonMetricResult | None = None
    total_reasoning_tokens: JsonMetricResult | None = None
    benchmark_duration: JsonMetricResult | None = None
    total_isl: JsonMetricResult | None = None
    total_osl: JsonMetricResult | None = None
    error_request_count: JsonMetricResult | None = None
    error_isl: JsonMetricResult | None = None
    total_error_isl: JsonMetricResult | None = None
    telemetry_data: TelemetryExportData | None = None
    input_config: BenchmarkConfig | None = None
    was_cancelled: bool | None = None
    error_summary: list[ErrorDetailsCount] | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
