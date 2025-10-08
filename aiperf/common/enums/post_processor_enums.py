# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums.base_enums import CaseInsensitiveStrEnum


class RecordProcessorType(CaseInsensitiveStrEnum):
    """Type of streaming record processor."""

    METRIC_RECORD = "metric_record"
    """Streamer that streams records and computes metrics from MetricType.RECORD and MetricType.AGGREGATE.
    This is the first stage of the metrics processing pipeline, and is done is a distributed manner across multiple service instances."""


class ResultsProcessorType(CaseInsensitiveStrEnum):
    """Type of streaming results processor."""

    METRIC_RESULTS = "metric_results"
    """Processor that processes the metric results from METRIC_RECORD and computes metrics from MetricType.DERIVED. as well as aggregates the results.
    This is the last stage of the metrics processing pipeline, and is done by the RecordsManager after all the service instances have completed their processing."""

    TELEMETRY_RESULTS = "telemetry_results"
    """Processor that processes telemetry records from GPU monitoring and computes metrics from MetricType.TELEMETRY.
    This processes per-GPU telemetry data and aggregates it using the MetricTelemetryDict and PerGpuMetricArray classes."""
