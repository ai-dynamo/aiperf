# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums.base_enums import CaseInsensitiveStrEnum


class RecordProcessorType(CaseInsensitiveStrEnum):
    """Type of streaming record processor.

    Record processors are responsible for streaming records and computing metrics from MetricType.RECORD and MetricType.AGGREGATE.
    This is the first stage of the processing pipeline, and is done is a distributed manner across multiple service instances.
    """

    METRIC_RECORD = "metric_record"
    """Streamer that streams records and computes metrics from MetricType.RECORD and MetricType.AGGREGATE.
    This is the first stage of the metrics processing pipeline, and is done is a distributed manner across multiple service instances."""


class ResultsProcessorType(CaseInsensitiveStrEnum):
    """Type of streaming results processor.

    Results processors are responsible for processing results from RecordProcessors and computing metrics from MetricType.DERIVED.
    as well as aggregating the results.
    This is the last stage of the processing pipeline, and is done from the single instance of the RecordsManager.
    """

    METRIC_RESULTS = "metric_results"
    """Processor that processes the metric results from METRIC_RECORD and computes metrics from MetricType.DERIVED. as well as aggregates the results.
    This is the last stage of the metrics processing pipeline, and is done from the RecordsManager after all the service instances have completed their processing."""

    RECORD_EXPORT = "record_export"
    """Processor that exports per-record metrics to JSONL files with display unit conversion and filtering.
    Only enabled when export_level is set to RECORDS."""
