# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums.base_enums import CaseInsensitiveStrEnum


class ConsoleExporterType(CaseInsensitiveStrEnum):
    API_ERRORS = "api_errors"
    ERRORS = "errors"
    EXPERIMENTAL_METRICS = "experimental_metrics"
    INTERNAL_METRICS = "internal_metrics"
    METRICS = "metrics"
    TELEMETRY = "telemetry"
    USAGE_DISCREPANCY_WARNING = "usage_discrepancy_warning"


class DataExporterType(CaseInsensitiveStrEnum):
    JSON = "json"
    CSV = "csv"
    RAW_RECORD_AGGREGATOR = "raw_record_aggregator"
    TIMESLICE_JSON = "timeslice_json"
    TIMESLICE_CSV = "timeslice_csv"


class ExportLevel(CaseInsensitiveStrEnum):
    """Export level for benchmark data."""

    SUMMARY = "summary"
    """Export only aggregated/summarized metrics (default, most compact)"""

    RECORDS = "records"
    """Export per-record metrics after aggregation with display unit conversion"""

    RAW = "raw"
    """Export raw parsed records with full request/response data (most detailed)"""
