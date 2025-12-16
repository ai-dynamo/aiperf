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
    SERVER_METRICS_JSON = "server_metrics_json"
    SERVER_METRICS_CSV = "server_metrics_csv"
    SERVER_METRICS_PARQUET = "server_metrics_parquet"
    TIMESLICE_JSON = "timeslice_json"
    TIMESLICE_CSV = "timeslice_csv"


class ServerMetricsFormat(CaseInsensitiveStrEnum):
    """Format options for server metrics export.

    Controls which output files are generated for server metrics data.
    Default selection is JSON + CSV (JSONL excluded to avoid large files).
    """

    JSON = "json"
    """Export aggregated statistics in JSON hybrid format with metrics keyed by name.
    Best for: Programmatic access, CI/CD pipelines, automated analysis."""

    CSV = "csv"
    """Export aggregated statistics in CSV tabular format organized by metric type.
    Best for: Spreadsheet analysis, Excel/Google Sheets, pandas DataFrames."""

    JSONL = "jsonl"
    """Export raw time-series records in line-delimited JSON format.
    Best for: Time-series analysis, debugging, visualizing metric evolution.
    Warning: Can generate very large files for long-running benchmarks."""

    PARQUET = "parquet"
    """Export raw time-series data with delta calculations in Parquet columnar format.
    Best for: Analytics with DuckDB/pandas/Polars, efficient storage, SQL queries.
    Includes cumulative deltas from reference point for counters and histograms."""


class ExportLevel(CaseInsensitiveStrEnum):
    """Export level for benchmark data."""

    SUMMARY = "summary"
    """Export only aggregated/summarized metrics (default, most compact)"""

    RECORDS = "records"
    """Export per-record metrics after aggregation with display unit conversion"""

    RAW = "raw"
    """Export raw parsed records with full request/response data (most detailed)"""
