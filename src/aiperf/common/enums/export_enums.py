# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums.base_enums import CaseInsensitiveStrEnum


class ExportFormat(CaseInsensitiveStrEnum):
    """Defines the file format for record-level exports."""

    JSON = "json"
    """JSON format."""

    JSONL = "jsonl"
    """JSON Lines format (one JSON object per line)."""

    CSV = "csv"
    """Comma-separated values format."""


class ExportLevel(CaseInsensitiveStrEnum):
    """Export level for benchmark data."""

    SUMMARY = "summary"
    """Export only aggregated/summarized metrics (default, most compact)"""

    RECORDS = "records"
    """Export per-record metrics after aggregation with display unit conversion"""

    RAW = "raw"
    """Export raw parsed records with full request/response data (most detailed)"""


class ListMetricAggregationMode(CaseInsensitiveStrEnum):
    """Aggregation strategy for list-valued metrics in benchmark summaries."""

    EXACT = "exact"
    """Preserve exact list values for aggregation and summary statistics."""

    TDIGEST = "tdigest"
    """Use t-digest sketches for scalable percentile aggregation of list metrics."""


class RecordExportFormat(CaseInsensitiveStrEnum):
    """Format options for per-record metrics export.

    Controls which output files are generated for per-record benchmark data.
    Default selection is JSONL only.
    """

    CSV = "csv"
    """Export per-record metrics in CSV tabular format with flat column layout.
    Best for: Spreadsheet analysis, tabular comparison, pandas DataFrames."""

    JSONL = "jsonl"
    """Export per-record metrics in line-delimited JSON with nested metadata.
    Best for: Programmatic access, structured analysis, debugging."""


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


class SummaryFormat(CaseInsensitiveStrEnum):
    """Defines the file format for summary exports."""

    JSON = "json"
    """JSON format for summaries."""

    YAML = "yaml"
    """YAML format for summaries."""
