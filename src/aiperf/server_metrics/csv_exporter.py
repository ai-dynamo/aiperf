# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
import io
import numbers
from collections import defaultdict
from decimal import Decimal
from typing import NamedTuple

from aiperf.common.enums import PrometheusMetricType, ServerMetricsFormat
from aiperf.common.exceptions import DataExporterDisabled
from aiperf.common.models.server_metrics_models import (
    CounterSeries,
    GaugeSeries,
    HistogramSeries,
    ServerMetricsEndpointSummary,
)
from aiperf.exporters.exporter_config import ExporterConfig, FileExportInfo
from aiperf.exporters.metrics_base_exporter import MetricsBaseExporter
from aiperf.exporters.utils import normalize_endpoint_display
from aiperf.server_metrics.units import infer_unit

# Base headers at the start of each row
BASE_HEADERS_START = ["Endpoint", "Type", "Metric", "Unit"]
# Base headers at the end of each row (Description comes after dynamic label columns)
BASE_HEADERS_END = ["Description"]
# Headers for transposed info metrics section
INFO_HEADERS = ["Endpoint", "Metric", "Key", "Value", "Description"]

# Stat keys for each metric type (matches model field names)
GAUGE_STAT_KEYS = [
    "avg", "min", "max", "std", "p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99"
]  # fmt: skip
COUNTER_STAT_KEYS = [
    "total", "rate", "rate_avg", "rate_min", "rate_max", "rate_std"
]  # fmt: skip
HISTOGRAM_STAT_KEYS = [
    "count", "count_rate", "sum", "sum_rate", "avg", "p1_estimate", "p5_estimate", "p10_estimate", "p25_estimate", "p50_estimate", "p75_estimate", "p90_estimate", "p95_estimate", "p99_estimate"
]  # fmt: skip

STAT_KEYS_MAP = {
    PrometheusMetricType.GAUGE: GAUGE_STAT_KEYS,
    PrometheusMetricType.COUNTER: COUNTER_STAT_KEYS,
    PrometheusMetricType.HISTOGRAM: HISTOGRAM_STAT_KEYS,
}


__all__ = ["ServerMetricsCsvExporter"]


class CsvMetricInfo(NamedTuple):
    """Information about a metric to be exported to a CSV file.

    Args:
        endpoint: Normalized endpoint display name
        metric_name: Prometheus metric name
        description: Metric description from HELP text
        unit: Inferred unit string (e.g., "seconds", "bytes") or None
        stats: Type-specific statistics (Gauge, Counter, or Histogram series)
        is_derived: Whether this is a derived metric
    """

    endpoint: str
    metric_name: str
    description: str
    unit: str | None
    stats: GaugeSeries | CounterSeries | HistogramSeries

    @property
    def is_info_metric(self) -> bool:
        """Check if this is an info metric (should skip label expansion)."""
        return self.metric_name.endswith("_info")


class ServerMetricsCsvExporter(MetricsBaseExporter):
    """Export server metrics to a separate CSV file organized by metric type.

    Exports server metrics in sections separated by Prometheus metric type
    (gauge, counter, histogram), with appropriate columns for each type.
    """

    def __init__(self, exporter_config: ExporterConfig, **kwargs) -> None:
        """Initialize the CSV exporter for server metrics.

        Args:
            exporter_config: Configuration containing user settings and server metrics results
            **kwargs: Additional arguments passed to base class

        Raises:
            DataExporterDisabled: If server metrics are disabled or no data is available
        """
        if exporter_config.user_config.server_metrics_disabled:
            raise DataExporterDisabled("Server metrics is disabled")

        # Check if CSV format is enabled
        if (
            ServerMetricsFormat.CSV
            not in exporter_config.user_config.server_metrics_formats
        ):
            raise DataExporterDisabled(
                "Server metrics CSV export disabled: format not selected"
            )

        # Check if server metrics data is available before initializing
        if (
            not exporter_config.server_metrics_results
            or not exporter_config.server_metrics_results.endpoint_summaries
        ):
            raise DataExporterDisabled(
                "Server metrics CSV export disabled: no server metrics data available"
            )

        super().__init__(exporter_config, **kwargs)
        self._file_path = (
            exporter_config.user_config.output.server_metrics_export_csv_file
        )
        self.trace_or_debug(
            lambda: f"Initializing ServerMetricsCsvExporter with config: {exporter_config}",
            lambda: f"Initializing ServerMetricsCsvExporter with file path: {self._file_path}",
        )

    def get_export_info(self) -> FileExportInfo:
        """Return export metadata for logging and user feedback.

        Returns:
            FileExportInfo with export type description and target file path
        """
        return FileExportInfo(
            export_type="Server Metrics CSV Export",
            file_path=self._file_path,
        )

    def _generate_content(self) -> str:
        """Generate CSV content for server metrics data organized by metric type.

        Creates separate sections for each metric type (gauge, counter, histogram),
        with info metrics in their own transposed section at the end.

        Returns:
            str: CSV content with server metrics organized by type
        """
        if not self._server_metrics_results:
            return ""

        endpoint_summaries = self._server_metrics_results.endpoint_summaries
        if not endpoint_summaries:
            self.warning(
                "No pre-computed server metrics summaries available. "
                "This may indicate a ZMQ serialization issue."
            )
            return ""

        # Group metrics by type across all endpoints
        metrics_by_type = self._group_metrics_by_type(endpoint_summaries)

        buf = io.StringIO()
        writer = csv.writer(buf)

        from aiperf import __version__ as aiperf_version

        # Add metadata as comments at the top of the file
        buf.write("# AIPerf Server Metrics Export (CSV)\n")
        buf.write(f"# aiperf_version: {aiperf_version}\n")
        buf.write("# schema_version: 1.0\n")
        buf.write(f"# benchmark_id: {self._server_metrics_results.benchmark_id}\n")
        buf.write(
            "# Note: Same benchmark_id and version appear in JSON and Parquet exports\n"
        )
        buf.write("#\n")

        # Separate info metrics from gauges
        info_metrics: list[CsvMetricInfo] = []
        if PrometheusMetricType.GAUGE in metrics_by_type:
            gauge_metrics = metrics_by_type[PrometheusMetricType.GAUGE]
            info_metrics = [m for m in gauge_metrics if m.is_info_metric]
            metrics_by_type[PrometheusMetricType.GAUGE] = [
                m for m in gauge_metrics if not m.is_info_metric
            ]
            # Remove gauge section if empty after filtering
            if not metrics_by_type[PrometheusMetricType.GAUGE]:
                del metrics_by_type[PrometheusMetricType.GAUGE]

        # Write sections in order: gauge, counter, histogram
        section_order = [
            PrometheusMetricType.GAUGE,
            PrometheusMetricType.COUNTER,
            PrometheusMetricType.HISTOGRAM,
        ]
        first_section = True

        for metric_type in section_order:
            if metric_type not in metrics_by_type:
                continue

            if not first_section:
                writer.writerow([])

            self._write_section(writer, metric_type, metrics_by_type[metric_type])
            first_section = False

        # Write info metrics section at the end (transposed format)
        if info_metrics:
            if not first_section:
                writer.writerow([])
            self._write_info_section(writer, info_metrics)

        return buf.getvalue()

    def _group_metrics_by_type(
        self, endpoint_summaries: dict[str, ServerMetricsEndpointSummary]
    ) -> dict[PrometheusMetricType, list[CsvMetricInfo]]:
        """Group all metrics by their type across all endpoints.

        Args:
            endpoint_summaries: Dict mapping endpoint names to their summaries

        Returns:
            Dict mapping metric type to list of CsvMetricInfo objects.
        """
        metrics_by_type: dict[PrometheusMetricType, list[CsvMetricInfo]] = defaultdict(
            list
        )

        for endpoint_summary in endpoint_summaries.values():
            normalized_endpoint = normalize_endpoint_display(
                endpoint_summary.endpoint_url
            )

            for metric_name, metric_summary in endpoint_summary.metrics.items():
                metric_type = metric_summary.type
                unit = infer_unit(metric_name, metric_summary.description)
                unit_display_name = unit.display_name() if unit else None

                for series_item in metric_summary.series:
                    metrics_by_type[metric_type].append(
                        CsvMetricInfo(
                            endpoint=normalized_endpoint,
                            metric_name=metric_name,
                            description=metric_summary.description,
                            unit=unit_display_name,
                            stats=series_item,
                        )
                    )

        return dict(metrics_by_type)

    def _write_section(
        self,
        writer: csv.writer,
        metric_type: PrometheusMetricType,
        metrics: list[CsvMetricInfo],
    ) -> None:
        """Write a CSV section for a specific metric type with optimized layout.

        Creates a compact, readable CSV section by:
        1. Expanding labels into individual columns for easy filtering/pivoting
        2. Optimizing column order to minimize horizontal gaps (empty cells)
        3. Clustering rows by label patterns to group related metrics
        4. Including type-specific stat columns (avg/min/max for gauges, etc.)

        Label handling:
        - Non-info metrics: Labels become columns (e.g., "method", "status")
        - Info metrics: Label columns left blank (labels contain config, not dimensions)

        Layout optimizations reduce file size and improve readability by grouping
        metrics with similar label sets together, minimizing CSV sparsity.

        Args:
            writer: CSV writer to output rows to
            metric_type: Type of metrics in this section (GAUGE, COUNTER, or HISTOGRAM)
            metrics: List of CsvMetricInfo objects to export in this section
        """
        stat_keys = STAT_KEYS_MAP[metric_type]

        # Get optimal label column order (minimizes horizontal gaps)
        label_column_order = self._get_optimal_label_order(metrics)

        # Build header: start + stats + label columns + end
        header = BASE_HEADERS_START + stat_keys + label_column_order + BASE_HEADERS_END

        # Add metadata column for histogram
        if metric_type == PrometheusMetricType.HISTOGRAM:
            header.append("buckets")

        writer.writerow(header)

        # Sort by bitmap pattern (vertical clustering), then name/endpoint/labels
        sorted_metrics = sorted(
            metrics,
            key=lambda m: self._get_vertical_sort_key(m, label_column_order),
        )

        for metric in sorted_metrics:
            labels = metric.stats.labels or {}
            is_info = metric.is_info_metric

            # Start with base columns
            row = [
                metric.endpoint,
                metric_type,
                metric.metric_name,
                metric.unit or "",
            ]

            series = metric.stats
            for stat in stat_keys:
                stat_value = self._get_stat_value(series, stat)
                row.append(self._format_number(stat_value))

            # Add individual label columns (blank for info metrics)
            for label_key in label_column_order:
                if is_info:
                    row.append("")
                else:
                    row.append(labels.get(label_key, ""))

            # Add description
            row.append(metric.description)

            # Add metadata column for histogram (key=value;key2=value2 format)
            if metric_type == PrometheusMetricType.HISTOGRAM:
                buckets = metric.stats.buckets or {}
                row.append(
                    ";".join(
                        f"{k}={self._format_number(v)}" for k, v in buckets.items()
                    )
                )

            writer.writerow(row)

    def _write_info_section(
        self,
        writer: csv.writer,
        metrics: list[CsvMetricInfo],
    ) -> None:
        """Write info metrics in transposed key-value format.

        Info metrics (ending with "_info") typically contain configuration data
        in their labels (e.g., version numbers, build info, config settings).
        These are better displayed in a transposed format where each label
        becomes a separate row, rather than creating many sparse columns.

        Format: Each label key-value pair becomes one row with columns:
        [Endpoint, Metric, Key, Value, Description]

        This makes configuration data easy to read and compare across endpoints.

        Args:
            writer: CSV writer to output rows to
            metrics: List of info metrics to export in transposed format
        """
        writer.writerow(INFO_HEADERS)

        # Sort by metric name, then endpoint
        sorted_metrics = sorted(metrics, key=lambda x: (x.metric_name, x.endpoint))

        for metric in sorted_metrics:
            labels = metric.stats.labels or {}
            # Write one row per label key-value pair
            for key in sorted(labels.keys()):
                writer.writerow(
                    [
                        metric.endpoint,
                        metric.metric_name,
                        key,
                        labels[key],
                        metric.description,
                    ]
                )

    def _get_stat_value(
        self,
        series: GaugeSeries | CounterSeries | HistogramSeries,
        stat: str,
    ):
        """Get a statistic value by name from the series stats object.

        All series use the stats format for API consistency.

        Args:
            series: Series object with stats field
            stat: Name of the statistic field to retrieve (e.g., 'avg', 'p99', 'total')

        Returns:
            The stat value if present, None otherwise (renders as empty cell in CSV)
        """
        if series.stats is None:
            return None
        return getattr(series.stats, stat, None)

    def _format_number(self, value) -> str:
        """Format a numeric value for CSV output with consistent precision.

        Applies type-appropriate formatting to ensure CSV readability while
        maintaining necessary precision for metrics analysis:

        - None → empty string (sparse data)
        - bool → "True"/"False" (explicit boolean values)
        - int → no decimal places (exact counts)
        - float/Decimal → 4 decimal places (balance precision vs readability)

        The 4 decimal precision for floats is sufficient for most metrics
        (millisecond precision for seconds, sub-percent for percentages) while
        keeping CSV files compact and readable.

        Args:
            value: The value to format (None, bool, int, float, or Decimal)

        Returns:
            Formatted string suitable for CSV cell content
        """
        if value is None:
            return ""
        # Handle bools explicitly (bool is a subclass of int)
        if isinstance(value, bool):
            return str(value)
        # Integers
        if isinstance(value, numbers.Integral):
            return f"{int(value)}"
        # Real numbers and Decimal
        if isinstance(value, numbers.Real | Decimal):
            return f"{float(value):.4f}"
        return str(value)

    def _get_optimal_label_order(self, metrics: list[CsvMetricInfo]) -> list[str]:
        """Determine optimal column ordering for label columns to minimize sparse gaps.

        Groups related labels into "families" using Union-Find on co-occurrence, then
        orders columns so that labels exclusive to a single metric appear before
        shared "bridge" labels. This produces a more compact CSV where metrics with
        similar label sets have adjacent filled columns.

        Args:
            metrics: List of metric info objects to analyze for label patterns.

        Returns:
            Ordered list of label names optimized for minimal horizontal sparsity.
        """
        label_sets = [
            set(m.stats.labels.keys())
            for m in metrics
            if not m.is_info_metric and m.stats.labels
        ]
        all_labels = set().union(*label_sets) if label_sets else set()
        if not all_labels:
            return []

        # Union-Find to group co-occurring labels into families
        parent: dict[str, str] = {}

        def find(x: str) -> str:
            return x if parent.setdefault(x, x) == x else find(parent[x])

        for ls in label_sets:
            labels = sorted(ls)
            for a, b in zip(labels, labels[1:], strict=False):
                parent[find(b)] = find(a)

        # Group labels by family, identify bridges (appear in multiple label sets)
        families: dict[str, set[str]] = defaultdict(set)
        for label in all_labels:
            families[find(label)].add(label)
        bridges = {lbl for lbl in all_labels if sum(lbl in ls for ls in label_sets) > 1}

        # Within each family: exclusive labels first, then bridges (both sorted)
        result: list[str] = []
        for family in sorted(families.values(), key=min):
            result.extend(sorted(lbl for lbl in family if lbl not in bridges))
            result.extend(sorted(lbl for lbl in family if lbl in bridges))
        return result

    def _get_vertical_sort_key(
        self, metric: CsvMetricInfo, label_order: list[str]
    ) -> tuple[str, str, str, str]:
        """Generate a sort key that clusters rows with similar column fill patterns.

        Produces a tuple where the primary sort component is a binary pattern string
        representing which label columns are filled ('1') vs empty ('0'). Rows with
        identical fill patterns are then sub-sorted by metric name, endpoint, and
        label values for deterministic ordering.

        Args:
            metric: The metric info to generate a sort key for.
            label_order: The ordered list of label column names from
                `_get_optimal_label_order`.

        Returns:
            A 4-tuple of (fill_pattern, metric_name, endpoint, labels_str) for sorting.
        """
        labels = metric.stats.labels or {}
        # Build pattern: '1' for filled columns, '0' for empty
        pattern = "".join("1" if col in labels else "0" for col in label_order)
        return (
            pattern,
            metric.metric_name,
            metric.endpoint,
            str(labels) if labels else "",
        )
