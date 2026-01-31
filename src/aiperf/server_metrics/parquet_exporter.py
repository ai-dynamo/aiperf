# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parquet exporter for raw server metrics with delta calculations."""

from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from aiperf.common.enums import PrometheusMetricType, ServerMetricsFormat
from aiperf.common.exceptions import DataExporterDisabled
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.models.server_metrics_models import TimeRangeFilter
from aiperf.exporters.exporter_config import FileExportInfo
from aiperf.server_metrics.storage import (
    HistogramTimeSeries,
    ScalarTimeSeries,
    ServerMetricEntry,
)
from aiperf.server_metrics.units import infer_unit

if TYPE_CHECKING:
    from aiperf.server_metrics.accumulator import ServerMetricsAccumulator

__all__ = ["ServerMetricsParquetExporter"]


class ServerMetricsParquetExporter(AIPerfLoggerMixin):
    """Export raw server metrics time-series with delta calculations to Parquet format.

    Exports raw time-series data in columnar Parquet format with cumulative deltas
    applied at each timestamp. Uses normalized schema where histogram buckets are
    separate rows rather than separate columns, producing smaller files (50% size
    reduction) and better SQL query ergonomics.

    Delta calculations:
    - Gauges: Raw values at each timestamp (no delta)
    - Counters: Cumulative delta from reference point at each timestamp
    - Histograms: Cumulative sum/count/bucket deltas from reference at each timestamp

    Schema features:
    - Dynamic label columns discovered from metric labels (e.g., method, status, model)
    - Natural bucket values without sanitization (0.01, 0.1, +Inf - not bucket_0_01)
    - Inferred units for metrics (seconds, tokens, requests, ratio, etc.)
    - Single row per timestamp for gauges/counters
    - N rows per timestamp for histograms (one per bucket)

    The normalized schema enables SQL queries like:
    - WHERE bucket_le = '0.1' (natural values)
    - WHERE method='GET' AND status='200' (label filtering)
    - GROUP BY bucket_le (histogram reconstruction)

    Designed for analytics workflows using DuckDB, pandas, or Polars.
    """

    def __init__(
        self,
        server_metrics_accumulator: "ServerMetricsAccumulator",
        time_filter: TimeRangeFilter,
        **kwargs,
    ) -> None:
        """Initialize the Parquet exporter for server metrics.

        Validates that Parquet format is enabled and sets up file paths. The exporter
        accesses raw time-series data directly from the accumulator (which cannot be
        serialized through ZMQ), so it must be called in the same process where the
        accumulator exists (RecordsManager).

        Args:
            server_metrics_accumulator: Accumulator containing raw time-series data
            time_filter: Time range filter for profiling period (excludes warmup)
            **kwargs: Additional arguments passed to base class

        Raises:
            DataExporterDisabled: If server metrics are disabled or Parquet format not selected
        """
        self.user_config = server_metrics_accumulator.user_config
        if self.user_config.server_metrics_disabled:
            raise DataExporterDisabled("Server metrics is disabled")

        # Check if Parquet format is enabled
        if ServerMetricsFormat.PARQUET not in self.user_config.server_metrics_formats:
            raise DataExporterDisabled(
                "Server metrics Parquet export disabled: format not selected"
            )

        super().__init__(**kwargs)
        self._file_path = self.user_config.output.server_metrics_export_parquet_file
        self._accumulator = server_metrics_accumulator
        self._time_filter = time_filter
        self.trace_or_debug(
            lambda: f"Initializing ServerMetricsParquetExporter with config: {self.user_config}",
            lambda: f"Initializing ServerMetricsParquetExporter with file path: {self._file_path}",
        )

    def get_export_info(self) -> FileExportInfo:
        """Return export metadata for logging and user feedback.

        Returns:
            FileExportInfo with export type description and target file path
        """
        return FileExportInfo(
            export_type="Server Metrics Parquet Export",
            file_path=self._file_path,
        )

    async def export(self) -> FileExportInfo:
        """Export server metrics to Parquet file with normalized schema using streaming writes.

        Performs schema discovery (label keys), collects rows in batches with delta calculations,
        builds PyArrow schema with dynamic label columns, and writes to Parquet file incrementally
        with Snappy compression. Uses streaming writes to minimize memory usage.

        The normalized schema uses separate rows per histogram bucket instead of separate
        columns, producing files that are ~50% smaller and more SQL-friendly than wide schema.

        Returns:
            FileExportInfo with export type and file path

        Raises:
            ImportError: If pyarrow is not installed (handled gracefully with warning)
        """
        self.debug("Discovering label keys...")
        all_label_keys = self._discover_all_label_keys()

        # Filter label keys to avoid conflicts with reserved columns
        # Note: "endpoint" is a common Prometheus label, so we use "endpoint_url" for our column
        reserved_names = self._get_reserved_names()
        label_keys = {lk for lk in all_label_keys if lk not in reserved_names}

        self.debug(lambda: f"Found {len(label_keys)} label keys")

        # Build PyArrow schema with metadata
        schema = self._build_pyarrow_schema(label_keys)
        metadata = self._build_parquet_metadata(label_keys)
        schema = schema.with_metadata(metadata)

        # Stream rows in batches to minimize memory usage
        self.debug("Writing Parquet file with streaming batches...")
        total_rows = 0
        batch_size = 10_000  # Process 10K rows at a time to control memory
        batch_rows = []

        # Collect first batch to check if we have any data
        # This avoids creating empty files
        row_generator = self._collect_all_rows_generator(label_keys)
        for row in row_generator:
            batch_rows.append(row)
            if len(batch_rows) >= batch_size:
                break

        # If no data at all, skip file creation
        if not batch_rows:
            self.warning("No data to export. Skipping Parquet file creation.")
            return self.get_export_info()

        # We have data, open writer and write batches
        with pq.ParquetWriter(self._file_path, schema, compression="snappy") as writer:
            # Write first batch
            table = pa.table(
                {col: [r.get(col) for r in batch_rows] for col in schema.names},
                schema=schema,
            )
            writer.write_table(table)
            total_rows += len(batch_rows)
            self.trace(
                lambda: f"Wrote batch of {len(batch_rows):,} rows (total: {total_rows:,})"
            )
            batch_rows = []

            # Continue with remaining rows
            for row in row_generator:
                batch_rows.append(row)

                # Write batch when it reaches batch_size
                if len(batch_rows) >= batch_size:
                    table = pa.table(
                        {col: [r.get(col) for r in batch_rows] for col in schema.names},
                        schema=schema,
                    )
                    writer.write_table(table)
                    total_rows += len(batch_rows)
                    batch_count = len(batch_rows)
                    current_total = total_rows
                    self.trace(
                        lambda batch_count=batch_count,
                        current_total=current_total: f"Wrote batch of {batch_count:,} rows (total: {current_total:,})"
                    )
                    batch_rows = []

            # Write remaining rows
            if batch_rows:
                table = pa.table(
                    {col: [r.get(col) for r in batch_rows] for col in schema.names},
                    schema=schema,
                )
                writer.write_table(table)
                total_rows += len(batch_rows)

        # Validate export success (metadata already included in schema)
        try:
            if not self._file_path.exists():
                raise RuntimeError(f"Parquet file was not created: {self._file_path}")

            # Verify file can be read and check row count using metadata (no full read)
            parquet_metadata = pq.read_metadata(self._file_path)
            actual_rows = parquet_metadata.num_rows

            if actual_rows != total_rows:
                self.warning(
                    f"Row count mismatch: wrote {total_rows:,} rows but file contains {actual_rows:,} rows"
                )
            else:
                self.info(
                    f"Successfully wrote and validated {total_rows:,} rows to {self._file_path}"
                )

        except Exception as e:
            self.error(f"Failed to validate Parquet export: {e!r}")
            raise RuntimeError(f"Parquet export validation failed: {e!r}") from e

        return self.get_export_info()

    def _get_reserved_names(self) -> set[str]:
        """Get set of reserved column names that cannot be used as label keys.

        Returns:
            Set of reserved column names
        """
        return {
            "endpoint_url",
            "metric_name",
            "metric_type",
            "unit",
            "description",
            "timestamp_ns",
            "value",
            "sum",
            "count",
            "bucket_le",
            "bucket_count",
        }

    def _build_parquet_metadata(self, label_keys: set[str]) -> dict[bytes, bytes]:
        """Build Parquet file metadata for provenance tracking.

        Args:
            label_keys: Set of label columns in this file's schema

        Returns:
            Dictionary of metadata key-value pairs (both as bytes)
        """
        import socket
        import sys
        from importlib.metadata import version as get_version

        import orjson

        # Get AIPerf version from installed package
        try:
            aiperf_version = get_version("aiperf")
        except Exception:
            aiperf_version = "unknown"

        # Core metadata
        metadata = {
            b"aiperf.schema_version": b"1.0",
            b"aiperf.version": aiperf_version.encode("utf-8"),
            b"aiperf.benchmark_id": self.user_config.benchmark_id.encode("utf-8"),
            b"aiperf.export_timestamp_utc": datetime.now(timezone.utc)
            .isoformat()
            .encode("utf-8"),
            b"aiperf.exporter": b"ServerMetricsParquetExporter",
        }

        # System information for reproducibility
        metadata[b"aiperf.hostname"] = socket.gethostname().encode("utf-8")
        metadata[b"aiperf.python_version"] = (
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        ).encode()

        # Library versions
        try:
            metadata[b"aiperf.pyarrow_version"] = pa.__version__.encode("utf-8")
        except AttributeError:
            metadata[b"aiperf.pyarrow_version"] = b"unknown"

        # Time filter information
        if self._time_filter.start_ns is not None:
            metadata[b"aiperf.time_filter_start_ns"] = str(
                self._time_filter.start_ns
            ).encode("utf-8")

        if self._time_filter.end_ns is not None:
            metadata[b"aiperf.time_filter_end_ns"] = str(
                self._time_filter.end_ns
            ).encode("utf-8")

            # Calculate profiling duration
            if self._time_filter.start_ns is not None:
                duration_ns = self._time_filter.end_ns - self._time_filter.start_ns
                metadata[b"aiperf.profiling_duration_ns"] = str(duration_ns).encode(
                    "utf-8"
                )
                metadata[b"aiperf.profiling_duration_seconds"] = str(
                    duration_ns / 1_000_000_000
                ).encode("utf-8")

        # Benchmark configuration (full context)
        # Dump entire user config with exclude_unset to capture actual benchmark settings
        config_dict = self.user_config.model_dump(
            mode="json", exclude_unset=True, exclude_none=True
        )
        metadata[b"aiperf.input_config"] = orjson.dumps(config_dict)

        # Also add key config values for quick access (without parsing JSON)
        metadata[b"aiperf.model_names"] = orjson.dumps(
            self.user_config.endpoint.model_names
        )
        metadata[b"aiperf.concurrency"] = str(
            self.user_config.loadgen.concurrency
        ).encode("utf-8")

        if self.user_config.loadgen.request_rate is not None:
            metadata[b"aiperf.request_rate"] = str(
                self.user_config.loadgen.request_rate
            ).encode("utf-8")

        # Endpoint information
        hierarchy = self._accumulator.get_hierarchy_for_export()
        endpoint_urls = sorted(hierarchy.endpoints.keys())
        metadata[b"aiperf.endpoint_urls"] = orjson.dumps(endpoint_urls)
        metadata[b"aiperf.endpoint_count"] = str(len(endpoint_urls)).encode("utf-8")

        # Schema information for cross-file compatibility
        metadata[b"aiperf.label_columns"] = orjson.dumps(sorted(label_keys))
        metadata[b"aiperf.label_count"] = str(len(label_keys)).encode("utf-8")

        # Metric counts by type (pre-calculated to avoid file rewrite)
        type_counts = {"gauge": 0, "counter": 0, "histogram": 0}
        total_metrics = 0
        for time_series_collection in hierarchy.endpoints.values():
            for metric_entry in time_series_collection.metrics.values():
                total_metrics += 1
                if metric_entry.metric_type == PrometheusMetricType.GAUGE:
                    type_counts["gauge"] += 1
                elif metric_entry.metric_type == PrometheusMetricType.COUNTER:
                    type_counts["counter"] += 1
                elif metric_entry.metric_type == PrometheusMetricType.HISTOGRAM:
                    type_counts["histogram"] += 1

        metadata[b"aiperf.metric_count"] = str(total_metrics).encode("utf-8")
        metadata[b"aiperf.metric_type_counts"] = orjson.dumps(type_counts)

        # Usage note for cross-file queries
        metadata[b"aiperf.schema_note"] = (
            b"Label columns vary by endpoint/model. Use union_by_name=true for cross-file queries."
        )

        return metadata

    def _discover_all_label_keys(self) -> set[str]:
        """Discover all unique label keys across all metrics.

        Similar to CSV exporter's label discovery, scans all metrics to find
        unique label keys for dynamic column creation.

        Returns:
            Set of label key strings (e.g., {"method", "status", "endpoint_path"})
        """
        label_keys = set()
        hierarchy = self._accumulator.get_hierarchy_for_export()
        for (
            _endpoint_url,
            time_series,
        ) in hierarchy.endpoints.items():
            for metric_key, _metric_entry in time_series.metrics.items():
                if metric_key.labels_dict:
                    label_keys.update(metric_key.labels_dict.keys())
        return label_keys

    def _build_pyarrow_schema(self, label_keys: set[str]) -> "pa.Schema":
        """Build PyArrow schema with normalized histogram buckets.

        Normalized schema uses separate rows per bucket instead of separate columns.
        This produces smaller files and better SQL query ergonomics.

        Args:
            label_keys: Set of label key strings (already filtered to avoid conflicts)

        Returns:
            PyArrow schema with all columns (common + labels + values + bucket fields)
        """
        fields = [
            # Common columns (non-nullable required fields)
            pa.field("endpoint_url", pa.string()),
            pa.field("metric_name", pa.string()),
            pa.field("metric_type", pa.string()),
            pa.field("unit", pa.string(), nullable=True),
            pa.field("description", pa.string(), nullable=True),
            pa.field("timestamp_ns", pa.int64()),
        ]

        # Dynamic label columns (sorted alphabetically for consistency)
        for label_key in sorted(label_keys):
            fields.append(pa.field(label_key, pa.string(), nullable=True))

        # Value columns (nullable)
        fields.extend(
            [
                pa.field("value", pa.float64(), nullable=True),
                pa.field("sum", pa.float64(), nullable=True),
                pa.field("count", pa.float64(), nullable=True),
                # Normalized bucket representation (for histograms only)
                pa.field("bucket_le", pa.string(), nullable=True),
                pa.field("bucket_count", pa.float64(), nullable=True),
            ]
        )

        return pa.schema(fields)

    def _collect_all_rows_generator(self, label_keys: set[str]):
        """Generator that yields rows from all endpoints and metrics with delta calculations.

        Memory-efficient generator version that yields rows one at a time instead of
        collecting all rows in memory. Used for streaming writes to Parquet.

        Uses normalized schema where histogram buckets are separate rows rather than
        separate columns. This produces smaller files and better SQL ergonomics.

        Uses self._time_filter for time range filtering.

        Args:
            label_keys: Set of all label keys for column population

        Yields:
            Row dictionaries ready for PyArrow table creation
        """
        hierarchy = self._accumulator.get_hierarchy_for_export()
        for (
            endpoint_url,
            time_series_collection,
        ) in hierarchy.endpoints.items():
            for metric_key, metric_entry in time_series_collection.metrics.items():
                metric_type = metric_entry.metric_type
                labels_dict = metric_key.labels_dict

                if metric_type in (
                    PrometheusMetricType.GAUGE,
                    PrometheusMetricType.COUNTER,
                ):
                    rows = self._collect_scalar_rows(
                        endpoint_url,
                        metric_key.name,
                        metric_entry,
                        labels_dict,
                        label_keys,
                    )
                    yield from rows
                elif metric_type == PrometheusMetricType.HISTOGRAM:
                    rows = self._collect_histogram_rows(
                        endpoint_url,
                        metric_key.name,
                        metric_entry,
                        labels_dict,
                        label_keys,
                    )
                    yield from rows

    def _collect_all_rows(
        self,
        label_keys: set[str],
    ) -> list[dict]:
        """Collect all rows from all endpoints and metrics with delta calculations.

        Uses normalized schema where histogram buckets are separate rows rather than
        separate columns. This produces smaller files and better SQL ergonomics.

        Uses self._time_filter for time range filtering.

        Args:
            label_keys: Set of all label keys for column population

        Returns:
            List of row dictionaries ready for PyArrow table creation
        """
        rows = []
        hierarchy = self._accumulator.get_hierarchy_for_export()
        for (
            endpoint_url,
            time_series_collection,
        ) in hierarchy.endpoints.items():
            for metric_key, metric_entry in time_series_collection.metrics.items():
                metric_type = metric_entry.metric_type
                labels_dict = metric_key.labels_dict

                if metric_type in (
                    PrometheusMetricType.GAUGE,
                    PrometheusMetricType.COUNTER,
                ):
                    rows.extend(
                        self._collect_scalar_rows(
                            endpoint_url,
                            metric_key.name,
                            metric_entry,
                            labels_dict,
                            label_keys,
                        )
                    )
                elif metric_type == PrometheusMetricType.HISTOGRAM:
                    rows.extend(
                        self._collect_histogram_rows(
                            endpoint_url,
                            metric_key.name,
                            metric_entry,
                            labels_dict,
                            label_keys,
                        )
                    )
        return rows

    def _collect_scalar_rows(
        self,
        endpoint: str,
        metric_name: str,
        metric_entry: ServerMetricEntry,
        labels_dict: dict[str, str] | None,
        label_keys: set[str],
    ) -> list[dict]:
        """Collect rows for gauge or counter metrics with delta calculations.

        For gauges: exports raw values at each timestamp.
        For counters: exports cumulative deltas from reference point at each timestamp.

        Uses self._time_filter for time range filtering.

        Args:
            endpoint: Endpoint URL
            metric_name: Metric name
            metric_entry: Metric entry containing time series data
            labels_dict: Labels for this metric
            label_keys: All label keys for column population

        Returns:
            List of row dictionaries for this metric
        """
        time_series = metric_entry.data
        if not isinstance(time_series, ScalarTimeSeries):
            return []

        metric_type = metric_entry.metric_type
        is_gauge = metric_type == PrometheusMetricType.GAUGE

        # Infer unit for this metric
        unit = infer_unit(metric_name, metric_entry.description)
        unit_display = unit.display_name() if unit else None

        # Get filtered data
        time_mask = (
            time_series.get_time_mask(self._time_filter)
            if self._time_filter
            else np.ones(len(time_series), dtype=bool)
        )
        filtered_timestamps = time_series.timestamps[time_mask]
        filtered_values = time_series.values[time_mask]

        if len(filtered_timestamps) == 0:
            return []

        # For counters, compute cumulative deltas from reference
        if is_gauge:
            # Gauges: use raw values
            values_to_export = filtered_values
        else:
            # Counters: compute cumulative deltas from reference point
            reference_idx = (
                time_series.get_reference_idx(self._time_filter)
                if self._time_filter
                else None
            )
            reference_value = (
                time_series.values[reference_idx]
                if reference_idx is not None
                else filtered_values[0]
            )
            # Vectorized delta calculation
            deltas = filtered_values - reference_value
            # Handle counter resets (negative deltas become 0)
            values_to_export = np.maximum(deltas, 0.0)

        # Build rows
        assert len(filtered_timestamps) == len(values_to_export), (
            f"Array length mismatch: {len(filtered_timestamps)} timestamps != {len(values_to_export)} values"
        )

        rows = []
        for timestamp, value in zip(
            filtered_timestamps, values_to_export, strict=False
        ):
            row = {
                "endpoint_url": endpoint,
                "metric_name": metric_name,
                "metric_type": metric_type,
                "unit": unit_display,
                "description": metric_entry.description,
                "timestamp_ns": int(timestamp),
                # Individual label columns (None for missing labels)
                **{
                    label_key: labels_dict.get(label_key) if labels_dict else None
                    for label_key in label_keys
                },
                "value": float(value),
                "sum": None,
                "count": None,
                # Bucket fields (None for non-histogram metrics)
                "bucket_le": None,
                "bucket_count": None,
            }
            rows.append(row)

        return rows

    def _collect_histogram_rows(
        self,
        endpoint: str,
        metric_name: str,
        metric_entry: ServerMetricEntry,
        labels_dict: dict[str, str] | None,
        label_keys: set[str],
    ) -> list[dict]:
        """Collect rows for histogram metrics with delta calculations.

        Creates one row per bucket per timestamp (normalized schema).
        Each row includes cumulative deltas for sum/count, plus individual bucket delta.

        Uses self._time_filter for time range filtering.

        Args:
            endpoint: Endpoint URL
            metric_name: Metric name
            metric_entry: Metric entry containing time series data
            labels_dict: Labels for this metric
            label_keys: All label keys for column population

        Returns:
            List of row dictionaries for this metric (one per bucket per timestamp)
        """
        time_series = metric_entry.data
        if not isinstance(time_series, HistogramTimeSeries):
            return []

        if len(time_series) == 0:
            return []

        # Infer unit for this metric
        unit = infer_unit(metric_name, metric_entry.description)
        unit_display = unit.display_name() if unit else None

        # Get filtered data using indices (HistogramTimeSeries doesn't have get_time_mask)
        if self._time_filter:
            reference_idx, final_idx = time_series.get_indices_for_filter(
                self._time_filter
            )
            # Find first index in filter range
            first_idx = np.searchsorted(
                time_series.timestamps, self._time_filter.start_ns, side="left"
            )
        else:
            reference_idx = None
            first_idx = 0
            final_idx = len(time_series) - 1

        # Get filtered data slice
        filtered_timestamps = time_series.timestamps[first_idx : final_idx + 1]
        filtered_sums = time_series.sums[first_idx : final_idx + 1]
        filtered_counts = time_series.counts[first_idx : final_idx + 1]
        filtered_bucket_counts = time_series.bucket_counts[first_idx : final_idx + 1]

        if len(filtered_timestamps) == 0:
            return []

        # Get reference values for delta calculation
        if reference_idx is not None:
            reference_sum = time_series.sums[reference_idx]
            reference_count = time_series.counts[reference_idx]
            reference_buckets = time_series.bucket_counts[reference_idx]
        else:
            reference_sum = filtered_sums[0]
            reference_count = filtered_counts[0]
            reference_buckets = filtered_bucket_counts[0]

        # Vectorized delta calculations
        sum_deltas = filtered_sums - reference_sum
        count_deltas = filtered_counts - reference_count
        # For buckets: broadcast subtract (2D array - 1D array)
        bucket_deltas = filtered_bucket_counts - reference_buckets

        # Handle counter resets (negative deltas become 0)
        sum_deltas = np.maximum(sum_deltas, 0.0)
        count_deltas = np.maximum(count_deltas, 0.0)
        bucket_deltas = np.maximum(bucket_deltas, 0.0)

        # Build rows - one row per bucket per timestamp (normalized schema)
        rows = []
        bucket_les = time_series.bucket_les

        for i, timestamp in enumerate(filtered_timestamps):
            sum_delta = float(sum_deltas[i])
            count_delta = float(count_deltas[i])

            # Create one row per bucket for this timestamp
            for j, bucket_le in enumerate(bucket_les):
                bucket_delta = float(bucket_deltas[i, j])

                row = {
                    "endpoint_url": endpoint,
                    "metric_name": metric_name,
                    "metric_type": PrometheusMetricType.HISTOGRAM,
                    "unit": unit_display,
                    "description": metric_entry.description,
                    "timestamp_ns": int(timestamp),
                    # Individual label columns (None for missing labels)
                    **{
                        label_key: labels_dict.get(label_key) if labels_dict else None
                        for label_key in label_keys
                    },
                    "value": None,
                    "sum": sum_delta,
                    "count": count_delta,
                    # Bucket fields (normalized - one row per bucket)
                    "bucket_le": bucket_le,
                    "bucket_count": bucket_delta,
                }
                rows.append(row)

        return rows
