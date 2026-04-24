# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parquet exporter for raw server metrics with delta calculations."""

from collections.abc import Iterator
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq

from aiperf.common.enums import PrometheusMetricType, ServerMetricsFormat
from aiperf.common.exceptions import DataExporterDisabled
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.models.server_metrics_models import TimeRangeFilter
from aiperf.exporters.exporter_config import FileExportInfo
from aiperf.server_metrics.parquet_metadata import build_parquet_metadata
from aiperf.server_metrics.parquet_rows import (
    collect_histogram_rows,
    collect_scalar_rows,
)

if TYPE_CHECKING:
    from aiperf.server_metrics.accumulator import ServerMetricsAccumulator

__all__ = ["ServerMetricsParquetExporter"]

_BATCH_SIZE = 10_000
_RESERVED_COLUMN_NAMES: frozenset[str] = frozenset(
    {
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
)


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
        self.run = server_metrics_accumulator.run
        if self.run.cfg.server_metrics_disabled:
            raise DataExporterDisabled("Server metrics is disabled")

        if ServerMetricsFormat.PARQUET not in self.run.cfg.server_metrics_formats:
            raise DataExporterDisabled(
                "Server metrics Parquet export disabled: format not selected"
            )

        super().__init__(**kwargs)
        self._file_path = self.run.cfg.output.server_metrics_export_parquet_file
        self._accumulator = server_metrics_accumulator
        self._time_filter = time_filter
        self.trace_or_debug(
            lambda: f"Initializing ServerMetricsParquetExporter with config: {self.run.cfg}",
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

        Returns:
            FileExportInfo with export type and file path
        """
        self.debug("Discovering label keys...")
        all_label_keys = self._discover_all_label_keys()
        # "endpoint" is a common Prometheus label; we reserve "endpoint_url" for our column.
        label_keys = {lk for lk in all_label_keys if lk not in _RESERVED_COLUMN_NAMES}
        self.debug(lambda: f"Found {len(label_keys)} label keys")

        schema = self._build_pyarrow_schema(label_keys).with_metadata(
            build_parquet_metadata(
                accumulator=self._accumulator,
                time_filter=self._time_filter,
                label_keys=label_keys,
            )
        )

        self.debug("Writing Parquet file with streaming batches...")
        total_rows = self._stream_rows_to_parquet(schema, label_keys)
        if total_rows == 0:
            self.warning("No data to export. Skipping Parquet file creation.")
            return self.get_export_info()

        self._validate_export(total_rows)
        return self.get_export_info()

    def _stream_rows_to_parquet(self, schema: "pa.Schema", label_keys: set[str]) -> int:
        """Stream rows to a Parquet writer in fixed-size batches.

        Returns 0 if there was no data (and skips file creation). Otherwise
        returns the total number of rows written.
        """
        row_generator = self._collect_all_rows_generator(label_keys)
        batch_rows: list[dict] = []
        # Peek: collect a first batch to determine whether any data exists.
        for row in row_generator:
            batch_rows.append(row)
            if len(batch_rows) >= _BATCH_SIZE:
                break
        if not batch_rows:
            return 0

        total_rows = 0
        with pq.ParquetWriter(self._file_path, schema, compression="snappy") as writer:
            self._write_batch(writer, schema, batch_rows)
            total_rows += len(batch_rows)
            self.trace(
                lambda: f"Wrote batch of {len(batch_rows):,} rows (total: {total_rows:,})"
            )
            batch_rows = []
            for row in row_generator:
                batch_rows.append(row)
                if len(batch_rows) >= _BATCH_SIZE:
                    self._write_batch(writer, schema, batch_rows)
                    total_rows += len(batch_rows)
                    batch_count = len(batch_rows)
                    current_total = total_rows
                    self.trace(
                        lambda batch_count=batch_count,
                        current_total=current_total: f"Wrote batch of {batch_count:,} rows (total: {current_total:,})"
                    )
                    batch_rows = []
            if batch_rows:
                self._write_batch(writer, schema, batch_rows)
                total_rows += len(batch_rows)
        return total_rows

    @staticmethod
    def _write_batch(
        writer: pq.ParquetWriter, schema: "pa.Schema", batch_rows: list[dict]
    ) -> None:
        table = pa.table(
            {col: [r.get(col) for r in batch_rows] for col in schema.names},
            schema=schema,
        )
        writer.write_table(table)

    def _validate_export(self, total_rows: int) -> None:
        try:
            if not self._file_path.exists():
                raise RuntimeError(f"Parquet file was not created: {self._file_path}")
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

    def _discover_all_label_keys(self) -> set[str]:
        """Discover all unique label keys across all metrics.

        Similar to CSV exporter's label discovery, scans all metrics to find
        unique label keys for dynamic column creation.

        Returns:
            Set of label key strings (e.g., {"method", "status", "endpoint_path"})
        """
        label_keys: set[str] = set()
        hierarchy = self._accumulator.get_hierarchy_for_export()
        for time_series in hierarchy.endpoints.values():
            for metric_key in time_series.metrics:
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
            pa.field("endpoint_url", pa.string()),
            pa.field("metric_name", pa.string()),
            pa.field("metric_type", pa.string()),
            pa.field("unit", pa.string(), nullable=True),
            pa.field("description", pa.string(), nullable=True),
            pa.field("timestamp_ns", pa.int64()),
        ]
        for label_key in sorted(label_keys):
            fields.append(pa.field(label_key, pa.string(), nullable=True))
        fields.extend(
            [
                pa.field("value", pa.float64(), nullable=True),
                pa.field("sum", pa.float64(), nullable=True),
                pa.field("count", pa.float64(), nullable=True),
                pa.field("bucket_le", pa.string(), nullable=True),
                pa.field("bucket_count", pa.float64(), nullable=True),
            ]
        )
        return pa.schema(fields)

    def _collect_all_rows_generator(self, label_keys: set[str]) -> Iterator[dict]:
        """Yield rows from all endpoints and metrics with delta calculations.

        Memory-efficient generator version that yields rows one at a time instead of
        collecting all rows in memory. Used for streaming writes to Parquet.
        """
        hierarchy = self._accumulator.get_hierarchy_for_export()
        for endpoint_url, time_series_collection in hierarchy.endpoints.items():
            for metric_key, metric_entry in time_series_collection.metrics.items():
                metric_type = metric_entry.metric_type
                labels_dict = metric_key.labels_dict

                if metric_type in (
                    PrometheusMetricType.GAUGE,
                    PrometheusMetricType.COUNTER,
                ):
                    yield from collect_scalar_rows(
                        endpoint=endpoint_url,
                        metric_name=metric_key.name,
                        metric_entry=metric_entry,
                        labels_dict=labels_dict,
                        label_keys=label_keys,
                        time_filter=self._time_filter,
                    )
                elif metric_type == PrometheusMetricType.HISTOGRAM:
                    yield from collect_histogram_rows(
                        endpoint=endpoint_url,
                        metric_name=metric_key.name,
                        metric_entry=metric_entry,
                        labels_dict=labels_dict,
                        label_keys=label_keys,
                        time_filter=self._time_filter,
                    )
