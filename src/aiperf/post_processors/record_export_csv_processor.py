# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from aiperf.common.enums import (
    MetricFlags,
    MetricType,
    MetricValueType,
)
from aiperf.common.environment import Environment
from aiperf.common.exceptions import PostProcessorDisabled
from aiperf.common.hooks import on_init
from aiperf.common.metric_records_wire import MetricRecordMetadata, MetricRecordsData
from aiperf.common.mixins.buffered_csv_writer_mixin import BufferedCSVWriterMixin
from aiperf.common.models.record_models import MetricResult, MetricValue
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.metrics.metric_registry import MetricRegistry
from aiperf.post_processors.base_metrics_processor import BaseMetricsProcessor

if TYPE_CHECKING:
    from aiperf.config import BenchmarkRun

_METADATA_COLUMNS = [
    "request_num",
    "session_num",
    "x_request_id",
    "x_correlation_id",
    "conversation_id",
    "turn_index",
    "benchmark_phase",
    "worker_id",
    "record_processor_id",
    "credit_issued_ns",
    "request_start_ns",
    "request_ack_ns",
    "request_end_ns",
    "was_cancelled",
    "cancellation_time_ns",
    "error_code",
    "error_message",
]

_LIST_VALUE_TYPES = frozenset({MetricValueType.FLOAT_LIST, MetricValueType.INT_LIST})


class RecordExportCSVProcessor(BaseMetricsProcessor, BufferedCSVWriterMixin):
    """Exports per-record metrics to CSV with flat column layout.

    Columns are determined upfront from the MetricRegistry so the header is
    stable regardless of which metrics any individual record produces.
    """

    def __init__(
        self,
        service_id: str,
        run: BenchmarkRun,
        **kwargs,
    ):
        config = run.cfg
        artifacts = config.artifacts
        records_enabled = artifacts.records and artifacts.records is not False
        if not records_enabled:
            raise PostProcessorDisabled(
                "CSV record export processor is disabled (artifacts.records is not enabled)"
            )
        if not (isinstance(artifacts.records, list) and "csv" in artifacts.records):
            raise PostProcessorDisabled(
                "CSV record export disabled: 'csv' not in artifacts.records"
            )

        output_file = artifacts.profile_export_records_csv_file
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.unlink(missing_ok=True)

        super().__init__(
            output_file=output_file,
            batch_size=Environment.RECORD.EXPORT_BATCH_SIZE,
            run=run,
            **kwargs,
        )

        self.show_internal = (
            Environment.DEV.MODE and Environment.DEV.SHOW_INTERNAL_METRICS
        )
        self.show_experimental = (
            Environment.DEV.MODE and Environment.DEV.SHOW_EXPERIMENTAL_METRICS
        )
        self.export_per_chunk_data = config.artifacts.per_chunk_data

        # Build the fixed metric column list from the registry
        self._metric_headers: list[str] = self._build_metric_headers()

        self.info(f"CSV record metrics export enabled: {self.output_file}")
        if self.export_per_chunk_data:
            self.info("Per-chunk data export enabled (--export-per-chunk-data)")

    def _build_metric_headers(self) -> list[str]:
        """Build the ordered list of metric column headers from the registry.

        Applies the same filters as to_display_dict: excludes internal,
        experimental, NO_INDIVIDUAL_RECORDS, and optionally list-valued metrics.
        """
        candidates: list[tuple[int | None, str]] = []

        for metric_class in MetricRegistry.all_classes():
            if metric_class.type not in (MetricType.RECORD, MetricType.AGGREGATE):
                continue
            if metric_class.has_flags(MetricFlags.NO_INDIVIDUAL_RECORDS):
                continue
            if metric_class.has_flags(MetricFlags.INTERNAL) and not self.show_internal:
                continue
            if (
                metric_class.has_flags(MetricFlags.EXPERIMENTAL)
                and not self.show_experimental
            ):
                continue
            if (
                not self.export_per_chunk_data
                and metric_class.value_type in _LIST_VALUE_TYPES
            ):
                continue

            display_unit = metric_class.display_unit or metric_class.unit
            header = f"{metric_class.tag} ({display_unit})"
            candidates.append((metric_class.display_order, header))

        candidates.sort(key=lambda x: (x[0] is None, x[0] or 0))
        return [header for _, header in candidates]

    @on_init
    async def _set_csv_columns(self) -> None:
        """Set the CSV column headers from the pre-built metric headers."""
        all_columns = list(_METADATA_COLUMNS) + self._metric_headers
        self.set_csv_columns(all_columns)

    def _flatten_metadata(
        self,
        metadata: MetricRecordMetadata,
        error_code: str,
        error_message: str,
    ) -> list[str]:
        """Flatten metadata into a list of string values in _METADATA_COLUMNS order."""
        return [
            str(metadata.request_num) if metadata.request_num is not None else "",
            str(metadata.session_num),
            metadata.x_request_id or "",
            metadata.x_correlation_id or "",
            metadata.conversation_id or "",
            str(metadata.turn_index) if metadata.turn_index is not None else "",
            str(metadata.benchmark_phase),
            metadata.worker_id,
            metadata.record_processor_id,
            str(metadata.credit_issued_ns)
            if metadata.credit_issued_ns is not None
            else "",
            str(metadata.request_start_ns),
            str(metadata.request_ack_ns) if metadata.request_ack_ns is not None else "",
            str(metadata.request_end_ns),
            str(metadata.was_cancelled),
            str(metadata.cancellation_time_ns)
            if metadata.cancellation_time_ns is not None
            else "",
            error_code,
            error_message,
        ]

    def _build_row(
        self,
        metadata_row: list[str],
        display_metrics: dict[str, MetricValue],
    ) -> list[str]:
        """Build a flat CSV row from metadata and metrics."""
        metric_lookup: dict[str, str] = {}
        for tag, mv in display_metrics.items():
            header = f"{tag} ({mv.unit})"
            value = mv.value
            metric_lookup[header] = (
                f"{value:.6g}" if isinstance(value, float) else str(value)
            )

        return metadata_row + [
            metric_lookup.get(col_header, "") for col_header in self._metric_headers
        ]

    async def process_result(self, record_data: MetricRecordsData) -> None:
        try:
            metric_dict = MetricRecordDict(record_data.metrics)
            display_metrics = metric_dict.to_display_dict(
                MetricRegistry, self.show_internal, self.show_experimental
            )
            if not display_metrics and not record_data.error:
                return

            # Filter out list-valued metrics (per-chunk arrays) unless explicitly enabled
            if not self.export_per_chunk_data:
                display_metrics = {
                    k: v
                    for k, v in display_metrics.items()
                    if not isinstance(v.value, list)
                }

            error_code = ""
            error_message = ""
            if record_data.error:
                error_code = (
                    str(record_data.error.code)
                    if record_data.error.code is not None
                    else ""
                )
                error_message = record_data.error.message or ""

            metadata_row = self._flatten_metadata(
                record_data.metadata, error_code, error_message
            )
            row = self._build_row(metadata_row, display_metrics)
            await self.buffered_csv_write(row)

        except Exception as e:
            self.error(f"Failed to write CSV record metrics: {e}")

    async def summarize(self) -> list[MetricResult]:
        """No aggregation needed for CSV export."""
        return []
