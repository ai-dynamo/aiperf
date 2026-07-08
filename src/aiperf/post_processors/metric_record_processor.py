# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from aiperf.common.enums import MetricType
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.metric_records_wire import MetricRecordMetadata
from aiperf.common.models import ParsedResponseRecord
from aiperf.common.types import MetricTagT
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.post_processors.base_metrics_processor import BaseMetricsProcessor

if TYPE_CHECKING:
    from aiperf.config import BenchmarkRun


class MetricRecordProcessor(BaseMetricsProcessor):
    """Processor for metric records.

    This is the first stage of the metrics processing pipeline, and is done is a distributed manner across multiple service instances.
    It is responsible for streaming the records to the post processor, and computing the metrics from the records.
    It computes metrics from MetricType.RECORD and MetricType.AGGREGATE types."""

    def __init__(
        self,
        run: BenchmarkRun,
        **kwargs,
    ) -> None:
        super().__init__(run=run, **kwargs)

        # Store a reference to the parse_record function for valid metrics.
        # This is done to avoid extra attribute lookups.
        self.valid_parse_funcs: list[
            tuple[MetricTagT, Callable[[ParsedResponseRecord, MetricRecordDict], Any]]
        ] = [
            (metric.tag, metric.parse_record)  # type: ignore
            for metric in self._setup_metrics(
                MetricType.RECORD, MetricType.AGGREGATE, exclude_error_metrics=True
            )
        ]

        # Store a reference to the parse_record function for error metrics.
        # This is done to avoid extra attribute lookups.
        self.error_parse_funcs: list[
            tuple[MetricTagT, Callable[[ParsedResponseRecord, MetricRecordDict], Any]]
        ] = [
            (metric.tag, metric.parse_record)  # type: ignore
            for metric in self._setup_metrics(
                MetricType.RECORD, MetricType.AGGREGATE, error_metrics_only=True
            )
        ]

        # Store a reference to the parse_record function for cancelled metrics.
        # A client-cancelled request (--request-cancellation-rate) is NOT a
        # server error: it carries a code-499 RequestCancellationError so
        # record.valid is False, but routing it to the error path would inflate
        # error_request_count and deflate goodput, contradicting the credit-side
        # `cancelled` bucket. It gets its own parse funcs (CANCELLED_ONLY).
        self.cancelled_parse_funcs: list[
            tuple[MetricTagT, Callable[[ParsedResponseRecord, MetricRecordDict], Any]]
        ] = [
            (metric.tag, metric.parse_record)  # type: ignore
            for metric in self._setup_metrics(
                MetricType.RECORD, MetricType.AGGREGATE, cancelled_metrics_only=True
            )
        ]

    async def process_record(
        self, record: ParsedResponseRecord, metadata: MetricRecordMetadata
    ) -> MetricRecordDict:
        """Process a response record from the inference results parser."""
        record_metrics: MetricRecordDict = MetricRecordDict()
        # Check cancellation before validity: a cancelled record is invalid
        # (has a RequestCancellationError) but must not count as an error.
        if record.request.was_cancelled:
            parse_funcs = self.cancelled_parse_funcs
        elif record.valid:
            parse_funcs = self.valid_parse_funcs
        else:
            parse_funcs = self.error_parse_funcs
        # NOTE: Need to parse the record in a loop, as the parse_record function may depend on the results of previous metrics.
        for tag, parse_func in parse_funcs:
            try:
                record_metrics[tag] = parse_func(record, record_metrics)
            except NoMetricValue as e:
                self.trace(
                    lambda tag=tag, e=e: f"No metric value for metric '{tag}': {e!r}"
                )
            except Exception as e:  # noqa: BLE001 - per-metric; skip bad parse and continue
                self.warning(f"Error parsing record for metric '{tag}': {e!r}")
        return record_metrics
