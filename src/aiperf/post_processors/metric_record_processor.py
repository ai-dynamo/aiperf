# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from typing import Any

from aiperf.common.config import UserConfig
from aiperf.common.enums import MetricType
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import ParsedResponseRecord
from aiperf.common.models.record_models import MetricRecordMetadata
from aiperf.common.types import MetricTagT
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.post_processors.base_metrics_processor import BaseMetricsProcessor


class MetricRecordProcessor(BaseMetricsProcessor):
    """Processor for metric records.

    This is the first stage of the metrics processing pipeline, and is done is a distributed manner across multiple service instances.
    It is responsible for streaming the records to the post processor, and computing the metrics from the records.
    It computes metrics from MetricType.RECORD and MetricType.AGGREGATE types."""

    def __init__(
        self,
        user_config: UserConfig,
        **kwargs,
    ) -> None:
        super().__init__(user_config=user_config, **kwargs)

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

    async def process_record(
        self, record: ParsedResponseRecord, metadata: MetricRecordMetadata
    ) -> MetricRecordDict:
        """Process a response record from the inference results parser."""
        record_metrics: MetricRecordDict = MetricRecordDict()
        parse_funcs = self.valid_parse_funcs if record.valid else self.error_parse_funcs
        # NOTE: Need to parse the record in a loop, as the parse_record function may depend on the results of previous metrics.
        for tag, parse_func in parse_funcs:
            try:
                record_metrics[tag] = parse_func(record, record_metrics)
            except NoMetricValue as e:
                self.trace(
                    lambda tag=tag, e=e: f"No metric value for metric '{tag}': {e!r}"
                )
            except Exception as e:
                self.warning(f"Error parsing record for metric '{tag}': {e!r}")
        return record_metrics
