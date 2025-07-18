# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

import pandas as pd

from aiperf.common.constants import NANOS_PER_MILLIS
from aiperf.common.enums import MetricType, PostProcessorType
from aiperf.common.factories import PostProcessorFactory
from aiperf.common.models import MetricResult, ParsedResponseRecord
from aiperf.services.records_manager.metrics.base_metric import BaseMetric

logger = logging.getLogger(__name__)


@PostProcessorFactory.register(PostProcessorType.METRIC_SUMMARY)
class MetricSummary:
    """
    MetricSummary is a post-processor that generates a summary of metrics from the records.
    It processes the records to extract relevant metrics and returns them in a structured format.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.debug("Initializing MetricSummary post-processor")

        self._metrics = []
        for metric_cls in BaseMetric.get_all().values():
            self._metrics.append(metric_cls())

    def process(self, records: list[ParsedResponseRecord]) -> None:
        """
        Process the records to generate a summary of metrics.

        :param records: The input records to be processed.
        :return: A dictionary containing the summarized metrics.
        """
        for record in records:
            for metric in self._metrics:
                if metric.type == MetricType.METRIC_OF_RECORDS:
                    metric.update_value(record=record)
            for metric in self._metrics:
                if metric.type == MetricType.METRIC_OF_METRICS:
                    metric.update_value(metrics={m.tag: m for m in self._metrics})
            for metric in self._metrics:
                if metric.type == MetricType.METRIC_OF_BOTH:
                    metric.update_value(
                        record=record, metrics={m.tag: m for m in self._metrics}
                    )

        # TODO: Fix this after we add support for dependencies
        # between metrics of metrics
        # This is a workaround to ensure that metrics of metrics
        # are updated after all records are processed
        for metric in self._metrics:
            if metric.type == MetricType.METRIC_OF_METRICS:
                metric.update_value(metrics={m.tag: m for m in self._metrics})
        for metric in self._metrics:
            if metric.type == MetricType.METRIC_OF_BOTH:
                metric.update_value(
                    # TODO: Where does this `record` value come from? Is this wrong?
                    record=record,
                    metrics={m.tag: m for m in self._metrics},
                )

    def get_metrics_summary(self) -> list[MetricResult]:
        metrics_summary = []

        df = pd.DataFrame({metric.tag: metric.values() for metric in self._metrics})

        for metric in self._metrics:
            res: MetricResult = record_from_dataframe(df, metric)
            metrics_summary.append(res)
        return metrics_summary


def record_from_dataframe(
    df: pd.DataFrame,
    metric: BaseMetric,
) -> MetricResult:
    """Create a Record from a DataFrame."""
    column = df[metric.tag]
    return MetricResult(
        tag=metric.tag,
        header=metric.header,
        unit=metric.unit.name,
        avg=column.mean() / NANOS_PER_MILLIS,
        min=column.min() / NANOS_PER_MILLIS,
        max=column.max() / NANOS_PER_MILLIS,
        p1=column.quantile(0.01) / NANOS_PER_MILLIS,
        p5=column.quantile(0.05) / NANOS_PER_MILLIS,
        p25=column.quantile(0.25) / NANOS_PER_MILLIS,
        p50=column.quantile(0.50) / NANOS_PER_MILLIS,
        p75=column.quantile(0.75) / NANOS_PER_MILLIS,
        p90=column.quantile(0.90) / NANOS_PER_MILLIS,
        p95=column.quantile(0.95) / NANOS_PER_MILLIS,
        p99=column.quantile(0.99) / NANOS_PER_MILLIS,
        std=column.std() / NANOS_PER_MILLIS,
        count=int(column.count()),
        streaming_only=metric.streaming_only,
    )
