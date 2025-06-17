# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

from aiperf.common.enums import MetricType, PostProcessorType
from aiperf.common.factories import PostProcessorFactory
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

    def process(self, records: list) -> None:
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
                elif metric.type == MetricType.METRIC_OF_BOTH:
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
            elif metric.type == MetricType.METRIC_OF_BOTH:
                metric.update_value(
                    record=record, metrics={m.tag: m for m in self._metrics}
                )

    def get_metrics_summary(self) -> dict:
        metrics_summary = {}
        for metric in self._metrics:
            metrics_summary[metric.tag] = metric.values()

        return metrics_summary
