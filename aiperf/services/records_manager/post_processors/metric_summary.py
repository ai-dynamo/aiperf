# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

from aiperf.common.enums import PostProcessorType
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
                metric.add_record(record)

    def get_metrics(self) -> dict:
        metrics_summary = {}
        for metric in self._metrics:
            metrics_summary[metric.tag] = metric.get_metrics()

        return metrics_summary
