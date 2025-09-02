# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import MetricFlags, MetricTimeUnit
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import ParsedResponseRecord
from aiperf.metrics.base_record_metric import BaseRecordMetric
from aiperf.metrics.metric_dicts import MetricRecordDict


class CreditDropLatencyMetric(BaseRecordMetric[int]):
    """
    Post-processor for calculating Credit Drop Latency metrics from records. This is an internal metric that is
    intended to be used for debugging and performance analysis of the AIPerf internal system.

    It exposes how long it took from when a credit was dropped, to when the actual request was sent. This will
    include the time it took to query the DatasetManager to get the Turn, as well as the time it took to format
    the request.

    Formula:
        Credit Drop Latency = Request Start Time - Credit Drop Received Time
    """

    tag = "credit_drop_latency"
    header = "Credit Drop Latency"
    short_header = "Credit Latency"
    unit = MetricTimeUnit.NANOSECONDS
    display_unit = MetricTimeUnit.MILLISECONDS
    flags = MetricFlags.INTERNAL
    required_metrics = None

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> int:
        """
        This method extracts the credit drop latency from the record and returns it.

        Raises:
            ValueError: If the record does not include a credit drop latency.
        """
        if not record.request.credit_drop_latency:
            raise NoMetricValue("Credit Drop Latency is not included in the record.")

        return record.request.credit_drop_latency
