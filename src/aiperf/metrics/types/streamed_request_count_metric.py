# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Visible aggregate counter of requests that streamed on the wire.

This is the user-facing streaming denominator, displayed beside Request Count. It
mirrors ``RequestCountMetric`` but counts only records whose underlying request
streamed (``RequestRecord.streamed``); non-streamed records raise
``NoMetricValue`` and contribute nothing. The hidden per-record gate that streaming
metrics depend on is the companion ``StreamedRequestMetric`` predicate.
"""

from aiperf.common.constants import STREAMED_REQUEST_COUNT_TAG
from aiperf.common.enums import GenericMetricUnit, MetricFlags
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import ParsedResponseRecord
from aiperf.metrics.base_aggregate_counter_metric import BaseAggregateCounterMetric
from aiperf.metrics.metric_dicts import MetricRecordDict


class StreamedRequestCountMetric(BaseAggregateCounterMetric[int]):
    """Counts requests that actually streamed on the wire.

    Formula:
        ```
        Streamed Request Count = Sum(1 if request.streamed else skip)
        ```
    """

    tag = STREAMED_REQUEST_COUNT_TAG
    header = "Streamed Request Count"
    short_header = "Streamed Requests"
    short_header_hide_unit = True
    unit = GenericMetricUnit.REQUESTS
    display_order = 1101
    flags = MetricFlags.STREAMING_TOKENS_ONLY | MetricFlags.NO_INDIVIDUAL_RECORDS
    required_metrics = None

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> int:
        """Return 1 iff the underlying request streamed; else raise to skip it.

        Raises:
            NoMetricValue: If the request did not stream on the wire.
        """
        if not record.request.streamed:
            raise NoMetricValue("request did not stream")
        return 1
