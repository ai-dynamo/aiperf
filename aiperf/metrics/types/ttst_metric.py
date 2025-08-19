# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from aiperf.common.enums import MetricFlags, MetricTimeUnit
from aiperf.common.models import ParsedResponseRecord
from aiperf.metrics import BaseRecordMetric
from aiperf.metrics.metric_dicts import MetricRecordDict


class TTSTMetric(BaseRecordMetric[int]):
    """
    Post-processor for calculating Time to Second Token (TTST) metrics from records.

    Formula:
        TTST = Second Response Timestamp - First Response Timestamp
    """

    tag = "ttst"
    header = "Time to Second Token"
    short_header = "TTST"
    unit = MetricTimeUnit.NANOSECONDS
    display_unit = MetricTimeUnit.MILLISECONDS
    display_order = 200
    flags = MetricFlags.STREAMING_TOKENS_ONLY
    required_metrics = None

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> int:
        """
        This method extracts the timestamps from the first and second response in the given
        RequestRecord object, computes the difference (TTST), and returns the result.

        Raises:
            ValueError: If the record does not have at least two responses, or if the second response is before the first response.
        """

        if len(record.responses) < 2:
            raise ValueError(
                "Record must have at least two responses to calculate TTST."
            )
        if record.responses[1].perf_ns < record.responses[0].perf_ns:
            raise ValueError(
                "Second response timestamp must be greater than or equal to the first response timestamp."
            )

        first_response_ts: int = record.responses[0].perf_ns
        second_response_ts: int = record.responses[1].perf_ns
        return second_response_ts - first_response_ts
