# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from aiperf.common.enums import (
    AggregationKind,
    MetricConsoleGroup,
    MetricFlags,
    MetricTimeUnit,
)
from aiperf.common.models import ParsedResponseRecord
from aiperf.metrics import BaseAggregateMetric
from aiperf.metrics.metric_dicts import MetricRecordDict


class MinRequestTimestampMetric(BaseAggregateMetric[int]):
    """
    Post-processor for calculating the minimum request time stamp metric from records.

    Formula:
        Minimum Request Timestamp = Min(Request Timestamps)
    """

    tag = "min_request_timestamp"
    header = "Minimum Request Timestamp"
    short_header = "Min Req"
    short_header_hide_unit = True
    unit = MetricTimeUnit.NANOSECONDS
    flags = MetricFlags.NO_INDIVIDUAL_RECORDS | MetricFlags.INTERNAL
    console_group = MetricConsoleGroup.NONE
    aggregation_kind = AggregationKind.MIN
    required_metrics = None

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> int:
        """Return the request timestamp.

        The accumulator folds these to the run minimum via ``aggregation_kind``.
        """
        # NOTE: Use the request timestamp_ns, not the start_perf_ns, because we want wall-clock timestamps,
        return record.timestamp_ns
