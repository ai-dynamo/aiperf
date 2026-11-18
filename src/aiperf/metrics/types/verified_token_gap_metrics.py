# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import (
    GenericMetricUnit,
    MetricConsoleGroup,
    MetricFlags,
    MetricTimeUnit,
)
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import ParsedResponseRecord
from aiperf.metrics import BaseRecordMetric
from aiperf.metrics.metric_dicts import MetricRecordDict


class VerifiedTokenGapLatencyMetric(BaseRecordMetric[list[int]]):
    """Pooled delivery gaps with one verified output token at each end."""

    tag = "verified_token_gap_latency"
    header = "Verified Token Gap Latency"
    short_header = "Token Gap"
    unit = MetricTimeUnit.NANOSECONDS
    display_unit = MetricTimeUnit.MILLISECONDS
    flags = (
        MetricFlags.STREAMING_TOKENS_ONLY
        | MetricFlags.NO_INDIVIDUAL_RECORDS
        | MetricFlags.VERIFIED_TOKEN_GAPS_ONLY
    )
    console_group = MetricConsoleGroup.NONE

    def _parse_record(
        self, record: ParsedResponseRecord, record_metrics: MetricRecordDict
    ) -> list[int]:
        if not record.verified_token_gaps_ns:
            raise NoMetricValue("No verified token gaps")
        return record.verified_token_gaps_ns


class VerifiedTokenGapRequestCoverageMetric(BaseRecordMetric[int]):
    """One for requests with at least one verified gap, zero for other eligible requests."""

    tag = "verified_token_gap_request_coverage"
    header = "Verified Token Gap Request Coverage"
    short_header = "Token Gap Coverage"
    unit = GenericMetricUnit.RATIO
    flags = VerifiedTokenGapLatencyMetric.flags
    console_group = MetricConsoleGroup.NONE

    def _parse_record(
        self, record: ParsedResponseRecord, record_metrics: MetricRecordDict
    ) -> int:
        if record.verified_token_gaps_ns is None:
            raise NoMetricValue("Request is outside verified token gap coverage")
        return int(bool(record.verified_token_gaps_ns))
