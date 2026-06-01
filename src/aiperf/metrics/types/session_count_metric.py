# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import GenericMetricUnit, MetricFlags
from aiperf.common.models import ParsedResponseRecord
from aiperf.metrics.base_aggregate_counter_metric import BaseAggregateCounterMetric
from aiperf.metrics.metric_dicts import MetricRecordDict


class SessionCountMetric(BaseAggregateCounterMetric[int]):
    """
    The total number of completed sessions (multi-turn conversations).

    A session completes when the final turn of a root conversation returns.
    Only valid records reach this metric (invalid/errored records are converted
    to errors and excluded upstream), so this is a goodput count: it increments
    for each valid record whose ``is_final_turn`` is set and whose
    ``agent_depth`` is 0. DAG children (``agent_depth > 0``) belong to their
    parent's session and are not counted as separate sessions, matching the
    timing-side ``CreditCounter.completed_sessions`` semantics.

    Formula:
        ```
        Session Count = Sum(valid records where is_final_turn and agent_depth == 0)
        ```
    """

    tag = "session_count"
    header = "Session Count"
    short_header = "Sessions"
    short_header_hide_unit = True
    unit = GenericMetricUnit.SESSIONS
    display_order = 1105
    flags = MetricFlags.LARGER_IS_BETTER | MetricFlags.NO_INDIVIDUAL_RECORDS
    required_metrics = None

    def _parse_record(
        self, record: ParsedResponseRecord, record_metrics: MetricRecordDict
    ) -> int:
        """Return 1 for the final turn of a root session, else 0."""
        ctx = record.request.request_info
        if ctx is None or not ctx.is_final_turn or ctx.agent_depth != 0:
            return 0
        return 1
