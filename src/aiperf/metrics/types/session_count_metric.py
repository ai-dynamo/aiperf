# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import GenericMetricUnit, MetricFlags
from aiperf.common.models import ParsedResponseRecord
from aiperf.metrics.base_aggregate_counter_metric import BaseAggregateCounterMetric
from aiperf.metrics.metric_dicts import MetricRecordDict


class SessionCountMetric(BaseAggregateCounterMetric[int]):
    """Count successfully completed root sessions.

    A session completes when its final root-conversation turn returns a valid
    response. DAG child records belong to the root session and do not increment
    this counter.
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
        """Return one for a final root turn, otherwise zero."""
        ctx = record.request.request_info
        return int(ctx is not None and ctx.is_final_turn and ctx.agent_depth == 0)
