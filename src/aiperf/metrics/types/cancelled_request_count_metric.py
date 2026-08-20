# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import GenericMetricUnit, MetricConsoleGroup, MetricFlags
from aiperf.metrics.base_aggregate_counter_metric import BaseAggregateCounterMetric


class CancelledRequestCountMetric(BaseAggregateCounterMetric[int]):
    """Total number of requests deliberately cancelled by the client."""

    tag = "cancelled_request_count"
    header = "Cancelled Request Count"
    short_header = "Cancelled Count"
    short_header_hide_unit = True
    unit = GenericMetricUnit.REQUESTS
    display_order = 1076
    flags = MetricFlags.CANCELLED_ONLY | MetricFlags.NO_INDIVIDUAL_RECORDS
    console_group = MetricConsoleGroup.NONE
    required_metrics = None
