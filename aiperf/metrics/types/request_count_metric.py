# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import GenericMetricUnit, MetricFlags
from aiperf.metrics.base_aggregate_counter_metric import BaseAggregateCounterMetric


class RequestCountMetric(BaseAggregateCounterMetric[int]):
    """
    This is the total number of valid requests processed by the benchmark.
    It is incremented for each valid request.

    Formula:
        ```
        Request Count = Sum(Valid Requests)
        ```
    """

    tag = "request_count"
    header = "Request Count"
    short_header = "Requests"
    short_header_hide_unit = True
    unit = GenericMetricUnit.REQUESTS
    display_order = 1100
    flags = MetricFlags.LARGER_IS_BETTER | MetricFlags.NO_INDIVIDUAL_RECORDS
    required_metrics = None
