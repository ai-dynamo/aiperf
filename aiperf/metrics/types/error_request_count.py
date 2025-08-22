# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import GenericMetricUnit, MetricFlags
from aiperf.metrics.base_aggregate_counter_metric import BaseAggregateCounterMetric


class ErrorRequestCountMetric(BaseAggregateCounterMetric[int]):
    """
    This is the total number of error requests processed by the benchmark.
    It is incremented for each error request.

    Formula:
        ```
        Error Request Count = Sum(Error Requests)
        ```
    """

    tag = "error_request_count"
    header = "Error Request Count"
    short_header = "Error Count"
    short_header_hide_unit = True
    unit = GenericMetricUnit.REQUESTS
    flags = MetricFlags.ERROR_ONLY
    required_metrics = None
