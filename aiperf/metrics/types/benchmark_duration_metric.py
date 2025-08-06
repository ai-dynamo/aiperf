# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import MetricFlags, MetricTimeUnit
from aiperf.metrics.base_derived_metric import BaseDerivedMetric
from aiperf.metrics.metric_dicts import MetricResultsDict
from aiperf.metrics.types.max_response_metric import MaxResponseTimestampMetric
from aiperf.metrics.types.min_request_metric import MinRequestTimestampMetric


class BenchmarkDurationMetric(BaseDerivedMetric[int]):
    """
    Post-processor for calculating the Benchmark Duration metric.

    Formula:
        Benchmark Duration = Maximum Response Timestamp - Minimum Request Timestamp
    """

    tag = "benchmark_duration"
    header = "Benchmark Duration"
    unit = MetricTimeUnit.NANOSECONDS
    display_unit = MetricTimeUnit.SECONDS
    flags = MetricFlags.HIDDEN
    required_metrics = {
        MinRequestTimestampMetric.tag,
        MaxResponseTimestampMetric.tag,
    }

    def _derive_value(
        self,
        metric_results: MetricResultsDict,
    ) -> int:
        min_req_time = metric_results[MinRequestTimestampMetric.tag]
        max_res_time = metric_results[MaxResponseTimestampMetric.tag]

        if min_req_time is None or max_res_time is None:
            raise ValueError(
                "Min request and max response are required to calculate benchmark duration."
            )

        if min_req_time >= max_res_time:  # type: ignore
            raise ValueError(
                "Min request must be less than max response to calculate benchmark duration."
            )

        return max_res_time - min_req_time  # type: ignore
