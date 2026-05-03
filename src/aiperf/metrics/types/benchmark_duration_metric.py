# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import MetricFlags, MetricTimeUnit
from aiperf.metrics.base_derived_metric import BaseDerivedMetric
from aiperf.metrics.metric_dicts import MetricResultsDict
from aiperf.metrics.types.max_response_metric import MaxResponseTimestampMetric
from aiperf.metrics.types.min_request_metric import MinRequestTimestampMetric


class BenchmarkDurationMetric(BaseDerivedMetric[int]):
    """
    This is the duration of the benchmark, from the first request to the last response.

    Formula:
        ```
        Benchmark Duration = Maximum Response Timestamp - Minimum Request Timestamp
        ```
    """

    tag = "benchmark_duration"
    header = "Benchmark Duration"
    short_header = "Duration"
    short_header_hide_unit = True
    unit = MetricTimeUnit.NANOSECONDS
    display_unit = MetricTimeUnit.SECONDS
    flags = MetricFlags.NO_CONSOLE
    required_metrics = {
        MinRequestTimestampMetric.tag,
        MaxResponseTimestampMetric.tag,
    }

    def _derive_value(
        self,
        metric_results: MetricResultsDict,
    ) -> int:
        min_req_time = metric_results.get_or_raise(MinRequestTimestampMetric)
        max_res_time = metric_results.get_or_raise(MaxResponseTimestampMetric)

        if min_req_time >= max_res_time:  # type: ignore
            raise ValueError(
                "Min request must be less than max response to calculate benchmark duration."
            )

        return max_res_time - min_req_time  # type: ignore
