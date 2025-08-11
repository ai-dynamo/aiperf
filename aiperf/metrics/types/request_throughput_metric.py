# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import MetricFlags, MetricOverTimeUnit
from aiperf.metrics.base_derived_metric import BaseDerivedMetric
from aiperf.metrics.metric_dicts import MetricResultsDict
from aiperf.metrics.types.benchmark_duration_metric import BenchmarkDurationMetric
from aiperf.metrics.types.request_count_metric import RequestCountMetric


class RequestThroughputMetric(BaseDerivedMetric[float]):
    """
    Post Processor for calculating Request throughput metrics from records.

    Formula:
        Request Throughput = Valid Request Count / Benchmark Duration (seconds)
    """

    tag = "request_throughput"
    header = "Request Throughput"
    unit = MetricOverTimeUnit.REQUESTS_PER_SECOND
    display_order = 900
    flags = MetricFlags.LARGER_IS_BETTER
    required_metrics = {
        RequestCountMetric.tag,
        BenchmarkDurationMetric.tag,
    }

    def _derive_value(
        self,
        metric_results: MetricResultsDict,
    ) -> float:
        benchmark_duration = metric_results[BenchmarkDurationMetric.tag]
        if benchmark_duration is None or benchmark_duration == 0:
            raise ValueError(
                "Benchmark duration is required and must be greater than 0 to calculate request throughput."
            )

        request_count = metric_results[RequestCountMetric.tag]
        if request_count is None:
            raise ValueError(
                "Request count is required to calculate request throughput."
            )

        benchmark_duration_converted = metric_results.get_converted(  # type: ignore
            BenchmarkDurationMetric,
            self.unit.time_unit,  # type: ignore
        )
        return request_count / benchmark_duration_converted  # type: ignore
