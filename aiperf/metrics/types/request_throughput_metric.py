# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import MetricOverTimeUnit
from aiperf.common.enums.metric_enums import MetricFlags
from aiperf.metrics.base_derived_metric import BaseDerivedMetric
from aiperf.metrics.metric_dicts import MetricResultsDict
from aiperf.metrics.types.benchmark_duration import BenchmarkDurationMetric
from aiperf.metrics.types.valid_request_count import ValidRequestCountMetric


class RequestThroughputMetric(BaseDerivedMetric[float]):
    """
    Post Processor for calculating Request throughput metrics from records.

    Formula:
        Request Throughput = Valid Request Count / Benchmark Duration (seconds)
    """

    tag = "request_throughput"
    header = "Request Throughput"
    unit = MetricOverTimeUnit.REQUESTS_PER_SECOND
    flags = MetricFlags.LARGER_IS_BETTER
    required_metrics = {
        ValidRequestCountMetric.tag,
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

        valid_request_count = metric_results[ValidRequestCountMetric.tag]
        if valid_request_count is None:
            raise ValueError(
                "Valid request count is required to calculate request throughput."
            )

        benchmark_duration_converted = metric_results.get_converted(  # type: ignore
            BenchmarkDurationMetric,
            self.unit.time_unit,  # type: ignore
        )
        return valid_request_count / benchmark_duration_converted  # type: ignore
