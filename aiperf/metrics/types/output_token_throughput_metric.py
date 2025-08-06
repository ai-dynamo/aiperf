# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import MetricFlags, MetricOverTimeUnit
from aiperf.metrics.base_derived_metric import BaseDerivedMetric
from aiperf.metrics.metric_dicts import MetricResultsDict
from aiperf.metrics.types.benchmark_duration_metric import BenchmarkDurationMetric
from aiperf.metrics.types.benchmark_token_count import BenchmarkTokenCountMetric


class OutputTokenThroughputMetric(BaseDerivedMetric[float]):
    """
    Post Processor for calculating Output Token Throughput Metric.

    Formula:
        Output Token Throughput = Benchmark Token Count / Benchmark Duration (seconds)
    """

    tag = "output_token_throughput"
    header = "Output Token Throughput"
    unit = MetricOverTimeUnit.TOKENS_PER_SECOND
    flags = MetricFlags.PRODUCES_TOKENS_ONLY | MetricFlags.LARGER_IS_BETTER
    required_metrics = {
        BenchmarkTokenCountMetric.tag,
        BenchmarkDurationMetric.tag,
    }

    def _derive_value(
        self,
        metric_results: MetricResultsDict,
    ) -> float:
        benchmark_token_count = metric_results[BenchmarkTokenCountMetric.tag]
        benchmark_duration = metric_results[BenchmarkDurationMetric.tag]
        if benchmark_duration is None or benchmark_duration == 0:
            raise ValueError("Benchmark duration is not available.")

        benchmark_duration_converted = metric_results.get_converted(  # type: ignore
            BenchmarkDurationMetric,
            self.unit.time_unit,  # type: ignore
        )
        return benchmark_token_count / benchmark_duration_converted  # type: ignore
