# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import MetricFlags, MetricOverTimeUnit
from aiperf.metrics.base_derived_metric import BaseDerivedMetric
from aiperf.metrics.metric_dicts import MetricResultsDict
from aiperf.metrics.types.benchmark_token_count import BenchmarkTokenCountMetric
from aiperf.metrics.types.inter_token_latency import InterTokenLatencyMetric


class OutputTokenThroughputPerUserMetric(BaseDerivedMetric[float]):
    """
    Post Processor for calculating Output Token Throughput per user metrics from records.

    Formula:
        Output Token Throughput Per User = Sum(Inter-Token Latencies) / Benchmark Token Count
    """

    tag = "output_token_throughput_per_user"
    header = "Output Token Throughput Per User"
    unit = MetricOverTimeUnit.TOKENS_PER_SECOND_PER_USER
    flags = MetricFlags.STREAMING_TOKENS_ONLY | MetricFlags.LARGER_IS_BETTER
    required_metrics = {
        InterTokenLatencyMetric.tag,
        BenchmarkTokenCountMetric.tag,
    }

    def _derive_value(
        self,
        metric_results: MetricResultsDict,
    ) -> float:
        inter_token_latencies = metric_results[InterTokenLatencyMetric.tag]
        benchmark_token_count = metric_results[BenchmarkTokenCountMetric.tag]
        return sum(inter_token_latencies) / benchmark_token_count  # type: ignore
