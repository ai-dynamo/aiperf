# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import MetricFlags, MetricOverTimeUnit
from aiperf.common.exceptions import NoMetricValue
from aiperf.metrics import BaseDerivedMetric
from aiperf.metrics.metric_dicts import MetricResultsDict
from aiperf.metrics.types.benchmark_duration_metric import BenchmarkDurationMetric
from aiperf.metrics.types.input_sequence_length_metric import (
    TotalInputSequenceLengthMetric,
)
from aiperf.metrics.types.output_sequence_length_metric import (
    TotalOutputSequenceLengthMetric,
)


class TotalTokenThroughputMetric(BaseDerivedMetric[float]):
    """
    Post Processor for calculating Total Token Throughput Metric.

    Formula:
        Total Token Throughput = (Total Input Tokens + Total Output Tokens) / Benchmark Duration (seconds)
    """

    tag = "total_token_throughput"
    header = "Total Token Throughput"
    short_header = "Total TPS"
    short_header_hide_unit = True
    unit = MetricOverTimeUnit.TOKENS_PER_SECOND
    flags = (
        MetricFlags.PRODUCES_TOKENS_ONLY
        | MetricFlags.LARGER_IS_BETTER
        | MetricFlags.NO_CONSOLE
    )
    required_metrics = {
        TotalInputSequenceLengthMetric.tag,
        TotalOutputSequenceLengthMetric.tag,
        BenchmarkDurationMetric.tag,
    }

    def _derive_value(
        self,
        metric_results: MetricResultsDict,
    ) -> float:
        total_input_tokens = metric_results.get_or_raise(TotalInputSequenceLengthMetric)
        total_output_tokens = metric_results.get_or_raise(
            TotalOutputSequenceLengthMetric
        )
        benchmark_duration_converted = metric_results.get_converted_or_raise(
            BenchmarkDurationMetric,
            self.unit.time_unit,  # type: ignore
        )
        if benchmark_duration_converted == 0:
            raise NoMetricValue(
                "Benchmark duration is zero, cannot calculate total token throughput metric"
            )
        return (total_input_tokens + total_output_tokens) / benchmark_duration_converted  # type: ignore
