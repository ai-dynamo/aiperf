# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import MetricFlags, MetricOverTimeUnit
from aiperf.common.models.record_models import ParsedResponseRecord
from aiperf.metrics import BaseDerivedMetric, BaseRecordMetric
from aiperf.metrics.metric_dicts import MetricRecordDict, MetricResultsDict
from aiperf.metrics.types.benchmark_duration_metric import BenchmarkDurationMetric
from aiperf.metrics.types.inter_token_latency_metric import InterTokenLatencyMetric
from aiperf.metrics.types.output_sequence_length_metric import (
    TotalOutputSequenceLengthMetric,
)


class OutputTokenThroughputMetric(BaseDerivedMetric[float]):
    """
    Post Processor for calculating Output Token Throughput Metric.

    Formula:
        Output Token Throughput = Benchmark Token Count / Benchmark Duration (seconds)
    """

    tag = "output_token_throughput"
    header = "Output Token Throughput"
    short_header = "Output TPS"
    short_header_hide_unit = True
    unit = MetricOverTimeUnit.TOKENS_PER_SECOND
    display_order = 800
    flags = MetricFlags.PRODUCES_TOKENS_ONLY | MetricFlags.LARGER_IS_BETTER
    required_metrics = {
        TotalOutputSequenceLengthMetric.tag,
        BenchmarkDurationMetric.tag,
    }

    def _derive_value(
        self,
        metric_results: MetricResultsDict,
    ) -> float:
        total_osl = metric_results.get_or_raise(TotalOutputSequenceLengthMetric)
        benchmark_duration_converted = metric_results.get_converted_or_raise(
            BenchmarkDurationMetric,
            self.unit.time_unit,  # type: ignore
        )
        return total_osl / benchmark_duration_converted  # type: ignore


class OutputTokenThroughputPerUserMetric(BaseRecordMetric[float]):
    """
    Post Processor for calculating Output Token Throughput Per User Metric.

    Formula:
        Output Token Throughput Per User = 1 / Inter-Token Latency (seconds)
    """

    tag = "output_token_throughput_per_user"
    header = "Output Token Throughput Per User"
    short_header = "Output TPS/User"
    short_header_hide_unit = True
    unit = MetricOverTimeUnit.TOKENS_PER_SECOND_PER_USER
    display_order = 500
    flags = MetricFlags.STREAMING_TOKENS_ONLY | MetricFlags.LARGER_IS_BETTER
    required_metrics = {
        InterTokenLatencyMetric.tag,
    }

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> float:
        """This method calculates the output token throughput per user by computing the inverse of the inter-token latency."""
        converted_itl = record_metrics.get_converted_or_raise(
            InterTokenLatencyMetric,
            self.unit.time_unit,  # type: ignore
        )
        return 1 / converted_itl
