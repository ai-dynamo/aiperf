# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import MetricFlags, MetricOverTimeUnit
from aiperf.metrics import BaseDerivedMetric
from aiperf.metrics.metric_dicts import MetricResultsDict
from aiperf.metrics.types.benchmark_duration_metric import BenchmarkDurationMetric
from aiperf.metrics.types.input_sequence_length_metric import (
    TotalInputSequenceLengthMetric,
)


class InputTokenThroughputMetric(BaseDerivedMetric[float]):
    """
    System-level prefill throughput. Mirrors ``OutputTokenThroughputMetric``
    on the input side so the realtime stats and assessment blocks can show
    both halves of ``total_token_throughput`` separately. Useful for
    long-context agentic workloads where prefill dominates: input throughput
    can be 100x output throughput, and tracking them separately is the only
    way to see prefill saturation.

    Formula:
        Input Token Throughput = Total Input Tokens / Benchmark Duration (seconds)
    """

    tag = "input_token_throughput"
    header = "Input Token Throughput"
    short_header = "Input TPS"
    short_header_hide_unit = True
    unit = MetricOverTimeUnit.TOKENS_PER_SECOND
    display_order = 805
    flags = MetricFlags.LARGER_IS_BETTER
    # Default console_group (DEFAULT) so the metric flows through
    # filter_display_metrics into the realtime stats block. Setting NONE
    # would drop it as a "hidden" metric — that's why
    # ``total_token_throughput`` doesn't appear in the realtime line and
    # always rendered ``-``. Mirrors ``OutputTokenThroughputMetric`` which
    # also uses the default group.
    required_metrics = {
        TotalInputSequenceLengthMetric.tag,
        BenchmarkDurationMetric.tag,
    }

    def _derive_value(
        self,
        metric_results: MetricResultsDict,
    ) -> float:
        total_isl = metric_results.get_or_raise(TotalInputSequenceLengthMetric)
        duration = metric_results.observation_duration(self.unit.time_unit)  # type: ignore
        return total_isl / duration  # type: ignore
