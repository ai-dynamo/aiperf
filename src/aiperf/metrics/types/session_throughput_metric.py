# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import MetricFlags, MetricOverTimeUnit
from aiperf.common.exceptions import NoMetricValue
from aiperf.metrics.base_derived_metric import BaseDerivedMetric
from aiperf.metrics.metric_dicts import MetricResultsDict
from aiperf.metrics.types.benchmark_duration_metric import BenchmarkDurationMetric
from aiperf.metrics.types.session_count_metric import SessionCountMetric


class SessionThroughputMetric(BaseDerivedMetric[float]):
    """
    Post Processor for calculating completed-session throughput.

    This is the sustained rate at which whole multi-turn sessions complete over
    the benchmark window. Under saturation it is the metric of interest for
    "sessions per hour at max throughput" runs.

    Formula:
        ```
        Session Throughput = Session Count / Benchmark Duration (hours)
        ```
    """

    tag = "session_throughput"
    header = "Session Throughput"
    short_header = "Sessions/hr"
    short_header_hide_unit = True
    unit = MetricOverTimeUnit.SESSIONS_PER_HOUR
    display_order = 905
    flags = MetricFlags.LARGER_IS_BETTER
    required_metrics = {
        SessionCountMetric.tag,
        BenchmarkDurationMetric.tag,
    }

    def _derive_value(
        self,
        metric_results: MetricResultsDict,
    ) -> float:
        session_count = metric_results.get_or_raise(SessionCountMetric)
        benchmark_duration_hours = metric_results.get_converted_or_raise(
            BenchmarkDurationMetric,
            self.unit.time_unit,  # type: ignore
        )
        if benchmark_duration_hours == 0:
            raise NoMetricValue(
                "Benchmark duration cannot be zero for throughput calculation"
            )
        return session_count / benchmark_duration_hours  # type: ignore
