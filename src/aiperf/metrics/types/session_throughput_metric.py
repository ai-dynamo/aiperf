# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import MetricFlags, MetricOverTimeUnit
from aiperf.metrics.base_derived_metric import BaseDerivedMetric
from aiperf.metrics.metric_dicts import MetricResultsDict
from aiperf.metrics.types.benchmark_duration_metric import BenchmarkDurationMetric
from aiperf.metrics.types.session_count_metric import SessionCountMetric


class SessionThroughputMetric(BaseDerivedMetric[float]):
    """Calculate successfully completed root sessions per hour."""

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

    def _derive_value(self, metric_results: MetricResultsDict) -> float:
        session_count = metric_results.get_or_raise(SessionCountMetric)
        duration = metric_results.observation_duration(self.unit.time_unit)
        return session_count / duration  # type: ignore[operator]
