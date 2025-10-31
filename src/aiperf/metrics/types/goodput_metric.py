# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import MetricFlags, MetricOverTimeUnit
from aiperf.common.exceptions import NoMetricValue
from aiperf.metrics.base_derived_metric import BaseDerivedMetric
from aiperf.metrics.metric_dicts import MetricResultsDict
from aiperf.metrics.types.benchmark_duration_metric import BenchmarkDurationMetric
from aiperf.metrics.types.good_request_count_metric import GoodRequestCountMetric


class GoodputMetric(BaseDerivedMetric[float]):
    """
    Postprocessor for calculating the Goodput metric.

    Formula:
    Goodput = Good request count / Benchmark Duration (seconds)
    """

    tag = "goodput"
    header = "Goodput"
    short_header = "Goodput"
    short_header_hide_unit = True
    unit = MetricOverTimeUnit.REQUESTS_PER_SECOND
    display_order = 1000
    flags = MetricFlags.GOODPUT
    required_metrics = {GoodRequestCountMetric.tag, BenchmarkDurationMetric.tag}

    def _derive_value(self, metric_results: MetricResultsDict) -> float:
        tag = GoodRequestCountMetric.tag
        if tag not in metric_results:
            raise NoMetricValue(f"Metric '{tag}' is not available for the run.")
        good_request_count = metric_results[tag]

        benchmark_duration_converted = metric_results.get_converted_or_raise(
            BenchmarkDurationMetric,
            self.unit.time_unit,  # type: ignore
        )
        return good_request_count / benchmark_duration_converted  # type: ignore
