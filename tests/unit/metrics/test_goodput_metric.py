# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pytest import approx

from aiperf.metrics.metric_dicts import MetricResultsDict
from aiperf.metrics.types.benchmark_duration_metric import BenchmarkDurationMetric
from aiperf.metrics.types.good_request_count_metric import GoodRequestCountMetric
from aiperf.metrics.types.goodput_metric import GoodputMetric


class TestGoodputMetric:
    def test_goodput(self):
        metric = GoodputMetric()

        metric_results = MetricResultsDict()
        metric_results[GoodRequestCountMetric.tag] = 15
        metric_results[BenchmarkDurationMetric.tag] = 1_000_000_000

        result = metric.derive_value(metric_results)
        assert result == approx(15.0)

    def test_goodput_zero_count(self):
        metric = GoodputMetric()

        metric_results = MetricResultsDict()
        metric_results[GoodRequestCountMetric.tag] = 0
        metric_results[BenchmarkDurationMetric.tag] = 2_000_000_000

        result = metric.derive_value(metric_results)
        assert result == 0.0
