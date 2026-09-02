# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pytest import approx

from aiperf.metrics.metric_dicts import MetricResultsDict
from aiperf.metrics.types.error_request_count import ErrorRequestCountMetric
from aiperf.metrics.types.good_request_count_metric import GoodRequestCountMetric
from aiperf.metrics.types.good_request_fraction_metric import GoodRequestFractionMetric
from aiperf.metrics.types.request_count_metric import RequestCountMetric


class TestGoodRequestFractionMetric:
    def test_basic_fraction_no_errors(self):
        metric = GoodRequestFractionMetric()
        results = MetricResultsDict()
        results[GoodRequestCountMetric.tag] = 18
        results[RequestCountMetric.tag] = 20
        assert metric.derive_value(results) == approx(0.9)

    def test_all_good(self):
        metric = GoodRequestFractionMetric()
        results = MetricResultsDict()
        results[GoodRequestCountMetric.tag] = 20
        results[RequestCountMetric.tag] = 20
        assert metric.derive_value(results) == approx(1.0)

    def test_none_good(self):
        metric = GoodRequestFractionMetric()
        results = MetricResultsDict()
        results[GoodRequestCountMetric.tag] = 0
        results[RequestCountMetric.tag] = 20
        assert metric.derive_value(results) == approx(0.0)

    def test_errors_count_in_denominator(self):
        # 18 good / (20 valid + 5 errors) = 18/25 = 0.72; the failed
        # requests must drag the fraction down.
        metric = GoodRequestFractionMetric()
        results = MetricResultsDict()
        results[GoodRequestCountMetric.tag] = 18
        results[RequestCountMetric.tag] = 20
        results[ErrorRequestCountMetric.tag] = 5
        assert metric.derive_value(results) == approx(18 / 25)

    def test_all_errors_zero_fraction(self):
        metric = GoodRequestFractionMetric()
        results = MetricResultsDict()
        results[ErrorRequestCountMetric.tag] = 10
        assert metric.derive_value(results) == approx(0.0)

    def test_zero_total_returns_zero(self):
        metric = GoodRequestFractionMetric()
        results = MetricResultsDict()
        results[GoodRequestCountMetric.tag] = 0
        results[RequestCountMetric.tag] = 0
        assert metric.derive_value(results) == 0.0

    def test_missing_good_count_reports_zero(self):
        metric = GoodRequestFractionMetric()
        results = MetricResultsDict()
        results[RequestCountMetric.tag] = 20
        assert metric.derive_value(results) == 0.0

    def test_registered_in_metric_registry(self):
        from aiperf.metrics.metric_registry import MetricRegistry

        cls = MetricRegistry.get_class("good_request_fraction")
        assert cls is GoodRequestFractionMetric

    def test_counter_dependencies_are_optional(self):
        assert GoodRequestFractionMetric.required_metrics is None
        assert GoodRequestFractionMetric.optional_metrics == frozenset(
            {
                GoodRequestCountMetric.tag,
                RequestCountMetric.tag,
                ErrorRequestCountMetric.tag,
            }
        )

    def test_registry_orders_optional_counters_before_fraction(self):
        from aiperf.metrics.metric_registry import MetricRegistry

        order = MetricRegistry.create_dependency_order_for(
            [GoodRequestFractionMetric.tag]
        )
        fraction_index = order.index(GoodRequestFractionMetric.tag)
        for tag in GoodRequestFractionMetric.optional_metrics:
            assert order.index(tag) < fraction_index
