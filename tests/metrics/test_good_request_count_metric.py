# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.exceptions import MetricTypeError
from aiperf.metrics.metric_registry import MetricRegistry
from aiperf.metrics.types.good_request_count_metric import GoodRequestCountMetric
from aiperf.metrics.types.request_latency_metric import RequestLatencyMetric
from tests.metrics.conftest import create_record, run_simple_metrics_pipeline


class TestGoodRequestCountMetric:
    def setup_method(self):
        GoodRequestCountMetric.set_slos({})

    def test_unknown_tag_raises(self, monkeypatch):
        def mock_get_class(tag):
            raise MetricTypeError(f"Metric class with tag '{tag}' not found")

        monkeypatch.setattr(MetricRegistry, "get_class", mock_get_class)

        with pytest.raises(ValueError, match="Unknown metric tag"):
            GoodRequestCountMetric.set_slos({"does_not_exist": 123})

    def test_set_slos_populates_required_metrics(self):
        GoodRequestCountMetric.set_slos(
            {
                RequestLatencyMetric.tag: 250.0,
            }
        )
        assert GoodRequestCountMetric.required_metrics == {RequestLatencyMetric.tag}

    def test_counts_good_requests(self):
        GoodRequestCountMetric.set_slos({RequestLatencyMetric.tag: 250.0})

        records = [
            create_record(start_ns=0, responses=[100_000_000]),  # 100ms -> good
            create_record(
                start_ns=100_000_000, responses=[400_000_000]
            ),  # 300ms -> bad
            create_record(
                start_ns=200_000_000, responses=[450_000_000]
            ),  # 250ms -> good
        ]

        metrics = run_simple_metrics_pipeline(
            records,
            RequestLatencyMetric.tag,
            GoodRequestCountMetric.tag,
        )

        assert metrics[GoodRequestCountMetric.tag] == 2.0

    def test_no_slos_configured_returns_zero(self):
        records = [create_record(start_ns=0, responses=[100_000_000])]
        metrics = run_simple_metrics_pipeline(
            records,
            RequestLatencyMetric.tag,
            GoodRequestCountMetric.tag,
        )
        assert metrics[GoodRequestCountMetric.tag] == 0.0
