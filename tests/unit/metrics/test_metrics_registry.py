# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.exceptions import MetricTypeError
from aiperf.metrics.metric_registry import MetricRegistry
from aiperf.metrics.types.benchmark_duration_metric import BenchmarkDurationMetric
from aiperf.metrics.types.inter_token_latency_metric import InterTokenLatencyMetric
from aiperf.metrics.types.request_count_metric import RequestCountMetric
from aiperf.metrics.types.request_throughput_metric import RequestThroughputMetric


class TestMetricRegistry:
    def test_create_dependency_order_for(self):
        """Test that the dependency order is created correctly."""
        order = MetricRegistry.create_dependency_order_for(tags=[])
        assert order == []

    def test_create_dependency_order_for_with_tags(self):
        """Test that the dependency order is created correctly with all dependencies."""
        order = MetricRegistry.create_dependency_order_for(
            [
                RequestThroughputMetric.tag,
                RequestCountMetric.tag,
                BenchmarkDurationMetric.tag,
            ]
        )
        # The order should include all recursive dependencies
        # RequestThroughputMetric depends on benchmark_duration and request_count
        # BenchmarkDurationMetric depends on min_request_timestamp and max_response_timestamp
        # Those timestamps depend on request_latency
        assert RequestCountMetric.tag in order
        assert BenchmarkDurationMetric.tag in order
        assert RequestThroughputMetric.tag in order
        # Dependencies should come before dependents
        assert order.index(RequestCountMetric.tag) < order.index(
            RequestThroughputMetric.tag
        )
        assert order.index(BenchmarkDurationMetric.tag) < order.index(
            RequestThroughputMetric.tag
        )

    def test_create_dependency_order_includes_all_dependencies(self):
        """Test that the dependency order includes all recursive dependencies."""
        assert len(RequestThroughputMetric.required_metrics) > 0, (
            "RequestThroughputMetric must have dependencies, or this test is invalid"
        )
        order = MetricRegistry.create_dependency_order_for(
            tags=[RequestThroughputMetric.tag]
        )
        # Should include RequestThroughputMetric and all its recursive dependencies
        assert RequestThroughputMetric.tag in order
        for dep in RequestThroughputMetric.required_metrics:
            assert dep in order, f"Dependency {dep} should be in the order"
            # Dependencies should come before the dependent
            assert order.index(dep) < order.index(RequestThroughputMetric.tag)

    def test_create_dependency_order_for_circular_dependency(self):
        """Test that a circular dependency raises an error."""
        # Create a circular dependency by adding a dependency on RequestThroughputMetric to InterTokenLatencyMetric
        itl_original_required_metrics = InterTokenLatencyMetric.required_metrics
        rtt_original_required_metrics = RequestThroughputMetric.required_metrics
        InterTokenLatencyMetric.required_metrics = {RequestThroughputMetric.tag}
        RequestThroughputMetric.required_metrics = {InterTokenLatencyMetric.tag}

        try:
            with pytest.raises(
                MetricTypeError, match="Circular dependency detected among metrics"
            ):
                MetricRegistry.create_dependency_order_for(
                    [InterTokenLatencyMetric.tag, RequestThroughputMetric.tag]
                )
        finally:
            # Put it back to the original values
            InterTokenLatencyMetric.required_metrics = itl_original_required_metrics
            RequestThroughputMetric.required_metrics = rtt_original_required_metrics

    def test_get_class_or_none_returns_class_for_known_tag(self):
        """get_class_or_none returns the metric class for a registered tag."""
        cls = MetricRegistry.get_class_or_none(RequestCountMetric.tag)
        assert cls is RequestCountMetric

    def test_get_class_or_none_returns_none_for_unknown_tag(self):
        """get_class_or_none returns None — no exception — for an unknown tag.

        This is the contract the JSON exporter relies on: tags from other
        registries (e.g. GPU telemetry) must not raise here.
        """
        assert MetricRegistry.get_class_or_none("definitely_not_a_real_metric") is None
        assert MetricRegistry.get_class_or_none("gpu_power_usage") is None

    def test_get_class_or_none_matches_get_class_for_known_tags(self):
        """For registered tags, both lookups return the same class."""
        for tag in (
            RequestCountMetric.tag,
            RequestThroughputMetric.tag,
            BenchmarkDurationMetric.tag,
            InterTokenLatencyMetric.tag,
        ):
            assert MetricRegistry.get_class_or_none(tag) is MetricRegistry.get_class(
                tag
            )
