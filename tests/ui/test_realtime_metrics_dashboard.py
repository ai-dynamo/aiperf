# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.config import ServiceConfig
from aiperf.common.config.dev_config import DeveloperConfig
from aiperf.common.models import MetricResult
from aiperf.metrics.types.benchmark_duration_metric import BenchmarkDurationMetric
from aiperf.metrics.types.error_request_count import ErrorRequestCountMetric
from aiperf.metrics.types.inter_token_latency_metric import InterTokenLatencyMetric
from aiperf.metrics.types.output_token_count import (
    OutputTokenCountMetric,
)
from aiperf.metrics.types.request_latency_metric import RequestLatencyMetric
from aiperf.metrics.types.thinking_efficiency_metrics import ThinkingEfficiencyMetric
from aiperf.metrics.types.ttft_metric import TTFTMetric
from aiperf.ui.dashboard.realtime_metrics_dashboard import RealtimeMetricsTable


@pytest.fixture
def service_config_show_internal_false():
    return ServiceConfig(developer=DeveloperConfig(show_internal_metrics=False))


@pytest.fixture
def service_config_show_internal_true():
    return ServiceConfig(developer=DeveloperConfig(show_internal_metrics=True))


class TestRealtimeMetricsTable:
    @pytest.mark.parametrize(
        "metric_tag, show_internal, should_skip",
        [
            # ERROR_ONLY metrics - always skipped
            (ErrorRequestCountMetric.tag, False, True),
            (ErrorRequestCountMetric.tag, True, True),
            # EXPERIMENTAL metrics - skipped if show_internal_metrics is False
            (ThinkingEfficiencyMetric.tag, False, True),
            (ThinkingEfficiencyMetric.tag, True, False),
            # NO_CONSOLE metrics - skipped if show_internal_metrics is False
            (BenchmarkDurationMetric.tag, False, True),
            (BenchmarkDurationMetric.tag, True, False),
            (OutputTokenCountMetric.tag, False, True),
            (OutputTokenCountMetric.tag, True, False),
            # Normal metrics - always shown
            (RequestLatencyMetric.tag, False, False),
            (RequestLatencyMetric.tag, True, False),
            (TTFTMetric.tag, False, False),
            (TTFTMetric.tag, True, False),
            (InterTokenLatencyMetric.tag, False, False),
            (InterTokenLatencyMetric.tag, True, False),
        ],
    )  # fmt: skip
    def test_should_skip_logic_with_real_metrics(
        self, metric_tag, show_internal, should_skip
    ):
        """Test that metrics are skipped based on flags and configuration using real metrics"""
        service_config = ServiceConfig(
            developer=DeveloperConfig(show_internal_metrics=show_internal)
        )
        table = RealtimeMetricsTable(service_config)

        metric_result = MetricResult(
            tag=metric_tag,
            header="Test Metric",
            unit="ms",
            avg=1.0,
        )

        assert table._should_skip(metric_result) is should_skip
