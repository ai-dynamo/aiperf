# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

import pytest

from aiperf.common.environment import Environment
from aiperf.common.models import MetricResult
from aiperf.metrics.types.benchmark_duration_metric import BenchmarkDurationMetric
from aiperf.metrics.types.error_request_count import ErrorRequestCountMetric
from aiperf.metrics.types.inter_token_latency_metric import InterTokenLatencyMetric
from aiperf.metrics.types.output_token_count import (
    OutputTokenCountMetric,
)
from aiperf.metrics.types.request_latency_metric import RequestLatencyMetric
from aiperf.metrics.types.theoretical_prefix_cache_metric import (
    TheoreticalPrefixCacheHitMetric,
)
from aiperf.metrics.types.ttft_metric import TTFTMetric
from aiperf.ui.dashboard.realtime_metrics_dashboard import RealtimeMetricsTable


class TestRealtimeMetricsTable:
    @pytest.mark.parametrize(
        "metric_tag, show_internal, should_skip",
        [
            # ERROR_ONLY metrics - always skipped
            (ErrorRequestCountMetric.tag, False, True),
            (ErrorRequestCountMetric.tag, True, True),
            # NO_CONSOLE metrics - skipped unless SHOW_INTERNAL_METRICS is True
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
            # Externally-injected accumulator metric (CACHE group) - always shown
            (TheoreticalPrefixCacheHitMetric.tag, False, False),
            (TheoreticalPrefixCacheHitMetric.tag, True, False),
        ],
    )  # fmt: skip
    def test_should_skip_logic_with_real_metrics(
        self, metric_tag, show_internal, should_skip
    ):
        """Test that metrics are skipped based on flags and configuration using real metrics"""
        with patch.object(Environment.DEV, "SHOW_INTERNAL_METRICS", show_internal):
            run = Mock()
            table = RealtimeMetricsTable(run)

            metric_result = MetricResult(
                tag=metric_tag,
                header="Test Metric",
                unit="ms",
                avg=1.0,
            )

            assert table._should_skip(metric_result) is should_skip


class TestRealtimeMetricsTableExternallyInjectedTags:
    """Registered accumulator tags render; unregistered tags cannot kill the table.

    The theoretical_prefix_cache_hit MetricResult is minted by a standalone
    accumulator (not the record-metric pipeline), so it reaches the dashboard
    on every graph-IR run. Before its display class was registered, the strict
    MetricRegistry.get_class calls in the update path raised MetricTypeError on
    every tick and no rows rendered at all.
    """

    def _mounted_table(self) -> RealtimeMetricsTable:
        table = RealtimeMetricsTable(Mock())
        table.data_table = Mock()
        table.data_table.is_mounted = True
        return table

    def _rendered_headers(self, table: RealtimeMetricsTable) -> list[str]:
        return [call.args[0].plain for call in table.data_table.add_row.call_args_list]

    def test_update_renders_theoretical_prefix_cache_hit_row(self):
        table = self._mounted_table()
        table.update(
            [
                MetricResult(
                    tag=TheoreticalPrefixCacheHitMetric.tag,
                    header="ignored - display metadata comes from the registry",
                    unit="%",
                    avg=42.5,
                ),
            ]
        )
        assert self._rendered_headers(table) == ["Theoretical Prefix Cache Hit (%)"]

    def test_update_with_unregistered_tag_renders_fallback_and_sorts_last(self):
        table = self._mounted_table()
        table.update(
            [
                MetricResult(
                    tag="not_a_registered_metric",
                    header="Mystery",
                    unit="widgets",
                    avg=1.0,
                ),
                MetricResult(
                    tag=TheoreticalPrefixCacheHitMetric.tag,
                    header="ignored",
                    unit="%",
                    avg=42.5,
                ),
            ]
        )
        assert self._rendered_headers(table) == [
            "Theoretical Prefix Cache Hit (%)",
            "Mystery (widgets)",
        ]

    def test_should_skip_returns_false_for_unregistered_tag(self):
        table = RealtimeMetricsTable(Mock())
        metric_result = MetricResult(
            tag="not_a_registered_metric", header="Mystery", unit="widgets", avg=1.0
        )
        assert table._should_skip(metric_result) is False
