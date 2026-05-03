# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from rich.console import Console

from aiperf.common.config import EndpointConfig, ServiceConfig, UserConfig
from aiperf.common.constants import NANOS_PER_MILLIS
from aiperf.common.models import MetricResult, ProfileResults
from aiperf.exporters.console_metrics_exporter import ConsoleMetricsExporter
from aiperf.exporters.exporter_config import ExporterConfig
from aiperf.metrics.display_units import to_display_unit
from aiperf.metrics.metric_registry import MetricRegistry
from aiperf.metrics.types.benchmark_duration_metric import BenchmarkDurationMetric
from aiperf.metrics.types.credit_drop_latency_metric import CreditDropLatencyMetric
from aiperf.metrics.types.error_request_count import ErrorRequestCountMetric
from aiperf.metrics.types.inter_token_latency_metric import InterTokenLatencyMetric
from aiperf.metrics.types.output_token_count import OutputTokenCountMetric
from aiperf.metrics.types.request_latency_metric import RequestLatencyMetric
from aiperf.metrics.types.ttft_metric import TTFTMetric
from aiperf.plugin.enums import EndpointType


@pytest.fixture
def mock_endpoint_config():
    return EndpointConfig(
        type=EndpointType.CHAT,
        streaming=True,
        model_names=["test-model"],
    )


@pytest.fixture
def sample_records():
    return [
        MetricResult(
            tag="time_to_first_token",
            header="Time to First Token",
            unit="ms",
            avg=120.5,
            min=110.0,
            max=130.0,
            p99=128.0,
            p90=125.0,
            p75=122.0,
        ),
        MetricResult(
            tag="request_latency",
            header="Request Latency",
            unit="ms",
            avg=15.3,
            min=12.1,
            max=21.4,
            p99=20.5,
            p90=18.7,
            p75=16.2,
        ),
        MetricResult(
            tag="inter_token_latency",
            header="Inter Token Latency",
            unit="ms",
            avg=3.7,
            min=2.9,
            max=5.1,
            p99=4.9,
            p90=4.5,
            p75=4.0,
        ),
        MetricResult(
            tag="request_throughput",
            header="Request Throughput",
            unit="requests/sec",
            avg=95.0,
        ),
    ]


@pytest.fixture
def mock_exporter_config(sample_records, mock_endpoint_config):
    input_config = UserConfig(endpoint=mock_endpoint_config)
    return ExporterConfig(
        results=ProfileResults(
            records=sample_records,
            start_ns=0,
            end_ns=0,
            completed=0,
        ),
        user_config=input_config,
        service_config=ServiceConfig(),
        telemetry_results=None,
    )


class TestConsoleExporter:
    @pytest.mark.asyncio
    async def test_export_prints_expected_table(self, mock_exporter_config, capsys):
        exporter = ConsoleMetricsExporter(mock_exporter_config)
        await exporter.export(Console(width=115))
        output = capsys.readouterr().out
        assert "NVIDIA AIPerf | LLM Metrics" in output
        assert "Time to First Token (ms)" in output or "Time to First Token" in output
        assert "Request Latency (ms)" in output or "Request Latency" in output
        assert "Inter Token Latency (ms)" in output or "Inter Token Latency" in output
        assert "Request Throughput" in output
        assert "requests/sec" in output

    @pytest.mark.parametrize(
        "metric_tag, should_show",
        [
            # ERROR_ONLY flags - always hidden
            (ErrorRequestCountMetric.tag, False),  # ERROR_ONLY flag
            # NO_CONSOLE flags - hidden
            (BenchmarkDurationMetric.tag, False),  # NO_CONSOLE flag
            (OutputTokenCountMetric.tag, False),  # NO_CONSOLE flag
            (CreditDropLatencyMetric.tag, False),  # INTERNAL flag
            # INTERNAL flags - hidden
            (CreditDropLatencyMetric.tag, False),  # INTERNAL flag
            # Normal metrics - shown
            (InterTokenLatencyMetric.tag, True),  # Normal metric
            (RequestLatencyMetric.tag, True),  # Normal metric
            (TTFTMetric.tag, True),  # Normal metric
        ],
    )  # fmt: skip
    def test_should_show_metrics_based_on_flags(
        self,
        mock_endpoint_config: EndpointConfig,
        metric_tag,
        should_show,
    ):
        """Test that metrics are shown/hidden based on their flags"""
        user_config = UserConfig(endpoint=mock_endpoint_config)
        service_config = ServiceConfig()
        config = ExporterConfig(
            results=ProfileResults(
                records=[],
                start_ns=0,
                end_ns=0,
                completed=0,
            ),
            user_config=user_config,
            service_config=service_config,
            telemetry_results=None,
        )
        exporter = ConsoleMetricsExporter(config)

        record = MetricResult(
            tag=metric_tag,
            header="Test Metric",
            unit="ms",
            avg=1.0,
        )
        assert exporter._should_show(record) is should_show

    def test_format_row_formats_values_correctly(self, mock_exporter_config):
        exporter = ConsoleMetricsExporter(mock_exporter_config)
        # Request latency metric expects values in nanoseconds (native unit)
        # but displays in milliseconds.
        record = MetricResult(
            tag="request_latency",
            header="Request Latency",
            unit="ns",
            avg=10.123 * NANOS_PER_MILLIS,
            min=None,
            max=20.0 * NANOS_PER_MILLIS,
            p99=None,
            p90=15.5 * NANOS_PER_MILLIS,
            p50=12.3 * NANOS_PER_MILLIS,
        )
        record = to_display_unit(record, MetricRegistry)
        row = exporter._format_row(record)
        # This asserts that the display is unit converted correctly
        assert row[0] == "Request Latency (ms)"
        assert row[1] == "10.12"
        assert row[2] == "[dim]N/A[/dim]"
        assert row[3] == "20.00"
        assert row[4] == "[dim]N/A[/dim]"
        assert row[5] == "15.50"
        assert row[6] == "12.30"

    def test_get_title_returns_expected_string(self, mock_exporter_config):
        exporter = ConsoleMetricsExporter(mock_exporter_config)
        assert exporter._get_title() == "NVIDIA AIPerf | LLM Metrics"
