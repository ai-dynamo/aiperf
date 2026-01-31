# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from rich.console import Console

from aiperf.common.config import EndpointConfig, OutputConfig, ServiceConfig, UserConfig
from aiperf.common.exceptions import ConsoleExporterDisabled
from aiperf.common.models import MetricResult, ProfileResults
from aiperf.exporters.exporter_config import ExporterConfig
from aiperf.exporters.http_trace_console_exporter import HttpTraceConsoleExporter
from aiperf.metrics.types.http_trace_metrics import (
    HttpBlockedMetric,
    HttpChunksReceivedMetric,
    HttpChunksSentMetric,
    HttpConnectingMetric,
    HttpConnectionOverheadMetric,
    HttpConnectionReusedMetric,
    HttpDataReceivedMetric,
    HttpDataSentMetric,
    HttpDnsLookupMetric,
    HttpDurationMetric,
    HttpReceivingMetric,
    HttpSendingMetric,
    HttpTotalTimeMetric,
    HttpWaitingMetric,
)
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
def sample_http_trace_records():
    """Sample HTTP trace metric records for testing."""
    return [
        MetricResult(
            tag="http_req_blocked",
            header="HTTP Blocked",
            unit="ms",
            avg=5.0,
            min=1.0,
            max=15.0,
            p99=14.0,
            p90=12.0,
            p50=5.0,
        ),
        MetricResult(
            tag="http_req_dns_lookup",
            header="HTTP DNS Lookup",
            unit="ms",
            avg=2.0,
            min=0.5,
            max=8.0,
            p99=7.5,
            p90=6.0,
            p50=2.0,
        ),
        MetricResult(
            tag="http_req_connecting",
            header="HTTP Connecting",
            unit="ms",
            avg=25.0,
            min=10.0,
            max=50.0,
            p99=48.0,
            p90=40.0,
            p50=25.0,
        ),
        MetricResult(
            tag="http_req_sending",
            header="HTTP Sending",
            unit="ms",
            avg=1.5,
            min=0.5,
            max=5.0,
            p99=4.5,
            p90=3.5,
            p50=1.5,
        ),
        MetricResult(
            tag="http_req_waiting",
            header="HTTP Waiting (TTFB)",
            unit="ms",
            avg=100.0,
            min=50.0,
            max=200.0,
            p99=190.0,
            p90=170.0,
            p50=100.0,
        ),
        MetricResult(
            tag="http_req_receiving",
            header="HTTP Receiving",
            unit="ms",
            avg=50.0,
            min=20.0,
            max=100.0,
            p99=95.0,
            p90=85.0,
            p50=50.0,
        ),
        MetricResult(
            tag="http_req_duration",
            header="HTTP Duration (excl. conn)",
            unit="ms",
            avg=151.5,
            min=70.5,
            max=305.0,
            p99=290.0,
            p90=259.0,
            p50=151.5,
        ),
    ]


@pytest.fixture
def sample_mixed_records(sample_http_trace_records):
    """Mix of HTTP trace records and regular metrics."""
    regular_records = [
        MetricResult(
            tag="time_to_first_token",
            header="Time to First Token",
            unit="ms",
            avg=120.5,
            min=110.0,
            max=130.0,
            p99=128.0,
            p90=125.0,
            p50=119.0,
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
            p50=15.0,
        ),
    ]
    return sample_http_trace_records + regular_records


def make_exporter_config(
    records: list[MetricResult],
    endpoint_config: EndpointConfig,
    show_trace_timing: bool = True,
) -> ExporterConfig:
    """Create an ExporterConfig with the specified settings."""
    user_config = UserConfig(
        endpoint=endpoint_config,
        output=OutputConfig(show_trace_timing=show_trace_timing),
    )
    return ExporterConfig(
        results=ProfileResults(
            records=records,
            start_ns=0,
            end_ns=0,
            completed=0,
        ),
        user_config=user_config,
        service_config=ServiceConfig(),
        telemetry_results=None,
    )


class TestHttpTraceConsoleExporter:
    """Tests for HttpTraceConsoleExporter."""

    def test_raises_when_disabled(self, mock_endpoint_config):
        """Test that exporter raises ConsoleExporterDisabled when flag is False."""
        config = make_exporter_config(
            records=[],
            endpoint_config=mock_endpoint_config,
            show_trace_timing=False,
        )
        with pytest.raises(ConsoleExporterDisabled) as exc_info:
            HttpTraceConsoleExporter(config)

        assert "HTTP trace timing is not enabled" in str(exc_info.value)

    def test_creates_successfully_when_enabled(self, mock_endpoint_config):
        """Test that exporter creates successfully when flag is True."""
        config = make_exporter_config(
            records=[],
            endpoint_config=mock_endpoint_config,
            show_trace_timing=True,
        )
        exporter = HttpTraceConsoleExporter(config)
        assert exporter._show_trace_timing is True

    def test_get_title_returns_http_trace_title(self, mock_endpoint_config):
        """Test that _get_title returns the correct title."""
        config = make_exporter_config(
            records=[],
            endpoint_config=mock_endpoint_config,
            show_trace_timing=True,
        )
        exporter = HttpTraceConsoleExporter(config)
        assert exporter._get_title() == "NVIDIA AIPerf | HTTP Trace Timing"

    @pytest.mark.parametrize(
        "metric_class, should_show",
        [
            # HTTP trace metrics - should be shown
            (HttpBlockedMetric, True),
            (HttpDnsLookupMetric, True),
            (HttpConnectingMetric, True),
            (HttpSendingMetric, True),
            (HttpWaitingMetric, True),
            (HttpReceivingMetric, True),
            (HttpConnectionReusedMetric, True),
            (HttpDataSentMetric, True),
            (HttpChunksSentMetric, True),
            (HttpDataReceivedMetric, True),
            (HttpChunksReceivedMetric, True),
            (HttpConnectionOverheadMetric, True),
            (HttpDurationMetric, True),
            (HttpTotalTimeMetric, True),
            # Regular metrics - should NOT be shown
            (TTFTMetric, False),
            (RequestLatencyMetric, False),
        ],
    )  # fmt: skip
    def test_should_show_only_http_trace_metrics(
        self,
        mock_endpoint_config,
        metric_class,
        should_show,
    ):
        """Test that only HTTP trace metrics are shown."""
        config = make_exporter_config(
            records=[],
            endpoint_config=mock_endpoint_config,
            show_trace_timing=True,
        )
        exporter = HttpTraceConsoleExporter(config)

        record = MetricResult(
            tag=metric_class.tag,
            header="Test Metric",
            unit="ms",
            avg=1.0,
        )
        assert exporter._should_show(record) is should_show

    @pytest.mark.asyncio
    async def test_export_prints_http_trace_table(
        self, sample_http_trace_records, mock_endpoint_config, capsys
    ):
        """Test that export prints the HTTP trace table with correct content."""
        config = make_exporter_config(
            records=sample_http_trace_records,
            endpoint_config=mock_endpoint_config,
            show_trace_timing=True,
        )
        exporter = HttpTraceConsoleExporter(config)
        await exporter.export(Console(width=120))

        output = capsys.readouterr().out
        assert "NVIDIA AIPerf | HTTP Trace Timing" in output
        assert "HTTP Blocked" in output
        assert "HTTP DNS Lookup" in output
        assert "HTTP Connecting" in output
        assert "HTTP Sending" in output
        assert "HTTP Waiting (TTFB)" in output
        assert "HTTP Receiving" in output
        assert "HTTP Duration (excl. conn)" in output

    @pytest.mark.asyncio
    async def test_export_filters_non_trace_metrics(
        self, sample_mixed_records, mock_endpoint_config, capsys
    ):
        """Test that regular metrics are filtered out from the output."""
        config = make_exporter_config(
            records=sample_mixed_records,
            endpoint_config=mock_endpoint_config,
            show_trace_timing=True,
        )
        exporter = HttpTraceConsoleExporter(config)
        await exporter.export(Console(width=120))

        output = capsys.readouterr().out
        # HTTP trace metrics should be present
        assert "HTTP Blocked" in output
        assert "HTTP Duration (excl. conn)" in output
        # Regular metrics should NOT be present
        assert "Time to First Token" not in output
        assert "Request Latency" not in output

    @pytest.mark.asyncio
    async def test_export_with_no_records_returns_early(
        self, mock_endpoint_config, capsys
    ):
        """Test that export returns early when there are no records."""
        config = make_exporter_config(
            records=[],
            endpoint_config=mock_endpoint_config,
            show_trace_timing=True,
        )
        exporter = HttpTraceConsoleExporter(config)
        await exporter.export(Console(width=120))

        output = capsys.readouterr().out
        # Should not print anything when there are no records
        assert "HTTP Trace Timing" not in output
