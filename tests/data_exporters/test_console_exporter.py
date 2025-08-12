# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from rich.console import Console

from aiperf.common.config import EndpointConfig, ServiceConfig, UserConfig
from aiperf.common.config.dev_config import DeveloperConfig
from aiperf.common.constants import NANOS_PER_MILLIS
from aiperf.common.enums import EndpointType
from aiperf.common.models import MetricResult, ProfileResults
from aiperf.exporters import ConsoleMetricsExporter, ExporterConfig


@pytest.fixture
def mock_endpoint_config():
    return EndpointConfig(
        type=EndpointType.OPENAI_CHAT_COMPLETIONS,
        streaming=True,
        model_names=["test-model"],
    )


@pytest.fixture
def sample_records():
    return [
        MetricResult(
            tag="ttft",
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
    )


class TestConsoleExporter:
    @pytest.mark.asyncio
    async def test_export_prints_expected_table(self, mock_exporter_config, capsys):
        exporter = ConsoleMetricsExporter(mock_exporter_config)
        await exporter.export(Console(width=100))
        output = capsys.readouterr().out
        assert "NVIDIA AIPerf | LLM Metrics" in output
        assert "Time to First Token (ms)" in output
        assert "Request Latency (ms)" in output
        assert "Inter Token Latency (ms)" in output
        assert "Request Throughput (requests/sec)" in output

    @pytest.mark.parametrize(
        "show_internal_metrics, is_hidden_metric, should_skip",
        [
            (True, True, False),  # Show internal metrics, hidden metric -> don't skip
            (False, True, True),  # Don't show internal metrics, hidden metric -> skip
            (
                False,
                False,
                False,
            ),  # Don't show internal metrics, normal metric -> don't skip
            (True, False, False),  # Show internal metrics, normal metric -> don't skip
        ],
    )
    def test_should_skip_logic(
        self,
        mock_endpoint_config: EndpointConfig,
        show_internal_metrics,
        is_hidden_metric,
        should_skip,
        monkeypatch,
    ):
        # Mock the configs to control show_internal_metrics
        user_config = UserConfig(endpoint=mock_endpoint_config)
        service_config = ServiceConfig(
            developer=DeveloperConfig(show_internal_metrics=show_internal_metrics)
        )
        with monkeypatch.context() as m:
            m.setattr(
                "aiperf.exporters.console_metrics_exporter.AIPERF_DEV_MODE",
                show_internal_metrics,
            )
            config = ExporterConfig(
                results=ProfileResults(
                    records=[],
                    start_ns=0,
                    end_ns=0,
                    completed=0,
                ),
                user_config=user_config,
                service_config=service_config,
            )
            exporter = ConsoleMetricsExporter(config)

            # Test with actual metric behavior: use hidden metrics vs normal metrics
            if is_hidden_metric:
                # Use a metric that has HIDDEN flags (like benchmark_duration which has HIDDEN flag)
                record = MetricResult(
                    tag="benchmark_duration",
                    header="Benchmark Duration",
                    unit="s",
                    avg=1.0,
                )
            else:
                # Use a normal metric without HIDDEN flags
                record = MetricResult(
                    tag="request_latency",
                    header="Request Latency",
                    unit="ns",
                    avg=1.0,
                )
            assert exporter._should_skip(record) is should_skip

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
            p75=12.3 * NANOS_PER_MILLIS,
        )
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
