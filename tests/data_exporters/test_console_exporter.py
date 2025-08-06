# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.config.endpoint_config import EndpointConfig
from aiperf.common.config.user_config import UserConfig
from aiperf.common.enums import EndpointType
from aiperf.common.models import MetricResult
from aiperf.common.models.record_models import ProfileResults
from aiperf.exporters import ConsoleExporter
from aiperf.exporters.exporter_config import ExporterConfig


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
            tag="Time to First Token",
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
            tag="Request Latency",
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
            tag="Inter Token Latency",
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
            tag="Request Throughput",
            header="Request Throughput",
            unit="per sec",
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
    )


class TestConsoleExporter:
    @pytest.mark.asyncio
    async def test_export_prints_expected_table(self, mock_exporter_config, capsys):
        exporter = ConsoleExporter(mock_exporter_config)
        await exporter.export(width=100)
        output = capsys.readouterr().out
        assert "NVIDIA AIPerf | LLM Metrics" in output
        assert "Time to First Token (ms)" in output
        assert "Request Latency (ms)" in output
        assert "Inter Token Latency (ms)" in output
        assert "Request Throughput (per sec)" in output

    @pytest.skip(
        reason="TODO: Metric refactor work in progress", allow_module_level=True
    )
    @pytest.mark.parametrize(
        "enable_streaming, is_streaming_only_metric, should_skip",
        [
            (True, True, False),
            (False, True, True),
            (False, False, False),
            (True, False, False),
        ],
    )
    def test_should_skip_logic(
        self,
        mock_endpoint_config,
        enable_streaming,
        is_streaming_only_metric,
        should_skip,
    ):
        mock_endpoint_config.streaming = enable_streaming
        input_config = UserConfig(endpoint=mock_endpoint_config)
        config = ExporterConfig(
            results=ProfileResults(
                records=[],
                start_ns=0,
                end_ns=0,
                completed=0,
            ),
            user_config=input_config,
        )
        exporter = ConsoleExporter(config)
        record = MetricResult(
            tag="ttft",
            header="Time to First Token",
            unit="ms",
            avg=1.0,
        )
        assert exporter._should_skip(record) is should_skip

    @pytest.skip(
        reason="TODO: Metric refactor work in progress", allow_module_level=True
    )
    def test_format_row_formats_values_correctly(self, mock_exporter_config):
        exporter = ConsoleExporter(mock_exporter_config)
        record = MetricResult(
            tag="Request Latency",
            header="Request Latency",
            unit="ms",
            avg=10.123,
            min=None,
            max=20.0,
            p99=None,
            p90=15.5,
            p75=12.3,
        )
        row = exporter._format_row(record)
        assert row[0] == "Request Latency (ms)"
        assert row[1] == "10.12"
        assert row[2] == "[dim]N/A[/dim]"
        assert row[3] == "20.00"
        assert row[4] == "[dim]N/A[/dim]"
        assert row[5] == "15.50"
        assert row[6] == "12.30"

    @pytest.skip(
        reason="TODO: Metric refactor work in progress", allow_module_level=True
    )
    def test_get_title_returns_expected_string(self, mock_exporter_config):
        exporter = ConsoleExporter(mock_exporter_config)
        assert exporter._get_title() == "NVIDIA AIPerf | LLM Metrics"
