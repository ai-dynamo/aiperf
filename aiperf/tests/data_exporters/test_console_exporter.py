# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.config.endpoint_config import EndPointConfig
from aiperf.data_exporter import ConsoleExporter
from aiperf.data_exporter.record import Record


@pytest.fixture
def endpoint_config():
    return EndPointConfig(type="llm", streaming=True)


@pytest.fixture
def sample_records():
    return [
        Record(
            name="Time to First Token",
            unit="ms",
            avg=120.5,
            min=110.0,
            max=130.0,
            p99=128.0,
            p90=125.0,
            p75=122.0,
        ),
        Record(
            name="Request Latency",
            unit="ms",
            avg=15.3,
            min=12.1,
            max=21.4,
            p99=20.5,
            p90=18.7,
            p75=16.2,
        ),
        Record(
            name="Inter Token Latency",
            unit="ms",
            avg=3.7,
            min=2.9,
            max=5.1,
            p99=4.9,
            p90=4.5,
            p75=4.0,
            streaming_only=True,
        ),
        Record(
            name="Request Throughput",
            unit="per sec",
            avg=95.0,
            min=None,
            max=None,
            p99=None,
            p90=None,
            p75=None,
        ),
    ]


class TestConsoleExporter:
    def test_export_prints_expected_table(
        self, endpoint_config, sample_records, capsys
    ):
        exporter = ConsoleExporter(endpoint_config)
        exporter.export(sample_records, width=100)  # fixed width for consistent output
        captured = capsys.readouterr()
        output = captured.out
        assert "NVIDIA AIPerf | LLM Metrics" in output
        assert "Time to First Token (ms)" in output
        assert "Request Latency (ms)" in output
        assert "Inter Token Latency (ms)" in output
        assert "Request Throughput (per sec)" in output

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
        self, endpoint_config, enable_streaming, is_streaming_only_metric, should_skip
    ):
        endpoint_config.streaming = enable_streaming
        exporter = ConsoleExporter(endpoint_config)
        record = Record(
            name="Test Metric",
            unit="ms",
            avg=1.0,
            streaming_only=is_streaming_only_metric,
        )
        assert exporter._should_skip(record) is should_skip

    def test_format_row_formats_values_correctly(self, endpoint_config):
        exporter = ConsoleExporter(endpoint_config)
        record = Record(
            name="Request Latency",
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
        assert row[2] == "N/A"
        assert row[3] == "20.00"
        assert row[4] == "N/A"
        assert row[5] == "15.50"
        assert row[6] == "12.30"

    def test_get_title_returns_expected_string(self, endpoint_config):
        exporter = ConsoleExporter(endpoint_config)
        assert exporter._get_title() == "NVIDIA AIPerf | LLM Metrics"
