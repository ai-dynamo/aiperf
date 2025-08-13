# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
from pathlib import Path

import pytest

from aiperf.common.config import EndpointConfig, ServiceConfig, UserConfig
from aiperf.common.constants import NANOS_PER_MILLIS
from aiperf.common.enums import EndpointType
from aiperf.common.models import MetricResult
from aiperf.exporters.exporter_config import ExporterConfig
from aiperf.exporters.json_exporter import JsonExporter


class TestJsonExporter:
    @pytest.fixture
    def sample_records(self):
        return [
            MetricResult(
                tag="ttft",
                header="Time to First Token",
                unit="ns",
                avg=123.0 * NANOS_PER_MILLIS,
                min=100.0 * NANOS_PER_MILLIS,
                max=150.0 * NANOS_PER_MILLIS,
                p1=101.0 * NANOS_PER_MILLIS,
                p5=105.0 * NANOS_PER_MILLIS,
                p25=110.0 * NANOS_PER_MILLIS,
                p50=120.0 * NANOS_PER_MILLIS,
                p75=130.0 * NANOS_PER_MILLIS,
                p90=140.0 * NANOS_PER_MILLIS,
                p95=None,
                p99=149.0 * NANOS_PER_MILLIS,
                std=10.0 * NANOS_PER_MILLIS,
            )
        ]

    @pytest.fixture
    def mock_user_config(self):
        return UserConfig(
            endpoint=EndpointConfig(
                model_names=["test-model"],
                type=EndpointType.OPENAI_CHAT_COMPLETIONS,
                custom_endpoint="custom_endpoint",
            )
        )

    @pytest.fixture
    def mock_results(self, sample_records):
        class MockResults:
            def __init__(self, metrics):
                self.metrics = metrics
                self.start_ns = None
                self.end_ns = None

            @property
            def records(self):
                return self.metrics

            @property
            def has_results(self):
                return bool(self.metrics)

            @property
            def was_cancelled(self):
                return False

            @property
            def error_summary(self):
                return []

        return MockResults(sample_records)

    @pytest.mark.asyncio
    async def test_json_exporter_creates_expected_json(
        self, mock_results, mock_user_config
    ):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            mock_user_config.output.artifact_directory = output_dir

            exporter_config = ExporterConfig(
                results=mock_results,
                user_config=mock_user_config,
                service_config=ServiceConfig(),
            )

            exporter = JsonExporter(exporter_config)
            await exporter.export()

            expected_file = output_dir / "profile_export_aiperf.json"
            assert expected_file.exists()

            with open(expected_file) as f:
                data = json.load(f)

            assert "records" in data
            records = data["records"]
            assert isinstance(records, dict)
            assert len(records) == 1
            assert "ttft" in records
            assert records["ttft"]["unit"] == "ms"
            assert records["ttft"]["avg"] == 123.0
            assert records["ttft"]["p1"] == 101.0

            assert "input_config" in data
            assert isinstance(data["input_config"], dict)
            # TODO: Uncomment this once we have expanded the output config to include all important fields
            # assert "output" in data["input_config"]
            # assert data["input_config"]["output"]["artifact_directory"] == str(
            #     output_dir
            # )
