# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
from pathlib import Path

import pytest

from aiperf.common.config.endpoint_config import EndpointConfig
from aiperf.common.config.user_config import UserConfig
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
                unit="ms",
                avg=123.0,
                min=100.0,
                max=150.0,
                p1=101.0,
                p5=105.0,
                p25=110.0,
                p50=120.0,
                p75=130.0,
                p90=140.0,
                p95=None,
                p99=149.0,
                std=10.0,
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

            assert "input_config" in data
            assert isinstance(data["input_config"], dict)
            assert "output" in data["input_config"]
            assert data["input_config"]["output"]["artifact_directory"] == str(
                output_dir
            )
