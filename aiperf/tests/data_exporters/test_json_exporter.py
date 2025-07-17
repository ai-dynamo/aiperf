# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
from pathlib import Path

import pytest

from aiperf.common.config.user_config import UserConfig
from aiperf.common.record_models import MetricResult
from aiperf.data_exporter.exporter_config import ExporterConfig
from aiperf.data_exporter.json_exporter import JsonExporter


class TestJsonExporter:
    @pytest.fixture
    def sample_records(self):
        return [
            MetricResult(
                tag="Test Metric",
                header="Tes t Metric",
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
                streaming_only=False,
            )
        ]

    @pytest.fixture
    def mock_user_config(self):
        return UserConfig(model_names=["test-model"])

    @pytest.mark.asyncio
    async def test_json_exporter_creates_expected_json(
        self, sample_records, mock_user_config
    ):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            mock_user_config.output.artifact_directory = output_dir

            exporter_config = ExporterConfig(
                results=sample_records,
                input_config=mock_user_config,
            )

            exporter = JsonExporter(exporter_config)
            await exporter.export()

            expected_file = output_dir / "profile_export_aiperf.json"
            assert expected_file.exists()

            with open(expected_file) as f:
                data = json.load(f)

            assert "Test Metric" in data
            assert data["Test Metric"]["unit"] == "ms"
            assert data["Test Metric"]["avg"] == 123.0

            assert "input_config" in data
            assert isinstance(data["input_config"], dict)
            assert "output" in data["input_config"]
            assert data["input_config"]["output"]["artifact_directory"] == str(
                output_dir
            )
