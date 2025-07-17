# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest

from aiperf.common.config import EndPointConfig, OutputConfig, UserConfig
from aiperf.common.enums.endpoints import EndpointType
from aiperf.common.record_models import MetricResult
from aiperf.data_exporter.exporter_manager import ExporterManager


@pytest.fixture
def endpoint_config():
    return EndPointConfig(type=EndpointType.OPENAI_CHAT_COMPLETIONS, streaming=True)


@pytest.fixture
def output_config(tmp_path):
    return OutputConfig(artifact_directory=tmp_path)


@pytest.fixture
def sample_records():
    return [
        MetricResult(
            tag="Latency",
            unit="ms",
            avg=10.0,
            header="test-header",
        )
    ]


@pytest.fixture
def mock_user_config(endpoint_config, output_config):
    config = UserConfig(model_names=["test-model"])
    config.endpoint = endpoint_config
    config.output = output_config
    return config


class TestExporterManager:
    @pytest.mark.asyncio
    async def test_export(
        self, endpoint_config, output_config, sample_records, mock_user_config
    ):
        mock_exporter_instance = MagicMock()
        mock_exporter_class = MagicMock(return_value=mock_exporter_instance)

        with patch(
            "aiperf.common.factories.DataExporterFactory.get_all_classes",
            return_value=[mock_exporter_class],
        ):
            manager = ExporterManager(
                results=sample_records,
                input_config=mock_user_config,
            )
            await manager.export_all()
        mock_exporter_class.assert_called_once()
        mock_exporter_instance.export.assert_called_once()
