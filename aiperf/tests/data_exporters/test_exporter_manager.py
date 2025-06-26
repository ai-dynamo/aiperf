# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest

from aiperf.common.config import EndPointConfig, OutputConfig, UserConfig
from aiperf.data_exporter.exporter_manager import ExporterManager
from aiperf.data_exporter.record import Record


@pytest.fixture
def endpoint_config():
    return EndPointConfig(type="llm", streaming=True)


@pytest.fixture
def output_config(tmp_path):
    return OutputConfig(artifact_directory=tmp_path)


@pytest.fixture
def sample_records():
    return [Record(name="Latency", unit="ms", avg=10.0)]


@pytest.fixture
def mock_user_config(endpoint_config, output_config):
    config = UserConfig()
    config.endpoint = endpoint_config
    config.output = output_config
    return config


class TestExporterManager:
    def test_export(
        self, endpoint_config, output_config, sample_records, mock_user_config
    ):
        mock_exporter_instance = MagicMock()
        mock_exporter_class = MagicMock(return_value=mock_exporter_instance)

        with patch(
            "aiperf.common.factories.DataExporterFactory.get_all_classes",
            return_value=[mock_exporter_class],
        ):
            manager = ExporterManager(
                records=sample_records,
                user_config=mock_user_config,
            )
            manager.export()
        mock_exporter_class.assert_called_once()
        mock_exporter_instance.export.assert_called_once()
