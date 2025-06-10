# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest

from aiperf.common.config.endpoint_config import EndPointConfig
from aiperf.common.data_exporter.exporter_manager import ExporterManager
from aiperf.common.data_exporter.record import Record


@pytest.fixture
def endpoint_config():
    return EndPointConfig(type="llm", streaming=True)


@pytest.fixture
def sample_records():
    return [Record(name="Latency", unit="ms", avg=10.0)]


class TestExporterManager:
    def test_export(self, endpoint_config, sample_records):
        with patch(
            "aiperf.common.data_exporter.data_exporter_factory.DataExporterFactory.create_data_exporters"
        ) as mock_create:
            mock_console_exporter = MagicMock()
            mock_create.return_value = [mock_console_exporter]

            manager = ExporterManager(endpoint_config)
            manager.export(sample_records)

            mock_console_exporter.export.assert_called_once_with(sample_records)
