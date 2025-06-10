# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.config.endpoint_config import EndPointConfig
from aiperf.common.data_exporter import ConsoleExporter
from aiperf.common.data_exporter.data_exporter_factory import DataExporterFactory


class TestDataExporterFactory:
    def test_create_data_exporters(self):
        endpoint_config = EndPointConfig(type="embeddings", streaming=True)
        factory = DataExporterFactory()
        exporters = factory.create_data_exporters(endpoint_config)

        assert len(exporters) == 1
        assert isinstance(exporters[0], ConsoleExporter)
