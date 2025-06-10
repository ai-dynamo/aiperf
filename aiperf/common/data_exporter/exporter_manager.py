# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.config.endpoint_config import EndPointConfig
from aiperf.common.data_exporter.data_exporter_factory import DataExporterFactory
from aiperf.common.data_exporter.record import Record


class ExporterManager:
    def __init__(self, endpoint_config: EndPointConfig):
        factory = DataExporterFactory()
        self.exporters = factory.create_data_exporters(endpoint_config)

    def export(self, records: list[Record]) -> None:
        for exporter in self.exporters:
            exporter.export(records)
