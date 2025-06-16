# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.config.endpoint.endpoint_config import EndPointConfig
from aiperf.common.factories import DataExporterFactory
from aiperf.data_exporter.record import Record


class ExporterManager:
    """
    ExporterManager is responsible for exporting records using all
    registered data exporters.
    """

    def __init__(self, endpoint_config: EndPointConfig):
        self.endpoint_config = endpoint_config
        self.exporter_classes = DataExporterFactory.get_all_classes()

    def export(self, records: list[Record]) -> None:
        for exporter_class in self.exporter_classes:
            exporter = exporter_class(self.endpoint_config)
            exporter.export(records)
