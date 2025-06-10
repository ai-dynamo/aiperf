# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from aiperf.common.config.endpoint_config import EndPointConfig
from aiperf.common.data_exporter import ConsoleExporter
from aiperf.common.data_exporter.interface import DataExporterInterface


class DataExporterFactory:
    def create_data_exporters(
        self, endpoint_config: EndPointConfig
    ) -> list[DataExporterInterface]:
        return [ConsoleExporter(endpoint_config)]
