# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

from aiperf.common.enums import DataExporterType
from aiperf.common.factories import DataExporterFactory
from aiperf.data_exporter.exporter_config import ExporterConfig


@DataExporterFactory.register(DataExporterType.JSON)
class JsonExporter:
    """
    A class to export records to a JSON file.
    """

    def __init__(self, exporter_config: ExporterConfig) -> None:
        self._records = exporter_config.records
        self._output_directory = exporter_config.input_config.output.artifact_directory
        self._input_config = exporter_config.input_config

    def export(self) -> None:
        filename = self._output_directory / "profile_export_aiperf.json"
        self._output_directory.mkdir(parents=True, exist_ok=True)
        export_data = {record.name: record.to_dict() for record in self._records}
        if self._input_config:
            input_config_dict = self._input_config.model_dump()
            input_config_serializable = json.loads(
                json.dumps(input_config_dict, default=str)
            )
            export_data["input_config"] = input_config_serializable
        with open(filename, "w") as f:
            json.dump(export_data, f, indent=2)
