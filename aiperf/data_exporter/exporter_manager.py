# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.config import UserConfig
from aiperf.common.factories import DataExporterFactory
from aiperf.data_exporter.exporter_config import ExporterConfig
from aiperf.data_exporter.record import Record


class ExporterManager:
    """
    ExporterManager is responsible for exporting records using all
    registered data exporters.
    """

    def __init__(self, records: list[Record], user_config: UserConfig):
        self._records = records
        self._user_config = user_config
        self._exporter_classes = DataExporterFactory.get_all_classes()

    def export(self) -> None:
        exporter_config = self._create_exporter_config()
        for exporter_class in self._exporter_classes:
            exporter = exporter_class(exporter_config)
            exporter.export()

    def _create_exporter_config(self) -> ExporterConfig:
        """
        Create ExporterConfig that is used by all exporters.
        """
        exporter_config = ExporterConfig(
            records=self._records,
            input_config=self._user_config,
        )
        return exporter_config
