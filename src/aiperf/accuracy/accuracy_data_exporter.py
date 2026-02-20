# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.exceptions import DataExporterDisabled
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.exporters.exporter_config import ExporterConfig, FileExportInfo


class AccuracyDataExporter(AIPerfLoggerMixin):
    """Data exporter for accuracy benchmarking results.

    Exports per-problem grading results to CSV for offline analysis.
    Self-disables when accuracy mode is not enabled.
    """

    def __init__(self, exporter_config: ExporterConfig, **kwargs) -> None:
        if not exporter_config.user_config.accuracy.enabled:
            raise DataExporterDisabled(
                "Accuracy data exporter is disabled: accuracy mode is not enabled"
            )

        super().__init__(**kwargs)
        self.exporter_config = exporter_config

    def get_export_info(self) -> FileExportInfo:
        raise NotImplementedError

    async def export(self) -> None:
        raise NotImplementedError
