# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from aiperf.common.exceptions import ConsoleExporterDisabled
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.exporters.exporter_config import ExporterConfig

if TYPE_CHECKING:
    from rich.console import Console


class AccuracyConsoleExporter(AIPerfLoggerMixin):
    """Console exporter for accuracy benchmarking results.

    Displays accuracy metrics (overall score, per-task breakdown) to the console.
    Self-disables when accuracy mode is not enabled.
    """

    def __init__(self, exporter_config: ExporterConfig, **kwargs) -> None:
        if not exporter_config.user_config.accuracy.enabled:
            raise ConsoleExporterDisabled(
                "Accuracy console exporter is disabled: accuracy mode is not enabled"
            )

        super().__init__(**kwargs)
        self.exporter_config = exporter_config

    async def export(self, console: Console) -> None:
        raise NotImplementedError
