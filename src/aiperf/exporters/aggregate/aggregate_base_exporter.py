# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Base class for aggregate exporters."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import aiofiles

from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.orchestrator.aggregation.base import AggregateResult


@dataclass(slots=True)
class AggregateExporterConfig:
    """Configuration for aggregate exporters.

    Simpler than ExporterConfig because aggregate exports don't need:
    - ProfileResults (single-run data)
    - TelemetryExportData (per-run telemetry)
    - ServerMetricsResults (per-run server metrics)
    - Full UserConfig (just need output directory)

    Attributes:
        result: AggregateResult to export
        output_dir: Directory where export file will be written
    """

    result: AggregateResult
    output_dir: Path


class AggregateBaseExporter(AIPerfLoggerMixin, ABC):
    """Base class for all aggregate exporters.

    Provides common functionality:
    - File writing logic
    - Directory creation
    - Error handling
    - Logging

    Subclasses implement:
    - _generate_content() - Format-specific content generation
    - get_file_name() - Output file name
    """

    def __init__(self, config: AggregateExporterConfig, **kwargs) -> None:
        """Initialize aggregate exporter.

        Args:
            config: Configuration for the exporter
            **kwargs: Additional arguments passed to AIPerfLoggerMixin
        """
        super().__init__(**kwargs)
        self._config = config
        self._result = config.result
        self._output_dir = Path(config.output_dir)

    @abstractmethod
    def get_file_name(self) -> str:
        """Return the output file name.

        Returns:
            str: File name (e.g., "profile_export_aiperf_aggregate.json")
        """
        pass

    @abstractmethod
    def _generate_content(self) -> str:
        """Generate export content string.

        Subclasses implement format-specific content generation.

        Returns:
            str: Complete content string ready to write to file
        """
        pass

    async def export(self) -> Path:
        """Export aggregate result to file.

        Creates output directory, generates content, and writes to file.

        Returns:
            Path: Path to written file

        Raises:
            Exception: If file writing fails
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)

        file_path = self._output_dir / self.get_file_name()

        self.debug(lambda: f"Exporting aggregate data to: {file_path}")

        try:
            content = self._generate_content()

            async with aiofiles.open(file_path, "w", newline="", encoding="utf-8") as f:
                await f.write(content)

            self.info(f"Exported aggregate data to: {file_path}")
            return file_path

        except Exception as e:
            self.error(f"Failed to export to {file_path}: {e}")
            raise
