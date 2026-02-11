# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from collections.abc import Iterable

import aiofiles

from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.models import MetricResult
from aiperf.exporters.exporter_config import ExporterConfig


class MetricsBaseExporter(AIPerfLoggerMixin, ABC):
    """Base class for all metrics exporters with common functionality."""

    def __init__(self, exporter_config: ExporterConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self._results = exporter_config.results
        self._telemetry_results = exporter_config.telemetry_results
        self._server_metrics_results = exporter_config.server_metrics_results
        self._user_config = exporter_config.user_config
        self._output_directory = exporter_config.user_config.output.artifact_directory

    def _prepare_metrics(
        self, metric_results: Iterable[MetricResult]
    ) -> dict[str, MetricResult]:
        """Build a dict of metrics keyed by tag for export.

        Metrics are already filtered and in display units from summarize().

        Args:
            metric_results: Metric results from summarize()

        Returns:
            dict of metrics ready for export
        """
        return {metric.tag: metric for metric in metric_results}

    @abstractmethod
    def _generate_content(self) -> str:
        """Generate export content string.

        Subclasses must implement this to generate format-specific content
        using instance data members (self._results, self._telemetry_results, etc.).

        Returns:
            str: Complete content string ready to write to file
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _generate_content()"
        )

    async def export(self) -> None:
        """Export inference and telemetry data to file.

        Creates output directory, generates content, and writes to file.
        Handles common file writing logic for all exporters.

        Raises:
            Exception: If file writing fails
        """
        self._output_directory.mkdir(parents=True, exist_ok=True)

        self.debug(lambda: f"Exporting data to file: {self._file_path}")

        try:
            content = self._generate_content()

            async with aiofiles.open(
                self._file_path, "w", newline="", encoding="utf-8"
            ) as f:
                await f.write(content)

        except Exception as e:
            self.error(f"Failed to export to {self._file_path}: {e}")
            raise
