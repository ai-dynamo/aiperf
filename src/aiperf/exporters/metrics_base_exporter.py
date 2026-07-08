# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import os
from abc import ABC, abstractmethod
from collections.abc import Iterable

import aiofiles

from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.models import MetricResult
from aiperf.exporters.exporter_config import ExporterConfig
from aiperf.metrics.metric_registry import MetricRegistry


class MetricsBaseExporter(AIPerfLoggerMixin, ABC):
    """Base class for all metrics exporters with common functionality."""

    def __init__(self, exporter_config: ExporterConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self._results = exporter_config.results
        self._telemetry_results = exporter_config.telemetry_results
        self._server_metrics_results = exporter_config.server_metrics_results
        self._config = exporter_config.config
        self._metric_registry = MetricRegistry
        self._output_directory = exporter_config.config.artifacts.dir

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

        Creates output directory, generates content, and atomically publishes
        the file via a ``.partial`` sidecar + ``os.replace``. The atomic swap
        matters for the key exports (``profile_export_aiperf.json``/``.csv``):
        a mid-write ENOSPC or kill truncates only the sidecar, never the final
        artifact the K8s results-ready marker and the operator's
        ``_key_files_materialized`` gate depend on. On failure the sidecar is
        removed so a truncated file never lingers under the key name.

        Raises:
            Exception: If file writing fails
        """
        self._output_directory.mkdir(parents=True, exist_ok=True)

        self.debug(lambda: f"Exporting data to file: {self._file_path}")

        tmp_path = self._file_path.with_name(f"{self._file_path.name}.partial")
        try:
            content = self._generate_content()

            async with aiofiles.open(tmp_path, "w", newline="", encoding="utf-8") as f:
                await f.write(content)

            os.replace(tmp_path, self._file_path)

        except Exception as e:
            self.error(f"Failed to export to {self._file_path}: {e}")
            with contextlib.suppress(OSError):
                tmp_path.unlink()
            raise
