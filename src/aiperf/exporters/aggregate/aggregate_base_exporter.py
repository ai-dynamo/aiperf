# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Base class for aggregate exporters."""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import aiofiles

from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.scenario.submission_outcome import (
    CONTEXT_OVERFLOW_REASON,
    RUN_CANCELLED_REASON,
    compute_submission_outcome,
)
from aiperf.orchestrator.aggregation.base import AggregateResult

__all__ = [
    "CONTEXT_OVERFLOW_REASON",
    "RUN_CANCELLED_REASON",
    "AggregateBaseExporter",
    "AggregateExporterConfig",
    "_build_run_metadata_dict",
    "compute_submission_outcome",
]


def _build_run_metadata_dict(
    *,
    scenario_name: str | None,
    submission_valid: bool | None,
    submission_invalid_reasons: list[str] | None = None,
) -> dict:
    """Build the scenario-submission sub-dict for the aggregate export.

    Mirrors the single-run ``RunInfo`` submission surface for the multi-run
    (``--num-profile-runs > 1`` / sweep) aggregate export. Returns an empty
    dict when ``scenario_name`` is ``None`` so non-scenario runs are not
    polluted with submission-tracking fields. When ``scenario_name`` is set,
    returns the ``scenario`` name plus a coerced ``submission_valid`` bool, and
    includes ``submission_invalid_reasons`` only when that list is non-empty.

    Args:
        scenario_name: Active scenario identifier, or ``None`` for a
            non-scenario run.
        submission_valid: Whether the run is a valid scenario submission.
            Coerced to ``bool`` (``None`` becomes ``False``) when emitted.
        submission_invalid_reasons: Machine-readable reason codes (e.g.
            ``"unsafe_override"``, ``"context_overflow_rate_exceeded"``).

    Returns:
        A dict suitable for merging into the top-level aggregate JSON output.
    """
    md: dict = {}
    if scenario_name is not None:
        md["scenario"] = scenario_name
        md["submission_valid"] = bool(submission_valid)
        if submission_invalid_reasons:
            md["submission_invalid_reasons"] = list(submission_invalid_reasons)
    return md


@dataclass(slots=True)
class AggregateExporterConfig:
    """Configuration for aggregate exporters.

    Simpler than ExporterConfig because aggregate exports don't need:
    - ProfileResults (single-run data)
    - TelemetryExportData (per-run telemetry)
    - ServerMetricsResults (per-run server metrics)
    - Full benchmark config (just need output directory)

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
        await asyncio.to_thread(self._output_dir.mkdir, parents=True, exist_ok=True)

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
