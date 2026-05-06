# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared export helpers for aggregate results.

Module-level functions that any strategy can call to export confidence
or sweep aggregate artifacts. Each function uses a single asyncio.run()
call to batch concurrent file writes.
"""

import asyncio
import logging
from pathlib import Path

from aiperf.orchestrator.aggregation.base import AggregateResult

logger = logging.getLogger(__name__)

__all__ = [
    "export_confidence",
    "export_detailed",
    "export_sweep",
]


def export_confidence(
    aggregate: AggregateResult, output_dir: Path
) -> tuple[Path, Path]:
    """Export confidence aggregate to JSON and CSV.

    Args:
        aggregate: Confidence aggregate result to export
        output_dir: Directory where export files will be written

    Returns:
        Tuple of (json_path, csv_path)
    """
    from aiperf.exporters.aggregate import (
        AggregateConfidenceCsvExporter,
        AggregateConfidenceJsonExporter,
        AggregateExporterConfig,
    )

    config = AggregateExporterConfig(result=aggregate, output_dir=output_dir)

    async def _export() -> tuple[Path, Path]:
        await asyncio.to_thread(output_dir.mkdir, parents=True, exist_ok=True)
        json_path, csv_path = await asyncio.gather(
            AggregateConfidenceJsonExporter(config).export(),
            AggregateConfidenceCsvExporter(config).export(),
        )
        return json_path, csv_path

    json_path, csv_path = asyncio.run(_export())
    logger.info(f"Confidence aggregate JSON: {json_path}")
    logger.info(f"Confidence aggregate CSV: {csv_path}")
    return json_path, csv_path


def export_detailed(aggregate: AggregateResult, output_dir: Path) -> Path:
    """Export detailed/collated aggregate to JSON.

    Args:
        aggregate: Detailed aggregate result to export
        output_dir: Directory where export file will be written

    Returns:
        Path to the exported JSON file
    """
    from aiperf.exporters.aggregate import (
        AggregateDetailedJsonExporter,
        AggregateExporterConfig,
    )

    config = AggregateExporterConfig(result=aggregate, output_dir=output_dir)

    async def _export() -> Path:
        await asyncio.to_thread(output_dir.mkdir, parents=True, exist_ok=True)
        return await AggregateDetailedJsonExporter(config).export()

    json_path = asyncio.run(_export())
    logger.info(f"Collated aggregate JSON: {json_path}")
    return json_path


def export_sweep(aggregate: AggregateResult, output_dir: Path) -> tuple[Path, Path]:
    """Export sweep aggregate to JSON and CSV.

    Args:
        aggregate: Sweep aggregate result to export
        output_dir: Directory where export files will be written

    Returns:
        Tuple of (json_path, csv_path)
    """
    from aiperf.exporters.aggregate import (
        AggregateExporterConfig,
        AggregateSweepCsvExporter,
        AggregateSweepJsonExporter,
    )

    config = AggregateExporterConfig(result=aggregate, output_dir=output_dir)

    async def _export() -> tuple[Path, Path]:
        await asyncio.to_thread(output_dir.mkdir, parents=True, exist_ok=True)
        json_path, csv_path = await asyncio.gather(
            AggregateSweepJsonExporter(config).export(),
            AggregateSweepCsvExporter(config).export(),
        )
        return json_path, csv_path

    json_path, csv_path = asyncio.run(_export())
    logger.info(f"Sweep aggregate JSON: {json_path}")
    logger.info(f"Sweep aggregate CSV: {csv_path}")
    return json_path, csv_path
