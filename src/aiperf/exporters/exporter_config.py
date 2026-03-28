# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from aiperf.common.models import ProfileResults
from aiperf.common.models.export_models import TelemetryExportData
from aiperf.common.models.server_metrics_models import ServerMetricsResults

if TYPE_CHECKING:
    from aiperf.config import BenchmarkConfig


@dataclass(slots=True)
class ExporterConfig:
    """Configuration for the exporter."""

    results: ProfileResults | None
    """Profiling results from the benchmark run."""

    config: BenchmarkConfig
    """Benchmark configuration used for this run."""

    telemetry_results: TelemetryExportData | None
    """Telemetry data collected during the run."""

    server_metrics_results: ServerMetricsResults | None = None
    """Server-side metrics results, if collected."""


@dataclass(slots=True)
class FileExportInfo:
    """Information about a file export."""

    export_type: str
    """Type of export (e.g., "json", "csv")."""

    file_path: Path
    """Filesystem path where the export was written."""
