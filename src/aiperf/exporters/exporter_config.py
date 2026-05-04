# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.models import ProfileResults
from aiperf.common.models.export_models import TelemetryExportData
from aiperf.common.models.server_metrics_models import ServerMetricsResults


@dataclass(slots=True)
class ExporterConfig:
    """Configuration for the exporter."""

    results: ProfileResults | None
    """Profiling results from the benchmark run."""

    user_config: UserConfig
    """User-facing configuration for this run."""

    service_config: ServiceConfig | None
    """Service-level configuration for this run."""

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
