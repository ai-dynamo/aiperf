# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from aiperf.common.models import ProfileResults
from aiperf.common.models.export_models import TelemetryExportData
from aiperf.common.models.server_metrics_models import ServerMetricsResults

if TYPE_CHECKING:
    from aiperf.config.config import BenchmarkConfig
    from aiperf.config.resolution.plan import BenchmarkRun


@dataclass(slots=True)
class ExporterConfig:
    """Configuration for the exporter."""

    results: ProfileResults | None
    cfg: "BenchmarkConfig"
    telemetry_results: TelemetryExportData | None
    server_metrics_results: ServerMetricsResults | None = None
    run: "BenchmarkRun | None" = None


@dataclass(slots=True)
class FileExportInfo:
    """Information about a file export."""

    export_type: str
    file_path: Path
