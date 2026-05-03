# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
    user_config: UserConfig
    service_config: ServiceConfig | None
    telemetry_results: TelemetryExportData | None
    server_metrics_results: ServerMetricsResults | None = None


@dataclass(slots=True)
class FileExportInfo:
    """Information about a file export."""

    export_type: str
    file_path: Path
