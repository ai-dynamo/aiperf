# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aiperf.common.models import ProfileResults
from aiperf.common.models.export_models import TelemetryExportData
from aiperf.common.models.server_metrics_models import ServerMetricsResults

if TYPE_CHECKING:
    from aiperf.analysis.energy_analyzer import EnergyEfficiencySummary
    from aiperf.config import BenchmarkConfig
    from aiperf.post_processors.steady_state_analyzer import SteadyStateSummary


@dataclass(slots=True)
class ExporterConfig:
    """Configuration for the exporter.

    ``cfg`` is the canonical attribute name; ``config`` is retained as an
    alias because parts of the codebase (and most of the tests) still spell
    it the old way. Both point at the same underlying object — see
    :meth:`__post_init__`.
    """

    results: ProfileResults | None
    """Profiling results from the benchmark run."""

    config: BenchmarkConfig | None = None
    """Benchmark configuration used for this run (alias of ``cfg``)."""

    cfg: BenchmarkConfig | None = None
    """Benchmark configuration used for this run (alias of ``config``)."""

    telemetry_results: TelemetryExportData | None = None
    """Telemetry data collected during the run."""

    server_metrics_results: ServerMetricsResults | None = None
    """Server-side metrics results, if collected."""

    steady_state_results: SteadyStateSummary | None = None
    """Steady-state windowed metrics results, if computed."""

    energy_efficiency_results: EnergyEfficiencySummary | None = None
    """Energy efficiency metrics results, if computed."""

    run: Any = None
    """Optional ``BenchmarkRun`` reference (mlflow exporter reads benchmark_id off this)."""

    def __post_init__(self) -> None:
        # Synchronize the two aliases so call sites can read either.
        if self.cfg is None and self.config is not None:
            self.cfg = self.config
        elif self.config is None and self.cfg is not None:
            self.config = self.cfg
        if self.cfg is None:
            raise TypeError(
                "ExporterConfig requires either ``cfg=`` or ``config=`` to be set."
            )


@dataclass(slots=True)
class FileExportInfo:
    """Information about a file export."""

    export_type: str
    """Type of export (e.g., "json", "csv")."""

    file_path: Path
    """Filesystem path where the export was written."""
