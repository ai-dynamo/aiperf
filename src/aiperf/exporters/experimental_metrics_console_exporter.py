# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from aiperf.common.enums import MetricFlags
from aiperf.common.environment import Environment
from aiperf.common.exceptions import ConsoleExporterDisabled
from aiperf.exporters.console_metrics_exporter import ConsoleMetricsExporter
from aiperf.exporters.exporter_config import ExporterConfig


class ConsoleExperimentalMetricsExporter(ConsoleMetricsExporter):
    """Console exporter for EXPERIMENTAL metrics, gated on dev mode."""

    title = "[yellow]NVIDIA AIPerf | Experimental Metrics[/yellow]"
    require_flags = MetricFlags.EXPERIMENTAL
    exclude_flags = MetricFlags.ERROR_ONLY
    console_groups = None

    def _check_enabled(self, exporter_config: ExporterConfig) -> None:
        if not (Environment.DEV.MODE and Environment.DEV.SHOW_EXPERIMENTAL_METRICS):
            raise ConsoleExporterDisabled(
                "Experimental metrics are not enabled, skipping console export"
            )
