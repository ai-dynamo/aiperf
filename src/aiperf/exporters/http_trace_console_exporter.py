# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import MetricFlags
from aiperf.common.exceptions import ConsoleExporterDisabled
from aiperf.exporters.console_metrics_exporter import ConsoleMetricsExporter
from aiperf.exporters.exporter_config import ExporterConfig


class HttpTraceConsoleExporter(ConsoleMetricsExporter):
    """Console exporter for HTTP trace timing metrics (k6-style breakdown).

    Gated on the `--show-trace-timing` user config flag.
    """

    title = "NVIDIA AIPerf | HTTP Trace Timing"
    require_flags = MetricFlags.HTTP_TRACE_ONLY
    exclude_flags = MetricFlags.ERROR_ONLY
    console_groups = None

    def _check_enabled(self, exporter_config: ExporterConfig) -> None:
        if not exporter_config.cfg.artifacts.show_trace_timing:
            raise ConsoleExporterDisabled(
                "HTTP trace timing is not enabled, skipping console export"
            )
