# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import MetricFlags
from aiperf.common.exceptions import ConsoleExporterDisabled
from aiperf.common.models import MetricResult
from aiperf.exporters.console_metrics_exporter import ConsoleMetricsExporter
from aiperf.exporters.exporter_config import ExporterConfig
from aiperf.metrics.metric_registry import MetricRegistry


class HttpTraceConsoleExporter(ConsoleMetricsExporter):
    """A class that exports HTTP trace timing metrics to the console.

    This exporter displays detailed HTTP trace timing breakdown following k6
    naming conventions: blocked, DNS lookup, connecting, sending, waiting (TTFB),
    receiving, and total duration. It is enabled via the --show-trace-timing flag.
    """

    def __init__(self, exporter_config: ExporterConfig, **kwargs) -> None:
        super().__init__(exporter_config=exporter_config, **kwargs)
        self._show_trace_timing = exporter_config.user_config.output.show_trace_timing
        if not self._show_trace_timing:
            raise ConsoleExporterDisabled(
                "HTTP trace timing is not enabled, skipping console export"
            )

    def _should_show(self, record: MetricResult) -> bool:
        metric_class = MetricRegistry.get_class(record.tag)
        # Only show HTTP trace metrics
        return metric_class.has_flags(MetricFlags.HTTP_TRACE_ONLY)

    def _get_title(self) -> str:
        return "NVIDIA AIPerf | HTTP Trace Timing"
