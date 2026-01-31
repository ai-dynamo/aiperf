# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from aiperf.common.enums import MetricFlags
from aiperf.common.environment import Environment
from aiperf.common.exceptions import ConsoleExporterDisabled
from aiperf.common.models import MetricResult
from aiperf.exporters.console_metrics_exporter import ConsoleMetricsExporter
from aiperf.exporters.exporter_config import ExporterConfig
from aiperf.metrics.metric_registry import MetricRegistry


class ConsoleExperimentalMetricsExporter(ConsoleMetricsExporter):
    """A class that exports experimental metrics to the console.

    This is a special exporter that is used to export experimental metrics to the console.
    """

    def __init__(self, exporter_config: ExporterConfig, **kwargs) -> None:
        super().__init__(exporter_config=exporter_config, **kwargs)
        self._show_experimental_metrics = (
            Environment.DEV.MODE and Environment.DEV.SHOW_EXPERIMENTAL_METRICS
        )
        if not self._show_experimental_metrics:
            raise ConsoleExporterDisabled(
                "Experimental metrics are not enabled, skipping console export"
            )

    def _should_show(self, record: MetricResult) -> bool:
        metric_class = MetricRegistry.get_class(record.tag)
        # Only show experimental metrics
        return metric_class.has_flags(MetricFlags.EXPERIMENTAL)

    def _get_title(self) -> str:
        return "[yellow]NVIDIA AIPerf | Experimental Metrics[/yellow]"
