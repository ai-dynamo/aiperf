# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from rich.console import Console

from aiperf.common.constants import AIPERF_DEV_MODE
from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import MetricFlags
from aiperf.common.enums.data_exporter_enums import ConsoleExporterType
from aiperf.common.factories import ConsoleExporterFactory
from aiperf.common.models import MetricResult
from aiperf.common.protocols import ConsoleExporterProtocol
from aiperf.exporters.console_metrics_exporter import ConsoleMetricsExporter
from aiperf.exporters.exporter_config import ExporterConfig
from aiperf.metrics.metric_registry import MetricRegistry


@implements_protocol(ConsoleExporterProtocol)
@ConsoleExporterFactory.register(ConsoleExporterType.INTERNAL_METRICS)
class ConsoleInternalMetricsExporter(ConsoleMetricsExporter):
    """A class that exports internal metrics to the console.

    This is a special exporter that is used to export internal metrics to the console.
    It is only applicable to internal metrics and is not applicable to user-facing metrics.
    """

    def __init__(self, exporter_config: ExporterConfig, **kwargs) -> None:
        super().__init__(exporter_config=exporter_config, **kwargs)
        self._show_internal_metrics = AIPERF_DEV_MODE and (
            exporter_config.service_config.developer.show_internal_metrics
        )

    async def export(self, console: Console) -> None:
        if not self._show_internal_metrics:
            self.debug("Internal metrics are not enabled, skipping export")
            return
        await super().export(console)

    def _should_show(self, record: MetricResult) -> bool:
        metric_class = MetricRegistry.get_class(record.tag)
        # Only show internal metrics
        return metric_class.has_flags(MetricFlags.INTERNAL)

    def _get_title(self) -> str:
        return "[yellow]NVIDIA AIPerf | Internal Metrics[/yellow]"
