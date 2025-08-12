# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from datetime import datetime

from rich.console import Console
from rich.table import Table

from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import MetricFlags
from aiperf.common.enums.data_exporter_enums import ConsoleExporterType
from aiperf.common.factories import ConsoleExporterFactory
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.models import MetricResult
from aiperf.common.protocols import ConsoleExporterProtocol
from aiperf.exporters.display_units_utils import to_display_unit
from aiperf.exporters.exporter_config import ExporterConfig
from aiperf.metrics.metric_registry import MetricRegistry


@implements_protocol(ConsoleExporterProtocol)
@ConsoleExporterFactory.register(ConsoleExporterType.METRICS)
class ConsoleMetricsExporter(AIPerfLoggerMixin):
    """A class that exports data to the console"""

    STAT_COLUMN_KEYS = ["avg", "min", "max", "p99", "p90", "p75", "std"]

    def __init__(self, exporter_config: ExporterConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self._results = exporter_config.results
        self._endpoint_type = exporter_config.user_config.endpoint.type
        self._show_internal_metrics = (
            exporter_config.user_config.output.show_internal_metrics
        )

    async def export(self, console: Console, width: int | None = None) -> None:
        if not self._results.records:
            self.warning("No records to export")
            return

        table = Table(title=self._get_title(), width=width)
        table.add_column("Metric", justify="right", style="cyan")
        for key in self.STAT_COLUMN_KEYS:
            table.add_column(key, justify="right", style="green")
        self._construct_table(table, self._results.records)

        console.print("\n")
        console.print(table)
        console.file.flush()

    def _construct_table(self, table: Table, records: list[MetricResult]) -> None:
        records = sorted(
            (to_display_unit(r, MetricRegistry()) for r in records),
            key=lambda x: MetricRegistry.get_class(x.tag).display_order or sys.maxsize,
        )
        for record in records:
            if self._should_skip(record):
                continue
            table.add_row(*self._format_row(record))

    def _should_skip(self, record: MetricResult) -> bool:
        metric_class = MetricRegistry.get_class(record.tag)
        if metric_class.has_flags(MetricFlags.ERROR_ONLY):
            return True
        return (
            metric_class.has_flags(MetricFlags.HIDDEN)
            and not self._show_internal_metrics
        )

    def _format_row(self, record: MetricResult) -> list[str]:
        metric_class = MetricRegistry.get_class(record.tag)
        display_unit = metric_class.display_unit or metric_class.unit
        row = [f"{record.header} ({display_unit})"]
        for stat in self.STAT_COLUMN_KEYS:
            value = getattr(record, stat, None)
            if value is None:
                row.append("[dim]N/A[/dim]")
                continue

            if isinstance(value, datetime):
                value = value.strftime("%Y-%m-%d %H:%M:%S")
            elif isinstance(value, int | float):
                value = f"{value:,.2f}"
            else:
                value = str(value)
            row.append(value)
        return row

    def _get_title(self) -> str:
        return f"NVIDIA AIPerf | {self._endpoint_type.metrics_title}"
