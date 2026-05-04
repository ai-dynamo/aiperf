# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from collections.abc import Iterable
from datetime import datetime

from rich.box import Box
from rich.console import Console, RenderableType
from rich.table import Table

from aiperf.common.enums import MetricFlags
from aiperf.common.exceptions import MetricTypeError
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.models import MetricResult
from aiperf.exporters.exporter_config import ExporterConfig
from aiperf.metrics.metric_registry import MetricRegistry


class ConsoleMetricsExporter(AIPerfLoggerMixin):
    """Render benchmark metrics to a Rich table on the console.

    The defaults reproduce the standard end-of-run table. Construct with explicit
    ``stat_keys`` / ``box`` / ``title`` / ``metric_filter`` to render a custom
    table (e.g. realtime ticks) without subclassing.
    """

    DEFAULT_STAT_KEYS = ("avg", "min", "max", "p99", "p90", "p50", "std")

    def __init__(
        self,
        exporter_config: ExporterConfig | None = None,
        *,
        stat_keys: Iterable[str] | None = None,
        box: Box | None = None,
        title: str | None = None,
        metric_filter: Iterable[str] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._results = exporter_config.results if exporter_config else None
        self._endpoint_type = (
            exporter_config.user_config.endpoint.type if exporter_config else None
        )
        self.stat_keys = tuple(stat_keys) if stat_keys else self.DEFAULT_STAT_KEYS
        self.box = box
        self.title = title
        self.metric_filter = set(metric_filter) if metric_filter is not None else None

    async def export(self, console: Console) -> None:
        if not self._results or not self._results.records:
            self.debug("No records to export")
            return

        self._print_renderable(
            console, self.get_renderable(self._results.records, console)
        )

    def _print_renderable(self, console: Console, renderable: RenderableType) -> None:
        console.print("\n")
        console.print(renderable)
        console.file.flush()

    def get_renderable(
        self, records: Iterable[MetricResult], console: Console
    ) -> RenderableType:
        title = self.title if self.title is not None else self._get_title()
        table_kwargs: dict = {"title": title}
        if self.box is not None:
            table_kwargs["box"] = self.box
        table = Table(**table_kwargs)
        table.add_column("Metric", justify="right", style="cyan")
        for key in self.stat_keys:
            table.add_column(key, justify="right", style="green")
        self._construct_table(table, records)
        return table

    def _construct_table(self, table: Table, records: Iterable[MetricResult]) -> None:
        # Records are already in display units from summarize()
        sorted_records = sorted(
            records,
            key=lambda x: self._display_order(x.tag),
        )
        for record in sorted_records:
            if not self._should_show(record):
                continue
            table.add_row(*self._format_row(record))

    @staticmethod
    def _display_order(tag: str) -> int:
        """Return the display order for a metric tag, defaulting to last for unregistered tags."""
        try:
            return MetricRegistry.get_class(tag).display_order or sys.maxsize
        except MetricTypeError:
            return sys.maxsize

    def _should_show(self, record: MetricResult) -> bool:
        """Only show metrics that are not error-only or hidden."""
        if self.metric_filter is not None and record.tag not in self.metric_filter:
            return False
        try:
            metric_class = MetricRegistry.get_class(record.tag)
        except MetricTypeError:
            return True
        return metric_class.missing_flags(
            MetricFlags.ERROR_ONLY
            | MetricFlags.NO_CONSOLE
            | MetricFlags.INTERNAL
            | MetricFlags.EXPERIMENTAL
        )

    def _format_row(self, record: MetricResult) -> list[str]:
        delimiter = "\n" if len(record.header) > 30 else " "
        row = [f"{record.header}{delimiter}({record.unit})"]
        for stat in self.stat_keys:
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
        from aiperf.plugin import plugins

        if self._endpoint_type is None:
            return "NVIDIA AIPerf"
        metadata = plugins.get_endpoint_metadata(self._endpoint_type)
        return f"NVIDIA AIPerf | {metadata.metrics_title}"
