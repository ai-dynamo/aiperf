# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from collections.abc import Iterable
from datetime import datetime

from rich.console import Console, RenderableType
from rich.table import Table

from aiperf.common.enums import MetricConsoleGroup, MetricFlags
from aiperf.common.exceptions import MetricTypeError
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.models import MetricResult
from aiperf.exporters.exporter_config import ExporterConfig
from aiperf.metrics.cache_reporting_hint import (
    CACHE_REPORTING_HINT,
    usage_without_cache_in_results,
)
from aiperf.metrics.metric_registry import MetricRegistry


class ConsoleMetricsExporter(AIPerfLoggerMixin):
    """A class that exports data to the console"""

    STAT_COLUMN_KEYS = ["avg", "min", "max", "p99", "p90", "p50", "std"]

    def __init__(self, exporter_config: ExporterConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self._results = exporter_config.results
        self._endpoint_type = exporter_config.config.endpoint.type

    async def export(self, console: Console) -> None:
        if not self._results.records:
            self.debug("No records to export")
            return

        self._print_renderable(
            console, self.get_renderable(self._results.records, console)
        )

        # Persist the cache-reporting hint in the final summary (the mid-run log
        # line is ephemeral in dashboard mode); see cache_reporting_hint.
        if usage_without_cache_in_results(self._results.records):
            console.print(f"\n[yellow]{CACHE_REPORTING_HINT}[/yellow]")

    def _print_renderable(self, console: Console, renderable: RenderableType) -> None:
        console.print("\n")
        console.print(renderable)
        console.file.flush()

    def get_renderable(
        self, records: Iterable[MetricResult], console: Console
    ) -> RenderableType:
        table = Table(title=self._get_title())
        table.add_column("Metric", justify="right", style="cyan")
        for key in self.STAT_COLUMN_KEYS:
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

    # Flags that exclude a metric from the console summary table. Shared with
    # WandbDataExporter so the uploaded table mirrors the console's visibility.
    EXCLUDE_FLAGS = (
        MetricFlags.ERROR_ONLY | MetricFlags.INTERNAL | MetricFlags.EXPERIMENTAL
    )

    @classmethod
    def should_show_metric_class(cls, metric_class: type) -> bool:
        """Whether a metric class belongs in the console summary table.

        Single source of truth for console visibility, reused by
        WandbDataExporter so its uploaded summary table mirrors the console.
        """
        # NO_CONSOLE was retired in favour of ``MetricConsoleGroup.NONE``;
        # treat any metric whose console_group is NONE as if it had the
        # legacy NO_CONSOLE flag set so the filter still excludes it from
        # the console table.
        if getattr(metric_class, "console_group", None) == MetricConsoleGroup.NONE:
            return False
        return metric_class.missing_flags(cls.EXCLUDE_FLAGS)

    def _should_show(self, record: MetricResult) -> bool:
        """Only show metrics that are not error-only or hidden."""
        try:
            metric_class = MetricRegistry.get_class(record.tag)
        except MetricTypeError:
            return True
        return self.should_show_metric_class(metric_class)

    def _format_row(self, record: MetricResult) -> list[str]:
        delimiter = "\n" if len(record.header) > 30 else " "
        row = [f"{record.header}{delimiter}({record.unit})"]
        for stat in self.STAT_COLUMN_KEYS:
            value = getattr(record, stat, None)
            if value is None:
                row.append("[dim]N/A[/dim]")
                continue

            if isinstance(value, datetime):
                value = value.strftime("%Y-%m-%d %H:%M:%S")
            elif isinstance(value, (int, float)):
                value = f"{value:,.2f}"
            else:
                value = str(value)
            row.append(value)
        return row

    def _get_title(self) -> str:
        from aiperf.plugin import plugins

        metadata = plugins.get_endpoint_metadata(self._endpoint_type)
        return f"NVIDIA AIPerf | {metadata.metrics_title}"
