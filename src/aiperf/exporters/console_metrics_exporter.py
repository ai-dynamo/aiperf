# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from collections.abc import Iterable
from datetime import datetime
from typing import ClassVar

from rich.console import Console, Group, RenderableType
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
    """Generic console metrics exporter.

    Records are filtered by group membership and flags, then rendered as one
    table per `MetricConsoleGroup`, in the order given by `console_groups`.
    Set `console_groups = None` to skip the group filter entirely (used by
    the flag-driven subclasses: internal, experimental, HTTP trace), or
    `split_by_group = False` to keep the group filter but render a single
    merged table (used by the steady-state table).
    """

    STAT_COLUMN_KEYS = ["avg", "min", "max", "p99", "p90", "p50", "std"]

    console_groups: ClassVar[tuple[MetricConsoleGroup, ...] | None] = (
        MetricConsoleGroup.DEFAULT,
        MetricConsoleGroup.USAGE,
        MetricConsoleGroup.CACHE,
        MetricConsoleGroup.PREDICTION,
        MetricConsoleGroup.AUDIO,
        MetricConsoleGroup.REASONING,
    )
    """Groups to include. `None` means no group filter (every record that
    passes the flag filter is shown)."""

    split_by_group: ClassVar[bool] = True
    """When `True`, render one table per non-empty group from `console_groups`.
    When `False`, render every matching record in a single table — useful when
    you want group-based filtering without separate tables."""

    def __init__(self, exporter_config: ExporterConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self._results = exporter_config.results
        self._endpoint_type = exporter_config.config.endpoint.type

    async def export(self, console: Console) -> None:
        if not self._results.records:
            self.debug("No records to export")
            return

        renderable = self.get_renderable(self._results.records, console)
        if renderable is None:
            return
        self._print_renderable(console, renderable)

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
    ) -> RenderableType | None:
        records = list(records)
        if self.console_groups is None or not self.split_by_group:
            visible = [r for r in records if self._should_show(r)]
            if not visible:
                return None
            return self._build_table(self._get_title(), visible)

        grouped = self._group_records(records)
        tables = [
            self._build_table(self._get_group_title(group), grouped[group])
            for group in self.console_groups
            if grouped.get(group)
        ]
        if not tables:
            return None
        if len(tables) == 1:
            return tables[0]
        return Group(*tables)

    def _group_records(
        self, records: list[MetricResult]
    ) -> dict[MetricConsoleGroup, list[MetricResult]]:
        grouped: dict[MetricConsoleGroup, list[MetricResult]] = {}
        for record in records:
            if not self._should_show(record):
                continue
            try:
                metric_class = MetricRegistry.get_class(record.tag)
            except MetricTypeError:
                continue
            grouped.setdefault(metric_class.console_group, []).append(record)
        return grouped

    def _build_table(self, title: str, records: list[MetricResult]) -> Table:
        table = Table(title=title)
        table.add_column("Metric", justify="right", style="cyan")
        for key in self.STAT_COLUMN_KEYS:
            table.add_column(key, justify="right", style="green")
        self._construct_table(table, records)
        return table

    def _construct_table(self, table: Table, records: list[MetricResult]) -> None:
        # Records are already in display units from summarize()
        sorted_records = sorted(
            records,
            key=lambda x: self._display_order(x.tag),
        )
        for record in sorted_records:
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
        """Only show metrics that pass the group filter and are not error-only or hidden."""
        try:
            metric_class = MetricRegistry.get_class(record.tag)
        except MetricTypeError:
            return False
        if (
            self.console_groups is not None
            and metric_class.console_group not in self.console_groups
        ):
            return False
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

    def _get_group_title(self, group: MetricConsoleGroup) -> str:
        """Return the table title for a console group.

        Defaults to the main title for `DEFAULT`, and `<main>: <Group>` for any
        other group. Subclasses can override per-group naming.
        """
        if group == MetricConsoleGroup.DEFAULT:
            return self._get_title()
        return f"{self._get_title()}: {group.name.title()}"
