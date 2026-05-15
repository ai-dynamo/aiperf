# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from datetime import datetime
from typing import ClassVar

from rich.console import Console, Group, RenderableType
from rich.table import Table

from aiperf.common.enums import MetricConsoleGroup, MetricFlags
from aiperf.common.exceptions import MetricTypeError
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.models import MetricResult
from aiperf.exporters.exporter_config import ExporterConfig
from aiperf.metrics.metric_registry import MetricRegistry


class ConsoleMetricsExporter(AIPerfLoggerMixin):
    """Generic console metrics exporter.

    Records are filtered by `require_flags` / `exclude_flags` and rendered as
    one table per `MetricConsoleGroup`, in the order given by `console_groups`.
    Set `console_groups = None` to render a single table containing every
    record that passes the flag filter, regardless of group — used by the
    flag-driven variants (internal, experimental, HTTP trace).
    """

    STAT_COLUMN_KEYS = ["avg", "min", "max", "p99", "p90", "p50", "std"]

    title: ClassVar[str | None] = None
    """Override for the exporter title. None means derive from the endpoint metadata."""

    require_flags: ClassVar[MetricFlags] = MetricFlags.NONE
    """Records must have ALL of these flags. `NONE` means no requirement."""

    exclude_flags: ClassVar[MetricFlags] = (
        MetricFlags.ERROR_ONLY | MetricFlags.INTERNAL | MetricFlags.EXPERIMENTAL
    )
    """Records that have ANY of these flags are hidden."""

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
        self._endpoint_type = exporter_config.cfg.endpoint.type
        self._check_enabled(exporter_config)

    def _check_enabled(self, exporter_config: ExporterConfig) -> None:
        """Raise `ConsoleExporterDisabled` if this exporter should not run."""

    async def export(self, console: Console) -> None:
        if not self._results.records:
            self.debug("No records to export")
            return

        renderable = self.get_renderable(self._results.records, console)
        if renderable is None:
            return
        self._print_renderable(console, renderable)

    def _print_renderable(self, console: Console, renderable: RenderableType) -> None:
        console.print("\n")
        console.print(renderable)
        console.file.flush()

    def get_renderable(
        self, records: list[MetricResult], console: Console
    ) -> RenderableType | None:
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
        def _sort_key(x: MetricResult) -> int:
            try:
                return MetricRegistry.get_class(x.tag).display_order or sys.maxsize
            except MetricTypeError:
                return sys.maxsize

        for record in sorted(records, key=_sort_key):
            table.add_row(*self._format_row(record))

    def _should_show(self, record: MetricResult) -> bool:
        try:
            metric_class = MetricRegistry.get_class(record.tag)
        except MetricTypeError:
            return False
        if (
            self.console_groups is not None
            and metric_class.console_group not in self.console_groups
        ):
            return False
        if self.require_flags != MetricFlags.NONE and not metric_class.has_flags(
            self.require_flags
        ):
            return False
        return metric_class.missing_flags(self.exclude_flags)

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
            elif isinstance(value, int | float):
                value = f"{value:,.2f}"
            else:
                value = str(value)
            row.append(value)
        return row

    def _get_title(self) -> str:
        if self.title is not None:
            return self.title
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
