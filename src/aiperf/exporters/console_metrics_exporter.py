# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from collections.abc import Iterable
from datetime import datetime
from typing import ClassVar

from rich.box import Box
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

    Records are filtered by `require_flags` / `exclude_flags` and rendered as
    one table per `MetricConsoleGroup`, in the order given by `console_groups`.
    Set `console_groups = None` to render a single table containing every
    record that passes the flag filter, regardless of group — used by the
    flag-driven variants (internal, experimental, HTTP trace).

    The defaults reproduce the standard end-of-run table. Construct with explicit
    ``stat_keys`` / ``box`` / ``title`` / ``metric_filter`` to render a custom
    table (e.g. realtime ticks) without subclassing.
    """

    DEFAULT_STAT_KEYS = ("avg", "min", "max", "p99", "p90", "p50", "std")
    # Back-compat alias: the W&B data exporter (#1049) reads the stat column
    # keys off the class. Keep both names pointing at the same tuple.
    STAT_COLUMN_KEYS = DEFAULT_STAT_KEYS

    title: ClassVar[str | None] = None
    """Subclass-level title override. None means derive from the endpoint metadata."""

    require_flags: ClassVar[MetricFlags] = MetricFlags.NONE
    """Records must have ALL of these flags. `NONE` means no requirement."""

    exclude_flags: ClassVar[MetricFlags] = (
        MetricFlags.ERROR_ONLY | MetricFlags.INTERNAL | MetricFlags.EXPERIMENTAL
    )
    """Records that have ANY of these flags are hidden."""

    console_groups: ClassVar[tuple[MetricConsoleGroup, ...] | None] = (
        MetricConsoleGroup.EFFECTIVE,
        MetricConsoleGroup.ACTIVE,
        MetricConsoleGroup.USAGE,
        MetricConsoleGroup.CACHE,
        MetricConsoleGroup.PREDICTION,
        MetricConsoleGroup.AUDIO,
        MetricConsoleGroup.REASONING,
        MetricConsoleGroup.DEFAULT,
    )
    """Groups to include. `None` means no group filter (every record that
    passes the flag filter is shown)."""

    split_by_group: ClassVar[bool] = True
    """When `True`, render one table per non-empty group from `console_groups`.
    When `False`, render every matching record in a single table — useful when
    you want group-based filtering without separate tables."""

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
            exporter_config.cfg.endpoint.type if exporter_config else None
        )
        self.stat_keys = tuple(stat_keys) if stat_keys else self.DEFAULT_STAT_KEYS
        self.box = box
        if title is not None:
            self.title = title
        self.metric_filter = set(metric_filter) if metric_filter is not None else None
        if exporter_config is not None:
            self._check_enabled(exporter_config)

    def _check_enabled(self, exporter_config: ExporterConfig) -> None:
        """Raise `ConsoleExporterDisabled` if this exporter should not run."""

    async def export(self, console: Console) -> None:
        if not self._results or not self._results.records:
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
        records_list = records if isinstance(records, list) else list(records)
        if self.console_groups is None or not self.split_by_group:
            visible = [r for r in records_list if self._should_show(r)]
            if not visible:
                return None
            return self._build_table(self._get_title(), visible)

        grouped = self._group_records(records_list)
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
            grouped.setdefault(self._record_group(record), []).append(record)
        return grouped

    @staticmethod
    def _record_group(record: MetricResult) -> MetricConsoleGroup:
        """Resolve a record's console group: registered metric ClassVar first,
        then the inline `record.console_group` override (used by analyzer-
        injected results whose tags are not in MetricRegistry), defaulting to
        `DEFAULT`."""
        try:
            return MetricRegistry.get_class(record.tag).console_group
        except MetricTypeError:
            return record.console_group or MetricConsoleGroup.DEFAULT

    def _build_table(self, title: str, records: list[MetricResult]) -> Table:
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
        for record in sorted(records, key=lambda x: self._display_order(x.tag)):
            table.add_row(*self._format_row(record))

    @staticmethod
    def _display_order(tag: str) -> int:
        """Return the display order for a metric tag, defaulting to last for unregistered tags."""
        try:
            return MetricRegistry.get_class(tag).display_order or sys.maxsize
        except MetricTypeError:
            return sys.maxsize

    def _should_show(self, record: MetricResult) -> bool:
        if self.metric_filter is not None and record.tag not in self.metric_filter:
            return False
        try:
            metric_class = MetricRegistry.get_class(record.tag)
        except MetricTypeError:
            # Unregistered tag (analyzer-injected or external plugin metric):
            # an unregistered tag has no metric class and therefore no flags, so
            # it can never satisfy a `require_flags` requirement. Reject it for
            # require_flags-gated exporters (internal/experimental/HTTP-trace).
            if self.require_flags != MetricFlags.NONE:
                return False
            # Otherwise honor the inline `record.console_group` override against
            # the group filter.
            if self.console_groups is not None:
                inline_group = record.console_group or MetricConsoleGroup.DEFAULT
                if inline_group not in self.console_groups:
                    return False
            return True
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
        if self.title is not None:
            return self.title
        from aiperf.plugin import plugins

        if self._endpoint_type is None:
            return "NVIDIA AIPerf"
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
