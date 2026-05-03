# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from contextlib import suppress
from datetime import datetime
from typing import Any

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Container
from textual.widget import Widget
from textual.widgets import Static
from textual.widgets.data_table import ColumnKey, RowDoesNotExist, RowKey

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.config.service_config import ServiceConfig
from aiperf.common.enums import MetricFlags
from aiperf.common.environment import Environment
from aiperf.common.models.record_models import MetricResult
from aiperf.metrics.metric_registry import MetricRegistry
from aiperf.ui.dashboard.custom_widgets import MaximizableWidget, NonFocusableDataTable

_logger = AIPerfLogger(__name__)


class RealtimeMetricsTable(Widget):
    DEFAULT_CSS = """
    RealtimeMetricsTable {
        height: 1fr;
    }
    NonFocusableDataTable {
        height: 1fr;
    }
    """

    STATS_FIELDS = ["avg", "min", "max", "p99", "p90", "p50", "std"]
    COLUMNS = ["Metric", *STATS_FIELDS]

    def __init__(self, service_config: ServiceConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self.service_config = service_config
        self.data_table: NonFocusableDataTable | None = None
        self._columns_initialized = False
        self._column_keys: dict[str, ColumnKey] = {}
        self._metric_row_keys: dict[str, RowKey] = {}
        self.metrics: list[MetricResult] = []

    def compose(self) -> ComposeResult:
        self.data_table = NonFocusableDataTable(
            cursor_type="row", show_cursor=False, zebra_stripes=True
        )
        yield self.data_table

    def on_mount(self) -> None:
        if self.data_table and not self._columns_initialized:
            self._initialize_columns()
            if self.metrics:
                self.update(self.metrics)

    def _should_skip(self, metric: MetricResult) -> bool:
        """Determine if a metric should be skipped.

        INTERNAL and EXPERIMENTAL metrics are already filtered upstream by
        summarize(), so only ERROR_ONLY and NO_CONSOLE need filtering here.
        """
        metric_class = MetricRegistry.get_class(metric.tag)
        if metric_class.has_flags(MetricFlags.ERROR_ONLY):
            return True
        return (
            metric_class.has_flags(MetricFlags.NO_CONSOLE)
            and not Environment.DEV.SHOW_INTERNAL_METRICS
        )

    def _initialize_columns(self) -> None:
        """Initialize table columns."""
        for col in self.COLUMNS:
            self._column_keys[col] = self.data_table.add_column(  # type: ignore
                Text(col, justify="right")
            )
        self._columns_initialized = True

    def update(self, metrics: list[MetricResult]) -> None:
        """Update the metrics table."""
        self.metrics = metrics

        if not self.data_table or not self.data_table.is_mounted:
            return

        if not self._columns_initialized:
            self._initialize_columns()

        metrics = [
            metric
            for metric in sorted(
                metrics,
                key=lambda m: MetricRegistry.get_class(m.tag).display_order
                or sys.maxsize,
            )
            if not self._should_skip(metric)
        ]
        _logger.debug(lambda: f"Updating metrics table with {len(metrics)} metrics")
        for metric in metrics:
            row_cells = self._format_metric_row(metric)
            if metric.tag in self._metric_row_keys:
                row_key = self._metric_row_keys[metric.tag]
                try:
                    _ = self.data_table.get_row_index(row_key)
                    self._update_single_row(row_cells, row_key)
                    continue
                except RowDoesNotExist:
                    # Row doesn't exist, fall through to add as new
                    pass

            # Add new metric row
            row_key = self.data_table.add_row(*row_cells)
            self._metric_row_keys[metric.tag] = row_key

    def _update_single_row(self, row_cells: list[Text], row_key: RowKey) -> None:
        """Update a single row's cells."""
        for col_name, cell_value in zip(self.COLUMNS, row_cells, strict=True):
            try:
                self.data_table.update_cell(  # type: ignore
                    row_key, self._column_keys[col_name], cell_value, update_width=True
                )
            except Exception as e:
                _logger.warning(
                    f"Error updating cell {col_name} with value {cell_value}: {e!r}"
                )

    def _format_metric_row(self, metric: MetricResult) -> list[Text]:
        """Format worker data into table row cells.

        Note: Metrics are pre-converted to display units by summarize(),
        so values can be used directly without conversion.
        """
        metric_class = MetricRegistry.get_class(metric.tag)
        short_header = metric_class.short_header or metric_class.header
        # Use the metric's unit directly (already converted to display unit)
        if not metric_class.short_header_hide_unit and metric.unit:
            short_header = f"{short_header} ({metric.unit})"
        return [
            Text(
                short_header,
                style="bold cyan",
                justify="right",
            ),
            *[
                self._format_metric_value(getattr(metric, field))
                for field in self.STATS_FIELDS
            ],
        ]

    def _format_metric_value(self, value: Any | None) -> Text:
        """Format a metric value.

        Note: Values are pre-converted to display units by summarize(),
        so no unit conversion is needed here.
        """
        if value is None:
            return Text("N/A", justify="right", style="dim")

        if isinstance(value, datetime):
            value_str = value.strftime("%Y-%m-%d %H:%M:%S")
        elif isinstance(value, int | float):
            value_str = f"{value:,.2f}"
        else:
            value_str = str(value)
        return Text(value_str, justify="right", style="green")


class RealtimeMetricsDashboard(Container, MaximizableWidget):
    DEFAULT_CSS = """
    RealtimeMetricsDashboard {
        border: round $primary;
        border-title-color: $primary;
        border-title-style: bold;
        border-title-align: center;
        height: 1fr;
        layout: vertical;
        margin: 0 0 0 0;
        scrollbar-gutter: auto;
    }
    #realtime-metrics-status {
        height: 100%;
        width: 100%;
        color: $warning;
        text-style: italic;
        content-align: center middle;
    }
    .hidden {
        display: none;
    }
    """

    def __init__(self, service_config: ServiceConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self.service_config = service_config
        self.metrics_table: RealtimeMetricsTable | None = None
        self.metrics: list[MetricResult] = []
        self.border_title = "Real-Time Metrics"

    def compose(self) -> ComposeResult:
        self.metrics_table = RealtimeMetricsTable(
            service_config=self.service_config, id="metrics-table", classes="hidden"
        )
        yield self.metrics_table
        yield Static(
            "No metrics available yet. Please wait...",
            id="realtime-metrics-status",
        )

    def on_realtime_metrics(self, metrics: list[MetricResult]) -> None:
        """Handle metrics updates."""
        if not self.metrics:
            with suppress(Exception):
                self.query_one("#metrics-table", RealtimeMetricsTable).remove_class("hidden")  # fmt: skip
                self.query_one("#realtime-metrics-status", Static).add_class("hidden")

        self.metrics = metrics
        if self.metrics_table:
            self.metrics_table.update(metrics)
