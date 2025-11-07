# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import suppress

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll
from textual.widget import Widget
from textual.widgets import Static
from textual.widgets.data_table import ColumnKey, RowDoesNotExist, RowKey

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.config.service_config import ServiceConfig
from aiperf.common.models.record_models import MetricResult
from aiperf.ui.dashboard.custom_widgets import MaximizableWidget, NonFocusableDataTable

_logger = AIPerfLogger(__name__)


class GPUMetricsTable(Widget):
    """Display metrics table for a single GPU."""

    DEFAULT_CSS = """
    GPUMetricsTable {
        height: auto;
        margin: 0 0 1 0;
    }
    GPUMetricsTable Static {
        background: $boost;
        padding: 0 1;
        margin: 0 0 0 0;
        text-style: bold;
    }
    NonFocusableDataTable {
        height: auto;
        max-height: 20;
    }
    """

    STATS_FIELDS = ["current", "avg", "min", "max", "p99", "p90", "p50", "std"]
    COLUMNS = ["Metric", *STATS_FIELDS]

    def __init__(
        self, endpoint: str, gpu_uuid: str, gpu_index: int, model_name: str, **kwargs
    ):
        super().__init__(**kwargs)
        self.endpoint = endpoint
        self.gpu_uuid = gpu_uuid
        self.gpu_index = gpu_index
        self.model_name = model_name
        self.data_table: NonFocusableDataTable | None = None
        self._columns_initialized = False
        self._column_keys: dict[str, ColumnKey] = {}
        self._metric_row_keys: dict[str, RowKey] = {}

    def compose(self) -> ComposeResult:
        """Compose GPU metrics table with header and data table."""
        yield Static(f"{self.endpoint} | GPU {self.gpu_index} | {self.model_name}")
        self.data_table = NonFocusableDataTable(
            cursor_type="row", show_cursor=False, zebra_stripes=True
        )
        yield self.data_table

    def on_mount(self) -> None:
        """Initialize table columns on mount."""
        if self.data_table and not self._columns_initialized:
            self._initialize_columns()

    def _initialize_columns(self) -> None:
        """Initialize table columns."""
        for i, col in enumerate(self.COLUMNS):
            if i == 0:
                self._column_keys[col] = self.data_table.add_column(  # type: ignore
                    Text(col, justify="left")
                )
            else:
                self._column_keys[col] = self.data_table.add_column(  # type: ignore
                    Text(col, justify="right")
                )
        self._columns_initialized = True

    def update(self, metrics: list[MetricResult]) -> None:
        """Update table with metrics for this GPU."""
        if not self.data_table or not self.data_table.is_mounted:
            return

        if not self._columns_initialized:
            self._initialize_columns()

        for metric in metrics:
            row_cells = self._format_metric_row(metric)
            if metric.tag in self._metric_row_keys:
                row_key = self._metric_row_keys[metric.tag]
                try:
                    _ = self.data_table.get_row_index(row_key)
                    self._update_single_row(row_cells, row_key)
                    continue
                except RowDoesNotExist:
                    _logger.warning(f"Row key {row_key} no longer exists, re-adding")

            row_key = self.data_table.add_row(*row_cells)
            self._metric_row_keys[metric.tag] = row_key

        if self.data_table:
            self.data_table.refresh()

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
        """Format metric data into table row cells.

        Header format: "GPU Power Usage | localhost:9401 | GPU 0 | Model"
        We only want the metric name (first part before |).
        Display format: "Metric name (unit)" to match console export.
        """
        metric_name = (
            metric.header.split(" | ")[0] if " | " in metric.header else metric.header
        )

        metric_display = f"{metric_name} ({metric.unit})"

        return [
            Text(metric_display, style="bold cyan", justify="left"),
            *[
                self._format_value(getattr(metric, field))
                for field in self.STATS_FIELDS
            ],
        ]

    def _format_value(self, value) -> Text:
        """Format a metric value for display (matches console exporter format)."""
        if value is None:
            return Text("N/A", justify="right", style="dim")

        if not isinstance(value, int | float):
            return Text(str(value), justify="right", style="green")

        value_str = f"{value:.2e}" if abs(value) >= 1000000 else f"{value:,.2f}"

        return Text(value_str, justify="right", style="green")


class SingleNodeView(VerticalScroll):
    """Display all GPUs for a single node directly."""

    DEFAULT_CSS = """
    SingleNodeView {
        height: 100%;
        padding: 1 2;
    }
    SingleNodeView > Static {
        background: $panel;
        padding: 0 1;
        margin: 0 0 1 0;
        text-style: bold;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.gpu_tables: dict[str, GPUMetricsTable] = {}

    def compose(self) -> ComposeResult:
        """Compose single node view (no initial widgets - GPU tables added dynamically)."""
        yield from ()

    def update(self, metrics: list[MetricResult]) -> None:
        """Update display with all GPUs from all nodes."""

        gpus = self._group_metrics_by_gpu(metrics)

        for gpu_key, gpu_metrics in gpus.items():
            if gpu_key not in self.gpu_tables:
                if not self.is_mounted:
                    continue

                endpoint, gpu_index, gpu_uuid, model_name = self._extract_gpu_info(
                    gpu_metrics[0]
                )
                gpu_table = GPUMetricsTable(endpoint, gpu_uuid, gpu_index, model_name)
                self.gpu_tables[gpu_key] = gpu_table
                self.mount(gpu_table)

            self.gpu_tables[gpu_key].update(gpu_metrics)

    def _group_metrics_by_gpu(
        self, metrics: list[MetricResult]
    ) -> dict[str, list[MetricResult]]:
        """Group metrics by GPU."""
        gpus = {}
        for metric in metrics:
            gpu_key = self._extract_gpu_key_from_tag(metric.tag)
            gpus.setdefault(gpu_key, []).append(metric)
        return gpus

    def _extract_gpu_key_from_tag(self, tag: str) -> str:
        """Extract GPU identifier from metric tag including endpoint info.

        Tag format: metric_name_dcgm_http___localhost_9400_metrics_gpu0_uuid
        Returns: dcgm_http___localhost_9400_metrics_0_uuid
        """
        if "_dcgm_" not in tag or "_gpu" not in tag:
            return "unknown"

        dcgm_and_gpu = tag.split("_dcgm_")[1]
        return dcgm_and_gpu.replace("_gpu", "_")

    def _extract_gpu_info(self, metric: MetricResult) -> tuple[str, int, str, str]:
        """Extract endpoint, GPU index, UUID, and model name from metric.

        Header format: "GPU Power Usage | localhost:9401 | GPU 0 | NVIDIA RTX 6000..."
        Tag format: metric_name_dcgm_url_gpu0_uuid
        """
        parts = metric.header.split(" | ")
        if len(parts) >= 4:
            endpoint = parts[1]
            gpu_text = parts[2]
            model_name = parts[3]
            gpu_index = int(gpu_text.split()[1]) if len(gpu_text.split()) > 1 else 0
        else:
            endpoint = "unknown"
            gpu_index = 0
            model_name = "GPU"

        tag_parts = metric.tag.split("_")
        gpu_uuid = tag_parts[-1] if tag_parts else "unknown"

        return endpoint, gpu_index, gpu_uuid, model_name


class RealtimeTelemetryDashboard(Container, MaximizableWidget):
    """Main telemetry dashboard - auto-switches between single/multi node."""

    DEFAULT_CSS = """
    RealtimeTelemetryDashboard {
        border: round $primary;
        border-title-color: $primary;
        border-title-style: bold;
        border-title-align: center;
        height: 1fr;
        layout: vertical;
    }
    #all-nodes-view {
        height: 100%;
    }
    .hidden {
        display: none;
    }
    #telemetry-status {
        height: 100%;
        width: 100%;
        color: $warning;
        text-style: italic;
        content-align: center middle;
    }
    """

    def __init__(self, service_config: ServiceConfig, **kwargs):
        super().__init__(**kwargs)
        self.service_config = service_config
        self.all_nodes_view: SingleNodeView | None = None
        self.metrics: list[MetricResult] = []
        self.border_title = "Real-Time GPU Telemetry"

    def compose(self) -> ComposeResult:
        """Compose the dashboard."""
        yield Static(
            "No telemetry data available yet. Please wait...",
            id="telemetry-status",
        )

        self.all_nodes_view = SingleNodeView(id="all-nodes-view", classes="hidden")
        yield self.all_nodes_view

    def set_status_message(self, message: str) -> None:
        """Update the status message text.

        Args:
            message: The new status message to display.
        """
        with suppress(Exception):
            status_widget = self.query_one("#telemetry-status")
            status_widget.update(message)
            status_widget.remove_class("hidden")
            self.all_nodes_view.add_class("hidden")

    def on_realtime_telemetry_metrics(self, metrics: list[MetricResult]) -> None:
        """Handle GPU telemetry metrics updates."""

        if not self.metrics:
            with suppress(Exception):
                self.query_one("#telemetry-status").add_class("hidden")
                self.all_nodes_view.remove_class("hidden")

        self.metrics = metrics
        self.all_nodes_view.update(metrics)
