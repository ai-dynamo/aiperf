# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from rich.text import Text
from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import DataTable
from textual.widgets.data_table import ColumnKey, RowDoesNotExist, RowKey

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.enums import WorkerStatus
from aiperf.common.models import WorkerStats
from aiperf.ui.utils import format_bytes

_logger = AIPerfLogger(__name__)


WORKER_STATUS_STYLES = {
    WorkerStatus.HEALTHY: "bold #6fbc76",
    WorkerStatus.HIGH_LOAD: "bold yellow",
    WorkerStatus.ERROR: "bold red",
    WorkerStatus.IDLE: "dim",
    WorkerStatus.STALE: "dim white",
}


class NonFocusableDataTable(DataTable, can_focus=False):
    """DataTable that cannot receive focus.
    This is done to prevent the table from focusing when the user clicks on it, which would cause the table to darken its background."""


class WorkerStatusTable(Widget):
    DEFAULT_CSS = """
    WorkerStatusTable {
        height: 1fr;
    }
    NonFocusableDataTable {
        height: 1fr;
    }
    """

    COLUMNS = ["Worker ID", "Status", "In-flight", "Completed", "Failed", "CPU", "Memory", "Total Read", "Total Write"]  # fmt: skip

    def __init__(self) -> None:
        super().__init__()
        self.data_table: NonFocusableDataTable | None = None
        self._worker_row_keys: dict[str, RowKey] = {}
        self._columns_initialized = False
        self._column_keys: dict[str, ColumnKey] = {}

    def compose(self) -> ComposeResult:
        self.data_table = NonFocusableDataTable(cursor_type="row", show_cursor=False)
        yield self.data_table

    def on_mount(self) -> None:
        if self.data_table and not self._columns_initialized:
            self._initialize_columns()

    def _initialize_columns(self) -> None:
        """Initialize table columns."""
        for col in self.COLUMNS:
            self._column_keys[col] = self.data_table.add_column(  # type: ignore
                Text(col, justify="right")
            )
        self._columns_initialized = True

    def update_single_worker(self, worker_stats: WorkerStats) -> None:
        """Update a single worker's row."""
        if not self.data_table or not self.data_table.is_mounted:
            return

        if not self._columns_initialized:
            self._initialize_columns()

        row_cells = self._format_worker_row(worker_stats)

        if worker_stats.worker_id in self._worker_row_keys:
            row_key = self._worker_row_keys[worker_stats.worker_id]
            try:
                _ = self.data_table.get_row_index(row_key)
                self._update_single_row(row_cells, row_key)
                return
            except RowDoesNotExist:
                # Row doesn't exist, fall through to add as new
                pass

        # Add new worker row
        row_key = self.data_table.add_row(*row_cells)
        self._worker_row_keys[worker_stats.worker_id] = row_key

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

    @staticmethod
    def _format_memory(memory_bytes: int | None) -> str:
        """Format memory usage."""
        return format_bytes(memory_bytes) if memory_bytes is not None else "N/A"

    @staticmethod
    def _format_cpu(cpu_usage: float | None) -> str:
        """Format CPU usage percentage."""
        return f"{cpu_usage:5.01f}%" if cpu_usage is not None else "N/A"

    def _format_worker_row(self, worker_stats: WorkerStats) -> list[Text]:
        """Format worker data into table row cells."""
        row_data = [
            Text(worker_stats.worker_id, style="bold cyan", justify="right"),
            Text(
                worker_stats.status.replace("_", " ").title(),
                style=WORKER_STATUS_STYLES[worker_stats.status],
                justify="right",
            ),
            Text(f"{worker_stats.task_stats.in_progress:,}", justify="right"),
            Text(f"{worker_stats.task_stats.completed:,}", justify="right"),
            Text(f"{worker_stats.task_stats.failed:,}", justify="right"),
        ]

        health = worker_stats.health

        if health:
            row_data.extend([
                Text(self._format_cpu(health.cpu_usage), justify="right"),
                Text(self._format_memory(health.memory_usage), justify="right"),
            ])  # fmt: skip
        else:
            row_data.extend([
                Text("N/A", justify="right"),
                Text("N/A", justify="right"),
            ])  # fmt: skip

        if health and health.io_counters:
            row_data.extend([
                Text(format_bytes(health.io_counters.read_chars), justify="right"),
                Text(format_bytes(health.io_counters.write_chars), justify="right"),
            ])  # fmt: skip
        else:
            row_data.extend([
                Text("N/A", justify="right"),
                Text("N/A", justify="right"),
            ])  # fmt: skip
        return row_data
