# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from rich.text import Text
from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets.data_table import ColumnKey, RowDoesNotExist, RowKey

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.enums import WorkerStartupState, WorkerStatus
from aiperf.common.models import WorkerGroupStats, WorkerStats
from aiperf.ui.dashboard.custom_widgets import NonFocusableDataTable
from aiperf.ui.utils import format_bytes

_logger = AIPerfLogger(__name__)


WORKER_STATUS_STYLES = {
    WorkerStatus.HEALTHY: "bold #6fbc76",
    WorkerStatus.HIGH_LOAD: "bold yellow",
    WorkerStatus.ERROR: "bold red",
    WorkerStatus.IDLE: "dim",
    WorkerStatus.STALE: "dim white",
}


class WorkerStatusTable(Widget):
    DEFAULT_CSS = """
    WorkerStatusTable {
        height: 1fr;
    }
    NonFocusableDataTable {
        height: 1fr;
    }
    """

    COLUMNS = ["Group ID", "Status", "Ready", "In-flight", "Completed", "Failed", "CPU%", "Memory"]  # fmt: skip

    def __init__(self) -> None:
        super().__init__()
        self.data_table: NonFocusableDataTable | None = None
        self._group_row_keys: dict[str, RowKey] = {}
        self._columns_initialized = False
        self._column_keys: dict[str, ColumnKey] = {}

    def compose(self) -> ComposeResult:
        self.data_table = NonFocusableDataTable(
            cursor_type="row", show_cursor=False, zebra_stripes=True
        )
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

    def update_group(self, group_id: str, group: WorkerGroupStats) -> None:
        """Update a single worker-group's row."""
        if not self.data_table or not self.data_table.is_mounted:
            return

        if not self._columns_initialized:
            self._initialize_columns()

        row_cells = self._format_group_row(group_id, group)

        if group_id in self._group_row_keys:
            row_key = self._group_row_keys[group_id]
            try:
                _ = self.data_table.get_row_index(row_key)
                self._update_single_row(row_cells, row_key)
                return
            except RowDoesNotExist:
                # Row doesn't exist, fall through to add as new
                pass

        row_key = self.data_table.add_row(*row_cells)
        self._group_row_keys[group_id] = row_key

    def update_single_worker(self, worker_stats: WorkerStats) -> None:
        """No-op shim for legacy ON_WORKER_UPDATE callers.

        Superseded by ``update_group``; the WorkerTrackerMixin now folds
        per-worker WORKER_HEALTH into a synthetic ``local`` group and fires
        ``ON_WORKER_GROUP_UPDATE`` for both real-WGM and fake-in-process modes.
        Kept on the class to avoid AttributeError in the legacy hook bridge.
        """
        return

    def _update_single_row(self, row_cells: list[Text], row_key: RowKey) -> None:
        """Update a single row's cells."""
        for col_name, cell_value in zip(self.COLUMNS, row_cells, strict=True):
            try:
                self.data_table.update_cell(  # type: ignore
                    row_key, self._column_keys[col_name], cell_value, update_width=True
                )
            except Exception as e:  # noqa: BLE001 - best-effort UI cell update; any textual/rich error is logged and skipped
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

    def _format_group_row(self, group_id: str, group: WorkerGroupStats) -> list[Text]:
        """Format worker-group data into table row cells."""
        status_text = group.status.replace("_", " ").title()
        if (
            group.startup_state is not None
            and group.startup_state != WorkerStartupState.READY
        ):
            startup_text = group.startup_state.replace("_", " ").title()
            status_text = f"{status_text} ({startup_text})"

        denom = group.declared_workers or len(group.workers)
        ready_text = f"{group.ready_workers}/{denom}"

        health = group.health
        cpu_text = self._format_cpu(health.cpu_usage) if health else "N/A"
        mem_text = self._format_memory(health.memory_usage) if health else "N/A"

        return [
            Text(group_id, style="bold cyan", justify="right"),
            Text(
                status_text,
                style=WORKER_STATUS_STYLES.get(group.status, ""),
                justify="right",
            ),
            Text(ready_text, justify="right"),
            Text(f"{group.task_stats.in_progress:,}", justify="right"),
            Text(f"{group.task_stats.completed:,}", justify="right"),
            Text(f"{group.task_stats.failed:,}", justify="right"),
            Text(cpu_text, justify="right"),
            Text(mem_text, justify="right"),
        ]
