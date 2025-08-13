# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
from collections import Counter

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.css.query import NoMatches, WrongType
from textual.events import Click
from textual.widgets import Label

from aiperf.common.enums import WorkerStatus
from aiperf.common.models import WorkerStats
from aiperf.ui.dashboard.worker_status_table import WorkerStatusTable


class WorkerDashboard(Container):
    DEFAULT_CSS = """
    WorkerDashboard {
        border: round $primary;
        border-title-color: $primary;
        border-title-style: bold;
        height: 1fr;
        layout: vertical;
    }

    #summary-content {
        height: 1;
        layout: horizontal;
        align: left middle;
        margin: 0 1 0 1;
    }

    .summary-item { margin: 0 1; }
    .summary-title { text-style: bold; }
    .summary-healthy { color: $success; text-style: bold; }
    .summary-high-load { color: $warning; text-style: bold; }
    .summary-error { color: $error; text-style: bold; }
    .summary-idle { color: $text-muted; }
    .summary-stale { color: $surface-darken-1; }

    #table-section {
        height: 1fr;
        margin: 1 0 0 1;
    }
    """

    ALLOW_MAXIMIZE = True

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.worker_stats: dict[str, WorkerStats] = {}
        self.table_widget: WorkerStatusTable | None = None
        self.border_title = "Worker Status"

    def compose(self) -> ComposeResult:
        with Vertical(id="worker-dashboard-content"):
            with Horizontal(id="summary-content"):
                yield Label("Summary: ", classes="summary-item summary-title")
                for status in WorkerStatus:
                    # Create a label for each possible status type
                    yield Label(
                        f"0 {status.replace('_', ' ')}",
                        id=f"{status.replace('_', '-').lower()}-count",
                        classes=f"summary-item summary-{status.replace('_', '-').lower()}",
                    )
                    yield Label("•", classes="summary-item")

            with Container(id="table-section"):
                self.table_widget = WorkerStatusTable()
                yield self.table_widget

    def on_worker_update(self, worker_id: str, worker_stats: WorkerStats) -> None:
        """Handle individual worker updates."""
        self.worker_stats[worker_id] = worker_stats
        if self.table_widget:
            self.table_widget.update_single_worker(worker_stats)

    def on_worker_status_summary(
        self, worker_status_summary: dict[str, WorkerStatus]
    ) -> None:
        """Handle worker status summary updates."""
        summary = Counter(worker_status_summary.values())

        # For each status type, update the label with the count of workers in that status
        for status in WorkerStatus:
            with contextlib.suppress(WrongType, NoMatches):
                self.query_one(
                    f"#{status.replace('_', '-').lower()}-count", Label
                ).update(f"{summary[status]} {status.replace('_', ' ')}")

    def on_click(self, event: Click) -> None:
        """Handle click events to toggle the maximize state of the widget."""
        if event.chain == 2:
            event.stop()
            self.toggle_maximize()

    def toggle_maximize(self) -> None:
        """Toggle the maximize state of the widget."""
        if not self.is_maximized:
            self.screen.maximize(self)
        else:
            self.screen.minimize()
