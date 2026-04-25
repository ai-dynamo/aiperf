# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
from collections import Counter

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.css.query import NoMatches, WrongType
from textual.widgets import Label

from aiperf.common.enums import WorkerStartupState, WorkerStatus
from aiperf.common.models import WorkerGroupStats, WorkerStats
from aiperf.ui.dashboard.custom_widgets import MaximizableWidget
from aiperf.ui.dashboard.worker_status_table import WorkerStatusTable


class WorkerDashboard(Container, MaximizableWidget):
    DEFAULT_CSS = """
    WorkerDashboard {
        border: round $primary;
        border-title-color: $primary;
        border-title-style: bold;
        border-title-align: center;
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

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.worker_groups: dict[str, WorkerGroupStats] = {}
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

    def on_worker_group_update(
        self, group_id: str, group_stats: WorkerGroupStats
    ) -> None:
        """Handle a worker-group rolled-up snapshot."""
        self.worker_groups[group_id] = group_stats
        if self.table_widget:
            self.table_widget.update_group(group_id, group_stats)

        # Recompute the per-worker-status counts across all known groups so
        # the summary chips stay in sync with whatever the WGM is reporting.
        statuses: list[WorkerStatus] = []
        for group in self.worker_groups.values():
            for child in group.workers.values():
                statuses.append(child.status)
        summary = Counter(statuses)
        for status in WorkerStatus:
            with contextlib.suppress(WrongType, NoMatches):
                self.query_one(
                    f"#{status.replace('_', '-').lower()}-count", Label
                ).update(f"{summary[status]} {status.replace('_', ' ')}")

    def on_worker_update(self, worker_id: str, worker_stats: WorkerStats) -> None:
        """Legacy ON_WORKER_UPDATE bridge — superseded by on_worker_group_update.

        Kept as a no-op so the fake-in-process WorkerTrackerMixin path that still
        emits ``ON_WORKER_UPDATE`` for external listeners does not crash here.
        """
        return

    def on_worker_status_summary(
        self,
        worker_status_summary: dict[str, WorkerStatus],
        worker_startup_states: dict[str, WorkerStartupState] | None = None,
    ) -> None:
        """Legacy ON_WORKER_STATUS_SUMMARY bridge — superseded by on_worker_group_update.

        Kept as a no-op for symmetry with on_worker_update.
        """
        return
