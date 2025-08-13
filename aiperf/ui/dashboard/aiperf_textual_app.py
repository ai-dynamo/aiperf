# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import signal

from rich.console import RenderableType
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Footer

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.constants import AIPERF_DEV_MODE
from aiperf.common.enums import WorkerStatus
from aiperf.common.models import RecordsStats, RequestsStats, WorkerStats
from aiperf.ui.dashboard.aiperf_theme import AIPERF_THEME
from aiperf.ui.dashboard.custom_header import CustomHeader
from aiperf.ui.dashboard.progress_dashboard import ProgressDashboard
from aiperf.ui.dashboard.rich_log_viewer import RichLogViewer
from aiperf.ui.dashboard.worker_dashboard import WorkerDashboard

_logger = AIPerfLogger(__name__)


class AIPerfTextualApp(App):
    """
    AIPerf Textual App.

    This is the main application class for the Textual UI. It is responsible for
    composing the application layout and handling the application commands.
    """

    ENABLE_COMMAND_PALETTE = False
    """Disable the command palette that is enabled by default in Textual."""

    CSS = """
    #main-container {
        height: 100%;
    }
    #dashboard-section {
        height: 3fr;
        min-height: 14;
    }
    Tab {
        text-style: bold;
    }
    #logs-section {
        height: 2fr;
        max-height: 16;
    }
    #logs-section.hidden {
        display: none;
    }
    #dashboard-section.logs-hidden {
        height: 1fr;
    }
    """

    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("1", "minimize_all_panels", "Overview"),
        ("2", "toggle_maximize('progress')", "Progress"),
        ("3", "toggle_maximize('workers')", "Workers"),
        ("4", "toggle_maximize('logs')", "Logs"),
        ("escape", "restore_all_panels", "Restore View"),
        ("l", "toggle_hide_log_viewer", "Toggle Logs"),
    ]

    def __init__(self) -> None:
        super().__init__()

        self.title = "NVIDIA AIPerf"
        if AIPERF_DEV_MODE:
            self.title = "NVIDIA AIPerf (Developer Mode)"

        self.log_viewer: RichLogViewer | None = None
        self.progress_dashboard: ProgressDashboard | None = None
        self.worker_dashboard: WorkerDashboard | None = None
        self.profile_results: list[RenderableType] = []
        self.logs_hidden = False

    def on_mount(self) -> None:
        self.register_theme(AIPERF_THEME)
        self.theme = AIPERF_THEME.name

    def compose(self) -> ComposeResult:
        """Compose the full application layout."""
        yield CustomHeader()

        # NOTE: SIM117 is disabled because nested with statements are recommended for textual ui layouts
        with Vertical(id="main-container"):
            with Container(id="dashboard-section"):  # noqa: SIM117
                with Horizontal(id="overview-section"):
                    self.progress_dashboard = ProgressDashboard(id="progress")
                    yield self.progress_dashboard
                    self.worker_dashboard = WorkerDashboard(id="workers")
                    yield self.worker_dashboard

            with Container(id="logs-section"):
                self.log_viewer = RichLogViewer(id="logs")
                yield self.log_viewer

        yield Footer()

    async def on_profiling_progress(self, profiling_stats: RequestsStats) -> None:
        """Update the overview and progress dashboard with the profiling progress."""
        if self.progress_dashboard:
            self.progress_dashboard.on_profiling_progress(profiling_stats)

    async def on_warmup_progress(self, warmup_stats: RequestsStats) -> None:
        """Update the overview and progress dashboard with the warmup progress."""
        if self.progress_dashboard:
            self.progress_dashboard.on_warmup_progress(warmup_stats)

    async def on_records_progress(self, records_stats: RecordsStats) -> None:
        """Update the overview and progress dashboard with the records progress."""
        if self.progress_dashboard:
            self.progress_dashboard.on_records_progress(records_stats)

    async def on_worker_status_summary(
        self, worker_status_summary: dict[str, WorkerStatus]
    ) -> None:
        """Update the overview and worker dashboard with the worker status summary."""
        if self.worker_dashboard:
            self.worker_dashboard.on_worker_status_summary(worker_status_summary)

    async def on_worker_update(self, worker_id: str, worker_stats: WorkerStats) -> None:
        """Update the overview and worker dashboard with the worker update."""
        if self.worker_dashboard:
            self.worker_dashboard.on_worker_stats_update(worker_id, worker_stats)

    async def action_quit(self) -> None:
        """Stop the UI and forward the signal to the main process."""
        self.exit(return_code=0)
        # Forward the signal to the main process
        os.kill(os.getpid(), signal.SIGINT)

    async def action_toggle_hide_log_viewer(self) -> None:
        """Toggle the visibility of the log viewer section."""
        try:
            if self.logs_hidden:
                self.query_one("#logs-section").remove_class("hidden")
                self.query_one("#dashboard-section").remove_class("logs-hidden")
            else:
                self.query_one("#logs-section").add_class("hidden")
                self.query_one("#dashboard-section").add_class("logs-hidden")
            self.logs_hidden = not self.logs_hidden

            _logger.debug(f"Logs {'hidden' if self.logs_hidden else 'shown'}")
        except Exception as e:
            _logger.error(f"Error toggling log viewer: {e!r}")

    async def action_restore_all_panels(self) -> None:
        """Restore all panels."""
        self.screen.minimize()
        if self.logs_hidden:
            _logger.info("Restoring log viewer")
            await self.action_toggle_hide_log_viewer()

    async def action_minimize_all_panels(self) -> None:
        """Minimize all panels."""
        self.screen.minimize()

    async def action_toggle_maximize(self, panel_id: str) -> None:
        """Toggle the maximize state of the panel with the given id."""
        panel = self.query_one(f"#{panel_id}")
        if panel and panel.is_maximized:
            self.screen.minimize()
        else:
            self.screen.maximize(panel)
