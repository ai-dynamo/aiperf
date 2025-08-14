# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import signal
from contextlib import suppress

from rich.console import RenderableType
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Footer

from aiperf.common.config.service_config import ServiceConfig
from aiperf.common.constants import AIPERF_DEV_MODE
from aiperf.common.enums import WorkerStatus
from aiperf.common.models import MetricResult, RecordsStats, RequestsStats, WorkerStats
from aiperf.controller.system_controller import SystemController
from aiperf.ui.dashboard.aiperf_theme import AIPERF_THEME
from aiperf.ui.dashboard.custom_widgets import ProgressHeader
from aiperf.ui.dashboard.progress_dashboard import ProgressDashboard
from aiperf.ui.dashboard.realtime_metrics_dashboard import RealtimeMetricsDashboard
from aiperf.ui.dashboard.rich_log_viewer import RichLogViewer
from aiperf.ui.dashboard.worker_dashboard import WorkerDashboard


class AIPerfTextualApp(App):
    """
    AIPerf Textual App.

    This is the main application class for the Textual UI. It is responsible for
    composing the application layout and handling the application commands.
    """

    ENABLE_COMMAND_PALETTE = False
    """Disable the command palette that is enabled by default in Textual."""

    ALLOW_IN_MAXIMIZED_VIEW = "ProgressHeader, Footer"
    """Allow the custom header and footer to be displayed when a panel is maximized."""

    NOTIFICATION_TIMEOUT = 3
    """The timeout for notifications in seconds."""

    CSS = """
    #main-container {
        height: 100%;
    }
    #dashboard-section {
        height: 3fr;
        min-height: 14;
    }
    #logs-section {
        height: 2fr;
        max-height: 16;
    }
    #workers-section {
        height: 3;
    }
    #progress-section {
        width: 1fr;
    }
    #metrics-section {
        width: 2fr;
    }
    .hidden {
        display: none;
    }
    """

    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("1", "minimize_all_panels", "Overview"),
        ("2", "toggle_maximize('progress')", "Progress"),
        ("3", "toggle_maximize('metrics')", "Metrics"),
        ("4", "toggle_maximize('workers')", "Workers"),
        ("5", "toggle_maximize('logs')", "Logs"),
        ("escape", "restore_all_panels", "Restore View"),
        Binding("ctrl+s", "screenshot", "Save Screenshot", show=False),
        Binding("l", "toggle_hide_log_viewer", "Toggle Logs", show=False),
    ]

    def __init__(
        self, service_config: ServiceConfig, controller: SystemController
    ) -> None:
        super().__init__()

        self.title = "NVIDIA AIPerf"
        if AIPERF_DEV_MODE:
            self.title = "NVIDIA AIPerf (Developer Mode)"

        self.log_viewer: RichLogViewer | None = None
        self.progress_dashboard: ProgressDashboard | None = None
        self.progress_header: ProgressHeader | None = None
        self.worker_dashboard: WorkerDashboard | None = None
        self.realtime_metrics_dashboard: RealtimeMetricsDashboard | None = None
        self.profile_results: list[RenderableType] = []
        self.service_config = service_config
        self.controller: SystemController = controller
        self._warmup_stats: RequestsStats | None = None
        self._profiling_stats: RequestsStats | None = None
        self._records_stats: RecordsStats | None = None
        self._metrics: list[MetricResult] | None = None

    def on_mount(self) -> None:
        self.register_theme(AIPERF_THEME)
        self.theme = AIPERF_THEME.name

    def compose(self) -> ComposeResult:
        """Compose the full application layout."""
        self.progress_header = ProgressHeader(title=self.title, id="progress-header")
        yield self.progress_header

        # NOTE: SIM117 is disabled because nested with statements are recommended for textual ui layouts
        with Vertical(id="main-container"):
            with Container(id="dashboard-section"):  # noqa: SIM117
                with Horizontal(id="overview-section"):
                    with Container(id="progress-section"):
                        self.progress_dashboard = ProgressDashboard(id="progress")
                        yield self.progress_dashboard

                    with Container(id="metrics-section", classes="hidden"):
                        self.realtime_metrics_dashboard = RealtimeMetricsDashboard(
                            service_config=self.service_config, id="metrics"
                        )
                        yield self.realtime_metrics_dashboard

            with Container(id="workers-section", classes="hidden"):
                self.worker_dashboard = WorkerDashboard(id="workers")
                yield self.worker_dashboard

            with Container(id="logs-section"):
                self.log_viewer = RichLogViewer(id="logs")
                yield self.log_viewer

        yield Footer()

    async def action_quit(self) -> None:
        """Stop the UI and forward the signal to the main process."""
        self.exit(return_code=0)
        # Forward the signal to the main process
        os.kill(os.getpid(), signal.SIGINT)

    async def action_toggle_hide_log_viewer(self) -> None:
        """Toggle the visibility of the log viewer section."""
        with suppress(Exception):
            self.query_one("#logs-section").toggle_class("hidden")

    async def action_restore_all_panels(self) -> None:
        """Restore all panels."""
        self.screen.minimize()
        with suppress(Exception):
            self.query_one("#logs-section").remove_class("hidden")

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

    async def on_warmup_progress(self, warmup_stats: RequestsStats) -> None:
        """Forward warmup progress updates to the Textual App."""
        if not self._warmup_stats:
            self.query_one("#progress-section").remove_class("hidden")
        self._warmup_stats = warmup_stats
        if self.progress_dashboard:
            async with self.progress_dashboard.batch():
                self.progress_dashboard.on_warmup_progress(warmup_stats)
        if self.progress_header:
            self.progress_header.update_progress(
                header="Warmup",
                progress=warmup_stats.finished,
                total=warmup_stats.total_expected_requests,
            )

    async def on_profiling_progress(self, profiling_stats: RequestsStats) -> None:
        """Forward requests phase progress updates to the Textual App."""
        if not self._profiling_stats:
            self.query_one("#progress-section").remove_class("hidden")
        self._profiling_stats = profiling_stats
        if self.progress_dashboard:
            async with self.progress_dashboard.batch():
                self.progress_dashboard.on_profiling_progress(profiling_stats)
        if self.progress_header:
            self.progress_header.update_progress(
                header="Profiling",
                progress=profiling_stats.finished,
                total=profiling_stats.total_expected_requests,
            )

    async def on_records_progress(self, records_stats: RecordsStats) -> None:
        """Forward records progress updates to the Textual App."""
        self._records_stats = records_stats
        if self.progress_dashboard:
            async with self.progress_dashboard.batch():
                self.progress_dashboard.on_records_progress(records_stats)

        if (
            self._profiling_stats
            and self._profiling_stats.is_complete
            and self.progress_header
        ):
            self.progress_header.update_progress(
                header="Records",
                progress=self._profiling_stats.finished,
                total=self._profiling_stats.total_expected_requests,
            )

    async def on_worker_update(self, worker_id: str, worker_stats: WorkerStats):
        """Forward worker updates to the Textual App."""
        if self.worker_dashboard:
            async with self.worker_dashboard.batch():
                self.worker_dashboard.on_worker_update(worker_id, worker_stats)

    async def on_worker_status_summary(self, worker_status_summary: dict[str, WorkerStatus]) -> None:  # fmt: skip
        """Forward worker status summary updates to the Textual App."""
        if self.worker_dashboard:
            async with self.worker_dashboard.batch():
                self.worker_dashboard.on_worker_status_summary(worker_status_summary)

    async def on_realtime_metrics(self, metrics: list[MetricResult]) -> None:
        """Forward real-time metrics updates to the Textual App."""
        if not self._metrics:
            self.query_one("#metrics-section").remove_class("hidden")
        self._metrics = metrics
        if self.realtime_metrics_dashboard:
            async with self.realtime_metrics_dashboard.batch():
                self.realtime_metrics_dashboard.on_realtime_metrics(metrics)
