# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import signal
from contextlib import suppress
from typing import TYPE_CHECKING

from rich.console import RenderableType
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Footer

from aiperf.common.config.service_config import ServiceConfig
from aiperf.common.enums import GPUTelemetryMode, WorkerStatus
from aiperf.common.environment import Environment
from aiperf.common.messages import StartRealtimeTelemetryCommand
from aiperf.common.mixins import CombinedPhaseStats
from aiperf.common.models import MetricResult, WorkerStats
from aiperf.ui.dashboard.aiperf_theme import AIPERF_THEME
from aiperf.ui.dashboard.progress_dashboard import ProgressDashboard
from aiperf.ui.dashboard.progress_header import ProgressHeader
from aiperf.ui.dashboard.realtime_metrics_dashboard import RealtimeMetricsDashboard
from aiperf.ui.dashboard.realtime_telemetry_dashboard import RealtimeTelemetryDashboard
from aiperf.ui.dashboard.rich_log_viewer import RichLogViewer
from aiperf.ui.dashboard.worker_dashboard import WorkerDashboard

if TYPE_CHECKING:
    from aiperf.controller.system_controller import SystemController


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
    #telemetry-section {
        height: 3fr;
        min-height: 14;
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
        ("5", "toggle_maximize_telemetry", "GPU Telemetry"),
        ("6", "toggle_maximize('logs')", "Logs"),
        ("escape", "restore_all_panels", "Restore View"),
        Binding("ctrl+s", "screenshot", "Save Screenshot", show=False),
        Binding("l", "toggle_hide_log_viewer", "Toggle Logs", show=False),
        Binding("c", "copy_logs", "Copy Logs", show=False),
    ]

    def __init__(
        self, service_config: ServiceConfig, controller: SystemController
    ) -> None:
        super().__init__()

        self.title = "NVIDIA AIPerf"
        if Environment.DEV.MODE:
            self.title = "NVIDIA AIPerf (Developer Mode)"

        self.log_viewer: RichLogViewer | None = None
        self.progress_dashboard: ProgressDashboard | None = None
        self.progress_header: ProgressHeader | None = None
        self.worker_dashboard: WorkerDashboard | None = None
        self.realtime_metrics_dashboard: RealtimeMetricsDashboard | None = None
        self.realtime_telemetry_dashboard: RealtimeTelemetryDashboard | None = None
        self.profile_results: list[RenderableType] = []
        self.service_config = service_config
        self.controller: SystemController = controller
        self._warmup_stats: CombinedPhaseStats | None = None
        self._profiling_stats: CombinedPhaseStats | None = None
        self._records_stats: CombinedPhaseStats | None = None
        self._has_result_data = False

    def on_mount(self) -> None:
        self.register_theme(AIPERF_THEME)
        self.theme = AIPERF_THEME.name
        # Maximize log viewer initially until result data arrives
        if not self._has_result_data and self.log_viewer:
            self.screen.maximize(self.log_viewer)

    def _on_first_result_data(self) -> None:
        """Called when the first result data arrives - minimizes the log viewer."""
        self._has_result_data = True
        # Restore to normal view when data starts coming in
        self.screen.minimize()
        if self.log_viewer:
            # Scroll down as things may have shifted when the log viewer is un-maximized
            self.log_viewer.scroll_end(duration=0.2)

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

                    with Container(id="metrics-section"):
                        self.realtime_metrics_dashboard = RealtimeMetricsDashboard(
                            service_config=self.service_config, id="metrics"
                        )
                        yield self.realtime_metrics_dashboard

            with Container(id="workers-section", classes="hidden"):
                self.worker_dashboard = WorkerDashboard(id="workers")
                yield self.worker_dashboard

            with Container(id="telemetry-section", classes="hidden"):
                self.realtime_telemetry_dashboard = RealtimeTelemetryDashboard(
                    service_config=self.service_config, id="telemetry"
                )
                yield self.realtime_telemetry_dashboard

            with Container(id="logs-section"):
                self.log_viewer = RichLogViewer(id="logs")
                yield self.log_viewer

        yield Footer()

    async def action_quit(self) -> None:
        """Stop the UI and forward the signal to the main process."""
        self.exit(return_code=0)
        # Clear the references to the widgets to ensure they do not get updated after the app is stopped
        self.worker_dashboard = None
        self.progress_dashboard = None
        self.progress_header = None
        self.realtime_metrics_dashboard = None
        self.log_viewer = None

        # Forward the signal to the main process
        # IMPORTANT: This is necessary, otherwise the process will hang
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

    async def action_toggle_maximize_telemetry(self) -> None:
        """Toggle the maximize state of the telemetry panel and enable realtime GPU telemetry if needed."""
        if (
            self.controller.user_config.gpu_telemetry_mode
            != GPUTelemetryMode.REALTIME_DASHBOARD
        ):
            self.controller.user_config.gpu_telemetry_mode = (
                GPUTelemetryMode.REALTIME_DASHBOARD
            )
            if self.realtime_telemetry_dashboard:
                self.realtime_telemetry_dashboard.set_status_message(
                    "Enabling live GPU telemetry..."
                )

            await self.controller.publish(
                StartRealtimeTelemetryCommand(
                    service_id=self.controller.service_id,
                )
            )

        await self.action_toggle_maximize("telemetry")

    async def action_copy_logs(self) -> None:
        """Copy all log content to clipboard."""
        if self.log_viewer:
            log_text = self.log_viewer.get_log_text()
            if log_text:
                self.copy_to_clipboard(log_text)
                self.notify(
                    f"Copied {len(log_text):,} characters to clipboard",
                    title="Logs Copied",
                )
            else:
                self.notify("No logs to copy", severity="warning")

    async def on_warmup_progress(self, warmup_stats: CombinedPhaseStats) -> None:
        """Forward warmup progress updates to the Textual App."""
        if not self._has_result_data:
            self._on_first_result_data()
        self._warmup_stats = warmup_stats

        if self.progress_dashboard:
            async with self.progress_dashboard.batch():
                self.progress_dashboard.on_warmup_progress(warmup_stats)

        if self.progress_header:
            # During grace period, show progress as completed+cancelled out of sent
            if warmup_stats.timeout_triggered:
                total = warmup_stats.requests_sent
                completed = (
                    warmup_stats.requests_completed + warmup_stats.requests_cancelled
                )
                progress = (completed / total * 100) if total > 0 else 0
                self.progress_header.update_progress(
                    header="Warmup Grace",
                    progress=progress,
                    total=100,
                )
            else:
                progress = warmup_stats.requests_progress_percent
                if progress is not None:
                    self.progress_header.update_progress(
                        header="Warmup",
                        progress=progress,
                        total=100,
                    )

    async def on_profiling_progress(self, profiling_stats: CombinedPhaseStats) -> None:
        """Forward requests phase progress updates to the Textual App."""
        if not self._has_result_data:
            self._on_first_result_data()
        self._profiling_stats = profiling_stats
        if self.progress_dashboard:
            async with self.progress_dashboard.batch():
                self.progress_dashboard.on_profiling_progress(profiling_stats)
        if self.progress_header:
            # During grace period, show progress as completed+cancelled out of sent
            if profiling_stats.timeout_triggered:
                total = profiling_stats.requests_sent
                completed = (
                    profiling_stats.requests_completed
                    + profiling_stats.requests_cancelled
                )
                progress = (completed / total * 100) if total > 0 else 0
                self.progress_header.update_progress(
                    header="Grace Period",
                    progress=progress,
                    total=100,
                )
            else:
                progress = profiling_stats.requests_progress_percent
                if progress is not None:
                    self.progress_header.update_progress(
                        header="Profiling",
                        progress=progress,
                        total=100,
                    )

    async def on_records_progress(self, records_stats: CombinedPhaseStats) -> None:
        """Forward records progress updates to the Textual App."""
        self._records_stats = records_stats
        if self.progress_dashboard:
            async with self.progress_dashboard.batch():
                self.progress_dashboard.on_records_progress(records_stats)

        pct = records_stats.records_progress_percent
        if (
            self._profiling_stats
            and self._profiling_stats.is_requests_complete
            and self.progress_header
            and pct is not None
            and pct > 0
        ):
            self.progress_header.update_progress(
                header="Records",
                progress=records_stats.records_progress_percent,
                total=100,
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
        if self.realtime_metrics_dashboard:
            async with self.realtime_metrics_dashboard.batch():
                self.realtime_metrics_dashboard.on_realtime_metrics(metrics)

    async def on_realtime_telemetry_metrics(self, metrics: list[MetricResult]) -> None:
        """Forward real-time GPU telemetry metrics updates to the Textual App."""
        if self.realtime_telemetry_dashboard:
            async with self.realtime_telemetry_dashboard.batch():
                self.realtime_telemetry_dashboard.on_realtime_telemetry_metrics(metrics)
