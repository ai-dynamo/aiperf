# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time
from collections.abc import Callable

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
)
from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Container
from textual.timer import Timer
from textual.visual import VisualType
from textual.widgets import Static

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.enums import CreditPhase
from aiperf.common.environment import Environment
from aiperf.common.mixins import CombinedPhaseStats
from aiperf.ui.dashboard.custom_widgets import MaximizableWidget
from aiperf.ui.utils import format_elapsed_time, format_eta


class ProgressDashboard(Container, MaximizableWidget):
    """Textual widget that displays Rich progress bars for profile execution."""

    DEFAULT_CSS = """
    ProgressDashboard {
        height: 1fr;
        border: round $primary;
        border-title-color: $primary;
        border-title-style: bold;
        border-title-align: center;
        padding: 0 1 0 1;
    }
    #status-display {
        height: auto;
        margin: 0 1 0 1;
    }
    #progress-display {
        height: auto;
        margin: 0 1 0 1;
    }
    #stats-display {
        height: auto;
    }
    #stats-display.no-stats {
        height: 1fr;
        content-align: center middle;
        color: $warning;
        text-style: italic;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.border_title = "Profile Progress"

        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            expand=False,
        )

        self.task_ids: dict[str, TaskID] = {}
        self.progress_widget: Static | None = None
        self.stats_widget: Static | None = None
        self.records_stats: CombinedPhaseStats | None = None
        self._phase_stats: dict[str, CombinedPhaseStats] = {}
        self.refresh_timer: Timer | None = None

    def on_mount(self) -> None:
        """Set up the refresh timer when the widget is mounted."""
        self.refresh_timer = self.set_interval(
            Environment.UI.SPINNER_REFRESH_RATE, self.refresh_timer_callback
        )

    def on_unmount(self) -> None:
        """Clean up the timer when the widget is unmounted."""
        if self.refresh_timer:
            self.refresh_timer.stop()

    def refresh_timer_callback(self) -> None:
        """Callback for the refresh timer to update the progress widget."""
        if self.progress_widget:
            self.progress_widget.update(self.progress)

    def compose(self) -> ComposeResult:
        self.progress_widget = Static(self.progress, id="progress-display")
        yield self.progress_widget

        self.stats_widget = Static(
            "Waiting for profile data...",
            id="stats-display",
            classes="no-stats",
        )
        yield self.stats_widget

    def create_or_update_progress(
        self,
        name: str,
        stats: CombinedPhaseStats,
        callback: Callable[[CombinedPhaseStats], tuple[float, float]],
    ) -> None:
        """Create or update the progress for a given task."""
        progress_percent, is_complete = callback(stats)
        task_id = self.task_ids.get(name)
        if task_id is None:
            self.task_ids[name] = self.progress.add_task(name, total=100)
        elif task_id is not None:
            self.progress.update(task_id, completed=progress_percent, total=100)
            if is_complete:
                self.progress.update(
                    task_id,
                    description=f"[green]{name}[/green]",
                )

    def _get_grace_period_progress(self, stats: CombinedPhaseStats) -> float:
        """Calculate grace period progress based on duration or request completion."""
        # If we have a finite grace period duration, use time-based progress
        if (
            stats.expected_grace_period_sec
            and stats.expected_grace_period_sec != float("inf")
            and stats.sent_end_ns
        ):
            elapsed = (time.time_ns() - stats.sent_end_ns) / NANOS_PER_SECOND
            return min((elapsed / stats.expected_grace_period_sec) * 100, 100)
        # Otherwise fall back to request-based progress
        if stats.requests_sent > 0:
            return (
                (stats.requests_completed + stats.requests_cancelled)
                / stats.requests_sent
                * 100
            )
        return 0

    def on_phase_progress(self, phase_stats: CombinedPhaseStats) -> None:
        """Callback for any phase progress updates."""
        phase = phase_stats.phase
        is_new = phase not in self._phase_stats
        self._phase_stats[phase] = phase_stats

        if is_new:
            self.query_one("#stats-display").remove_class("no-stats")

        label = phase.title()
        if phase_stats.timeout_triggered:
            self.create_or_update_progress(
                label,
                phase_stats,
                lambda stats: (100, True),
            )
            self.create_or_update_progress(
                f"{label} Grace",
                phase_stats,
                lambda stats: (
                    self._get_grace_period_progress(stats),
                    stats.is_requests_complete,
                ),
            )
        else:
            self.create_or_update_progress(
                label,
                phase_stats,
                lambda stats: (
                    stats.requests_progress_percent,
                    stats.is_requests_complete,
                ),
            )
        self.update_display(phase, phase_stats)

    def on_records_progress(self, records_stats: CombinedPhaseStats) -> None:
        """Callback for records progress updates."""
        # Only track and show records when we have actual progress (not 0 or None)
        pct = records_stats.records_progress_percent
        if pct is not None and pct > 0:
            if not self.records_stats:
                self.query_one("#stats-display").remove_class("no-stats")
            self.records_stats = records_stats
            self.create_or_update_progress(
                "Records",
                records_stats,
                lambda stats: (
                    stats.records_progress_percent,
                    stats.is_records_complete,
                ),
            )
        # Use last non-excluded phase stats for the display
        display_stats = self._get_latest_results_phase_stats()
        if display_stats:
            self.update_display(display_stats.phase, display_stats)

    def update_display(
        self, phase: CreditPhase, stats: CombinedPhaseStats | None = None
    ) -> None:
        """Update the progress display."""
        if self.progress_widget:
            self.progress_widget.update(self.progress)
        if self.stats_widget:
            self.stats_widget.update(self.create_stats_table(phase, stats))

    def _get_latest_results_phase_stats(self) -> CombinedPhaseStats | None:
        """Get the latest non-excluded phase stats."""
        for stats in reversed(list(self._phase_stats.values())):
            if not stats.exclude_from_results:
                return stats
        return None

    def _get_status(self) -> Text:
        """Get the status of the profile."""
        if self.records_stats and self.records_stats.is_records_complete:
            return Text("Complete", style="bold green")

        results_stats = self._get_latest_results_phase_stats()
        if results_stats and results_stats.is_requests_complete:
            return Text("Processing", style="bold green")
        if results_stats and results_stats.timeout_triggered:
            return Text("Grace Period", style="bold yellow")
        if results_stats:
            return Text(results_stats.phase.title(), style="bold yellow")

        # Check for any excluded phase in progress
        for stats in reversed(list(self._phase_stats.values())):
            if stats.exclude_from_results:
                if stats.timeout_triggered:
                    return Text(f"{stats.phase.title()} Grace", style="bold yellow")
                return Text(stats.phase.title(), style="bold yellow")

        if not self._phase_stats:
            return Text("Waiting for profile data...", style="dim")
        return Text("Running", style="bold yellow")

    def create_stats_table(
        self, phase: CreditPhase, stats: CombinedPhaseStats | None = None
    ) -> VisualType:
        """Create a table with the profile status and progress."""
        stats_table = Table.grid(padding=(0, 1, 0, 0))
        stats_table.add_column(style="bold cyan", justify="right")
        stats_table.add_column(style="bold white")

        if not stats:
            return stats_table

        stats_table.add_row("Status:", self._get_status())
        self._add_progress_row(stats_table, stats)

        # Show live concurrency during profiling and grace period (until requests complete)
        if not stats.is_requests_complete:
            stats_table.add_row(
                "Live Concurrency:",
                f"{stats.in_flight_requests or 0:,} requests in flight",
            )

        self._add_errors_row(stats_table)

        stats_table.add_row(
            "Request Rate:", f"{stats.requests_per_second or 0:,.1f} requests/s"
        )

        if self.records_stats:
            stats_table.add_row(
                "Processing Rate:",
                f"{self.records_stats.records_per_second or 0:,.1f} records/s",
            )

        self._add_timing_rows(stats_table, stats)
        return stats_table

    def _add_progress_row(self, stats_table: Table, stats: CombinedPhaseStats) -> None:
        """Add the Progress row based on which expected bound is active."""
        # During grace period, show progress as completed+cancelled out of sent
        if stats.timeout_triggered:
            completed = stats.requests_completed + stats.requests_cancelled
            sent = stats.requests_sent
            pct = (completed / sent * 100) if sent > 0 else 0
            stats_table.add_row(
                "Progress:",
                f"{completed:,} / {sent:,} requests returned ({pct:.1f}%)",
            )
        elif stats.total_expected_requests:
            stats_table.add_row(
                "Progress:",
                f"{stats.requests_completed or 0:,} / {stats.total_expected_requests:,} requests "
                f"({stats.requests_progress_percent:.1f}%)",
            )
        elif stats.expected_num_sessions:
            stats_table.add_row(
                "Progress:",
                f"{stats.completed_sessions or 0:,} / {stats.expected_num_sessions:,} user sessions "
                f"({stats.requests_progress_percent:.1f}%)",
            )
        elif stats.expected_duration_sec:
            stats_table.add_row(
                "Progress:",
                f"{stats.requests_elapsed_time or 0:.1f} / {stats.expected_duration_sec:.1f} seconds "
                f"({stats.requests_progress_percent:.1f}%)",
            )

    def _add_errors_row(self, stats_table: Table) -> None:
        """Add the Errors row with color-coded percent when records stats exist."""
        if not self.records_stats:
            return
        error_percent = self.records_stats.records_error_percent
        error_color = (
            "green" if error_percent == 0 else "red" if error_percent > 10 else "yellow"
        )
        stats_table.add_row(
            "Errors:",
            f"[{error_color}]{self.records_stats.error_records or 0:,} / {self.records_stats.total_records or 0:,} "
            f"({error_percent:.1f}%)[/{error_color}]",
        )

    def _add_timing_rows(self, stats_table: Table, stats: CombinedPhaseStats) -> None:
        """Add Elapsed / Grace Period / ETA rows depending on request completion."""
        if not stats.is_requests_complete:
            # Display request stats while profiling
            if stats.start_ns:
                stats_table.add_row(
                    "Elapsed:", format_elapsed_time(stats.requests_elapsed_time)
                )
            # Show grace period elapsed time when in grace period
            if stats.timeout_triggered and stats.sent_end_ns:
                grace_elapsed_sec = (
                    time.time_ns() - stats.sent_end_ns
                ) / NANOS_PER_SECOND
                stats_table.add_row(
                    "Grace Period:", format_elapsed_time(grace_elapsed_sec)
                )
            elif stats.requests_eta_sec:
                stats_table.add_row("ETA:", format_eta(stats.requests_eta_sec))
        elif self.records_stats:
            # Display record processing stats after profiling
            if self.records_stats.start_ns:
                stats_table.add_row(
                    "Elapsed:",
                    format_elapsed_time(self.records_stats.records_elapsed_time),
                )
            if self.records_stats.records_eta_sec:
                stats_table.add_row(
                    "Records ETA:", format_eta(self.records_stats.records_eta_sec)
                )
