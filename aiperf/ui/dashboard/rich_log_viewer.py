# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import multiprocessing
from datetime import datetime
from typing import TYPE_CHECKING

from rich.highlighter import Highlighter, ReprHighlighter
from rich.text import Text
from textual.events import Click
from textual.widgets import RichLog

from aiperf.common.hooks import background_task
from aiperf.common.mixins import AIPerfLifecycleMixin
from aiperf.common.utils import yield_to_event_loop

if TYPE_CHECKING:
    from aiperf.ui.dashboard.aiperf_textual_app import AIPerfTextualApp


class RichLogViewer(RichLog):
    """RichLogViewer is a widget that displays log records in a rich format."""

    # NOTE: MaximizableWidget is not used here because the RichLog widget is not compatible with it.
    ALLOW_MAXIMIZE = True
    """Allow the widget to be maximized."""

    DEFAULT_CSS = """
    RichLogViewer {
        border: round $primary;
        border-title-color: $primary;
        border-title-style: bold;
        border-title-align: center;
        layout: vertical;
        scrollbar-gutter: stable;
        &:focus {
            background-tint: $primary 0%;
        }
    }
    """

    MAX_LOG_LINES = 2000
    MAX_LOG_MESSAGE_LENGTH = 500

    LOG_LEVEL_STYLES = {
        "TRACE": "dim",
        "DEBUG": "dim",
        "INFO": "cyan",
        "NOTICE": "blue",
        "WARNING": "yellow",
        "SUCCESS": "green",
        "ERROR": "red",
        "CRITICAL": "red",
    }

    def __init__(self, **kwargs) -> None:
        super().__init__(
            highlight=True,
            markup=True,
            wrap=True,
            auto_scroll=True,
            max_lines=self.MAX_LOG_LINES,
            **kwargs,
        )
        self.border_title = "Application Logs"
        self.highlighter: Highlighter = ReprHighlighter()

    def display_log_record(self, log_data: dict) -> None:
        timestamp = datetime.fromtimestamp(log_data["created"]).strftime("%H:%M:%S.%f")[:-3]  # fmt: skip
        level_style = self.LOG_LEVEL_STYLES.get(log_data["levelname"], "white")

        formatted_log = Text.assemble(
            Text.from_markup(f"[dim]{timestamp}[/dim] "),
            Text.from_markup(
                f"[bold][{level_style}]{log_data['levelname']}[/{level_style}][/bold] "
            ),
            Text.from_markup(f"[bold]{log_data['name']}[/bold] "),
            self.highlighter(
                Text.from_markup(log_data["msg"][: self.MAX_LOG_MESSAGE_LENGTH])
            ),
        )
        self.write(formatted_log)

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


class LogConsumer(AIPerfLifecycleMixin):
    """LogConsumer is a class that consumes log records from the shared log queue
    and displays them in the RichLogViewer."""

    def __init__(
        self, log_queue: multiprocessing.Queue, app: "AIPerfTextualApp", **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.log_queue = log_queue
        self.app = app

    LOG_REFRESH_INTERVAL = 0.1

    @background_task(immediate=True, interval=LOG_REFRESH_INTERVAL)
    async def _consume_logs(self) -> None:
        """Consume log records from the queue and display them.

        This is a background task that runs every LOG_REFRESH_INTERVAL seconds
        to consume log records from the queue and display them in the log viewer.
        """
        if self.app.log_viewer is None:
            return

        # Process all pending log records
        while not self.log_queue.empty():
            try:
                log_data = self.log_queue.get_nowait()
                self.app.log_viewer.display_log_record(log_data)
                await yield_to_event_loop()
            except Exception:
                # Silently ignore queue errors to avoid recursion
                break
