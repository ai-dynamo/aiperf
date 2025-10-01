# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import logging
from contextlib import suppress
from datetime import datetime
from typing import TYPE_CHECKING

from rich.highlighter import Highlighter, ReprHighlighter
from rich.text import Text
from textual.events import Click
from textual.widgets import RichLog

from aiperf.common.hooks import background_task
from aiperf.common.mixins.aiperf_lifecycle_mixin import AIPerfLifecycleMixin
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
        self.log_queue = asyncio.Queue(maxsize=1000)
        self._removed_log_handlers = []

        # Create and register the dedicated log handler
        self._log_handler: UILogHandler | None = None

        self._setup_log_handler()

    def _setup_log_handler(self) -> None:
        """Set up the dedicated log handler for this viewer."""
        self._log_handler = UILogHandler(self)
        root_logger = logging.getLogger()
        # for handler in root_logger.handlers[:]:
        #     if not isinstance(handler, logging.FileHandler):
        #         self._removed_log_handlers.append(handler)
        #         root_logger.removeHandler(handler)
        root_logger.addHandler(self._log_handler)

    def _cleanup_log_handler(self) -> None:
        """Clean up the log handler."""
        root_logger = logging.getLogger()
        if self._log_handler is not None:
            if self._log_handler in root_logger.handlers:
                root_logger.removeHandler(self._log_handler)
            self._log_handler = None
        for handler in self._removed_log_handlers:
            root_logger.addHandler(handler.copy())

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

    def unmount(self):
        """Clean up when the widget is unmounted."""
        self._cleanup_log_handler()
        super().unmount()

    def _write_log_record(self, record: logging.LogRecord) -> None:
        """Write a log record to the widget."""
        with suppress(Exception):
            timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S.%f")[:-3]  # fmt: skip
            level_style = self.LOG_LEVEL_STYLES.get(record.levelname, "white")
            message = record.getMessage()[: self.MAX_LOG_MESSAGE_LENGTH]

            formatted_log = Text.assemble(
                Text.from_markup(f"[dim]{timestamp}[/dim] "),
                Text.from_markup(
                    f"[bold][{level_style}]{record.levelname}[/{level_style}][/bold] "
                ),
                Text.from_markup(f"[bold]{record.name}[/bold] "),
                self.highlighter(Text.from_markup(message)),
            )
            self.write(formatted_log, scroll_end=self.auto_scroll)


class UILogHandler(logging.Handler):
    """Lightweight logging handler that directly interfaces with RichLogViewer.

    This handler is specifically designed for the RichLogViewer and provides
    direct, high-performance log forwarding without callbacks or queues.
    """

    def __init__(self, log_viewer: RichLogViewer) -> None:
        super().__init__()
        self.log_viewer = log_viewer
        self.write = log_viewer.write

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record directly to the RichLogViewer with maximum performance."""
        with suppress(asyncio.QueueFull):
            self.log_viewer.log_queue.put_nowait(record)


class LogConsumer(AIPerfLifecycleMixin):
    """LogConsumer is a class that consumes log records from the shared log queue
    and displays them in the RichLogViewer."""

    def __init__(self, app: "AIPerfTextualApp", **kwargs) -> None:
        super().__init__(**kwargs)
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
        while not self.app.log_viewer.log_queue.empty():
            try:
                log_data = self.app.log_viewer.log_queue.get_nowait()
                self.app.log_viewer._write_log_record(log_data)
                await yield_to_event_loop()
            except Exception:
                # Silently ignore queue errors to avoid recursion
                break
