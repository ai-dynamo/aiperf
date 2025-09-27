# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
from datetime import datetime

from rich.highlighter import Highlighter, ReprHighlighter
from rich.text import Text
from textual.await_remove import AwaitRemove
from textual.events import Click
from textual.widgets import RichLog


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

        # Create and register the dedicated log handler
        self._log_handler: UILogHandler | None = None
        self._setup_log_handler()

    def _setup_log_handler(self) -> None:
        """Set up the dedicated log handler for this viewer."""
        self._log_handler = UILogHandler(self)
        root_logger = logging.getLogger()
        root_logger.addHandler(self._log_handler)

    def _cleanup_log_handler(self) -> None:
        """Clean up the log handler."""
        if self._log_handler is not None:
            root_logger = logging.getLogger()
            if self._log_handler in root_logger.handlers:
                root_logger.removeHandler(self._log_handler)
            self._log_handler = None

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

    def remove(self) -> AwaitRemove:
        """Clean up when the widget is removed."""
        self._cleanup_log_handler()
        return super().remove()


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
        try:
            # Direct formatting without intermediate dictionary
            timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S.%f")[
                :-3
            ]
            level_style = self.log_viewer.LOG_LEVEL_STYLES.get(
                record.levelname, "white"
            )
            message = record.getMessage()[: self.log_viewer.MAX_LOG_MESSAGE_LENGTH]

            # Direct assembly and write - no intermediate steps
            formatted_log = Text.assemble(
                Text.from_markup(f"[dim]{timestamp}[/dim] "),
                Text.from_markup(
                    f"[bold][{level_style}]{record.levelname}[/{level_style}][/bold] "
                ),
                Text.from_markup(f"[bold]{record.name}[/bold] "),
                self.log_viewer.highlighter(Text.from_markup(message)),
            )
            self.write(formatted_log)
        except Exception:
            # Do not log to prevent recursion
            pass
