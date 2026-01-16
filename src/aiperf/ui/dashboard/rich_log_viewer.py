# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import multiprocessing
from collections import deque
from datetime import datetime
from typing import TYPE_CHECKING

from rich.highlighter import ReprHighlighter
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import ScrollableContainer
from textual.events import Click, MouseScrollDown, MouseScrollUp
from textual.widgets import Static

from aiperf.common.environment import Environment
from aiperf.common.hooks import background_task
from aiperf.common.mixins import AIPerfLifecycleMixin
from aiperf.common.utils import yield_to_event_loop

if TYPE_CHECKING:
    from aiperf.ui.dashboard.aiperf_textual_app import AIPerfTextualApp


class SelectableRichLog(ScrollableContainer):
    """A RichLog-like widget that supports text selection with character-level wrapping."""

    ALLOW_MAXIMIZE = True
    """Allow the widget to be maximized."""

    DEFAULT_CSS = """
    SelectableRichLog {
        background: $surface;
        color: $text;
        border: round $primary;
        border-title-color: $primary;
        border-title-style: bold;
        border-title-align: center;
        padding: 0 1;
        scrollbar-gutter: stable;
        scrollbar-background: $panel;
        scrollbar-color: $primary;
        scrollbar-color-hover: $primary-lighten-1;
        scrollbar-color-active: $primary-lighten-2;
    }

    SelectableRichLog > .log-content {
        width: 100%;
        height: auto;
        background: $surface;
        padding: 0;
    }
    """

    MAX_LOG_LINES = 2000
    MAX_LOG_MESSAGE_LENGTH = 500
    DEFAULT_WIDTH = 120
    INDENT_WIDTH_THRESHOLD = 90

    LOG_LEVEL_STYLES = {
        "TRACE": "dim",
        "DEBUG": "dim",
        "INFO": "cyan",
        "NOTICE": "blue",
        "WARNING": "yellow",
        "SUCCESS": "green",
        "ERROR": "red",
        "CRITICAL": "bold red",
    }

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.border_title = "Application Logs"
        self._log_lines: deque[Text] = deque(maxlen=self.MAX_LOG_LINES)
        self._content_widget: Static | None = None
        self.highlighter = ReprHighlighter()
        self._auto_scroll = True

    def compose(self) -> ComposeResult:
        self._content_widget = Static(classes="log-content")
        yield self._content_widget

    def display_log_record(self, log_data: dict) -> None:
        """Display a log record with character-level wrapping."""
        try:
            timestamp = datetime.fromtimestamp(log_data["created"]).strftime("%H:%M:%S.%f")[:-3]  # fmt: skip
            levelname_raw = log_data["levelname"]
            filename = log_data.get("filename", log_data["name"])
            lineno = log_data.get("lineno", "")
            message = log_data["msg"][: self.MAX_LOG_MESSAGE_LENGTH]

            level_style = self.LOG_LEVEL_STYLES.get(levelname_raw, "white")
            log_suffix = f"({filename}:{lineno})" if lineno else f"({filename})"

            # Calculate widths
            console_width = (
                max(self.size.width - 4, 40)
                if self.size.width > 0
                else self.DEFAULT_WIDTH
            )
            target_width = console_width - 2

            prefix = f"[{timestamp}] {levelname_raw:<8} "
            prefix_len = len(prefix)
            content_width = target_width - prefix_len

            # Combine message and suffix for character-level wrapping
            full_content = f"{message} {log_suffix}"
            suffix_start_pos = len(message) + 1

            # Only indent continuation lines on wide displays (90+)
            indent_continuations = console_width >= self.INDENT_WIDTH_THRESHOLD
            continuation_width = content_width if indent_continuations else target_width

            # Manual character-level wrapping
            lines = []
            remaining = full_content
            is_first_line = True
            while remaining:
                line_width = content_width if is_first_line else continuation_width
                if len(remaining) <= line_width:
                    lines.append(remaining)
                    break
                lines.append(remaining[:line_width])
                remaining = remaining[line_width:]
                is_first_line = False

            # Build output with proper styling
            parts = []
            char_pos = 0
            for i, line in enumerate(lines):
                if i > 0:
                    parts.append(Text("\n"))
                    if indent_continuations:
                        parts.append(Text(" " * prefix_len))
                else:
                    parts.append(Text(f"[{timestamp}] ", style="log.time"))
                    parts.append(Text(f"{levelname_raw:<8} ", style=level_style))

                line_end_pos = char_pos + len(line)

                # Apply styling: message gets highlighting, suffix gets dim italic
                if char_pos >= suffix_start_pos:
                    # Entire line is suffix
                    parts.append(Text(line, style="dim italic"))
                elif line_end_pos <= suffix_start_pos:
                    # Entire line is message
                    parts.append(self.highlighter(Text(line)))
                else:
                    # Line contains both message and suffix
                    msg_chars = suffix_start_pos - char_pos
                    parts.append(self.highlighter(Text(line[:msg_chars])))
                    parts.append(Text(line[msg_chars:], style="dim italic"))

                char_pos = line_end_pos

            formatted_log = Text.assemble(*parts)
            self._log_lines.append(formatted_log)
            self._update_display()

        except Exception as e:
            self._log_lines.append(Text(f"ERROR formatting log: {e}"))
            self._update_display()

    def _update_display(self) -> None:
        """Update the display with all log lines."""
        if self._content_widget is None:
            return

        combined = Text()
        for i, line in enumerate(self._log_lines):
            if i > 0:
                combined.append("\n")
            combined.append(line)

        self._content_widget.update(combined)
        if self._auto_scroll:
            self.scroll_end(animate=False)

    def _is_at_bottom(self) -> bool:
        """Check if scrolled to the bottom (within a small threshold)."""
        return self.scroll_y >= self.max_scroll_y - 2

    def on_mouse_scroll_up(self, event: MouseScrollUp) -> None:
        """User scrolled up - disable auto-scroll."""
        self._auto_scroll = False

    def on_mouse_scroll_down(self, event: MouseScrollDown) -> None:
        """User scrolled down - re-enable auto-scroll if at bottom."""
        # Use call_after_refresh to check position after scroll completes
        self.call_after_refresh(self._check_auto_scroll)

    def _check_auto_scroll(self) -> None:
        """Check if we should re-enable auto-scroll."""
        if self._is_at_bottom():
            self._auto_scroll = True

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

    def get_log_text(self) -> str:
        """Get all log content as plain text for clipboard copy."""
        return "\n".join(log_line.plain for log_line in self._log_lines)


class RichLogViewer(SelectableRichLog):
    """RichLogViewer with colorized output and text selection support via clipboard copy."""

    pass


class LogConsumer(AIPerfLifecycleMixin):
    """LogConsumer is a class that consumes log records from the shared log queue
    and displays them in the RichLogViewer."""

    def __init__(
        self, log_queue: multiprocessing.Queue, app: "AIPerfTextualApp", **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.log_queue = log_queue
        self.app = app

    @background_task(immediate=True, interval=Environment.UI.LOG_REFRESH_INTERVAL)
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
