# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib

from textual.containers import Horizontal
from textual.events import Click
from textual.widget import Widget
from textual.widgets import DataTable, ProgressBar, Static


class NonFocusableDataTable(DataTable, can_focus=False):
    """DataTable that cannot receive focus.
    This is done to prevent the table from focusing when the user clicks on it, which would cause the table to darken its background."""


class MaximizableWidget(Widget):
    """Mixin that allows a widget to be maximized by double-clicking on it."""

    ALLOW_MAXIMIZE = True
    """Allow the widget to be maximized."""

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


class ProgressHeader(Widget):
    """Custom header for the progress dashboard."""

    DEFAULT_CSS = """
    ProgressHeader {
        dock: top;
        width: 100%;
        background: $footer-background;
        color: $primary;
        text-style: bold;
        height: 1;
    }
    .bar--indeterminate {
        color: $primary;
        background: $secondary;
    }

    .bar--complete {
        color: $error;
    }
    PercentageStatus {
        color: $primary;
    }
    ETAStatus {
        color: $primary;
    }
    #padding {
        width: 1fr;
    }
    #progress-bar {
        width: auto;
        background: $footer-background;
        align: right middle;
        padding-right: 1;
    }
    #header-title {
        width: 1fr;
        content-align: center middle;
        color: $primary;
    }
    #progress-name {
        width: auto;
        content-align: right middle;
        color: $primary;
        padding-right: 1;
    }
    #progress-name.warmup, #progress-bar.warmup {
        color: $warning;
    }
    #progress-name.profiling, #progress-bar.profiling {
        color: $primary;
    }
    #progress-name.records, #progress-bar.records {
        color: $success;
    }
    """

    def __init__(self, title: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title = title
        self.progress_name = ""

    def compose(self):
        with Horizontal():
            yield Static(id="padding")
            yield Static(self.title, id="header-title")
            yield Static(id="progress-name")
            yield ProgressBar(
                id="progress-bar",
                total=100,
                show_eta=False,
                show_percentage=True,
            )

    def update_progress(
        self, header: str, progress: float, total: float | None = None
    ) -> None:
        """Update the progress of the progress bar."""
        with contextlib.suppress(Exception):
            bar = self.query_one(ProgressBar)
            if self.progress_name != header:
                self.query_one("#progress-name", Static).remove_class(
                    "warmup", "profiling", "records"
                ).add_class(header.lower()).update(header)
                bar.remove_class("warmup", "profiling", "records")
                bar.add_class(header.lower())
                self.progress_name = header
            bar.update(progress=progress, total=total)
            self.refresh()
