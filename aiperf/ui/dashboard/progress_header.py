# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib

from textual.containers import Horizontal
from textual.widget import Widget
from textual.widgets import ProgressBar, Static


class ProgressHeader(Widget):
    """Custom header for the progress dashboard."""

    DEFAULT_CSS = """
    ProgressHeader {
        dock: top;
        width: 100%;
        background: $footer-background;
        color: $warning;
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
        width: 1fr;
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
        width: 1fr;
        content-align: left middle;
        color: $primary;
        padding-left: 1;
    }
    #progress-name.warmup, #progress-bar.warmup, PercentageStatus.warmup {
        color: $warning;
    }
    #progress-name.profiling, #progress-bar.profiling, PercentageStatus.profiling {
        color: $primary;
    }
    #progress-name.records, #progress-bar.records, PercentageStatus.records{
        color: $success;
    }
    """

    def __init__(self, title: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title = title
        self.progress_name = ""

    def compose(self):
        with Horizontal():
            yield Static(id="progress-name")
            yield Static(self.title, id="header-title")
            yield ProgressBar(
                id="progress-bar",
                total=100,
                show_eta=False,
                show_percentage=True,
                classes="hidden",
            )
            yield Static(id="padding")

    def update_progress(
        self, header: str, progress: float, total: float | None = None
    ) -> None:
        """Update the progress of the progress bar."""
        with contextlib.suppress(Exception):
            bar = self.query_one(ProgressBar)
            if self.progress_name != header:
                bar.remove_class("hidden")
                self.query_one("#padding").add_class("hidden")
                self.query_one("#progress-name", Static).remove_class(
                    "warmup", "profiling", "records"
                ).add_class(header.lower()).update(header)
                self.query_one("PercentageStatus").remove_class(
                    "warmup", "profiling", "records"
                ).add_class(header.lower())
                self.progress_name = header
            bar.update(progress=progress, total=total)
            self.refresh()
