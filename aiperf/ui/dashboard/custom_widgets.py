# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from textual.events import Click
from textual.widget import Widget
from textual.widgets import DataTable


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
