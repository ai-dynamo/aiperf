# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib

from textual.dom import NoScreen
from textual.events import Mount
from textual.widget import Widget
from textual.widgets._header import HeaderTitle


class CustomHeader(Widget):
    """Custom header alternative to the default Textual header. It removes
    some of the default abilities that we don't want.
    """

    DEFAULT_CSS = """
    CustomHeader {
        dock: top;
        width: 100%;
        background: $primary;
        color: $text;
        text-style: bold;
        height: 1;
    }
    """

    DEFAULT_CLASSES = ""

    def __init__(
        self,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ):
        super().__init__(name=name, id=id, classes=classes)

    def compose(self):
        yield HeaderTitle()

    @property
    def screen_title(self) -> str:
        screen_title = self.screen.title
        title = screen_title if screen_title is not None else self.app.title
        return title

    @property
    def screen_sub_title(self) -> str:
        screen_sub_title = self.screen.sub_title
        sub_title = (
            screen_sub_title if screen_sub_title is not None else self.app.sub_title
        )
        return sub_title

    def _on_mount(self, event: Mount) -> None:
        async def set_title() -> None:
            with contextlib.suppress(NoScreen):
                self.query_one(HeaderTitle).text = self.screen_title

        async def set_sub_title() -> None:
            with contextlib.suppress(NoScreen):
                self.query_one(HeaderTitle).sub_text = self.screen_sub_title

        self.watch(self.app, "title", set_title)
        self.watch(self.app, "sub_title", set_sub_title)
        self.watch(self.screen, "title", set_title)
        self.watch(self.screen, "sub_title", set_sub_title)
