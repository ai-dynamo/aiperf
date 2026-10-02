# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic Rich console sizing for tests that assert on rendered output.

`Console.size` short-circuits to the explicit dimensions only when *both* width
and height were supplied. Width alone falls through to size detection, whose
first branch hard-codes 80x25 for a dumb terminal — so under `TERM=dumb` (CI
logs, non-interactive agent shells) `Console(width=120)` silently renders at 80,
as does `console.width = 200` on a shared console and a `COLUMNS` override,
which that branch returns before ever reading. The same tests render at the
requested width under `TERM=xterm`, so assertions on table text pass on a
developer machine and fail in CI, or the reverse, with Rich wrapping or
ellipsizing cells at a width nobody chose.

Pin both dimensions through these helpers instead of sizing consoles inline.
"""

from collections.abc import Iterator
from contextlib import contextmanager

from rich.console import Console

TEST_CONSOLE_WIDTH = 120
"""Wide enough that realistic table rows render untruncated."""

TEST_CONSOLE_HEIGHT = 100
"""Arbitrary. Present only to satisfy the both-dimensions requirement."""


def fixed_console(width: int = TEST_CONSOLE_WIDTH, **kwargs: object) -> Console:
    """A `Console` that renders at exactly `width`, independent of environment."""
    return Console(width=width, height=TEST_CONSOLE_HEIGHT, **kwargs)  # type: ignore[arg-type]


@contextmanager
def fixed_width(console: Console, width: int = TEST_CONSOLE_WIDTH) -> Iterator[Console]:
    """Pin an already-constructed console (e.g. a module-level singleton) to `width`.

    Reads the private `_width`/`_height` to restore afterwards because the public
    getters report the *computed* size, which would bake this override into the
    singleton for every later test in the session.
    """
    prev_width, prev_height = console._width, console._height
    console.width = width
    console.height = TEST_CONSOLE_HEIGHT
    try:
        yield console
    finally:
        console.width = prev_width  # type: ignore[assignment]
        console.height = prev_height  # type: ignore[assignment]
