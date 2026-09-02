# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``exit_on_error`` must not let Rich eat square brackets in exception text.

Exception messages routinely contain brackets — ``list[str]``,
``uv pip install 'aiperf[accuracy]'`` — which Rich parses as style tags and
silently drops when a raw string is rendered in a Panel. That corrupted the
missing-lighteval hint (``'aiperf[accuracy]'`` → ``'aiperf'``). The fix escapes
the exception text before substitution; this test pins it.
"""

from __future__ import annotations

import io

import pytest
from rich.console import Console

import aiperf.cli_utils as cli_utils
from aiperf.cli_utils import exit_on_error


def test_exit_on_error_preserves_bracketed_exception_text(monkeypatch) -> None:
    buffer = io.StringIO()
    # force_terminal=False keeps output plain; the point is the literal text,
    # not styling.
    monkeypatch.setattr(
        cli_utils, "console", Console(file=buffer, force_terminal=False, width=200)
    )

    with (
        pytest.raises(SystemExit) as exc_info,
        exit_on_error(RuntimeError, message="{e}", title="Error", show_traceback=False),
    ):
        raise RuntimeError("Install with: uv pip install 'aiperf[accuracy]'.")

    assert exc_info.value.code == 1
    output = buffer.getvalue()
    # The bracketed extra must survive verbatim — not be stripped to 'aiperf'.
    assert "aiperf[accuracy]" in output
