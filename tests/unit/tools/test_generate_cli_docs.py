# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the CLI documentation generator."""

from typing import Any

import pytest

from tools import generate_cli_docs
from tools.generate_cli_docs import extract_commands


class _FakeGroup:
    """A subcommand-only group whose lazy resolution is broken."""

    _commands: dict[str, Any] = {}
    default_command = None
    help = "a group"


class _FakeApp:
    def __init__(self) -> None:
        self._commands = {"broken-group": _FakeGroup()}


def test_unresolvable_command_group_fails_generation(monkeypatch) -> None:
    """A group that cannot resolve must abort, not vanish from the docs.

    Swallowing the failure drops the group AND every descendant command from
    docs/cli-options.md while the generator still exits 0, so an import or
    registration bug lands as a quietly incomplete page that the pre-commit
    hook accepts. The documentation gate only means something if an
    unresolvable command group is a hard failure.
    """

    def _boom(cmd: Any) -> None:
        raise ImportError("no module named 'aiperf.cli_commands.thing'")

    monkeypatch.setattr(generate_cli_docs, "_resolve_lazy_commands", _boom)

    with pytest.raises(RuntimeError, match="broken-group") as excinfo:
        extract_commands(_FakeApp())

    assert isinstance(excinfo.value.__cause__, ImportError), (
        "the original resolution failure must stay attached for debugging"
    )


def test_real_cli_resolves_every_command_group() -> None:
    """The shipped CLI must actually resolve, not merely tolerate failure."""
    from aiperf.cli import app

    commands = extract_commands(app)

    assert commands, "no commands extracted from the real app"
    assert any(name == "profile" for name, _ in commands)
