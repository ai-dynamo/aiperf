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


def test_allowlisted_command_group_warns_instead_of_aborting(monkeypatch) -> None:
    """A declared-unresolvable group must not break every unrelated commit.

    The generator runs from pre-commit on any `.py` change, so a blanket raise
    makes one broken command group block every contributor until it is fixed.
    An entry in the allowlist is a reviewed, visible admission that the group
    is missing from the page -- the omission is warned about rather than
    silently accepted, and every group NOT listed still hard-fails.
    """

    def _boom(cmd: Any) -> None:
        raise ValueError("Cannot apply configuration to imported App")

    monkeypatch.setattr(generate_cli_docs, "_resolve_lazy_commands", _boom)
    monkeypatch.setattr(
        generate_cli_docs, "KNOWN_UNRESOLVABLE_COMMANDS", frozenset({"broken-group"})
    )

    warnings: list[str] = []
    monkeypatch.setattr(generate_cli_docs, "print_warning", warnings.append)

    assert extract_commands(_FakeApp()) == []
    assert any("broken-group" in w for w in warnings), (
        f"the skipped group must be announced, got {warnings}"
    )


def test_allowlist_only_covers_its_own_entries(monkeypatch) -> None:
    """A non-empty allowlist must not turn every resolve failure into a warning."""

    def _boom(cmd: Any) -> None:
        raise ValueError("Cannot apply configuration to imported App")

    monkeypatch.setattr(generate_cli_docs, "_resolve_lazy_commands", _boom)
    monkeypatch.setattr(
        generate_cli_docs,
        "KNOWN_UNRESOLVABLE_COMMANDS",
        frozenset({"some-other-group"}),
    )

    with pytest.raises(RuntimeError, match="broken-group"):
        extract_commands(_FakeApp())


def test_real_cli_resolves_every_command_group() -> None:
    """The shipped CLI must actually resolve, not merely tolerate failure."""
    from aiperf.cli import app

    commands = extract_commands(app)

    assert commands, "no commands extracted from the real app"
    assert any(name == "profile" for name, _ in commands)
