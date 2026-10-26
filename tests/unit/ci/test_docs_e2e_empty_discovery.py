# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Discovering no servers at all must fail, not pass.

The harness rejects every narrower way a documented command can leave the
suite: a bad tag attribute drops the command and fails the run, a duplicate
setup tag skips the server and fails the run, an unterminated fence refuses
the block and fails the run. Discovering nothing is all of those at once, and
was the one case that still reported success -- a GPU job that booted a
runner, built the image and executed nothing, reported green.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
HARNESS = REPO / "tests/ci/test_docs_end_to_end"
sys.path.insert(0, str(HARNESS))

import main as harness_main  # noqa: E402
import parser as harness_parser  # noqa: E402


def _run(monkeypatch, discovered: dict, *argv: str) -> int:
    monkeypatch.setattr(
        harness_parser.MarkdownParser, "parse_directory", lambda self, root: discovered
    )
    monkeypatch.setattr(sys, "argv", ["main.py", *argv])
    return harness_main.main()


def test_discovering_no_servers_fails_the_run(monkeypatch) -> None:
    assert _run(monkeypatch, {}, "--all-servers") == 1


def test_discovering_no_servers_fails_a_dry_run_too(monkeypatch) -> None:
    """``--dry-run`` lists what would run; "nothing" is the answer that matters."""
    assert _run(monkeypatch, {}, "--all-servers", "--dry-run") == 1


def test_a_discovered_server_still_reaches_the_dry_run(monkeypatch) -> None:
    """The guard must cost nothing on the path that actually has work."""
    from data_types import Command, Server

    command = Command(
        tag_name="aiperf-run-demo-endpoint-server",
        command="aiperf profile --model m",
        file_path="docs/demo.md",
        start_line=1,
        end_line=4,
    )
    server = Server(
        name="demo",
        setup_command=command,
        health_check_command=command,
        aiperf_commands=[command],
    )
    assert _run(monkeypatch, {"demo": server}, "--all-servers", "--dry-run") == 0
