# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One broken guide must not take the whole suite down.

A malformed tag attribute (``timeout=abc``) drops its own command rather than
aborting discovery. If that was a server's only run command, the server is left
incomplete -- and failing validation outright would skip every other documented
server too, turning one typo into a suite-wide outage that hides every real
result. The broken server is reported and skipped; the run still fails.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pytest import param

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tests/ci/test_docs_end_to_end"))

from data_types import Command, Server  # noqa: E402
from test_runner import EndToEndTestRunner  # noqa: E402


def _command(tag: str) -> Command:
    return Command(
        tag_name=tag, command="echo hi", file_path="doc.md", start_line=1, end_line=2
    )


def _server(name: str, *, with_runs: bool = True) -> Server:
    return Server(
        name=name,
        setup_command=_command(f"setup-{name}-endpoint-server"),
        health_check_command=_command(f"health-check-{name}-endpoint-server"),
        aiperf_commands=[_command(f"aiperf-run-{name}-endpoint-server")]
        if with_runs
        else [],
        files=[],
    )


@pytest.mark.parametrize(
    "missing",
    [
        param("setup_command", id="no-setup"),
        param("health_check_command", id="no-health-check"),
        param("aiperf_commands", id="no-run-commands"),
    ],
)  # fmt: skip
def test_only_the_incomplete_server_is_reported(missing: str) -> None:
    broken = _server("broken")
    if missing == "aiperf_commands":
        broken.aiperf_commands = []
    else:
        setattr(broken, missing, None)

    servers = {"good": _server("good"), "broken": broken}

    invalid = EndToEndTestRunner()._validate_servers(servers)

    assert invalid == ["broken"], (
        "a complete server must stay runnable when another one is incomplete"
    )


def test_a_fully_valid_set_reports_nothing_invalid() -> None:
    servers = {"a": _server("a"), "b": _server("b")}

    assert EndToEndTestRunner()._validate_servers(servers) == []
