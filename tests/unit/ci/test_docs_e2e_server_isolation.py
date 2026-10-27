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


def test_a_rejected_command_fails_the_run_without_skipping_the_server(
    monkeypatch,
) -> None:
    """A complete server that lost one command to a parse error still runs.

    Validation deliberately does not reject it -- the remaining commands are
    real coverage and should execute. But the run as a whole must fail, or a
    documented command silently stops being tested.
    """
    degraded = _server("degraded")
    degraded.parse_errors.append("dropped the command at doc.md:1")
    servers = {"good": _server("good"), "degraded": degraded}

    runner = EndToEndTestRunner()
    tested: list[str] = []
    monkeypatch.setattr(runner, "_cleanup_all_containers", lambda: None)
    monkeypatch.setattr(runner, "_cleanup", lambda: None)
    monkeypatch.setattr(runner, "_build_aiperf_container", lambda: True)
    monkeypatch.setattr(
        runner, "_test_server", lambda server: tested.append(server.name) or True
    )

    assert runner.run_tests(servers) is False, (
        "a command the parser rejected must not report a pass"
    )
    assert sorted(tested) == ["degraded", "good"], (
        "the surviving commands are real coverage and must still execute"
    )


def test_a_clean_set_still_passes(monkeypatch) -> None:
    runner = EndToEndTestRunner()
    monkeypatch.setattr(runner, "_cleanup_all_containers", lambda: None)
    monkeypatch.setattr(runner, "_cleanup", lambda: None)
    monkeypatch.setattr(runner, "_build_aiperf_container", lambda: True)
    monkeypatch.setattr(runner, "_test_server", lambda server: True)

    assert runner.run_tests({"a": _server("a"), "b": _server("b")}) is True


DUPLICATE_DOC = """\
<!-- setup-{name}-endpoint-server -->
```bash
echo boot
```
<!-- /setup-{name}-endpoint-server -->
<!-- health-check-{name}-endpoint-server -->
```bash
echo ok
```
<!-- /health-check-{name}-endpoint-server -->
<!-- aiperf-run-{name}-endpoint-server -->
```bash
aiperf profile --model m
```
<!-- /aiperf-run-{name}-endpoint-server -->
"""

SECOND_SETUP = """\
<!-- setup-{name}-endpoint-server -->
```bash
echo other-boot
```
<!-- /setup-{name}-endpoint-server -->
"""


def _parse_docs(tmp_path: Path, **docs: str) -> dict:
    from parser import MarkdownParser  # noqa: PLC0415

    parser = MarkdownParser()
    for stem, text in docs.items():
        doc = tmp_path / f"{stem}.md"
        doc.write_text(text, encoding="utf-8")
        parser._parse_file(str(doc))
    return parser.servers


def test_a_duplicate_setup_does_not_abort_discovery(tmp_path) -> None:
    """Two guides disagreeing about one server says nothing about the others.

    This used to call ``sys.exit(1)`` mid-parse, so a duplicate tag in one
    guide took every other documented server with it and no other result was
    ever reported.
    """
    servers = _parse_docs(
        tmp_path,
        a_dup=DUPLICATE_DOC.format(name="dup") + SECOND_SETUP.format(name="dup"),
        b_good=DUPLICATE_DOC.format(name="good"),
    )

    assert sorted(servers) == ["dup", "good"]
    assert servers["dup"].command_conflicts == ["setup"]
    assert servers["good"].command_conflicts == []


@pytest.mark.parametrize(
    "kind",
    [
        param("setup", id="duplicate-setup"),
        param("health-check", id="duplicate-health-check"),
    ],
)  # fmt: skip
def test_a_conflicted_server_is_skipped_and_the_run_fails(tmp_path, kind) -> None:
    """Unrecoverable, unlike a dropped command: neither guide can be chosen.

    Running either would benchmark a server booted in a way no guide
    documents, so the server is skipped rather than degraded.
    """
    second = SECOND_SETUP.format(name="dup")
    if kind == "health-check":
        second = second.replace("setup-dup", "health-check-dup").replace(
            "echo other-boot", "echo other-ok"
        )
    servers = _parse_docs(
        tmp_path,
        a_dup=DUPLICATE_DOC.format(name="dup") + second,
        b_good=DUPLICATE_DOC.format(name="good"),
    )

    runner = EndToEndTestRunner.__new__(EndToEndTestRunner)
    invalid = EndToEndTestRunner._validate_servers(runner, servers)

    assert invalid == ["dup"]
    assert [n for n in servers if n not in invalid] == ["good"]


def test_the_first_definition_is_the_one_kept(tmp_path) -> None:
    """Which one wins does not matter -- the server never runs -- but the
    message has to be able to name both, so the first is left in place."""
    servers = _parse_docs(
        tmp_path,
        a_dup=DUPLICATE_DOC.format(name="dup") + SECOND_SETUP.format(name="dup"),
    )

    assert servers["dup"].setup_command.command.strip() == "echo boot"
