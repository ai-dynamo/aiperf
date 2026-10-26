# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A malformed tag attribute must isolate to its own command.

`int(attr)` raising inside the parse loop aborts `parse_directory` for every
document, so one typo in one guide silently takes the whole docs-e2e suite
with it. Zero and negative values are rejected rather than accepted because
both are quietly destructive: `timeout=0` falls through to the global default
via an `or`, and `timeout=-1` kills the command the instant it starts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pytest import param

HARNESS = Path(__file__).resolve().parents[3] / "tests/ci/test_docs_end_to_end"
sys.path.insert(0, str(HARNESS))

from parser import MarkdownParser  # noqa: E402

GOOD = """\
<!-- setup-demo-endpoint-server -->
```bash
echo start
```
<!-- health-check-demo-endpoint-server -->
```bash
echo ok
```
<!-- aiperf-run-demo-endpoint-server -->
```bash
aiperf profile --model m
```
<!-- /aiperf-run-demo-endpoint-server -->
"""

BAD_TEMPLATE = """\
<!-- aiperf-run-demo-endpoint-server {attr} -->
```bash
aiperf profile --model bad
```
<!-- /aiperf-run-demo-endpoint-server -->
"""


def _parse(tmp_path: Path, *docs: str) -> dict:
    for n, text in enumerate(docs):
        (tmp_path / f"doc{n}.md").write_text(text)
    result = MarkdownParser().parse_directory(tmp_path)
    return result if isinstance(result, dict) else {s.name: s for s in result}


@pytest.mark.parametrize(
    "attr",
    [
        param("timeout=abc", id="timeout-not-a-number"),
        param("timeout=1h", id="timeout-with-unit"),
        param("weight=abc", id="weight-not-a-number"),
        param("timeout=0", id="timeout-zero-would-hit-the-global-default"),
        param("timeout=-1", id="timeout-negative-would-kill-immediately"),
        param("weight=0", id="weight-zero"),
    ],
)  # fmt: skip
def test_malformed_attribute_drops_only_its_own_command(tmp_path, attr) -> None:
    servers = _parse(tmp_path, GOOD, BAD_TEMPLATE.format(attr=attr))

    # The valid guide still parses -- discovery was not aborted.
    assert "demo" in servers
    commands = servers["demo"].aiperf_commands
    assert [c.command.strip() for c in commands] == ["aiperf profile --model m"]


def test_the_dropped_command_is_recorded_on_its_server(tmp_path) -> None:
    """Dropping the command must not also drop the fact that it was dropped.

    The sharper half of the bug: the server still has one valid run command,
    so it validates, executes that one, and reports a pass while the rejected
    command never ran and nothing in the result says so.
    """
    servers = _parse(tmp_path, GOOD, BAD_TEMPLATE.format(attr="timeout=abc"))

    errors = servers["demo"].parse_errors
    assert len(errors) == 1, "the rejected command left no trace in the result"
    assert "doc1.md:1" in errors[0]


def test_a_clean_parse_records_no_errors(tmp_path) -> None:
    assert _parse(tmp_path, GOOD)["demo"].parse_errors == []


def test_valid_attributes_are_applied(tmp_path) -> None:
    servers = _parse(
        tmp_path,
        GOOD.replace(
            "<!-- aiperf-run-demo-endpoint-server -->",
            "<!-- aiperf-run-demo-endpoint-server weight=3 timeout=1800 -->",
        ),
    )
    command = servers["demo"].aiperf_commands[0]
    assert command.weight == 3
    assert command.timeout == 1800


@pytest.mark.parametrize(
    "attr",
    [
        param("timout=3600", id="misspelled-timeout"),  # codespell:ignore
        param("weigth=250", id="misspelled-weight"),  # codespell:ignore
        param("Timeout=3600", id="wrong-case"),
        param("path=config.yaml", id="file-attribute-on-a-run-tag"),
    ],
)  # fmt: skip
def test_an_unknown_attribute_drops_its_command(tmp_path, attr) -> None:
    """An attribute the harness does not read is almost always a typo.

    Ignoring it is the quiet kind of wrong: a misspelled `timeout=` reads as a
    raised deadline, the command keeps the 1200s default, and the SIGKILL twenty
    minutes later looks like a slow command rather than an attribute that
    never applied. Treated like an invalid value -- drop the command, record
    it, fail the run.
    """
    servers = _parse(tmp_path, GOOD, BAD_TEMPLATE.format(attr=attr))

    commands = servers["demo"].aiperf_commands
    assert [c.command.strip() for c in commands] == ["aiperf profile --model m"]
    assert len(servers["demo"].parse_errors) == 1
    assert attr.split("=")[0] in servers["demo"].parse_errors[0]


def test_an_unknown_attribute_on_a_file_tag_drops_the_fixture(tmp_path) -> None:
    doc = """\
<!-- setup-file-demo-endpoint-server path=config.yaml weight=3 -->
```yaml
a: 1
```
"""
    servers = _parse(tmp_path, GOOD, doc)

    assert servers["demo"].files == []
    assert len(servers["demo"].parse_errors) == 1
    assert "weight" in servers["demo"].parse_errors[0]


def test_a_comment_that_is_not_a_harness_tag_is_left_alone(tmp_path) -> None:
    """Only harness tags are policed; ordinary markdown comments are not."""
    servers = _parse(tmp_path, GOOD, "<!-- prettier-ignore foo=bar -->\n")

    assert servers["demo"].parse_errors == []


@pytest.mark.parametrize(
    "kind,attr",
    [
        param("setup", "timeout=7200", id="setup-timeout"),
        param("setup", "weight=500", id="setup-weight"),
        param("health-check", "timeout=7200", id="health-check-timeout"),
        param("health-check", "weight=500", id="health-check-weight"),
    ],
)  # fmt: skip
def test_a_run_attribute_on_a_lifecycle_tag_is_rejected(tmp_path, kind, attr) -> None:
    """Nothing reads `weight=` or `timeout=` off a setup or health-check command.

    They are read only from ``aiperf_commands`` -- by ``shard_commands`` and
    ``resolve_command_timeout``. Accepted here they would set a field no code
    consults, leaving the guide looking like its boot deadline moved. That is
    the same failure as a misspelled attribute, so it gets the same answer.
    """
    doc = f"""\
<!-- {kind}-solo-endpoint-server {attr} -->
```bash
echo hi
```
<!-- /{kind}-solo-endpoint-server -->
"""
    servers = _parse(tmp_path, doc)

    server = servers["solo"]
    assert server.setup_command is None
    assert server.health_check_command is None
    assert len(server.parse_errors) == 1
    assert attr.split("=")[0] in server.parse_errors[0]


def test_a_lifecycle_tag_without_attributes_still_parses(tmp_path) -> None:
    """The rejection must not cost the overwhelmingly common bare tag."""
    servers = _parse(tmp_path, GOOD)

    assert servers["demo"].setup_command is not None
    assert servers["demo"].health_check_command is not None
    assert servers["demo"].parse_errors == []
