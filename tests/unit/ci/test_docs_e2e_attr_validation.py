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
