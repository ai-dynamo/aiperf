# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the docs-e2e per-command ``timeout=`` annotation.

Sweeps and multi-phase workflows run far longer than a single-point benchmark.
With only the shared ``AIPERF_COMMAND_TIMEOUT``, such a guide is either
untestable or forces the global ceiling up for every other command.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pytest import param

HARNESS = Path(__file__).resolve().parents[3] / "tests/ci/test_docs_end_to_end"
sys.path.insert(0, str(HARNESS))

from constants import AIPERF_COMMAND_TIMEOUT  # noqa: E402
from parser import MarkdownParser  # noqa: E402

SERVER = "vllm-default-openai"


def _cmd(tmp_path: Path, tag_attrs: str):
    doc = tmp_path / "g.md"
    doc.write_text(
        f"""
<!-- aiperf-run-{SERVER}-endpoint-server{tag_attrs} -->
```bash
aiperf profile --model m
```
<!-- /aiperf-run-{SERVER}-endpoint-server -->
""",
        encoding="utf-8",
    )
    p = MarkdownParser()
    p._parse_file(str(doc))
    return p.servers[SERVER].aiperf_commands[0]


def test_timeout_defaults_to_none_so_the_shared_ceiling_applies(tmp_path: Path) -> None:
    command = _cmd(tmp_path, "")
    assert command.timeout is None
    assert (command.timeout or AIPERF_COMMAND_TIMEOUT) == AIPERF_COMMAND_TIMEOUT


def test_timeout_annotation_is_parsed(tmp_path: Path) -> None:
    assert _cmd(tmp_path, " timeout=3600").timeout == 3600


@pytest.mark.parametrize(
    "attrs,weight,timeout",
    [
        param(" weight=150 timeout=3600", 150, 3600, id="weight-then-timeout"),
        param(" timeout=3600 weight=150", 150, 3600, id="timeout-then-weight"),
    ],
)  # fmt: skip
def test_weight_and_timeout_are_order_independent(
    tmp_path: Path, attrs: str, weight: int, timeout: int
) -> None:
    command = _cmd(tmp_path, attrs)
    assert (command.weight, command.timeout) == (weight, timeout)


def test_weight_alone_still_works(tmp_path: Path) -> None:
    """The original single-attribute form must keep parsing."""
    command = _cmd(tmp_path, " weight=300")
    assert command.weight == 300
    assert command.timeout is None


def test_the_runner_uses_the_per_command_timeout() -> None:
    """Pins the value at the point the watchdog consumes it.

    Asserting only that the parser produced `timeout=` leaves the runner free
    to pass `AIPERF_COMMAND_TIMEOUT` to `threading.Timer` regardless, which no
    parser-level test can detect.
    """
    from test_runner import resolve_command_timeout

    class _Cmd:
        timeout = 5400

    assert resolve_command_timeout(_Cmd()) == 5400

    class _NoTimeout:
        timeout = None

    assert resolve_command_timeout(_NoTimeout()) == AIPERF_COMMAND_TIMEOUT
