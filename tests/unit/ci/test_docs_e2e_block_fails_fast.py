# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A tagged block of several commands must fail on the first error.

Commands are delivered to the container over stdin and run by bash. Without
``-e``, bash reports only the LAST command's exit status, so a guide whose
setup step crashes still passes as long as its final command succeeds. That is
the "reads as covered, never actually checked" outcome this suite exists to
prevent, and it is invisible: the shard goes green.

Eleven of the tagged blocks run more than one command -- generating a trace
fixture, resolving a path, then profiling -- so this is not hypothetical.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from aiperf.common.constants import IS_WINDOWS

HARNESS = Path(__file__).resolve().parents[3] / "tests/ci/test_docs_end_to_end"
sys.path.insert(0, str(HARNESS))

pytestmark = pytest.mark.skipif(
    IS_WINDOWS, reason="docs-e2e commands run in a Linux container; needs bash"
)


def _bash_flags() -> str:
    """The flags the runner actually passes, read from its source.

    Asserting on the real argv rather than a copy: a test that hardcodes
    ``-se`` would keep passing if the runner reverted to ``-s``.
    """
    source = (HARNESS / "test_runner.py").read_text()
    marker = source.index('self.aiperf_container_id,\n                "bash",')
    tail = source[marker : marker + 200]
    start = tail.index('"bash",') + len('"bash",')
    raw = tail[start : tail.index("]", start)].strip().strip(",").strip('"')
    return raw.lstrip("-")


def test_the_runner_asks_bash_to_fail_fast() -> None:
    assert "e" in _bash_flags(), (
        "the runner must pass -e; without it only the last command's exit "
        "status is observed and an earlier failure is silently ignored"
    )


def test_an_early_failure_is_invisible_without_the_flag() -> None:
    """Pins why the flag is needed, not just that it is present."""
    block = "false\nfalse\ntrue\n"
    without = subprocess.run(
        ["bash", "-s"], input=block, text=True, capture_output=True
    )
    assert without.returncode == 0, "baseline assumption changed"

    with_flag = subprocess.run(
        ["bash", f"-{_bash_flags()}"], input=block, text=True, capture_output=True
    )
    assert with_flag.returncode != 0, (
        "a block whose earlier commands fail must not report success"
    )


def test_a_fully_successful_block_still_passes() -> None:
    block = "true\necho ok\n"
    result = subprocess.run(
        ["bash", f"-{_bash_flags()}"], input=block, text=True, capture_output=True
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
