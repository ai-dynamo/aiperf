# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A tagged command must survive the trip to the container verbatim.

Guides routinely pass JSON in single quotes -- ``--extra-inputs
'{"temperature": 0}'``. Interpolating that into a ``bash -c '<command>'``
*shell string* strips the quotes, so AIPerf receives ``{temperature: 0}`` and
rejects it as invalid JSON. Passing the command as an argv element keeps it
intact, and every guide using a quote stays taggable.
"""

from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path

import pytest
from pytest import param

from aiperf.common.constants import IS_WINDOWS

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tests/ci/test_docs_end_to_end"))

from test_runner import build_command_exec_argv  # noqa: E402

# The docs-e2e runner only ever executes inside a Linux container, and these
# tests assert on real shell behaviour, so there is nothing to verify on a
# platform without bash.
pytestmark = pytest.mark.skipif(
    IS_WINDOWS, reason="docs-e2e commands run in a Linux container; needs bash"
)

QUOTED = """aiperf profile --model m --extra-inputs '{"temperature": 0}'"""

# docs/tutorials/local-tokenizer.md: a `python -c "..."` setup line whose inner
# single quotes would close an outer `bash -c '...'` wrapper early.
PYTHON_SETUP = (
    'python -c "from transformers import AutoTokenizer; '
    "AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B')"
    ".save_pretrained('./local-tokenizer')\"\n"
    "\naiperf profile --tokenizer ./local-tokenizer --model Qwen/Qwen3-0.6B"
)


@pytest.mark.parametrize(
    "command",
    [
        param(QUOTED, id="single-quoted-json"),
        param("aiperf profile --model m --header 'X-Api-Key: abc'", id="quoted-header"),
        param('aiperf profile --model m --extra-inputs "a=b"', id="double-quoted"),
        param("aiperf profile --model m", id="plain"),
        param(PYTHON_SETUP, id="python-c-with-inner-single-quotes"),
    ],
)  # fmt: skip
def test_command_reaches_the_shell_unmodified(command: str) -> None:
    """argv delivery is byte-exact; `bash -c '<cmd>'` interpolation is not."""
    argv = build_command_exec_argv("unused", f"printf '%s' {shlex.quote(command)}")
    proc = subprocess.run(
        argv[argv.index("bash") :],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == command


def test_single_quote_interpolation_is_what_broke() -> None:
    """Pins the old behaviour so the reason for stdin delivery stays visible.

    ``f"bash -c '{command}'"`` run through an outer shell lets the command's
    own single quotes close the outer quoting early, so the JSON payload is
    truncated mid-token -- which is exactly the CI failure:
    ``Failed to parse JSON string: '{temperature:'``.
    """
    outer = f"bash -c 'echo ARGS: {QUOTED}'"
    mangled = subprocess.run(
        outer, shell=True, capture_output=True, text=True
    ).stdout.strip()

    assert mangled.endswith("{temperature:")
    assert '{"temperature": 0}' not in mangled
