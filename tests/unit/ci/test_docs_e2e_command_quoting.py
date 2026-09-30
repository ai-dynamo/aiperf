# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A tagged command must survive the trip to the container verbatim.

Guides routinely pass JSON in single quotes -- ``--extra-inputs
'{"temperature": 0}'``. Interpolating that into ``bash -c '<command>'``
strips the quotes, so AIPerf receives ``{temperature: 0}`` and rejects it as
invalid JSON. Passing the command over stdin keeps it intact, and every guide
using a quote stays taggable.
"""

from __future__ import annotations

import subprocess

import pytest
from pytest import param

QUOTED = """aiperf profile --model m --extra-inputs '{"temperature": 0}'"""


@pytest.mark.parametrize(
    "command",
    [
        param(QUOTED, id="single-quoted-json"),
        param("aiperf profile --model m --header 'X-Api-Key: abc'", id="quoted-header"),
        param('aiperf profile --model m --extra-inputs "a=b"', id="double-quoted"),
        param("aiperf profile --model m", id="plain"),
    ],
)  # fmt: skip
def test_command_reaches_the_shell_unmodified(command: str) -> None:
    """stdin delivery is byte-exact; `bash -c '<cmd>'` interpolation is not."""
    proc = subprocess.run(
        ["bash", "-s"],
        input=f"printf '%s' {chr(34)}$(cat <<'AIPERF_EOF'\n{command}\nAIPERF_EOF\n)\x22",
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
