# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A fixture path from a guide must not be able to run commands.

`path=` is an attribute of a markdown comment, so it is ordinary documentation
text that reaches a shell. The absolute/`..` guard stops the file being written
outside the working directory, but it does nothing about metacharacters: an
unquoted `cat > {target}` turns `path=x;touch PWNED` into a second command
executing inside the CI container. Content was already safe (it is piped over
stdin); the path was not.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path, PurePosixPath

import pytest
from pytest import param

from aiperf.common.constants import IS_WINDOWS

HARNESS = Path(__file__).resolve().parents[3] / "tests/ci/test_docs_end_to_end"
sys.path.insert(0, str(HARNESS))

from test_runner import build_fixture_write_command  # noqa: E402

pytestmark = pytest.mark.skipif(
    IS_WINDOWS, reason="docs-e2e fixtures are written inside a Linux container"
)


def _fixture_shell_command(path: str) -> str:
    return build_fixture_write_command(PurePosixPath(path))


@pytest.mark.parametrize(
    "path, sentinel",
    [
        param("x;touch PWNED", "PWNED", id="command-separator"),
        param("$(touch PWNED2).yaml", "PWNED2", id="command-substitution"),
        param("`touch PWNED3`.yaml", "PWNED3", id="backtick-substitution"),
        param("a b.yaml", "a", id="word-splitting-creates-a-stray-file"),
        param("sub/$(touch PWNED4).yaml", "PWNED4", id="metacharacter-in-parent"),
    ],
)  # fmt: skip
def test_hostile_fixture_path_executes_nothing(tmp_path, path, sentinel) -> None:
    subprocess.run(
        ["bash", "-c", _fixture_shell_command(path)],
        input="content",
        text=True,
        cwd=tmp_path,
        capture_output=True,
        timeout=30,
    )
    assert not (tmp_path / sentinel).exists(), (
        f"fixture path {path!r} executed a command or split into words, "
        f"creating {sentinel!r}"
    )


def test_an_ordinary_nested_path_still_works(tmp_path) -> None:
    result = subprocess.run(
        ["bash", "-c", _fixture_shell_command("configs/bench.yaml")],
        input="model: m\n",
        text=True,
        cwd=tmp_path,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "configs/bench.yaml").read_text() == "model: m\n"
