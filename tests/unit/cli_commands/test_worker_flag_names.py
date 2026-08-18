# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep local and Kubernetes worker-count CLI names disjoint."""

from __future__ import annotations

import subprocess
import sys

import pytest
from cyclopts.exceptions import UnknownOptionError
from pytest import param

from aiperf.cli import app


def _help(*command: str) -> str:
    code = (
        "import aiperf.cli\n"
        "try:\n"
        f"    aiperf.cli.app({[*command, '--help']!r})\n"
        "except SystemExit:\n"
        "    pass\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


@pytest.mark.parametrize(
    "command",
    [
        param(("kube", "profile"), id="kube-profile"),
        param(("kube", "generate"), id="kube-generate"),
        param(("kube", "sweep"), id="kube-sweep"),
    ],
)  # fmt: skip
def test_kube_commands_expose_only_total_workers(command: tuple[str, ...]) -> None:
    output = _help(*command)

    assert "--total-workers" in output
    assert "--workers-max" not in output
    assert "--max-workers" not in output
    for legacy_flag in ("--workers-max", "--max-workers"):
        with pytest.raises(UnknownOptionError):
            app.parse_args(
                [
                    *command,
                    "--model",
                    "test-model",
                    "--image",
                    "test",
                    legacy_flag,
                    "2",
                ],
                exit_on_error=False,
                print_error=False,
            )


def test_local_profile_retains_workers_max() -> None:
    output = _help("profile")

    assert "--workers-max" in output
    assert "--max-workers" in output
    assert "--total-workers" not in output
