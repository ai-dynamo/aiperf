# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep local and Kubernetes worker-count CLI names disjoint."""

from __future__ import annotations

import subprocess
import sys


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


def test_local_profile_retains_workers_max() -> None:
    output = _help("profile")

    assert "--workers-max" in output
    assert "--max-workers" in output
    assert "--total-workers" not in output
