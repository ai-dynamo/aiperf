# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep local and Kubernetes worker-count CLI names disjoint."""

from __future__ import annotations

import contextlib
import io


def _help(*command: str) -> str:
    """Render ``aiperf <command> --help`` in-process and return its stdout.

    Rendering help does not need process isolation: cyclopts writes to the
    ambient stdout and raises SystemExit, both of which are captured here.
    Spawning a real interpreter per assertion cost ~1s of import time for a
    pure string check.
    """
    import aiperf.cli

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.suppress(SystemExit):
        aiperf.cli.app([*command, "--help"])
    return buf.getvalue()


def test_local_profile_retains_workers_max() -> None:
    output = _help("profile")

    assert "--workers-max" in output
    assert "--max-workers" in output
    assert "--total-workers" not in output
