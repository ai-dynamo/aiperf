# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI command for rendering AIPerf swim-lane plots."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter

app = App(
    name="swim-lane",
    help="Render a per-session swim-lane PNG with concurrency curve underneath.",
)


@app.default
def swim_lane(
    run_dirs: list[Path],
    out: Annotated[Path | None, Parameter(name=["-o", "--out"])] = None,
) -> None:
    """Render a session swim-lane PNG for one or more AIPerf run directories.

    Each run directory must contain ``profile_export.jsonl``. The neighboring
    ``profile_export_aiperf.json`` (when present) is used to draw ramp-done and
    benchmark-end axvlines. Output defaults to ``<run_dir>/swim_lane.png``.

    Examples:
        # Plot a single run, writing to <run_dir>/swim_lane.png
        aiperf analyze swim-lane ./artifacts/my-run/

        # Multiple runs in one invocation
        aiperf analyze swim-lane ./artifacts/run_a/ ./artifacts/run_b/

        # Single run with explicit output path
        aiperf analyze swim-lane ./artifacts/my-run/ -o /tmp/lanes.png

    Args:
        run_dirs: One or more AIPerf run directories.
        out: Output PNG path. Only valid when a single run directory is given.
    """
    from aiperf.analysis.swim_lane import SwimLaneError, plot_swim_lane

    if out is not None and len(run_dirs) > 1:
        print("error: --out only valid with a single run dir", file=sys.stderr)
        sys.exit(2)

    failures = 0
    for run_dir in run_dirs:
        try:
            saved = plot_swim_lane(run_dir, out=out)
        except SwimLaneError as e:
            print(f"skip {run_dir}: {e}", file=sys.stderr)
            failures += 1
            continue
        print(f"saved {saved}")
    if failures and failures == len(run_dirs):
        sys.exit(1)
