# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Backfill `sweep.json` markers for legacy sweeps that ran before
`write_child_sweep_marker` existed.

Walks <results_dir>/<ns>/sweeps/*/aggregate.json, reads child_runs[],
and drops sweep.json into each existing child results dir. Children
that have no results dir are skipped silently — they would have
nothing to render anyway.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import orjson

from aiperf.sweep_controller.k8s_executor import write_child_sweep_marker

logger = logging.getLogger(__name__)


def backfill_sweep_markers(base_dir: Path) -> int:
    """Return the number of markers written."""
    written = 0
    if not base_dir.is_dir():
        return 0
    for ns_dir in sorted(base_dir.iterdir()):
        if not ns_dir.is_dir():
            continue
        sweeps_root = ns_dir / "sweeps"
        if not sweeps_root.is_dir():
            continue
        for sweep_dir in sorted(sweeps_root.iterdir()):
            if not sweep_dir.is_dir():
                continue
            agg_path = sweep_dir / "aggregate.json"
            if not agg_path.is_file():
                continue
            try:
                doc = orjson.loads(agg_path.read_bytes())
            except (OSError, orjson.JSONDecodeError) as e:
                logger.warning("skipping unreadable %s: %s", agg_path, e)
                continue
            sweep_name = sweep_dir.name
            for child in doc.get("child_runs") or []:
                if not isinstance(child, dict):
                    continue
                child_ns = child.get("namespace") or ns_dir.name
                child_name = child.get("name")
                if not child_name:
                    continue
                child_dir = base_dir / child_ns / child_name
                if not child_dir.is_dir():
                    continue
                write_child_sweep_marker(
                    base_dir=base_dir,
                    namespace=child_ns,
                    child_name=child_name,
                    sweep_name=sweep_name,
                    variation_index=int(child.get("variation_index") or 0),
                    variation_label=str(child.get("variation_label") or ""),
                    trial_index=child.get("trial_index"),
                )
                written += 1
    return written


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1:
        print("usage: backfill_sweep_markers.py <results_dir>", file=sys.stderr)
        return 2
    n = backfill_sweep_markers(Path(args[0]))
    print(f"wrote {n} sweep.json markers")
    return 0


if __name__ == "__main__":
    sys.exit(main())
