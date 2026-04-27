# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-shot wipe of pre-epoch results from the AIPerf operator PVC.

A pre-epoch dir is one whose contents are NOT exclusively decimal-seconds
epoch subdirs. This includes:

- ``<ns>/<name>/profile_export_aiperf.json`` directly (no <epoch>/).
- ``<ns>/<name>/legacy/...`` (the old migration target).
- ``<ns>/sweeps/<name>/aggregate.json`` directly.

Run on the operator pod via ``kubectl exec``::

    kubectl exec -n acasagrande-aiperf deploy/aiperf-operator -c operator -- \\
        python /app/tools/wipe_pre_epoch_results.py /data --apply
"""

from __future__ import annotations

import logging
import re
import shutil
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Mirrors EPOCH_RE in results_layout.py post-Task-1 (no |^legacy$ branch).
_EPOCH_RE = re.compile(r"^\d{9,11}$")
_RESERVED_NAMES = {"sweeps"}


def _is_pure_epoch_dir(p: Path) -> bool:
    """A dir is "pure-epoch" if every immediate child is an epoch subdir
    (or the LATEST_POINTER pointer file)."""
    has_any_epoch = False
    for child in p.iterdir():
        if child.is_file():
            if child.name == "latest.txt":
                continue
            return False
        if not _EPOCH_RE.match(child.name):
            return False
        has_any_epoch = True
    return has_any_epoch


def scan_pre_epoch(base: Path) -> list[Path]:
    """Return the list of <name> directories that look pre-epoch."""
    targets: list[Path] = []
    if not base.is_dir():
        return targets
    for ns_dir in sorted(base.iterdir()):
        if not ns_dir.is_dir():
            continue
        # Job dirs: <ns>/<name>/...
        for name_dir in sorted(ns_dir.iterdir()):
            if not name_dir.is_dir():
                continue
            if name_dir.name in _RESERVED_NAMES:
                continue
            if not _is_pure_epoch_dir(name_dir):
                targets.append(name_dir)
        # Sweep dirs: <ns>/sweeps/<name>/...
        sweeps_root = ns_dir / "sweeps"
        if sweeps_root.is_dir():
            for sweep_dir in sorted(sweeps_root.iterdir()):
                if not sweep_dir.is_dir():
                    continue
                if not _is_pure_epoch_dir(sweep_dir):
                    targets.append(sweep_dir)
    return targets


def wipe_pre_epoch(base: Path, *, dry_run: bool = True) -> int:
    """Delete every pre-epoch dir found by ``scan_pre_epoch``.

    Returns the number of dirs deleted (or that would have been deleted in dry-run).
    """
    targets = scan_pre_epoch(base)
    for t in targets:
        if dry_run:
            logger.info("DRY-RUN would delete %s", t)
        else:
            logger.info("DELETING %s", t)
            shutil.rmtree(t)
    return len(targets)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = argv if argv is not None else sys.argv[1:]
    if not args or args[0] in {"-h", "--help"}:
        print(
            "usage: wipe_pre_epoch_results.py <base_dir> [--apply]\n"
            "  default is dry-run; pass --apply to actually delete.",
            file=sys.stderr,
        )
        return 2
    base = Path(args[0])
    apply = "--apply" in args[1:]
    n = wipe_pre_epoch(base, dry_run=not apply)
    print(
        f"{'wiped' if apply else 'would wipe'} {n} pre-epoch directories under {base}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
