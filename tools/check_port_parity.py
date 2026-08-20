# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check a reconstructed branch against the port's source-of-truth reference.

The Kubernetes feature has been re-ported several times: each generation was
re-authored onto fresh ``main`` and its content restored as hand-picked
"vertical slices" rather than rebased or cherry-picked. That method silently
drops anything living outside a named slice -- a one-line ``asyncio.to_thread``
inside a pre-existing non-Kubernetes file, or three hooks inside a shared
config -- because no slice claims it and no test fails without it.

This gate diffs the reference against the current tree and reports every file
where the reference still holds content the reconstruction does not. Divergences
that are deliberate (the reconstruction moved, rewrote, or intentionally dropped
something) are recorded in ``tools/port_parity_baseline.json`` with a reason, so
the check fails only on *new* drift.

Usage::

    python tools/check_port_parity.py --ref ajc/k8s-clean-port
    python tools/check_port_parity.py --ref ajc/k8s-clean-port --update-baseline
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

BASELINE_PATH = Path(__file__).parent / "port_parity_baseline.json"

# Paths where a reference-only delta is never interesting: generated artifacts,
# lockfiles, and scratch trees that no reconstruction is expected to carry.
IGNORED_PREFIXES: tuple[str, ...] = (
    "uv.lock",
    "dev/benchmarks/",
    "docs/superpowers/",
    ".superpowers/",
)


def _git(*args: str) -> str:
    """Run a git command from the repo root and return its stdout."""
    result = subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def _ref_exists(ref: str) -> bool:
    """Return whether the reference is resolvable in this clone."""
    try:
        _git("rev-parse", "--verify", f"{ref}^{{commit}}")
    except RuntimeError:
        return False
    return True


def find_reference_only_content(ref: str, target: str) -> dict[str, int]:
    """Map each file to the count of lines the reference has and target lacks.

    ``git diff ref target`` reports removals relative to the reference, so a
    removed line is content that exists in ``ref`` and not in ``target`` -- the
    direction that catches a dropped port.
    """
    raw = _git("diff", "--numstat", ref, target)
    dropped: dict[str, int] = {}
    for line in raw.splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        _added, removed, path = parts
        if removed == "-":  # binary file
            continue
        if path.startswith(IGNORED_PREFIXES):
            continue
        if int(removed) > 0:
            dropped[path] = int(removed)
    return dropped


def load_baseline() -> dict[str, Any]:
    """Load the accepted-divergence baseline, or an empty one if absent."""
    if not BASELINE_PATH.exists():
        return {"accepted": {}}
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


def main() -> int:
    """Compare the working branch against the reference and report new drift."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ref",
        default="ajc/k8s-clean-port",
        help="Source-of-truth ref the current branch was reconstructed from.",
    )
    parser.add_argument(
        "--target", default="HEAD", help="Ref to check (defaults to HEAD)."
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Record current divergences as accepted, each needing a reason.",
    )
    args = parser.parse_args()

    if not _ref_exists(args.ref):
        # A fresh clone has no port-history refs. Skipping is correct: this is a
        # release gate for the porting worktree, not a universal build step.
        print(f"port-parity: reference {args.ref!r} not present, skipping.")
        return 0

    dropped = find_reference_only_content(args.ref, args.target)
    baseline = load_baseline()
    accepted: dict[str, Any] = baseline.get("accepted", {})

    if args.update_baseline:
        merged = dict(accepted)
        for path, count in sorted(dropped.items()):
            entry = merged.get(path)
            reason = entry.get("reason", "TODO: explain") if entry else "TODO: explain"
            merged[path] = {"lines": count, "reason": reason}
        BASELINE_PATH.write_text(
            json.dumps({"accepted": merged}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"port-parity: baseline updated with {len(merged)} accepted path(s).")
        return 0

    new_drift = {
        path: count
        for path, count in dropped.items()
        # A larger delta than recorded means content was dropped since the
        # baseline was taken, even though the path itself is already accepted.
        if path not in accepted or count > accepted[path].get("lines", 0)
    }

    if not new_drift:
        print(
            f"port-parity: OK. {len(dropped)} accepted divergence(s) vs {args.ref}, "
            "no new dropped content."
        )
        return 0

    print(f"port-parity: FAILED against {args.ref}\n")
    print("These files still hold content in the reference that this branch lacks.")
    print("Port it, or record it as deliberate with --update-baseline + a reason.\n")
    for path, count in sorted(new_drift.items(), key=lambda kv: -kv[1]):
        was = accepted.get(path, {}).get("lines")
        grew = f" (baseline allowed {was})" if was is not None else ""
        print(f"  {count:>6} line(s)  {path}{grew}")
    print(f"\n  git diff {args.ref} {args.target} -- <path>")
    return 1


if __name__ == "__main__":
    sys.exit(main())
