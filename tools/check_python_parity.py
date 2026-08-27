#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Enforce the port-and-keep contract for the Python stack.

This branch reimplements AIPerf's execution engine in Rust while KEEPING the
Python stack intact as the cross-engine parity oracle. Porting a feature means
adding the native implementation, never deleting the Python one.

That contract was broken silently and repeatedly. A series of
``merge: integrate origin main <feature> port`` merges resolved to the mainline
side and reverted the Python their own port branches had correctly carried over
from ``origin/main`` -- roughly 4500 lines across 99 files, including whole
modules and their tests. Nothing in git flagged it, because a merge that drops
one parent's content is not recorded as a deletion. It surfaced only as
NameError at runtime, months later.

The invariant below makes that class of regression impossible to land quietly:

1. Every file ``origin/main`` has under ``src/`` and ``tests/`` must exist here
   and be byte-identical, except for the small explicit ``ALLOWED_MODIFIED``
   set. Deleting or editing main's Python is what the bug looked like, so both
   are refused by default.
2. New files under ``src/`` must live in ``src/aiperf/rust_shims/`` (the Rust
   port cordon) or be named in ``ALLOWED_NEW``.
3. New files under ``tests/`` are unrestricted; branch-only tests covering
   native behavior are expected.

Adding to the allowlists is a deliberate, reviewable act. Silently reverting
main's Python is not.

Usage::

    python3 tools/check_python_parity.py [--base origin/main]

Exits 0 when the invariant holds, 1 otherwise.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys

# The Rust-port cordon. Everything here exists solely to serve the native
# runtime and has no counterpart on origin/main.
CORDON = "src/aiperf/rust_shims/"

# Files from origin/main this branch is permitted to modify in place. Each entry
# needs a reason. Keep this list as close to empty as the wiring allows.
ALLOWED_MODIFIED: dict[str, str] = {
    "src/aiperf/common/random_generator.py": (
        "selects the rust RNG backend and routes to aiperf.rust_shims.rng_parity"
    ),
}

# Directory prefixes from origin/main this branch is permitted to modify. Prefer
# ALLOWED_MODIFIED; use this only when one mechanical change spans a whole tree.
ALLOWED_MODIFIED_PREFIXES: dict[str, str] = {}

# On origin/main the in-tree Python mock server is pip-installed
# (tests/aiperf_mock_server/pyproject.toml) and imported as the top-level
# `aiperf_mock_server`. This branch cannot install it: its [project.scripts]
# declares an `aiperf-mock-server` console script that would collide on the venv
# PATH with the native Rust binary of the same name. The test tree therefore
# imports it as a subpackage of tests/ instead.
#
# Rather than exempt all of tests/, this permits exactly that rewrite: a modified
# test file passes only if applying the substitution to origin/main's content
# reproduces this branch's content byte for byte. Any other edit still fails.
MOCK_SERVER_IMPORT = re.compile(r"(?<!tests\.)\baiperf_mock_server\b")
MOCK_SERVER_REPLACEMENT = "tests.aiperf_mock_server"


def _is_only_mock_server_rewrite(base_text: str, head_text: str) -> bool:
    """Report whether head differs from base solely by the mock-server rewrite."""
    return MOCK_SERVER_IMPORT.sub(MOCK_SERVER_REPLACEMENT, base_text) == head_text


def _blob_text(ref: str, path: str) -> str | None:
    """Return the decoded file contents at a ref, or None when unreadable."""
    try:
        raw = subprocess.run(
            ["git", "show", f"{ref}:{path}"], capture_output=True, check=True
        ).stdout
    except subprocess.CalledProcessError:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


# Branch-only files under src/ that are not in the cordon. Each needs a reason.
ALLOWED_NEW: dict[str, str] = {
    "src/aiperf/entrypoint.py": (
        "console-script entry for aiperf-python; the native binary owns 'aiperf'"
    ),
    "src/aiperf/config/templates/dynosim_offline_replay.yaml": (
        "embedded by Rust via include_str! at cli/src/config/templates_data.rs"
    ),
}

TRACKED_ROOTS = ("src/", "tests/")


def _git(*args: str) -> str:
    """Run a git command and return stdout, raising on failure."""
    result = subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True
    )
    return result.stdout


def _tree_files(ref: str) -> set[str]:
    """List files under the tracked roots for one git ref."""
    out = _git("ls-tree", "-r", "--name-only", ref, "--", *TRACKED_ROOTS)
    return {line for line in out.splitlines() if line}


def _blob_id(ref: str, path: str) -> str | None:
    """Return the blob hash for a path at a ref, or None when absent."""
    try:
        return _git("rev-parse", f"{ref}:{path}").strip()
    except subprocess.CalledProcessError:
        return None


def check(base: str, head: str) -> list[str]:
    """Return a list of invariant violations, empty when the tree is clean."""
    violations: list[str] = []
    base_files = _tree_files(base)
    head_files = _tree_files(head)

    for path in sorted(base_files):
        if path not in head_files:
            violations.append(
                f"DELETED from {base}: {path}\n"
                f"    Porting a feature to Rust does not remove its Python. If this "
                f"removal is intended, say so explicitly in the commit."
            )
            continue
        if _blob_id(base, path) == _blob_id(head, path):
            continue
        if path in ALLOWED_MODIFIED:
            continue
        if any(path.startswith(prefix) for prefix in ALLOWED_MODIFIED_PREFIXES):
            continue
        base_text = _blob_text(base, path)
        head_text = _blob_text(head, path)
        if (
            base_text is not None
            and head_text is not None
            and _is_only_mock_server_rewrite(base_text, head_text)
        ):
            continue
        violations.append(
            f"MODIFIED vs {base}: {path}\n"
            f"    Add it to ALLOWED_MODIFIED in this script with a reason, or "
            f"move the change into {CORDON}."
        )

    for path in sorted(head_files - base_files):
        if not path.startswith("src/"):
            continue
        if path.startswith(CORDON) or path in ALLOWED_NEW:
            continue
        violations.append(
            f"NEW outside the cordon: {path}\n"
            f"    Rust-port Python belongs in {CORDON}. Otherwise add it to "
            f"ALLOWED_NEW with a reason."
        )

    return violations


def main() -> int:
    """Check the Python parity invariant and report any violations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="origin/main", help="base ref to compare against")
    parser.add_argument("--head", default="HEAD", help="ref to check")
    args = parser.parse_args()

    try:
        violations = check(args.base, args.head)
    except subprocess.CalledProcessError as error:
        print(f"git failed: {error.stderr.strip() or error}", file=sys.stderr)
        return 2

    if violations:
        print(
            f"Python parity contract violated ({len(violations)} issue(s)).\n"
            f"This branch keeps {args.base}'s Python intact; the Rust port is additive.\n",
            file=sys.stderr,
        )
        for violation in violations:
            print(f"  {violation}\n", file=sys.stderr)
        return 1

    print(f"Python parity contract holds against {args.base}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
