#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Guard: keep the native-Rust architecture map in lockstep with specs and crates.

Enforces the "Keeping these docs current" mandate in the four agent instruction
files. A change FAILS if it:

  * changes any ``specs/*.md`` (other than ``specs/README.md``) without also
    updating its canonical entry in ``specs/README.md``; and, for a spec ADD /
    REMOVE / RENAME, without also updating the canonical index in ``llms.txt``.
  * ADDS / REMOVES / RENAMES a crate (a ``rust/<name>/Cargo.toml``) without
    updating all four agent instruction files (the crate-topology table) AND
    ``llms.txt`` (the crate table).

Modes:
    check_docs_current.py                # staged changes  (pre-commit)
    check_docs_current.py --base <ref>   # <ref>...HEAD    (CI, e.g. origin/main)

Bypass (use sparingly, e.g. a pure doc-typo sweep):
    DOCS_GUARD_SKIP=1 check_docs_current.py

Exit codes: 0 = ok / skipped, 1 = violation, 2 = git error.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

AGENT_FILES = {
    "AGENTS.md",
    "CLAUDE.md",
    ".github/copilot-instructions.md",
    ".cursor/rules/python.mdc",
}
SPECS_INDEX = "specs/README.md"
LLMS = "llms.txt"

_SPEC_MD = re.compile(r"^specs/.+\.md$")
_CRATE_MANIFEST = re.compile(r"^rust/[^/]+/Cargo\.toml$")


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, check=True
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        sys.stderr.write(f"check-docs-current: git failed: {' '.join(cmd)}\n{exc}\n")
        raise SystemExit(2) from exc


def changed(base: str | None) -> list[tuple[str, str]]:
    """Return (status, path) pairs; a rename is split into a D + an A."""
    if base:
        out = _run(["git", "diff", "--name-status", "-M", f"{base}...HEAD", "--"])
    else:
        out = _run(["git", "diff", "--cached", "--name-status", "-M", "--"])
    rows: list[tuple[str, str]] = []
    for line in out.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        status = parts[0]
        if status.startswith("R") and len(parts) == 3:
            rows.append(("D", parts[1]))
            rows.append(("A", parts[2]))
        else:
            rows.append((status[0], parts[-1]))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--base",
        default=None,
        help="git ref to diff against (<base>...HEAD); omit to use the staged index",
    )
    args = ap.parse_args()

    if os.environ.get("DOCS_GUARD_SKIP"):
        sys.stderr.write(
            "check-docs-current: SKIPPED via DOCS_GUARD_SKIP — the architecture "
            "map may now be stale; update specs/README.md + llms.txt manually.\n"
        )
        return 0

    rows = changed(args.base)
    paths = {p for _, p in rows}

    spec_rows = [(s, p) for s, p in rows if _SPEC_MD.match(p) and p != SPECS_INDEX]
    spec_touched = bool(spec_rows)
    spec_addremove = any(s in ("A", "D") for s, _ in spec_rows)

    crate_rows = [(s, p) for s, p in rows if _CRATE_MANIFEST.match(p)]
    crate_addremove = any(s in ("A", "D") for s, _ in crate_rows)

    problems: list[str] = []

    if spec_touched and SPECS_INDEX not in paths:
        touched = ", ".join(sorted(p for _, p in spec_rows))
        problems.append(
            f"specs changed ({touched}) but {SPECS_INDEX} was not updated.\n"
            f"    -> add, update, rename, or remove the corresponding canonical "
            f"index entries."
        )
    if spec_addremove and LLMS not in paths:
        problems.append(
            f"a spec was added/removed/renamed but {LLMS} was not updated.\n"
            f"    -> update the specs index in {LLMS}."
        )
    if crate_addremove:
        missing = AGENT_FILES - paths
        if missing:
            problems.append(
                "a crate was added/removed/renamed but the agent instruction files "
                "were not all updated: " + ", ".join(sorted(missing)) + ".\n"
                "    -> update the crate-topology table + dependency-direction line "
                "in ALL FOUR (they must stay byte-identical; run "
                "tools/check_agent_files_sync.py)."
            )
        if LLMS not in paths:
            problems.append(
                f"a crate was added/removed/renamed but {LLMS} was not updated.\n"
                f"    -> update the crate table in {LLMS}."
            )

    if problems:
        sys.stderr.write(
            "check-docs-current: the architecture map is out of sync with this "
            "change.\nDocs must move in the SAME commit (see 'Keeping these docs "
            "current' in CLAUDE.md).\n\n"
        )
        for i, p in enumerate(problems, 1):
            sys.stderr.write(f"  {i}. {p}\n")
        sys.stderr.write(
            "\nIf this change genuinely needs none of the above, bypass with "
            "DOCS_GUARD_SKIP=1 and explain why in the commit message.\n"
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
