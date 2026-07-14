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
Rewrite the agent instruction files so they are byte-identical below their
per-tool header preambles.

This is the write-side companion to ``tools/check_agent_files_sync.py``: the
checker reports drift, this script fixes it. It copies the *body* (everything
from the first ``# AIPerf`` H1 to EOF) of the source file onto every target,
preserving each target's own header (HTML SPDX comment, Cursor frontmatter,
etc.) untouched.

Files kept in sync:
  - AGENTS.md
  - CLAUDE.md
  - .github/copilot-instructions.md
  - .cursor/rules/python.mdc

The default source is ``CLAUDE.md`` (the canonical human-facing file); override
with ``--source``. Use ``--check`` for a non-mutating dry run that exits 1 if
any target would change — suitable for CI / pre-commit alongside the checker.

Usage:
    python tools/sync_agent_files.py                 # sync all from CLAUDE.md
    python tools/sync_agent_files.py --source AGENTS.md
    python tools/sync_agent_files.py --check         # dry run, nonzero on drift
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

BODY_ANCHOR = "# AIPerf"
DEFAULT_SOURCE = "CLAUDE.md"
TARGETS = (
    "AGENTS.md",
    "CLAUDE.md",
    ".github/copilot-instructions.md",
    ".cursor/rules/python.mdc",
)


def split_header_body(path: Path) -> tuple[str, str]:
    """Return ``(header, body)`` split at the first ``# AIPerf`` H1.

    The header is everything above the anchor (kept verbatim per file); the body
    is the anchor line through EOF. The anchor match is exact rather than a
    generic ``# `` so a stray heading in a header preamble is never mistaken for
    the body start.
    """
    text = path.read_text()
    lines = text.splitlines(keepends=True)
    for idx, line in enumerate(lines):
        if line.startswith(BODY_ANCHOR):
            return "".join(lines[:idx]), "".join(lines[idx:])
    raise SystemExit(f"{path}: could not find '{BODY_ANCHOR}' H1 — header detection failed.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default=DEFAULT_SOURCE,
        choices=TARGETS,
        help=f"File whose body is the source of truth (default: {DEFAULT_SOURCE}).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Dry run: do not write; exit 1 if any target would change.",
    )
    args = parser.parse_args()

    source_path = REPO_ROOT / args.source
    if not source_path.is_file():
        print(f"ERROR: missing source file: {args.source}", file=sys.stderr)
        return 1
    _, canonical_body = split_header_body(source_path)

    changed: list[str] = []
    for rel in TARGETS:
        path = REPO_ROOT / rel
        if not path.is_file():
            print(f"ERROR: missing required agent file: {rel}", file=sys.stderr)
            return 1
        header, body = split_header_body(path)
        if body == canonical_body:
            continue
        changed.append(rel)
        if not args.check:
            path.write_text(header + canonical_body)

    if not changed:
        print(f"All agent files already match {args.source} below their headers.")
        return 0

    if args.check:
        print(
            f"ERROR: {len(changed)} agent file(s) differ from {args.source}: "
            f"{', '.join(changed)}\nRun `python tools/sync_agent_files.py` to fix.",
            file=sys.stderr,
        )
        return 1

    print(f"Synced {len(changed)} file(s) from {args.source}: {', '.join(changed)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
