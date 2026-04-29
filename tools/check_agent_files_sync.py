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
Check that the four agent instruction files stay byte-identical below their
per-tool header preambles.

Files compared:
  - AGENTS.md                          (canonical source for the diff output)
  - CLAUDE.md
  - .github/copilot-instructions.md
  - .cursor/rules/python.mdc

The "body" of each file is everything from the first H1 (``# AIPerf``) to EOF.
Anything above that line is treated as a per-tool header (HTML SPDX comment,
Cursor frontmatter, etc.) and ignored. If any two bodies differ, prints a
unified diff against AGENTS.md and exits 1.

Usage:
    python tools/check_agent_files_sync.py
"""

from __future__ import annotations

import difflib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

CANONICAL = "AGENTS.md"
TARGETS = (
    "AGENTS.md",
    "CLAUDE.md",
    ".github/copilot-instructions.md",
    ".cursor/rules/python.mdc",
)


def extract_body(path: Path) -> str:
    """Return the file content from the first ``# AIPerf`` H1 onward.

    The H1 anchor is intentionally exact rather than a generic ``# `` match so
    a stray heading inside a header preamble can never be mistaken for the
    start of the body.
    """
    text = path.read_text()
    lines = text.splitlines(keepends=True)
    for idx, line in enumerate(lines):
        if line.startswith("# AIPerf"):
            return "".join(lines[idx:])
    raise SystemExit(f"{path}: could not find '# AIPerf' H1 — header detection failed.")


def main() -> int:
    bodies: dict[str, str] = {}
    for rel in TARGETS:
        path = REPO_ROOT / rel
        if not path.is_file():
            print(f"ERROR: missing required agent file: {rel}", file=sys.stderr)
            return 1
        bodies[rel] = extract_body(path)

    canonical_body = bodies[CANONICAL]
    drift = [rel for rel, body in bodies.items() if body != canonical_body]

    if not drift:
        return 0

    print(
        f"ERROR: agent instruction files have drifted from {CANONICAL}.\n"
        "These four files must share identical content below their headers — "
        "see CLAUDE.md 'Four-File Sync Rule'.\n",
        file=sys.stderr,
    )
    canonical_lines = canonical_body.splitlines(keepends=True)
    for rel in drift:
        diff = difflib.unified_diff(
            canonical_lines,
            bodies[rel].splitlines(keepends=True),
            fromfile=CANONICAL,
            tofile=rel,
            n=3,
        )
        sys.stderr.writelines(diff)
        print("", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
