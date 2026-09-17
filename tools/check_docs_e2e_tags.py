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

"""Check that new tutorials carry runnable docs-e2e tags.

A tutorial whose ``aiperf`` commands are untagged is never executed by
``test_docs_end_to_end``, so it can rot silently: a renamed flag or a removed
dataset leaves a copy-pasteable command that no longer works, and nothing
fails until a user tries it.

This gate is deliberately scoped to *newly added* files. The existing
untagged backlog is tracked separately; failing on it would block every PR.

Usage:
    python tools/check_docs_e2e_tags.py                  # diff against origin/main
    python tools/check_docs_e2e_tags.py --base HEAD~1
    python tools/check_docs_e2e_tags.py --all            # report the whole backlog
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# The harness imports its own modules flatly (``from constants import ...``),
# so its directory has to be on the path rather than its parent package.
sys.path.insert(0, str(REPO_ROOT / "tests" / "ci" / "test_docs_end_to_end"))

from parser import MarkdownParser  # noqa: E402

# Docs that legitimately contain no runnable command, or whose commands
# cannot run in CI. Keep this list short and justified.
ALLOWLIST = {
    "docs/cli-options.md",  # generated reference; every flag appears as prose
    "docs/environment-variables.md",  # generated reference
}


def added_markdown_files(base: str) -> list[Path]:
    """Return docs/*.md files added (not modified) relative to ``base``."""
    try:
        out = subprocess.run(
            ["git", "diff", "--diff-filter=A", "--name-only", base, "--", "docs/"],
            capture_output=True,
            text=True,
            check=True,
            cwd=REPO_ROOT,
        ).stdout
    except subprocess.CalledProcessError as e:
        print(f"::warning::Could not diff against {base}: {e}", file=sys.stderr)
        return []
    return [Path(line) for line in out.split() if line.endswith(".md")]


def has_runnable_command(path: Path) -> bool:
    """Whether the file contains at least one ``aiperf`` command in a code block."""
    infence = False
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            infence = not infence
            continue
        if infence and stripped.startswith("aiperf "):
            return True
    return False


def tagged_run_count(path: Path) -> int:
    """Number of runnable commands the harness's own parser finds in ``path``.

    Reuses ``MarkdownParser`` rather than matching the tag with a local regex,
    so this gate and the test that consumes the tags can never disagree about
    what counts as tagged.
    """
    try:
        parser = MarkdownParser()
        parser._parse_file(str(path))
    except Exception as e:  # pragma: no cover - parser has its own tests
        print(f"::warning::Could not parse {path}: {e}", file=sys.stderr)
        return 0
    return sum(len(server.aiperf_commands) for server in parser.servers.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="origin/main", help="Git ref to diff against")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Report every untagged doc, not just new ones",
    )
    args = parser.parse_args()

    if args.all:
        candidates = sorted((REPO_ROOT / "docs").rglob("*.md"))
        candidates = [p.relative_to(REPO_ROOT) for p in candidates]
    else:
        candidates = added_markdown_files(args.base)

    offenders = []
    for rel in candidates:
        if str(rel) in ALLOWLIST:
            continue
        path = REPO_ROOT / rel
        if not path.exists() or not has_runnable_command(path):
            continue
        if tagged_run_count(path) == 0:
            offenders.append(rel)

    if not offenders:
        print("All checked docs with runnable commands carry docs-e2e tags.")
        return

    label = "doc(s)" if args.all else "newly added doc(s)"
    print(f"\n{len(offenders)} {label} contain runnable `aiperf` commands but no")
    print("docs-e2e tags, so nothing in CI ever executes them:\n")
    for rel in offenders:
        print(f"  {rel}")
    print(
        "\nTag at least one command so the guide is exercised. Wrap the block:\n"
        "\n"
        "    <!-- aiperf-run-vllm-default-openai-endpoint-server -->\n"
        "    ```bash\n"
        "    aiperf profile --model ... \n"
        "    ```\n"
        "    <!-- /aiperf-run-vllm-default-openai-endpoint-server -->\n"
        "\n"
        "See tests/ci/test_docs_end_to_end/README or an already-tagged tutorial\n"
        "such as docs/tutorials/sharegpt.md for the available server names.\n"
    )
    if not args.all:
        sys.exit(1)


if __name__ == "__main__":
    main()
