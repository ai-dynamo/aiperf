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

A tutorial whose ``aiperf profile`` commands are untagged is never executed by
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
import re
import subprocess
import sys
from collections.abc import Iterator
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
    """Return docs/*.md files added (not modified) relative to ``base``.

    Raises on a failed diff rather than returning nothing: an unavailable base
    ref would otherwise make the gate report success having inspected no files
    at all, which is worse than no gate -- it reads as a passing check.

    ``-z`` because ``--name-only`` quotes paths containing whitespace and
    splitting on whitespace would shear them into fragments that resolve to
    nothing, silently skipping exactly the file being added.
    """
    try:
        out = subprocess.run(
            [
                "git",
                "diff",
                "-z",
                "--diff-filter=A",
                "--name-only",
                base,
                "--",
                "docs/",
            ],
            capture_output=True,
            text=True,
            check=True,
            cwd=REPO_ROOT,
        ).stdout
    except subprocess.CalledProcessError as e:
        raise SystemExit(
            f"::error::Could not diff against {base}: {e.stderr.strip() or e}. "
            "Refusing to report success without checking any files."
        ) from e
    return [Path(entry) for entry in out.split("\0") if entry.endswith(".md")]


# A command may be prefixed with environment assignments
# (``AIPERF_HTTP_VIDEO_POLL_INTERVAL=0.5 aiperf profile ...``), which must be
# stripped before the subcommand is read, or the line reads as no command at
# all and the doc silently escapes the gate.
_ENV_ASSIGNMENT = re.compile(r"""^[A-Za-z_][A-Za-z0-9_]*=(?:"[^"]*"|'[^']*'|\S*)\s+""")


def invokes_aiperf_profile(line: str) -> bool:
    """Whether ``line`` starts a benchmark the docs-e2e harness could run.

    Only ``aiperf profile`` counts. The harness boots a model server and runs
    the tagged block against it; it has no cluster for ``aiperf kube``, no tty
    for ``aiperf chat``, and nothing for ``aiperf plot`` to read until a
    profile run has produced an artifact directory. Treating every ``aiperf``
    line as runnable made the gate demand tags on guides it cannot execute --
    18 of them under ``docs/kubernetes/`` alone -- so a new Kubernetes page
    could not be added without either a tag that does nothing or an entry in
    the allowlist.
    """
    text = line.strip()
    while match := _ENV_ASSIGNMENT.match(text):
        text = text[match.end() :]
    return text.startswith("aiperf profile")


def has_runnable_command(path: Path) -> bool:
    """Whether the file shows an ``aiperf profile`` run a reader could copy.

    Only fenced blocks count: prose naming a flag is not a command, and gating
    on prose would flag every reference page.
    """
    return any(
        invokes_aiperf_profile(line)
        for line in _fenced_lines(path.read_text(encoding="utf-8"))
    )


def _fenced_lines(text: str) -> Iterator[str]:
    """Yield the lines inside fenced code blocks.

    CommonMark allows tildes as well as backticks, and a closing fence must use
    the opener's character and be at least as long. Toggling on any backtick
    run missed tilde-fenced blocks entirely -- so a guide written with tildes
    had its commands invisible here and slipped through untagged -- and let a
    short backtick run inside a longer fence close it early.
    """
    opener: tuple[str, int] | None = None
    for line in text.splitlines():
        stripped = line.strip()
        marker = stripped[:1]
        if marker in ("`", "~"):
            run = len(stripped) - len(stripped.lstrip(marker))
            if opener is None:
                if run >= 3:
                    opener = (marker, run)
                continue
            char, length = opener
            closes = marker == char and run >= length and not stripped[run:].strip()
            if closes:
                opener = None
            continue
        if opener is not None:
            yield line


def tagged_run_count(path: Path) -> int:
    """Reuses ``MarkdownParser`` rather than matching the tag with a local
    regex, so this gate and the suite that consumes the tags can never disagree
    about what counts as tagged.
    """
    try:
        parser = MarkdownParser()
        parser._parse_file(str(path))
    except Exception as e:  # pragma: no cover - parser has its own tests
        print(f"::warning::Could not parse {path}: {e}", file=sys.stderr)
        return 0
    # Count only tagged blocks that actually invoke aiperf profile. The parser
    # categorises a block by its tag without inspecting the body, so a doc
    # could tag an `echo ok` block and satisfy this gate while its real
    # `aiperf profile` command sits untagged beside it.
    return sum(
        1
        for server in parser.servers.values()
        for command in server.aiperf_commands
        if any(invokes_aiperf_profile(line) for line in command.command.splitlines())
    )


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
    print(
        f"\n{len(offenders)} {label} contain runnable `aiperf profile` commands but no"
    )
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
        "See tests/ci/test_docs_end_to_end/README.md or an already-tagged\n"
        "tutorial such as docs/tutorials/sharegpt.md for the server names.\n"
    )
    if not args.all:
        sys.exit(1)


if __name__ == "__main__":
    main()
