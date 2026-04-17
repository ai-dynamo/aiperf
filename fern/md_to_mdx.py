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
Convert standard GitHub Markdown to Fern MDX format.

Two transformations are applied in a single pass:

1. HTML comments -> MDX comments
       <!-- tag-name -->   ->  {/* tag-name */}
       <!-- /tag-name -->  ->  {/* /tag-name */}

2. GitHub callouts -> Fern admonitions
       > [!NOTE]           ->  <Note>...</Note>
       > [!TIP]            ->  <Tip>...</Tip>
       > [!IMPORTANT]      ->  <Info>...</Info>
       > [!WARNING]        ->  <Warning>...</Warning>
       > [!CAUTION]        ->  <Error>...</Error>

Run as part of the docs-website sync pipeline on every merge to main.
Designed to be reusable across repos — teams opt in by including this file.

Usage:
    # Convert all markdown files in a directory (in-place)
    python fern/md_to_mdx.py --dir fern/pages-dev

    # Convert a single file in-place
    python fern/md_to_mdx.py fern/pages-dev/tutorials/example.md

    # Preview changes without writing
    python fern/md_to_mdx.py --dir fern/pages-dev --dry-run

    # Run tests
    python fern/md_to_mdx.py --test
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# ── Code fence awareness ─────────────────────────────────────────────────────

# Splits text into alternating non-fence / fence segments.
# Odd-indexed segments are inside fenced code blocks and must not be transformed.
_FENCE_SPLIT = re.compile(r"(^```.*?^```)", re.MULTILINE | re.DOTALL)


def _apply_outside_fences(text: str, fn) -> str:
    """Apply fn only to text outside fenced code blocks."""
    parts = _FENCE_SPLIT.split(text)
    return "".join(fn(p) if i % 2 == 0 else p for i, p in enumerate(parts))


# ── HTML comment -> MDX comment ──────────────────────────────────────────────

# re.DOTALL so . matches newlines, handling multiline HTML comments
HTML_COMMENT_PATTERN = re.compile(r"<!--\s*(.*?)\s*-->", re.DOTALL)


def _convert_comments(text: str) -> str:
    """Replace <!-- ... --> with {/* ... */}, skipping fenced code blocks."""
    return _apply_outside_fences(
        text,
        lambda s: HTML_COMMENT_PATTERN.sub(lambda m: f"{{/* {m.group(1)} */}}", s),
    )


# ── GitHub callouts -> Fern admonitions ──────────────────────────────────────

GITHUB_TO_FERN = {
    "NOTE": "Note",
    "TIP": "Tip",
    "IMPORTANT": "Info",
    "WARNING": "Warning",
    "CAUTION": "Error",
}

GITHUB_ADMONITION_PATTERN = re.compile(
    r"^(?P<indent>[ \t]*)>[ \t]*\[!(?P<type>NOTE|TIP|IMPORTANT|WARNING|CAUTION)\][ \t]*\n"
    r"(?P<content>(?:(?P=indent)>[ \t]*.*\n?)+)",
    re.MULTILINE | re.IGNORECASE,
)


def _extract_blockquote_content(content: str, indent: str) -> str:
    lines = content.split("\n")
    extracted = []
    for line in lines:
        if line.startswith(indent + ">"):
            stripped = line[len(indent) + 1 :]
            if stripped.startswith(" "):
                stripped = stripped[1:]
            extracted.append(stripped)
        elif line.strip() == "":
            break
        else:
            break
    return "\n".join(extracted).rstrip()


def _convert_admonition(match: re.Match) -> str:
    indent = match.group("indent")
    alert_type = match.group("type").upper()
    fern_tag = GITHUB_TO_FERN.get(alert_type, "Note")
    content = _extract_blockquote_content(match.group("content"), indent)
    content_lines = content.split("\n")
    if len(content_lines) == 1 and len(content) < 100:
        return f"{indent}<{fern_tag}>{content}</{fern_tag}>\n"
    formatted = "\n".join(content_lines)
    return f"{indent}<{fern_tag}>\n{formatted}\n{indent}</{fern_tag}>\n"


def _convert_admonitions(text: str) -> str:
    """Replace GitHub > [!TYPE] callouts with Fern <Tag> admonitions, skipping fenced code blocks."""
    return _apply_outside_fences(
        text,
        lambda s: GITHUB_ADMONITION_PATTERN.sub(_convert_admonition, s),
    )


# ── Public API ────────────────────────────────────────────────────────────────


def convert(text: str) -> str:
    """Apply all transformations to convert GitHub Markdown to Fern MDX."""
    text = _convert_comments(text)
    text = _convert_admonitions(text)
    return text


def process_file(path: Path, dry_run: bool = False) -> bool:
    """Convert a single file. Returns True if the file was (or would be) changed."""
    original = path.read_text(encoding="utf-8")
    converted = convert(original)
    if converted == original:
        return False
    if not dry_run:
        path.write_text(converted, encoding="utf-8")
    return True


def process_directory(dir_path: Path, dry_run: bool = False) -> int:
    """Convert all markdown files in a directory. Returns count of changed files."""
    changed = 0
    for file_path in sorted(dir_path.rglob("*.md")):
        if process_file(file_path, dry_run=dry_run):
            action = "Would convert" if dry_run else "Converted"
            print(f"  {action}: {file_path}")
            changed += 1
    return changed


# ── Tests ─────────────────────────────────────────────────────────────────────


def run_tests() -> bool:
    passed = 0
    failed = 0

    def test(name: str, input_text: str, expected: str) -> None:
        nonlocal passed, failed
        result = convert(input_text)
        if result == expected:
            print(f"  PASS: {name}")
            passed += 1
        else:
            print(f"  FAIL: {name}")
            print(f"    Input:    {repr(input_text)}")
            print(f"    Expected: {repr(expected)}")
            print(f"    Got:      {repr(result)}")
            failed += 1

    print("Running tests...\n")

    # HTML comment -> MDX comment
    test(
        "Opening HTML comment tag",
        "<!-- aiperf-run-vllm-default-openai-endpoint-server -->\n",
        "{/* aiperf-run-vllm-default-openai-endpoint-server */}\n",
    )
    test(
        "Closing HTML comment tag",
        "<!-- /aiperf-run-vllm-default-openai-endpoint-server -->\n",
        "{/* /aiperf-run-vllm-default-openai-endpoint-server */}\n",
    )
    test(
        "Tag surrounded by content",
        "Some text\n<!-- my-tag -->\n```bash\necho hi\n```\n<!-- /my-tag -->\nMore text\n",
        "Some text\n{/* my-tag */}\n```bash\necho hi\n```\n{/* /my-tag */}\nMore text\n",
    )
    test(
        "Multiline HTML comment",
        "<!--\nCopyright 2026 NVIDIA\n-->\n\n# Title\n",
        "{/* Copyright 2026 NVIDIA */}\n\n# Title\n",
    )

    # GitHub callouts -> Fern admonitions
    test(
        "NOTE callout",
        "> [!NOTE]\n> This is a note.\n",
        "<Note>This is a note.</Note>\n",
    )
    test(
        "TIP callout",
        "> [!TIP]\n> Helpful tip.\n",
        "<Tip>Helpful tip.</Tip>\n",
    )
    test(
        "IMPORTANT -> Info",
        "> [!IMPORTANT]\n> Key info.\n",
        "<Info>Key info.</Info>\n",
    )
    test(
        "WARNING callout",
        "> [!WARNING]\n> Watch out.\n",
        "<Warning>Watch out.</Warning>\n",
    )
    test(
        "CAUTION -> Error",
        "> [!CAUTION]\n> Dangerous.\n",
        "<Error>Dangerous.</Error>\n",
    )
    test(
        "Multiline callout",
        "> [!NOTE]\n> Line one.\n> Line two.\n",
        "<Note>\nLine one.\nLine two.\n</Note>\n",
    )
    test(
        "Admonition in document",
        "# Header\n\n> [!WARNING]\n> Be careful.\n\nMore text.\n",
        "# Header\n\n<Warning>Be careful.</Warning>\n\nMore text.\n",
    )

    # Code-fence awareness — content inside fences must not be transformed
    test(
        "HTML comment inside code fence unchanged",
        "```bash\n<!-- tag-name -->\n```\n",
        "```bash\n<!-- tag-name -->\n```\n",
    )
    test(
        "Callout inside code fence unchanged",
        "```\n> [!NOTE]\n> This is inside a fence.\n```\n",
        "```\n> [!NOTE]\n> This is inside a fence.\n```\n",
    )
    test(
        "Comment outside fence converted, inside unchanged",
        "<!-- before -->\n```bash\n<!-- inside -->\n```\n<!-- after -->\n",
        "{/* before */}\n```bash\n<!-- inside -->\n```\n{/* after */}\n",
    )

    # No change
    test(
        "Plain markdown unchanged",
        "# Just a header\n\nSome text.\n",
        "# Just a header\n\nSome text.\n",
    )

    print(f"\n{'=' * 40}")
    print(f"Results: {passed} passed, {failed} failed")
    return failed == 0


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert GitHub Markdown to Fern MDX format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input", nargs="?", help="Single file to convert in-place")
    parser.add_argument(
        "--dir", "-d", type=Path, help="Directory of markdown files to convert"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print what would change without writing"
    )
    parser.add_argument("--test", "-t", action="store_true", help="Run test cases")
    args = parser.parse_args()

    if args.test:
        sys.exit(0 if run_tests() else 1)

    if args.dir:
        if not args.dir.is_dir():
            print(f"Error: {args.dir} is not a directory", file=sys.stderr)
            sys.exit(1)
        count = process_directory(args.dir, dry_run=args.dry_run)
        action = "Would convert" if args.dry_run else "Converted"
        print(f"\n{action} {count} file(s)")
    elif args.input:
        path = Path(args.input)
        if not path.is_file():
            print(f"Error: {path} is not a file", file=sys.stderr)
            sys.exit(1)
        changed = process_file(path, dry_run=args.dry_run)
        if changed:
            action = "Would convert" if args.dry_run else "Converted"
            print(f"{action}: {path}")
        else:
            print(f"No changes: {path}")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
