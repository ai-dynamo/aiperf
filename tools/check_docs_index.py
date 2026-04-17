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
Check that every markdown file in docs/ is listed in docs/index.yml.

Fails with a non-zero exit code if any file is missing, so this can be
used as a CI gate to prevent docs from being silently excluded from the
Fern documentation site.

Files listed in ALLOWLIST are intentionally excluded and will not cause
a failure.

Usage:
    python tools/check_docs_index.py
    python tools/check_docs_index.py --docs-dir docs --index docs/index.yml
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Files intentionally excluded from the index (relative to docs/)
ALLOWLIST = {
    "accuracy/accuracy_stubs.md",
}


def get_indexed_paths(index_path: Path) -> set[str]:
    """Extract all path: values from index.yml."""
    content = index_path.read_text(encoding="utf-8")
    return set(re.findall(r"^\s*path:\s*(.+)$", content, re.MULTILINE))


def get_all_docs(docs_dir: Path) -> set[str]:
    """Return all .md files in docs/ relative to docs_dir, excluding index.md."""
    return {
        str(f.relative_to(docs_dir))
        for f in docs_dir.rglob("*.md")
        if f.name != "index.md"
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check docs/index.yml is complete")
    parser.add_argument(
        "--docs-dir", type=Path, default=Path("docs"), help="Path to docs directory"
    )
    parser.add_argument(
        "--index", type=Path, default=Path("docs/index.yml"), help="Path to index.yml"
    )
    args = parser.parse_args()

    if not args.docs_dir.is_dir():
        print(f"Error: docs directory not found: {args.docs_dir}", file=sys.stderr)
        sys.exit(1)

    if not args.index.is_file():
        print(f"Error: index file not found: {args.index}", file=sys.stderr)
        sys.exit(1)

    all_docs = get_all_docs(args.docs_dir)
    indexed = get_indexed_paths(args.index)
    missing = sorted(all_docs - indexed - ALLOWLIST)

    if missing:
        print(
            f"ERROR: {len(missing)} file(s) in docs/ are not listed in {args.index}:\n"
        )
        for f in missing:
            print(f"  - {f}")
        print(
            f"\nAdd the missing file(s) to {args.index} or to the ALLOWLIST in this script."
        )
        sys.exit(1)

    print(
        f"OK: all {len(all_docs) - len(ALLOWLIST)} docs files are listed in {args.index}"
    )


if __name__ == "__main__":
    main()
