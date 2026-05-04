#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One-off helper: walk *.py files, find triple-quoted YAML literals,
run them through tools.migrate_config_yaml.migrate_yaml_text, splice back.

Heuristic for detecting a YAML literal: any triple-quoted string assignment
where the contents contain a top-level body key (`models:`, `endpoint:`,
etc.) at column 0 of the dedented contents. Conservative — false negatives
preferred over false positives.

Run only after Plan A's models/loader/sweep tasks land, before bulk
fixture migration. Discarded after Task 17.
"""

from __future__ import annotations

import re
import sys
import textwrap
from pathlib import Path

from tools.migrate_config_yaml import BODY_KEYS, migrate_yaml_text


_TRIPLE_QUOTE_RE = re.compile(
    r'(?P<indent>[ \t]*)(?P<var>\w+)\s*=\s*(?P<quote>"""|\'\'\')(?P<body>.*?)(?P=quote)',
    re.DOTALL,
)


def _looks_like_yaml(body: str) -> bool:
    """Cheap detector: does the dedented body have a top-level body key?"""
    dedented = textwrap.dedent(body).lstrip("\n")
    for key in BODY_KEYS:
        if re.search(rf"(?m)^{key}\s*:", dedented):
            return True
    return False


def _migrate_literal(match: re.Match[str]) -> str:
    body = match.group("body")
    if not _looks_like_yaml(body):
        return match.group(0)
    dedented = textwrap.dedent(body).lstrip("\n")
    try:
        migrated = migrate_yaml_text(dedented)
    except Exception as exc:  # noqa: BLE001 - tolerate non-YAML literals
        # If body looked like YAML by heuristic but doesn't parse, leave it alone.
        print(f"  skipped (non-YAML or parse error): {exc}", file=sys.stderr)
        return match.group(0)
    indent = match.group("indent") + "    "
    indented = textwrap.indent(migrated.rstrip("\n"), indent)
    return (
        f'{match.group("indent")}{match.group("var")} = '
        f'{match.group("quote")}\n{indented}\n'
        f'{match.group("indent")}{match.group("quote")}'
    )


def migrate_python_file(path: Path) -> bool:
    """Returns True if the file was modified."""
    original = path.read_text(encoding="utf-8")
    new_text = _TRIPLE_QUOTE_RE.sub(_migrate_literal, original)
    if new_text != original:
        path.write_text(new_text, encoding="utf-8")
        return True
    return False


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    paths = [Path(a) for a in args]
    changed = 0
    for path in paths:
        if path.is_file() and path.suffix == ".py":
            if migrate_python_file(path):
                changed += 1
                print(f"migrated: {path}")
        elif path.is_dir():
            for py in path.rglob("*.py"):
                if migrate_python_file(py):
                    changed += 1
                    print(f"migrated: {py}")
    print(f"{changed} files changed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
