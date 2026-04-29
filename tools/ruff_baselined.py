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

"""Run ruff with an out-of-band baseline for selected rules.

Enforces ``PLR0915``, ``PLR0912``, ``C901``, ``TID251``, ``BLE001``,
``S110``, ``S112``, ``ANN201``, and ``D103`` (nine rules — see ``RULES``)
**without** polluting the source tree with ``# noqa`` comments and
**without** grandfathering entire files. Each violation is matched
against ``tools/ruff_baseline.json`` using a stable key:

    (rule, file, identifier)

where identifier comes from ``_resolve_identifier()``:

* For function-scope rules (PLR0915 / PLR0912 / C901 / BLE001 / S110 /
  S112 / ANN201 / D103), the **enclosing function qualname** resolved
  from the file's AST (e.g. ``MyClass.do_thing`` for a method, or just
  ``my_func`` for a module-level function); ``None`` if the violation
  is outside any function.
* For TID251, ``<enclosing-function>::<banned call expression>`` (e.g.
  ``render::json.dumps``), with ``<module>`` substituted when the call
  is at module scope. Namespacing the banned call by its enclosing
  function prevents one grandfathered ``json.dumps`` site from masking
  a brand-new ``json.dumps`` added in a different function in the same
  file.

Because the key excludes line numbers, unrelated edits above a
grandfathered site don't re-trigger the check. Because the key is
per-identifier (not per-file), a new offending function added to a
grandfathered file *does* fire the rule.

Usage:
    python tools/ruff_baselined.py               # check src/aiperf/
    python tools/ruff_baselined.py <files...>    # check specific files
    python tools/ruff_baselined.py --regenerate-baseline
    python tools/ruff_baselined.py --show-baselined

Exit codes:
    0 — no new violations
    1 — one or more new violations
    2 — ruff or tool failure
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import orjson

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_PATH = Path(__file__).resolve().parent / "ruff_baseline.json"
SRC_ROOT = REPO_ROOT / "src" / "aiperf"

# Rules enforced via this wrapper. They are intentionally NOT in
# [tool.ruff.lint] select in pyproject.toml — the wrapper is the gate.
#
# Mapping to the rules in llm-codebase-ergonomics.md /
# llm-codebase-ergonomics-extended.md:
#   PLR0915, PLR0912, C901  — Rule 2 (function size) / nesting-depth
#   TID251                  — Rule 8 (stdlib-json ban via banned-api)
#   BLE001, S110, S112      — R14 (loud failures, no silent swallows)
#   ANN201                  — R11 (types as documentation, narrow: public
#                             function return types only)
#   D103                    — R13 (docstrings on public functions)
RULES = [
    "PLR0915",
    "PLR0912",
    "C901",
    "TID251",
    "BLE001",
    "S110",
    "S112",
    "ANN201",
    "D103",
]

_TID251_BANNED_RE = re.compile(r"`([^`]+)`\s+is banned")


@dataclass(frozen=True)
class Violation:
    rule: str
    path: str  # relative to REPO_ROOT
    line: int
    col: int
    message: str

    def identifier(self) -> str | None:
        """Return the stable identifier for baselining.

        Resolved once per violation using the surrounding source. May
        return ``None`` if the file can't be parsed (in which case the
        violation is not baselineable).
        """
        return _resolve_identifier(self)


def _resolve_identifier(v: Violation) -> str | None:
    if v.rule == "TID251":
        m = _TID251_BANNED_RE.search(v.message)
        if not m:
            return None
        call = m.group(1)
        # Distinguish sites within a file by enclosing function so that a
        # new ``json.dumps`` in a brand-new function isn't hidden by an
        # existing grandfathered one in another function.
        fn = _enclosing_function(v.path, v.line) or "<module>"
        return f"{fn}::{call}"
    # Function-scope rules — identifier is the enclosing function.
    return _enclosing_function(v.path, v.line)


_function_cache: dict[tuple[str, int], str | None] = {}
_tree_cache: dict[str, ast.Module | None] = {}


def _parse(rel: str) -> ast.Module | None:
    if rel not in _tree_cache:
        try:
            _tree_cache[rel] = ast.parse((REPO_ROOT / rel).read_text())
        except (SyntaxError, UnicodeDecodeError, FileNotFoundError):
            _tree_cache[rel] = None
    return _tree_cache[rel]


def _enclosing_function(rel: str, line: int) -> str | None:
    key = (rel, line)
    if key in _function_cache:
        return _function_cache[key]
    tree = _parse(rel)
    if tree is None:
        _function_cache[key] = None
        return None
    best: tuple[int, str] | None = None  # (start_line, name)
    stack: list[tuple[ast.AST, list[str]]] = [(tree, [])]
    while stack:
        node, path_names = stack.pop()
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                stack.append((child, [*path_names, child.name]))
            elif isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                new_path = [*path_names, child.name]
                end = child.end_lineno or child.lineno
                if child.lineno <= line <= end:
                    qualname = ".".join(new_path)
                    if best is None or child.lineno > best[0]:
                        best = (child.lineno, qualname)
                # Recurse into nested defs.
                stack.append((child, new_path))
            else:
                stack.append((child, path_names))
    result = best[1] if best else None
    _function_cache[key] = result
    return result


# ---------------------------------------------------------------------------
# Baseline IO
# ---------------------------------------------------------------------------


def load_baseline() -> set[tuple[str, str, str]]:
    if not BASELINE_PATH.exists():
        return set()
    data = orjson.loads(BASELINE_PATH.read_bytes())
    return {tuple(entry) for entry in data.get("violations", [])}


def write_baseline(keys: set[tuple[str, str, str]]) -> None:
    BASELINE_PATH.write_bytes(
        orjson.dumps(
            {
                "_comment": (
                    "Out-of-band ruff baseline for tools/ruff_baselined.py. "
                    "Key: (rule, file, identifier). Regenerate with: "
                    "python tools/ruff_baselined.py --regenerate-baseline. "
                    "Entries here are grandfathered violations — prefer "
                    "fixing the underlying code over extending the list."
                ),
                "rules": RULES,
                "violations": [list(k) for k in sorted(keys)],
            },
            option=orjson.OPT_INDENT_2,
        )
        + b"\n"
    )


# ---------------------------------------------------------------------------
# Ruff invocation
# ---------------------------------------------------------------------------


def run_ruff(paths: list[str]) -> list[Violation]:
    cmd = [
        sys.executable,
        "-m",
        "ruff",
        "check",
        "--select",
        ",".join(RULES),
        "--output-format",
        "json",
        "--no-cache",
        *paths,
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=False)
    if result.returncode not in (0, 1):
        sys.stderr.buffer.write(result.stderr)
        raise RuntimeError(
            f"ruff exited with code {result.returncode} (expected 0 or 1)"
        )
    if not result.stdout.strip():
        return []
    raw = orjson.loads(result.stdout)
    violations: list[Violation] = []
    for entry in raw:
        path = entry["filename"]
        abs_path = (REPO_ROOT / path).resolve()
        try:
            rel = str(abs_path.relative_to(REPO_ROOT))
        except ValueError:
            rel = path
        loc = entry["location"]
        violations.append(
            Violation(
                rule=entry["code"],
                path=rel,
                line=int(loc["row"]),
                col=int(loc["column"]),
                message=entry["message"],
            )
        )
    return violations


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _default_paths() -> list[str]:
    return [str(SRC_ROOT.relative_to(REPO_ROOT))]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*")
    parser.add_argument(
        "--regenerate-baseline",
        action="store_true",
        help="Overwrite tools/ruff_baseline.json with current violations.",
    )
    parser.add_argument(
        "--show-baselined",
        action="store_true",
        help="Also print violations hidden by the baseline.",
    )
    args = parser.parse_args()

    paths = args.files or _default_paths()

    try:
        violations = run_ruff(paths)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # Build keys. Violations without a resolvable identifier cannot be
    # baselined — always surface them.
    resolved: list[tuple[Violation, tuple[str, str, str] | None]] = []
    for v in violations:
        ident = v.identifier()
        key = (v.rule, v.path, ident) if ident else None
        resolved.append((v, key))

    if args.regenerate_baseline:
        # Always regenerate from the full tree for consistency.
        if args.files:
            full_violations = run_ruff(_default_paths())
            resolved = [
                (v, (v.rule, v.path, v.identifier()) if v.identifier() else None)
                for v in full_violations
            ]
        keys = {k for _, k in resolved if k is not None}
        write_baseline(keys)
        print(
            f"wrote {len(keys)} baseline entries "
            f"to {BASELINE_PATH.relative_to(REPO_ROOT)}"
        )
        unresolved = [v for v, k in resolved if k is None]
        if unresolved:
            print(
                f"warning: {len(unresolved)} violation(s) could not be "
                f"baselined (no stable identifier):",
                file=sys.stderr,
            )
            for v in unresolved:
                print(
                    f"  {v.path}:{v.line}:{v.col} {v.rule} {v.message}",
                    file=sys.stderr,
                )
        return 0

    baseline = load_baseline()
    new: list[Violation] = []
    baselined: list[Violation] = []
    for v, key in resolved:
        if key is not None and key in baseline:
            baselined.append(v)
        else:
            new.append(v)

    if args.show_baselined:
        print(f"--- {len(baselined)} baselined violation(s) ---")
        for v in baselined:
            print(f"  [{v.rule}] {v.path}:{v.line}:{v.col}  {v.message}")

    if not new:
        print(
            f"ruff-baselined: OK ({len(violations)} total, "
            f"{len(baselined)} baselined, 0 new)"
        )
        return 0

    print(f"ruff-baselined: {len(new)} NEW violation(s)", file=sys.stderr)
    for v in new:
        ident = v.identifier() or "?"
        print(
            f"  [{v.rule}] {v.path}:{v.line}:{v.col}  in {ident}  — {v.message}",
            file=sys.stderr,
        )
    print(
        "\nIf the violation is unavoidable (rare), regenerate the baseline:\n"
        "  python tools/ruff_baselined.py --regenerate-baseline",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
