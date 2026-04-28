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

"""LLM-ergonomics checks for AIPerf.

Fails the build on *new* violations of the 8 rules in
``artifacts/code-review-2026-04-21/llm-codebase-ergonomics.md``.

Existing violations are grandfathered via a baseline file at
``tools/ergonomics_baseline.json``. Regenerate the baseline with::

    python tools/check_ergonomics.py --regenerate-baseline

Checks (each can also be run in isolation with --only <check>):

    file-size           files under src/aiperf/ must be <= 500 lines
    function-size       functions must be <= 80 lines
    nesting-depth       control-flow nesting must be <= 5 levels
    keyword-only-args   functions with >=5 positional args must use ``*,``
    module-state        no module-level mutable dict/list/set assignments
    duplicate-classes   class names must be unique across src/aiperf/
    pydantic-fields     Pydantic models must have <=30 ``Field(...)`` decls
    stdlib-json         no ``import json`` / ``json.dumps`` / ``json.loads``

Usage:
    python tools/check_ergonomics.py                 # run all checks
    python tools/check_ergonomics.py --only file-size
    python tools/check_ergonomics.py --regenerate-baseline
    python tools/check_ergonomics.py file1.py file2.py   # check specific files

Thresholds live at the top of this file; adjust as the codebase cleans up.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src" / "aiperf"
BASELINE_PATH = Path(__file__).resolve().parent / "ergonomics_baseline.json"

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

MAX_FILE_LINES = 500
MAX_FUNCTION_LINES = 80
MAX_NESTING_DEPTH = 5
MAX_POSITIONAL_ARGS = 4  # >= 5 positional args without `*,` is an error
MAX_PYDANTIC_FIELDS = 30
MIN_EXCEPTION_MESSAGE_WORDS = 3  # R10: error messages must carry context

CHECKS = [
    "file-size",
    "function-size",
    "nesting-depth",
    "keyword-only-args",
    "module-state",
    "duplicate-classes",
    "pydantic-fields",
    "stdlib-json",
    "exception-message",
]


# ---------------------------------------------------------------------------
# Violation model + baseline
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Violation:
    check: str
    file: str  # relative to REPO_ROOT
    line: int
    identifier: str  # function/class name, or "" for file-level
    message: str

    def key(self) -> tuple[str, str, str]:
        """Baseline key: ``(check, file, identifier)``.

        Line numbers are deliberately excluded so that unrelated edits
        above a known violation do not re-trigger the check. The
        ``identifier`` for AST-backed function/class checks is the
        qualname (e.g. ``FixedTrialsStrategy.__init__``), so methods
        on different classes that legitimately share a short name (a
        Protocol/ABC contract, a dunder, repeated Dash inner-callback
        names) each get their own baseline entry instead of collapsing.

        For call-site checks (``stdlib-json``, ``exception-message``)
        the identifier is ``<enclosing-qualname>::<call-or-message>``,
        so the same call in different functions doesn't collapse. When
        the same call repeats inside a single function, ``_disambiguate``
        appends an ``#N`` occurrence suffix so each site is individually
        baselineable and a baselined site cannot mask a later sibling.
        """
        return (self.check, self.file, self.identifier)


def _disambiguate(violations: list[Violation]) -> list[Violation]:
    """Append ``#N`` to the 2nd+ occurrence of each (check, file, identifier).

    Safety net for the few checks where the qualname-scoped identifier
    can still legitimately repeat — e.g. two ``json.loads(`` calls in
    the same function (same ``qualname::json.loads(`` identifier), or
    two ``raise ValueError("bad")`` sites with the same literal prefix
    in the same function. Without ``#N``, those collapse to a single
    baseline key and a freshly added 3rd call would inherit the
    grandfathering of the existing 2 — defeating the "new violations
    only" guarantee. Sorting by ``(file, check, line, identifier)``
    keeps numbering stable across runs given a stable file.
    """
    counts: dict[tuple[str, str, str], int] = defaultdict(int)
    out: list[Violation] = []
    for v in sorted(violations, key=lambda x: (x.file, x.check, x.line, x.identifier)):
        base = (v.check, v.file, v.identifier)
        counts[base] += 1
        n = counts[base]
        if n == 1:
            out.append(v)
        else:
            out.append(
                Violation(
                    check=v.check,
                    file=v.file,
                    line=v.line,
                    identifier=f"{v.identifier}#{n}",
                    message=v.message,
                )
            )
    return out


def load_baseline() -> set[tuple[str, str, str]]:
    if not BASELINE_PATH.exists():
        return set()
    data = json.loads(BASELINE_PATH.read_text())
    return {tuple(entry) for entry in data.get("violations", [])}


def write_baseline(violations: list[Violation]) -> None:
    keys = sorted({v.key() for v in violations})
    BASELINE_PATH.write_text(
        json.dumps(
            {
                "_comment": (
                    "Pre-existing violations of tools/check_ergonomics.py. "
                    "Regenerate with: python tools/check_ergonomics.py "
                    "--regenerate-baseline. Key: (check, file, identifier). "
                    "Function/class identifiers are qualnames "
                    "(e.g. 'FixedTrialsStrategy.__init__'); call-site "
                    "identifiers (stdlib-json, exception-message) are "
                    "'<enclosing-qualname>::<call-or-message>'. A trailing "
                    "'#N' (N>=2) marks the Nth occurrence of an identifier "
                    "that genuinely repeats inside one scope (see "
                    "_disambiguate). New entries here should be rare and "
                    "justified; prefer fixing the underlying violation."
                ),
                "violations": [list(k) for k in keys],
            },
            indent=2,
        )
        + "\n"
    )


# ---------------------------------------------------------------------------
# AST utilities
# ---------------------------------------------------------------------------


def _parse(path: Path) -> ast.Module:
    """Parse a Python source file. Surface parse failures loudly.

    A swallowed SyntaxError or UnicodeDecodeError used to drop the file
    from every AST-backed check, which let the runner end with
    ``ergonomics: OK`` while a changed file silently went unchecked.
    Now we re-raise so the gate fails and the user sees the path.
    """
    try:
        return ast.parse(path.read_text())
    except (SyntaxError, UnicodeDecodeError) as exc:
        try:
            rel = path.relative_to(REPO_ROOT)
        except ValueError:
            rel = path
        raise RuntimeError(f"failed to parse {rel}: {exc}") from exc


_DefNode = ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef


def _qualname_walk(tree: ast.AST):
    """Yield ``(qualname, node)`` for every def in ``tree``.

    ``qualname`` is dot-joined from enclosing classes and functions —
    e.g. ``FixedTrialsStrategy.__init__`` for an ``__init__`` inside a
    class, or ``register_callbacks.create_custom_plot`` for an inner
    function. This is the only stable identifier across files where
    multiple classes legitimately share a method name (Protocol/ABC
    implementations, dunders, Dash callback patterns) — the previous
    short-name identifier collapsed all of those into one baseline key.
    """

    def walk(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, _DefNode):
                qn = f"{prefix}.{child.name}" if prefix else child.name
                yield qn, child
                yield from walk(child, qn)
            else:
                yield from walk(child, prefix)

    yield from walk(tree, "")


def _enclosing_func_map(tree: ast.Module) -> dict[int, str]:
    """Map ``id(node) -> enclosing-function-qualname`` for every AST node.

    Used by checks that key on a call site or raise site so the
    identifier can include the enclosing function scope (so two
    ``json.loads`` calls in different functions don't collapse to the
    same baseline key).
    """
    result: dict[int, str] = {}

    def walk(node, fn_qualname, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, _DefNode):
                qn = f"{prefix}.{child.name}" if prefix else child.name
                child_fn = (
                    qn
                    if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef)
                    else fn_qualname
                )
                result[id(child)] = fn_qualname or "<module>"
                walk(child, child_fn, qn)
            else:
                result[id(child)] = fn_qualname or "<module>"
                walk(child, fn_qualname, prefix)

    walk(tree, None, "")
    return result


def _max_depth(node: ast.AST, depth: int = 0) -> int:
    nesting_types = (
        ast.If,
        ast.For,
        ast.While,
        ast.Try,
        ast.With,
        ast.AsyncFor,
        ast.AsyncWith,
    )
    if isinstance(node, nesting_types):
        depth += 1
    best = depth
    for child in ast.iter_child_nodes(node):
        best = max(best, _max_depth(child, depth))
    return best


def _is_pydantic_model(cls: ast.ClassDef) -> bool:
    for base in cls.bases:
        name = ast.unparse(base)
        if any(
            tag in name
            for tag in (
                "BaseModel",
                "AIPerfBaseModel",
                "BaseConfig",
                "BaseSettings",
                "CamelModel",
            )
        ):
            return True
    return False


def _pydantic_field_count(cls: ast.ClassDef) -> int:
    count = 0
    for node in cls.body:
        if not isinstance(node, ast.AnnAssign):
            continue
        if not isinstance(node.target, ast.Name):
            continue
        val = node.value
        if val is None:
            # bare annotation — counts as a pydantic field
            count += 1
        elif isinstance(val, ast.Call):
            fname = (
                val.func.id
                if isinstance(val.func, ast.Name)
                else (val.func.attr if isinstance(val.func, ast.Attribute) else "")
            )
            if fname == "Field":
                count += 1
        else:
            count += 1  # literal default
    return count


# ---------------------------------------------------------------------------
# Per-file checks
# ---------------------------------------------------------------------------


def check_file_size(path: Path, rel: str) -> list[Violation]:
    lines = len(path.read_text().splitlines())
    if lines > MAX_FILE_LINES:
        return [
            Violation(
                check="file-size",
                file=rel,
                line=lines,
                identifier="",
                message=f"file has {lines} lines (>{MAX_FILE_LINES})",
            )
        ]
    return []


def check_function_size(tree: ast.Module, rel: str) -> list[Violation]:
    out: list[Violation] = []
    for qualname, node in _qualname_walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            end = node.end_lineno or node.lineno
            length = end - node.lineno + 1
            if length > MAX_FUNCTION_LINES:
                out.append(
                    Violation(
                        check="function-size",
                        file=rel,
                        line=node.lineno,
                        identifier=qualname,
                        message=f"function '{qualname}' is {length} lines (>{MAX_FUNCTION_LINES})",
                    )
                )
    return out


def check_nesting_depth(tree: ast.Module, rel: str) -> list[Violation]:
    out: list[Violation] = []
    for qualname, node in _qualname_walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            depth = _max_depth(node)
            if depth > MAX_NESTING_DEPTH:
                out.append(
                    Violation(
                        check="nesting-depth",
                        file=rel,
                        line=node.lineno,
                        identifier=qualname,
                        message=f"function '{qualname}' has nesting depth {depth} (>{MAX_NESTING_DEPTH})",
                    )
                )
    return out


def check_keyword_only_args(tree: ast.Module, rel: str) -> list[Violation]:
    out: list[Violation] = []
    for qualname, node in _qualname_walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        args = node.args
        pos = len(args.args) + len(args.posonlyargs)
        if args.args and args.args[0].arg in ("self", "cls"):
            pos -= 1
        kwonly = len(args.kwonlyargs)
        if pos > MAX_POSITIONAL_ARGS and kwonly == 0:
            out.append(
                Violation(
                    check="keyword-only-args",
                    file=rel,
                    line=node.lineno,
                    identifier=qualname,
                    message=f"function '{qualname}' has {pos} positional args without '*,' separator",
                )
            )
    return out


def check_module_state(tree: ast.Module, rel: str) -> list[Violation]:
    mutable = {
        "dict",
        "list",
        "set",
        "deque",
        "defaultdict",
        "Counter",
        "OrderedDict",
        "WeakSet",
        "WeakValueDictionary",
    }
    out: list[Violation] = []
    for node in tree.body:
        if not isinstance(node, ast.Assign | ast.AnnAssign):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for tgt in targets:
            if not isinstance(tgt, ast.Name):
                continue
            name = tgt.id
            if name.isupper() or name == "__all__":
                continue  # constants are OK
            val = node.value
            if val is None:
                continue
            kind: str | None = None
            if isinstance(val, ast.Dict | ast.List | ast.Set):
                kind = type(val).__name__
            elif isinstance(val, ast.Call):
                func = val.func
                fname = (
                    func.id
                    if isinstance(func, ast.Name)
                    else (func.attr if isinstance(func, ast.Attribute) else None)
                )
                if fname in mutable:
                    kind = f"{fname}()"
            if kind:
                out.append(
                    Violation(
                        check="module-state",
                        file=rel,
                        line=node.lineno,
                        identifier=name,
                        message=f"module-level mutable '{name}' ({kind}) — wrap in a class",
                    )
                )
    return out


def check_pydantic_fields(tree: ast.Module, rel: str) -> list[Violation]:
    out: list[Violation] = []
    for qualname, node in _qualname_walk(tree):
        if isinstance(node, ast.ClassDef) and _is_pydantic_model(node):
            n = _pydantic_field_count(node)
            if n > MAX_PYDANTIC_FIELDS:
                out.append(
                    Violation(
                        check="pydantic-fields",
                        file=rel,
                        line=node.lineno,
                        identifier=qualname,
                        message=f"model '{qualname}' has {n} fields (>{MAX_PYDANTIC_FIELDS}) — split into sub-models",
                    )
                )
    return out


_JSON_CALLS = {"dumps", "loads", "dump", "load"}


def check_stdlib_json(tree: ast.Module, rel: str) -> list[Violation]:
    """AST-walk replacement for the previous regex-based scan.

    The earlier version matched raw source text and so flagged ``json.loads``
    inside docstrings, comments, and string literals (false positives in
    docs and examples). Walking the AST inspects only executable nodes:
    ``import json``, ``from json import ...``, and attribute calls
    ``json.dumps(...)`` / ``json.loads(...)`` / etc.

    Identifier is namespaced by enclosing function (``<qualname>::call``)
    so a baselined ``json.loads`` in one function does not mask a new
    ``json.loads`` added to a sibling function in the same file.
    """
    enclosing = _enclosing_func_map(tree)
    out: list[Violation] = []
    for node in ast.walk(tree):
        scope = enclosing.get(id(node), "<module>")
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "json":
                    out.append(
                        Violation(
                            check="stdlib-json",
                            file=rel,
                            line=node.lineno,
                            identifier=f"{scope}::import json",
                            message="stdlib 'json' is banned; use 'orjson' instead",
                        )
                    )
        elif isinstance(node, ast.ImportFrom) and node.module == "json":
            names = ", ".join(alias.name for alias in node.names)
            out.append(
                Violation(
                    check="stdlib-json",
                    file=rel,
                    line=node.lineno,
                    identifier=f"{scope}::from json import {names}",
                    message="stdlib 'json' is banned; use 'orjson' instead",
                )
            )
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "json"
            and node.func.attr in _JSON_CALLS
        ):
            call = f"json.{node.func.attr}("
            out.append(
                Violation(
                    check="stdlib-json",
                    file=rel,
                    line=node.lineno,
                    identifier=f"{scope}::{call}",
                    message=f"stdlib '{call}' is banned; use orjson.dumps/orjson.loads",
                )
            )
    return out


def check_exception_message(tree: ast.Module, rel: str) -> list[Violation]:
    """R10 — raise sites must carry diagnostic context.

    Flags ``raise Cls("short")`` where the message is a plain string
    literal shorter than ``MIN_EXCEPTION_MESSAGE_WORDS`` words. An f-string
    or concatenation is assumed to carry dynamic context and is accepted
    regardless of literal length.

    Identifier is namespaced by enclosing function so a baselined raise
    in one function does not mask a new short-message raise added to
    another function in the same file.
    """
    enclosing = _enclosing_func_map(tree)
    out: list[Violation] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Raise):
            continue
        if not isinstance(node.exc, ast.Call) or not node.exc.args:
            continue
        first = node.exc.args[0]
        if not isinstance(first, ast.Constant) or not isinstance(first.value, str):
            continue  # f-string, JoinedStr, Name, etc. — dynamic, assumed OK
        words = first.value.split()
        if len(words) >= MIN_EXCEPTION_MESSAGE_WORDS:
            continue
        exc_name = (
            node.exc.func.id
            if isinstance(node.exc.func, ast.Name)
            else (
                node.exc.func.attr if isinstance(node.exc.func, ast.Attribute) else "?"
            )
        )
        scope = enclosing.get(id(node), "<module>")
        out.append(
            Violation(
                check="exception-message",
                file=rel,
                line=node.lineno,
                identifier=f"{scope}::{exc_name}:{first.value[:40]}",
                message=(
                    f"raise {exc_name}({first.value!r}) — message too terse "
                    f"(<{MIN_EXCEPTION_MESSAGE_WORDS} words); include the "
                    f"operation, the input, and a likely cause"
                ),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Tree-wide checks (need all files together)
# ---------------------------------------------------------------------------


def check_duplicate_classes(
    trees: dict[Path, ast.Module], rels: dict[Path, str]
) -> list[Violation]:
    """Flag class names defined in more than one module under src/aiperf/."""
    # Allowlist: framework entry points / duplicate `main`-style is fine.
    allowlist = set()
    locations: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for path, tree in trees.items():
        rel = rels[path]
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                locations[node.name].append((rel, node.lineno))

    out: list[Violation] = []
    for name, locs in locations.items():
        if name in allowlist or len(locs) < 2:
            continue
        # report each duplicate site; baseline keys on (check, file, name)
        locs_str = ", ".join(f"{f}:{ln}" for f, ln in locs)
        for rel, lineno in locs:
            out.append(
                Violation(
                    check="duplicate-classes",
                    file=rel,
                    line=lineno,
                    identifier=name,
                    message=f"class '{name}' duplicated across: {locs_str}",
                )
            )
    return out


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _iter_py_files(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        if p.is_file() and p.suffix == ".py":
            out.append(p)
        elif p.is_dir():
            out.extend(sorted(p.rglob("*.py")))
    return [f for f in out if "__pycache__" not in f.parts]


def _run_per_file(
    path: Path, rel: str, enabled: set[str]
) -> tuple[list[Violation], ast.Module]:
    out: list[Violation] = []
    if "file-size" in enabled:
        out.extend(check_file_size(path, rel))
    tree = _parse(path)
    if "stdlib-json" in enabled:
        out.extend(check_stdlib_json(tree, rel))
    if "function-size" in enabled:
        out.extend(check_function_size(tree, rel))
    if "nesting-depth" in enabled:
        out.extend(check_nesting_depth(tree, rel))
    if "keyword-only-args" in enabled:
        out.extend(check_keyword_only_args(tree, rel))
    if "module-state" in enabled:
        out.extend(check_module_state(tree, rel))
    if "pydantic-fields" in enabled:
        out.extend(check_pydantic_fields(tree, rel))
    if "exception-message" in enabled:
        out.extend(check_exception_message(tree, rel))
    return out, tree


def collect_violations(files: list[Path], enabled: set[str]) -> list[Violation]:
    violations: list[Violation] = []
    trees: dict[Path, ast.Module] = {}
    rels: dict[Path, str] = {}
    for path in files:
        try:
            rel = str(path.relative_to(REPO_ROOT))
        except ValueError:
            rel = str(path)
        rels[path] = rel
        per_file, tree = _run_per_file(path, rel, enabled)
        violations.extend(per_file)
        if "duplicate-classes" in enabled:
            trees[path] = tree
    if "duplicate-classes" in enabled and trees:
        violations.extend(check_duplicate_classes(trees, rels))
    return _disambiguate(violations)


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM-ergonomics checks (AIPerf)")
    parser.add_argument(
        "files",
        nargs="*",
        help="Files to check (defaults to src/aiperf/). Accepts dirs too.",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        choices=CHECKS,
        help="Only run the named check(s). Repeatable.",
    )
    parser.add_argument(
        "--regenerate-baseline",
        action="store_true",
        help="Overwrite tools/ergonomics_baseline.json with current violations.",
    )
    parser.add_argument(
        "--show-baselined",
        action="store_true",
        help="Also print baselined (grandfathered) violations.",
    )
    args = parser.parse_args()

    enabled = set(args.only) if args.only else set(CHECKS)

    # Resolve input files. For duplicate-classes we need the whole tree;
    # warn if user passed a subset.
    if args.files:
        paths = [Path(p).resolve() for p in args.files]
        files = _iter_py_files(paths)
        if "duplicate-classes" in enabled and not args.regenerate_baseline:
            # duplicate check is only meaningful tree-wide; scope it down
            enabled = enabled - {"duplicate-classes"}
    else:
        files = _iter_py_files([SRC_ROOT])

    if not files:
        print("no python files to check", file=sys.stderr)
        return 0

    violations = collect_violations(files, enabled)

    if args.regenerate_baseline:
        # baseline always captures the full tree
        all_files = _iter_py_files([SRC_ROOT])
        all_violations = collect_violations(all_files, set(CHECKS))
        write_baseline(all_violations)
        print(
            f"wrote {len(all_violations)} violations "
            f"across {len({v.key() for v in all_violations})} keys to "
            f"{BASELINE_PATH.relative_to(REPO_ROOT)}"
        )
        return 0

    baseline = load_baseline()
    new = [v for v in violations if v.key() not in baseline]
    baselined = [v for v in violations if v.key() in baseline]

    if args.show_baselined and baselined:
        print(f"--- {len(baselined)} baselined (grandfathered) violations ---")
        for v in baselined:
            print(f"  [{v.check}] {v.file}:{v.line}  {v.message}")

    if not new:
        print(
            f"ergonomics: OK ({len(violations)} total, "
            f"{len(baselined)} baselined, 0 new)"
        )
        return 0

    print(f"ergonomics: {len(new)} NEW violation(s)", file=sys.stderr)
    by_check: dict[str, list[Violation]] = defaultdict(list)
    for v in new:
        by_check[v.check].append(v)
    for check in CHECKS:
        items = by_check.get(check, [])
        if not items:
            continue
        print(f"\n[{check}] ({len(items)})", file=sys.stderr)
        for v in items:
            print(f"  {v.file}:{v.line}  {v.message}", file=sys.stderr)
    print(
        "\nIf a violation is unavoidable, fix it at the site. "
        "To accept a new baselined entry (rare), run:\n"
        "  python tools/check_ergonomics.py --regenerate-baseline",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
