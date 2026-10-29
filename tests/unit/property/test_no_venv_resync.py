# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mechanical "global invariant" test: unit tests never re-sync the venv they run in.

``uv run`` and ``uv sync`` reconcile the project ``.venv`` with ``uv.lock``
before doing anything else, so a test that shells out to either rewrites the
dependencies of the very process that spawned it. When the venv has drifted
from the lock, modules already imported stay at the old version while freshly
imported ones come from the new one, and the suite fails in ways that depend on
import order rather than on the code under test.

The 2026-10-28 nightly hit exactly that: two ``aiperf kube`` CLI tests ran
``uv run aiperf`` from the repo root, which downgraded transformers 5.21.x to
5.15.1 mid-session, and 14 tokenizer tests then died on ``ImportError: cannot
import name 'GGUF_TOKENIZER_MAPPING' from 'transformers.integrations'`` —
because the in-memory 5.21.x ``transformers.integrations`` lazy module does not
export a name the 5.15.1 module on disk asks for.

Subprocess CLI tests invoke the CLI as ``sys.executable -m aiperf`` instead.
"""

from __future__ import annotations

import ast
import pathlib
from dataclasses import dataclass

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
UNIT_ROOT = PROJECT_ROOT / "tests" / "unit"

# ``uv`` subcommands that reconcile the project environment with the lockfile.
# ``uv pip install`` mutates the environment too, but it is not part of the
# project-sync path this invariant is about.
SYNCING_SUBCOMMANDS = frozenset({"run", "sync"})

# The only flag that stops the reconciliation. ``--frozen`` and ``--locked``
# constrain the lockfile, not the environment, so they still sync.
NO_SYNC_FLAG = "--no-sync"

# ``uv run`` options that consume the next token as their value, so the command
# boundary sits one token further along. Anything not listed is assumed
# valueless, which errs toward flagging: a missed ``--no-sync`` is the bug this
# guard exists to catch, whereas a false positive is visible and resolved by
# reordering the options.
UV_RUN_VALUE_OPTIONS = frozenset(
    {
        "--allow-insecure-host",
        "--cache-dir",
        "--color",
        "--config-file",
        "--constraint",
        "--directory",
        "--env-file",
        "--exclude-newer",
        "--extra-index-url",
        "--find-links",
        "--fork-strategy",
        "--group",
        "--index",
        "--index-url",
        "--keyring-provider",
        "--link-mode",
        "--module",
        "--no-binary",
        "--only-binary",
        "--override",
        "--package",
        "--prerelease",
        "--project",
        "--python",
        "--python-platform",
        "--python-preference",
        "--resolution",
        "--script",
        "--with",
        "--with-requirements",
    }
)


def _leading_literals(sequence: ast.List | ast.Tuple) -> list[str]:
    """The run of string constants at the head of a sequence literal.

    Stops at the first non-constant element so an argv such as
    ``["uv", "run", *extra]`` still yields ``["uv", "run"]``.
    """
    literals: list[str] = []
    for element in sequence.elts:
        if not (isinstance(element, ast.Constant) and isinstance(element.value, str)):
            break
        literals.append(element.value)
    return literals


def _uv_option_region(head: list[str]) -> list[str]:
    """The run of uv's own options immediately after the subcommand.

    Walks past each option's value where the option is known to take one, so
    ``--python 3.12`` does not end the region and hide a later ``--no-sync``.
    ``uv run`` stops parsing its own options at the command name, so a
    ``--no-sync`` placed after the command is passed through to the command and
    does not stop the reconciliation.
    """
    options: list[str] = []
    index = 2
    while index < len(head):
        token = head[index]
        if not token.startswith("-"):
            break
        options.append(token)
        index += 2 if token in UV_RUN_VALUE_OPTIONS else 1
    return options


@dataclass(frozen=True)
class _Binding:
    """A name bound to a list or tuple literal in one lexical scope."""

    scope: ast.AST
    name: str
    line: int
    value: ast.List | ast.Tuple


def _parent_map(tree: ast.Module) -> dict[int, ast.AST]:
    return {
        id(child): parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }


def _scope_chain(node: ast.AST, parents: dict[int, ast.AST]) -> list[ast.AST]:
    """Enclosing function scopes innermost first, ending at the module."""
    chain: list[ast.AST] = []
    current = parents.get(id(node))
    while current is not None:
        if isinstance(current, ast.FunctionDef | ast.AsyncFunctionDef | ast.Module):
            chain.append(current)
        current = parents.get(id(current))
    return chain


def _literal_bindings(tree: ast.Module, parents: dict[int, ast.AST]) -> list[_Binding]:
    """List and tuple literals bound to a name, tagged with their scope.

    Lets ``cmd = ["uv", "run", "aiperf"]`` followed by ``subprocess.run(cmd)`` be
    inspected without letting two functions that both use ``cmd`` classify each
    other's calls.
    """
    bindings: list[_Binding] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign):
            targets, value = [node.target], node.value
        else:
            continue
        if not isinstance(value, ast.List | ast.Tuple):
            continue
        scope = _scope_chain(node, parents)[0]
        bindings += [
            _Binding(scope, target.id, node.lineno, value)
            for target in targets
            if isinstance(target, ast.Name)
        ]
    return bindings


def _resolve_name(
    name: str, line: int, scopes: list[ast.AST], bindings: list[_Binding]
) -> ast.List | ast.Tuple | None:
    """The literal bound to *name* before *line*, innermost scope first."""
    for scope in scopes:
        candidates = [
            binding
            for binding in bindings
            if binding.scope is scope and binding.name == name and binding.line < line
        ]
        if candidates:
            return max(candidates, key=lambda binding: binding.line).value
    return None


def _argv_of(
    node: ast.Call, bindings: list[_Binding], parents: dict[int, ast.AST]
) -> ast.List | ast.Tuple | None:
    """The argv literal of a call, from its first positional arg or ``args``.

    Deliberately callee-agnostic: any call handed a list headed ``uv run`` is
    worth flagging, whether it goes through ``subprocess.run``,
    ``subprocess.Popen`` or a wrapper.
    """
    candidates = list(node.args[:1])
    candidates += [kw.value for kw in node.keywords if kw.arg == "args"]
    scopes = _scope_chain(node, parents)
    for candidate in candidates:
        if isinstance(candidate, ast.List | ast.Tuple):
            return candidate
        if isinstance(candidate, ast.Name):
            resolved = _resolve_name(candidate.id, node.lineno, scopes, bindings)
            if resolved is not None:
                return resolved
    return None


def _offending_lines(tree: ast.Module) -> list[int]:
    parents = _parent_map(tree)
    bindings = _literal_bindings(tree, parents)
    lines: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        argv = _argv_of(node, bindings, parents)
        if argv is None:
            continue
        head = _leading_literals(argv)
        if len(head) < 2 or head[0] != "uv" or head[1] not in SYNCING_SUBCOMMANDS:
            continue
        if NO_SYNC_FLAG in _uv_option_region(head):
            continue
        lines.append(node.lineno)
    return lines


def _syncing_subprocess_calls() -> list[str]:
    offenders: list[str] = []
    for path in sorted(UNIT_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        location = path.relative_to(PROJECT_ROOT)
        offenders += [f"{location}:{line}" for line in _offending_lines(tree)]
    return offenders


def test_unit_tests_do_not_resync_the_venv() -> None:
    offenders = _syncing_subprocess_calls()
    assert not offenders, (
        "These tests shell out to `uv run` or `uv sync`, which reconciles the "
        "project .venv with uv.lock before executing and so rewrites the "
        "dependencies of the running suite. Invoke the CLI as "
        "`sys.executable -m aiperf`, or pass `--no-sync`:\n  " + "\n  ".join(offenders)
    )


def test_offending_lines_flags_a_syncing_subprocess_call() -> None:
    source = 'subprocess.run(["uv", "run", "aiperf", "kube"], cwd=ROOT)'
    assert _offending_lines(ast.parse(source)) == [1]


def test_offending_lines_rejects_a_flag_owned_by_the_child() -> None:
    """A trailing ``--no-sync`` reaches the child, so uv still reconciles."""
    misplaced = 'subprocess.run(["uv", "run", "aiperf", "--no-sync"])'
    assert _offending_lines(ast.parse(misplaced)) == [1]


def test_offending_lines_accepts_the_no_sync_option() -> None:
    guarded = 'subprocess.run(["uv", "run", "--no-sync", "aiperf"])'
    assert _offending_lines(ast.parse(guarded)) == []


def test_offending_lines_accepts_no_sync_after_an_option_value() -> None:
    for source in (
        'subprocess.run(["uv", "run", "--python", "3.12", "--no-sync", "aiperf"])',
        'subprocess.run(["uv", "run", "--project", "path", "--no-sync", "aiperf"])',
    ):
        assert _offending_lines(ast.parse(source)) == []


def test_offending_lines_resolves_a_command_variable() -> None:
    source = 'cmd = ["uv", "run", "aiperf"]\nsubprocess.run(cmd, cwd=ROOT)'
    assert _offending_lines(ast.parse(source)) == [2]


def test_offending_lines_resolves_a_name_per_scope() -> None:
    source = (
        "def safe():\n"
        '    cmd = ["aiperf", "kube"]\n'
        "    subprocess.run(cmd)\n"
        "\n"
        "def unsafe():\n"
        '    cmd = ["uv", "run", "aiperf"]\n'
        "    subprocess.run(cmd)\n"
    )
    assert _offending_lines(ast.parse(source)) == [7]


def test_offending_lines_reads_the_args_keyword() -> None:
    source = 'subprocess.run(args=["uv", "run", "aiperf"])'
    assert _offending_lines(ast.parse(source)) == [1]


def test_offending_lines_ignores_non_syncing_uv_subcommands() -> None:
    source = 'subprocess.run(["uv", "pip", "install", "x"])'
    assert _offending_lines(ast.parse(source)) == []
