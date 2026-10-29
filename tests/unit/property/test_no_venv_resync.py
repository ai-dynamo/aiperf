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

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
UNIT_ROOT = PROJECT_ROOT / "tests" / "unit"

# ``uv`` subcommands that reconcile the project environment with the lockfile.
# ``uv pip install`` mutates the environment too, but it is not part of the
# project-sync path this invariant is about.
SYNCING_SUBCOMMANDS = frozenset({"run", "sync"})

# The only flag that stops the reconciliation. ``--frozen`` and ``--locked``
# constrain the lockfile, not the environment, so they still sync.
NO_SYNC_FLAG = "--no-sync"


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

    ``uv run`` stops parsing its own options at the command name, so a
    ``--no-sync`` placed after the command is passed through to the command and
    does not stop the reconciliation. Only the leading option run counts.
    """
    options: list[str] = []
    for token in head[2:]:
        if not token.startswith("-"):
            break
        options.append(token)
    return options


def _literal_bindings(tree: ast.Module) -> dict[str, ast.List | ast.Tuple]:
    """Names bound to a list or tuple literal anywhere in the module.

    Lets ``cmd = ["uv", "run", "aiperf"]`` followed by ``subprocess.run(cmd)`` be
    inspected. Last literal binding wins; a name later rebound to something that
    is not a literal keeps its earlier literal, which can only over-report.
    """
    bindings: dict[str, ast.List | ast.Tuple] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        if not isinstance(node.value, ast.List | ast.Tuple):
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                bindings[target.id] = node.value
    return bindings


def _argv_of(
    node: ast.Call, bindings: dict[str, ast.List | ast.Tuple]
) -> ast.List | ast.Tuple | None:
    """The argv literal of a call, from its first positional arg or ``args``.

    Deliberately callee-agnostic: any call handed a list headed ``uv run`` is
    worth flagging, whether it goes through ``subprocess.run``,
    ``subprocess.Popen`` or a wrapper.
    """
    candidates = list(node.args[:1])
    candidates += [kw.value for kw in node.keywords if kw.arg == "args"]
    for candidate in candidates:
        if isinstance(candidate, ast.List | ast.Tuple):
            return candidate
        if isinstance(candidate, ast.Name) and candidate.id in bindings:
            return bindings[candidate.id]
    return None


def _offending_lines(tree: ast.Module) -> list[int]:
    bindings = _literal_bindings(tree)
    lines: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        argv = _argv_of(node, bindings)
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


def test_offending_lines_resolves_a_command_variable() -> None:
    source = 'cmd = ["uv", "run", "aiperf"]\nsubprocess.run(cmd, cwd=ROOT)'
    assert _offending_lines(ast.parse(source)) == [2]


def test_offending_lines_reads_the_args_keyword() -> None:
    source = 'subprocess.run(args=["uv", "run", "aiperf"])'
    assert _offending_lines(ast.parse(source)) == [1]


def test_offending_lines_ignores_non_syncing_uv_subcommands() -> None:
    source = 'subprocess.run(["uv", "pip", "install", "x"])'
    assert _offending_lines(ast.parse(source)) == []
