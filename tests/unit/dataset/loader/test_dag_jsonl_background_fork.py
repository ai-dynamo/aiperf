# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``DagFork.background`` (fork-and-continue) loader semantics."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from aiperf.common.enums import ConversationBranchMode
from aiperf.dataset.loader.dag_jsonl import DagJsonlLoader, DagLoadError


def _write_jsonl(text: str) -> Path:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(text.strip())
        return Path(f.name)


def _conv(loader: DagJsonlLoader, sid: str):
    return next(c for c in loader.load() if c.session_id == sid)


def _branch(loader: DagJsonlLoader, sid: str, branch_id: str):
    conv = _conv(loader, sid)
    for b in conv.branches:
        if b.branch_id == branch_id:
            return b
    raise KeyError(f"branch_id {branch_id!r} not found on session {sid!r}")


class TestBareStringForkUnchanged:
    """Bare-string ``forks: ["c"]`` continues to mean foreground FORK
    (parent terminates after the fork). Existing fixtures must parse
    identically to pre-DagFork behavior."""

    def test_bare_string_emits_fg_fork_branch(self) -> None:
        path = _write_jsonl("""
{"session_id":"r","turns":[{"messages":[{"role":"user","content":"t0"}],"forks":["c"]}]}
{"session_id":"c","turns":[{"messages":[{"role":"user","content":"child"}]}]}
""")
        loader = DagJsonlLoader(filename=path)
        b = _branch(loader, "r", "r:0")
        assert b.mode == ConversationBranchMode.FORK
        assert b.background is False
        assert b.child_conversation_ids == ["c"]

    def test_bare_string_on_non_final_turn_rejected(self) -> None:
        """The historical fail-fast behavior: foreground FORK on a non-
        final turn is rejected (would terminate the parent before its
        next turn could fire)."""
        path = _write_jsonl("""
{"session_id":"r","turns":[{"messages":[{"role":"user","content":"t0"}],"forks":["c"]},{"messages":[{"role":"user","content":"t1"}]}]}
{"session_id":"c","turns":[{"messages":[{"role":"user","content":"child"}]}]}
""")
        with pytest.raises(DagLoadError, match="foreground FORK"):
            DagJsonlLoader(filename=path).load()


class TestObjectFormForkBackground:
    """Object-form ``forks: [{child, background: true}]`` opts into
    fork-and-continue. Parent runs its remaining turns while the child
    fans out with inherited context."""

    def test_object_form_bg_true_emits_bg_branch(self) -> None:
        path = _write_jsonl("""
{"session_id":"r","turns":[{"messages":[{"role":"user","content":"t0"}],"forks":[{"child":"c","background":true}]},{"messages":[{"role":"user","content":"t1"}]}]}
{"session_id":"c","turns":[{"messages":[{"role":"user","content":"child"}]}]}
""")
        loader = DagJsonlLoader(filename=path)
        b = _branch(loader, "r", "r:0")
        assert b.mode == ConversationBranchMode.FORK
        assert b.background is True
        assert b.child_conversation_ids == ["c"]

    def test_object_form_bg_false_equivalent_to_bare_string(self) -> None:
        """Explicit ``background: false`` should parse identically to
        bare-string and trip the same must-be-last-turn rule."""
        path = _write_jsonl("""
{"session_id":"r","turns":[{"messages":[{"role":"user","content":"t0"}],"forks":[{"child":"c","background":false}]},{"messages":[{"role":"user","content":"t1"}]}]}
{"session_id":"c","turns":[{"messages":[{"role":"user","content":"child"}]}]}
""")
        with pytest.raises(DagLoadError, match="foreground FORK"):
            DagJsonlLoader(filename=path).load()

    def test_multiple_bg_entries_collapse_to_one_branch(self) -> None:
        """All BG forks on a turn share scheduling semantics, so they
        collapse into a single ConversationBranchInfo with all children."""
        path = _write_jsonl("""
{"session_id":"r","turns":[{"messages":[{"role":"user","content":"t0"}],"forks":[{"child":"c1","background":true},{"child":"c2","background":true}]},{"messages":[{"role":"user","content":"t1"}]}]}
{"session_id":"c1","turns":[{"messages":[{"role":"user","content":"c1"}]}]}
{"session_id":"c2","turns":[{"messages":[{"role":"user","content":"c2"}]}]}
""")
        loader = DagJsonlLoader(filename=path)
        b = _branch(loader, "r", "r:0")
        assert b.background is True
        assert b.child_conversation_ids == ["c1", "c2"]

    def test_bg_forks_on_multiple_non_final_turns(self) -> None:
        """A parent may BG-fork on every turn; each turn gets its own
        per-turn branch_id (no collision with neighbors)."""
        path = _write_jsonl("""
{"session_id":"r","turns":[{"messages":[{"role":"user","content":"t0"}],"forks":[{"child":"c1","background":true}]},{"messages":[{"role":"user","content":"t1"}],"forks":[{"child":"c2","background":true}]},{"messages":[{"role":"user","content":"t2"}]}]}
{"session_id":"c1","turns":[{"messages":[{"role":"user","content":"c1"}]}]}
{"session_id":"c2","turns":[{"messages":[{"role":"user","content":"c2"}]}]}
""")
        loader = DagJsonlLoader(filename=path)
        b0 = _branch(loader, "r", "r:0")
        b1 = _branch(loader, "r", "r:1")
        assert b0.background is True and b0.child_conversation_ids == ["c1"]
        assert b1.background is True and b1.child_conversation_ids == ["c2"]


class TestMixedForegroundAndBackground:
    """When a turn has both FG and BG fork entries, the loader emits two
    ConversationBranchInfo entries with disambiguating ``:fork`` /
    ``:bg_fork`` suffixes (mirroring the ``:fork``/``:spawn`` suffixes
    used when fork+spawn coexist on a turn)."""

    def test_mixed_on_final_turn_emits_two_branches(self) -> None:
        path = _write_jsonl("""
{"session_id":"r","turns":[{"messages":[{"role":"user","content":"t0"}],"forks":["c1",{"child":"c2","background":true}]}]}
{"session_id":"c1","turns":[{"messages":[{"role":"user","content":"c1"}]}]}
{"session_id":"c2","turns":[{"messages":[{"role":"user","content":"c2"}]}]}
""")
        loader = DagJsonlLoader(filename=path)
        fg = _branch(loader, "r", "r:0:fork")
        bg = _branch(loader, "r", "r:0:bg_fork")
        assert fg.background is False and fg.child_conversation_ids == ["c1"]
        assert bg.background is True and bg.child_conversation_ids == ["c2"]

    def test_mixed_on_non_final_turn_rejects_fg_with_clear_error(self) -> None:
        """FG fork would terminate the parent before its next turn fires,
        but BG fork on the same turn implies the parent should continue.
        The pair is contradictory authoring; reject with a clear message."""
        path = _write_jsonl("""
{"session_id":"r","turns":[{"messages":[{"role":"user","content":"t0"}],"forks":["c1",{"child":"c2","background":true}]},{"messages":[{"role":"user","content":"t1"}]}]}
{"session_id":"c1","turns":[{"messages":[{"role":"user","content":"c1"}]}]}
{"session_id":"c2","turns":[{"messages":[{"role":"user","content":"c2"}]}]}
""")
        with pytest.raises(DagLoadError, match="foreground FORK"):
            DagJsonlLoader(filename=path).load()


class TestDuplicateChildAcrossForkBackgroundFlags:
    """Same child appearing in both FG and BG fork entries on the same
    turn is rejected — even though the branch_ids differ, the orchestrator
    would still dispatch the child twice with ambiguous parent-completion
    semantics."""

    def test_duplicate_across_fg_bg_rejected(self) -> None:
        path = _write_jsonl("""
{"session_id":"r","turns":[{"messages":[{"role":"user","content":"t0"}],"forks":["dup",{"child":"dup","background":true}]}]}
{"session_id":"dup","turns":[{"messages":[{"role":"user","content":"x"}]}]}
""")
        with pytest.raises(DagLoadError, match="duplicate"):
            DagJsonlLoader(filename=path).load()


class TestMultiParentForkRejectionAcrossBgFg:
    """A single child reached by both an FG fork (one parent) and a BG
    fork (different parent) violates the FORK single-parent constraint
    (context-inheritance ambiguity). The validator rejects regardless of
    the background flag combination."""

    def test_fg_and_bg_parents_of_same_child_rejected(self) -> None:
        path = _write_jsonl("""
{"session_id":"r1","turns":[{"messages":[{"role":"user","content":"t0"}],"forks":[{"child":"shared","background":true}]},{"messages":[{"role":"user","content":"t1"}]}]}
{"session_id":"r2","turns":[{"messages":[{"role":"user","content":"r2"}],"forks":["shared"]}]}
{"session_id":"shared","turns":[{"messages":[{"role":"user","content":"x"}]}]}
""")
        with pytest.raises(
            DagLoadError, match="FORK-mode children require a single parent"
        ):
            DagJsonlLoader(filename=path).load()
