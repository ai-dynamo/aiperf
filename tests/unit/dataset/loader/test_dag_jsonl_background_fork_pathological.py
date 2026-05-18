# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial / pathological test cases for ``DagFork.background`` to
ensure the loader fails fast and predictably on every authoring foot-gun
we can think of, and accepts the legitimate edge cases (BG on final turn,
nested BG, BG with mixed siblings).

If a case here regresses, the loader's strict-mode posture is broken —
not just a typo bug. These are the cases where authors will typically
hand-author DAG fixtures and where Pydantic's ``extra="forbid"`` and the
loader's structural validators are the only thing standing between a
typo and a silently-different benchmark."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from aiperf.common.enums import ConversationBranchMode
from aiperf.dataset.loader.dag_jsonl import DagJsonlLoader, DagLoadError
from aiperf.dataset.loader.dag_jsonl_models import DagFork


def _write_jsonl(text: str) -> Path:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(text.strip())
        return Path(f.name)


def _load(text: str):
    return DagJsonlLoader(filename=_write_jsonl(text)).load()


def _branches(convs, sid: str):
    return next(c.branches for c in convs if c.session_id == sid)


# ============================================================================
# Pydantic strict-mode (DagFork field validation)
# ============================================================================


class TestDagForkPydanticStrictMode:
    """The loader leans on Pydantic ``extra="forbid"`` to catch typo'd field
    names before they silently default. These tests pin that posture."""

    def test_extra_field_on_dag_fork_rejected(self) -> None:
        """Typo'd field name should NOT silently default — reject."""
        with pytest.raises(ValidationError, match="(?i)extra"):
            DagFork(child="c", background=True, bg=True)  # type: ignore[call-arg]

    def test_missing_child_rejected(self) -> None:
        """``child`` is required; an object without it must error before
        the loader sees a session-resolution attempt."""
        with pytest.raises(ValidationError, match="(?i)child"):
            DagFork(background=True)  # type: ignore[call-arg]

    def test_empty_child_rejected(self) -> None:
        """``child=""`` must reject (min_length=1)."""
        with pytest.raises(ValidationError, match="(?i)at least 1"):
            DagFork(child="", background=False)

    def test_misspelled_background_field_name_rejected(self) -> None:
        """A typo in the field name (here: ``bgnd:`` instead of
        ``background``) must be caught at load time. The ``str | DagFork``
        union means Pydantic tries the string branch first and reports a
        confusing "Input should be a valid string" error rather than the
        more useful "extra field not permitted" from the DagFork branch.
        The typo IS caught (run fails fast), but the error message is
        not great. Pinning the current behavior so a future improvement
        to the error UX is visible as a test wording update."""
        text = (
            '{"session_id":"r","turns":['
            '{"messages":[{"role":"user","content":"t0"}],'
            '"forks":[{"child":"c","bgnd":true}]}]}\n'
            '{"session_id":"c","turns":[{"messages":[{"role":"user","content":"x"}]}]}'
        )
        with pytest.raises(DagLoadError):
            _load(text)

    def test_background_string_true_coerced_to_bool(self) -> None:
        """Pydantic 2 default ``bool`` coerces ``"true"`` and ``"True"``
        strings to True. Authors who hand-author with the wrong type get
        the intent-preserving coercion, not an error.

        If you want STRICT bool behavior — reject any non-bool — add
        ``StrictBool`` to the field. Pinning the current loose behavior
        so a future tightening is a visible test update."""
        f = DagFork(child="c", background="true")  # type: ignore[arg-type]
        assert f.background is True

    def test_background_int_one_coerced_to_bool(self) -> None:
        """Same coercion: ``1`` becomes True, ``0`` becomes False."""
        f1 = DagFork(child="c", background=1)  # type: ignore[arg-type]
        f0 = DagFork(child="c", background=0)  # type: ignore[arg-type]
        assert f1.background is True
        assert f0.background is False


# ============================================================================
# Empty / degenerate fork lists
# ============================================================================


class TestEmptyAndDegenerateFork:
    def test_empty_forks_list_is_a_no_op(self) -> None:
        """``forks: []`` (empty) on any turn must be accepted and produce
        no branches — same as omitting the key entirely."""
        text = (
            '{"session_id":"r","turns":['
            '{"messages":[{"role":"user","content":"t0"}],"forks":[]}]}'
        )
        convs = _load(text)
        assert _branches(convs, "r") == []

    def test_bg_fork_on_one_turn_parent(self) -> None:
        """Parent has only 1 turn; ``forks: [{c, bg:True}]`` is technically
        valid (last turn) — ``background`` flag is redundant since there's
        no continuation, but loader accepts it."""
        text = (
            '{"session_id":"r","turns":['
            '{"messages":[{"role":"user","content":"t0"}],'
            '"forks":[{"child":"c","background":true}]}]}\n'
            '{"session_id":"c","turns":[{"messages":[{"role":"user","content":"x"}]}]}'
        )
        convs = _load(text)
        b = _branches(convs, "r")[0]
        assert b.background is True
        assert b.mode == ConversationBranchMode.FORK


# ============================================================================
# Cycle detection across BG fork edges
# ============================================================================


class TestCycleDetectionAcrossBgForks:
    """BG forks add edges to the conversation graph just like FG forks.
    Cycle detection must traverse them."""

    def test_self_cycle_via_bg_fork_rejected(self) -> None:
        """A session BG-forking itself is a self-loop; cycle detector
        must catch it."""
        text = (
            '{"session_id":"r","turns":['
            '{"messages":[{"role":"user","content":"t0"}],'
            '"forks":[{"child":"r","background":true}]},'
            '{"messages":[{"role":"user","content":"t1"}]}]}'
        )
        with pytest.raises(DagLoadError, match="cycle"):
            _load(text)

    def test_two_node_cycle_through_bg_fork_rejected(self) -> None:
        """A -> bg-fork -> B -> bg-fork -> A: cycle must be rejected."""
        text = (
            '{"session_id":"a","turns":['
            '{"messages":[{"role":"user","content":"a0"}],'
            '"forks":[{"child":"b","background":true}]},'
            '{"messages":[{"role":"user","content":"a1"}]}]}\n'
            '{"session_id":"b","turns":['
            '{"messages":[{"role":"user","content":"b0"}],'
            '"forks":[{"child":"a","background":true}]},'
            '{"messages":[{"role":"user","content":"b1"}]}]}'
        )
        with pytest.raises(DagLoadError, match="cycle"):
            _load(text)

    def test_mixed_fg_bg_cycle_rejected(self) -> None:
        """Cycle through a mix of FG and BG fork edges must still be
        rejected — cycle detector is mode-agnostic."""
        text = (
            '{"session_id":"a","turns":['
            '{"messages":[{"role":"user","content":"a0"}],'
            '"forks":[{"child":"b","background":true}]},'
            '{"messages":[{"role":"user","content":"a1"}]}]}\n'
            '{"session_id":"b","turns":['
            '{"messages":[{"role":"user","content":"b0"}],"forks":["a"]}]}'
        )
        with pytest.raises(DagLoadError, match="cycle"):
            _load(text)


# ============================================================================
# Cross-edge multi-parent constraint
# ============================================================================


class TestMultiParentAcrossModes:
    """A child reached as a FORK target (FG or BG) by more than one parent
    is ambiguous (which parent's accumulator does the child seed from?).
    Reject regardless of background flag combinations."""

    def test_bg_bg_multi_parent_rejected(self) -> None:
        """Child BG-forked from two different parents."""
        text = (
            '{"session_id":"p1","turns":['
            '{"messages":[{"role":"user","content":"p1.t0"}],'
            '"forks":[{"child":"shared","background":true}]},'
            '{"messages":[{"role":"user","content":"p1.t1"}]}]}\n'
            '{"session_id":"p2","turns":['
            '{"messages":[{"role":"user","content":"p2.t0"}],'
            '"forks":[{"child":"shared","background":true}]},'
            '{"messages":[{"role":"user","content":"p2.t1"}]}]}\n'
            '{"session_id":"shared","turns":[{"messages":[{"role":"user","content":"x"}]}]}'
        )
        with pytest.raises(DagLoadError, match="single parent"):
            _load(text)

    def test_fg_bg_multi_parent_within_one_session_same_turn_rejected(self) -> None:
        """One session lists the same child as both bare-string AND
        ``background=true`` on the same turn. Even though the loader emits
        them as two distinct ConversationBranchInfo entries (FG vs BG),
        the same child appearing in both lists is a duplicate and the
        check_branch_duplicates pass should reject."""
        text = (
            '{"session_id":"r","turns":['
            '{"messages":[{"role":"user","content":"t0"}],'
            '"forks":["dup",{"child":"dup","background":true}]}]}\n'
            '{"session_id":"dup","turns":[{"messages":[{"role":"user","content":"x"}]}]}'
        )
        with pytest.raises(DagLoadError, match="duplicate"):
            _load(text)


# ============================================================================
# Mode mismatches: BG-fork target also referenced as something else
# ============================================================================


class TestModeMismatchAcrossBgFork:
    """Regression-pin the existing schema decision that a child reachable
    as both a FORK and SPAWN target on the same turn is allowed (runs
    twice, once per mode). If a future change tightens this, the test
    rewording will surface the design shift."""

    def test_bg_fork_target_also_spawn_target_on_same_turn_accepted(self) -> None:
        """A child reached as both a FORK and SPAWN target on the same
        turn is INTENTIONALLY accepted today (per ``check_branch_duplicates``
        docstring: "Fork-vs-spawn cross-pollination on the same turn is a
        distinct case (different modes, disambiguated branch_ids) and is
        intentionally allowed.").

        Result: the child runs TWICE, once as a FORK child (sticky-routed,
        inheriting context) and once as a SPAWN child (fresh, free
        routing), with two distinct correlation_ids. Counted twice in
        ``children_spawned``. This is by design but worth a regression
        test so a future tightening (e.g. rejecting the cross-mode
        duplicate) is a visible test update, not a silent semantics shift."""
        text = (
            '{"session_id":"r","turns":['
            '{"messages":[{"role":"user","content":"t0"}],'
            '"forks":[{"child":"c","background":true}],"spawns":["c"]},'
            '{"messages":[{"role":"user","content":"t1"}]}]}\n'
            '{"session_id":"c","turns":[{"messages":[{"role":"user","content":"x"}]}]}'
        )
        convs = _load(text)
        r_bs = _branches(convs, "r")
        # Two branches on r: one BG fork, one SPAWN, both naming "c".
        modes = sorted(b.mode for b in r_bs)
        assert modes == sorted(
            [ConversationBranchMode.FORK, ConversationBranchMode.SPAWN]
        )
        assert all("c" in b.child_conversation_ids for b in r_bs)


# ============================================================================
# BG fork stacked across turns (long parent, many BG fan-outs)
# ============================================================================


class TestStackedBgForks:
    """A long parent that BG-forks on multiple turns must produce one
    branch per turn (no collision between per-turn branch_ids) and the
    must-be-last rule must keep waiving for each one."""

    def test_bg_fork_on_every_non_final_turn_of_long_parent(self) -> None:
        """5-turn parent BG-forks on turns 0,1,2,3 (every non-final).
        Verifies no branch_id collision and each emits its own branch."""
        children = "\n".join(
            f'{{"session_id":"c{i}","turns":[{{"messages":[{{"role":"user","content":"c{i}"}}]}}]}}'
            for i in range(4)
        )
        parent_turns = []
        for i in range(4):
            parent_turns.append(
                f'{{"messages":[{{"role":"user","content":"t{i}"}}],'
                f'"forks":[{{"child":"c{i}","background":true}}]}}'
            )
        parent_turns.append('{"messages":[{"role":"user","content":"t4"}]}')
        text = (
            '{"session_id":"r","turns":[' + ",".join(parent_turns) + "]}\n" + children
        )
        convs = _load(text)
        bs = _branches(convs, "r")
        assert len(bs) == 4
        for i, b in enumerate(bs):
            assert b.background is True
            assert b.branch_id == f"r:{i}"
            assert b.child_conversation_ids == [f"c{i}"]


# ============================================================================
# Deep nesting (BG-fork's child has its own BG-fork)
# ============================================================================


class TestNestedBgForks:
    """A BG-fork target may itself BG-fork. The structural rules must
    apply identically at any depth."""

    def test_two_deep_bg_fork_chain(self) -> None:
        """Root BG-forks A; A BG-forks B. Both chains must validate."""
        text = (
            '{"session_id":"r","turns":['
            '{"messages":[{"role":"user","content":"r0"}],'
            '"forks":[{"child":"a","background":true}]},'
            '{"messages":[{"role":"user","content":"r1"}]}]}\n'
            '{"session_id":"a","turns":['
            '{"messages":[{"role":"user","content":"a0"}],'
            '"forks":[{"child":"b","background":true}]},'
            '{"messages":[{"role":"user","content":"a1"}]}]}\n'
            '{"session_id":"b","turns":[{"messages":[{"role":"user","content":"b"}]}]}'
        )
        convs = _load(text)
        # Sanity-check: r BG-forks a, a BG-forks b, all three sessions present.
        assert sorted(c.session_id for c in convs) == ["a", "b", "r"]
        r_bs = _branches(convs, "r")
        a_bs = _branches(convs, "a")
        assert r_bs[0].background is True
        assert r_bs[0].child_conversation_ids == ["a"]
        assert a_bs[0].background is True
        assert a_bs[0].child_conversation_ids == ["b"]


# ============================================================================
# BG-fork + SPAWN_JOIN on later turn
# ============================================================================


class TestBgForkCoexistsWithSpawnJoin:
    """A parent may BG-fork on an early turn AND have a normal
    SPAWN_JOIN gate on a later turn (e.g. 'fire-and-forget side agent
    + sync sub-agent'). Both must coexist on the same parent without
    interference."""

    def test_bg_fork_then_spawn_join_on_later_turn(self) -> None:
        """Turn 0 BG-forks a side agent (fire-and-forget). Turn 1 has
        a SPAWN with auto-join (parent suspends until that child drains).
        Turn 2 fires after the join."""
        text = (
            '{"session_id":"r","turns":['
            '{"messages":[{"role":"user","content":"t0"}],'
            '"forks":[{"child":"side","background":true}]},'
            '{"messages":[{"role":"user","content":"t1"}],"spawns":["sync"]},'
            '{"messages":[{"role":"user","content":"t2"}]}]}\n'
            '{"session_id":"side","turns":[{"messages":[{"role":"user","content":"side"}]}]}\n'
            '{"session_id":"sync","turns":[{"messages":[{"role":"user","content":"sync"}]}]}'
        )
        convs = _load(text)
        r_bs = _branches(convs, "r")
        # Two branches on r: BG fork on t0, SPAWN on t1.
        assert len(r_bs) == 2
        bg = next(b for b in r_bs if b.mode == ConversationBranchMode.FORK)
        sp = next(b for b in r_bs if b.mode == ConversationBranchMode.SPAWN)
        assert bg.background is True
        assert bg.child_conversation_ids == ["side"]
        assert sp.background is False  # SPAWN ignores background
        assert sp.child_conversation_ids == ["sync"]
