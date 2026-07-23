# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial / pathological test cases for ``DagFork.background`` strict-mode loader posture."""

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


class TestDagForkPydanticStrictMode:
    """The loader leans on Pydantic ``extra="forbid"`` to catch typo'd field"""

    def test_extra_field_on_dag_fork_rejected(self) -> None:
        """Typo'd field name should NOT silently default — reject."""
        with pytest.raises(ValidationError, match="(?i)extra"):
            DagFork(child="c", background=True, bg=True)  # type: ignore[call-arg]

    def test_missing_child_rejected(self) -> None:
        """``child`` is required; an object without it must error before"""
        with pytest.raises(ValidationError, match="(?i)child"):
            DagFork(background=True)  # type: ignore[call-arg]

    def test_empty_child_rejected(self) -> None:
        """``child=""`` must reject (min_length=1)."""
        with pytest.raises(ValidationError, match="(?i)at least 1"):
            DagFork(child="", background=False)

    def test_misspelled_background_field_name_rejected(self) -> None:
        """A typo in the field name (here: ``bgnd:`` instead of"""
        text = (
            '{"session_id":"r","turns":['
            '{"messages":[{"role":"user","content":"t0"}],'
            '"forks":[{"child":"c","bgnd":true}]}]}\n'
            '{"session_id":"c","turns":[{"messages":[{"role":"user","content":"x"}]}]}'
        )
        with pytest.raises(DagLoadError):
            _load(text)

    def test_background_string_true_coerced_to_bool(self) -> None:
        """Pydantic 2 default ``bool`` coerces ``"true"`` and ``"True"``"""
        f = DagFork(child="c", background="true")  # type: ignore[arg-type]
        assert f.background is True

    def test_background_int_one_coerced_to_bool(self) -> None:
        """Same coercion: ``1`` becomes True, ``0`` becomes False."""
        f1 = DagFork(child="c", background=1)  # type: ignore[arg-type]
        f0 = DagFork(child="c", background=0)  # type: ignore[arg-type]
        assert f1.background is True
        assert f0.background is False


class TestEmptyAndDegenerateFork:
    def test_empty_forks_list_is_a_no_op(self) -> None:
        """``forks: []`` (empty) on any turn must be accepted and produce"""
        text = (
            '{"session_id":"r","turns":['
            '{"messages":[{"role":"user","content":"t0"}],"forks":[]}]}'
        )
        convs = _load(text)
        assert _branches(convs, "r") == []

    def test_bg_fork_on_one_turn_parent(self) -> None:
        """Parent has only 1 turn; ``forks: [{c, bg:True}]`` is technically"""
        text = (
            '{"session_id":"r","turns":['
            '{"messages":[{"role":"user","content":"t0"}],'
            '"forks":[{"child":"c","background":true}]}]}\n'
            '{"session_id":"c","turns":[{"messages":[{"role":"user","content":"x"}]}]}'
        )
        convs = _load(text)
        b = _branches(convs, "r")[0]
        assert b.is_background is True
        assert b.mode == ConversationBranchMode.FORK


class TestCycleDetectionAcrossBgForks:
    """BG forks add edges to the conversation graph just like FG forks."""

    def test_self_cycle_via_bg_fork_rejected(self) -> None:
        """A session BG-forking itself is a self-loop; cycle detector"""
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
        """Cycle through a mix of FG and BG fork edges must still be"""
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


class TestMultiParentAcrossModes:
    """A child reached as a FORK target (FG or BG) by more than one parent"""

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
        """One session lists the same child as both bare-string AND"""
        text = (
            '{"session_id":"r","turns":['
            '{"messages":[{"role":"user","content":"t0"}],'
            '"forks":["dup",{"child":"dup","background":true}]}]}\n'
            '{"session_id":"dup","turns":[{"messages":[{"role":"user","content":"x"}]}]}'
        )
        with pytest.raises(DagLoadError, match="duplicate"):
            _load(text)


class TestModeMismatchAcrossBgFork:
    """Regression-pin the existing schema decision that a child reachable"""

    def test_bg_fork_target_also_spawn_target_on_same_turn_accepted(self) -> None:
        """A child reached as both a FORK and SPAWN target on the same"""
        text = (
            '{"session_id":"r","turns":['
            '{"messages":[{"role":"user","content":"t0"}],'
            '"forks":[{"child":"c","background":true}],"spawns":["c"]},'
            '{"messages":[{"role":"user","content":"t1"}]}]}\n'
            '{"session_id":"c","turns":[{"messages":[{"role":"user","content":"x"}]}]}'
        )
        convs = _load(text)
        r_bs = _branches(convs, "r")
        modes = sorted(b.mode for b in r_bs)
        assert modes == sorted(
            [ConversationBranchMode.FORK, ConversationBranchMode.SPAWN]
        )
        assert all("c" in b.child_conversation_ids for b in r_bs)


class TestStackedBgForks:
    """A long parent that BG-forks on multiple turns must produce one"""

    def test_bg_fork_on_every_non_final_turn_of_long_parent(self) -> None:
        """5-turn parent BG-forks on turns 0,1,2,3 (every non-final)."""
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
            assert b.is_background is True
            assert b.branch_id == f"r:{i}"
            assert b.child_conversation_ids == [f"c{i}"]


class TestNestedBgForks:
    """A BG-fork target may itself BG-fork. The structural rules must"""

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
        assert sorted(c.session_id for c in convs) == ["a", "b", "r"]
        r_bs = _branches(convs, "r")
        a_bs = _branches(convs, "a")
        assert r_bs[0].is_background is True
        assert r_bs[0].child_conversation_ids == ["a"]
        assert a_bs[0].is_background is True
        assert a_bs[0].child_conversation_ids == ["b"]


class TestBgForkCoexistsWithSpawnJoin:
    """A parent may BG-fork on an early turn AND have a normal"""

    def test_bg_fork_then_spawn_join_on_later_turn(self) -> None:
        """Turn 0 BG-forks a side agent (fire-and-forget). Turn 1 has"""
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
        assert len(r_bs) == 2
        bg = next(b for b in r_bs if b.mode == ConversationBranchMode.FORK)
        sp = next(b for b in r_bs if b.mode == ConversationBranchMode.SPAWN)
        assert bg.is_background is True
        assert bg.child_conversation_ids == ["side"]
        assert sp.is_background is False
        assert sp.child_conversation_ids == ["sync"]
