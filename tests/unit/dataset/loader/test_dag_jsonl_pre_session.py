# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Phase 2b loader tests for the ``pre_session_spawns`` conversation shorthand.

Covers:

- ``pre_session_spawns: [session_id, ...]`` on a DagConversation desugars
  into a single ``ConversationBranchInfo`` attached to turn 0, with
  ``mode=SPAWN``, ``is_background=True``, ``dispatch_timing="pre"``.
- Missing ``pre_session_spawns`` key -> no pre branch emitted.
- A child listed in ``pre_session_spawns`` that isn't declared in the
  dataset is rejected in ``_resolve_and_validate``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aiperf.common.enums import ConversationBranchMode
from aiperf.dataset.loader.dag_jsonl import DagJsonlLoader, DagLoadError


def _write(tmp_path: Path, lines: list[dict]) -> Path:
    p = tmp_path / "dag.jsonl"
    p.write_text("\n".join(json.dumps(line) for line in lines))
    return p


def test_pre_session_spawns_desugars_to_branch_with_dispatch_timing_pre(
    tmp_path: Path,
):
    path = _write(
        tmp_path,
        [
            {
                "session_id": "root",
                "pre_session_spawns": ["early_child"],
                "turns": [
                    {"messages": [{"role": "user", "content": "hi"}]},
                    {"messages": [{"role": "user", "content": "bye"}]},
                ],
            },
            {
                "session_id": "early_child",
                "turns": [{"messages": [{"role": "user", "content": "c"}]}],
            },
        ],
    )
    convs = DagJsonlLoader(filename=path).load()
    root = next(c for c in convs if c.session_id == "root")
    assert len(root.branches) == 1
    branch = root.branches[0]
    assert branch.branch_id == "root:pre"
    assert branch.mode == ConversationBranchMode.SPAWN
    # is_background was renamed to dispatch_timing="pre" in cp-main-sync.
    assert branch.dispatch_timing == "pre"
    assert branch.child_conversation_ids == ["early_child"]
    # Attached to turn 0's branch_ids.
    assert "root:pre" in root.turns[0].branch_ids
    # No SPAWN_JOIN prereq emitted (background).
    for turn in root.turns:
        for prereq in turn.prerequisites:
            assert prereq.branch_id != "root:pre"


def test_pre_session_spawns_default_empty_list(tmp_path: Path):
    """A conversation without ``pre_session_spawns`` emits no pre branch."""
    path = _write(
        tmp_path,
        [
            {
                "session_id": "root",
                "turns": [
                    {"messages": [{"role": "user", "content": "hi"}]},
                ],
            }
        ],
    )
    convs = DagJsonlLoader(filename=path).load()
    root = next(c for c in convs if c.session_id == "root")
    assert root.branches == []


def test_pre_session_spawns_rejects_duplicate_child(tmp_path: Path):
    """Duplicate child_conversation_ids in ``pre_session_spawns`` are
    rejected at load time with a location-rich error. Per-turn forks/spawns
    already reject this via ``check_branch_duplicates``; pre-session must
    match because it desugars into the same single SPAWN branch and would
    otherwise double-dispatch the child and double-count its join gate.
    """
    path = _write(
        tmp_path,
        [
            {
                "session_id": "root",
                "pre_session_spawns": ["child", "child"],
                "turns": [{"messages": [{"role": "user", "content": "hi"}]}],
            },
            {
                "session_id": "child",
                "turns": [{"messages": [{"role": "user", "content": "x"}]}],
            },
        ],
    )
    with pytest.raises(
        DagLoadError,
        match=r"session 'root' duplicate child_conversation_id 'child' "
        r"in pre_session_spawns",
    ):
        DagJsonlLoader(filename=path).load()


def test_pre_session_spawns_child_must_exist_in_dataset(tmp_path: Path):
    """Children referenced by ``pre_session_spawns`` must be declared; a
    missing id is rejected at _resolve_and_validate time."""
    path = _write(
        tmp_path,
        [
            {
                "session_id": "root",
                "pre_session_spawns": ["missing_child"],
                "turns": [
                    {"messages": [{"role": "user", "content": "hi"}]},
                ],
            }
        ],
    )
    with pytest.raises(
        DagLoadError, match="branch target 'missing_child' not declared"
    ):
        DagJsonlLoader(filename=path).load()


def test_pre_session_spawns_multiple_children(tmp_path: Path):
    """Multiple pre-session children end up in one branch's
    child_conversation_ids."""
    path = _write(
        tmp_path,
        [
            {
                "session_id": "root",
                "pre_session_spawns": ["a", "b"],
                "turns": [
                    {"messages": [{"role": "user", "content": "hi"}]},
                ],
            },
            {
                "session_id": "a",
                "turns": [{"messages": [{"role": "user", "content": "x"}]}],
            },
            {
                "session_id": "b",
                "turns": [{"messages": [{"role": "user", "content": "y"}]}],
            },
        ],
    )
    convs = DagJsonlLoader(filename=path).load()
    root = next(c for c in convs if c.session_id == "root")
    assert len(root.branches) == 1
    assert root.branches[0].child_conversation_ids == ["a", "b"]


def test_pre_session_spawns_coexists_with_per_turn_spawns(tmp_path: Path):
    """A root may declare both a pre-session branch and a regular
    per-turn SPAWN on turn 0. Two distinct branches are emitted; the
    orchestrator distinguishes them by branch_id."""
    path = _write(
        tmp_path,
        [
            {
                "session_id": "root",
                "pre_session_spawns": ["early"],
                "turns": [
                    {
                        "messages": [{"role": "user", "content": "hi"}],
                        "spawns": ["post_child"],
                    },
                    {"messages": [{"role": "user", "content": "bye"}]},
                ],
            },
            {
                "session_id": "early",
                "turns": [{"messages": [{"role": "user", "content": "c"}]}],
            },
            {
                "session_id": "post_child",
                "turns": [{"messages": [{"role": "user", "content": "c"}]}],
            },
        ],
    )
    convs = DagJsonlLoader(filename=path).load()
    root = next(c for c in convs if c.session_id == "root")
    branch_ids = {b.branch_id for b in root.branches}
    assert "root:pre" in branch_ids
    # Legacy single-spawn shorthand on turn 0 with no fork gets "root:0".
    assert "root:0" in branch_ids
    pre = next(b for b in root.branches if b.branch_id == "root:pre")
    assert pre.dispatch_timing == "pre"
    post = next(b for b in root.branches if b.branch_id == "root:0")
    assert post.dispatch_timing == "post"


def test_pre_session_spawns_target_also_fork_targeted_is_rejected(
    tmp_path: Path,
):
    """A session listed in another root's ``pre_session_spawns`` cannot
    simultaneously be a FORK target — pre-session children dispatch with
    no parent context (fresh SPAWN), while FORK children inherit context.
    Allowing both would silently choose one path at runtime."""
    path = _write(
        tmp_path,
        [
            {
                "session_id": "root_a",
                "turns": [
                    {
                        "messages": [{"role": "user", "content": "hi"}],
                        "forks": ["shared"],
                    },
                ],
            },
            {
                "session_id": "root_b",
                "pre_session_spawns": ["shared"],
                "turns": [{"messages": [{"role": "user", "content": "ok"}]}],
            },
            {
                "session_id": "shared",
                "turns": [{"messages": [{"role": "user", "content": "c"}]}],
            },
        ],
    )
    with pytest.raises(DagLoadError) as excinfo:
        DagJsonlLoader(filename=path).load()
    msg = str(excinfo.value)
    assert "shared" in msg
    assert "pre_session_spawns" in msg
    assert "FORK target" in msg
    assert "SPAWN-mode" in msg


# ===========================================================================
# Adversarial: pre_session_spawns duplicate-child detection
# ===========================================================================


def test_pre_session_spawns_rejects_non_adjacent_duplicate(tmp_path: Path):
    """A duplicate child surrounded by other entries must still be
    caught — the dedup check is not adjacency-sensitive.
    """
    path = _write(
        tmp_path,
        [
            {
                "session_id": "root",
                "pre_session_spawns": ["a", "b", "a"],
                "turns": [{"messages": [{"role": "user", "content": "hi"}]}],
            },
            {
                "session_id": "a",
                "turns": [{"messages": [{"role": "user", "content": "x"}]}],
            },
            {
                "session_id": "b",
                "turns": [{"messages": [{"role": "user", "content": "y"}]}],
            },
        ],
    )
    with pytest.raises(
        DagLoadError,
        match=r"session 'root' duplicate child_conversation_id 'a' "
        r"in pre_session_spawns",
    ):
        DagJsonlLoader(filename=path).load()


def test_pre_session_spawns_rejects_triplicate_on_first_dup(tmp_path: Path):
    """Three of the same child reports the first duplicate (the second
    occurrence), not waiting for the third. Stable error-on-first-fail.
    """
    path = _write(
        tmp_path,
        [
            {
                "session_id": "root",
                "pre_session_spawns": ["x", "x", "x"],
                "turns": [{"messages": [{"role": "user", "content": "hi"}]}],
            },
            {
                "session_id": "x",
                "turns": [{"messages": [{"role": "user", "content": "x"}]}],
            },
        ],
    )
    with pytest.raises(
        DagLoadError,
        match=r"session 'root' duplicate child_conversation_id 'x' "
        r"in pre_session_spawns",
    ):
        DagJsonlLoader(filename=path).load()


def test_pre_session_spawns_single_child_passes(tmp_path: Path):
    """A single-child ``pre_session_spawns`` cannot be a duplicate; load
    must succeed and produce one ``:pre`` branch."""
    path = _write(
        tmp_path,
        [
            {
                "session_id": "root",
                "pre_session_spawns": ["only"],
                "turns": [{"messages": [{"role": "user", "content": "hi"}]}],
            },
            {
                "session_id": "only",
                "turns": [{"messages": [{"role": "user", "content": "x"}]}],
            },
        ],
    )
    convs = DagJsonlLoader(filename=path).load()
    root = next(c for c in convs if c.session_id == "root")
    pre_branches = [b for b in root.branches if b.branch_id == "root:pre"]
    assert len(pre_branches) == 1
    assert pre_branches[0].child_conversation_ids == ["only"]


def test_pre_session_spawns_all_unique_passes(tmp_path: Path):
    """A long list of unique children must load cleanly — dedup must not
    false-positive on shared prefixes or substring overlap.
    """
    children = ["alpha", "alphabet", "al", "a", "beta", "gamma"]
    path = _write(
        tmp_path,
        [
            {
                "session_id": "root",
                "pre_session_spawns": children,
                "turns": [{"messages": [{"role": "user", "content": "hi"}]}],
            },
            *[
                {
                    "session_id": cid,
                    "turns": [{"messages": [{"role": "user", "content": cid}]}],
                }
                for cid in children
            ],
        ],
    )
    convs = DagJsonlLoader(filename=path).load()
    root = next(c for c in convs if c.session_id == "root")
    pre = next(b for b in root.branches if b.branch_id == "root:pre")
    assert pre.child_conversation_ids == children


def test_pre_session_spawns_duplicate_caught_before_disjoint_check(tmp_path: Path):
    """Authoring mistake combining a duplicate AND a fork-overlap: dedup
    fires first at intake (before ``validate_pre_session_spawns_disjoint
    _from_forks`` runs later in ``_resolve_and_validate``), so the
    surfaced error names the duplicate, not the disjoint conflict.
    """
    path = _write(
        tmp_path,
        [
            {
                "session_id": "root",
                "pre_session_spawns": ["c", "c"],
                "turns": [
                    {
                        "messages": [{"role": "user", "content": "hi"}],
                        "forks": [{"child": "c"}],
                    },
                ],
            },
            {
                "session_id": "c",
                "turns": [{"messages": [{"role": "user", "content": "x"}]}],
            },
        ],
    )
    with pytest.raises(
        DagLoadError,
        match=r"session 'root' duplicate child_conversation_id 'c' "
        r"in pre_session_spawns",
    ):
        DagJsonlLoader(filename=path).load()
