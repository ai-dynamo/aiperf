# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Phase 1 loader tests for the delayed-spawn shorthand (``DagSpawn`` object).

Covers the new object-form entry in ``DagTurn.spawns``:

- ``{"children": [...], "join_at": N}`` writes a SPAWN_JOIN prereq on
  ``turns[N]`` instead of the legacy ``turns[idx+1]``.
- ``join_at <= idx`` (forward/self-ref) and ``join_at >= num_turns`` are
  rejected at desugar time with :class:`DagLoadError`.
- Legacy string entries still desugar to turns[idx+1] unchanged.
- Mixed legacy + object forms on the same turn coexist via suffixed
  branch ids.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aiperf.common.enums import ConversationBranchMode, PrerequisiteKind
from aiperf.dataset.loader.dag_jsonl import DagJsonlLoader, DagLoadError


def _write(tmp_path: Path, lines: list[dict]) -> Path:
    p = tmp_path / "dag.jsonl"
    p.write_text("\n".join(json.dumps(line) for line in lines))
    return p


def test_delayed_spawn_object_form_desugars_correctly(tmp_path: Path):
    """A DagSpawn with ``join_at=3`` writes the prereq on turn 3, not
    turn idx+1."""
    path = _write(
        tmp_path,
        [
            {
                "session_id": "root",
                "turns": [
                    {
                        "messages": [{"role": "user", "content": "u"}],
                        "spawns": [{"children": ["child"], "join_at": 3}],
                    },
                    {"messages": [{"role": "user", "content": "u1"}]},
                    {"messages": [{"role": "user", "content": "u2"}]},
                    {"messages": [{"role": "user", "content": "u3"}]},
                ],
            },
            {
                "session_id": "child",
                "turns": [{"messages": [{"role": "user", "content": "c"}]}],
            },
        ],
    )
    convs = DagJsonlLoader(filename=path).load()
    root = next(c for c in convs if c.session_id == "root")
    assert len(root.branches) == 1
    branch = root.branches[0]
    assert branch.mode == ConversationBranchMode.SPAWN
    assert branch.child_conversation_ids == ["child"]

    # Prereq on turn 3, NOT turn 1.
    assert root.turns[1].prerequisites == []
    assert root.turns[2].prerequisites == []
    assert len(root.turns[3].prerequisites) == 1
    p = root.turns[3].prerequisites[0]
    assert p.kind == PrerequisiteKind.SPAWN_JOIN
    assert p.branch_id == branch.branch_id


def test_delayed_spawn_join_at_forward_ref_rejected(tmp_path: Path):
    """``join_at <= idx`` is rejected: self-ref or backward-ref."""
    path = _write(
        tmp_path,
        [
            {
                "session_id": "root",
                "turns": [
                    {"messages": [{"role": "user", "content": "u0"}]},
                    {
                        "messages": [{"role": "user", "content": "u1"}],
                        "spawns": [{"children": ["child"], "join_at": 0}],
                    },
                    {"messages": [{"role": "user", "content": "u2"}]},
                ],
            },
            {
                "session_id": "child",
                "turns": [{"messages": [{"role": "user", "content": "c"}]}],
            },
        ],
    )
    with pytest.raises(DagLoadError, match="must be strictly greater"):
        DagJsonlLoader(filename=path).load()


def test_delayed_spawn_join_at_out_of_range_rejected(tmp_path: Path):
    """``join_at >= num_turns`` is rejected."""
    path = _write(
        tmp_path,
        [
            {
                "session_id": "root",
                "turns": [
                    {
                        "messages": [{"role": "user", "content": "u0"}],
                        "spawns": [{"children": ["child"], "join_at": 5}],
                    },
                    {"messages": [{"role": "user", "content": "u1"}]},
                ],
            },
            {
                "session_id": "child",
                "turns": [{"messages": [{"role": "user", "content": "c"}]}],
            },
        ],
    )
    with pytest.raises(DagLoadError, match="out of range"):
        DagJsonlLoader(filename=path).load()


def test_delayed_spawn_legacy_string_form_still_works(tmp_path: Path):
    """Existing bare-string ``spawns`` entries continue to desugar to
    turns[idx+1] with no change."""
    path = _write(
        tmp_path,
        [
            {
                "session_id": "root",
                "turns": [
                    {
                        "messages": [{"role": "user", "content": "u"}],
                        "spawns": ["child"],
                    },
                    {"messages": [{"role": "user", "content": "u1"}]},
                ],
            },
            {
                "session_id": "child",
                "turns": [{"messages": [{"role": "user", "content": "c"}]}],
            },
        ],
    )
    convs = DagJsonlLoader(filename=path).load()
    root = next(c for c in convs if c.session_id == "root")
    assert len(root.branches) == 1
    assert len(root.turns[1].prerequisites) == 1
    assert root.turns[1].prerequisites[0].branch_id == root.branches[0].branch_id


def test_delayed_spawn_mixed_legacy_and_object_forms_same_turn(tmp_path: Path):
    """Mixing a legacy string entry with a DagSpawn object on the same turn
    desugars each to its own ConversationBranchInfo with a suffixed
    branch_id. Phase 1 still routes through the v1 validator which rejects
    two gated branches on one spawning turn — so this authoring is legal
    only when at least one entry is terminal/background. Verify the pure
    desugar without invoking the validator."""
    loader = DagJsonlLoader(filename=tmp_path / "unused.jsonl")
    # Inject internal state directly to exercise _desugar_forks without
    # round-tripping through the validator.
    from aiperf.common.models.dataset_models import Conversation, Turn

    root = Conversation(
        session_id="root",
        turns=[
            Turn(raw_messages=[{"role": "user", "content": "u"}]),
            Turn(raw_messages=[{"role": "user", "content": "u1"}]),
            Turn(raw_messages=[{"role": "user", "content": "u2"}]),
            Turn(raw_messages=[{"role": "user", "content": "u3"}]),
        ],
    )
    loader._conversations["root"] = root
    # Group 1: legacy bare string with join_at=None -> default idx+1=1.
    # Group 2: DagSpawn object with join_at=3.
    loader._inline_forks["root"] = [[], [], [], []]
    loader._inline_spawns["root"] = [
        [
            (["legacy_child"], None),
            (["delayed_child"], 3),
        ],
        [],
        [],
        [],
    ]
    loader._desugar_forks()

    # Two branches with suffixed ids.
    assert len(root.branches) == 2
    ids = {b.branch_id for b in root.branches}
    assert "root:0:spawn" in ids
    assert "root:0:spawn1" in ids

    # Legacy prereq on turn 1, delayed prereq on turn 3.
    prereqs_on_t1 = [p.branch_id for p in root.turns[1].prerequisites]
    prereqs_on_t3 = [p.branch_id for p in root.turns[3].prerequisites]
    assert "root:0:spawn" in prereqs_on_t1
    assert "root:0:spawn1" in prereqs_on_t3
