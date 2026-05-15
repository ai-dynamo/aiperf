# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path

from aiperf.common.enums import ConversationBranchMode, PrerequisiteKind
from aiperf.dataset.loader.dag_jsonl import DagJsonlLoader


def _write(tmp_path: Path, lines: list[dict]) -> Path:
    p = tmp_path / "dag.jsonl"
    p.write_text("\n".join(json.dumps(line) for line in lines))
    return p


def test_non_terminal_spawns_desugar_to_branch_plus_prereq(tmp_path: Path):
    path = _write(
        tmp_path,
        [
            {
                "session_id": "root",
                "turns": [
                    {
                        "messages": [{"role": "user", "content": "hi"}],
                        "spawns": ["child"],
                    },
                    {"messages": [{"role": "user", "content": "after"}]},
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
    # Branch on turn 0
    assert len(root.branches) == 1
    assert root.branches[0].mode == ConversationBranchMode.SPAWN
    # Prereq on turn 1 referencing the branch
    assert len(root.turns[1].prerequisites) == 1
    p = root.turns[1].prerequisites[0]
    assert p.kind == PrerequisiteKind.SPAWN_JOIN
    assert p.branch_id == root.branches[0].branch_id


def test_terminal_spawns_produce_branch_without_prereq(tmp_path: Path):
    # Last turn spawns: no "next turn" to attach a prereq to -> branch only.
    path = _write(
        tmp_path,
        [
            {
                "session_id": "root",
                "turns": [
                    {
                        "messages": [{"role": "user", "content": "hi"}],
                        "spawns": ["child"],
                    },
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
    # No prereqs anywhere (only one turn in root).
    assert all(not t.prerequisites for t in root.turns)
