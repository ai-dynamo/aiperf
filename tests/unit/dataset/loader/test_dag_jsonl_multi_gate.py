# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Phase 2 loader tests: multi-entry ``spawns`` with distinct ``join_at``.

Phase 1 validator rejected multiple gated branches declared on a single
spawning turn. Phase 2 lifts that rejection, so JSONL inputs with two
DagSpawn entries on the same turn — each with a different ``join_at`` —
now load successfully and pass the orchestrator_v1 validator.
"""

from __future__ import annotations

import json
from pathlib import Path

from aiperf.common.enums import ConversationBranchMode, PrerequisiteKind
from aiperf.common.validators.orchestrator_v1 import (
    validate_for_orchestrator_v1,
)
from aiperf.dataset.loader.dag_jsonl import DagJsonlLoader


def _write(tmp_path: Path, lines: list[dict]) -> Path:
    p = tmp_path / "dag_multi_gate.jsonl"
    p.write_text("\n".join(json.dumps(line) for line in lines))
    return p


def _conversation_to_metadata(conversations):
    """Build DatasetMetadata via the same translation the orchestrator uses."""
    from aiperf.common.models import (
        ConversationMetadata,
        DatasetMetadata,
        TurnMetadata,
    )
    from aiperf.plugin.enums import DatasetSamplingStrategy

    metas = []
    for conv in conversations:
        turns = [
            TurnMetadata(
                branch_ids=list(t.branch_ids),
                prerequisites=list(t.prerequisites),
            )
            for t in conv.turns
        ]
        metas.append(
            ConversationMetadata(
                conversation_id=conv.session_id,
                turns=turns,
                branches=list(conv.branches),
            )
        )
    return DatasetMetadata(
        conversations=metas,
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )


def test_multi_spawn_entries_with_distinct_join_at_loads(tmp_path: Path):
    """Turn 0 with two DagSpawn entries — one gating at T=1, one at T=3 —
    loads successfully and passes the v1 validator (Phase 2)."""
    path = _write(
        tmp_path,
        [
            {
                "session_id": "root",
                "turns": [
                    {
                        "messages": [{"role": "user", "content": "u0"}],
                        "spawns": [
                            {"children": ["child_a"], "join_at": 1},
                            {"children": ["child_b"], "join_at": 3},
                        ],
                    },
                    {"messages": [{"role": "user", "content": "u1"}]},
                    {"messages": [{"role": "user", "content": "u2"}]},
                    {"messages": [{"role": "user", "content": "u3"}]},
                ],
            },
            {
                "session_id": "child_a",
                "turns": [{"messages": [{"role": "user", "content": "ca"}]}],
            },
            {
                "session_id": "child_b",
                "turns": [{"messages": [{"role": "user", "content": "cb"}]}],
            },
        ],
    )
    convs = DagJsonlLoader(filename=path).load()
    root = next(c for c in convs if c.session_id == "root")

    # Two branches with suffixed ids on turn 0.
    spawn_branches = [
        b for b in root.branches if b.mode == ConversationBranchMode.SPAWN
    ]
    assert len(spawn_branches) == 2
    branch_ids = {b.branch_id for b in spawn_branches}
    assert "root:0:spawn" in branch_ids
    assert "root:0:spawn1" in branch_ids

    # First entry gated at T=1; second at T=3.
    prereqs_t1 = [p.branch_id for p in root.turns[1].prerequisites]
    prereqs_t3 = [p.branch_id for p in root.turns[3].prerequisites]
    assert "root:0:spawn" in prereqs_t1
    assert "root:0:spawn1" in prereqs_t3

    # v1 validator accepts the multi-gated shape in Phase 2.
    metadata = _conversation_to_metadata(convs)
    validate_for_orchestrator_v1(metadata)

    # Sanity: both prereqs are SPAWN_JOIN.
    assert root.turns[1].prerequisites[0].kind == PrerequisiteKind.SPAWN_JOIN
    assert root.turns[3].prerequisites[0].kind == PrerequisiteKind.SPAWN_JOIN
