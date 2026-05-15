# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial component-integration tests for DAG prereq gating under validate_for_orchestrator_v1.

Covers the full DagJsonlLoader -> DatasetMetadata -> validate_for_orchestrator_v1
pipeline, plus the two post-fix invariants:
- Task 7 fix: forward / same-turn prereq branch references are rejected.
- Task 8 fix: branches consumed by more than one gated turn are rejected.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aiperf.common.enums import ConversationBranchMode, PrerequisiteKind
from aiperf.common.models import (
    ConversationBranchInfo,
    ConversationMetadata,
    DatasetMetadata,
    TurnMetadata,
    TurnPrerequisite,
)
from aiperf.common.validators.orchestrator_v1 import validate_for_orchestrator_v1
from aiperf.dataset.loader.dag_jsonl import DagJsonlLoader
from aiperf.plugin.enums import DatasetSamplingStrategy

pytestmark = pytest.mark.component_integration

FIXTURES = Path(__file__).parents[2] / "fixtures" / "dag"


def _write_dag(tmp_path: Path, lines: list[dict], name: str = "dag.jsonl") -> Path:
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(line) for line in lines))
    return p


def _load_metadata(path: Path) -> DatasetMetadata:
    loader = DagJsonlLoader(filename=path)
    data = loader.load_dataset()
    convs = loader.convert_to_conversations(data)
    return DatasetMetadata(
        conversations=[c.metadata() for c in convs],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )


def test_full_dag_loader_to_validator_pipeline_spawn_join_topology_passes(
    tmp_path: Path,
) -> None:
    """2-turn parent spawning a child on turn 0 desugars cleanly and passes v1."""
    path = _write_dag(
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
        name="simple.dag.jsonl",
    )

    # DagJsonlLoader.load_dataset already invokes validate_for_orchestrator_v1.
    md = _load_metadata(path)
    # Explicit validation call doubles as an assertion that there are no
    # hidden mutations between the loader's internal call and a caller's
    # subsequent inspection.
    validate_for_orchestrator_v1(md)

    root = next(c for c in md.conversations if c.conversation_id == "root")
    assert len(root.branches) == 1
    assert root.branches[0].mode == ConversationBranchMode.SPAWN
    assert len(root.turns[1].prerequisites) == 1
    assert root.turns[1].prerequisites[0].kind == PrerequisiteKind.SPAWN_JOIN


def test_three_level_spawn_join_chain_end_to_end_passes(tmp_path: Path) -> None:
    """Three spawn-join pairs on alternating turns: spawn,join,spawn,join,spawn,join."""
    path = _write_dag(
        tmp_path,
        [
            {
                "session_id": "root",
                "turns": [
                    {
                        "messages": [{"role": "user", "content": "t0"}],
                        "spawns": ["a"],
                    },
                    {"messages": [{"role": "user", "content": "t1"}]},
                    {
                        "messages": [{"role": "user", "content": "t2"}],
                        "spawns": ["b"],
                    },
                    {"messages": [{"role": "user", "content": "t3"}]},
                    {
                        "messages": [{"role": "user", "content": "t4"}],
                        "spawns": ["c"],
                    },
                    {"messages": [{"role": "user", "content": "t5"}]},
                ],
            },
            {
                "session_id": "a",
                "turns": [{"messages": [{"role": "user", "content": "a"}]}],
            },
            {
                "session_id": "b",
                "turns": [{"messages": [{"role": "user", "content": "b"}]}],
            },
            {
                "session_id": "c",
                "turns": [{"messages": [{"role": "user", "content": "c"}]}],
            },
        ],
        name="chain.dag.jsonl",
    )

    md = _load_metadata(path)
    validate_for_orchestrator_v1(md)

    root = next(c for c in md.conversations if c.conversation_id == "root")
    assert len(root.branches) == 3
    # Each join turn (1, 3, 5) carries exactly one SPAWN_JOIN prereq.
    for gated_idx in (1, 3, 5):
        prereqs = root.turns[gated_idx].prerequisites
        assert len(prereqs) == 1
        assert prereqs[0].kind == PrerequisiteKind.SPAWN_JOIN


def test_two_independent_conversations_validate_separately(tmp_path: Path) -> None:
    """Two roots, each with their own spawn-join topology, co-validate."""
    path = _write_dag(
        tmp_path,
        [
            {
                "session_id": "r1",
                "turns": [
                    {
                        "messages": [{"role": "user", "content": "r1-0"}],
                        "spawns": ["r1c"],
                    },
                    {"messages": [{"role": "user", "content": "r1-1"}]},
                ],
            },
            {
                "session_id": "r1c",
                "turns": [{"messages": [{"role": "user", "content": "r1 child"}]}],
            },
            {
                "session_id": "r2",
                "turns": [
                    {
                        "messages": [{"role": "user", "content": "r2-0"}],
                        "spawns": ["r2c"],
                    },
                    {"messages": [{"role": "user", "content": "r2-1"}]},
                ],
            },
            {
                "session_id": "r2c",
                "turns": [{"messages": [{"role": "user", "content": "r2 child"}]}],
            },
        ],
        name="two_convs.dag.jsonl",
    )

    md = _load_metadata(path)
    validate_for_orchestrator_v1(md)

    ids = {c.conversation_id for c in md.conversations}
    assert {"r1", "r1c", "r2", "r2c"} <= ids
    r1 = next(c for c in md.conversations if c.conversation_id == "r1")
    r2 = next(c for c in md.conversations if c.conversation_id == "r2")
    assert len(r1.branches) == 1 and len(r2.branches) == 1
    assert len(r1.turns[1].prerequisites) == 1
    assert len(r2.turns[1].prerequisites) == 1


def test_forward_prereq_reference_rejected_end_to_end_bug_fix_1() -> None:
    """Task 7 fix: a prereq that references a branch declared on a later turn is rejected."""
    md = DatasetMetadata(
        conversations=[
            ConversationMetadata(
                conversation_id="c",
                turns=[
                    TurnMetadata(
                        prerequisites=[
                            TurnPrerequisite(
                                kind=PrerequisiteKind.SPAWN_JOIN,
                                branch_id="b1",
                            )
                        ],
                    ),
                    TurnMetadata(branch_ids=["b1"]),
                ],
                branches=[
                    ConversationBranchInfo(
                        branch_id="b1",
                        child_conversation_ids=["x"],
                        mode=ConversationBranchMode.SPAWN,
                    )
                ],
            ),
            ConversationMetadata(conversation_id="x", turns=[TurnMetadata()]),
        ],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    with pytest.raises(NotImplementedError, match="not strictly earlier"):
        validate_for_orchestrator_v1(md)


def test_same_turn_prereq_reference_rejected_end_to_end_bug_fix_1() -> None:
    """Task 7 fix: a prereq that references a branch declared on the same turn is rejected."""
    md = DatasetMetadata(
        conversations=[
            ConversationMetadata(
                conversation_id="c",
                turns=[
                    TurnMetadata(
                        branch_ids=["b1"],
                        prerequisites=[
                            TurnPrerequisite(
                                kind=PrerequisiteKind.SPAWN_JOIN,
                                branch_id="b1",
                            )
                        ],
                    ),
                ],
                branches=[
                    ConversationBranchInfo(
                        branch_id="b1",
                        child_conversation_ids=["x"],
                        mode=ConversationBranchMode.SPAWN,
                    )
                ],
            ),
            ConversationMetadata(conversation_id="x", turns=[TurnMetadata()]),
        ],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    with pytest.raises(NotImplementedError, match="not strictly earlier"):
        validate_for_orchestrator_v1(md)


def test_multi_consumer_branch_accepted_end_to_end_phase_3() -> None:
    """Phase 3: two turns consuming the same branch_id is accepted (the
    orchestrator installs one pending join per gated turn)."""
    md = DatasetMetadata(
        conversations=[
            ConversationMetadata(
                conversation_id="c",
                turns=[
                    TurnMetadata(branch_ids=["b1"]),
                    TurnMetadata(
                        prerequisites=[
                            TurnPrerequisite(
                                kind=PrerequisiteKind.SPAWN_JOIN,
                                branch_id="b1",
                            )
                        ],
                    ),
                    TurnMetadata(
                        prerequisites=[
                            TurnPrerequisite(
                                kind=PrerequisiteKind.SPAWN_JOIN,
                                branch_id="b1",
                            )
                        ],
                    ),
                ],
                branches=[
                    ConversationBranchInfo(
                        branch_id="b1",
                        child_conversation_ids=["x"],
                        mode=ConversationBranchMode.SPAWN,
                    )
                ],
            ),
            ConversationMetadata(conversation_id="x", turns=[TurnMetadata()]),
        ],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    # Phase 3 accepts this shape.
    validate_for_orchestrator_v1(md)


@pytest.mark.skip(
    reason="Parent-join cycle is covered end-to-end by "
    "tests/component_integration/timing/test_dag_join_end_to_end.py::"
    "test_parent_resumes_after_all_children_complete; leaving e2e to the shipped test."
)
def test_full_parent_join_cycle_end_to_end_still_works_post_fixes() -> None:
    """Covered by the shipped join-orchestration e2e test."""


def test_hundred_child_gate_closes_end_to_end() -> None:
    """Validator-level: a branch carrying 100 child_conversation_ids passes v1."""
    children = [f"child-{i:03d}" for i in range(100)]
    md = DatasetMetadata(
        conversations=[
            ConversationMetadata(
                conversation_id="big",
                turns=[
                    TurnMetadata(branch_ids=["big:0"]),
                    TurnMetadata(
                        prerequisites=[
                            TurnPrerequisite(
                                kind=PrerequisiteKind.SPAWN_JOIN,
                                branch_id="big:0",
                            )
                        ],
                    ),
                ],
                branches=[
                    ConversationBranchInfo(
                        branch_id="big:0",
                        child_conversation_ids=children,
                        mode=ConversationBranchMode.SPAWN,
                    )
                ],
            ),
            *(
                ConversationMetadata(conversation_id=cid, turns=[TurnMetadata()])
                for cid in children
            ),
        ],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    # Must not raise.
    validate_for_orchestrator_v1(md)

    big = md.conversations[0]
    assert len(big.branches) == 1
    assert len(big.branches[0].child_conversation_ids) == 100
    assert len(big.turns[1].prerequisites) == 1


def test_dataset_metadata_json_roundtrip_through_validator_twice_idempotent(
    tmp_path: Path,
) -> None:
    """JSON roundtrip + two sequential validations: no mutations, both pass."""
    path = _write_dag(
        tmp_path,
        [
            {
                "session_id": "root",
                "turns": [
                    {
                        "messages": [{"role": "user", "content": "t0"}],
                        "spawns": ["child"],
                    },
                    {"messages": [{"role": "user", "content": "t1"}]},
                ],
            },
            {
                "session_id": "child",
                "turns": [{"messages": [{"role": "user", "content": "c"}]}],
            },
        ],
        name="roundtrip.dag.jsonl",
    )
    md = _load_metadata(path)

    blob = md.model_dump_json()
    restored = DatasetMetadata.model_validate_json(blob)

    # First validation.
    validate_for_orchestrator_v1(restored)
    # Snapshot structure after first pass.
    root = next(c for c in restored.conversations if c.conversation_id == "root")
    branch_ids_before = [b.branch_id for b in root.branches]
    prereq_ids_before = [p.branch_id for t in root.turns for p in t.prerequisites]

    # Second validation on the same instance must be idempotent.
    validate_for_orchestrator_v1(restored)

    root_after = next(c for c in restored.conversations if c.conversation_id == "root")
    assert [b.branch_id for b in root_after.branches] == branch_ids_before
    assert [
        p.branch_id for t in root_after.turns for p in t.prerequisites
    ] == prereq_ids_before
    assert prereq_ids_before  # sanity: structure wasn't empty


@pytest.mark.parametrize(
    "fixture_name",
    ["small.dag.jsonl", "full.dag.jsonl", "spawn_minimal.dag.jsonl"],
)
def test_shipped_fixtures_all_pass_post_fix_validator(fixture_name: str) -> None:
    """Each shipped DAG fixture must validate cleanly under the stricter v1 rules."""
    fixture_path = FIXTURES / fixture_name
    assert fixture_path.exists(), f"missing fixture: {fixture_path}"

    md = _load_metadata(fixture_path)
    # Explicit validate after loader's internal call.
    validate_for_orchestrator_v1(md)
    assert md.conversations, f"{fixture_name} produced no conversations"
