# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial shape / JSON round-trip tests for DAG prereq and branch metadata."""

from aiperf.common.enums import (
    ConversationBranchMode,
    PrerequisiteKind,
)
from aiperf.common.models import (
    ConversationBranchInfo,
    ConversationMetadata,
    DatasetMetadata,
    Turn,
    TurnMetadata,
    TurnPrerequisite,
)
from aiperf.plugin.enums import DatasetSamplingStrategy


def _kind_kwargs(kind: PrerequisiteKind) -> dict:
    """Return the minimal extra kwargs a given PrerequisiteKind needs, beyond ``kind``.

    The v1 orchestrator rejects most of these at load time, but pydantic
    construction should succeed for all kinds with any of the optional fields
    populated. We stay semantically consistent so the round-trip is meaningful.
    """
    if kind == PrerequisiteKind.SPAWN_JOIN:
        return {"branch_id": "b:0"}
    if kind == PrerequisiteKind.CHILD_SESSION_COMPLETE:
        return {"child_conversation_ids": ["c:0"]}
    if kind == PrerequisiteKind.TIMER:
        return {"timer_seconds": 1.5}
    if kind == PrerequisiteKind.EXTERNAL_EVENT:
        return {"event_name": "ready"}
    if kind == PrerequisiteKind.BARRIER:
        return {"barrier_id": "bar:0"}
    return {}


def test_dataset_metadata_json_roundtrip_preserves_prereqs_all_kinds():
    """DatasetMetadata json round-trip preserves one TurnPrerequisite per PrerequisiteKind value."""
    prereqs = [
        TurnPrerequisite(kind=kind, **_kind_kwargs(kind)) for kind in PrerequisiteKind
    ]
    turn_meta = TurnMetadata(prerequisites=prereqs)
    conv = ConversationMetadata(conversation_id="c0", turns=[turn_meta])
    ds = DatasetMetadata(
        conversations=[conv],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )

    restored = DatasetMetadata.model_validate_json(ds.model_dump_json())

    restored_prereqs = restored.conversations[0].turns[0].prerequisites
    assert len(restored_prereqs) == len(list(PrerequisiteKind))
    assert [p.kind for p in restored_prereqs] == [k for k in PrerequisiteKind]
    # Spot-check that the per-kind optional field round-trips.
    by_kind = {p.kind: p for p in restored_prereqs}
    assert by_kind[PrerequisiteKind.SPAWN_JOIN].branch_id == "b:0"
    assert by_kind[PrerequisiteKind.CHILD_SESSION_COMPLETE].child_conversation_ids == [
        "c:0"
    ]
    assert by_kind[PrerequisiteKind.TIMER].timer_seconds == 1.5
    assert by_kind[PrerequisiteKind.EXTERNAL_EVENT].event_name == "ready"
    assert by_kind[PrerequisiteKind.BARRIER].barrier_id == "bar:0"


def test_turn_metadata_copied_deep_from_turn_mutation_isolated():
    """Mutating the TurnMetadata.prerequisites list returned by Turn.metadata() must not leak back."""
    p = TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="b")
    turn = Turn(prerequisites=[p])
    meta = turn.metadata()

    meta.prerequisites.append(
        TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="b2")
    )

    assert len(turn.prerequisites) == 1
    assert turn.prerequisites[0].branch_id == "b"


def test_turn_metadata_default_prerequisites_distinct_per_instance():
    """Two TurnMetadata() instances must NOT share a default prerequisites list (no default-factory aliasing)."""
    a = TurnMetadata()
    b = TurnMetadata()

    a.prerequisites.append(
        TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="x")
    )

    assert b.prerequisites == []
    assert a.prerequisites is not b.prerequisites


def test_conversation_branch_info_json_roundtrip_preserves_dispatch_timing_and_mode():
    """ConversationBranchInfo round-trips mode + dispatch_timing + child ids verbatim."""
    info = ConversationBranchInfo(
        branch_id="b",
        child_conversation_ids=["c"],
        mode=ConversationBranchMode.SPAWN,
        dispatch_timing="pre",
    )

    restored = ConversationBranchInfo.model_validate_json(info.model_dump_json())

    assert restored.branch_id == "b"
    assert restored.child_conversation_ids == ["c"]
    assert restored.mode == ConversationBranchMode.SPAWN
    assert restored.dispatch_timing == "pre"


def test_metadata_with_ten_thousand_prereqs_on_one_turn_roundtrips():
    """10_000-entry prerequisites list survives JSON round-trip (length + spot-checks)."""
    n = 10_000
    prereqs = [
        TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id=f"b:{i}")
        for i in range(n)
    ]
    turn_meta = TurnMetadata(prerequisites=prereqs)
    conv = ConversationMetadata(conversation_id="c0", turns=[turn_meta])
    ds = DatasetMetadata(
        conversations=[conv],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )

    restored = DatasetMetadata.model_validate_json(ds.model_dump_json())
    restored_prereqs = restored.conversations[0].turns[0].prerequisites

    assert len(restored_prereqs) == n
    assert restored_prereqs[0].branch_id == "b:0"
    assert restored_prereqs[n // 2].branch_id == f"b:{n // 2}"
    assert restored_prereqs[-1].branch_id == f"b:{n - 1}"
    # All entries share the same kind.
    assert restored_prereqs[0].kind == PrerequisiteKind.SPAWN_JOIN
    assert restored_prereqs[-1].kind == PrerequisiteKind.SPAWN_JOIN


def test_deeply_nested_conversations_root_child_grandchild_serialize():
    """3-level DAG (root -> child -> grandchild) metadata survives JSON round-trip."""
    root_branch = ConversationBranchInfo(
        branch_id="root:0",
        child_conversation_ids=["child"],
        mode=ConversationBranchMode.SPAWN,
    )
    child_branch = ConversationBranchInfo(
        branch_id="child:0",
        child_conversation_ids=["grandchild"],
        mode=ConversationBranchMode.SPAWN,
    )
    root = ConversationMetadata(
        conversation_id="root",
        turns=[TurnMetadata(branch_ids=["root:0"])],
        branches=[root_branch],
        is_root=True,
        agent_depth=0,
    )
    child = ConversationMetadata(
        conversation_id="child",
        turns=[TurnMetadata(branch_ids=["child:0"])],
        branches=[child_branch],
        is_root=False,
        agent_depth=1,
        parent_conversation_id="root",
    )
    grandchild = ConversationMetadata(
        conversation_id="grandchild",
        turns=[TurnMetadata()],
        branches=[],
        is_root=False,
        agent_depth=2,
        parent_conversation_id="child",
    )
    ds = DatasetMetadata(
        conversations=[root, child, grandchild],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )

    restored = DatasetMetadata.model_validate_json(ds.model_dump_json())

    assert len(restored.conversations) == 3
    by_id = {c.conversation_id: c for c in restored.conversations}
    assert by_id["root"].agent_depth == 0
    assert by_id["child"].agent_depth == 1
    assert by_id["child"].parent_conversation_id == "root"
    assert by_id["grandchild"].agent_depth == 2
    assert by_id["grandchild"].parent_conversation_id == "child"
    assert by_id["root"].branches[0].child_conversation_ids == ["child"]
    assert by_id["child"].branches[0].child_conversation_ids == ["grandchild"]
    assert by_id["grandchild"].branches == []


def test_conversation_metadata_empty_branches_roundtrip():
    """ConversationMetadata with branches=[] preserves the empty list across JSON round-trip."""
    conv = ConversationMetadata(
        conversation_id="x",
        turns=[TurnMetadata()],
        branches=[],
    )

    restored = ConversationMetadata.model_validate_json(conv.model_dump_json())

    assert restored.branches == []
    assert restored.conversation_id == "x"
    assert len(restored.turns) == 1


def test_branch_id_duplicate_across_conversations_is_legal():
    """Two separate conversations each declaring a branch with the same branch_id must validate."""
    branch_a = ConversationBranchInfo(
        branch_id="shared:0",
        child_conversation_ids=["child_a"],
        mode=ConversationBranchMode.SPAWN,
    )
    branch_b = ConversationBranchInfo(
        branch_id="shared:0",
        child_conversation_ids=["child_b"],
        mode=ConversationBranchMode.SPAWN,
    )
    conv_a = ConversationMetadata(
        conversation_id="conv_a",
        turns=[TurnMetadata()],
        branches=[branch_a],
    )
    conv_b = ConversationMetadata(
        conversation_id="conv_b",
        turns=[TurnMetadata()],
        branches=[branch_b],
    )

    ds = DatasetMetadata(
        conversations=[conv_a, conv_b],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )

    assert len(ds.conversations) == 2
    assert ds.conversations[0].branches[0].branch_id == "shared:0"
    assert ds.conversations[1].branches[0].branch_id == "shared:0"
    assert ds.conversations[0].branches[0].child_conversation_ids == ["child_a"]
    assert ds.conversations[1].branches[0].child_conversation_ids == ["child_b"]


def test_conversation_branch_info_duplicate_child_conversation_ids_preserved_verbatim():
    """Duplicate child_conversation_ids are preserved (no implicit de-duplication) pre- and post-round-trip."""
    info = ConversationBranchInfo(
        branch_id="b",
        child_conversation_ids=["c", "c"],
        mode=ConversationBranchMode.SPAWN,
    )
    assert info.child_conversation_ids == ["c", "c"]

    restored = ConversationBranchInfo.model_validate_json(info.model_dump_json())

    assert restored.child_conversation_ids == ["c", "c"]
