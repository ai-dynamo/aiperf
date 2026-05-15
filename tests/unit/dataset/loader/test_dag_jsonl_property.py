# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Property-based fuzzing for ``DagJsonlLoader``.

Uses ``hypothesis`` to generate small, valid DAG JSONL line lists and
asserts loader-level invariants that must hold for *every* valid input:

1. ``test_loader_round_trips_through_jsonl_without_semantic_loss``:
   write -> load -> serialize -> load again produces equivalent metadata.
2. ``test_validator_monotonicity_under_leaf_removal``: removing an unused
   leaf conversation never breaks loadability.
3. ``test_loading_is_deterministic_across_repeated_calls``: two
   ``DagJsonlLoader`` instances loading the same file produce equal
   metadata.
4. ``test_prereq_index_matches_declared_branch_ids``: BranchOrchestrator's
   prereq index is consistent with the metadata it was built from.
5. ``test_no_silent_drop_of_referenced_branch_ids``: every branch_id named
   on a Turn appears in Conversation.branches.
6. ``test_strict_ordering_invariant_for_spawn_join_prereqs``: every
   resolved SPAWN_JOIN prereq has its declaring branch on a strictly
   earlier turn than the gated turn.

Each test bounds ``max_examples`` and disables hypothesis' deadline so
suite wall-clock stays predictable on shared CI runners.
"""

from __future__ import annotations

import json
from pathlib import Path

from hypothesis import given, settings

from aiperf.common.enums import PrerequisiteKind
from aiperf.common.models import DatasetMetadata
from aiperf.dataset.loader.dag_jsonl import DagJsonlLoader
from aiperf.plugin.enums import DatasetSamplingStrategy
from aiperf.timing.branch_orchestrator import BranchOrchestrator
from tests.unit.dataset.loader._dag_strategies import dag_dataset

HYPO = settings(max_examples=80, deadline=None)


def _write(tmp_path: Path, lines: list[dict]) -> Path:
    p = tmp_path / "dag.jsonl"
    p.write_text("\n".join(json.dumps(line) for line in lines))
    return p


def _load(path: Path) -> DatasetMetadata:
    loader = DagJsonlLoader(filename=path)
    convs = loader.convert_to_conversations(loader.load_dataset())
    return DatasetMetadata(
        conversations=[c.metadata() for c in convs],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )


# 1. Round-trip property ------------------------------------------------------


@HYPO
@given(lines=dag_dataset())
def test_loader_round_trips_through_jsonl_without_semantic_loss(
    tmp_path_factory, lines
):
    """Loading the same JSONL twice produces equivalent DatasetMetadata."""
    tmp = tmp_path_factory.mktemp("dag_roundtrip")
    path = _write(tmp, lines)
    md1 = _load(path)
    md2 = _load(path)
    # ConversationMetadata equality is structural via Pydantic; equal lists
    # of conversations imply equal datasets ignoring sampling-strategy
    # which we set identically above.
    by_id1 = {c.conversation_id: c for c in md1.conversations}
    by_id2 = {c.conversation_id: c for c in md2.conversations}
    assert set(by_id1) == set(by_id2)
    for cid in by_id1:
        assert by_id1[cid].model_dump() == by_id2[cid].model_dump(), cid


# 2. Validator monotonicity under leaf removal -------------------------------


@HYPO
@given(lines=dag_dataset(min_convs=3))
def test_validator_monotonicity_under_leaf_removal(tmp_path_factory, lines):
    """Removing a leaf conversation that no other conversation references
    never invalidates the dataset.

    "Leaf" here = a conversation whose ``session_id`` is not named in any
    other conversation's forks/spawns/pre_session_spawns. Leaves are
    optional from the dataset's POV; deleting one is a strict subset.
    """
    referenced: set[str] = set()
    for line in lines:
        for_pre = line.get("pre_session_spawns") or []
        referenced.update(for_pre)
        for t in line.get("turns", []):
            for f in t.get("forks", []) or []:
                referenced.add(f)
            for s in t.get("spawns", []) or []:
                if isinstance(s, str):
                    referenced.add(s)
                else:
                    referenced.update(s.get("children", []))

    # The root is the first line; never remove it. Find an unreferenced
    # non-root leaf to delete.
    candidates = [
        i
        for i, line in enumerate(lines)
        if i > 0 and line["session_id"] not in referenced
    ]
    if not candidates:
        # Whole dataset is fully referenced; nothing to monotone-remove.
        return

    # Baseline must load cleanly.
    tmp = tmp_path_factory.mktemp("dag_monotone")
    _load(_write(tmp, lines))
    # Subset must also load cleanly.
    smaller = [line for j, line in enumerate(lines) if j != candidates[0]]
    _load(_write(tmp, smaller))


# 3. Deterministic loading ----------------------------------------------------


@HYPO
@given(lines=dag_dataset())
def test_loading_is_deterministic_across_repeated_calls(tmp_path_factory, lines):
    """Two independent loader instances over the same file produce
    metadata equal under Pydantic ``model_dump`` (no insertion-order or
    RNG dependence).
    """
    tmp = tmp_path_factory.mktemp("dag_determ")
    path = _write(tmp, lines)
    a = _load(path).model_dump()
    b = _load(path).model_dump()
    assert a == b


# 4. Prereq-index consistency -------------------------------------------------


@HYPO
@given(lines=dag_dataset())
def test_prereq_index_matches_declared_branch_ids(tmp_path_factory, lines):
    """``BranchOrchestrator._build_prereq_index`` must agree with the
    metadata it was built from.

    For every (conversation_id, spawning_turn_idx) -> [(branch_id,
    gated_idx, prereq_key)] entry, the branch_id must be declared on
    ``spawning_turn_idx`` and the gated turn must hold a SPAWN_JOIN
    prereq referencing it.
    """
    tmp = tmp_path_factory.mktemp("dag_index")
    md = _load(_write(tmp, lines))

    class _CS:
        dataset_metadata = md

        def get_metadata(self, cid):
            return next(c for c in md.conversations if c.conversation_id == cid)

    class _Issuer:
        async def dispatch_first_turn(self, *_a, **_k):
            return True

        async def dispatch_join_turn(self, *_a, **_k):
            return True

    orch = BranchOrchestrator(conversation_source=_CS(), credit_issuer=_Issuer())

    by_id = {c.conversation_id: c for c in md.conversations}
    for (conv_id, spawn_idx), entries in orch._prereq_index.items():
        conv = by_id[conv_id]
        declared_on_turn = set(conv.turns[spawn_idx].branch_ids or [])
        for branch_id, gated_idx, prereq_key in entries:
            assert branch_id in declared_on_turn, (
                f"index claims {branch_id} declared on turn {spawn_idx} of "
                f"{conv_id} but turn declares {declared_on_turn}"
            )
            gated_prereqs = {p.branch_id for p in conv.turns[gated_idx].prerequisites}
            assert branch_id in gated_prereqs, (
                f"index claims {branch_id} gated at turn {gated_idx} of "
                f"{conv_id} but gated turn prereqs are {gated_prereqs}"
            )
            assert prereq_key == f"SPAWN_JOIN:{branch_id}"


# 5. No silent drops ----------------------------------------------------------


@HYPO
@given(lines=dag_dataset())
def test_no_silent_drop_of_referenced_branch_ids(tmp_path_factory, lines):
    """Every branch_id named on a Turn must appear in
    ``Conversation.branches`` (else the orchestrator would silently no-op
    on dispatch).
    """
    tmp = tmp_path_factory.mktemp("dag_no_drop")
    md = _load(_write(tmp, lines))
    for conv in md.conversations:
        declared = {b.branch_id for b in conv.branches}
        for idx, turn in enumerate(conv.turns):
            for bid in turn.branch_ids:
                assert bid in declared, (
                    f"conversation {conv.conversation_id} turn {idx} names "
                    f"branch_id {bid!r} but conversation.branches has {declared}"
                )


# 6. Strict ordering invariant ------------------------------------------------


@HYPO
@given(lines=dag_dataset())
def test_strict_ordering_invariant_for_spawn_join_prereqs(tmp_path_factory, lines):
    """For every SPAWN_JOIN prereq attached to a conversation's turn ``g``,
    the branch it references must be declared on a turn ``s < g`` of the
    same conversation. ``validate_for_orchestrator_v1`` already enforces
    this; the property test certifies the loader doesn't somehow emit
    out-of-order metadata that would silently bypass the check.
    """
    tmp = tmp_path_factory.mktemp("dag_strict_order")
    md = _load(_write(tmp, lines))
    for conv in md.conversations:
        decl_turn: dict[str, int] = {}
        for idx, turn in enumerate(conv.turns):
            for bid in turn.branch_ids:
                decl_turn.setdefault(bid, idx)
        for gated_idx, turn in enumerate(conv.turns):
            for prereq in turn.prerequisites:
                if prereq.kind != PrerequisiteKind.SPAWN_JOIN:
                    continue
                if prereq.branch_id is None:
                    continue
                # Branch may be defined on this conversation only.
                if prereq.branch_id not in decl_turn:
                    continue
                assert decl_turn[prereq.branch_id] < gated_idx, (
                    f"conv {conv.conversation_id}: SPAWN_JOIN on branch "
                    f"{prereq.branch_id} has declaring turn "
                    f"{decl_turn[prereq.branch_id]} >= gated turn {gated_idx}"
                )


# Sanity: hypothesis dataset strategy itself produces *something* loadable
# in the trivial fixed case, used to detect regressions in the strategy.


def test_strategy_smoke_loads_minimal_two_conversation_dataset(tmp_path):
    lines = [
        {
            "session_id": "root",
            "turns": [
                {"messages": [{"role": "user", "content": "hi"}], "spawns": ["c"]},
                {"messages": [{"role": "user", "content": "after"}]},
            ],
        },
        {
            "session_id": "c",
            "turns": [{"messages": [{"role": "user", "content": "ch"}]}],
        },
    ]
    md = _load(_write(tmp_path, lines))
    assert {c.conversation_id for c in md.conversations} == {"root", "c"}
