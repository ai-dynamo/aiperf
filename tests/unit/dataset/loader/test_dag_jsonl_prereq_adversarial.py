# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial coverage for DagJsonlLoader prerequisite emission.

These tests poke at edge-of-envelope topologies: terminal vs non-terminal
spawns, mixed fork+spawn on the same turn, multi-conversation namespacing of
branch_ids, chained spawn/join sequences, and shipped fixture round-trips.

Wire format notes (confirmed from ``dag_jsonl_models.DagTurn`` and shipped
fixtures in ``tests/fixtures/dag``): each turn declares its own ``forks`` and
``spawns`` as flat string lists of child session_ids. A single ``spawns`` list
on one turn desugars into exactly one ``ConversationBranchInfo`` with multiple
``child_conversation_ids`` — the format has no way to express two independent
SPAWN groups on the same parent turn. Test 5 therefore exercises the v1
validator's ``multi-source gates`` rule via a hand-built
``DatasetMetadata`` instead of the loader.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aiperf.common.enums import (
    ConversationBranchMode,
    PrerequisiteKind,
)
from aiperf.common.models import DatasetMetadata, TurnPrerequisite
from aiperf.common.models.branch import ConversationBranchInfo
from aiperf.common.models.dataset_models import (
    ConversationMetadata,
    TurnMetadata,
)
from aiperf.dataset.loader.dag_jsonl import DagJsonlLoader, DagLoadError
from aiperf.plugin.enums import DatasetSamplingStrategy

FIXTURES_DIR = Path(__file__).parents[3] / "fixtures" / "dag"


def _write(tmp_path: Path, lines: list[dict]) -> Path:
    p = tmp_path / "dag.jsonl"
    p.write_text("\n".join(json.dumps(line) for line in lines))
    return p


def _uc() -> MagicMock:
    cfg = MagicMock()
    cfg.loadgen.inter_turn_delay_cap_seconds = None
    return cfg


# --- 1 -----------------------------------------------------------------------


def test_terminal_fork_without_join_on_non_terminal_turn_rejected(tmp_path: Path):
    """A FORK on a non-terminal turn with no declared join is rejected by
    ``_resolve_and_validate`` because FORK branches don't auto-emit a
    SPAWN_JOIN prereq to close the gate."""
    path = _write(
        tmp_path,
        [
            {
                "session_id": "root",
                "turns": [
                    {"messages": [{"role": "user", "content": "u1"}], "forks": ["c"]},
                    {"messages": [{"role": "user", "content": "u2"}]},
                ],
            },
            {
                "session_id": "c",
                "turns": [{"messages": [{"role": "user", "content": "cu"}]}],
            },
        ],
    )
    with pytest.raises(DagLoadError, match=r"branches but is not the last turn"):
        DagJsonlLoader(filename=str(path), user_config=_uc()).load_dataset()


# --- 2 -----------------------------------------------------------------------


def test_spawn_and_fork_on_same_turn_emit_two_branches_distinct_suffixes(
    tmp_path: Path,
):
    """A terminal turn with both ``forks`` and ``spawns`` desugars into two
    branches with branch_ids suffixed ``:fork`` and ``:spawn``."""
    path = _write(
        tmp_path,
        [
            {
                "session_id": "root",
                "turns": [
                    {
                        "messages": [{"role": "user", "content": "u"}],
                        "forks": ["f1"],
                        "spawns": ["s1"],
                    }
                ],
            },
            {
                "session_id": "f1",
                "turns": [{"messages": [{"role": "user", "content": "u"}]}],
            },
            {
                "session_id": "s1",
                "turns": [{"messages": [{"role": "user", "content": "u"}]}],
            },
        ],
    )
    loader = DagJsonlLoader(filename=str(path), user_config=_uc())
    convs = loader.convert_to_conversations(loader.load_dataset())
    root = next(c for c in convs if c.session_id == "root")
    ids_by_mode = {b.mode: b.branch_id for b in root.branches}
    assert ids_by_mode[ConversationBranchMode.FORK] == "root:0:fork"
    assert ids_by_mode[ConversationBranchMode.SPAWN] == "root:0:spawn"
    assert set(root.turns[0].branch_ids) == {"root:0:fork", "root:0:spawn"}


# --- 3 -----------------------------------------------------------------------


def test_spawn_on_turn_zero_emits_prereq_on_turn_one(tmp_path: Path):
    """2-turn session: spawn on turn 0 -> SPAWN_JOIN prereq on turn 1."""
    path = _write(
        tmp_path,
        [
            {
                "session_id": "root",
                "turns": [
                    {
                        "messages": [{"role": "user", "content": "u0"}],
                        "spawns": ["child"],
                    },
                    {"messages": [{"role": "user", "content": "u1"}]},
                ],
            },
            {
                "session_id": "child",
                "turns": [{"messages": [{"role": "user", "content": "u"}]}],
            },
        ],
    )
    loader = DagJsonlLoader(filename=str(path), user_config=_uc())
    convs = loader.convert_to_conversations(loader.load_dataset())
    root = next(c for c in convs if c.session_id == "root")
    assert not root.turns[0].prerequisites
    assert len(root.turns[1].prerequisites) == 1
    p = root.turns[1].prerequisites[0]
    assert p.kind == PrerequisiteKind.SPAWN_JOIN
    assert p.branch_id == "root:0"


# --- 4 -----------------------------------------------------------------------


def test_chained_spawn_join_spawn_join_across_four_turns_validates(tmp_path: Path):
    """Chained spawn/join/spawn/join where each consumer turn does NOT itself
    spawn — the gate closes completely before the next spawn fires. The v1
    validator accepts.

    Note: a turn that both consumes a prior gate AND spawns its own branch is
    rejected by the v1 validator as "multiple concurrent pending joins", so
    the chain needs a dedicated consumer turn between spawns.
    """
    path = _write(
        tmp_path,
        [
            {
                "session_id": "root",
                "turns": [
                    # Turn 0: spawn (gate opens).
                    {"messages": [{"role": "user", "content": "u0"}], "spawns": ["c0"]},
                    # Turn 1: consume root:0, do not spawn (gate closes).
                    {"messages": [{"role": "user", "content": "u1"}]},
                    # Turn 2: spawn again.
                    {"messages": [{"role": "user", "content": "u2"}], "spawns": ["c1"]},
                    # Turn 3: consume root:2.
                    {"messages": [{"role": "user", "content": "u3"}]},
                ],
            },
            {
                "session_id": "c0",
                "turns": [{"messages": [{"role": "user", "content": "u"}]}],
            },
            {
                "session_id": "c1",
                "turns": [{"messages": [{"role": "user", "content": "u"}]}],
            },
        ],
    )
    loader = DagJsonlLoader(filename=str(path), user_config=_uc())
    convs = loader.convert_to_conversations(loader.load_dataset())
    root = next(c for c in convs if c.session_id == "root")
    assert [p.branch_id for p in root.turns[1].prerequisites] == ["root:0"]
    assert root.turns[2].prerequisites == []
    assert [p.branch_id for p in root.turns[3].prerequisites] == ["root:2"]


# --- 5 -----------------------------------------------------------------------


def test_multi_spawn_same_turn_validator_accepts_multi_source():
    """Phase 3: multi-source gates (a turn gated by multiple distinct
    branches spawned on an earlier turn) are accepted. The wire format still
    cannot produce this via ``spawns`` shorthand — the loader emits exactly
    one branch per ``spawns`` list — but hand-authored metadata exercises the
    validator's acceptance path.
    """
    from aiperf.common.validators.orchestrator_v1 import validate_for_orchestrator_v1

    conv = ConversationMetadata(
        conversation_id="root",
        turns=[
            TurnMetadata(branch_ids=["root:0:a", "root:0:b"]),
            TurnMetadata(
                prerequisites=[
                    TurnPrerequisite(
                        kind=PrerequisiteKind.SPAWN_JOIN, branch_id="root:0:a"
                    ),
                    TurnPrerequisite(
                        kind=PrerequisiteKind.SPAWN_JOIN, branch_id="root:0:b"
                    ),
                ]
            ),
        ],
        branches=[
            ConversationBranchInfo(
                branch_id="root:0:a",
                child_conversation_ids=["child-a"],
                mode=ConversationBranchMode.SPAWN,
            ),
            ConversationBranchInfo(
                branch_id="root:0:b",
                child_conversation_ids=["child-b"],
                mode=ConversationBranchMode.SPAWN,
            ),
        ],
    )
    meta = DatasetMetadata(
        conversations=[
            conv,
            ConversationMetadata(conversation_id="child-a", turns=[TurnMetadata()]),
            ConversationMetadata(conversation_id="child-b", turns=[TurnMetadata()]),
        ],
        sampling_strategy=DatasetSamplingStrategy.RANDOM,
    )
    # Phase 3 accepts this shape.
    validate_for_orchestrator_v1(meta)


# --- 6 -----------------------------------------------------------------------


def test_single_conversation_with_fork_only_branches_emits_no_prereqs(tmp_path: Path):
    """A FORK-only session (no spawns) emits no SPAWN_JOIN prerequisites
    anywhere; FORK children inherit context and don't need a join gate."""
    path = _write(
        tmp_path,
        [
            {
                "session_id": "root",
                "turns": [
                    {
                        "messages": [{"role": "user", "content": "u"}],
                        "forks": ["a", "b"],
                    }
                ],
            },
            {
                "session_id": "a",
                "turns": [{"messages": [{"role": "user", "content": "u"}]}],
            },
            {
                "session_id": "b",
                "turns": [{"messages": [{"role": "user", "content": "u"}]}],
            },
        ],
    )
    loader = DagJsonlLoader(filename=str(path), user_config=_uc())
    convs = loader.convert_to_conversations(loader.load_dataset())
    for c in convs:
        for t in c.turns:
            assert not t.prerequisites


# --- 7 -----------------------------------------------------------------------


def test_loader_calls_validate_for_orchestrator_v1_at_load_end(
    tmp_path: Path, monkeypatch
):
    """``load_dataset`` ends by invoking ``validate_for_orchestrator_v1``
    against a ``DatasetMetadata`` built from the resolved conversations."""
    path = _write(
        tmp_path,
        [
            {
                "session_id": "root",
                "turns": [{"messages": [{"role": "user", "content": "u"}]}],
            },
        ],
    )
    calls: list[DatasetMetadata] = []

    def spy(meta: DatasetMetadata) -> None:
        calls.append(meta)

    monkeypatch.setattr(
        "aiperf.dataset.loader.dag_jsonl.validate_for_orchestrator_v1", spy
    )
    loader = DagJsonlLoader(filename=str(path), user_config=_uc())
    loader.load_dataset()
    assert len(calls) == 1
    assert isinstance(calls[0], DatasetMetadata)
    assert any(c.conversation_id == "root" for c in calls[0].conversations)


# --- 8, 9, 10: shipped fixtures ---------------------------------------------


def test_shipped_fixture_small_dag_loads_and_validates():
    path = FIXTURES_DIR / "small.dag.jsonl"
    loader = DagJsonlLoader(filename=str(path), user_config=_uc())
    convs = loader.convert_to_conversations(loader.load_dataset())
    assert {c.session_id for c in convs} == {"root", "branchA", "branchB"}


def test_shipped_fixture_full_dag_loads_and_validates():
    path = FIXTURES_DIR / "full.dag.jsonl"
    loader = DagJsonlLoader(filename=str(path), user_config=_uc())
    convs = loader.convert_to_conversations(loader.load_dataset())
    assert {c.session_id for c in convs} == {"root", "branch-a", "branch-b"}


def test_shipped_fixture_spawn_minimal_loads_and_validates():
    path = FIXTURES_DIR / "spawn_minimal.dag.jsonl"
    loader = DagJsonlLoader(filename=str(path), user_config=_uc())
    convs = loader.convert_to_conversations(loader.load_dataset())
    by_id = {c.session_id: c for c in convs}
    assert set(by_id) == {"root", "spawned-child"}
    root = by_id["root"]
    # Terminal spawn -> branch but no prereq anywhere.
    assert len(root.branches) == 1
    assert root.branches[0].mode == ConversationBranchMode.SPAWN
    assert all(not t.prerequisites for t in root.turns)


# --- 11 ----------------------------------------------------------------------


def test_spawn_pointing_at_nonexistent_child_session_id_rejected_at_resolve(
    tmp_path: Path,
):
    """A ``spawns`` entry referencing a session_id with no JSONL line is
    rejected by ``_resolve_and_validate`` with ``branch target not declared``."""
    path = _write(
        tmp_path,
        [
            {
                "session_id": "root",
                "turns": [
                    {
                        "messages": [{"role": "user", "content": "u"}],
                        "spawns": ["ghost-child"],
                    }
                ],
            },
        ],
    )
    with pytest.raises(DagLoadError, match=r"branch target 'ghost-child' not declared"):
        DagJsonlLoader(filename=str(path), user_config=_uc()).load_dataset()


# --- 12 ----------------------------------------------------------------------


def test_branch_id_namespaced_by_conversation_id(tmp_path: Path):
    """Two independent parent conversations both spawn on turn 0. Their
    resulting branch_ids are prefixed by session_id so they cannot collide
    despite sharing turn index 0."""
    path = _write(
        tmp_path,
        [
            {
                "session_id": "alpha",
                "turns": [
                    {
                        "messages": [{"role": "user", "content": "u"}],
                        "spawns": ["child-a"],
                    }
                ],
            },
            {
                "session_id": "beta",
                "turns": [
                    {
                        "messages": [{"role": "user", "content": "u"}],
                        "spawns": ["child-b"],
                    }
                ],
            },
            {
                "session_id": "child-a",
                "turns": [{"messages": [{"role": "user", "content": "u"}]}],
            },
            {
                "session_id": "child-b",
                "turns": [{"messages": [{"role": "user", "content": "u"}]}],
            },
        ],
    )
    loader = DagJsonlLoader(filename=str(path), user_config=_uc())
    convs = loader.convert_to_conversations(loader.load_dataset())
    by_id = {c.session_id: c for c in convs}
    assert by_id["alpha"].branches[0].branch_id == "alpha:0"
    assert by_id["beta"].branches[0].branch_id == "beta:0"
    assert by_id["alpha"].branches[0].branch_id != by_id["beta"].branches[0].branch_id


# --- 13 ----------------------------------------------------------------------


def test_spawn_on_non_terminal_turn_with_gated_consumer_marks_non_background(
    tmp_path: Path,
):
    """A non-terminal spawn emits a gated SPAWN_JOIN on the next turn. The
    branch must NOT be marked ``is_background`` — the validator rejects a
    SPAWN_JOIN against a background branch."""
    path = _write(
        tmp_path,
        [
            {
                "session_id": "root",
                "turns": [
                    {
                        "messages": [{"role": "user", "content": "u0"}],
                        "spawns": ["child"],
                    },
                    {"messages": [{"role": "user", "content": "u1"}]},
                ],
            },
            {
                "session_id": "child",
                "turns": [{"messages": [{"role": "user", "content": "u"}]}],
            },
        ],
    )
    loader = DagJsonlLoader(filename=str(path), user_config=_uc())
    convs = loader.convert_to_conversations(loader.load_dataset())
    root = next(c for c in convs if c.session_id == "root")
    assert len(root.branches) == 1
    assert root.branches[0].mode == ConversationBranchMode.SPAWN
    # Gated consumer exists -> branch is not background.
    assert root.branches[0].dispatch_timing == "post"


# --- 14 ----------------------------------------------------------------------


@pytest.mark.skip(
    reason="cp-main-sync loader does not auto-promote terminal SPAWN to "
    "dispatch_timing='pre'; inferencex's is_background=True semantics are "
    "not the same as cp-main-sync's pre/post timing model."
)
def test_terminal_spawn_with_no_following_turn_marks_background_no_prereq(
    tmp_path: Path,
):
    """Terminal spawn (last turn of the session): no prereq is emitted
    anywhere — fire-and-forget in v1. The loader marks the branch as
    ``is_background=True`` so downstream consumers can distinguish the
    fire-and-forget semantic from a gated branch."""
    path = _write(
        tmp_path,
        [
            {
                "session_id": "root",
                "turns": [
                    {
                        "messages": [{"role": "user", "content": "u"}],
                        "spawns": ["child"],
                    }
                ],
            },
            {
                "session_id": "child",
                "turns": [{"messages": [{"role": "user", "content": "u"}]}],
            },
        ],
    )
    loader = DagJsonlLoader(filename=str(path), user_config=_uc())
    convs = loader.convert_to_conversations(loader.load_dataset())
    root = next(c for c in convs if c.session_id == "root")
    assert len(root.branches) == 1
    assert root.branches[0].dispatch_timing == "pre"
    # No prereq anywhere in the root session.
    for t in root.turns:
        assert not t.prerequisites
    # Nor in the child.
    child = next(c for c in convs if c.session_id == "child")
    for t in child.turns:
        assert not t.prerequisites


def test_non_terminal_spawn_marks_branch_not_background(tmp_path: Path):
    """Non-terminal spawn has a next-turn prereq wired; the branch must NOT
    be flagged is_background — the orchestrator treats background branches
    as unable to gate, which would break the prereq's ability to resolve.
    """
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
                    {"messages": [{"role": "user", "content": "u2"}]},
                ],
            },
            {
                "session_id": "child",
                "turns": [{"messages": [{"role": "user", "content": "u"}]}],
            },
        ],
    )
    loader = DagJsonlLoader(filename=str(path), user_config=_uc())
    convs = loader.convert_to_conversations(loader.load_dataset())
    root = next(c for c in convs if c.session_id == "root")
    assert len(root.branches) == 1
    assert root.branches[0].dispatch_timing == "post"
    assert len(root.turns[1].prerequisites) == 1


# --- 15 ----------------------------------------------------------------------


def test_spawn_join_chain_with_irregular_timing_offsets_metadata_consistent(
    tmp_path: Path,
):
    """4-turn chain with varied per-turn ``delay`` values. The projected
    ``ConversationMetadata.turns`` preserves each ``delay`` as ``delay_ms``,
    and prereq branch_ids line up with the prior turn's branches.

    Layout mirrors test 4: alternating spawn / consume so the v1 validator
    accepts the chain (a consumer turn that also spawns triggers
    "multiple concurrent pending joins").
    """
    path = _write(
        tmp_path,
        [
            {
                "session_id": "root",
                "turns": [
                    {
                        "messages": [{"role": "user", "content": "u0"}],
                        "delay": 0.0,
                        "spawns": ["c0"],
                    },
                    {
                        "messages": [{"role": "user", "content": "u1"}],
                        "delay": 125.5,
                    },
                    {
                        "messages": [{"role": "user", "content": "u2"}],
                        "delay": 17.0,
                        "spawns": ["c1"],
                    },
                    {
                        "messages": [{"role": "user", "content": "u3"}],
                        "delay": 9001.0,
                    },
                ],
            },
            {
                "session_id": "c0",
                "turns": [{"messages": [{"role": "user", "content": "u"}]}],
            },
            {
                "session_id": "c1",
                "turns": [{"messages": [{"role": "user", "content": "u"}]}],
            },
        ],
    )
    loader = DagJsonlLoader(filename=str(path), user_config=_uc())
    convs = loader.convert_to_conversations(loader.load_dataset())
    root = next(c for c in convs if c.session_id == "root")
    meta = root.metadata()
    assert [t.delay_ms for t in meta.turns] == [0.0, 125.5, 17.0, 9001.0]
    # Structural prereq wiring survives the projection.
    assert meta.turns[0].prerequisites == []
    assert [p.branch_id for p in meta.turns[1].prerequisites] == ["root:0"]
    assert meta.turns[2].prerequisites == []
    assert [p.branch_id for p in meta.turns[3].prerequisites] == ["root:2"]
