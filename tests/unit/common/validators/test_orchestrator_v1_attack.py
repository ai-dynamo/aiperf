# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hostile-input unit tests for ``validate_for_orchestrator_v1``.

These tests construct ``DatasetMetadata`` programmatically — bypassing the
loaders' safety nets — to attack the v1 orchestrator validator directly.
Every rejection assertion pins the contract from ``CLAUDE.md``:

    "New validators must follow this shape" -> ``"<loc>: <reason>"``

i.e. each ``NotImplementedError`` (or ``ValueError`` for dup-prereq) must
embed both the offending conversation_id AND a turn index or branch_id so
authors can locate the offending construct without grepping.
"""

from __future__ import annotations

import time

import pytest

from aiperf.common.enums import ConversationBranchMode, PrerequisiteKind
from aiperf.common.models import (
    ConversationMetadata,
    DatasetMetadata,
    TurnMetadata,
    TurnPrerequisite,
)
from aiperf.common.models.branch import ConversationBranchInfo
from aiperf.common.validators.orchestrator_v1 import validate_for_orchestrator_v1
from aiperf.plugin.enums import DatasetSamplingStrategy

# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _dataset(*convs: ConversationMetadata) -> DatasetMetadata:
    return DatasetMetadata(
        conversations=list(convs),
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )


def _spawn_branch(
    branch_id: str,
    children: list[str],
    *,
    dispatch_timing: str = "post",
) -> ConversationBranchInfo:
    return ConversationBranchInfo(
        branch_id=branch_id,
        child_conversation_ids=children,
        mode=ConversationBranchMode.SPAWN,
        dispatch_timing=dispatch_timing,
    )


def _fork_branch(branch_id: str, children: list[str]) -> ConversationBranchInfo:
    return ConversationBranchInfo(
        branch_id=branch_id,
        child_conversation_ids=children,
        mode=ConversationBranchMode.FORK,
    )


def _turn(
    branch_ids: list[str] | None = None, prereqs: list[TurnPrerequisite] | None = None
) -> TurnMetadata:
    return TurnMetadata(
        branch_ids=list(branch_ids) if branch_ids else [],
        prerequisites=list(prereqs) if prereqs else [],
    )


def _conv(
    conv_id: str,
    turns: list[TurnMetadata],
    branches: list[ConversationBranchInfo] | None = None,
    *,
    is_root: bool = True,
    agent_depth: int = 0,
    parent_conversation_id: str | None = None,
) -> ConversationMetadata:
    return ConversationMetadata(
        conversation_id=conv_id,
        turns=turns,
        branches=list(branches) if branches else [],
        is_root=is_root,
        agent_depth=agent_depth,
        parent_conversation_id=parent_conversation_id,
    )


# ===========================================================================
# 1. Pre-session branch belt-and-suspenders
# ===========================================================================


def test_pre_session_root_depth0_turn0_accepts():
    """Canonical pre-session SPAWN: root, agent_depth=0, branch declared on
    turn 0. Should validate cleanly."""
    parent = _conv(
        "root",
        [_turn(branch_ids=["root:pre"])],
        branches=[_spawn_branch("root:pre", ["child"], dispatch_timing="pre")],
        is_root=True,
        agent_depth=0,
    )
    child = _conv(
        "child", [_turn()], is_root=False, agent_depth=1, parent_conversation_id="root"
    )
    validate_for_orchestrator_v1(_dataset(parent, child))


def test_pre_session_non_root_rejects_with_branch_id():
    """is_root=False is one belt: rejected even when agent_depth=0."""
    parent = _conv(
        "subroot",
        [_turn(branch_ids=["b:pre"])],
        branches=[_spawn_branch("b:pre", ["leaf"], dispatch_timing="pre")],
        is_root=False,
        agent_depth=0,
    )
    leaf = _conv(
        "leaf",
        [_turn()],
        is_root=False,
        agent_depth=2,
        parent_conversation_id="subroot",
    )
    with pytest.raises(NotImplementedError) as exc:
        validate_for_orchestrator_v1(_dataset(parent, leaf))
    msg = str(exc.value)
    assert "subroot" in msg
    assert "b:pre" in msg
    assert "is_root=False" in msg


def test_pre_session_agent_depth_positive_rejects_with_branch_id():
    """agent_depth > 0 is the other belt: rejected even when is_root=True."""
    parent = _conv(
        "deepish",
        [_turn(branch_ids=["b:pre"])],
        branches=[_spawn_branch("b:pre", ["leaf"], dispatch_timing="pre")],
        is_root=True,
        agent_depth=5,
    )
    leaf = _conv(
        "leaf",
        [_turn()],
        is_root=False,
        agent_depth=6,
        parent_conversation_id="deepish",
    )
    with pytest.raises(NotImplementedError) as exc:
        validate_for_orchestrator_v1(_dataset(parent, leaf))
    msg = str(exc.value)
    assert "deepish" in msg
    assert "b:pre" in msg
    assert "agent_depth=5" in msg


def test_pre_session_both_belts_fail_rejects():
    """Both belts wrong: error names the offending conv + branch."""
    parent = _conv(
        "bad",
        [_turn(branch_ids=["bad:pre"])],
        branches=[_spawn_branch("bad:pre", ["leaf"], dispatch_timing="pre")],
        is_root=False,
        agent_depth=5,
    )
    leaf = _conv(
        "leaf", [_turn()], is_root=False, agent_depth=6, parent_conversation_id="bad"
    )
    with pytest.raises(NotImplementedError, match=r"bad.*bad:pre"):
        validate_for_orchestrator_v1(_dataset(parent, leaf))


def test_pre_session_fork_dispatch_timing_rejects_at_validator():
    """``ConversationBranchInfo``'s field validator rejects FORK+pre at
    construction. The orchestrator-validator defense-in-depth path is
    therefore unreachable from a public constructor — pin the field-level
    rejection so the contract surface stays visible."""
    with pytest.raises(Exception) as exc:
        ConversationBranchInfo(
            branch_id="b",
            child_conversation_ids=["c"],
            mode=ConversationBranchMode.FORK,
            dispatch_timing="pre",
        )
    assert "pre" in str(exc.value).lower()


def test_pre_session_fork_bypass_via_model_construct_rejects():
    """Bypass the field validator with ``model_construct`` so we can hit the
    validator's FORK-pre defense-in-depth path."""
    bad = ConversationBranchInfo.model_construct(
        branch_id="x:pre",
        child_conversation_ids=["child"],
        mode=ConversationBranchMode.FORK,
        dispatch_timing="pre",
        background=False,
    )
    parent = _conv(
        "px",
        [_turn(branch_ids=["x:pre"])],
        branches=[bad],
    )
    child = _conv(
        "child", [_turn()], is_root=False, agent_depth=1, parent_conversation_id="px"
    )
    with pytest.raises(NotImplementedError) as exc:
        validate_for_orchestrator_v1(_dataset(parent, child))
    msg = str(exc.value)
    assert "px" in msg
    assert "x:pre" in msg
    assert "FORK" in msg or "SPAWN" in msg


def test_pre_session_branch_attached_to_turn_3_rejects_with_turn_index():
    """pre branch must be declared on turn 0; declaring it on turn 3 fails
    with the turn index in the message."""
    parent = _conv(
        "late",
        [
            _turn(),
            _turn(),
            _turn(),
            _turn(branch_ids=["late:pre"]),
        ],
        branches=[_spawn_branch("late:pre", ["child"], dispatch_timing="pre")],
    )
    child = _conv(
        "child", [_turn()], is_root=False, agent_depth=1, parent_conversation_id="late"
    )
    with pytest.raises(NotImplementedError) as exc:
        validate_for_orchestrator_v1(_dataset(parent, child))
    msg = str(exc.value)
    assert "late" in msg
    assert "late:pre" in msg
    assert "turn 3" in msg


def test_pre_session_branch_not_declared_on_any_turn_rejects():
    """A pre-session branch in ``branches`` that no turn references must
    fail with a 'not attached to any turn' style message."""
    parent = _conv(
        "orphan",
        [_turn(branch_ids=[]), _turn(branch_ids=[])],
        branches=[_spawn_branch("orphan:pre", ["child"], dispatch_timing="pre")],
    )
    child = _conv(
        "child",
        [_turn()],
        is_root=False,
        agent_depth=1,
        parent_conversation_id="orphan",
    )
    with pytest.raises(NotImplementedError) as exc:
        validate_for_orchestrator_v1(_dataset(parent, child))
    msg = str(exc.value)
    assert "orphan" in msg
    assert "orphan:pre" in msg


# ===========================================================================
# 2. Branch shape
# ===========================================================================


def test_branch_mode_outside_supported_set_rejects():
    """Bypass enum on a ``ConversationBranchInfo`` via ``model_construct`` to
    smuggle a 'future' mode value past the validator."""
    bad = ConversationBranchInfo.model_construct(
        branch_id="b0",
        child_conversation_ids=["c"],
        mode="loopback",  # not in _SUPPORTED_BRANCH_MODES
        dispatch_timing="post",
        background=False,
    )
    parent = _conv("p", [_turn(branch_ids=["b0"])], branches=[bad])
    child = _conv(
        "c", [_turn()], is_root=False, agent_depth=1, parent_conversation_id="p"
    )
    with pytest.raises(NotImplementedError) as exc:
        validate_for_orchestrator_v1(_dataset(parent, child))
    msg = str(exc.value)
    assert "p" in msg
    assert "b0" in msg
    assert "not supported" in msg


def test_branch_empty_child_list_accepts():
    """Empty ``child_conversation_ids`` is structurally legal — the
    validator only checks each entry resolves, so [] passes. Pin current
    behavior so any future tightening forces an explicit test update."""
    parent = _conv(
        "p",
        [_turn(branch_ids=["b0"])],
        branches=[_spawn_branch("b0", [])],
    )
    validate_for_orchestrator_v1(_dataset(parent))


def test_branch_child_id_unknown_rejects_naming_bad_id():
    """A child_conversation_id with no matching conversation must be
    rejected, naming the missing id."""
    parent = _conv(
        "p",
        [_turn(branch_ids=["b0"])],
        branches=[_spawn_branch("b0", ["ghost"])],
    )
    with pytest.raises(NotImplementedError) as exc:
        validate_for_orchestrator_v1(_dataset(parent))
    msg = str(exc.value)
    assert "p" in msg
    assert "b0" in msg
    assert "ghost" in msg


def test_branch_id_collision_same_turn_rejects_with_turn_index():
    """Two branches declared on the same turn with the same branch_id must
    fail with the turn index in the message."""
    branch_a = _spawn_branch("dup", ["c1"])
    branch_b = _spawn_branch("dup", ["c2"])
    parent = _conv(
        "p",
        [_turn(branch_ids=["dup", "dup"])],
        branches=[branch_a, branch_b],
    )
    c1 = _conv(
        "c1", [_turn()], is_root=False, agent_depth=1, parent_conversation_id="p"
    )
    c2 = _conv(
        "c2", [_turn()], is_root=False, agent_depth=1, parent_conversation_id="p"
    )
    with pytest.raises(NotImplementedError) as exc:
        validate_for_orchestrator_v1(_dataset(parent, c1, c2))
    msg = str(exc.value)
    assert "p" in msg
    assert "turn 0" in msg
    assert "dup" in msg


def test_branch_id_empty_string_accepts_currently():
    """An empty-string branch_id is structurally usable (validator only
    checks uniqueness per turn and resolution of children). Pin behavior."""
    parent = _conv(
        "p",
        [_turn(branch_ids=[""])],
        branches=[_spawn_branch("", ["c"])],
    )
    child = _conv(
        "c", [_turn()], is_root=False, agent_depth=1, parent_conversation_id="p"
    )
    validate_for_orchestrator_v1(_dataset(parent, child))


def test_branch_thousand_child_spawn_accepts():
    """Wide-fanout SPAWN: 1000 children must validate without issue."""
    children_ids = [f"c{i}" for i in range(1000)]
    parent = _conv(
        "wide",
        [_turn(branch_ids=["wide:0"])],
        branches=[_spawn_branch("wide:0", children_ids)],
    )
    children = [
        _conv(
            cid, [_turn()], is_root=False, agent_depth=1, parent_conversation_id="wide"
        )
        for cid in children_ids
    ]
    validate_for_orchestrator_v1(_dataset(parent, *children))


# ===========================================================================
# 3. TurnPrerequisite (SPAWN_JOIN)
# ===========================================================================


def test_prereq_branch_id_not_in_dataset_rejects():
    """SPAWN_JOIN referencing a non-existent branch_id must reject with the
    bad id and turn index."""
    prereq = TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="ghost")
    parent = _conv("p", [_turn(), _turn(prereqs=[prereq])])
    with pytest.raises(NotImplementedError) as exc:
        validate_for_orchestrator_v1(_dataset(parent))
    msg = str(exc.value)
    assert "p" in msg
    assert "turn 1" in msg
    assert "ghost" in msg


def test_prereq_on_turn_0_with_no_branches_rejects():
    """Turn 0 cannot have a SPAWN_JOIN prereq — nothing earlier could have
    spawned. Rejection must mention turn 0 and the missing branch."""
    prereq = TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="x")
    parent = _conv("p", [_turn(prereqs=[prereq])])
    with pytest.raises(NotImplementedError) as exc:
        validate_for_orchestrator_v1(_dataset(parent))
    msg = str(exc.value)
    assert "p" in msg
    assert "turn 0" in msg
    assert "x" in msg


def test_prereq_kind_timer_rejects_with_turn_loc():
    """Non-SPAWN_JOIN kinds must reject with the turn-level location."""
    prereq = TurnPrerequisite(kind=PrerequisiteKind.TIMER, timer_seconds=1.5)
    parent = _conv("p", [_turn(), _turn(prereqs=[prereq])])
    with pytest.raises(NotImplementedError) as exc:
        validate_for_orchestrator_v1(_dataset(parent))
    msg = str(exc.value)
    assert "p" in msg
    assert "turn 1" in msg
    assert "timer" in msg.lower() or "TIMER" in msg


def test_prereq_kind_external_event_rejects():
    prereq = TurnPrerequisite(kind=PrerequisiteKind.EXTERNAL_EVENT, event_name="go")
    parent = _conv("p", [_turn(), _turn(prereqs=[prereq])])
    with pytest.raises(NotImplementedError) as exc:
        validate_for_orchestrator_v1(_dataset(parent))
    msg = str(exc.value)
    assert "p" in msg
    assert "turn 1" in msg


def test_prereq_kind_barrier_rejects():
    prereq = TurnPrerequisite(kind=PrerequisiteKind.BARRIER, barrier_id="bar1")
    parent = _conv("p", [_turn(), _turn(prereqs=[prereq])])
    with pytest.raises(NotImplementedError) as exc:
        validate_for_orchestrator_v1(_dataset(parent))
    msg = str(exc.value)
    assert "p" in msg
    assert "turn 1" in msg


def test_prereq_no_branches_in_conv_rejects():
    """SPAWN_JOIN on a conversation that has no branches at all: the
    referenced branch_id cannot be found."""
    prereq = TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="noexist")
    parent = _conv("p", [_turn(), _turn(prereqs=[prereq])])
    with pytest.raises(NotImplementedError) as exc:
        validate_for_orchestrator_v1(_dataset(parent))
    msg = str(exc.value)
    assert "p" in msg
    assert "noexist" in msg


def test_prereq_pointing_at_fork_branch_accepts():
    """SPAWN_JOIN pointing at a FORK branch is currently accepted by the
    validator (no SPAWN-vs-FORK type check). Pin behavior so any future
    tightening is visible."""
    branch = _fork_branch("b0", ["c"])
    prereq = TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="b0")
    parent = _conv(
        "p",
        [_turn(branch_ids=["b0"]), _turn(prereqs=[prereq])],
        branches=[branch],
    )
    child = _conv(
        "c", [_turn()], is_root=False, agent_depth=1, parent_conversation_id="p"
    )
    validate_for_orchestrator_v1(_dataset(parent, child))


def test_prereq_pointing_at_pre_session_branch_rejects():
    """Pre-session (fire-and-forget) SPAWN branches cannot be SPAWN_JOIN
    targets."""
    branch = _spawn_branch("pre0", ["c"], dispatch_timing="pre")
    prereq = TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="pre0")
    parent = _conv(
        "p",
        [_turn(branch_ids=["pre0"]), _turn(prereqs=[prereq])],
        branches=[branch],
    )
    child = _conv(
        "c", [_turn()], is_root=False, agent_depth=1, parent_conversation_id="p"
    )
    with pytest.raises(NotImplementedError) as exc:
        validate_for_orchestrator_v1(_dataset(parent, child))
    msg = str(exc.value)
    assert "p" in msg
    assert "pre0" in msg
    assert "turn 1" in msg


def test_prereq_reserved_per_child_field_rejects():
    """Setting reserved ``child_conversation_ids`` on TurnPrerequisite must
    reject with turn loc."""
    prereq = TurnPrerequisite(
        kind=PrerequisiteKind.SPAWN_JOIN,
        branch_id="b0",
        child_conversation_ids=["c"],
    )
    parent = _conv(
        "p",
        [_turn(branch_ids=["b0"]), _turn(prereqs=[prereq])],
        branches=[_spawn_branch("b0", ["c"])],
    )
    child = _conv(
        "c", [_turn()], is_root=False, agent_depth=1, parent_conversation_id="p"
    )
    with pytest.raises(NotImplementedError) as exc:
        validate_for_orchestrator_v1(_dataset(parent, child))
    msg = str(exc.value)
    assert "p" in msg
    assert "turn 1" in msg


def test_prereq_same_turn_self_reference_rejects():
    """Prereq referencing a branch declared on the same turn (not strictly
    earlier) must reject."""
    branch = _spawn_branch("b0", ["c"])
    prereq = TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="b0")
    parent = _conv(
        "p",
        [_turn(branch_ids=["b0"], prereqs=[prereq])],
        branches=[branch],
    )
    child = _conv(
        "c", [_turn()], is_root=False, agent_depth=1, parent_conversation_id="p"
    )
    with pytest.raises(NotImplementedError) as exc:
        validate_for_orchestrator_v1(_dataset(parent, child))
    msg = str(exc.value)
    assert "p" in msg
    assert "b0" in msg
    assert "turn 0" in msg


def test_prereq_forward_reference_rejects():
    """Prereq on turn 0 referencing a branch declared on turn 2: forward
    reference, rejected."""
    branch = _spawn_branch("b0", ["c"])
    prereq = TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="b0")
    parent = _conv(
        "p",
        [
            _turn(prereqs=[prereq]),
            _turn(),
            _turn(branch_ids=["b0"]),
        ],
        branches=[branch],
    )
    child = _conv(
        "c", [_turn()], is_root=False, agent_depth=1, parent_conversation_id="p"
    )
    with pytest.raises(NotImplementedError) as exc:
        validate_for_orchestrator_v1(_dataset(parent, child))
    msg = str(exc.value)
    assert "p" in msg
    assert "b0" in msg
    assert "turn 0" in msg


def test_prereq_duplicate_branch_id_on_same_turn_rejects_valueerror():
    """Two SPAWN_JOIN prereqs on the same gated turn referencing the same
    branch_id: documented to raise ValueError, not NotImplementedError."""
    branch = _spawn_branch("b0", ["c"])
    p1 = TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="b0")
    p2 = TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="b0")
    parent = _conv(
        "p",
        [_turn(branch_ids=["b0"]), _turn(prereqs=[p1, p2])],
        branches=[branch],
    )
    child = _conv(
        "c", [_turn()], is_root=False, agent_depth=1, parent_conversation_id="p"
    )
    with pytest.raises(ValueError) as exc:
        validate_for_orchestrator_v1(_dataset(parent, child))
    msg = str(exc.value)
    assert "p" in msg
    assert "b0" in msg
    assert "turn 1" in msg


# ===========================================================================
# 4. ConversationMetadata edges
# ===========================================================================


def test_conv_with_branches_but_empty_turns_accepts_currently():
    """branches declared but turns=[]: branch_declaration_turn is empty.
    Validator currently allows this (branches are unreferenced); pin
    behavior."""
    parent = _conv(
        "p",
        turns=[],
        branches=[_spawn_branch("b0", ["c"])],
    )
    child = _conv(
        "c", [_turn()], is_root=False, agent_depth=1, parent_conversation_id="p"
    )
    validate_for_orchestrator_v1(_dataset(parent, child))


def test_conv_with_negative_agent_depth_accepts():
    """agent_depth=-1 is non-sensical but only matters in the pre-session
    branch path (where agent_depth > 0 is the failure mode). With no pre
    branch, the validator does not gate on it. Pin current behavior."""
    parent = _conv(
        "p",
        [_turn(branch_ids=["b0"])],
        branches=[_spawn_branch("b0", ["c"])],
        is_root=True,
        agent_depth=-1,
    )
    child = _conv(
        "c", [_turn()], is_root=False, agent_depth=0, parent_conversation_id="p"
    )
    validate_for_orchestrator_v1(_dataset(parent, child))


def test_conv_with_very_deep_agent_depth_accepts():
    """agent_depth=999 with no pre-session branch is structurally fine."""
    parent = _conv(
        "p",
        [_turn(branch_ids=["b0"])],
        branches=[_spawn_branch("b0", ["c"])],
        is_root=True,
        agent_depth=999,
    )
    child = _conv(
        "c", [_turn()], is_root=False, agent_depth=1000, parent_conversation_id="p"
    )
    validate_for_orchestrator_v1(_dataset(parent, child))


def test_conv_parent_self_reference_accepts_currently():
    """parent_conversation_id pointing at self is suspicious but the
    validator does not check parent topology. Pin behavior."""
    parent = _conv(
        "p",
        [_turn()],
        is_root=False,
        agent_depth=1,
        parent_conversation_id="p",
    )
    validate_for_orchestrator_v1(_dataset(parent))


def test_conv_duplicate_conversation_id_in_metadata_accepts_currently():
    """Same conversation_id appearing twice: validator does not assert
    uniqueness. Pin behavior so any future tightening is visible."""
    a = _conv("dup", [_turn()])
    b = _conv("dup", [_turn()])
    validate_for_orchestrator_v1(_dataset(a, b))


# ===========================================================================
# 5. DatasetMetadata edges
# ===========================================================================


def test_empty_dataset_accepts():
    """Degenerate empty dataset must validate cleanly."""
    validate_for_orchestrator_v1(_dataset())


def test_ten_thousand_conversations_validates_under_5_seconds():
    """10k conversations each with a SPAWN branch and one child: validator
    must complete inside 5 seconds (perf regression guard)."""
    convs: list[ConversationMetadata] = []
    for i in range(10_000):
        root_id = f"r{i}"
        child_id = f"r{i}c"
        convs.append(
            _conv(
                root_id,
                [_turn(branch_ids=[f"b{i}"])],
                branches=[_spawn_branch(f"b{i}", [child_id])],
            )
        )
        convs.append(
            _conv(
                child_id,
                [_turn()],
                is_root=False,
                agent_depth=1,
                parent_conversation_id=root_id,
            )
        )
    md = _dataset(*convs)
    t0 = time.perf_counter()
    validate_for_orchestrator_v1(md)
    elapsed = time.perf_counter() - t0
    assert elapsed < 5.0, f"validator took {elapsed:.2f}s on 10k conversations"


def test_mixed_fork_and_spawn_branches_same_turn_accepts():
    """A single turn declaring both a FORK and a SPAWN branch is legal."""
    parent = _conv(
        "p",
        [_turn(branch_ids=["f0", "s0"])],
        branches=[
            _fork_branch("f0", ["cf"]),
            _spawn_branch("s0", ["cs"]),
        ],
    )
    cf = _conv(
        "cf", [_turn()], is_root=False, agent_depth=1, parent_conversation_id="p"
    )
    cs = _conv(
        "cs", [_turn()], is_root=False, agent_depth=1, parent_conversation_id="p"
    )
    validate_for_orchestrator_v1(_dataset(parent, cf, cs))


def test_one_hundred_unique_branch_ids_single_turn_accepts():
    """100 distinct branch_ids on one turn must validate."""
    child_ids = [f"c{i}" for i in range(100)]
    branch_ids = [f"b{i}" for i in range(100)]
    parent = _conv(
        "p",
        [_turn(branch_ids=branch_ids)],
        branches=[
            _spawn_branch(bid, [cid])
            for bid, cid in zip(branch_ids, child_ids, strict=True)
        ],
    )
    children = [
        _conv(cid, [_turn()], is_root=False, agent_depth=1, parent_conversation_id="p")
        for cid in child_ids
    ]
    validate_for_orchestrator_v1(_dataset(parent, *children))


# ===========================================================================
# 6. Global FORK single-parent check
# ===========================================================================


def test_two_fork_parents_claiming_same_child_rejects():
    """Two different conversations FORK-claiming the same child must be
    rejected with both parent ids in the message."""
    p1 = _conv(
        "p1",
        [_turn(branch_ids=["p1:0"])],
        branches=[_fork_branch("p1:0", ["shared"])],
    )
    p2 = _conv(
        "p2",
        [_turn(branch_ids=["p2:0"])],
        branches=[_fork_branch("p2:0", ["shared"])],
    )
    shared = _conv("shared", [_turn()], is_root=False, agent_depth=1)
    with pytest.raises(NotImplementedError) as exc:
        validate_for_orchestrator_v1(_dataset(p1, p2, shared))
    msg = str(exc.value)
    assert "shared" in msg
    assert "p1" in msg
    assert "p2" in msg


def test_two_spawn_parents_claiming_same_child_accepts():
    """Two SPAWN-parents claiming the same child are NOT rejected — SPAWN
    children get fresh context so multi-parent is fine. Pin behavior."""
    p1 = _conv(
        "p1",
        [_turn(branch_ids=["p1:0"])],
        branches=[_spawn_branch("p1:0", ["shared"])],
    )
    p2 = _conv(
        "p2",
        [_turn(branch_ids=["p2:0"])],
        branches=[_spawn_branch("p2:0", ["shared"])],
    )
    shared = _conv("shared", [_turn()], is_root=False, agent_depth=1)
    validate_for_orchestrator_v1(_dataset(p1, p2, shared))
