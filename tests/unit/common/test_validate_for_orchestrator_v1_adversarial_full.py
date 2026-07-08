# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Full adversarial coverage for ``validate_for_orchestrator_v1``.

Bypasses the loader and constructs ``DatasetMetadata`` directly to exercise
edge-of-envelope shapes the loader's shorthand cannot author:

- Every Phase 2b ``dispatch_timing`` rejection path (combined with FORK,
  blocking SPAWN, non-root, non-turn-0).
- Programmatic-bypass of the ``TurnPrerequisite`` reserved fields.
- ``PrerequisiteKind`` other than SPAWN_JOIN.
- Branch ``mode`` outside FORK/SPAWN.
- Multi-source / multi-consumer Phase 3 acceptance regressions.
- Strictly-prior boundary values (N vs N+1 vs <N).
- FORK multi-parent pattern at the validator (currently NOT enforced
  globally — documented).
- Background-not-gated rule.
- child_conversation_ids referencing non-existent conversations.
- Duplicate branch_ids on the same turn.
- Self-spawn / cyclic spawn graphs (would recurse without bound at replay).
- Duplicate ConversationBranchInfo descriptors sharing one branch_id.
- Dangling turn branch_ids with no matching branch descriptor.
- Empty dataset graceful handling.
- JSON round-trip idempotency with validation.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from pytest import param

from aiperf.common.enums import (
    ConversationBranchMode,
    PrerequisiteKind,
)
from aiperf.common.models import (
    ConversationBranchInfo,
    ConversationMetadata,
    DatasetMetadata,
    TurnMetadata,
    TurnPrerequisite,
)
from aiperf.common.validators.orchestrator_v1 import validate_for_orchestrator_v1
from aiperf.plugin.enums import DatasetSamplingStrategy


def _md(
    branches: list[ConversationBranchInfo] | None = None,
    turns: list[TurnMetadata] | None = None,
    *,
    agent_depth: int = 0,
    extra_conversations: list[ConversationMetadata] | None = None,
) -> DatasetMetadata:
    branches = branches or []
    turns = turns or [TurnMetadata()]
    child_ids: set[str] = set()
    for b in branches:
        child_ids.update(b.child_conversation_ids)
    children = [
        ConversationMetadata(conversation_id=cid, turns=[TurnMetadata()])
        for cid in sorted(child_ids)
    ]
    return DatasetMetadata(
        conversations=[
            ConversationMetadata(
                conversation_id="r",
                turns=turns,
                branches=branches,
                agent_depth=agent_depth,
            ),
            *children,
            *(extra_conversations or []),
        ],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )


# ---------------------------------------------------------------------------
# 21. dispatch_timing="pre" combined with each invalid mode
# ---------------------------------------------------------------------------


def test_pre_dispatch_with_fork_rejected():
    # v2 moved this rejection to construction-time: the ConversationBranchInfo
    # field validator rejects FORK + dispatch_timing="pre" before the dataset
    # validator ever runs (stricter, not a gap).
    with pytest.raises(ValidationError, match="reserved for SPAWN"):
        ConversationBranchInfo(
            branch_id="r:pre",
            child_conversation_ids=["c"],
            mode=ConversationBranchMode.FORK,
            dispatch_timing="pre",
        )


@pytest.mark.xfail(
    strict=True,
    reason="PORT DEVIATION: v2 _check_pre_session_branch DROPPED the "
    "'pre-session dispatch requires is_background=True' check. v2 re-keyed "
    "fire-and-forget gating off is_background onto dispatch_timing='pre' to "
    "support main's dag_jsonl background-forks, so a blocking (is_background="
    "False) pre-session SPAWN now validates with no equivalent rejection.",
)
def test_pre_dispatch_with_blocking_spawn_rejected():
    """v2 PORT DEVIATION: agentx rejected a blocking (is_background=False)
    pre-session SPAWN; v2's _check_pre_session_branch DROPPED that check
    (fire-and-forget gating was re-keyed off ``is_background`` onto
    ``dispatch_timing='pre'``). A blocking pre-session SPAWN now validates,
    so the agentx rejection no longer fires. The fire-and-forget INTENT is
    instead asserted by the SPAWN_JOIN-gating tests; here we pin the dropped
    behavior as a strict xfail.
    """
    branch = ConversationBranchInfo(
        branch_id="r:pre",
        child_conversation_ids=["c"],
        mode=ConversationBranchMode.SPAWN,
        background=False,  # blocking
        dispatch_timing="pre",
    )
    md = _md([branch], [TurnMetadata(branch_ids=["r:pre"]), TurnMetadata()])
    with pytest.raises(
        NotImplementedError, match="pre-session dispatch requires is_background=True"
    ):
        validate_for_orchestrator_v1(md)


def test_pre_dispatch_background_spawn_on_non_root_rejected():
    branch = ConversationBranchInfo(
        branch_id="r:pre",
        child_conversation_ids=["c"],
        mode=ConversationBranchMode.SPAWN,
        background=True,
        dispatch_timing="pre",
    )
    md = _md(
        [branch],
        [TurnMetadata(branch_ids=["r:pre"]), TurnMetadata()],
        agent_depth=1,
    )
    with pytest.raises(NotImplementedError, match="requires a root conversation"):
        validate_for_orchestrator_v1(md)


def test_pre_dispatch_background_spawn_on_non_turn_0_rejected():
    branch = ConversationBranchInfo(
        branch_id="r:pre",
        child_conversation_ids=["c"],
        mode=ConversationBranchMode.SPAWN,
        background=True,
        dispatch_timing="pre",
    )
    # Branch declared on turn 1, not turn 0 — rejected.
    md = _md(
        [branch],
        [
            TurnMetadata(),
            TurnMetadata(branch_ids=["r:pre"]),
            TurnMetadata(),
        ],
    )
    with pytest.raises(NotImplementedError, match="must be declared on turn 0"):
        validate_for_orchestrator_v1(md)


def test_pre_dispatch_background_spawn_valid_shape_accepted():
    branch = ConversationBranchInfo(
        branch_id="r:pre",
        child_conversation_ids=["c"],
        mode=ConversationBranchMode.SPAWN,
        background=True,
        dispatch_timing="pre",
    )
    md = _md([branch], [TurnMetadata(branch_ids=["r:pre"]), TurnMetadata()])
    validate_for_orchestrator_v1(md)


# ---------------------------------------------------------------------------
# 22. Invalid Literal value for dispatch_timing
# ---------------------------------------------------------------------------


def test_dispatch_timing_invalid_literal_pydantic_rejects():
    """Pydantic enforces the ``Literal["pre", "post"]`` type."""
    with pytest.raises(ValidationError):
        ConversationBranchInfo(
            branch_id="r:0",
            child_conversation_ids=["c"],
            mode=ConversationBranchMode.SPAWN,
            dispatch_timing="middle",  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# 23. Reserved TurnPrerequisite fields snuck through
# ---------------------------------------------------------------------------


def _ok_branch(branch_id: str = "r:0", child: str = "c") -> ConversationBranchInfo:
    return ConversationBranchInfo(
        branch_id=branch_id,
        child_conversation_ids=[child],
        mode=ConversationBranchMode.SPAWN,
    )


def test_validator_rejects_barrier_id_field():
    p = TurnPrerequisite(
        kind=PrerequisiteKind.SPAWN_JOIN, branch_id="r:0", barrier_id="b1"
    )
    md = _md(
        [_ok_branch()],
        [TurnMetadata(branch_ids=["r:0"]), TurnMetadata(prerequisites=[p])],
    )
    with pytest.raises(NotImplementedError, match="barrier"):
        validate_for_orchestrator_v1(md)


def test_validator_rejects_timer_seconds_field():
    p = TurnPrerequisite(
        kind=PrerequisiteKind.SPAWN_JOIN, branch_id="r:0", timer_seconds=2.5
    )
    md = _md(
        [_ok_branch()],
        [TurnMetadata(branch_ids=["r:0"]), TurnMetadata(prerequisites=[p])],
    )
    with pytest.raises(NotImplementedError, match="timer"):
        validate_for_orchestrator_v1(md)


def test_validator_rejects_event_name_field():
    p = TurnPrerequisite(
        kind=PrerequisiteKind.SPAWN_JOIN, branch_id="r:0", event_name="ready"
    )
    md = _md(
        [_ok_branch()],
        [TurnMetadata(branch_ids=["r:0"]), TurnMetadata(prerequisites=[p])],
    )
    with pytest.raises(NotImplementedError, match="event"):
        validate_for_orchestrator_v1(md)


def test_validator_rejects_child_conversation_ids_field():
    p = TurnPrerequisite(
        kind=PrerequisiteKind.SPAWN_JOIN,
        branch_id="r:0",
        child_conversation_ids=["c"],
    )
    md = _md(
        [_ok_branch()],
        [TurnMetadata(branch_ids=["r:0"]), TurnMetadata(prerequisites=[p])],
    )
    with pytest.raises(NotImplementedError, match="per-child"):
        validate_for_orchestrator_v1(md)


# ---------------------------------------------------------------------------
# 24. PrerequisiteKind other than SPAWN_JOIN
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind",
    [param(k, id=k.name) for k in PrerequisiteKind if k != PrerequisiteKind.SPAWN_JOIN],
)  # fmt: skip
def test_validator_rejects_non_spawn_join_kinds(kind: PrerequisiteKind):
    p = TurnPrerequisite(kind=kind, branch_id="r:0")
    md = _md(
        [_ok_branch()],
        [TurnMetadata(branch_ids=["r:0"]), TurnMetadata(prerequisites=[p])],
    )
    with pytest.raises(NotImplementedError, match="not supported by v1 orchestrator"):
        validate_for_orchestrator_v1(md)


# ---------------------------------------------------------------------------
# 25-26. branch_id none / empty / whitespace / unicode
# ---------------------------------------------------------------------------


def test_validator_rejects_none_branch_id():
    p = TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id=None)
    md = _md(
        [_ok_branch()],
        [TurnMetadata(branch_ids=["r:0"]), TurnMetadata(prerequisites=[p])],
    )
    with pytest.raises(NotImplementedError, match="does not reference a prior branch"):
        validate_for_orchestrator_v1(md)


def test_validator_rejects_unresolved_branch_id_string_variants():
    """Empty, whitespace-only, and bogus branch_id strings are all rejected
    because none resolve against the conversation's branches_by_id."""
    for bid in ["", " ", "   ", "\t", "no_such_branch", "r:99"]:
        p = TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id=bid)
        md = _md(
            [_ok_branch()],
            [TurnMetadata(branch_ids=["r:0"]), TurnMetadata(prerequisites=[p])],
        )
        with pytest.raises(
            NotImplementedError, match="does not reference a prior branch"
        ):
            validate_for_orchestrator_v1(md)


def test_validator_accepts_unicode_branch_id_when_resolved():
    """A unicode branch_id is accepted when the branch and prereq agree."""
    branch = ConversationBranchInfo(
        branch_id="ブランチ:0",
        child_conversation_ids=["c"],
        mode=ConversationBranchMode.SPAWN,
    )
    p = TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="ブランチ:0")
    md = _md(
        [branch],
        [TurnMetadata(branch_ids=["ブランチ:0"]), TurnMetadata(prerequisites=[p])],
    )
    validate_for_orchestrator_v1(md)


# ---------------------------------------------------------------------------
# 27. Invalid mode
# ---------------------------------------------------------------------------


def test_invalid_branch_mode_pydantic_rejects():
    """Branch ``mode`` outside the enum is rejected at model construction."""
    with pytest.raises(ValidationError):
        ConversationBranchInfo(
            branch_id="r:0",
            child_conversation_ids=["c"],
            mode="DIAMOND",  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# 28-29. Phase 3 acceptance regressions
# ---------------------------------------------------------------------------


def test_two_spawn_join_prereqs_on_one_turn_phase3_accepted():
    """Multi-source gate: one turn with two SPAWN_JOIN prereqs from
    different branches is accepted post-Phase-3."""
    b0 = ConversationBranchInfo(
        branch_id="r:0",
        child_conversation_ids=["c0"],
        mode=ConversationBranchMode.SPAWN,
    )
    b1 = ConversationBranchInfo(
        branch_id="r:1",
        child_conversation_ids=["c1"],
        mode=ConversationBranchMode.SPAWN,
    )
    md = _md(
        [b0, b1],
        [
            TurnMetadata(branch_ids=["r:0"]),
            TurnMetadata(branch_ids=["r:1"]),
            TurnMetadata(
                prerequisites=[
                    TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="r:0"),
                    TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="r:1"),
                ]
            ),
        ],
    )
    validate_for_orchestrator_v1(md)


def test_one_branch_consumed_by_two_gates_phase3_accepted():
    """Multi-consumer: a single branch_id referenced by SPAWN_JOIN prereqs
    on two different gated turns is accepted post-Phase-3."""
    b = ConversationBranchInfo(
        branch_id="r:0",
        child_conversation_ids=["c"],
        mode=ConversationBranchMode.SPAWN,
    )
    md = _md(
        [b],
        [
            TurnMetadata(branch_ids=["r:0"]),
            TurnMetadata(
                prerequisites=[
                    TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="r:0")
                ]
            ),
            TurnMetadata(
                prerequisites=[
                    TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="r:0")
                ]
            ),
        ],
    )
    validate_for_orchestrator_v1(md)


# ---------------------------------------------------------------------------
# 30. Strictly-prior boundary
# ---------------------------------------------------------------------------


def test_strictly_prior_n_to_n_plus_one_accepted():
    """Spawn at turn N, gate at turn N+1 is the canonical legal shape."""
    b = ConversationBranchInfo(
        branch_id="r:0",
        child_conversation_ids=["c"],
        mode=ConversationBranchMode.SPAWN,
    )
    md = _md(
        [b],
        [
            TurnMetadata(branch_ids=["r:0"]),
            TurnMetadata(
                prerequisites=[
                    TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="r:0")
                ]
            ),
        ],
    )
    validate_for_orchestrator_v1(md)


def test_strictly_prior_same_turn_rejected():
    """Spawn AND gate on the SAME turn is rejected."""
    b = ConversationBranchInfo(
        branch_id="r:0",
        child_conversation_ids=["c"],
        mode=ConversationBranchMode.SPAWN,
    )
    md = _md(
        [b],
        [
            TurnMetadata(
                branch_ids=["r:0"],
                prerequisites=[
                    TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="r:0")
                ],
            ),
        ],
    )
    # v2 message-text drift: "strictly-prior" rejection now reads
    # "not strictly earlier than this turn".
    with pytest.raises(NotImplementedError, match="not strictly earlier"):
        validate_for_orchestrator_v1(md)


def test_strictly_prior_gate_before_spawn_rejected():
    """Gate at turn 0 referencing a branch declared on turn 1 is rejected
    (forward reference)."""
    b = ConversationBranchInfo(
        branch_id="r:1",
        child_conversation_ids=["c"],
        mode=ConversationBranchMode.SPAWN,
    )
    md = _md(
        [b],
        [
            TurnMetadata(
                prerequisites=[
                    TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="r:1")
                ]
            ),
            TurnMetadata(branch_ids=["r:1"]),
        ],
    )
    # v2 message-text drift: forward-reference (gate before spawn) now reads
    # "not strictly earlier than this turn".
    with pytest.raises(NotImplementedError, match="not strictly earlier"):
        validate_for_orchestrator_v1(md)


# ---------------------------------------------------------------------------
# 31. FORK multi-parent at validator level
# ---------------------------------------------------------------------------


def test_validator_enforces_fork_multi_parent_globally():
    """The FORK single-parent invariant is enforced globally by
    ``validate_for_orchestrator_v1``.

    Hand-authored ``DatasetMetadata`` with two FORK branches across two
    conversations pointing at the same child must be rejected (defense-in-
    depth for paths that bypass the loader's _resolve_and_validate).
    """
    b1 = ConversationBranchInfo(
        branch_id="r1:0",
        child_conversation_ids=["c"],
        mode=ConversationBranchMode.FORK,
    )
    b2 = ConversationBranchInfo(
        branch_id="r2:0",
        child_conversation_ids=["c"],
        mode=ConversationBranchMode.FORK,
    )
    md = DatasetMetadata(
        conversations=[
            ConversationMetadata(
                conversation_id="r1",
                turns=[TurnMetadata(branch_ids=["r1:0"])],
                branches=[b1],
            ),
            ConversationMetadata(
                conversation_id="r2",
                turns=[TurnMetadata(branch_ids=["r2:0"])],
                branches=[b2],
            ),
            ConversationMetadata(conversation_id="c", turns=[TurnMetadata()]),
        ],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    with pytest.raises(NotImplementedError, match="multiple FORK branches"):
        validate_for_orchestrator_v1(md)


# ---------------------------------------------------------------------------
# 32. Background branch with a SPAWN_JOIN prereq pointing at it
# ---------------------------------------------------------------------------


def test_background_branch_referenced_by_spawn_join_rejected():
    # v2 re-keys fire-and-forget from is_background to dispatch_timing="pre":
    # a pre-session SPAWN has no parent session at dispatch time, so it cannot
    # be SPAWN_JOIN-gated. The branch is declared on turn 0 (valid pre-session
    # shape) and then referenced by a SPAWN_JOIN on turn 1, which fires the
    # fire-and-forget rejection in _check_turn_prereq. This preserves the
    # original intent (a fire-and-forget branch cannot be SPAWN_JOIN-gated).
    b = ConversationBranchInfo(
        branch_id="r:pre",
        child_conversation_ids=["c"],
        mode=ConversationBranchMode.SPAWN,
        background=True,
        dispatch_timing="pre",
    )
    p = TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="r:pre")
    md = _md(
        [b],
        [TurnMetadata(branch_ids=["r:pre"]), TurnMetadata(prerequisites=[p])],
    )
    with pytest.raises(NotImplementedError, match="fire-and-forget"):
        validate_for_orchestrator_v1(md)


# ---------------------------------------------------------------------------
# 33. child_conversation_ids referencing a non-existent session
# ---------------------------------------------------------------------------


def test_branch_child_id_not_in_dataset_rejected():
    """A branch whose child_conversation_id isn't in the dataset is rejected."""
    b = ConversationBranchInfo(
        branch_id="r:0",
        child_conversation_ids=["ghost"],
        mode=ConversationBranchMode.SPAWN,
    )
    # Don't auto-create the child stub — bypass _md helper.
    md = DatasetMetadata(
        conversations=[
            ConversationMetadata(
                conversation_id="r",
                turns=[TurnMetadata(branch_ids=["r:0"]), TurnMetadata()],
                branches=[b],
            ),
        ],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    with pytest.raises(
        NotImplementedError, match="does not reference an existing conversation"
    ):
        validate_for_orchestrator_v1(md)


# ---------------------------------------------------------------------------
# 34. Duplicate branch_id on the same turn (Phase 2 rule)
# ---------------------------------------------------------------------------


def test_duplicate_branch_id_on_same_turn_rejected():
    b = ConversationBranchInfo(
        branch_id="r:0",
        child_conversation_ids=["c"],
        mode=ConversationBranchMode.SPAWN,
    )
    md = _md(
        [b],
        [
            TurnMetadata(branch_ids=["r:0", "r:0"]),  # duplicate
            TurnMetadata(),
        ],
    )
    with pytest.raises(
        NotImplementedError, match="declared multiple times on the same turn"
    ):
        validate_for_orchestrator_v1(md)


# ---------------------------------------------------------------------------
# 34b. Duplicate SPAWN_JOIN prereq on the same gated turn (Phase 2 rule)
# ---------------------------------------------------------------------------


def test_duplicate_prereq_branch_id_on_same_gated_turn_rejected():
    """Two TurnPrerequisite entries on the same gated turn referencing the
    same branch_id is an authoring duplicate; the orchestrator's prereq
    index would otherwise carry duplicate (branch_id, gated_turn_idx)
    tuples. Rejected at load time."""
    b = ConversationBranchInfo(
        branch_id="b:0",
        child_conversation_ids=["c"],
        mode=ConversationBranchMode.SPAWN,
    )
    md = _md(
        [b],
        [
            TurnMetadata(branch_ids=["b:0"]),
            TurnMetadata(
                prerequisites=[
                    TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="b:0"),
                    TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="b:0"),
                ]
            ),
        ],
    )
    with pytest.raises(
        ValueError, match="duplicate SPAWN_JOIN prerequisite for branch_id 'b:0'"
    ):
        validate_for_orchestrator_v1(md)


# ---------------------------------------------------------------------------
# 35. Empty dataset
# ---------------------------------------------------------------------------


def test_empty_dataset_no_op():
    """Empty conversation list is valid (no-op)."""
    md = DatasetMetadata(
        conversations=[],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    validate_for_orchestrator_v1(md)


def test_dataset_with_only_one_conversation_no_branches_no_op():
    md = DatasetMetadata(
        conversations=[
            ConversationMetadata(
                conversation_id="solo",
                turns=[TurnMetadata(), TurnMetadata()],
            )
        ],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    validate_for_orchestrator_v1(md)


# ---------------------------------------------------------------------------
# 36. JSON round-trip + validator idempotency
# ---------------------------------------------------------------------------


def test_complex_dataset_metadata_round_trip_then_validate():
    """A complex DatasetMetadata serializes to JSON, deserializes, and
    re-validates with no error and no shape drift."""
    branches = [
        ConversationBranchInfo(
            branch_id="r:0",
            child_conversation_ids=["c0"],
            mode=ConversationBranchMode.SPAWN,
        ),
        ConversationBranchInfo(
            branch_id="r:1",
            child_conversation_ids=["c1"],
            mode=ConversationBranchMode.FORK,
        ),
    ]
    md = DatasetMetadata(
        conversations=[
            ConversationMetadata(
                conversation_id="r",
                turns=[
                    TurnMetadata(branch_ids=["r:0"]),
                    TurnMetadata(
                        branch_ids=["r:1"],
                        has_forks=True,
                        prerequisites=[
                            TurnPrerequisite(
                                kind=PrerequisiteKind.SPAWN_JOIN, branch_id="r:0"
                            )
                        ],
                    ),
                ],
                branches=branches,
            ),
            ConversationMetadata(conversation_id="c0", turns=[TurnMetadata()]),
            ConversationMetadata(conversation_id="c1", turns=[TurnMetadata()]),
        ],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    blob = md.model_dump(mode="json")
    md2 = DatasetMetadata.model_validate(blob)
    blob2 = md2.model_dump(mode="json")
    assert blob == blob2
    validate_for_orchestrator_v1(md2)


# ---------------------------------------------------------------------------
# 37. Self-spawn / cyclic spawn graphs
# ---------------------------------------------------------------------------


def _spawning_conv(cid: str, child_id: str) -> ConversationMetadata:
    """Conversation ``cid`` whose turn 0 spawns ``child_id`` (one edge in the
    spawn graph). Every other aspect is valid so cycle rejection is the sole
    violation in the shapes below."""
    return ConversationMetadata(
        conversation_id=cid,
        turns=[TurnMetadata(branch_ids=[f"{cid}:0"]), TurnMetadata()],
        branches=[
            ConversationBranchInfo(
                branch_id=f"{cid}:0",
                child_conversation_ids=[child_id],
                mode=ConversationBranchMode.SPAWN,
            )
        ],
    )


def test_self_spawn_cycle_rejected():
    """A conversation spawning itself (r -> r) is the minimal spawn cycle;
    the v1 orchestrator would recurse on it without bound at replay time."""
    md = DatasetMetadata(
        conversations=[_spawning_conv("r", "r")],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    with pytest.raises(
        NotImplementedError, match="spawn graph contains a cycle"
    ) as exc:
        validate_for_orchestrator_v1(md)
    assert "(r -> r)" in str(exc.value)


def test_two_node_spawn_cycle_rejected():
    """Mutual spawn (r -> c -> r) is rejected with the full cycle path in the
    error message."""
    md = DatasetMetadata(
        conversations=[_spawning_conv("r", "c"), _spawning_conv("c", "r")],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    with pytest.raises(
        NotImplementedError, match="spawn graph contains a cycle"
    ) as exc:
        validate_for_orchestrator_v1(md)
    assert "(r -> c -> r)" in str(exc.value)
    assert "conversation 'r' turn 0 branch 'r:0' -> 'c'" in str(exc.value)
    assert "conversation 'c' turn 0 branch 'c:0' -> 'r'" in str(exc.value)


def test_spawn_cycle_downstream_of_dfs_start_rejected():
    """A cycle that does not include the DFS start node (r -> a -> b -> a) is
    still detected, and the reported path is the cycle itself (a -> b -> a),
    not the acyclic lead-in from r."""
    md = DatasetMetadata(
        conversations=[
            _spawning_conv("r", "a"),
            _spawning_conv("a", "b"),
            _spawning_conv("b", "a"),
        ],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    with pytest.raises(
        NotImplementedError, match="spawn graph contains a cycle"
    ) as exc:
        validate_for_orchestrator_v1(md)
    assert "(a -> b -> a)" in str(exc.value)


def test_undeclared_descriptor_back_edge_not_reported_as_cycle():
    """A ConversationBranchInfo that no turn's branch_ids declares is never
    dispatched (both orchestrator dispatch paths gate on turn membership), so
    its child_conversation_ids are not runtime spawn edges: an apparent cycle
    through an undeclared descriptor (r -> c declared, c -> r undeclared)
    must validate cleanly rather than be misreported as a spawn cycle."""
    md = DatasetMetadata(
        conversations=[
            _spawning_conv("r", "c"),
            ConversationMetadata(
                conversation_id="c",
                turns=[TurnMetadata()],
                branches=[
                    ConversationBranchInfo(
                        branch_id="c:undeclared",
                        child_conversation_ids=["r"],
                        mode=ConversationBranchMode.SPAWN,
                    )
                ],
            ),
        ],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    validate_for_orchestrator_v1(md)


# ---------------------------------------------------------------------------
# 38. Duplicate ConversationBranchInfo descriptors sharing one branch_id
# ---------------------------------------------------------------------------


def test_duplicate_branch_descriptor_rejected():
    """Two ConversationBranchInfo objects on one conversation sharing a
    branch_id would silently collapse under the orchestrator's dict-by-id,
    dropping the first descriptor's children; rejected at load time."""
    md = _md(
        [
            ConversationBranchInfo(
                branch_id="r:0",
                child_conversation_ids=["c"],
                mode=ConversationBranchMode.SPAWN,
            ),
            ConversationBranchInfo(
                branch_id="r:0",
                child_conversation_ids=["c2"],
                mode=ConversationBranchMode.SPAWN,
            ),
        ],
        [TurnMetadata(branch_ids=["r:0"]), TurnMetadata()],
    )
    with pytest.raises(
        NotImplementedError, match="declared by multiple ConversationBranchInfo"
    ) as exc:
        validate_for_orchestrator_v1(md)
    assert "conversation 'r'" in str(exc.value)
    assert "'r:0'" in str(exc.value)


# ---------------------------------------------------------------------------
# 39. Dangling turn branch_id with no matching branch descriptor
# ---------------------------------------------------------------------------


def test_dangling_branch_id_with_no_descriptors_rejected():
    """A turn declaring a branch_id when the conversation has no branch
    descriptors at all would make the orchestrator's lookup return None and
    the authored branch silently never spawn; rejected at load time."""
    md = _md(None, [TurnMetadata(branch_ids=["ghost"]), TurnMetadata()])
    with pytest.raises(
        NotImplementedError, match="has no matching ConversationBranchInfo"
    ) as exc:
        validate_for_orchestrator_v1(md)
    assert "conversation 'r' turn 0" in str(exc.value)
    assert "'ghost'" in str(exc.value)


def test_dangling_branch_id_on_later_turn_amid_valid_branch_rejected():
    """One resolvable branch does not mask a dangling branch_id on a later
    turn; the error pinpoints the declaring turn."""
    md = _md(
        [
            ConversationBranchInfo(
                branch_id="r:0",
                child_conversation_ids=["c"],
                mode=ConversationBranchMode.SPAWN,
            )
        ],
        [
            TurnMetadata(branch_ids=["r:0"]),
            TurnMetadata(branch_ids=["ghost"]),
            TurnMetadata(),
        ],
    )
    with pytest.raises(
        NotImplementedError, match="has no matching ConversationBranchInfo"
    ) as exc:
        validate_for_orchestrator_v1(md)
    assert "conversation 'r' turn 1" in str(exc.value)
    assert "'ghost'" in str(exc.value)


def test_acyclic_spawn_chain_with_shared_visited_node_accepted():
    """An acyclic chain (r -> c -> d) where a later DFS start (c) was already
    fully explored by an earlier one (r) validates cleanly; the DFS must skip
    already-explored start nodes instead of re-walking (or worse, misreporting
    a cycle on) them."""
    md = DatasetMetadata(
        conversations=[
            _spawning_conv("r", "c"),
            _spawning_conv("c", "d"),
            ConversationMetadata(conversation_id="d", turns=[TurnMetadata()]),
        ],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    validate_for_orchestrator_v1(md)


# ---------------------------------------------------------------------------
# 40. SPAWN_JOIN referencing a branch descriptor no turn declares
# ---------------------------------------------------------------------------


def test_prereq_referencing_undeclared_branch_descriptor_rejected():
    """A SPAWN_JOIN whose branch_id resolves to a ConversationBranchInfo that
    no turn's branch_ids declares can never be gated at runtime (the branch
    never spawns); rejected with the not-declared-on-any-turn message."""
    md = _md(
        [_ok_branch()],  # descriptor exists but no turn declares "r:0"
        [
            TurnMetadata(),
            TurnMetadata(
                prerequisites=[
                    TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="r:0")
                ]
            ),
        ],
    )
    with pytest.raises(NotImplementedError, match="is not declared on any turn") as exc:
        validate_for_orchestrator_v1(md)
    assert "conversation 'r' turn 1" in str(exc.value)
    assert "'r:0'" in str(exc.value)
