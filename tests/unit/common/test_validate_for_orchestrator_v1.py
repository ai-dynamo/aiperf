# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

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


def _one_conv_with(
    prereqs: list[TurnPrerequisite] | None = None,
    branches: list[ConversationBranchInfo] | None = None,
) -> DatasetMetadata:
    # Auto-generate stub ConversationMetadata for every child_conversation_id
    # referenced by any provided branch so the validator's child-existence
    # check is satisfied without every test needing to construct stubs.
    child_ids: set[str] = set()
    for b in branches or []:
        child_ids.update(b.child_conversation_ids)
    return DatasetMetadata(
        conversations=[
            ConversationMetadata(
                conversation_id="r",
                turns=[
                    TurnMetadata(branch_ids=["r:0"] if branches else []),
                    TurnMetadata(prerequisites=prereqs or []),
                ],
                branches=branches or [],
            ),
            *(
                ConversationMetadata(conversation_id=cid, turns=[TurnMetadata()])
                for cid in sorted(child_ids)
            ),
        ],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )


def _ok_branch() -> ConversationBranchInfo:
    return ConversationBranchInfo(
        branch_id="r:0",
        child_conversation_ids=["c"],
        mode=ConversationBranchMode.SPAWN,
    )


def test_validator_accepts_spawn_join_prereq():
    md = _one_conv_with(
        prereqs=[TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="r:0")],
        branches=[_ok_branch()],
    )
    validate_for_orchestrator_v1(md)


@pytest.mark.parametrize(
    "kind",
    [
        PrerequisiteKind.CHILD_SESSION_COMPLETE,
        PrerequisiteKind.TIMER,
        PrerequisiteKind.EXTERNAL_EVENT,
        PrerequisiteKind.BARRIER,
    ],
)
def test_validator_rejects_non_spawn_join_kinds(kind):
    md = _one_conv_with(
        prereqs=[TurnPrerequisite(kind=kind, branch_id="r:0")], branches=[_ok_branch()]
    )
    with pytest.raises(NotImplementedError, match="not supported by v1 orchestrator"):
        validate_for_orchestrator_v1(md)


def test_validator_rejects_per_child_prereq():
    md = _one_conv_with(
        prereqs=[
            TurnPrerequisite(
                kind=PrerequisiteKind.SPAWN_JOIN,
                branch_id="r:0",
                child_conversation_ids=["c"],
            )
        ],
        branches=[_ok_branch()],
    )
    with pytest.raises(NotImplementedError, match="per-child prerequisite subsets"):
        validate_for_orchestrator_v1(md)


def test_validator_rejects_barrier_id():
    md = _one_conv_with(
        prereqs=[
            TurnPrerequisite(
                kind=PrerequisiteKind.SPAWN_JOIN, branch_id="r:0", barrier_id="b"
            )
        ],
        branches=[_ok_branch()],
    )
    with pytest.raises(NotImplementedError, match="barrier-based"):
        validate_for_orchestrator_v1(md)


def test_validator_rejects_timer_seconds():
    md = _one_conv_with(
        prereqs=[
            TurnPrerequisite(
                kind=PrerequisiteKind.SPAWN_JOIN, branch_id="r:0", timer_seconds=1.0
            )
        ],
        branches=[_ok_branch()],
    )
    with pytest.raises(NotImplementedError, match="timer-based"):
        validate_for_orchestrator_v1(md)


def test_validator_rejects_event_name():
    md = _one_conv_with(
        prereqs=[
            TurnPrerequisite(
                kind=PrerequisiteKind.SPAWN_JOIN, branch_id="r:0", event_name="e"
            )
        ],
        branches=[_ok_branch()],
    )
    with pytest.raises(NotImplementedError, match="event-based"):
        validate_for_orchestrator_v1(md)


def test_validator_accepts_multiple_prereqs_on_one_turn_distinct_branches():
    """Phase 3: multi-source gates (one turn gated by multiple branches) are
    now supported; the orchestrator tracks each prereq independently under
    the same ``PendingBranchJoin.outstanding`` dict."""
    md = DatasetMetadata(
        conversations=[
            ConversationMetadata(
                conversation_id="r",
                turns=[
                    TurnMetadata(branch_ids=["r:0", "r:0b"]),
                    TurnMetadata(
                        prerequisites=[
                            TurnPrerequisite(
                                kind=PrerequisiteKind.SPAWN_JOIN, branch_id="r:0"
                            ),
                            TurnPrerequisite(
                                kind=PrerequisiteKind.SPAWN_JOIN, branch_id="r:0b"
                            ),
                        ]
                    ),
                ],
                branches=[
                    _ok_branch(),
                    ConversationBranchInfo(
                        branch_id="r:0b",
                        child_conversation_ids=["c2"],
                        mode=ConversationBranchMode.SPAWN,
                    ),
                ],
            ),
            ConversationMetadata(conversation_id="c", turns=[TurnMetadata()]),
            ConversationMetadata(conversation_id="c2", turns=[TurnMetadata()]),
        ],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    # Phase 3 accepts this shape.
    validate_for_orchestrator_v1(md)


def test_validator_rejects_prereq_pointing_at_unknown_branch():
    md = _one_conv_with(
        prereqs=[
            TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="missing")
        ],
        branches=[_ok_branch()],
    )
    with pytest.raises(NotImplementedError, match="does not reference a prior branch"):
        validate_for_orchestrator_v1(md)


def test_validator_rejects_background_branch_with_matching_prereq():
    # v2 re-keys fire-and-forget from is_background to dispatch_timing="pre":
    # a pre-session (fire-and-forget) SPAWN cannot be SPAWN_JOIN-gated because
    # no parent session exists at dispatch time. The branch is declared on
    # turn 0 (valid pre-session shape) and referenced by the prereq on turn 1.
    br = ConversationBranchInfo(
        branch_id="r:0",
        child_conversation_ids=["c"],
        mode=ConversationBranchMode.SPAWN,
        is_background=True,
        dispatch_timing="pre",
    )
    md = _one_conv_with(
        prereqs=[TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="r:0")],
        branches=[br],
    )
    with pytest.raises(NotImplementedError, match="fire-and-forget"):
        validate_for_orchestrator_v1(md)


def test_validator_accepts_overlapping_pending_joins_for_parent():
    # Phase 1: delayed joins are supported. Branch r:0 is spawned on turn 0
    # and consumed on turn 3 (K=3); branch r:1 is spawned on turn 1 — inside
    # the first gate's open window — and consumed on turn 4 (K=3). Parent
    # holds two concurrent future joins; the orchestrator's two-level
    # _future_joins map handles this naturally.
    md = DatasetMetadata(
        conversations=[
            ConversationMetadata(
                conversation_id="r",
                turns=[
                    TurnMetadata(branch_ids=["r:0"]),
                    TurnMetadata(branch_ids=["r:1"]),
                    TurnMetadata(),
                    TurnMetadata(
                        prerequisites=[
                            TurnPrerequisite(
                                kind=PrerequisiteKind.SPAWN_JOIN, branch_id="r:0"
                            )
                        ]
                    ),
                    TurnMetadata(
                        prerequisites=[
                            TurnPrerequisite(
                                kind=PrerequisiteKind.SPAWN_JOIN, branch_id="r:1"
                            )
                        ]
                    ),
                ],
                branches=[
                    ConversationBranchInfo(
                        branch_id="r:0",
                        child_conversation_ids=["c0"],
                        mode=ConversationBranchMode.SPAWN,
                    ),
                    ConversationBranchInfo(
                        branch_id="r:1",
                        child_conversation_ids=["c1"],
                        mode=ConversationBranchMode.SPAWN,
                    ),
                ],
            ),
            ConversationMetadata(conversation_id="c0", turns=[TurnMetadata()]),
            ConversationMetadata(conversation_id="c1", turns=[TurnMetadata()]),
        ],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    # No exception — Phase 1 accepts this shape.
    validate_for_orchestrator_v1(md)
