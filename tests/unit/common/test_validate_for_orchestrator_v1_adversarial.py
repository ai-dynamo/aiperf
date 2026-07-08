# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial coverage for validate_for_orchestrator_v1.

Complements the shipped happy-path tests in
``test_validate_for_orchestrator_v1.py`` with edge cases, unicode inputs,
degenerate shapes, and post-fix regression coverage for Task 7/8 behavior.
"""

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


# --- 1. Null branch_id on SPAWN_JOIN -----------------------------------------


def test_validator_rejects_null_branch_id_on_spawn_join_kind():
    md = _one_conv_with(
        prereqs=[TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id=None)],
        branches=[_ok_branch()],
    )
    with pytest.raises(
        NotImplementedError, match="does not reference a prior branch"
    ) as exc:
        validate_for_orchestrator_v1(md)
    assert "None" in str(exc.value)


# --- 2. Empty-string branch_id -----------------------------------------------


def test_validator_rejects_empty_string_branch_id():
    md = _one_conv_with(
        prereqs=[TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="")],
        branches=[_ok_branch()],
    )
    with pytest.raises(NotImplementedError, match="does not reference a prior branch"):
        validate_for_orchestrator_v1(md)


# --- 3. Whitespace branch_id -------------------------------------------------


def test_validator_rejects_whitespace_branch_id():
    md = _one_conv_with(
        prereqs=[TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="   ")],
        branches=[_ok_branch()],
    )
    with pytest.raises(NotImplementedError, match="does not reference a prior branch"):
        validate_for_orchestrator_v1(md)


# --- 4. Unicode branch_id passes through -------------------------------------


def test_validator_accepts_unicode_branch_id():
    unicode_id = "分支-🌲"
    br = ConversationBranchInfo(
        branch_id=unicode_id,
        child_conversation_ids=["c"],
        mode=ConversationBranchMode.SPAWN,
    )
    md = DatasetMetadata(
        conversations=[
            ConversationMetadata(
                conversation_id="r",
                turns=[
                    TurnMetadata(branch_ids=[unicode_id]),
                    TurnMetadata(
                        prerequisites=[
                            TurnPrerequisite(
                                kind=PrerequisiteKind.SPAWN_JOIN, branch_id=unicode_id
                            )
                        ]
                    ),
                ],
                branches=[br],
            ),
            ConversationMetadata(conversation_id="c", turns=[TurnMetadata()]),
        ],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    validate_for_orchestrator_v1(md)


# --- 5. Repeated same branch_id on one turn rejected ------------------------


def test_validator_rejects_repeated_same_branch_id_on_one_turn():
    """Two SPAWN_JOIN prereqs on the same gated turn referencing the same
    branch_id is an authoring duplicate and rejected at load time."""
    md = _one_conv_with(
        prereqs=[
            TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="r:0"),
            TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="r:0"),
        ],
        branches=[_ok_branch()],
    )
    with pytest.raises(
        ValueError, match="duplicate SPAWN_JOIN prerequisite for branch_id 'r:0'"
    ):
        validate_for_orchestrator_v1(md)


# --- 6. Empty dataset metadata passes ----------------------------------------


def test_validator_passes_on_empty_dataset_metadata():
    md = DatasetMetadata(
        conversations=[],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    validate_for_orchestrator_v1(md)


# --- 7. Conversation with no turns passes ------------------------------------


def test_validator_passes_on_conversation_with_no_turns():
    md = DatasetMetadata(
        conversations=[
            ConversationMetadata(
                conversation_id="r",
                turns=[],
                branches=[_ok_branch()],
            ),
            ConversationMetadata(conversation_id="c", turns=[TurnMetadata()]),
        ],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    validate_for_orchestrator_v1(md)


# --- 8. Prereq with no declared branches rejects -----------------------------


def test_validator_rejects_prereq_when_conversation_has_no_branches():
    md = DatasetMetadata(
        conversations=[
            ConversationMetadata(
                conversation_id="r",
                turns=[
                    TurnMetadata(
                        prerequisites=[
                            TurnPrerequisite(
                                kind=PrerequisiteKind.SPAWN_JOIN, branch_id="r:0"
                            )
                        ]
                    ),
                ],
                branches=[],
            )
        ],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    with pytest.raises(NotImplementedError, match="does not reference a prior branch"):
        validate_for_orchestrator_v1(md)


# --- 9. Branch with empty child_conversation_ids passes ----------------------


def test_validator_accepts_branch_with_empty_child_conversation_ids():
    br = ConversationBranchInfo(
        branch_id="r:0",
        child_conversation_ids=[],
        mode=ConversationBranchMode.SPAWN,
    )
    md = _one_conv_with(
        prereqs=[TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="r:0")],
        branches=[br],
    )
    validate_for_orchestrator_v1(md)


# --- 10. Multiple independent conversations pass -----------------------------


def test_validator_accepts_multiple_conversations_each_with_own_gating():
    conv_a = ConversationMetadata(
        conversation_id="a",
        turns=[
            TurnMetadata(branch_ids=["a:0"]),
            TurnMetadata(
                prerequisites=[
                    TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="a:0")
                ]
            ),
        ],
        branches=[
            ConversationBranchInfo(
                branch_id="a:0",
                child_conversation_ids=["ca"],
                mode=ConversationBranchMode.SPAWN,
            )
        ],
    )
    conv_b = ConversationMetadata(
        conversation_id="b",
        turns=[
            TurnMetadata(branch_ids=["b:0"]),
            TurnMetadata(
                prerequisites=[
                    TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="b:0")
                ]
            ),
        ],
        branches=[
            ConversationBranchInfo(
                branch_id="b:0",
                child_conversation_ids=["cb"],
                mode=ConversationBranchMode.SPAWN,
            )
        ],
    )
    md = DatasetMetadata(
        conversations=[
            conv_a,
            conv_b,
            ConversationMetadata(conversation_id="ca", turns=[TurnMetadata()]),
            ConversationMetadata(conversation_id="cb", turns=[TurnMetadata()]),
        ],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    validate_for_orchestrator_v1(md)


# --- 11. Three-level chain of non-overlapping spawn-join gates ---------------


def test_validator_accepts_three_level_chain_of_spawn_join_gates():
    # Current validator rejects any turn that both consumes AND spawns (treats
    # it as an overlapping pending-join gate), so the chain uses strict
    # spacer turns: each spawn is consumed on a dedicated turn before the next
    # spawn fires.
    md = DatasetMetadata(
        conversations=[
            ConversationMetadata(
                conversation_id="r",
                turns=[
                    TurnMetadata(branch_ids=["a"]),
                    TurnMetadata(
                        prerequisites=[
                            TurnPrerequisite(
                                kind=PrerequisiteKind.SPAWN_JOIN, branch_id="a"
                            )
                        ]
                    ),
                    TurnMetadata(branch_ids=["b"]),
                    TurnMetadata(
                        prerequisites=[
                            TurnPrerequisite(
                                kind=PrerequisiteKind.SPAWN_JOIN, branch_id="b"
                            )
                        ]
                    ),
                    TurnMetadata(branch_ids=["c"]),
                    TurnMetadata(
                        prerequisites=[
                            TurnPrerequisite(
                                kind=PrerequisiteKind.SPAWN_JOIN, branch_id="c"
                            )
                        ]
                    ),
                ],
                branches=[
                    ConversationBranchInfo(
                        branch_id="a",
                        child_conversation_ids=["ca"],
                        mode=ConversationBranchMode.SPAWN,
                    ),
                    ConversationBranchInfo(
                        branch_id="b",
                        child_conversation_ids=["cb"],
                        mode=ConversationBranchMode.SPAWN,
                    ),
                    ConversationBranchInfo(
                        branch_id="c",
                        child_conversation_ids=["cc"],
                        mode=ConversationBranchMode.SPAWN,
                    ),
                ],
            ),
            ConversationMetadata(conversation_id="ca", turns=[TurnMetadata()]),
            ConversationMetadata(conversation_id="cb", turns=[TurnMetadata()]),
            ConversationMetadata(conversation_id="cc", turns=[TurnMetadata()]),
        ],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    validate_for_orchestrator_v1(md)


# --- 12. Barrier rejection fires before multi-source aggregation -------------


def test_validator_rejects_barrier_id_before_checking_multi_source_count():
    md = DatasetMetadata(
        conversations=[
            ConversationMetadata(
                conversation_id="r",
                turns=[
                    TurnMetadata(branch_ids=["r:0", "r:0b"]),
                    TurnMetadata(
                        prerequisites=[
                            TurnPrerequisite(
                                kind=PrerequisiteKind.SPAWN_JOIN,
                                branch_id="r:0",
                                barrier_id="b1",
                            ),
                            TurnPrerequisite(
                                kind=PrerequisiteKind.SPAWN_JOIN,
                                branch_id="r:0b",
                                barrier_id="b2",
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
    with pytest.raises(NotImplementedError, match="barrier-based"):
        validate_for_orchestrator_v1(md)


# --- 13. same-turn self-reference (Task 7) -----------------------------------


def test_validator_rejects_same_turn_prereq_reference_post_fix():
    md = DatasetMetadata(
        conversations=[
            ConversationMetadata(
                conversation_id="r",
                turns=[
                    TurnMetadata(
                        branch_ids=["r:0"],
                        prerequisites=[
                            TurnPrerequisite(
                                kind=PrerequisiteKind.SPAWN_JOIN, branch_id="r:0"
                            )
                        ],
                    ),
                ],
                branches=[_ok_branch()],
            ),
            ConversationMetadata(conversation_id="c", turns=[TurnMetadata()]),
        ],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    # v2 message-text drift: the same-turn (not-strictly-earlier) rejection now
    # reads "not strictly earlier than this turn".
    with pytest.raises(NotImplementedError, match="not strictly earlier"):
        validate_for_orchestrator_v1(md)


# --- 14. forward prereq reference (Task 7) -----------------------------------


def test_validator_rejects_forward_prereq_reference_post_fix():
    md = DatasetMetadata(
        conversations=[
            ConversationMetadata(
                conversation_id="r",
                turns=[
                    TurnMetadata(
                        prerequisites=[
                            TurnPrerequisite(
                                kind=PrerequisiteKind.SPAWN_JOIN, branch_id="r:1"
                            )
                        ]
                    ),
                    TurnMetadata(branch_ids=["r:1"]),
                ],
                branches=[
                    ConversationBranchInfo(
                        branch_id="r:1",
                        child_conversation_ids=["c"],
                        mode=ConversationBranchMode.SPAWN,
                    )
                ],
            ),
            ConversationMetadata(conversation_id="c", turns=[TurnMetadata()]),
        ],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    # v2 message-text drift: forward-reference rejection now reads
    # "not strictly earlier than this turn".
    with pytest.raises(NotImplementedError, match="not strictly earlier"):
        validate_for_orchestrator_v1(md)


# --- 15. Phase 3: multiple gated consumers on same branch accepted ----------


def test_validator_accepts_multiple_turns_consuming_same_branch_phase_3():
    """Phase 3: one branch_id may be referenced by prereqs on multiple gated
    turns. The orchestrator's ``_future_joins[parent][gated_idx]`` gives each
    gate its own pending join entry."""
    md = DatasetMetadata(
        conversations=[
            ConversationMetadata(
                conversation_id="r",
                turns=[
                    TurnMetadata(branch_ids=["r:0"]),
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
                                kind=PrerequisiteKind.SPAWN_JOIN, branch_id="r:0"
                            )
                        ]
                    ),
                ],
                branches=[_ok_branch()],
            ),
            ConversationMetadata(conversation_id="c", turns=[TurnMetadata()]),
        ],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    # Phase 3 accepts; the orchestrator registers separate gates per gated turn.
    validate_for_orchestrator_v1(md)


# --- 16. FORK mode branch with matching SPAWN_JOIN prereq passes (documented) --


def test_validator_accepts_fork_mode_branch_with_matching_prereq():
    # NOTE: v1 validator's supported_modes is {FORK, SPAWN}, and there is no
    # cross-check that SPAWN_JOIN prereqs reference specifically a SPAWN-mode
    # branch. Currently accepted; documenting the behavior.
    br = ConversationBranchInfo(
        branch_id="r:0",
        child_conversation_ids=["c"],
        mode=ConversationBranchMode.FORK,
    )
    md = _one_conv_with(
        prereqs=[TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="r:0")],
        branches=[br],
    )
    validate_for_orchestrator_v1(md)


# --- 17. forward ref across multi-turn chain (Task 7) ------------------------


def test_validator_rejects_prereq_pointing_at_declared_branch_on_later_turn_chain():
    md = DatasetMetadata(
        conversations=[
            ConversationMetadata(
                conversation_id="r",
                turns=[
                    TurnMetadata(),
                    TurnMetadata(),
                    TurnMetadata(
                        prerequisites=[
                            TurnPrerequisite(
                                kind=PrerequisiteKind.SPAWN_JOIN, branch_id="r:4"
                            )
                        ]
                    ),
                    TurnMetadata(),
                    TurnMetadata(branch_ids=["r:4"]),
                ],
                branches=[
                    ConversationBranchInfo(
                        branch_id="r:4",
                        child_conversation_ids=["c"],
                        mode=ConversationBranchMode.SPAWN,
                    )
                ],
            ),
            ConversationMetadata(conversation_id="c", turns=[TurnMetadata()]),
        ],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    # v2 message-text drift: forward-reference across a multi-turn chain now
    # reads "not strictly earlier than this turn".
    with pytest.raises(NotImplementedError, match="not strictly earlier"):
        validate_for_orchestrator_v1(md)


def test_validator_accepts_consume_and_spawn_on_same_turn():
    """A turn that consumes one gate AND spawns a new branch on the same turn
    must validate: semantically, the gate closes at the start of the consumer
    turn (its prereq fires), the consumer dispatches, and the new spawn fires
    at end-of-turn when intercept() runs. No temporal overlap with the closing
    gate. Pre-fix the validator rejected this with ``idx <= gate_open_until``
    using an inclusive comparison; the fix relaxes to strict ``<``.
    """
    md = DatasetMetadata(
        conversations=[
            ConversationMetadata(
                conversation_id="r",
                turns=[
                    # Turn 0 spawns branch A.
                    TurnMetadata(branch_ids=["r:0"]),
                    # Turn 1 consumes A AND spawns branch B on the same turn.
                    TurnMetadata(
                        branch_ids=["r:1"],
                        prerequisites=[
                            TurnPrerequisite(
                                kind=PrerequisiteKind.SPAWN_JOIN, branch_id="r:0"
                            )
                        ],
                    ),
                    # Turn 2 consumes B.
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
    # Should NOT raise post-fix.
    validate_for_orchestrator_v1(md)


def test_validator_rejects_branch_child_conversation_id_not_in_dataset():
    """v1 requires every ConversationBranchInfo.child_conversation_ids entry to
    reference an existing conversation in the DatasetMetadata. A branch that
    spawns children which don't exist would leave the orchestrator unable to
    start those child sessions at runtime.
    """
    md = DatasetMetadata(
        conversations=[
            ConversationMetadata(
                conversation_id="r",
                turns=[
                    TurnMetadata(branch_ids=["r:0"]),
                    TurnMetadata(
                        prerequisites=[
                            TurnPrerequisite(
                                kind=PrerequisiteKind.SPAWN_JOIN, branch_id="r:0"
                            )
                        ]
                    ),
                ],
                branches=[
                    ConversationBranchInfo(
                        branch_id="r:0",
                        child_conversation_ids=["nonexistent_child"],
                        mode=ConversationBranchMode.SPAWN,
                    )
                ],
            )
        ],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    with pytest.raises(NotImplementedError, match="does not reference an existing"):
        validate_for_orchestrator_v1(md)


def test_validator_accepts_branch_child_conversation_id_resolves_to_another_conversation():
    """A branch whose child_conversation_ids resolve to conversations in the
    metadata must validate cleanly.
    """
    md = DatasetMetadata(
        conversations=[
            ConversationMetadata(
                conversation_id="r",
                turns=[
                    TurnMetadata(branch_ids=["r:0"]),
                    TurnMetadata(
                        prerequisites=[
                            TurnPrerequisite(
                                kind=PrerequisiteKind.SPAWN_JOIN, branch_id="r:0"
                            )
                        ]
                    ),
                ],
                branches=[
                    ConversationBranchInfo(
                        branch_id="r:0",
                        child_conversation_ids=["child_a", "child_b"],
                        mode=ConversationBranchMode.SPAWN,
                    )
                ],
            ),
            ConversationMetadata(conversation_id="child_a", turns=[TurnMetadata()]),
            ConversationMetadata(conversation_id="child_b", turns=[TurnMetadata()]),
        ],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    validate_for_orchestrator_v1(md)


def test_validator_accepts_multi_gated_branches_on_single_spawning_turn():
    """Phase 2: a turn declaring two branches each with their own consumer
    prereq is now accepted. The orchestrator's _future_joins[parent][gated_idx]
    dict-of-dict tracks each branch's gate independently. Multi-consumer per
    branch is still rejected (Phase 3 lifts that).
    """
    md = DatasetMetadata(
        conversations=[
            ConversationMetadata(
                conversation_id="r",
                turns=[
                    # Turn 0 spawns TWO gated branches.
                    TurnMetadata(branch_ids=["r:0:a", "r:0:b"]),
                    # Turn 1 consumes branch a.
                    TurnMetadata(
                        prerequisites=[
                            TurnPrerequisite(
                                kind=PrerequisiteKind.SPAWN_JOIN, branch_id="r:0:a"
                            )
                        ]
                    ),
                    # Turn 2 consumes branch b.
                    TurnMetadata(
                        prerequisites=[
                            TurnPrerequisite(
                                kind=PrerequisiteKind.SPAWN_JOIN, branch_id="r:0:b"
                            )
                        ]
                    ),
                ],
                branches=[
                    ConversationBranchInfo(
                        branch_id="r:0:a",
                        child_conversation_ids=["c_a"],
                        mode=ConversationBranchMode.SPAWN,
                    ),
                    ConversationBranchInfo(
                        branch_id="r:0:b",
                        child_conversation_ids=["c_b"],
                        mode=ConversationBranchMode.SPAWN,
                    ),
                ],
            ),
            ConversationMetadata(conversation_id="c_a", turns=[TurnMetadata()]),
            ConversationMetadata(conversation_id="c_b", turns=[TurnMetadata()]),
        ],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    validate_for_orchestrator_v1(md)


def test_duplicate_branch_id_on_same_turn_rejected():
    """Phase 2 guardrail: declaring the same branch_id twice on a single
    parent turn is rejected as an authoring bug. The orchestrator would
    otherwise spawn children under that branch twice and double-register
    the gate.
    """
    md = DatasetMetadata(
        conversations=[
            ConversationMetadata(
                conversation_id="r",
                turns=[
                    # Turn 0 declares branch "r:0" twice.
                    TurnMetadata(branch_ids=["r:0", "r:0"]),
                    TurnMetadata(
                        prerequisites=[
                            TurnPrerequisite(
                                kind=PrerequisiteKind.SPAWN_JOIN, branch_id="r:0"
                            )
                        ]
                    ),
                ],
                branches=[
                    ConversationBranchInfo(
                        branch_id="r:0",
                        child_conversation_ids=["c"],
                        mode=ConversationBranchMode.SPAWN,
                    ),
                ],
            ),
            ConversationMetadata(conversation_id="c", turns=[TurnMetadata()]),
        ],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    with pytest.raises(
        NotImplementedError, match="declared multiple times on the same turn"
    ):
        validate_for_orchestrator_v1(md)


def test_validator_accepts_one_gated_plus_one_background_branch_on_same_turn():
    """A spawning turn may carry one gated branch AND any number of background
    branches (the latter are fire-and-forget, don't participate in gating).
    """
    md = DatasetMetadata(
        conversations=[
            ConversationMetadata(
                conversation_id="r",
                turns=[
                    TurnMetadata(branch_ids=["r:0:gated", "r:0:bg"]),
                    TurnMetadata(
                        prerequisites=[
                            TurnPrerequisite(
                                kind=PrerequisiteKind.SPAWN_JOIN,
                                branch_id="r:0:gated",
                            )
                        ]
                    ),
                ],
                branches=[
                    ConversationBranchInfo(
                        branch_id="r:0:gated",
                        child_conversation_ids=["c_g"],
                        mode=ConversationBranchMode.SPAWN,
                    ),
                    ConversationBranchInfo(
                        branch_id="r:0:bg",
                        child_conversation_ids=["c_bg"],
                        mode=ConversationBranchMode.SPAWN,
                        background=True,
                    ),
                ],
            ),
            ConversationMetadata(conversation_id="c_g", turns=[TurnMetadata()]),
            ConversationMetadata(conversation_id="c_bg", turns=[TurnMetadata()]),
        ],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    validate_for_orchestrator_v1(md)
