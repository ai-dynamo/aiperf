# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fan-in validator coverage: multi-source gates and multi-consumer branches are accepted, while prior/background/non-SPAWN_JOIN shapes stay rejected."""

from __future__ import annotations

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
from aiperf.plugin.enums import DatasetSamplingStrategy


def _mk_child(cid: str) -> ConversationMetadata:
    return ConversationMetadata(conversation_id=cid, turns=[TurnMetadata()])


def test_fan_in_multi_source_gate_accepted():
    """A single gated turn with prereqs from two distinct branches spawned on different earlier turns is accepted."""
    conv = ConversationMetadata(
        conversation_id="r",
        turns=[
            TurnMetadata(branch_ids=["r:0:A"]),
            TurnMetadata(),
            TurnMetadata(branch_ids=["r:2:B"]),
            TurnMetadata(
                prerequisites=[
                    TurnPrerequisite(
                        kind=PrerequisiteKind.SPAWN_JOIN, branch_id="r:0:A"
                    ),
                    TurnPrerequisite(
                        kind=PrerequisiteKind.SPAWN_JOIN, branch_id="r:2:B"
                    ),
                ]
            ),
        ],
        branches=[
            ConversationBranchInfo(
                branch_id="r:0:A",
                child_conversation_ids=["ca"],
                mode=ConversationBranchMode.SPAWN,
            ),
            ConversationBranchInfo(
                branch_id="r:2:B",
                child_conversation_ids=["cb"],
                mode=ConversationBranchMode.SPAWN,
            ),
        ],
    )
    md = DatasetMetadata(
        conversations=[conv, _mk_child("ca"), _mk_child("cb")],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    validate_for_orchestrator_v1(md)


def test_fan_in_multi_source_gate_on_same_spawning_turn_accepted():
    """Two branches declared on the same spawning turn both gating the same later turn is accepted."""
    conv = ConversationMetadata(
        conversation_id="r",
        turns=[
            TurnMetadata(branch_ids=["r:0:A", "r:0:B"]),
            TurnMetadata(
                prerequisites=[
                    TurnPrerequisite(
                        kind=PrerequisiteKind.SPAWN_JOIN, branch_id="r:0:A"
                    ),
                    TurnPrerequisite(
                        kind=PrerequisiteKind.SPAWN_JOIN, branch_id="r:0:B"
                    ),
                ]
            ),
        ],
        branches=[
            ConversationBranchInfo(
                branch_id="r:0:A",
                child_conversation_ids=["ca"],
                mode=ConversationBranchMode.SPAWN,
            ),
            ConversationBranchInfo(
                branch_id="r:0:B",
                child_conversation_ids=["cb"],
                mode=ConversationBranchMode.SPAWN,
            ),
        ],
    )
    md = DatasetMetadata(
        conversations=[conv, _mk_child("ca"), _mk_child("cb")],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    validate_for_orchestrator_v1(md)


def test_fan_in_branch_consumed_by_multiple_gates_accepted():
    """One branch_id referenced by prereqs on multiple distinct gated turns is accepted."""
    conv = ConversationMetadata(
        conversation_id="r",
        turns=[
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
        branches=[
            ConversationBranchInfo(
                branch_id="r:0",
                child_conversation_ids=["c"],
                mode=ConversationBranchMode.SPAWN,
            )
        ],
    )
    md = DatasetMetadata(
        conversations=[conv, _mk_child("c")],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    validate_for_orchestrator_v1(md)


def test_fan_in_does_not_lift_strictly_prior_rejection():
    """Fan-in doesn't excuse a forward prereq reference."""
    conv = ConversationMetadata(
        conversation_id="r",
        turns=[
            TurnMetadata(
                prerequisites=[
                    TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="r:1")
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
    )
    md = DatasetMetadata(
        conversations=[conv, _mk_child("c")],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    # v2 message-text drift: the forward-reference rejection now reads
    # "not strictly earlier than this turn".
    with pytest.raises(NotImplementedError, match="not strictly earlier"):
        validate_for_orchestrator_v1(md)


def test_fan_in_does_not_lift_background_not_gated_rejection():
    """A fire-and-forget (pre-session SPAWN) branch referenced by a SPAWN_JOIN prereq is still rejected."""
    conv = ConversationMetadata(
        conversation_id="r",
        turns=[
            TurnMetadata(branch_ids=["r:pre", "r:0:ok"]),
            TurnMetadata(
                prerequisites=[
                    TurnPrerequisite(
                        kind=PrerequisiteKind.SPAWN_JOIN, branch_id="r:pre"
                    ),
                    TurnPrerequisite(
                        kind=PrerequisiteKind.SPAWN_JOIN, branch_id="r:0:ok"
                    ),
                ]
            ),
        ],
        branches=[
            ConversationBranchInfo(
                branch_id="r:pre",
                child_conversation_ids=["cb"],
                mode=ConversationBranchMode.SPAWN,
                is_background=True,
                dispatch_timing="pre",
            ),
            ConversationBranchInfo(
                branch_id="r:0:ok",
                child_conversation_ids=["co"],
                mode=ConversationBranchMode.SPAWN,
            ),
        ],
    )
    md = DatasetMetadata(
        conversations=[conv, _mk_child("cb"), _mk_child("co")],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    with pytest.raises(NotImplementedError, match="fire-and-forget"):
        validate_for_orchestrator_v1(md)


def test_fan_in_does_not_lift_non_spawn_join_rejection():
    """Non-SPAWN_JOIN prereq kinds are still rejected even on a multi-prereq turn."""
    conv = ConversationMetadata(
        conversation_id="r",
        turns=[
            TurnMetadata(branch_ids=["r:0:A", "r:0:B"]),
            TurnMetadata(
                prerequisites=[
                    TurnPrerequisite(
                        kind=PrerequisiteKind.SPAWN_JOIN, branch_id="r:0:A"
                    ),
                    TurnPrerequisite(kind=PrerequisiteKind.BARRIER, branch_id="r:0:B"),
                ]
            ),
        ],
        branches=[
            ConversationBranchInfo(
                branch_id="r:0:A",
                child_conversation_ids=["c"],
                mode=ConversationBranchMode.SPAWN,
            ),
            ConversationBranchInfo(
                branch_id="r:0:B",
                child_conversation_ids=["c2"],
                mode=ConversationBranchMode.SPAWN,
            ),
        ],
    )
    md = DatasetMetadata(
        conversations=[conv, _mk_child("c"), _mk_child("c2")],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    with pytest.raises(NotImplementedError, match="not supported by v1 orchestrator"):
        validate_for_orchestrator_v1(md)


def test_fan_in_does_not_lift_duplicate_branch_id_on_same_turn():
    """Declaring the same branch_id twice on a single turn remains rejected."""
    conv = ConversationMetadata(
        conversation_id="r",
        turns=[
            TurnMetadata(branch_ids=["r:0", "r:0"]),
            TurnMetadata(),
        ],
        branches=[
            ConversationBranchInfo(
                branch_id="r:0",
                child_conversation_ids=["c"],
                mode=ConversationBranchMode.SPAWN,
            )
        ],
    )
    md = DatasetMetadata(
        conversations=[conv, _mk_child("c")],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    with pytest.raises(NotImplementedError, match="multiple times"):
        validate_for_orchestrator_v1(md)
