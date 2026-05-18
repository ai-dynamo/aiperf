# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Phase 3 loader + validator round-trip: fan-in topologies.

Exercises DAG JSONL authoring shapes that were rejected in earlier phases
and are now accepted end-to-end:

- Two branches spawned on different parent turns both gate a single later
  turn (multi-source gate).
- One branch consumed by prereqs on two different gated turns (multi-consumer
  branch).

Because the current ``DagJsonlLoader`` wire format emits exactly one branch
per ``spawns`` list per turn, we hand-author the ``prerequisites`` in
metadata and run the validator directly. Loader-level multi-source
shorthand is out of scope for Phase 3.
"""

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


def _child(cid: str) -> ConversationMetadata:
    return ConversationMetadata(conversation_id=cid, turns=[TurnMetadata()])


def test_fan_in_authored_via_explicit_prerequisites():
    """A gated turn with two explicit SPAWN_JOIN prereqs from two different
    earlier spawning turns validates and the metadata round-trips."""
    conv = ConversationMetadata(
        conversation_id="root",
        turns=[
            TurnMetadata(branch_ids=["root:0:A"]),
            TurnMetadata(),
            TurnMetadata(branch_ids=["root:2:B"]),
            TurnMetadata(),
            TurnMetadata(),
            TurnMetadata(
                prerequisites=[
                    TurnPrerequisite(
                        kind=PrerequisiteKind.SPAWN_JOIN, branch_id="root:0:A"
                    ),
                    TurnPrerequisite(
                        kind=PrerequisiteKind.SPAWN_JOIN, branch_id="root:2:B"
                    ),
                ]
            ),
        ],
        branches=[
            ConversationBranchInfo(
                branch_id="root:0:A",
                child_conversation_ids=["a1"],
                mode=ConversationBranchMode.SPAWN,
            ),
            ConversationBranchInfo(
                branch_id="root:2:B",
                child_conversation_ids=["b1"],
                mode=ConversationBranchMode.SPAWN,
            ),
        ],
    )
    md = DatasetMetadata(
        conversations=[conv, _child("a1"), _child("b1")],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    validate_for_orchestrator_v1(md)

    # Metadata survives round-trip of branch_ids + prereq branch_ids.
    root = md.conversations[0]
    assert {b.branch_id for b in root.branches} == {"root:0:A", "root:2:B"}
    assert [p.branch_id for p in root.turns[5].prerequisites] == [
        "root:0:A",
        "root:2:B",
    ]


def test_branch_consumed_by_multiple_gates():
    """One branch_id referenced by prereqs on two different gated turns
    validates; the orchestrator installs one pending-join per gated turn."""
    conv = ConversationMetadata(
        conversation_id="root",
        turns=[
            TurnMetadata(branch_ids=["root:0"]),
            TurnMetadata(
                prerequisites=[
                    TurnPrerequisite(
                        kind=PrerequisiteKind.SPAWN_JOIN, branch_id="root:0"
                    )
                ]
            ),
            TurnMetadata(
                prerequisites=[
                    TurnPrerequisite(
                        kind=PrerequisiteKind.SPAWN_JOIN, branch_id="root:0"
                    )
                ]
            ),
        ],
        branches=[
            ConversationBranchInfo(
                branch_id="root:0",
                child_conversation_ids=["c1"],
                mode=ConversationBranchMode.SPAWN,
            )
        ],
    )
    md = DatasetMetadata(
        conversations=[conv, _child("c1")],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    validate_for_orchestrator_v1(md)


def test_fan_in_does_not_bypass_forward_ref_rejection():
    """Fan-in acceptance does not excuse forward-ref SPAWN_JOIN."""
    conv = ConversationMetadata(
        conversation_id="root",
        turns=[
            TurnMetadata(
                prerequisites=[
                    TurnPrerequisite(
                        kind=PrerequisiteKind.SPAWN_JOIN, branch_id="root:1"
                    ),
                ]
            ),
            TurnMetadata(branch_ids=["root:1"]),
        ],
        branches=[
            ConversationBranchInfo(
                branch_id="root:1",
                child_conversation_ids=["c"],
                mode=ConversationBranchMode.SPAWN,
            )
        ],
    )
    md = DatasetMetadata(
        conversations=[conv, _child("c")],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    with pytest.raises(NotImplementedError, match="not strictly earlier"):
        validate_for_orchestrator_v1(md)
