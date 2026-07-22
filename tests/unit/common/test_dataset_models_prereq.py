# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from aiperf.common.enums import PrerequisiteKind
from aiperf.common.models import Turn, TurnMetadata, TurnPrerequisite


def test_turn_defaults_empty_prerequisites() -> None:
    t = Turn()
    assert t.prerequisites == []


def test_turn_defaults_empty_branch_ids() -> None:
    t = Turn()
    assert t.branch_ids == []


def test_turn_carries_prerequisites() -> None:
    p = TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="b1")
    t = Turn(prerequisites=[p])
    assert len(t.prerequisites) == 1
    assert t.prerequisites[0].branch_id == "b1"


def test_turn_metadata_carries_prerequisites() -> None:
    p = TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="b1")
    m = TurnMetadata(prerequisites=[p])
    assert m.prerequisites == [p]


def test_turn_metadata_preserves_prerequisites_and_branch_ids() -> None:
    p = TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="b1")
    t = Turn(prerequisites=[p], branch_ids=["b1"])
    m = t.metadata()
    assert m.prerequisites == [p]
    assert m.branch_ids == ["b1"]
