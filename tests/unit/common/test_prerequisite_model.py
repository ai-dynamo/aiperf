# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from aiperf.common.enums import PrerequisiteKind
from aiperf.common.models import TurnPrerequisite


def test_minimal_spawn_join_prereq():
    p = TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="root:0")
    assert p.kind == PrerequisiteKind.SPAWN_JOIN
    assert p.branch_id == "root:0"
    assert p.child_conversation_ids is None
    assert p.barrier_id is None
    assert p.timer_seconds is None
    assert p.event_name is None


def test_prereq_is_frozen():
    p = TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="root:0")
    with pytest.raises(ValidationError):
        p.branch_id = "other"


def test_prereq_extra_field_forbidden():
    with pytest.raises(ValidationError):
        TurnPrerequisite(
            kind=PrerequisiteKind.SPAWN_JOIN, branch_id="root:0", garbage="x"
        )


def test_prereq_reserved_fields_accepted_at_model_layer():
    """Model accepts reserved fields; validator (separate task) rejects them."""
    p = TurnPrerequisite(
        kind=PrerequisiteKind.TIMER,
        timer_seconds=5.0,
        barrier_id="b1",
        child_conversation_ids=["c1"],
        event_name="e",
    )
    assert p.timer_seconds == 5.0
    assert p.barrier_id == "b1"
    assert p.child_conversation_ids == ["c1"]
    assert p.event_name == "e"
