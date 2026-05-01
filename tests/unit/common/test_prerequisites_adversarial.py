# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import math

import pytest
from pydantic import ValidationError

from aiperf.common.enums import PrerequisiteKind
from aiperf.common.models import TurnPrerequisite


def test_turn_prerequisite_rejects_unknown_kind_string():
    """Unknown kind strings must fail enum validation."""
    with pytest.raises(ValidationError):
        TurnPrerequisite.model_validate({"kind": "notreal"})


def test_turn_prerequisite_rejects_missing_kind():
    """`kind` is required; empty payload must fail."""
    with pytest.raises(ValidationError):
        TurnPrerequisite.model_validate({})


def test_turn_prerequisite_accepts_all_reserved_fields_simultaneously():
    """Every reserved field may be set at once on a single prerequisite."""
    p = TurnPrerequisite.model_validate(
        {
            "kind": "spawn_join",
            "branch_id": "b",
            "child_conversation_ids": ["c"],
            "barrier_id": "bar",
            "timer_seconds": 1.0,
            "event_name": "e",
        }
    )
    assert p.kind == PrerequisiteKind.SPAWN_JOIN
    assert p.branch_id == "b"
    assert p.child_conversation_ids == ["c"]
    assert p.barrier_id == "bar"
    assert p.timer_seconds == 1.0
    assert p.event_name == "e"


def test_turn_prerequisite_accepts_empty_string_branch_id():
    """Empty string is a structurally valid `branch_id`; semantic checks live elsewhere."""
    p = TurnPrerequisite.model_validate({"kind": "spawn_join", "branch_id": ""})
    assert p.branch_id == ""


def test_turn_prerequisite_accepts_negative_timer_seconds():
    """Negative timer values parse; documents a schema gap (no ge=0 constraint)."""
    p = TurnPrerequisite.model_validate({"kind": "timer", "timer_seconds": -1.0})
    assert p.timer_seconds == -1.0


def test_turn_prerequisite_accepts_nan_timer_seconds():
    """NaN timer values parse; documents a schema gap (no finite-float constraint)."""
    p = TurnPrerequisite.model_validate({"kind": "timer", "timer_seconds": math.nan})
    assert math.isnan(p.timer_seconds)


def test_turn_prerequisite_accepts_empty_child_conversation_ids_list():
    """Empty list is distinct from `None`; orchestrator treats `is not None` as per-child subset."""
    p = TurnPrerequisite.model_validate(
        {"kind": "spawn_join", "child_conversation_ids": []}
    )
    assert p.child_conversation_ids == []
    assert p.child_conversation_ids is not None


def test_turn_prerequisite_rejects_extra_field():
    """`extra="forbid"` must reject unknown keys."""
    with pytest.raises(ValidationError):
        TurnPrerequisite.model_validate({"kind": "spawn_join", "foo": "bar"})


def test_turn_prerequisite_json_roundtrip_preserves_all_reserved_fields():
    """Every reserved field survives a JSON dump/parse round trip."""
    original = TurnPrerequisite(
        kind=PrerequisiteKind.SPAWN_JOIN,
        branch_id="b",
        child_conversation_ids=["c1", "c2"],
        barrier_id="bar",
        timer_seconds=2.5,
        event_name="evt",
    )
    restored = TurnPrerequisite.model_validate_json(original.model_dump_json())
    assert restored == original
    assert restored.kind == original.kind
    assert restored.branch_id == original.branch_id
    assert restored.child_conversation_ids == original.child_conversation_ids
    assert restored.barrier_id == original.barrier_id
    assert restored.timer_seconds == original.timer_seconds
    assert restored.event_name == original.event_name


def test_turn_prerequisite_rejects_integer_branch_id():
    """Pydantic v2 does not coerce int to str for `str | None` fields."""
    with pytest.raises(ValidationError):
        TurnPrerequisite.model_validate({"kind": "spawn_join", "branch_id": 123})


def test_turn_prerequisite_instances_are_frozen():
    """TurnPrerequisite is immutable post-construction: mutating an attribute
    raises ValidationError. Freezing makes aliasing safe when a TurnPrerequisite
    instance is shared between a Turn and its derived TurnMetadata (or across
    JSON round-trip boundaries).
    """
    prereq = TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="b:0")
    with pytest.raises(ValidationError):
        prereq.branch_id = "mutated"
