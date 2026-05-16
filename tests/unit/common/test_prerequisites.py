# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import msgspec
import pytest

from aiperf.common.enums import PrerequisiteKind
from aiperf.common.models import TurnPrerequisite
from aiperf.common.models.base_models import _msgspec_dec_hook, _msgspec_enc_hook


def test_prerequisite_kind_values():
    assert PrerequisiteKind.SPAWN_JOIN == "spawn_join"


def test_prerequisite_kind_is_case_insensitive():
    assert PrerequisiteKind("SPAWN_JOIN") == PrerequisiteKind.SPAWN_JOIN
    assert PrerequisiteKind("spawn_join") == PrerequisiteKind.SPAWN_JOIN


def test_turn_prerequisite_minimal_spawn_join():
    p = TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="root:0")
    assert p.kind == PrerequisiteKind.SPAWN_JOIN
    assert p.branch_id == "root:0"
    assert p.child_conversation_ids is None
    assert p.barrier_id is None
    assert p.timer_seconds is None
    assert p.event_name is None


def test_turn_prerequisite_reserved_fields_accepted():
    p = TurnPrerequisite(
        kind=PrerequisiteKind.BARRIER,
        barrier_id="b1",
        timer_seconds=1.5,
        event_name="evt",
        child_conversation_ids=["c1", "c2"],
    )
    assert p.barrier_id == "b1"
    assert p.timer_seconds == 1.5
    assert p.event_name == "evt"
    assert p.child_conversation_ids == ["c1", "c2"]


def test_turn_prerequisite_round_trip_json():
    p = TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="root:0")
    encoded = msgspec.json.encode(p, enc_hook=_msgspec_enc_hook)
    restored = msgspec.convert(
        msgspec.json.decode(encoded), TurnPrerequisite, dec_hook=_msgspec_dec_hook
    )
    assert restored == p


def test_turn_prerequisite_forbids_unknown_fields():
    with pytest.raises(msgspec.ValidationError):
        msgspec.convert(
            {"kind": "spawn_join", "branch_id": "x", "unknown_field": 1},
            TurnPrerequisite,
            dec_hook=_msgspec_dec_hook,
        )
