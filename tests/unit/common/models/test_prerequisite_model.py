# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from aiperf.common.enums import PrerequisiteKind
from aiperf.common.models.prerequisites import TurnPrerequisite


class TestPrerequisiteKind:
    def test_members(self):
        assert PrerequisiteKind.SPAWN_JOIN == "spawn_join"
        assert PrerequisiteKind.CHILD_SESSION_COMPLETE == "child_session_complete"
        assert PrerequisiteKind.TIMER == "timer"
        assert PrerequisiteKind.EXTERNAL_EVENT == "external_event"
        assert PrerequisiteKind.BARRIER == "barrier"


class TestTurnPrerequisite:
    def test_construct_spawn_join(self):
        p = TurnPrerequisite(
            kind=PrerequisiteKind.SPAWN_JOIN,
            branch_id="root:0",
        )
        assert p.kind is PrerequisiteKind.SPAWN_JOIN
        assert p.branch_id == "root:0"
        assert p.child_conversation_ids is None
        assert p.barrier_id is None
        assert p.timer_seconds is None
        assert p.event_name is None

    def test_serialization_round_trip(self):
        p = TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="root:0")
        dumped = p.model_dump()
        restored = TurnPrerequisite.model_validate(dumped)
        assert restored == p

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            TurnPrerequisite(
                kind=PrerequisiteKind.SPAWN_JOIN,
                branch_id="root:0",
                unknown_field="boom",
            )

    def test_frozen(self):
        p = TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="root:0")
        with pytest.raises(ValidationError):
            p.branch_id = "other:1"

    def test_construct_timer_reserved(self):
        p = TurnPrerequisite(kind=PrerequisiteKind.TIMER, timer_seconds=1.5)
        assert p.kind is PrerequisiteKind.TIMER
        assert p.timer_seconds == 1.5
