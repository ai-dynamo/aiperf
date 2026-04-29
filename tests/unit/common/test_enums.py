# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


def test_prerequisite_kind_values():
    from aiperf.common.enums import PrerequisiteKind

    assert PrerequisiteKind.SPAWN_JOIN == "spawn_join"
    assert PrerequisiteKind.CHILD_SESSION_COMPLETE == "child_session_complete"
    assert PrerequisiteKind.TIMER == "timer"
    assert PrerequisiteKind.EXTERNAL_EVENT == "external_event"
    assert PrerequisiteKind.BARRIER == "barrier"


def test_prerequisite_kind_case_insensitive():
    from aiperf.common.enums import PrerequisiteKind

    assert PrerequisiteKind("SPAWN_JOIN") == PrerequisiteKind.SPAWN_JOIN
    assert PrerequisiteKind("spawn_join") == PrerequisiteKind.SPAWN_JOIN
