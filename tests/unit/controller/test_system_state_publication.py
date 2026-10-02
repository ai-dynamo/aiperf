# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for SystemController outer-lifecycle state publication.

The operator mirrors these transitions onto ``AIPerfJob.status.subPhase``, so a
transition must publish exactly once and a repeated set must publish not at all.
"""

from unittest.mock import AsyncMock

import pytest

from aiperf.common.enums import MessageType, SystemState
from aiperf.controller.system_controller import SystemController


def _state_messages(publish: AsyncMock) -> list:
    return [
        call.args[0]
        for call in publish.call_args_list
        if getattr(call.args[0], "message_type", None)
        == MessageType.SYSTEM_STATE_CHANGED
    ]


@pytest.mark.asyncio
async def test_state_transition_publishes_message(
    system_controller: SystemController,
) -> None:
    system_controller.publish = AsyncMock()
    await system_controller._set_system_state(SystemState.PROFILING)

    published = _state_messages(system_controller.publish)
    assert len(published) == 1
    assert published[0].state == SystemState.PROFILING


@pytest.mark.asyncio
async def test_repeated_state_does_not_republish(
    system_controller: SystemController,
) -> None:
    system_controller.publish = AsyncMock()
    await system_controller._set_system_state(SystemState.PROFILING)
    await system_controller._set_system_state(SystemState.PROFILING)

    published = _state_messages(system_controller.publish)
    assert len(published) == 1


@pytest.mark.asyncio
async def test_initial_state_is_initializing(
    system_controller: SystemController,
) -> None:
    assert system_controller._system_state == SystemState.INITIALIZING

    system_controller.publish = AsyncMock()
    await system_controller._set_system_state(SystemState.INITIALIZING)
    assert _state_messages(system_controller.publish) == []
