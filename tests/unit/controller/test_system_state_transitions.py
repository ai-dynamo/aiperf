# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``SystemController._set_system_state`` and the six wired call
sites that advance the controller's outer-lifecycle ``SystemState``.

The setter is the single mutation point for ``_system_state``. It is:
  * idempotent — no-op (no log, no publish) on same-state calls,
  * single-shot publisher — exactly one ``SystemStateChangedMessage`` per
    real transition, carrying the controller's ``service_id`` and the new
    state.

The full sequence under a healthy run is::

    INITIALIZING -> CONFIGURING -> READY -> PROFILING
                 -> PROCESSING -> STOPPING -> SHUTDOWN
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from aiperf.common.enums import SystemState
from aiperf.common.messages import SystemStateChangedMessage
from aiperf.controller.system_controller import SystemController


@pytest.fixture
def controller_with_mock_publish(
    system_controller: SystemController,
) -> SystemController:
    """Replace ``publish`` with an ``AsyncMock`` so we can assert the
    SystemStateChangedMessage payload without a live bus."""
    system_controller.publish = AsyncMock()  # type: ignore[method-assign]
    return system_controller


class TestSystemStateInitialDefault:
    def test_system_state_initializes_to_initializing(
        self, system_controller: SystemController
    ) -> None:
        assert system_controller._system_state == SystemState.INITIALIZING


class TestSetSystemStateIdempotency:
    @pytest.mark.asyncio
    async def test_set_system_state_same_state_is_noop(
        self, controller_with_mock_publish: SystemController
    ) -> None:
        ctrl = controller_with_mock_publish
        # Default is INITIALIZING; setting to INITIALIZING again should be a no-op.
        await ctrl._set_system_state(SystemState.INITIALIZING)
        assert ctrl._system_state == SystemState.INITIALIZING
        ctrl.publish.assert_not_called()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_set_system_state_repeated_transition_only_publishes_once(
        self, controller_with_mock_publish: SystemController
    ) -> None:
        ctrl = controller_with_mock_publish
        await ctrl._set_system_state(SystemState.CONFIGURING)
        await ctrl._set_system_state(SystemState.CONFIGURING)
        await ctrl._set_system_state(SystemState.CONFIGURING)
        assert ctrl.publish.call_count == 1  # type: ignore[attr-defined]
        assert ctrl._system_state == SystemState.CONFIGURING


class TestSetSystemStatePublish:
    @pytest.mark.asyncio
    async def test_set_system_state_publishes_message_with_state(
        self, controller_with_mock_publish: SystemController
    ) -> None:
        ctrl = controller_with_mock_publish
        await ctrl._set_system_state(SystemState.PROFILING)
        ctrl.publish.assert_awaited_once()  # type: ignore[attr-defined]
        published = ctrl.publish.await_args.args[0]  # type: ignore[attr-defined]
        assert isinstance(published, SystemStateChangedMessage)
        assert published.state == SystemState.PROFILING
        assert published.service_id == ctrl.service_id

    @pytest.mark.asyncio
    async def test_set_system_state_updates_field_before_publishing(
        self, controller_with_mock_publish: SystemController
    ) -> None:
        """A subscriber that re-reads ``_system_state`` from the controller
        in response to the message must observe the new value, not the
        old one. Update-before-publish is the contract."""
        ctrl = controller_with_mock_publish
        observed: list[SystemState] = []

        async def capture(_message: SystemStateChangedMessage) -> None:
            observed.append(ctrl._system_state)

        ctrl.publish = AsyncMock(side_effect=capture)  # type: ignore[method-assign]
        await ctrl._set_system_state(SystemState.READY)
        assert observed == [SystemState.READY]


class TestSystemStateFullSequence:
    """Walk the six real transitions in order and assert the field
    advances and exactly six SystemStateChangedMessages are published."""

    @pytest.mark.asyncio
    async def test_full_lifecycle_walk(
        self, controller_with_mock_publish: SystemController
    ) -> None:
        ctrl = controller_with_mock_publish
        sequence = [
            SystemState.CONFIGURING,
            SystemState.READY,
            SystemState.PROFILING,
            SystemState.PROCESSING,
            SystemState.STOPPING,
            SystemState.SHUTDOWN,
        ]
        for state in sequence:
            await ctrl._set_system_state(state)
            assert ctrl._system_state == state

        assert ctrl.publish.await_count == len(sequence)  # type: ignore[attr-defined]
        published_states = [
            call.args[0].state
            for call in ctrl.publish.await_args_list  # type: ignore[attr-defined]
        ]
        assert published_states == sequence

    @pytest.mark.asyncio
    async def test_full_lifecycle_with_intermixed_noops(
        self, controller_with_mock_publish: SystemController
    ) -> None:
        """Re-stamping the current state mid-sequence does not emit a
        duplicate message, but the next real transition still does."""
        ctrl = controller_with_mock_publish
        await ctrl._set_system_state(SystemState.CONFIGURING)
        await ctrl._set_system_state(SystemState.CONFIGURING)  # no-op
        await ctrl._set_system_state(SystemState.READY)
        await ctrl._set_system_state(SystemState.READY)  # no-op
        await ctrl._set_system_state(SystemState.PROFILING)

        assert ctrl.publish.await_count == 3  # type: ignore[attr-defined]
        published_states = [
            call.args[0].state
            for call in ctrl.publish.await_args_list  # type: ignore[attr-defined]
        ]
        assert published_states == [
            SystemState.CONFIGURING,
            SystemState.READY,
            SystemState.PROFILING,
        ]
        assert ctrl._system_state == SystemState.PROFILING
