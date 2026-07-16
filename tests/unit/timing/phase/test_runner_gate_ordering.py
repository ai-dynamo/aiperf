# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ordering invariants for the phase baseline handshake.

Proves at the unit level:
  - PhaseGateClient.before_phase awaits BEFORE phase_publisher.publish_phase_start
    is called (i.e. baseline collectors get their reading slot before any credit
    is published).
  - PhaseGateClient.after_phase fires AFTER _wait_for_returning_complete settles,
    once the phase has been finalised.
  - When AIPERF_BASELINE_GATE_ENABLED=false, PhaseGateClient is a strict no-op
    and never touches the command bus.

Uses the existing make_runner / cfg test scaffolding from test_runner.py for
fixtures, plus a recording sender to capture the ordering of gate vs. publish
calls via time.monotonic_ns() (the autouse no_sleep fixture only mocks
asyncio.sleep, not time.monotonic).
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.common.messages import (
    CommandMessage,
    PhaseEndGateCommand,
    PhaseGateGrantedResponse,
    PhaseStartGateCommand,
)
from aiperf.timing.phase.phase_gate import PhaseGateClient
from tests.unit.timing.phase.test_runner import MockStrategy, cfg, make_runner


class _RecordingSender:
    """Captures gate command sends with monotonic timestamps."""

    def __init__(self, events: list[tuple[str, int]]) -> None:
        self._events = events

    async def send_command_and_wait_for_response(
        self, message: CommandMessage, timeout: float | None = None
    ) -> PhaseGateGrantedResponse:
        if isinstance(message, PhaseStartGateCommand):
            label = "before_phase"
        elif isinstance(message, PhaseEndGateCommand):
            label = "after_phase"
        else:
            label = type(message).__name__
        self._events.append((label, time.monotonic_ns()))
        return PhaseGateGrantedResponse(
            command_id=message.command_id,
            service_id="system_controller",
            command=message.command,
            phase_id=message.phase_id,
        )


def _stamping_async_mock(events: list[tuple[str, int]], label: str) -> AsyncMock:
    async def _impl(*_a: object, **_k: object) -> None:
        events.append((label, time.monotonic_ns()))

    return AsyncMock(side_effect=_impl)


@pytest.fixture
def recording_sender_events() -> list[tuple[str, int]]:
    return []


@pytest.fixture
def recording_sender(
    recording_sender_events: list[tuple[str, int]],
) -> _RecordingSender:
    return _RecordingSender(recording_sender_events)


@pytest.fixture
def stamping_pub(recording_sender_events: list[tuple[str, int]]) -> MagicMock:
    m = MagicMock()
    m.publish_phase_start = _stamping_async_mock(
        recording_sender_events, "publish_phase_start"
    )
    m.publish_phase_sending_complete = _stamping_async_mock(
        recording_sender_events, "publish_phase_sending_complete"
    )
    m.publish_phase_complete = _stamping_async_mock(
        recording_sender_events, "publish_phase_complete"
    )
    m.publish_progress = AsyncMock()
    m.publish_credits_complete = _stamping_async_mock(
        recording_sender_events, "publish_credits_complete"
    )
    return m


@pytest.fixture
def conv_src() -> MagicMock:
    m = MagicMock()
    m.next = MagicMock()
    return m


@pytest.fixture
def router() -> MagicMock:
    m = MagicMock()
    m.send_credit = m.cancel_all_credits = AsyncMock()
    m.mark_credits_complete = MagicMock()
    m.wait_for_workers = AsyncMock()
    return m


@pytest.fixture
def conc() -> MagicMock:
    m = MagicMock()
    m.configure_for_phase = MagicMock()
    m.acquire_session_slot = AsyncMock(return_value=True)
    m.acquire_prefill_slot = AsyncMock(return_value=True)
    m.release_session_slot = m.release_prefill_slot = MagicMock()
    m.set_session_limit = m.set_prefill_limit = MagicMock()
    m.release_stuck_slots = MagicMock(return_value=(0, 0))
    return m


@pytest.fixture
def cancel_pol() -> MagicMock:
    m = MagicMock()
    m.next_cancellation_delay_ns = MagicMock(return_value=None)
    return m


@pytest.fixture
def callback_handler() -> MagicMock:
    m = MagicMock()
    m.register_phase = m.unregister_phase = MagicMock()
    m.on_credit_return = m.on_first_token = AsyncMock()
    return m


@pytest.mark.asyncio
async def test_before_phase_fires_before_publish_phase_start(
    conv_src: MagicMock,
    stamping_pub: MagicMock,
    router: MagicMock,
    conc: MagicMock,
    cancel_pol: MagicMock,
    callback_handler: MagicMock,
    recording_sender: _RecordingSender,
    recording_sender_events: list[tuple[str, int]],
) -> None:
    """The START gate must release before publish_phase_start is awaited."""
    gate = PhaseGateClient(
        sender=recording_sender,
        service_id="timing_manager_test",
        enabled=True,
        timeout_s=1.0,
    )
    runner = make_runner(
        cfg(),
        conv_src,
        stamping_pub,
        router,
        conc,
        cancel_pol,
        callback_handler,
    )
    runner._phase_gate = gate

    with patch(
        "aiperf.timing.phase.runner.plugins.get_class",
        return_value=lambda **_kw: MockStrategy(),
    ):
        runner._progress.all_credits_sent_event.set()
        runner._progress.all_credits_returned_event.set()
        await runner.run(is_final_phase=True)

    labels = [e[0] for e in recording_sender_events]
    assert "before_phase" in labels, f"no before_phase event recorded; got {labels}"
    assert "publish_phase_start" in labels, (
        f"no publish_phase_start event; got {labels}"
    )

    first_before = labels.index("before_phase")
    first_publish_start = labels.index("publish_phase_start")
    assert first_before < first_publish_start, (
        f"before_phase must precede publish_phase_start; got order: {labels}"
    )


@pytest.mark.asyncio
async def test_after_phase_fires_after_publish_phase_complete(
    conv_src: MagicMock,
    stamping_pub: MagicMock,
    router: MagicMock,
    conc: MagicMock,
    cancel_pol: MagicMock,
    callback_handler: MagicMock,
    recording_sender: _RecordingSender,
    recording_sender_events: list[tuple[str, int]],
) -> None:
    """The END gate must fire on the synchronous (final, non-seamless) path."""
    gate = PhaseGateClient(
        sender=recording_sender,
        service_id="timing_manager_test",
        enabled=True,
        timeout_s=1.0,
    )
    runner = make_runner(
        cfg(),
        conv_src,
        stamping_pub,
        router,
        conc,
        cancel_pol,
        callback_handler,
    )
    runner._phase_gate = gate

    with patch(
        "aiperf.timing.phase.runner.plugins.get_class",
        return_value=lambda **_kw: MockStrategy(),
    ):
        runner._progress.all_credits_sent_event.set()
        runner._progress.all_credits_returned_event.set()
        await runner.run(is_final_phase=True)

    labels = [e[0] for e in recording_sender_events]
    assert "after_phase" in labels, f"END gate never fired; events were {labels}"
    assert "publish_phase_complete" in labels, (
        f"phase never completed publish; events were {labels}"
    )

    last_complete = max(
        idx for idx, lbl in enumerate(labels) if lbl == "publish_phase_complete"
    )
    last_after = max(idx for idx, lbl in enumerate(labels) if lbl == "after_phase")
    assert last_after > last_complete, (
        f"after_phase must follow publish_phase_complete; got order: {labels}"
    )


@pytest.mark.asyncio
async def test_disabled_gate_does_not_send_commands(
    conv_src: MagicMock,
    stamping_pub: MagicMock,
    router: MagicMock,
    conc: MagicMock,
    cancel_pol: MagicMock,
    callback_handler: MagicMock,
    recording_sender: _RecordingSender,
    recording_sender_events: list[tuple[str, int]],
) -> None:
    """When AIPERF_BASELINE_GATE_ENABLED=false the gate is a strict no-op."""
    gate = PhaseGateClient(
        sender=recording_sender,
        service_id="timing_manager_test",
        enabled=False,
        timeout_s=1.0,
    )
    runner = make_runner(
        cfg(),
        conv_src,
        stamping_pub,
        router,
        conc,
        cancel_pol,
        callback_handler,
    )
    runner._phase_gate = gate

    with patch(
        "aiperf.timing.phase.runner.plugins.get_class",
        return_value=lambda **_kw: MockStrategy(),
    ):
        runner._progress.all_credits_sent_event.set()
        runner._progress.all_credits_returned_event.set()
        await runner.run(is_final_phase=True)

    gate_labels = [
        lbl
        for lbl, _ts in recording_sender_events
        if lbl in ("before_phase", "after_phase")
    ]
    assert gate_labels == [], (
        f"disabled gate sent commands; recorded gate events: {gate_labels}"
    )
    # Sanity: phase still ran end-to-end via the regular publish path.
    pub_labels = [
        lbl for lbl, _ts in recording_sender_events if lbl.startswith("publish_")
    ]
    assert "publish_phase_start" in pub_labels
    assert "publish_phase_complete" in pub_labels
