# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from aiperf.common.messages import (
    CommandMessage,
    PhaseEndGateCommand,
    PhaseGateGrantedResponse,
    PhaseStartGateCommand,
)
from aiperf.timing.phase.phase_gate import PhaseGateClient


class _StubSender:
    def __init__(self) -> None:
        self.sent: list[CommandMessage] = []

    async def send_command_and_wait_for_response(
        self, cmd: CommandMessage, timeout: float | None = None
    ) -> PhaseGateGrantedResponse:
        self.sent.append(cmd)
        return PhaseGateGrantedResponse(
            command_id=cmd.command_id,
            service_id="system_controller",
            command=cmd.command,
            phase_id=cmd.phase_id,
        )


@pytest.mark.asyncio
async def test_before_phase_sends_start_gate() -> None:
    sender = _StubSender()
    gate = PhaseGateClient(
        sender=sender,
        service_id="timing_manager_test",
        enabled=True,
        timeout_s=5.0,
    )
    await gate.before_phase("p1", "profiling")
    assert len(sender.sent) == 1
    assert isinstance(sender.sent[0], PhaseStartGateCommand)
    assert sender.sent[0].phase_id == "p1"
    assert sender.sent[0].phase_name == "profiling"
    assert sender.sent[0].service_id == "timing_manager_test"


@pytest.mark.asyncio
async def test_after_phase_sends_end_gate() -> None:
    sender = _StubSender()
    gate = PhaseGateClient(
        sender=sender,
        service_id="timing_manager_test",
        enabled=True,
        timeout_s=5.0,
    )
    await gate.after_phase("p1", "profiling")
    assert len(sender.sent) == 1
    assert isinstance(sender.sent[0], PhaseEndGateCommand)
    assert sender.sent[0].phase_id == "p1"
    assert sender.sent[0].phase_name == "profiling"
    assert sender.sent[0].service_id == "timing_manager_test"


@pytest.mark.asyncio
async def test_disabled_short_circuits() -> None:
    sender = _StubSender()
    gate = PhaseGateClient(
        sender=sender,
        service_id="timing_manager_test",
        enabled=False,
        timeout_s=5.0,
    )
    await gate.before_phase("p1", "profiling")
    await gate.after_phase("p1", "profiling")
    assert sender.sent == []
