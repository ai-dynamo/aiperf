# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from aiperf.common.enums import BaselineKind, CommandType, MessageType
from aiperf.common.messages import (
    PhaseBaselineAckMessage,
    PhaseBaselineRequestMessage,
    PhaseEndGateCommand,
    PhaseGateGrantedResponse,
    PhaseStartGateCommand,
)


def test_request_round_trip() -> None:
    msg = PhaseBaselineRequestMessage(
        phase_id="abc-123",
        phase_name="profiling",
        kind=BaselineKind.START,
    )
    assert msg.message_type == MessageType.PHASE_BASELINE_REQUEST
    parsed = PhaseBaselineRequestMessage.model_validate_json(msg.model_dump_json())
    assert parsed.phase_id == "abc-123"
    assert parsed.kind == BaselineKind.START


def test_ack_success_and_error() -> None:
    ok = PhaseBaselineAckMessage(
        service_id="svc-1",
        phase_id="abc",
        kind=BaselineKind.END,
        success=True,
    )
    assert ok.error is None
    bad = PhaseBaselineAckMessage(
        service_id="svc-1",
        phase_id="abc",
        kind=BaselineKind.END,
        success=False,
        error="DCGM connection refused",
    )
    assert bad.success is False
    assert "DCGM" in bad.error


def test_gate_commands_carry_phase_metadata() -> None:
    start = PhaseStartGateCommand(
        service_id="svc-1", command_id="c1", phase_id="abc", phase_name="warmup"
    )
    end = PhaseEndGateCommand(
        service_id="svc-1", command_id="c2", phase_id="abc", phase_name="warmup"
    )
    assert start.command == CommandType.PHASE_START_GATE
    assert end.command == CommandType.PHASE_END_GATE


def test_gate_granted_response() -> None:
    resp = PhaseGateGrantedResponse(
        service_id="svc-1",
        command=CommandType.PHASE_START_GATE,
        command_id="c1",
        phase_id="abc",
    )
    assert resp.phase_id == "abc"
