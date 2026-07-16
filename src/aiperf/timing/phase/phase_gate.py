# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TimingManager-side client for the phase baseline handshake.

A thin wrapper around send_command_and_wait_for_response that hides
PhaseStartGateCommand / PhaseEndGateCommand from PhaseRunner. Owns no
knowledge of telemetry, server-metrics, or any specific collector.
"""

from __future__ import annotations

import uuid
from typing import Protocol, TypeVar

from aiperf.common.messages import (
    CommandMessage,
    CommandResponse,
    PhaseEndGateCommand,
    PhaseStartGateCommand,
)
from aiperf.common.models.error_models import ErrorDetails
from aiperf.plugin.enums import ServiceType

_GateCommandT = TypeVar("_GateCommandT", PhaseStartGateCommand, PhaseEndGateCommand)


class _CommandSender(Protocol):
    async def send_command_and_wait_for_response(
        self, message: CommandMessage, timeout: float = ...
    ) -> CommandResponse | ErrorDetails: ...


class PhaseGateClient:
    """Sends PhaseStartGate/PhaseEndGate commands and waits for the response.

    The gate semantic is "released" if any response comes back at all; the
    payload is intentionally not inspected. On timeout the underlying sender
    returns an `ErrorDetails` (it does not raise), and the client treats that
    the same as a release so the benchmark keeps moving.
    """

    def __init__(
        self,
        sender: _CommandSender,
        service_id: str,
        enabled: bool,
        timeout_s: float,
    ) -> None:
        self._sender = sender
        self._service_id = service_id
        self._enabled = enabled
        self._timeout_s = timeout_s

    async def before_phase(self, phase_id: str, phase_name: str) -> None:
        """Block until SystemController releases the START gate (or timeout)."""
        await self._send_gate(PhaseStartGateCommand, phase_id, phase_name)

    async def after_phase(self, phase_id: str, phase_name: str) -> None:
        """Block until SystemController releases the END gate (or timeout)."""
        await self._send_gate(PhaseEndGateCommand, phase_id, phase_name)

    async def _send_gate(
        self,
        command_type: type[_GateCommandT],
        phase_id: str,
        phase_name: str,
    ) -> None:
        if not self._enabled:
            return
        await self._sender.send_command_and_wait_for_response(
            command_type(
                command_id=str(uuid.uuid4()),
                service_id=self._service_id,
                target_service_type=ServiceType.SYSTEM_CONTROLLER,
                phase_id=phase_id,
                phase_name=phase_name,
            ),
            timeout=self._timeout_s + 1.0,
        )
