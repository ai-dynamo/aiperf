# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mixin for services that take pre/post-phase baseline readings.

Subclasses implement ``collect_baseline(kind, phase_id, phase_name)`` and gain:
- automatic ServiceCapability.BASELINE_COLLECTOR registration via extra_capabilities
- a PhaseBaselineRequestMessage handler that calls collect_baseline and publishes
  PhaseBaselineAckMessage with success/error.

The mixin assumes the host class provides ``self.publish(msg)`` and ``self.service_id``
(both satisfied by BaseComponentService).
"""

from __future__ import annotations

from abc import abstractmethod
from typing import ClassVar

from aiperf.common.enums import BaselineKind, MessageType, ServiceCapability
from aiperf.common.hooks import on_message
from aiperf.common.messages import (
    PhaseBaselineAckMessage,
    PhaseBaselineRequestMessage,
)


class BaselineCollectorMixin:
    """Mix into a BaseComponentService to participate in the phase baseline handshake."""

    extra_capabilities: ClassVar[tuple[str, ...]] = (
        ServiceCapability.BASELINE_COLLECTOR,
    )

    @abstractmethod
    async def collect_baseline(
        self, kind: BaselineKind, phase_id: str, phase_name: str
    ) -> None:
        """Take a single point-in-time baseline reading.

        Implementations MUST be idempotent under retries (rare) and MUST NOT
        block longer than AIPERF_BASELINE_GATE_TIMEOUT_S; the coordinator
        will release the gate without their ack on timeout.
        """

    @on_message(MessageType.PHASE_BASELINE_REQUEST)
    async def _on_phase_baseline_request(
        self, message: PhaseBaselineRequestMessage
    ) -> None:
        """Drive collect_baseline and publish an ack with success/error status."""
        success = True
        error: str | None = None
        try:
            await self.collect_baseline(
                message.kind, message.phase_id, message.phase_name
            )
        except Exception as exc:  # per-collector fault tolerance
            success = False
            error = f"{type(exc).__name__}: {exc}"

        await self.publish(
            PhaseBaselineAckMessage(
                service_id=self.service_id,
                phase_id=message.phase_id,
                kind=message.kind,
                success=success,
                error=error,
            )
        )
