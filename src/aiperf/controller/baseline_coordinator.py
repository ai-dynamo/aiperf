# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SystemController-side coordinator for the phase baseline handshake.

Owns the set of services that registered with capability BASELINE_COLLECTOR,
fans out PhaseBaselineRequestMessage, and gathers PhaseBaselineAckMessage
responses with a per-gate timeout.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.enums import BaselineKind
from aiperf.common.messages import (
    PhaseBaselineAckMessage,
    PhaseBaselineRequestMessage,
)

_logger = AIPerfLogger(__name__)


class BaselineCoordinator:
    """Coordinates pre/post-phase baseline scrapes across registered collectors."""

    def __init__(
        self,
        publish: Callable[[PhaseBaselineRequestMessage], Awaitable[None]],
        gate_timeout_s: float,
    ) -> None:
        self._publish = publish
        self._gate_timeout_s = gate_timeout_s
        self._registered: set[str] = set()
        self._inflight: dict[
            tuple[str, BaselineKind],
            dict[str, asyncio.Future[PhaseBaselineAckMessage]],
        ] = {}

    @property
    def registered_count(self) -> int:
        return len(self._registered)

    def register(self, service_id: str) -> None:
        """Add a service to the registered baseline-collector set. Idempotent."""
        self._registered.add(service_id)

    def unregister(self, service_id: str) -> None:
        """Remove a service (e.g., on heartbeat-loss eviction). No-op if absent."""
        self._registered.discard(service_id)

    def handle_ack(self, ack: PhaseBaselineAckMessage) -> None:
        """Resolve the pending future for (phase_id, kind, service_id), if any."""
        pending = self._inflight.get((ack.phase_id, ack.kind))
        if pending is None:
            return
        fut = pending.get(ack.service_id)
        if fut is None or fut.done():
            return
        fut.set_result(ack)

    async def gate_phase(
        self, phase_id: str, phase_name: str, kind: BaselineKind
    ) -> None:
        """Block until all currently-registered collectors ack, or timeout fires."""
        registered = tuple(self._registered)
        if not registered:
            return

        pending: dict[str, asyncio.Future[PhaseBaselineAckMessage]] = {
            sid: asyncio.get_running_loop().create_future() for sid in registered
        }
        self._inflight[(phase_id, kind)] = pending

        await self._publish(
            PhaseBaselineRequestMessage(
                phase_id=phase_id, phase_name=phase_name, kind=kind
            )
        )

        try:
            results = await asyncio.wait_for(
                asyncio.gather(*pending.values(), return_exceptions=True),
                timeout=self._gate_timeout_s,
            )
            for ack in results:
                if isinstance(ack, PhaseBaselineAckMessage) and not ack.success:
                    _logger.warning(
                        f"Baseline {kind} for phase '{phase_name}' "
                        f"(id={phase_id[:8]}) collector {ack.service_id!r} "
                        f"reported failure: {ack.error}"
                    )
        except TimeoutError:
            unacked: list[str] = []
            for sid, f in pending.items():
                # A future is "acked" only if it completed via set_result (not
                # via cancellation from the wait_for timeout).
                if f.cancelled() or not f.done():
                    unacked.append(sid)
            unacked.sort()
            _logger.warning(
                f"Baseline {kind} gate for phase '{phase_name}' "
                f"(id={phase_id[:8]}) timed out after {self._gate_timeout_s}s; "
                f"proceeding without acks from {unacked}. "
                f"Increase AIPERF_BASELINE_GATE_TIMEOUT_S or set "
                f"AIPERF_BASELINE_GATE_ENABLED=0 to disable."
            )
        finally:
            self._inflight.pop((phase_id, kind), None)
