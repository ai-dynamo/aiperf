# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Phase lifecycle event publisher.

Publishes phase events (start, progress, complete) to message bus.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aiperf.common.enums import ProfileCancelReason
from aiperf.common.messages import ProfileCancelCommand
from aiperf.credit.messages import (
    CreditPhaseCompleteMessage,
    CreditPhaseProgressMessage,
    CreditPhasesConfiguredMessage,
    CreditPhaseSendingCompleteMessage,
    CreditPhaseStartMessage,
    CreditsCompleteMessage,
)

if TYPE_CHECKING:
    from aiperf.common.models import CreditPhaseStats
    from aiperf.common.models.branch_stats import BranchStats
    from aiperf.common.protocols import PubClientProtocol
    from aiperf.timing.config import CreditPhaseConfig


class PhasePublisher:
    """Publishes phase lifecycle events to message bus.

    Events: phase start, progress updates, sending complete, phase complete,
            credits complete.
    """

    def __init__(
        self,
        *,
        pub_client: PubClientProtocol,
        service_id: str,
    ):
        """Initialize publisher with message bus client."""
        self._pub_client = pub_client
        self._service_id = service_id

    async def publish_phases_configured(self, configs: list[CreditPhaseConfig]) -> None:
        """Publish phases configured event."""
        msg = CreditPhasesConfiguredMessage(
            service_id=self._service_id,
            configs=configs,
        )
        await self._pub_client.publish(msg)

    async def publish_phase_start(
        self, config: CreditPhaseConfig, phase_stats: CreditPhaseStats
    ) -> None:
        """Publish phase start event."""
        msg = CreditPhaseStartMessage(
            service_id=self._service_id,
            stats=phase_stats,
            config=config,
        )
        await self._pub_client.publish(msg)

    async def publish_phase_sending_complete(
        self, phase_stats: CreditPhaseStats
    ) -> None:
        """Publish phase sending complete event."""
        msg = CreditPhaseSendingCompleteMessage(
            service_id=self._service_id,
            stats=phase_stats,
        )
        await self._pub_client.publish(msg)

    async def publish_phase_complete(
        self,
        phase_stats: CreditPhaseStats,
        branch_stats: BranchStats | None = None,
    ) -> None:
        """Publish phase complete event.

        Args:
            phase_stats: Credit phase stats snapshot.
            branch_stats: Optional DAG sub-agent orchestrator counters for this
                phase. ``None`` when no orchestrator is attached or no children
                were spawned.
        """
        msg = CreditPhaseCompleteMessage(
            service_id=self._service_id,
            stats=phase_stats,
            branch_stats=branch_stats,
        )
        await self._pub_client.publish(msg)

    async def publish_progress(self, phase_stats: CreditPhaseStats) -> None:
        """Publish progress update."""
        msg = CreditPhaseProgressMessage(
            service_id=self._service_id,
            stats=phase_stats,
        )
        await self._pub_client.publish(msg)

    async def publish_credits_complete(self) -> None:
        """Publish credits complete event."""
        msg = CreditsCompleteMessage(service_id=self._service_id)
        await self._pub_client.publish(msg)

    async def publish_profile_cancel(
        self, reason: ProfileCancelReason = ProfileCancelReason.WARMUP_FAILURE
    ) -> None:
        """Broadcast ProfileCancelCommand to abort the run.

        Used by the agentic-replay WARMUP early-abort path: a terminal warmup
        failure means PROFILING must not start, so we broadcast the same command
        the ``--failed-request-threshold`` profiling abort uses. The records
        manager finalizes (cancelled) and the system controller shuts down
        (with a non-zero exit, since ``reason`` is an abort), instead of warmup
        running to teardown and hanging the run. Reuses the existing cancel
        handlers in timing/records/server-metrics/gpu-telemetry.
        """
        msg = ProfileCancelCommand(service_id=self._service_id, reason=reason)
        await self._pub_client.publish(msg)
