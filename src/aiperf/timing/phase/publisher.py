# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Phase lifecycle event publisher.

Publishes phase events (start, progress, complete) to message bus.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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

    async def publish_phase_complete(self, phase_stats: CreditPhaseStats) -> None:
        """Publish phase complete event."""
        msg = CreditPhaseCompleteMessage(
            service_id=self._service_id,
            stats=phase_stats,
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
