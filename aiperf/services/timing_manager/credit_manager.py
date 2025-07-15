# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Protocol, runtime_checkable

from aiperf.common.comms.base import PubClientProtocol
from aiperf.common.enums import CreditPhase
from aiperf.common.messages import (
    CreditPhaseCompleteMessage,
    CreditPhaseProgressMessage,
    CreditPhaseSendingCompleteMessage,
    CreditPhaseStartMessage,
    CreditsCompleteMessage,
)
from aiperf.common.mixins import AsyncTaskManagerMixin


@runtime_checkable
class CreditManagerProtocol(Protocol):
    """Defines the interface for a CreditManager.

    This is used to allow the credit issuing strategy to interact with the TimingManager
    in a decoupled way.
    """

    async def drop_credit(
        self,
        credit_phase: CreditPhase,
        conversation_id: str | None = None,
        credit_drop_ns: int | None = None,
    ) -> None: ...

    async def publish_progress(
        self, phase: CreditPhase, sent: int, completed: int
    ) -> None: ...

    async def publish_credits_complete(self) -> None: ...

    async def publish_phase_start(
        self,
        phase: CreditPhase,
        start_ns: int,
        total_requests: int | None,
        expected_duration_ns: int | None,
    ) -> None: ...

    async def publish_phase_sending_complete(
        self, phase: CreditPhase, sent_end_ns: int
    ) -> None: ...

    async def publish_phase_complete(self, phase: CreditPhase, end_ns: int) -> None: ...


class CreditPhaseMessagesMixin(AsyncTaskManagerMixin):
    """Mixin for services to implement the CreditManagerProtocol."""

    def __init__(self):
        super().__init__()
        self.pub_client: PubClientProtocol
        self.service_id: str

    async def publish_phase_start(
        self,
        phase: CreditPhase,
        start_ns: int,
        total_requests: int | None,
        expected_duration_ns: int | None,
    ) -> None:
        """Publish the phase start message."""
        self.execute_async(
            self.pub_client.publish(
                CreditPhaseStartMessage(
                    service_id=self.service_id,
                    phase=phase,
                    start_ns=start_ns,
                    total_requests=total_requests,
                    expected_duration_ns=expected_duration_ns,
                )
            )
        )

    async def publish_phase_sending_complete(
        self, phase: CreditPhase, sent_end_ns: int
    ) -> None:
        """Publish the phase sending complete message."""
        self.execute_async(
            self.pub_client.publish(
                CreditPhaseSendingCompleteMessage(
                    service_id=self.service_id,
                    phase=phase,
                    sent_end_ns=sent_end_ns,
                )
            )
        )

    async def publish_phase_complete(self, phase: CreditPhase, end_ns: int) -> None:
        """Publish the phase complete message."""
        self.execute_async(
            self.pub_client.publish(
                CreditPhaseCompleteMessage(
                    service_id=self.service_id,
                    phase=phase,
                    end_ns=end_ns,
                )
            )
        )

    async def publish_progress(
        self, phase: CreditPhase, sent: int, completed: int
    ) -> None:
        """Publish the progress message."""
        self.execute_async(
            self.pub_client.publish(
                CreditPhaseProgressMessage(
                    service_id=self.service_id,
                    phase=phase,
                    sent=sent,
                    completed=completed,
                )
            )
        )

    async def publish_credits_complete(self) -> None:
        """Publish the credits complete message."""
        self.execute_async(
            self.pub_client.publish(CreditsCompleteMessage(service_id=self.service_id))
        )
