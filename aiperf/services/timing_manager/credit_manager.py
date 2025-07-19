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
from aiperf.common.mixins import (
    AIPerfLoggerProtocol,
    AsyncTaskManagerMixin,
    AsyncTaskManagerProtocol,
)


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
        total_expected_requests: int | None,
        expected_duration_sec: float | None,
    ) -> None: ...

    async def publish_phase_sending_complete(
        self, phase: CreditPhase, sent_end_ns: int
    ) -> None: ...

    async def publish_phase_complete(
        self, phase: CreditPhase, completed: int, end_ns: int
    ) -> None: ...


@runtime_checkable
class CreditPhaseMessagesRequirements(
    AsyncTaskManagerProtocol, AIPerfLoggerProtocol, Protocol
):
    """Requirements for the CreditPhaseMessagesMixin. This is the list of attributes that must
    be provided by the class that uses this mixin."""

    pub_client: PubClientProtocol
    service_id: str


class CreditPhaseMessagesMixin(AsyncTaskManagerMixin, CreditPhaseMessagesRequirements):
    """Mixin for services to implement the CreditManagerProtocol.

    Requirements:
        This mixin must be used with a class that provides:
        - pub_client: PubClientProtocol
        - service_id: str
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(self, CreditPhaseMessagesRequirements):
            raise TypeError(
                "CreditPhaseMessagesMixin must be used with a class that provides CreditPhaseMessagesRequirements"
            )

    async def publish_phase_start(
        self,
        phase: CreditPhase,
        start_ns: int,
        total_expected_requests: int | None,
        expected_duration_sec: float | None,
    ) -> None:
        """Publish the phase start message."""
        self.execute_async(
            self.pub_client.publish(
                CreditPhaseStartMessage(
                    service_id=self.service_id,
                    phase=phase,
                    start_ns=start_ns,
                    # Only one of the below will be set, this is already validated in the strategy
                    total_expected_requests=total_expected_requests,
                    expected_duration_sec=expected_duration_sec,
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

    async def publish_phase_complete(
        self, phase: CreditPhase, completed: int, end_ns: int
    ) -> None:
        """Publish the phase complete message."""
        self.execute_async(
            self.pub_client.publish(
                CreditPhaseCompleteMessage(
                    service_id=self.service_id,
                    phase=phase,
                    completed=completed,
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
        self.debug("Publishing credits complete message")
        self.execute_async(
            self.pub_client.publish(CreditsCompleteMessage(service_id=self.service_id))
        )
