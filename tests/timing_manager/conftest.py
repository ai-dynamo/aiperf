# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import time

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.common.messages import (
    CreditDropMessage,
    CreditPhaseCompleteMessage,
    CreditPhaseProgressMessage,
    CreditPhaseSendingCompleteMessage,
    CreditPhaseStartMessage,
    CreditReturnMessage,
    CreditsCompleteMessage,
)
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.timing import CreditIssuingStrategy


class MockCreditManager(AIPerfLoggerMixin):
    """Mock implementation of CreditManagerProtocol for testing."""

    def __init__(self):
        super().__init__()
        self.dropped_timestamps = []
        self.dropped_credits = []
        self.progress_calls = []
        self.credits_complete_calls = []
        self.phase_start_calls = []
        self.phase_complete_calls = []
        self.phase_sending_complete_calls = []
        self.credit_strategy: CreditIssuingStrategy | None = None
        self.auto_credit_return = False

    async def drop_credit(
        self,
        credit_phase: CreditPhase,
        conversation_id: str | None = None,
        credit_drop_ns: int | None = None,
    ) -> None:
        """Mock drop_credit method."""
        self.dropped_timestamps.append(time.time_ns())
        self.dropped_credits.append(
            CreditDropMessage(
                service_id="test-service",
                phase=credit_phase,
                conversation_id=conversation_id,
                credit_drop_ns=credit_drop_ns,
            )
        )
        if not self.auto_credit_return:
            return

        if self.credit_strategy is None:
            self.logger.warning("Credit strategy not set, skipping credit return")
            return

        await self.credit_strategy._on_credit_return(
            CreditReturnMessage(
                service_id="test-service",
                phase=credit_phase,
            )
        )

    async def publish_progress(
        self, phase: CreditPhase, sent: int, completed: int
    ) -> None:
        """Mock publish_progress method."""
        self.progress_calls.append(
            CreditPhaseProgressMessage(
                phase=phase,
                sent=sent,
                completed=completed,
                service_id="test-service",
            )
        )

    async def publish_credits_complete(self) -> None:
        """Mock publish_credits_complete method."""
        self.credits_complete_calls.append(
            CreditsCompleteMessage(service_id="test-service")
        )

    async def publish_phase_start(
        self,
        phase: CreditPhase,
        start_ns: int,
        total_expected_requests: int | None,
        expected_duration_sec: float | None,
    ) -> None:
        """Mock publish_phase_start method."""
        self.phase_start_calls.append(
            CreditPhaseStartMessage(
                phase=phase,
                start_ns=start_ns,
                total_expected_requests=total_expected_requests,
                expected_duration_sec=expected_duration_sec,
                service_id="test-service",
            )
        )

    async def publish_phase_sending_complete(
        self, phase: CreditPhase, sent_end_ns: int
    ) -> None:
        """Mock publish_phase_sending_complete method."""
        self.phase_sending_complete_calls.append(
            CreditPhaseSendingCompleteMessage(
                phase=phase,
                sent_end_ns=sent_end_ns,
                service_id="test-service",
            )
        )

    async def publish_phase_complete(
        self, phase: CreditPhase, completed: int, end_ns: int
    ) -> None:
        """Mock publish_phase_complete method."""
        self.phase_complete_calls.append(
            CreditPhaseCompleteMessage(
                phase=phase,
                completed=completed,
                end_ns=end_ns,
                service_id="test-service",
            )
        )


@pytest.fixture
def mock_credit_manager():
    """Fixture providing a mock credit manager."""
    return MockCreditManager()
