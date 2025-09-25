# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import time
from collections import deque
from typing import Any, TypeVar

import pytest

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.enums import CreditPhase
from aiperf.common.messages import (
    CreditDropMessage,
    CreditPhaseCompleteMessage,
    CreditPhaseProgressMessage,
    CreditPhaseSendingCompleteMessage,
    CreditPhaseStartMessage,
    CreditsCompleteMessage,
    Message,
)
from aiperf.common.mixins.aiperf_lifecycle_mixin import AIPerfLifecycleMixin
from aiperf.common.models.credit_models import CreditPhaseStats
from aiperf.timing import CreditIssuingStrategy
from aiperf.timing.config import TimingManagerConfig
from tests.utils.time_traveler import TimeTraveler

T = TypeVar("T", bound=CreditIssuingStrategy)


_logger = AIPerfLogger(__name__)


class MockSemaphore:
    """Mock implementation of Semaphore for testing, with auto-return delay.

    This is a simple mock implementation of a semaphore that can be used to test the behavior of the
    CreditIssuingStrategy. It works by keeping track of the number of credits that are available
    and the number of credits that are being acquired and released.

    Args:
        value: The initial value of the semaphore.
        interval: The interval at which the semaphore is released. (simulating credit returns)
    """

    def __init__(self, value: int, interval: float) -> None:
        self.interval = interval
        self.value = value
        self.wait_count = 0
        self.acquire_count = 0
        self.release_count = 0
        self.next_release = deque()

    async def acquire(self):
        if self.locked():
            assert len(self.next_release) > 0
            release_time = self.next_release.popleft()
            if release_time > time.perf_counter():
                await asyncio.sleep(release_time - time.perf_counter())
                self.wait_count += 1

        self.value -= 1
        self.acquire_count += 1
        if self.interval is not None:
            self.next_release.append(time.perf_counter() + self.interval)

    def release(self):
        self.value += 1
        self.release_count += 1

    def locked(self):
        return self.value <= 0


class MockCreditManager(AIPerfLifecycleMixin):
    """Mock implementation of CreditManagerProtocol for testing."""

    def __init__(self, time_traveler: TimeTraveler, **kwargs):
        super().__init__(**kwargs)
        self.dropped_timestamps = []
        self.dropped_credits = []
        self.progress_calls = []
        self.credits_complete_calls = []
        self.phase_start_calls = []
        self.phase_complete_calls = []
        self.phase_sending_complete_calls = []
        self.credit_strategy: CreditIssuingStrategy | None = None
        self.time_traveler = time_traveler
        self.publish_calls = []

    def create_strategy(
        self,
        config: TimingManagerConfig,
        strategy_type: type[T],
        auto_return_delay: float | None = None,
    ) -> Any:
        """Create a credit issuing strategy."""
        self.credit_strategy = strategy_type(config, self)  # type: ignore
        if config.concurrency is not None:
            self.credit_strategy._semaphore = MockSemaphore(  # type: ignore
                config.concurrency, auto_return_delay
            )
            return self.credit_strategy, self.credit_strategy._semaphore  # type: ignore
        return self.credit_strategy

    async def publish(self, message: Message) -> None:
        """Mock publish method."""
        self.publish_calls.append(message)

    async def drop_credit(
        self,
        credit_phase: CreditPhase,
        conversation_id: str | None = None,
        credit_drop_ns: int | None = None,
        should_cancel: bool = False,
        cancel_after_ns: int = 0,
    ) -> None:
        """Mock drop_credit method."""
        drop_time_ns = self.time_traveler.time_ns()
        self.dropped_timestamps.append(drop_time_ns)
        self.dropped_credits.append(
            CreditDropMessage(
                service_id="test-service",
                phase=credit_phase,
                conversation_id=conversation_id,
                credit_drop_ns=credit_drop_ns,
                should_cancel=should_cancel,
                cancel_after_ns=cancel_after_ns,
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
        self, phase: CreditPhase, sent_end_ns: int, sent: int
    ) -> None:
        """Mock publish_phase_sending_complete method."""
        self.phase_sending_complete_calls.append(
            CreditPhaseSendingCompleteMessage(
                phase=phase,
                sent_end_ns=sent_end_ns,
                service_id="test-service",
                sent=sent,
            )
        )

    async def publish_phase_complete(
        self,
        phase: CreditPhase,
        completed: int,
        end_ns: int,
        timeout_triggered: bool = False,
    ) -> None:
        """Mock publish_phase_complete method."""
        self.phase_complete_calls.append(
            CreditPhaseCompleteMessage(
                phase=phase,
                completed=completed,
                end_ns=end_ns,
                timeout_triggered=timeout_triggered,
                service_id="test-service",
            )
        )

    async def run_strategy(self, strategy: CreditIssuingStrategy):
        """Run the full credit issuing strategy."""
        self.credit_strategy = strategy
        await self.credit_strategy.start()
        await self.wait_for_tasks()


@pytest.fixture
def mock_credit_manager(time_traveler: TimeTraveler):
    """Fixture providing a mock credit manager."""
    return MockCreditManager(time_traveler=time_traveler)


def profiling_phase_stats_from_config(config: TimingManagerConfig) -> CreditPhaseStats:
    """Create a phase stats object from a config."""
    return CreditPhaseStats(
        type=CreditPhase.PROFILING,
        start_ns=time.time_ns(),
        total_expected_requests=config.request_count,
    )
