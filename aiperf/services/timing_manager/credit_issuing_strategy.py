# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from abc import ABC, abstractmethod
from typing import Protocol

from aiperf.common.enums import CreditPhase
from aiperf.common.messages import CreditPhaseStats, CreditReturnMessage
from aiperf.common.mixins import AsyncTaskManagerMixin
from aiperf.services.timing_manager.config import TimingManagerConfig


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
        self, phase_stats_map: dict[CreditPhase, CreditPhaseStats]
    ) -> None: ...

    async def publish_credits_complete(self, cancelled: bool) -> None: ...
    async def publish_phase_start(self, phase_stats: CreditPhaseStats) -> None: ...
    async def publish_phase_sending_complete(
        self, phase_stats: CreditPhaseStats
    ) -> None: ...
    async def publish_phase_complete(self, phase_stats: CreditPhaseStats) -> None: ...


class CreditIssuingStrategy(AsyncTaskManagerMixin, ABC):
    """
    Base class for credit issuing strategies.
    """

    def __init__(
        self, config: TimingManagerConfig, credit_manager: CreditManagerProtocol
    ):
        super().__init__()
        self.logger = logging.getLogger(__class__.__name__)
        self.config = config
        self.credit_manager = credit_manager

        # The phases to run, in order
        self.phases: list[CreditPhaseStats] = []
        # The stats for each phase, keyed by phase type
        self.phase_stats: dict[CreditPhase, CreditPhaseStats] = {}

    @abstractmethod
    async def start(self) -> None:
        """Start the credit issuing strategy."""
        raise NotImplementedError("Start method must be implemented in subclass")

    async def stop(self) -> None:
        """Stop the credit issuing strategy."""
        await self.cancel_all_tasks()

    async def on_credit_return(self, message: CreditReturnMessage) -> None:
        """This is called by the credit manager when a credit is returned. It can be
        overridden in subclasses to handle the credit return."""
        return

    def all_phases_complete(self) -> bool:
        """Check if all phases are complete."""
        return all(phase.is_complete for phase in self.phases)
