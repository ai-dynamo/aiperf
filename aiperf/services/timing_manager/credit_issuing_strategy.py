# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from abc import ABC, abstractmethod
from typing import Protocol

from aiperf.common.messages import CreditReturnMessage
from aiperf.common.mixins import AsyncTaskManagerMixin
from aiperf.services.timing_manager.config import TimingManagerConfig


class CreditManagerProtocol(Protocol):
    """Defines the interface for a CreditManager.

    This is used to allow the credit issuing strategy to interact with the TimingManager
    in a decoupled way.
    """

    async def drop_credit(
        self,
        conversation_id: str | None = None,
        credit_drop_ns: int | None = None,
    ) -> None:
        """Drop a credit."""
        ...

    async def publish_progress(
        self, start_time_ns: int, total: int, completed: int
    ) -> None:
        """Publish the progress message."""
        ...

    async def publish_credits_complete(self, cancelled: bool) -> None:
        """Publish the credits complete message."""
        ...


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

    @abstractmethod
    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        await self.cancel_all_tasks()

    async def on_credit_return(self, message: CreditReturnMessage) -> None:
        """This is called by the credit manager when a credit is returned. It can be
        overridden in subclasses to handle the credit return."""
        return
