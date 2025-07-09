# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextlib
import sys
from dataclasses import dataclass

from aiperf.common.comms.base import (
    CommunicationClientAddressType,
    PullClientProtocol,
    PushClientProtocol,
    RequestClientProtocol,
)
from aiperf.common.config import ServiceConfig
from aiperf.common.constants import TASK_CANCEL_TIMEOUT_SHORT
from aiperf.common.credit_models import CreditDropMessage, CreditReturnMessage
from aiperf.common.enums import (
    MessageType,
    ServiceType,
)
from aiperf.common.factories import ServiceFactory
from aiperf.common.hooks import on_cleanup, on_configure, on_init, on_start, on_stop
from aiperf.common.messages import (
    CommandMessage,
    DatasetTimingRequest,
    DatasetTimingResponse,
)
from aiperf.common.service.base_component_service import BaseComponentService
from aiperf.services.timing_manager.concurrency_strategy import ConcurrencyStrategy
from aiperf.services.timing_manager.config import TimingManagerConfig, TimingMode
from aiperf.services.timing_manager.credit_issuing_strategy import CreditIssuingStrategy
from aiperf.services.timing_manager.fixed_schedule_strategy import FixedScheduleStrategy
from aiperf.services.timing_manager.rate_strategy import RateStrategy


@dataclass
class CreditDropInfo:
    amount: int = 1
    conversation_id: str | None = None
    credit_drop_ns: int | None = None


@ServiceFactory.register(ServiceType.TIMING_MANAGER)
class TimingManager(BaseComponentService):
    """
    The TimingManager service is responsible to generate the schedule and issuing
    timing credits for requests.
    """

    def __init__(
        self, service_config: ServiceConfig, service_id: str | None = None
    ) -> None:
        super().__init__(service_config=service_config, service_id=service_id)
        self.logger.debug("Initializing timing manager")

        self._credit_issuing_strategy: CreditIssuingStrategy | None = None

        self.tasks: set[asyncio.Task] = set()

        self.dataset_request_client: RequestClientProtocol = (
            self.comms.create_request_client(
                CommunicationClientAddressType.DATASET_MANAGER_PROXY_FRONTEND,
            )
        )
        self.credit_drop_client: PushClientProtocol = self.comms.create_push_client(
            CommunicationClientAddressType.CREDIT_DROP,
            bind=True,
        )
        self.credit_return_client: PullClientProtocol = self.comms.create_pull_client(
            CommunicationClientAddressType.CREDIT_RETURN,
            bind=True,
        )

    @property
    def service_type(self) -> ServiceType:
        """The type of service."""
        return ServiceType.TIMING_MANAGER

    @on_init
    async def _initialize(self) -> None:
        """Initialize timing manager-specific components."""
        self.logger.debug("Initializing timing manager")
        await self.credit_return_client.register_pull_callback(
            message_type=MessageType.CREDIT_RETURN,
            callback=self._on_credit_return,
        )

    @on_configure
    async def _configure(self, message: CommandMessage) -> None:
        """Configure the timing manager."""
        self.logger.debug("Configuring timing manager with message: %s", message)

        # config = TimingManagerConfig(message.data)
        config = TimingManagerConfig()
        assert isinstance(config, TimingManagerConfig)

        if config.timing_mode == TimingMode.FIXED_SCHEDULE:
            # This will block until the dataset is ready and the timing response is received
            dataset_timing_response: DatasetTimingResponse = (
                await self.dataset_request_client.request(
                    message=DatasetTimingRequest(
                        service_id=self.service_id,
                    ),
                )
            )
            self.logger.debug(
                "TM: Received dataset timing response: %s",
                dataset_timing_response,
            )
            # TODO: Pass dataset_timing_response to strategy
            self._credit_issuing_strategy = FixedScheduleStrategy(
                config, self._issue_credit_drop
            )
        elif config.timing_mode == TimingMode.CONCURRENCY:
            self._credit_issuing_strategy = ConcurrencyStrategy(
                config, self._issue_credit_drop
            )
        elif config.timing_mode == TimingMode.RATE:
            self._credit_issuing_strategy = RateStrategy(
                config, self._issue_credit_drop
            )

        assert isinstance(self._credit_issuing_strategy, CreditIssuingStrategy)
        await self._credit_issuing_strategy.initialize()

    @on_start
    async def _start(self) -> None:
        """Start the timing manager and issue credit drops according to the configured strategy."""
        self.logger.debug("Starting timing manager")
        # TODO: If not configured raise an exception

        if not self._credit_issuing_strategy:
            raise RuntimeError("No credit issuing strategy configured")

        task = asyncio.create_task(self._credit_issuing_strategy.start())
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    @on_stop
    async def _stop(self) -> None:
        """Stop the timing manager."""
        self.logger.debug("Stopping timing manager")
        for task in list(self.tasks):
            task.cancel()

        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(
                asyncio.gather(*self.tasks),
                timeout=TASK_CANCEL_TIMEOUT_SHORT,
            )
        self.tasks.clear()

    @on_cleanup
    async def _cleanup(self) -> None:
        """Clean up timing manager-specific components."""
        self.logger.debug("Cleaning up timing manager")

    async def _issue_credit_drop(self, credit_drop_info: CreditDropInfo) -> None:
        """Issue a credit drop."""
        task = asyncio.create_task(
            self.credit_drop_client.push(
                message=CreditDropMessage(
                    service_id=self.service_id,
                    amount=credit_drop_info.amount,
                    credit_drop_ns=credit_drop_info.credit_drop_ns,
                    conversation_id=credit_drop_info.conversation_id,
                ),
            )
        )
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    async def _on_credit_return(self, message: CreditReturnMessage) -> None:
        """Process a credit return message."""
        self.logger.debug("Processing credit return: %s", message)
        if self._credit_issuing_strategy:
            task = asyncio.create_task(
                self._credit_issuing_strategy.on_credit_return(message)
            )
            self.tasks.add(task)
            task.add_done_callback(self.tasks.discard)


def main() -> None:
    """Main entry point for the timing manager."""
    from aiperf.common.bootstrap import bootstrap_and_run_service

    bootstrap_and_run_service(TimingManager)


if __name__ == "__main__":
    sys.exit(main())
