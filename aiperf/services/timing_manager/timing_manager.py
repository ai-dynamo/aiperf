# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextlib
import sys
from dataclasses import dataclass

from aiperf.common.comms.client_enums import ClientType, PullClientType, PushClientType
from aiperf.common.config import ServiceConfig
from aiperf.common.enums import MessageType, ServiceState, ServiceType, Topic
from aiperf.common.factories import ServiceFactory
from aiperf.common.hooks import on_cleanup, on_configure, on_init, on_start, on_stop
from aiperf.common.messages import (
    CommandMessage,
    CreditReturnMessage,
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
        self._credit_drop_task: asyncio.Task | None = None
        self.dataset_timing_response: DatasetTimingResponse | None = None
        self._credit_issuing_strategy: CreditIssuingStrategy | None = None

    @property
    def service_type(self) -> ServiceType:
        """The type of service."""
        return ServiceType.TIMING_MANAGER

    @property
    def required_clients(self) -> list[ClientType]:
        """The communication clients required by the service."""
        return [
            *(super().required_clients or []),
            PullClientType.CREDIT_RETURN,
            PushClientType.CREDIT_DROP,
        ]

    @on_init
    async def _initialize(self) -> None:
        """Initialize timing manager-specific components."""
        self.logger.debug("Initializing timing manager")

    @on_configure
    async def _configure(self, message: CommandMessage) -> None:
        """Configure the timing manager."""
        self.logger.debug("Configuring timing manager with message: %s", message)

        # config = TimingManagerConfig(message.data)
        config = TimingManagerConfig()
        assert isinstance(config, TimingManagerConfig)

        if config.timing_mode == TimingMode.FIXED_SCHEDULE:
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
        """Start the timing manager."""
        self.logger.debug("Starting timing manager")

        # Setup credit return handling
        await self.comms.register_pull_callback(
            message_type=MessageType.CREDIT_RETURN,
            callback=self._on_credit_return,
        )
        await self.set_state(ServiceState.RUNNING)
        await asyncio.sleep(1.5)

        self.logger.debug("TM: Requesting dataset timing information")
        self.dataset_timing_response = await self.comms.request(
            topic=Topic.DATASET_TIMING,
            message=DatasetTimingRequest(
                service_id=self.service_id,
            ),
        )
        self.logger.debug(
            "TM: Received dataset timing response: %s",
            self.dataset_timing_response,
        )

        # Start the credit dropping task
        self._credit_drop_task = asyncio.create_task(self._issue_credit_drops())

    @on_stop
    async def _stop(self) -> None:
        """Stop the timing manager."""
        self.logger.debug("Stopping timing manager")
        if self._credit_drop_task and not self._credit_drop_task.done():
            self._credit_drop_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._credit_drop_task
            self._credit_drop_task = None

    @on_cleanup
    async def _cleanup(self) -> None:
        """Clean up timing manager-specific components."""
        self.logger.debug("Cleaning up timing manager")

    async def _issue_credit_drops(self) -> None:
        """Issue credit drops according to the schedule."""
        self.logger.debug("Starting credit drops")

        assert isinstance(self._credit_issuing_strategy, CreditIssuingStrategy)
        await self._credit_issuing_strategy.start()

    async def _issue_credit_drop(self, credit_drop_info: CreditDropInfo) -> None:
        # await self.comms.push(
        #     topic=Topic.CREDIT_DROP,
        #     message=CreditDropMessage(
        #         service_id=self.service_id,
        #         amount=1,
        #         conversation_id=conversation_id,
        #         credit_drop_ns=time.time_ns(),
        #     ),
        # )
        pass

    async def _on_credit_return(self, message: CreditReturnMessage) -> None:
        """Process a credit return message.

        Args:
            message: The credit return message received from the pull request
        """
        self.logger.debug("Processing credit return: %s", message)
        async with self._credit_lock:
            self._credits_available += message.amount


def main() -> None:
    """Main entry point for the timing manager."""
    from aiperf.common.bootstrap import bootstrap_and_run_service

    bootstrap_and_run_service(TimingManager)


if __name__ == "__main__":
    sys.exit(main())
