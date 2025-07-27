# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import sys

from aiperf.common.comms.base import (
    CommAddress,
    PullClientProtocol,
    PushClientProtocol,
    RequestClientProtocol,
)
from aiperf.common.config import ServiceConfig
from aiperf.common.config.user_config import UserConfig
from aiperf.common.enums import (
    CreditPhase,
    MessageType,
    ServiceType,
)
from aiperf.common.exceptions import InvalidStateError
from aiperf.common.factories import ServiceFactory
from aiperf.common.hooks import (
    on_configure,
    on_init,
    on_start,
    on_stop,
)
from aiperf.common.messages import (
    CommandMessage,
    CreditDropMessage,
    CreditReturnMessage,
    DatasetTimingRequest,
    DatasetTimingResponse,
)
from aiperf.services.base_component_service import BaseComponentService
from aiperf.services.timing_manager.config import (
    TimingManagerConfig,
    TimingMode,
)
from aiperf.services.timing_manager.credit_issuing_strategy import (
    CreditIssuingStrategy,
    CreditIssuingStrategyFactory,
)
from aiperf.services.timing_manager.credit_manager import CreditPhaseMessagesMixin


@ServiceFactory.register(ServiceType.TIMING_MANAGER)
class TimingManager(BaseComponentService, CreditPhaseMessagesMixin):
    """
    The TimingManager service is responsible to generate the schedule and issuing
    timing credits for requests.
    """

    def __init__(
        self,
        service_config: ServiceConfig,
        user_config: UserConfig | None,
        service_id: str | None = None,
    ) -> None:
        super().__init__(
            service_config=service_config,
            user_config=user_config,
            service_id=service_id,
        )
        self.debug("Initializing timing manager")

        self.dataset_request_client: RequestClientProtocol = (
            self.comms.create_request_client(
                CommAddress.DATASET_MANAGER_PROXY_FRONTEND,
            )
        )
        self.credit_drop_push_client: PushClientProtocol = (
            self.comms.create_push_client(
                CommAddress.CREDIT_DROP,
                bind=True,
            )
        )
        self.credit_return_pull_client: PullClientProtocol = (
            self.comms.create_pull_client(
                CommAddress.CREDIT_RETURN,
                bind=True,
            )
        )

        self._credit_issuing_strategy: CreditIssuingStrategy | None = None

    @property
    def service_type(self) -> ServiceType:
        """The type of service."""
        return ServiceType.TIMING_MANAGER

    @on_init
    async def _timing_manager_initialize(self) -> None:
        """Initialize timing manager-specific components."""
        self.debug("Initializing timing manager")
        self.config = TimingManagerConfig.from_user_config(self.user_config)
        await self.credit_return_pull_client.register_pull_callback(
            message_type=MessageType.CREDIT_RETURN,
            callback=self._on_credit_return,
        )

    @on_configure
    async def _timing_manager_configure(self, message: CommandMessage) -> None:
        """Configure the timing manager."""
        self.debug(lambda: f"Configuring timing manager with message: {message}")

        if self.config.timing_mode == TimingMode.FIXED_SCHEDULE:
            # This will block until the dataset is ready and the timing response is received
            dataset_timing_response: DatasetTimingResponse = (
                await self.dataset_request_client.request(
                    message=DatasetTimingRequest(
                        service_id=self.service_id,
                    ),
                )
            )
            self.debug(
                lambda: f"TM: Received dataset timing response: {dataset_timing_response}"
            )
            self.info("TM: Using fixed schedule strategy")
            self._credit_issuing_strategy = (
                CreditIssuingStrategyFactory.create_instance(
                    TimingMode.FIXED_SCHEDULE,
                    config=self.config,
                    credit_manager=self,
                    schedule=dataset_timing_response.timing_data,
                )
            )
        elif self.config.timing_mode == TimingMode.CONCURRENCY:
            self.info("TM: Using concurrency strategy")
            self._credit_issuing_strategy = (
                CreditIssuingStrategyFactory.create_instance(
                    TimingMode.CONCURRENCY,
                    config=self.config,
                    credit_manager=self,
                )
            )
        elif self.config.timing_mode == TimingMode.REQUEST_RATE:
            self.info("TM: Using request rate strategy")
            self._credit_issuing_strategy = (
                CreditIssuingStrategyFactory.create_instance(
                    TimingMode.REQUEST_RATE,
                    config=self.config,
                    credit_manager=self,
                )
            )

        if not self._credit_issuing_strategy:
            raise InvalidStateError("No credit issuing strategy configured")

    @on_start
    async def _timing_manager_start(self) -> None:
        """Start the timing manager and issue credit drops according to the configured strategy."""
        self.debug("Starting timing manager")

        if not self._credit_issuing_strategy:
            raise InvalidStateError("No credit issuing strategy configured")

        await asyncio.sleep(2)
        self.execute_async(self._credit_issuing_strategy.start())

    @on_stop
    async def _timing_manager_stop(self) -> None:
        """Stop the timing manager."""
        self.debug("Stopping timing manager")
        if self._credit_issuing_strategy:
            await self._credit_issuing_strategy.stop()
        await self.cancel_all_tasks()

    async def _on_credit_return(self, message: CreditReturnMessage) -> None:
        """Handle the credit return message."""
        self.debug(lambda: f"Timing manager received credit return message: {message}")
        if self._credit_issuing_strategy:
            await self._credit_issuing_strategy._on_credit_return(message)

    async def drop_credit(
        self,
        credit_phase: CreditPhase,
        conversation_id: str | None = None,
        credit_drop_ns: int | None = None,
    ) -> None:
        """Drop a credit."""
        self.execute_async(
            self.credit_drop_push_client.push(
                message=CreditDropMessage(
                    service_id=self.service_id,
                    phase=credit_phase,
                    credit_drop_ns=credit_drop_ns,
                    conversation_id=conversation_id,
                ),
            )
        )


def main() -> None:
    """Main entry point for the timing manager."""
    from aiperf.common.bootstrap import bootstrap_and_run_service

    bootstrap_and_run_service(TimingManager)


if __name__ == "__main__":
    sys.exit(main())
