# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import time

from aiperf.common.comms.base import (
    CommunicationClientAddressType,
    PullClientProtocol,
    PushClientProtocol,
    RequestClientProtocol,
)
from aiperf.common.config import ServiceConfig
from aiperf.common.config.user_config import UserConfig
from aiperf.common.credit_models import (
    CreditDropMessage,
    CreditPhaseCompleteMessage,
    CreditPhaseProgressMessage,
    CreditPhaseSendingCompleteMessage,
    CreditPhaseStartMessage,
    CreditPhaseStats,
    CreditReturnMessage,
    CreditsCompleteMessage,
)
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
    DatasetTimingRequest,
    DatasetTimingResponse,
)
from aiperf.common.mixins import AsyncTaskManagerMixin
from aiperf.common.service.base_component_service import BaseComponentService
from aiperf.services.timing_manager.concurrency_strategy import ConcurrencyStrategy
from aiperf.services.timing_manager.config import (
    TimingManagerConfig,
    TimingMode,
)
from aiperf.services.timing_manager.credit_issuing_strategy import CreditIssuingStrategy
from aiperf.services.timing_manager.fixed_schedule_strategy import FixedScheduleStrategy
from aiperf.services.timing_manager.request_rate_strategy import RequestRateStrategy


@ServiceFactory.register(ServiceType.TIMING_MANAGER)
class TimingManager(BaseComponentService, AsyncTaskManagerMixin):
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
        self.logger.debug("Initializing timing manager")

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

        self.start_time_ns = time.time_ns()
        self.config = TimingManagerConfig.from_user_config(self.user_config)
        self._credit_issuing_strategy: CreditIssuingStrategy | None = None

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

        if self.config.timing_mode == TimingMode.FIXED_SCHEDULE:
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
            self.logger.info("TM: Using fixed schedule strategy")
            self._credit_issuing_strategy = FixedScheduleStrategy(
                config=self.config,
                credit_manager=self,
                schedule=dataset_timing_response.timing_data,
            )
        elif self.config.timing_mode == TimingMode.CONCURRENCY:
            self.logger.info("TM: Using concurrency strategy")
            self._credit_issuing_strategy = ConcurrencyStrategy(
                config=self.config,
                credit_manager=self,
            )
        elif self.config.timing_mode == TimingMode.REQUEST_RATE:
            self.logger.info("TM: Using request rate strategy")
            self._credit_issuing_strategy = RequestRateStrategy(
                config=self.config,
                credit_manager=self,
            )

        if not self._credit_issuing_strategy:
            raise InvalidStateError("No credit issuing strategy configured")

    @on_start
    async def _start(self) -> None:
        """Start the timing manager and issue credit drops according to the configured strategy."""
        self.logger.debug("Starting timing manager")

        if not self._credit_issuing_strategy:
            raise InvalidStateError("No credit issuing strategy configured")

        # await asyncio.sleep(1)
        self.execute_async(self._credit_issuing_strategy.start())

    @on_stop
    async def _stop(self) -> None:
        """Stop the timing manager."""
        self.logger.debug("Stopping timing manager")
        if self._credit_issuing_strategy:
            await self._credit_issuing_strategy.stop()
        await self.cancel_all_tasks()

    async def _on_credit_return(self, message: CreditReturnMessage) -> None:
        """Handle the credit return message."""
        self.logger.debug("Timing manager received credit return message: %s", message)
        if self._credit_issuing_strategy:
            await self._credit_issuing_strategy.on_credit_return(message)

    async def publish_phase_start(self, phase_stats: CreditPhaseStats) -> None:
        """Publish the phase start message."""
        self.execute_async(
            self.pub_client.publish(
                CreditPhaseStartMessage(
                    service_id=self.service_id,
                    phase_stats=phase_stats,
                )
            )
        )

    async def publish_phase_complete(self, phase_stats: CreditPhaseStats) -> None:
        """Publish the phase complete message."""
        self.execute_async(
            self.pub_client.publish(
                CreditPhaseCompleteMessage(
                    service_id=self.service_id,
                    phase_stats=phase_stats,
                )
            )
        )

    async def publish_phase_sending_complete(
        self, phase_stats: CreditPhaseStats
    ) -> None:
        """Publish the phase sending complete message."""
        self.execute_async(
            self.pub_client.publish(
                CreditPhaseSendingCompleteMessage(
                    service_id=self.service_id,
                    phase_stats=phase_stats,
                )
            )
        )

    async def publish_progress(
        self, phase_stats: dict[CreditPhase, CreditPhaseStats]
    ) -> None:
        """Publish the progress message."""
        self.execute_async(
            self.pub_client.publish(
                CreditPhaseProgressMessage(
                    service_id=self.service_id,
                    phase_stats=phase_stats,
                    request_ns=time.time_ns(),
                )
            )
        )

    async def publish_credits_complete(self, cancelled: bool) -> None:
        """Publish the credits complete message."""
        self.execute_async(
            self.pub_client.publish(
                CreditsCompleteMessage(
                    service_id=self.service_id,
                    cancelled=cancelled,
                )
            )
        )

    async def drop_credit(
        self,
        credit_phase: CreditPhase,
        conversation_id: str | None = None,
        credit_drop_ns: int | None = None,
    ) -> None:
        """Drop a credit."""
        self.execute_async(
            self.credit_drop_client.push(
                message=CreditDropMessage(
                    service_id=self.service_id,
                    credit_phase=credit_phase,
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
