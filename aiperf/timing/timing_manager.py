# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.enums import (
    CommAddress,
    CommandType,
    CreditPhase,
    MessageType,
    ServiceType,
)
from aiperf.common.exceptions import InvalidStateError
from aiperf.common.factories import ServiceFactory
from aiperf.common.hooks import (
    on_command,
    on_init,
    on_pull_message,
    on_stop,
)
from aiperf.common.messages import (
    CommandMessage,
    CreditDropMessage,
    CreditReturnMessage,
    DatasetTimingRequest,
    DatasetTimingResponse,
)
from aiperf.common.messages.command_messages import (
    CommandAcknowledgedResponse,
    ProfileCancelCommand,
)
from aiperf.common.mixins import PullClientMixin
from aiperf.common.protocols import (
    PushClientProtocol,
    RequestClientProtocol,
)
from aiperf.services.base_component_service import BaseComponentService
from aiperf.timing.config import (
    TimingManagerConfig,
    TimingMode,
)
from aiperf.timing.credit_issuing_strategy import (
    CreditIssuingStrategy,
    CreditIssuingStrategyFactory,
)
from aiperf.timing.credit_manager import CreditPhaseMessagesMixin


@ServiceFactory.register(ServiceType.TIMING_MANAGER)
class TimingManager(PullClientMixin, BaseComponentService, CreditPhaseMessagesMixin):
    """
    The TimingManager service is responsible to generate the schedule and issuing
    timing credits for requests.
    """

    def __init__(
        self,
        service_config: ServiceConfig,
        user_config: UserConfig,
        service_id: str | None = None,
    ) -> None:
        super().__init__(
            service_config=service_config,
            user_config=user_config,
            service_id=service_id,
            pull_client_address=CommAddress.CREDIT_RETURN,
            pull_client_bind=True,
        )
        self.config = TimingManagerConfig.from_user_config(self.user_config)

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

        self._credit_issuing_strategy: CreditIssuingStrategy | None = None

    @on_init
    async def _timing_manager_initialize(self) -> None:
        """Initialize timing manager-specific components."""
        self.debug("Initializing timing manager")
        await self._configure()

    async def _configure(self) -> None:
        """Configure the timing manager."""
        self.debug("Configuring timing manager")

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
            self.info("Using fixed schedule strategy")
            self._credit_issuing_strategy = (
                CreditIssuingStrategyFactory.create_instance(
                    TimingMode.FIXED_SCHEDULE,
                    config=self.config,
                    credit_manager=self,
                    schedule=dataset_timing_response.timing_data,
                )
            )
        elif self.config.timing_mode == TimingMode.CONCURRENCY:
            self.info("Using concurrency strategy")
            self._credit_issuing_strategy = (
                CreditIssuingStrategyFactory.create_instance(
                    TimingMode.CONCURRENCY,
                    config=self.config,
                    credit_manager=self,
                )
            )
        elif self.config.timing_mode == TimingMode.REQUEST_RATE:
            self.info("Using request rate strategy")
            self._credit_issuing_strategy = (
                CreditIssuingStrategyFactory.create_instance(
                    TimingMode.REQUEST_RATE,
                    config=self.config,
                    credit_manager=self,
                )
            )

        if not self._credit_issuing_strategy:
            raise InvalidStateError("No credit issuing strategy configured")
        self.debug(
            lambda: f"Timing manager configured with credit issuing strategy: {self._credit_issuing_strategy}"
        )

    @on_command(CommandType.PROFILE_START)
    async def _on_start_profiling(self, message: CommandMessage) -> None:
        """Start the timing manager and issue credit drops according to the configured strategy."""
        self.debug("Starting profiling")

        self.debug("Waiting for timing manager to be initialized")
        await self.initialized_event.wait()
        self.debug("Timing manager initialized, starting profiling")

        if not self._credit_issuing_strategy:
            raise InvalidStateError("No credit issuing strategy configured")

        self.execute_async(self._credit_issuing_strategy.start())
        self.info("Profiling started")

    @on_command(CommandType.PROFILE_CANCEL)
    async def _handle_profile_cancel_command(
        self, message: ProfileCancelCommand
    ) -> None:
        self.debug(lambda: f"Received profile cancel command: {message}")
        await self.publish(
            CommandAcknowledgedResponse.from_command_message(message, self.service_id)
        )
        if self._credit_issuing_strategy:
            await self._credit_issuing_strategy.stop()

    @on_stop
    async def _timing_manager_stop(self) -> None:
        """Stop the timing manager."""
        self.debug("Stopping timing manager")
        if self._credit_issuing_strategy:
            await self._credit_issuing_strategy.stop()
        await self.cancel_all_tasks()

    @on_pull_message(MessageType.CREDIT_RETURN)
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
    main()
