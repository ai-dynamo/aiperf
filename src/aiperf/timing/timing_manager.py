# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from aiperf.common.base_component_service import BaseComponentService
from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import (
    CommAddress,
    CommandType,
    CreditPhase,
    MessageType,
    ServiceType,
)
from aiperf.common.environment import Environment
from aiperf.common.exceptions import InvalidStateError
from aiperf.common.factories import ServiceFactory
from aiperf.common.hooks import (
    on_command,
    on_pull_message,
    on_stop,
)
from aiperf.common.messages import (
    CommandAcknowledgedResponse,
    CommandMessage,
    CreditDropMessage,
    CreditReturnMessage,
    DatasetTimingRequest,
    DatasetTimingResponse,
    ProfileCancelCommand,
    ProfileConfigureCommand,
)
from aiperf.common.mixins import PullClientMixin
from aiperf.common.protocols import (
    PushClientProtocol,
    RequestClientProtocol,
    ServiceProtocol,
)
from aiperf.timing.config import (
    TimingManagerConfig,
    TimingMode,
)
from aiperf.timing.credit_issuing_strategy import (
    CreditIssuingStrategy,
    CreditIssuingStrategyFactory,
)
from aiperf.timing.credit_manager import CreditPhaseMessagesMixin


@implements_protocol(ServiceProtocol)
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
        self.debug("Timing manager __init__")
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

    @on_command(CommandType.PROFILE_CONFIGURE)
    async def _profile_configure_command(
        self, message: ProfileConfigureCommand
    ) -> None:
        """Configure the timing manager."""
        self.debug(f"Configuring credit issuing strategy for {self.service_id}")

        if self.config.timing_mode == TimingMode.FIXED_SCHEDULE:
            # This will block until the dataset is ready and the timing response is received
            dataset_timing_response: DatasetTimingResponse = await self.dataset_request_client.request(
                message=DatasetTimingRequest(
                    service_id=self.service_id,
                ),
                # NOTE: We use the dataset configuration timeout here because the dataset manager
                # may take longer than a normal request timeout to respond to this request. This is
                # because it blocks until the dataset is configured.
                timeout=Environment.DATASET.CONFIGURATION_TIMEOUT,
            )
            self.debug(
                lambda: f"Received dataset timing response: {dataset_timing_response}"
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
        else:
            self.info(f"Using {self.config.timing_mode.title()} strategy")
            self._credit_issuing_strategy = (
                CreditIssuingStrategyFactory.create_instance(
                    self.config.timing_mode,
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
        self.info(
            f"Credit issuing strategy for {self.config.timing_mode.title()} started"
        )

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
        if self.is_debug_enabled:
            self.debug(f"Timing manager received credit return message: {message}")
        if self._credit_issuing_strategy:
            await self._credit_issuing_strategy._on_credit_return(message)

    async def drop_credit(
        self,
        credit_phase: CreditPhase,
        credit_num: int,
        *,
        conversation_id: str | None = None,
        credit_drop_ns: int | None = None,
        should_cancel: bool = False,
        cancel_after_ns: int = 0,
    ) -> None:
        """Drop a credit."""
        self.execute_async(
            self.credit_drop_push_client.push(
                message=CreditDropMessage(
                    service_id=self.service_id,
                    phase=credit_phase,
                    credit_num=credit_num,
                    credit_drop_ns=credit_drop_ns,
                    conversation_id=conversation_id,
                    should_cancel=should_cancel,
                    cancel_after_ns=cancel_after_ns,
                ),
            )
        )


def main() -> None:
    """Main entry point for the timing manager."""
    from aiperf.common.bootstrap import bootstrap_and_run_service

    bootstrap_and_run_service(TimingManager)


if __name__ == "__main__":
    main()
