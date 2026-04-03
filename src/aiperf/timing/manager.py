# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import asyncio

from aiperf.common.base_component_service import BaseComponentService
from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.enums import CommandType, MessageType
from aiperf.common.environment import Environment
from aiperf.common.event_loop_monitor import EventLoopMonitor
from aiperf.common.exceptions import InvalidStateError
from aiperf.common.hooks import (
    on_command,
    on_message,
    on_stop,
)
from aiperf.common.messages import (
    CommandMessage,
    DatasetConfiguredNotification,
    ProfileCancelCommand,
    ProfileConfigureCommand,
)
from aiperf.common.models import DatasetMetadata
from aiperf.credit.direct_credit_router import DirectCreditRouter
from aiperf.credit.sticky_router import CreditRouterProtocol, StickyCreditRouter
from aiperf.timing.config import TimingConfig
from aiperf.timing.phase.publisher import PhasePublisher
from aiperf.timing.phase_orchestrator import PhaseOrchestrator
from aiperf.workers.worker import Worker


class TimingManager(BaseComponentService):
    """Service orchestrating credit issuance and request timing.

    Central Service for the credit system. Creates a PhaseOrchestrator
    which internally instantiates the appropriate TimingMode based on mode
    (REQUEST_RATE, FIXED_SCHEDULE, or USER_CENTRIC_RATE).

    Handles commands: PROFILE_CONFIGURE (create orchestrator),
                      PROFILE_START (begin credit issuance),
                      PROFILE_CANCEL (cancel gracefully).
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
        )
        self.debug("Timing manager __init__")
        self.config = TimingConfig.from_user_config(self.user_config)

        self.phase_publisher = PhasePublisher(
            pub_client=self.pub_client,
            service_id=self.service_id,
        )

        self._dataset_configured_event = asyncio.Event()
        self._dataset_metadata: DatasetMetadata | None = None

        if self._is_direct_worker_mode(service_config, user_config):
            # Direct: deliver credits directly to a co-located Worker
            direct_router = DirectCreditRouter()
            self._direct_worker = self._create_direct_worker(
                service_config, user_config
            )
            direct_router.attach_worker(self._direct_worker)
            self.attach_child_lifecycle(self._direct_worker)
            self.credit_router: CreditRouterProtocol = direct_router
        else:
            # Standard: ZMQ-based routing to separate worker processes
            self._direct_worker = None
            sticky_router = StickyCreditRouter(
                service_config=service_config,
                service_id=self.service_id,
            )
            self.attach_child_lifecycle(sticky_router)
            self.credit_router = sticky_router

        self.event_loop_monitor = EventLoopMonitor(self.service_id)

        self._phase_orchestrator: PhaseOrchestrator | None = None

    @on_message(MessageType.DATASET_CONFIGURED_NOTIFICATION)
    async def _on_dataset_configured_notification(
        self, message: DatasetConfiguredNotification
    ) -> None:
        """Store dataset metadata and signal configuration ready."""
        self.debug(
            lambda: f"Received dataset configured notification: "
            f"{len(message.metadata.conversations)} conversations, "
            f"{message.metadata.sampling_strategy.value} sampling strategy"
        )

        self._dataset_metadata = message.metadata
        self._dataset_configured_event.set()

    @on_command(CommandType.PROFILE_CONFIGURE)
    async def _profile_configure_command(
        self, message: ProfileConfigureCommand
    ) -> None:
        """Create and configure phase orchestrator."""
        self.info("Waiting for dataset to be configured before configuring timing")
        await asyncio.wait_for(
            self._dataset_configured_event.wait(),
            timeout=Environment.DATASET.CONFIGURATION_TIMEOUT,
        )

        if not self._dataset_metadata:
            raise InvalidStateError("Dataset metadata is not available")

        self.debug(f"Configuring phase orchestrator for {self.service_id}")

        # Create orchestrator that executes phases
        self._phase_orchestrator = PhaseOrchestrator(
            config=self.config,
            phase_publisher=self.phase_publisher,
            credit_router=self.credit_router,
            dataset_metadata=self._dataset_metadata,
        )
        await self._phase_orchestrator.initialize()

    @on_command(CommandType.PROFILE_START)
    async def _on_start_profiling(self, _message: CommandMessage) -> None:
        """Start credit issuance. Disables GC for stable timing."""
        if not self._phase_orchestrator:
            raise InvalidStateError("No phase orchestrator configured")

        # Start event loop health monitoring only during the benchmark
        self.event_loop_monitor.start()

        self.debug("Starting profiling")
        self.execute_async(self._phase_orchestrator.start())

    @on_command(CommandType.PROFILE_CANCEL)
    async def _handle_profile_cancel_command(
        self, message: ProfileCancelCommand
    ) -> None:
        """Cancel credit issuance gracefully.

        Stops new credits and cancels in-flight requests.
        """
        self.warning(f"Received profile cancel command: {message}")
        if self._phase_orchestrator:
            await self._phase_orchestrator.cancel()
            self.info("Phase orchestrator cancelled")

    @on_stop
    async def _timing_manager_stop(self) -> None:
        """Stop timing manager and re-enable GC."""
        self.debug("Stopping timing manager")

        if self._phase_orchestrator:
            await self._phase_orchestrator.stop()

        self.event_loop_monitor.stop()

    @staticmethod
    def _is_direct_worker_mode(
        service_config: ServiceConfig, user_config: UserConfig
    ) -> bool:
        """Determine whether to use direct worker mode.

        Resolution order:
        1. CLI flag (--workers-direct / --no-workers-direct) — explicit override
        2. Endpoint metadata (direct_worker: true in plugins.yaml) — default hint
        """
        cli_override = service_config.workers.direct
        if cli_override is not None:
            return cli_override
        try:
            return user_config._endpoint_metadata().direct_worker
        except Exception:
            return False

    def _create_direct_worker(
        self, service_config: ServiceConfig, user_config: UserConfig
    ) -> Worker:
        """Create a Worker co-located in TimingManager's process."""
        return Worker(
            service_config=service_config,
            user_config=user_config,
            service_id=f"{self.service_id}-worker",
        )


def main() -> None:
    """Main entry point for the timing manager."""
    from aiperf.common.bootstrap import bootstrap_and_run_service
    from aiperf.plugin.enums import ServiceType

    bootstrap_and_run_service(ServiceType.TIMING_MANAGER)


if __name__ == "__main__":
    main()
