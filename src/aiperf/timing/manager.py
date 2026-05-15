# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from aiperf.common.base_component_service import BaseComponentService
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
    DatasetConfigurationFailedNotification,
    DatasetConfiguredNotification,
    ProfileCancelCommand,
    ProfileConfigureCommand,
)
from aiperf.common.models import DatasetMetadata
from aiperf.credit.sticky_router import StickyCreditRouter
from aiperf.timing.config import TimingConfig
from aiperf.timing.phase.publisher import PhasePublisher
from aiperf.timing.phase_orchestrator import PhaseOrchestrator

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun


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
        run: BenchmarkRun,
        service_id: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            run=run,
            service_id=service_id,
            **kwargs,
        )
        self.debug("Timing manager __init__")
        self.config = TimingConfig.from_run(self.run)

        self.phase_publisher = PhasePublisher(
            pub_client=self.pub_client,
            service_id=self.service_id,
        )

        self._dataset_configured_event = asyncio.Event()
        self._dataset_failed_event = asyncio.Event()
        self._dataset_failure_reason: str | None = None
        self._dataset_metadata: DatasetMetadata | None = None

        # StickyCreditRouter handles everything: routing, sending, returns,
        # worker lifecycle. Created early to handle worker connections
        # immediately, as well as attaching to the lifecycle.
        self.sticky_router: StickyCreditRouter = StickyCreditRouter(
            run=run,
            service_id=self.service_id,
        )
        self.attach_child_lifecycle(self.sticky_router)
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

    @on_message(MessageType.DATASET_CONFIGURATION_FAILED)
    async def _on_dataset_configuration_failed(
        self, message: DatasetConfigurationFailedNotification
    ) -> None:
        """Abort the dataset-config wait when DatasetManager reports a failure.

        Without this, _profile_configure_command would block on
        _dataset_configured_event for the full DATASET.CONFIGURATION_TIMEOUT
        (300s default) even though the SystemController has already seen the
        CommandErrorResponse from DatasetManager and is trying to abort.
        """
        self.error(
            f"Received dataset configuration failed notification from "
            f"{message.service_id}: {message.error}"
        )
        self._dataset_failure_reason = message.error
        self._dataset_failed_event.set()

    @on_command(CommandType.PROFILE_CONFIGURE)
    async def _profile_configure_command(
        self, message: ProfileConfigureCommand
    ) -> None:
        """Create and configure phase orchestrator."""
        self.info("Waiting for dataset to be configured before configuring timing")
        await self._wait_for_dataset_or_failure()

        if self._dataset_failed_event.is_set():
            raise InvalidStateError(
                f"Dataset configuration failed: {self._dataset_failure_reason}"
            )

        if not self._dataset_metadata:
            raise InvalidStateError("Dataset metadata is not available")

        self.debug(f"Configuring phase orchestrator for {self.service_id}")

        # Create orchestrator that executes phases
        self._phase_orchestrator = PhaseOrchestrator(
            config=self.config,
            phase_publisher=self.phase_publisher,
            credit_router=self.sticky_router,
            dataset_metadata=self._dataset_metadata,
        )
        await self._phase_orchestrator.initialize()

    async def _wait_for_dataset_or_failure(self) -> None:
        """Wait for either the dataset-configured or dataset-failed event.

        Returns as soon as either event fires. Raises asyncio.TimeoutError
        on the existing 300s envelope (preserving prior behavior for the
        case where neither event ever arrives).
        """
        configured_task = asyncio.create_task(self._dataset_configured_event.wait())
        failed_task = asyncio.create_task(self._dataset_failed_event.wait())
        try:
            done, _ = await asyncio.wait(
                {configured_task, failed_task},
                timeout=Environment.DATASET.CONFIGURATION_TIMEOUT,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                raise asyncio.TimeoutError(
                    f"timed out waiting for dataset configuration after "
                    f"{Environment.DATASET.CONFIGURATION_TIMEOUT}s; check "
                    f"dataset-manager logs and consider raising "
                    f"AIPERF_DATASET_CONFIGURATION_TIMEOUT"
                )
        finally:
            for task in (configured_task, failed_task):
                if not task.done():
                    task.cancel()

    @on_command(CommandType.PROFILE_START)
    async def _on_start_profiling(self, _message: CommandMessage) -> None:
        """Start credit issuance.

        GC is already disabled for this process by the bootstrap path
        (``service_metadata.disable_gc=True`` for TimingManager); see
        ``aiperf.bootstrap``.
        """
        if not self._phase_orchestrator:
            raise InvalidStateError("No phase orchestrator configured")

        # Start event loop health monitoring only during the benchmark
        self.event_loop_monitor.start()

        self.debug("Starting profiling")
        task = self.execute_async(self._phase_orchestrator.start())
        task.add_done_callback(self._on_phase_orchestrator_done)

    def _on_phase_orchestrator_done(self, task: asyncio.Task) -> None:
        """Surface phase-orchestrator failures to the SystemController.

        ``execute_async`` is fire-and-forget, so a phase setup error (e.g.
        FixedScheduleStrategy rejecting an orphaned conversation with no
        first-turn timestamp) is otherwise stored on the task and never
        observed by the parent service. Without this hook, the run finishes
        with zero records but a clean ``os._exit(0)``, masking real bugs.
        Publish a ``BaseServiceErrorMessage`` so the SystemController can
        record it in its exit-error list and exit non-zero.

        Note: the orchestrator's ``_fail`` path raises ``CancelledError``
        after recording the original exception in the orchestrator's
        ``_exit_errors``. We therefore consult the orchestrator state
        rather than ``task.exception()`` (which is ``None`` for cancelled
        tasks) to decide whether the run actually failed.
        """
        from aiperf.common.enums import LifecycleState

        orchestrator = self._phase_orchestrator
        # task.exception() raises if the task was cancelled — guard with
        # cancelled() first. A bare CancelledError that wasn't preceded by
        # a real failure (e.g. user Ctrl+C) leaves the orchestrator in
        # STOPPED, not FAILED, and we shouldn't escalate that.
        if not task.cancelled():
            exc = task.exception()
            if exc is not None and not isinstance(exc, asyncio.CancelledError):
                self._publish_phase_failure(exc)
                return

        if orchestrator is not None and orchestrator.state == LifecycleState.FAILED:
            inner = orchestrator._exit_errors[0] if orchestrator._exit_errors else None
            err_details = inner.error_details if inner is not None else None
            self._publish_phase_failure_from_details(err_details)

    def _publish_phase_failure(self, exc: BaseException) -> None:
        from aiperf.common.messages import BaseServiceErrorMessage
        from aiperf.common.models.error_models import ErrorDetails

        self.error(f"Phase orchestrator failed: {exc!r}")
        self._publish_service_error_safely(
            BaseServiceErrorMessage(
                service_id=self.service_id,
                error=ErrorDetails.from_exception(exc),
            )
        )

    def _publish_phase_failure_from_details(self, details) -> None:
        from aiperf.common.messages import BaseServiceErrorMessage
        from aiperf.common.models.error_models import ErrorDetails

        self.error(f"Phase orchestrator entered FAILED state: {details}")
        self._publish_service_error_safely(
            BaseServiceErrorMessage(
                service_id=self.service_id,
                error=details
                or ErrorDetails(message="Phase orchestrator entered FAILED state"),
            )
        )

    def _publish_service_error_safely(self, message) -> None:
        try:
            self.execute_async(self.publish(message))
        except Exception as publish_error:
            self.debug(
                lambda e=publish_error: (
                    f"Failed to publish BaseServiceErrorMessage from phase failure "
                    f"(comms may already be down): {e!r}"
                )
            )

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
        """Stop the timing manager."""
        self.debug("Stopping timing manager")

        if self._phase_orchestrator:
            await self._phase_orchestrator.stop()

        self.event_loop_monitor.stop()


def main() -> None:
    """Main entry point for the timing manager."""
    from aiperf.common.bootstrap import bootstrap_and_run_service
    from aiperf.plugin.enums import ServiceType

    bootstrap_and_run_service(ServiceType.TIMING_MANAGER)


if __name__ == "__main__":
    main()
