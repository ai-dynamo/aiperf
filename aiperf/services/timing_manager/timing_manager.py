# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import contextlib
import sys
import time

from aiperf.common.comms.client_enums import ClientType, PullClientType, PushClientType
from aiperf.common.config.service_config import ServiceConfig
from aiperf.common.enums import ServiceState, ServiceType, Topic
from aiperf.common.hooks import on_cleanup, on_configure, on_init, on_start, on_stop
from aiperf.common.models import BasePayload, CreditDropPayload, Message
from aiperf.common.service.base_component_service import BaseComponentService


class TimingManager(BaseComponentService):
    """
    The TimingManager service is responsible to generate the schedule and issuing
    timing credits for requests.
    """

    def __init__(
        self, service_config: ServiceConfig, service_id: str | None = None
    ) -> None:
        super().__init__(service_config=service_config, service_id=service_id)
        self._credit_lock = asyncio.Lock()
        self._credits_available = 100
        self.logger.debug("Initializing timing manager")
        self._credit_drop_task: asyncio.Task | None = None

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
        # TODO: Implement timing manager initialization

    @on_start
    async def _start(self) -> None:
        """Start the timing manager."""
        self.logger.debug("Starting timing manager")
        # TODO: Implement timing manager start
        await self.comms.pull(
            topic=Topic.CREDIT_RETURN,
            callback=self._on_credit_return,
        )
        await self.set_state(ServiceState.RUNNING)
        await asyncio.sleep(3)

        self._credit_drop_task = asyncio.create_task(self._issue_credit_drops())

    @on_stop
    async def _stop(self) -> None:
        """Stop the timing manager."""
        self.logger.debug("Stopping timing manager")
        # TODO: Implement timing manager stop
        if self._credit_drop_task and not self._credit_drop_task.done():
            self._credit_drop_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._credit_drop_task
            self._credit_drop_task = None

    @on_cleanup
    async def _cleanup(self) -> None:
        """Clean up timing manager-specific components."""
        self.logger.debug("Cleaning up timing manager")
        # TODO: Implement timing manager cleanup

    @on_configure
    async def _configure(self, payload: BasePayload) -> None:
        """Configure the timing manager."""
        self.logger.debug(f"Configuring timing manager with payload: {payload}")
        # TODO: Implement timing manager configuration

    async def _issue_credit_drops(self) -> None:
        """Issue credit drops to workers."""
        self.logger.debug("Issuing credit drops to workers")
        # TODO: Actually implement real credit drop logic
        while not self.stop_event.is_set():
            try:
                await asyncio.sleep(0.1)

                async with self._credit_lock:
                    if self._credits_available <= 0:
                        self.logger.warning(
                            "No credits available, skipping credit drop"
                        )
                        continue
                    self.logger.debug("Issuing credit drop")
                    self._credits_available -= 1

                await self.comms.push(
                    topic=Topic.CREDIT_DROP,
                    message=self.create_message(
                        payload=CreditDropPayload(
                            amount=1,
                            timestamp=time.time_ns(),
                        ),
                    ),
                )
            except asyncio.CancelledError:
                self.logger.debug("Credit drop task cancelled")
                break
            except Exception as e:
                self.logger.error(f"Exception issuing credit drop: {e}")
                await asyncio.sleep(0.1)

    async def _on_credit_return(self, message: Message) -> None:
        """Process a credit return response.

        Args:
            message: The response received from the pull request
        """
        self.logger.debug(f"Processing credit return: {message.payload}")
        async with self._credit_lock:
            self._credits_available += message.payload.amount


def main() -> None:
    """Main entry point for the timing manager."""
    from aiperf.common.bootstrap import bootstrap_and_run_service

    bootstrap_and_run_service(TimingManager)


if __name__ == "__main__":
    sys.exit(main())
