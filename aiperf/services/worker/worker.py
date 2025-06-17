# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import sys

from aiperf.common.comms.client_enums import ClientType, PullClientType, PushClientType
from aiperf.common.config import ServiceConfig
from aiperf.common.enums import ServiceType, Topic
from aiperf.common.hooks import on_cleanup, on_init, on_run, on_start, on_stop
from aiperf.common.messages import CreditDropMessage, CreditReturnMessage
from aiperf.common.service.base_service import BaseService


class Worker(BaseService):
    """Worker is primarily responsible for converting the data into the appropriate
    format for the interface being used by the server. Also responsible for managing
    the conversation between turns.
    """

    def __init__(
        self, service_config: ServiceConfig, service_id: str | None = None
    ) -> None:
        super().__init__(service_config=service_config, service_id=service_id)
        self.logger.debug("Initializing worker")

    @property
    def service_type(self) -> ServiceType:
        """The type of service."""
        return ServiceType.WORKER

    @property
    def required_clients(self) -> list[ClientType]:
        """The communication clients required by the service."""
        return [
            *(super().required_clients or []),
            PullClientType.CREDIT_DROP,
            PushClientType.CREDIT_RETURN,
        ]

    @on_init
    async def _initialize(self) -> None:
        """Initialize worker-specific components."""
        self.logger.debug("Initializing worker")

    @on_run
    async def _run(self) -> None:
        """Automatically start the worker in the run method."""
        await self.start()

    @on_start
    async def _start(self) -> None:
        """Start the worker."""
        self.logger.debug("Starting worker")
        # Subscribe to the credit drop topic
        await self.comms.register_pull_callback(
            message_type=Topic.CREDIT_DROP,
            callback=self._process_credit_drop,
        )

    @on_stop
    async def _stop(self) -> None:
        """Stop the worker."""
        self.logger.debug("Stopping worker")

    @on_cleanup
    async def _cleanup(self) -> None:
        """Clean up worker-specific components."""
        self.logger.debug("Cleaning up worker")

    async def _process_credit_drop(self, message: CreditDropMessage) -> None:
        """Process a credit drop response.

        Args:
            message: The message received from the credit drop
        """
        self.logger.debug(f"Processing credit drop: {message}")
        # TODO: Implement actual worker logic
        await asyncio.sleep(1)  # Simulate some processing time

        self.logger.debug("Returning credits")
        await self.comms.push(
            topic=Topic.CREDIT_RETURN,
            message=CreditReturnMessage(
                service_id=self.service_id,
                amount=1,
            ),
        )


def main() -> None:
    """Main entry point for the worker."""

    import uvloop

    from aiperf.common.config import load_service_config

    # Load the service configuration
    cfg = load_service_config()

    # Create and run the worker
    worker = Worker(cfg)
    uvloop.run(worker.run_forever())


if __name__ == "__main__":
    sys.exit(main())
