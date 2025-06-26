# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import sys

from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.enums import ServiceType
from aiperf.common.factories import ServiceFactory
from aiperf.common.hooks import (
    on_cleanup,
    on_configure,
    on_init,
    on_start,
    on_stop,
)
from aiperf.common.messages import Message
from aiperf.common.service.base_component_service import BaseComponentService


@ServiceFactory.register(ServiceType.RECORDS_MANAGER)
class RecordsManager(BaseComponentService):
    """
    The RecordsManager service is primarily responsible for holding the
    results returned from the workers.
    """

    def __init__(
        self, service_config: ServiceConfig, service_id: str | None = None
    ) -> None:
        super().__init__(service_config=service_config, service_id=service_id)
        self.logger.debug("Initializing records manager")
        self.user_config: UserConfig | None = None
        self.configured_event = asyncio.Event()

    @property
    def service_type(self) -> ServiceType:
        """The type of service."""
        return ServiceType.RECORDS_MANAGER

    @on_init
    async def _initialize(self) -> None:
        """Initialize records manager-specific components."""
        self.logger.debug("Initializing records manager")
        # TODO: Implement records manager initialization

    @on_start
    async def _start(self) -> None:
        """Start the records manager."""
        self.logger.debug("Starting records manager")
        # TODO: Implement records manager start

    @on_stop
    async def _stop(self) -> None:
        """Stop the records manager."""
        self.logger.debug("Stopping records manager")
        # TODO: Implement records manager stop

    @on_cleanup
    async def _cleanup(self) -> None:
        """Clean up records manager-specific components."""
        self.logger.debug("Cleaning up records manager")
        # TODO: Implement records manager cleanup

    @on_configure
    async def _configure(self, message: Message) -> None:
        """Configure the records manager."""
        self.logger.debug(f"Configuring records manager with message: {message}")
        self.user_config = (
            message.data if isinstance(message.data, UserConfig) else None
        )
        self.configured_event.set()


def main() -> None:
    """Main entry point for the records manager."""

    from aiperf.common.bootstrap import bootstrap_and_run_service

    bootstrap_and_run_service(RecordsManager)


if __name__ == "__main__":
    sys.exit(main())
