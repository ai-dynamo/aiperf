# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import sys

from aiperf.common.config.service_config import ServiceConfig
from aiperf.common.decorators import (
    on_cleanup,
    on_configure,
    on_init,
    on_start,
    on_stop,
)
from aiperf.common.enums import ServiceType
from aiperf.common.models import BasePayload
from aiperf.common.service.base_component_service import BaseComponentService


class PostProcessorManager(BaseComponentService):
    """PostProcessorManager is primarily responsible for iterating over the
    records to generate metrics and other conclusions from the records.
    """

    def __init__(
        self, service_config: ServiceConfig, service_id: str | None = None
    ) -> None:
        super().__init__(service_config=service_config, service_id=service_id)
        self.logger.debug("Initializing post processor manager")

    @property
    def service_type(self) -> ServiceType:
        """The type of service."""
        return ServiceType.POST_PROCESSOR_MANAGER

    @on_init
    async def _initialize(self) -> None:
        """Initialize post processor manager-specific components."""
        self.logger.debug("Initializing post processor manager")
        # TODO: Implement post processor manager initialization

    @on_start
    async def _start(self) -> None:
        """Start the post processor manager."""
        self.logger.debug("Starting post processor manager")
        # TODO: Implement post processor manager start

    @on_stop
    async def _stop(self) -> None:
        """Stop the post processor manager."""
        self.logger.debug("Stopping post processor manager")
        # TODO: Implement post processor manager stop

    @on_cleanup
    async def _cleanup(self) -> None:
        """Clean up post processor manager-specific components."""
        self.logger.debug("Cleaning up post processor manager")
        # TODO: Implement post processor manager cleanup

    @on_configure
    async def _configure(self, payload: BasePayload) -> None:
        """Configure the post processor manager."""
        self.logger.debug(f"Configuring post processor manager with payload: {payload}")
        # TODO: Implement post processor manager configuration


def main() -> None:
    """Main entry point for the post processor manager."""

    from aiperf.common.bootstrap import bootstrap_and_run_service

    bootstrap_and_run_service(PostProcessorManager)


if __name__ == "__main__":
    sys.exit(main())
