# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import sys

from aiperf.common.config.service_config import ServiceConfig
from aiperf.common.enums import ServiceType
from aiperf.common.hooks import (
    on_cleanup,
    on_configure,
    on_init,
    on_start,
    on_stop,
)
from aiperf.common.models import BasePayload
from aiperf.common.service.base_component_service import BaseComponentService


class DatasetManager(BaseComponentService):
    """
    The DatasetManager primary responsibility is to manage the data generation or acquisition.
    For synthetic generation, it contains the code to generate the prompts or tokens.
    It will have an API for dataset acquisition of a dataset if available in a remote repository or database.
    """

    def __init__(
        self,
        service_config: ServiceConfig,
        service_id: str | None = None,
    ) -> None:
        super().__init__(service_config=service_config, service_id=service_id)
        self.logger.debug("Initializing dataset manager")

    @property
    def service_type(self) -> ServiceType:
        """The type of service."""
        return ServiceType.DATASET_MANAGER

    @on_init
    async def _initialize(self) -> None:
        """Initialize dataset manager-specific components."""
        self.logger.debug("Initializing dataset manager")
        # TODO: Implement dataset manager initialization

    @on_start
    async def _start(self) -> None:
        """Start the dataset manager."""
        self.logger.debug("Starting dataset manager")
        # TODO: Implement dataset manager start

    @on_stop
    async def _stop(self) -> None:
        """Stop the dataset manager."""
        self.logger.debug("Stopping dataset manager")
        # TODO: Implement dataset manager stop

    @on_cleanup
    async def _cleanup(self) -> None:
        """Clean up dataset manager-specific components."""
        self.logger.debug("Cleaning up dataset manager")
        # TODO: Implement dataset manager cleanup

    @on_configure
    async def _configure(self, payload: BasePayload) -> None:
        """Configure the dataset manager."""
        self.logger.debug(f"Configuring dataset manager with payload: {payload}")
        # TODO: Implement dataset manager configuration


def main() -> None:
    """Main entry point for the dataset manager."""

    from aiperf.common.bootstrap import bootstrap_and_run_service

    bootstrap_and_run_service(DatasetManager)


if __name__ == "__main__":
    sys.exit(main())
