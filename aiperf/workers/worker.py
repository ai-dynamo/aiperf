# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys

from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.enums import ServiceType
from aiperf.common.factories import (
    ServiceFactory,
)
from aiperf.common.hooks import (
    on_cleanup,
    on_configure,
    on_stop,
)
from aiperf.common.messages import (
    CommandMessage,
)
from aiperf.common.service.base_component_service import BaseComponentService


@ServiceFactory.register(ServiceType.WORKER)
class Worker(BaseComponentService):
    """Worker is primarily responsible for converting the data into the appropriate
    format for the interface being used by the server. Also responsible for managing
    the conversation between turns.
    """

    def __init__(
        self,
        service_config: ServiceConfig,
        user_config: UserConfig | None = None,
        service_id: str | None = None,
    ):
        super().__init__(service_config=service_config, service_id=service_id)

        self.logger.debug("Initializing worker process")
        self.user_config = user_config
        # self.worker: UniversalWorker | None = None

    @property
    def service_type(self) -> ServiceType:
        """The type of service."""
        return ServiceType.WORKER

    @on_configure
    async def _configure(self, message: CommandMessage) -> None:
        self.logger.debug("Configuring worker process %s", self.service_id)
        # self.worker = UniversalWorker(
        #     service_config=self.service_config,
        #     user_config=cast(UserConfig, message.data),
        #     service_id=f"{self.service_id}",
        # )
        # await self.worker.do_initialize()

    @on_stop
    async def _stop(self) -> None:
        self.logger.debug("Stopping worker process %s", self.service_id)
        # await self.worker.do_shutdown()

    @on_cleanup
    async def _cleanup(self) -> None:
        self.logger.debug("Cleaning up worker process %s", self.service_id)


def main() -> None:
    """Main entry point for the worker process."""

    from aiperf.common.bootstrap import bootstrap_and_run_service

    bootstrap_and_run_service(Worker)


if __name__ == "__main__":
    sys.exit(main())
