#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import asyncio
import sys

import uvloop

from aiperf.common.config.service_config import ServiceConfig
from aiperf.common.enums import (
    ClientType,
    PullClientType,
    PushClientType,
    ServiceType,
    Topic,
)
from aiperf.common.models.payload_models import CreditDropPayload, CreditReturnPayload
from aiperf.common.service.base_service import BaseService


class Worker(BaseService):
    """Worker responsible for sending requests to the server."""

    def __init__(self, service_config: ServiceConfig, service_id: str = None) -> None:
        super().__init__(service_config=service_config, service_id=service_id)
        self.logger.debug("Initializing worker")

    @property
    def service_type(self) -> ServiceType:
        """The type of service."""
        return ServiceType.WORKER

    @property
    def required_clients(self) -> list[ClientType]:
        """The communication clients required by the service."""
        return [PullClientType.CREDIT_DROP, PushClientType.CREDIT_RETURN]

    async def _initialize(self) -> None:
        """Initialize worker-specific components."""
        self.logger.debug("Initializing worker")

    async def _on_start(self) -> None:
        """Start the worker."""
        self.logger.debug("Starting worker")
        # Subscribe to the credit drop topic
        await self.comms.pull(
            topic=Topic.CREDIT_DROP,
            callback=self._process_credit_drop,
        )

    async def _on_stop(self) -> None:
        """Stop the worker."""
        self.logger.debug("Stopping worker")

    async def _cleanup(self) -> None:
        """Clean up worker-specific components."""
        self.logger.debug("Cleaning up worker")

    async def _process_credit_drop(self, payload: CreditDropPayload) -> None:
        """Process a credit drop response.

        Args:
            payload: The payload received from the credit drop response
        """
        self.logger.debug(f"Processing credit drop: {payload}")

        # TODO: Implement actual worker logic
        await asyncio.sleep(1)  # Simulate some processing time

        self.logger.debug("Returning credits")
        await self.comms.push(
            topic=Topic.CREDIT_RETURN,
            message=self.create_message(
                payload=CreditReturnPayload(amount=1),
            ),
        )


if __name__ == "__main__":
    uvloop.install()

    # Load the service configuration
    from aiperf.common.config.config_loader import load_service_config

    cfg = load_service_config()
    worker = Worker(cfg)
    sys.exit(uvloop.run(worker.run()))
