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

from pydantic import BaseModel

from aiperf.common.config.service_config import ServiceConfig
from aiperf.common.enums import ServiceType
from aiperf.common.errors.base_error import Error
from aiperf.services.system_controller.base_service_manager import BaseServiceManager


class ServiceKubernetesRunInfo(BaseModel):
    """Information about a service running in a Kubernetes pod."""

    pod_name: str
    node_name: str
    namespace: str


class KubernetesServiceManager(BaseServiceManager):
    """
    Service Manager for starting and stopping services in a Kubernetes cluster.
    """

    def __init__(
        self, required_service_types: list[ServiceType], config: ServiceConfig
    ):
        super().__init__(required_service_types, config)

    async def initialize_all_services(self) -> Error | None:
        """Initialize all required services as Kubernetes pods."""
        self.logger.debug("Initializing all required services as Kubernetes pods")
        # TODO: Implement Kubernetes
        raise NotImplementedError

    async def stop_all_services(self) -> Error | None:
        """Stop all required services as Kubernetes pods."""
        self.logger.debug("Stopping all required services as Kubernetes pods")
        # TODO: Implement Kubernetes
        raise NotImplementedError

    async def wait_for_all_services_registration(
        self, stop_event: asyncio.Event, timeout_seconds: int = 30
    ) -> Error | None:
        """Wait for all required services to be registered in Kubernetes."""
        self.logger.debug(
            "Waiting for all required services to be registered in Kubernetes"
        )
        # TODO: Implement Kubernetes
        raise NotImplementedError

    async def wait_for_all_services_start(self) -> Error | None:
        """Wait for all required services to be started in Kubernetes."""
        self.logger.debug(
            "Waiting for all required services to be started in Kubernetes"
        )
        # TODO: Implement Kubernetes
        raise NotImplementedError
