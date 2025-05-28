#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
import asyncio

from pydantic import BaseModel

from aiperf.common.config.service_config import ServiceConfig
from aiperf.common.enums import ServiceType
from aiperf.services.service_manager.base import BaseServiceManager


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

    async def initialize_all_services(self) -> None:
        """Initialize all required services as Kubernetes pods."""
        self.logger.debug("Initializing all required services as Kubernetes pods")
        # TODO: Implement Kubernetes
        raise NotImplementedError(
            "KubernetesServiceManager.initialize_all_services not implemented"
        )

    async def stop_all_services(self) -> None:
        """Stop all required services as Kubernetes pods."""
        self.logger.debug("Stopping all required services as Kubernetes pods")
        # TODO: Implement Kubernetes
        raise NotImplementedError(
            "KubernetesServiceManager.stop_all_services not implemented"
        )

    async def wait_for_all_services_registration(
        self, stop_event: asyncio.Event, timeout_seconds: int = 30
    ) -> None:
        """Wait for all required services to be registered in Kubernetes."""
        self.logger.debug(
            "Waiting for all required services to be registered in Kubernetes"
        )
        # TODO: Implement Kubernetes
        raise NotImplementedError(
            "KubernetesServiceManager.wait_for_all_services_registration not implemented"
        )

    async def wait_for_all_services_start(self) -> None:
        """Wait for all required services to be started in Kubernetes."""
        self.logger.debug(
            "Waiting for all required services to be started in Kubernetes"
        )
        # TODO: Implement Kubernetes
        raise NotImplementedError(
            "KubernetesServiceManager.wait_for_all_services_start not implemented"
        )
