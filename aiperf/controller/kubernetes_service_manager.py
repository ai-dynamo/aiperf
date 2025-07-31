# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio

from pydantic import BaseModel

from aiperf.common.config import ServiceConfig
from aiperf.common.config.user_config import UserConfig
from aiperf.common.constants import (
    DEFAULT_SERVICE_REGISTRATION_TIMEOUT,
    DEFAULT_SERVICE_START_TIMEOUT,
)
from aiperf.common.decorators import implements_protocol
from aiperf.common.enums.service_enums import ServiceRunType
from aiperf.common.factories import ServiceManagerFactory
from aiperf.common.protocols import ServiceManagerProtocol
from aiperf.common.types import ServiceTypeT
from aiperf.controller.base_service_manager import BaseServiceManager


class ServiceKubernetesRunInfo(BaseModel):
    """Information about a service running in a Kubernetes pod."""

    pod_name: str
    node_name: str
    namespace: str


@implements_protocol(ServiceManagerProtocol)
@ServiceManagerFactory.register(ServiceRunType.KUBERNETES)
class KubernetesServiceManager(BaseServiceManager):
    """
    Service Manager for starting and stopping services in a Kubernetes cluster.
    """

    def __init__(
        self,
        required_services: dict[ServiceTypeT, int],
        service_config: ServiceConfig,
        user_config: UserConfig,
        **kwargs,
    ):
        super().__init__(required_services, service_config, user_config, **kwargs)

    async def run_service(
        self, service_type: ServiceTypeT, num_replicas: int = 1
    ) -> None:
        """Run a service as a Kubernetes pod."""
        self.logger.debug(f"Running service {service_type} as a Kubernetes pod")
        # TODO: Implement Kubernetes
        raise NotImplementedError(
            "KubernetesServiceManager.run_service not implemented"
        )

    async def shutdown_all_services(self) -> list[BaseException | None]:
        """Stop all required services as Kubernetes pods."""
        self.logger.debug("Stopping all required services as Kubernetes pods")
        # TODO: Implement Kubernetes
        raise NotImplementedError(
            "KubernetesServiceManager.stop_all_services not implemented"
        )

    async def kill_all_services(self) -> list[BaseException | None]:
        """Kill all required services as Kubernetes pods."""
        self.logger.debug("Killing all required services as Kubernetes pods")
        # TODO: Implement Kubernetes
        raise NotImplementedError(
            "KubernetesServiceManager.kill_all_services not implemented"
        )

    async def wait_for_all_services_registration(
        self,
        stop_event: asyncio.Event,
        timeout_seconds: float = DEFAULT_SERVICE_REGISTRATION_TIMEOUT,
    ) -> None:
        """Wait for all required services to be registered in Kubernetes."""
        self.logger.debug(
            "Waiting for all required services to be registered in Kubernetes"
        )
        # TODO: Implement Kubernetes
        raise NotImplementedError(
            "KubernetesServiceManager.wait_for_all_services_registration not implemented"
        )

    async def wait_for_all_services_start(
        self,
        stop_event: asyncio.Event,
        timeout_seconds: float = DEFAULT_SERVICE_START_TIMEOUT,
    ) -> None:
        """Wait for all required services to be started in Kubernetes."""
        self.logger.debug(
            "Waiting for all required services to be started in Kubernetes"
        )
        # TODO: Implement Kubernetes
        raise NotImplementedError(
            "KubernetesServiceManager.wait_for_all_services_start not implemented"
        )
