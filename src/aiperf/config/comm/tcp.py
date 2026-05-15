# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated, ClassVar

from pydantic import Field, model_validator
from typing_extensions import Self

from aiperf.config.comm.base import BaseZMQCommunicationConfig, BaseZMQProxyConfig
from aiperf.plugin.enums import CommunicationBackend


def _event_bus_proxy_default() -> "ZMQTCPProxyConfig":
    """Default event-bus proxy ports, sourced from ``Environment.ZMQ``.

    Lazy import: ``Environment`` re-imports the config tree during bootstrap;
    a module-level import of ``Environment`` here would deadlock.
    """
    from aiperf.common.environment import Environment

    return ZMQTCPProxyConfig(
        frontend_port=Environment.ZMQ.EVENT_BUS_PROXY_FRONTEND_PORT,
        backend_port=Environment.ZMQ.EVENT_BUS_PROXY_BACKEND_PORT,
    )


class ZMQTCPProxyConfig(BaseZMQProxyConfig):
    """Configuration for TCP proxy."""

    host: str | None = Field(
        default=None,
        description="Host address for TCP connections",
    )
    frontend_port: int = Field(
        default=15555,
        ge=1,
        le=65535,
        description="Port for frontend address for proxy",
    )
    backend_port: int = Field(
        default=15556,
        ge=1,
        le=65535,
        description="Port for backend address for proxy",
    )
    control_port: int | None = Field(
        default=None,
        ge=1,
        le=65535,
        description="Port for control address for proxy",
    )
    capture_port: int | None = Field(
        default=None,
        ge=1,
        le=65535,
        description="Port for capture address for proxy",
    )

    def _addr(self, port: int) -> str:
        """Build a TCP address for the given port."""
        return f"tcp://{self.host or '127.0.0.1'}:{port}"

    @property
    def frontend_address(self) -> str:
        """Get the frontend address based on protocol configuration."""
        return self._addr(self.frontend_port)

    @property
    def backend_address(self) -> str:
        """Get the backend address based on protocol configuration."""
        return self._addr(self.backend_port)

    @property
    def control_address(self) -> str | None:
        """Get the control address based on protocol configuration."""
        return self._addr(self.control_port) if self.control_port else None

    @property
    def capture_address(self) -> str | None:
        """Get the capture address based on protocol configuration."""
        return self._addr(self.capture_port) if self.capture_port else None


class ZMQTCPConfig(BaseZMQCommunicationConfig):
    """Configuration for TCP transport."""

    comm_backend: ClassVar[CommunicationBackend] = CommunicationBackend.ZMQ_TCP

    @model_validator(mode="after")
    def validate_host(self) -> Self:
        """Fill in the host address for the proxy configs if not provided."""
        for proxy_config in [
            self.dataset_manager_proxy_config,
            self.event_bus_proxy_config,
            self.raw_inference_proxy_config,
        ]:
            if proxy_config.host is None:
                proxy_config.host = self.host
        return self

    host: Annotated[
        str,
        Field(
            description="Host address for internal ZMQ TCP communication between AIPerf services. Defaults to `127.0.0.1` (localhost) for "
            "single-machine deployments. For distributed setups, set to a reachable IP address. All internal service-to-service communication "
            "(message bus, dataset manager, workers) uses this host for TCP sockets.",
        ),
    ] = "127.0.0.1"
    records_push_pull_port: Annotated[
        int,
        Field(
            default=5557,
            ge=1,
            le=65535,
            description="Port for inference push/pull messages",
        ),
    ] = 5557
    credit_router_port: Annotated[
        int,
        Field(
            default=5564,
            ge=1,
            le=65535,
            description="Port for credit router (ROUTER-DEALER streaming)",
        ),
    ] = 5564
    credit_return_router_port: Annotated[
        int,
        Field(
            default=5668,
            ge=1,
            le=65535,
            description="Port for credit return router (ROUTER-DEALER credit returns)",
        ),
    ] = 5668
    control_port: Annotated[
        int,
        Field(
            default=5667,
            ge=1,
            le=65535,
            description="Port for control channel (ROUTER-DEALER)",
        ),
    ] = 5667
    dataset_manager_proxy_config: Annotated[  # type: ignore
        ZMQTCPProxyConfig,
        Field(
            description="Configuration for the ZMQ Proxy for the dataset manager.",
        ),
    ] = ZMQTCPProxyConfig(
        frontend_port=5661,
        backend_port=5662,
    )
    event_bus_proxy_config: Annotated[  # type: ignore
        ZMQTCPProxyConfig,
        Field(
            default_factory=_event_bus_proxy_default,
            description="Configuration for the ZMQ Proxy for the event bus.",
        ),
    ]
    raw_inference_proxy_config: Annotated[  # type: ignore
        ZMQTCPProxyConfig,
        Field(
            description="Configuration for the ZMQ Proxy for raw inference.",
        ),
    ] = ZMQTCPProxyConfig(
        frontend_port=5665,
        backend_port=5666,
    )

    @property
    def records_push_pull_address(self) -> str:
        """Get the records push/pull address based on protocol configuration."""
        return f"tcp://{self.host}:{self.records_push_pull_port}"

    @property
    def credit_router_address(self) -> str:
        """Get the credit router address for streaming ROUTER-DEALER."""
        return f"tcp://{self.host}:{self.credit_router_port}"

    @property
    def credit_return_router_address(self) -> str:
        """Get the credit return router address for dedicated return channel."""
        return f"tcp://{self.host}:{self.credit_return_router_port}"

    @property
    def control_address(self) -> str:
        """Get the control channel address."""
        return f"tcp://{self.host}:{self.control_port}"

    @property
    def group_lifecycle_address(self) -> str:
        """Get the group-local lifecycle channel address."""
        return f"tcp://{self.host}:{self.control_port + 1}"
