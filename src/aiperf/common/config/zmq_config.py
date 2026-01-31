# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Annotated, ClassVar

from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self

from aiperf.common.config.cli_parameter import CLIParameter, DisableCLI
from aiperf.common.config.groups import Groups
from aiperf.common.enums import CommAddress
from aiperf.plugin.enums import CommunicationBackend


class BaseZMQProxyConfig(BaseModel, ABC):
    """Configuration Protocol for ZMQ Proxy."""

    @property
    @abstractmethod
    def frontend_address(self) -> str:
        """Get the frontend address based on protocol configuration."""

    @property
    @abstractmethod
    def backend_address(self) -> str:
        """Get the backend address based on protocol configuration."""

    @property
    @abstractmethod
    def control_address(self) -> str | None:
        """Get the control address based on protocol configuration."""

    @property
    @abstractmethod
    def capture_address(self) -> str | None:
        """Get the capture address based on protocol configuration."""


class BaseZMQCommunicationConfig(BaseModel, ABC):
    """Configuration for ZMQ communication."""

    comm_backend: ClassVar[CommunicationBackend]

    # Proxy config options to be overridden by subclasses
    event_bus_proxy_config: ClassVar[BaseZMQProxyConfig]
    dataset_manager_proxy_config: ClassVar[BaseZMQProxyConfig]
    raw_inference_proxy_config: ClassVar[BaseZMQProxyConfig]

    @property
    @abstractmethod
    def records_push_pull_address(self) -> str:
        """Get the inference push/pull address based on protocol configuration."""

    @property
    @abstractmethod
    def credit_router_address(self) -> str:
        """Get the credit router address for bidirectional ROUTER-DEALER credit routing."""

    def get_address(self, address_type: CommAddress) -> str:
        """Get the actual address based on the address type."""
        address_map = {
            CommAddress.EVENT_BUS_PROXY_FRONTEND: self.event_bus_proxy_config.frontend_address,
            CommAddress.EVENT_BUS_PROXY_BACKEND: self.event_bus_proxy_config.backend_address,
            CommAddress.DATASET_MANAGER_PROXY_FRONTEND: self.dataset_manager_proxy_config.frontend_address,
            CommAddress.DATASET_MANAGER_PROXY_BACKEND: self.dataset_manager_proxy_config.backend_address,
            CommAddress.CREDIT_ROUTER: self.credit_router_address,
            CommAddress.RECORDS: self.records_push_pull_address,
            CommAddress.RAW_INFERENCE_PROXY_FRONTEND: self.raw_inference_proxy_config.frontend_address,
            CommAddress.RAW_INFERENCE_PROXY_BACKEND: self.raw_inference_proxy_config.backend_address,
        }

        if address_type not in address_map:
            raise ValueError(f"Invalid address type: {address_type}")

        return address_map[address_type]


class ZMQTCPProxyConfig(BaseZMQProxyConfig):
    """Configuration for TCP proxy."""

    host: str | None = Field(
        default=None,
        description="Host address for TCP connections",
    )
    frontend_port: int = Field(
        default=15555, description="Port for frontend address for proxy"
    )
    backend_port: int = Field(
        default=15556, description="Port for backend address for proxy"
    )
    control_port: int | None = Field(
        default=None, description="Port for control address for proxy"
    )
    capture_port: int | None = Field(
        default=None, description="Port for capture address for proxy"
    )

    @property
    def host_address(self) -> str:
        """Get the host address based on protocol configuration."""
        return self.host or "127.0.0.1"

    @property
    def frontend_address(self) -> str:
        """Get the frontend address based on protocol configuration."""
        return f"tcp://{self.host_address}:{self.frontend_port}"

    @property
    def backend_address(self) -> str:
        """Get the backend address based on protocol configuration."""
        return f"tcp://{self.host_address}:{self.backend_port}"

    @property
    def control_address(self) -> str | None:
        """Get the control address based on protocol configuration."""
        return (
            f"tcp://{self.host_address}:{self.control_port}"
            if self.control_port
            else None
        )

    @property
    def capture_address(self) -> str | None:
        """Get the capture address based on protocol configuration."""
        return (
            f"tcp://{self.host_address}:{self.capture_port}"
            if self.capture_port
            else None
        )


class ZMQIPCProxyConfig(BaseZMQProxyConfig):
    """Configuration for IPC proxy."""

    path: Path | None = Field(default=None, description="Path for IPC sockets")
    name: str = Field(default="proxy", description="Name for IPC sockets")
    enable_control: bool = Field(default=False, description="Enable control socket")
    enable_capture: bool = Field(default=False, description="Enable capture socket")

    @property
    def frontend_address(self) -> str:
        """Get the frontend address based on protocol configuration."""
        if self.path is None:
            raise ValueError("Path is required for IPC transport")
        return f"ipc://{self.path / self.name}_frontend.ipc"

    @property
    def backend_address(self) -> str:
        """Get the backend address based on protocol configuration."""
        if self.path is None:
            raise ValueError("Path is required for IPC transport")
        return f"ipc://{self.path / self.name}_backend.ipc"

    @property
    def control_address(self) -> str | None:
        """Get the control address based on protocol configuration."""
        if self.path is None:
            raise ValueError("Path is required for IPC transport")
        return (
            f"ipc://{self.path / self.name}_control.ipc"
            if self.enable_control
            else None
        )

    @property
    def capture_address(self) -> str | None:
        """Get the capture address based on protocol configuration."""
        if self.path is None:
            raise ValueError("Path is required for IPC transport")
        return (
            f"ipc://{self.path / self.name}_capture.ipc"
            if self.enable_capture
            else None
        )


class ZMQTCPConfig(BaseZMQCommunicationConfig):
    """Configuration for TCP transport."""

    _CLI_GROUP = Groups.ZMQ_COMMUNICATION
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
        CLIParameter(
            name=("--zmq-host"),
            group=_CLI_GROUP,
        ),
    ] = "127.0.0.1"
    records_push_pull_port: Annotated[int, DisableCLI()] = Field(
        default=5557, description="Port for inference push/pull messages"
    )
    credit_router_port: Annotated[int, DisableCLI()] = Field(
        default=5564, description="Port for credit router (ROUTER-DEALER streaming)"
    )
    dataset_manager_proxy_config: Annotated[  # type: ignore
        ZMQTCPProxyConfig, DisableCLI()
    ] = Field(
        default=ZMQTCPProxyConfig(
            frontend_port=5661,
            backend_port=5662,
        ),
        description="Configuration for the ZMQ Proxy. If provided, the proxy will be created and started.",
    )
    event_bus_proxy_config: Annotated[  # type: ignore
        ZMQTCPProxyConfig, DisableCLI()
    ] = Field(
        default=ZMQTCPProxyConfig(
            frontend_port=5663,
            backend_port=5664,
        ),
        description="Configuration for the ZMQ Proxy. If provided, the proxy will be created and started.",
    )
    raw_inference_proxy_config: Annotated[  # type: ignore
        ZMQTCPProxyConfig, DisableCLI()
    ] = Field(
        default=ZMQTCPProxyConfig(
            frontend_port=5665,
            backend_port=5666,
        ),
        description="Configuration for the ZMQ Proxy. If provided, the proxy will be created and started.",
    )

    @property
    def records_push_pull_address(self) -> str:
        """Get the records push/pull address based on protocol configuration."""
        return f"tcp://{self.host}:{self.records_push_pull_port}"

    @property
    def credit_router_address(self) -> str:
        """Get the credit router address for streaming ROUTER-DEALER."""
        return f"tcp://{self.host}:{self.credit_router_port}"


class ZMQIPCConfig(BaseZMQCommunicationConfig):
    """Configuration for IPC transport."""

    _CLI_GROUP = Groups.ZMQ_COMMUNICATION
    comm_backend: ClassVar[CommunicationBackend] = CommunicationBackend.ZMQ_IPC

    @model_validator(mode="after")
    def validate_path(self) -> Self:
        """Validate provided path or create a temporary path for IPC sockets."""
        if self.path is None:
            self.path = Path(tempfile.mkdtemp()) / "aiperf"
        if not self.path.exists():
            self.path.mkdir(parents=True, exist_ok=True)
        for proxy_config in [
            self.dataset_manager_proxy_config,
            self.event_bus_proxy_config,
            self.raw_inference_proxy_config,
        ]:
            if proxy_config.path is None:
                proxy_config.path = self.path
        return self

    path: Annotated[
        Path | None,
        Field(
            description="Directory path for ZMQ IPC (Inter-Process Communication) socket files. When using IPC transport instead of TCP, "
            "AIPerf creates Unix domain socket files in this directory for faster local communication. Auto-generated in system temp directory "
            "if not specified. Only applicable when using IPC communication backend.",
        ),
        CLIParameter(
            name=("--zmq-ipc-path"),
            group=_CLI_GROUP,
        ),
    ] = None

    dataset_manager_proxy_config: Annotated[  # type: ignore
        ZMQIPCProxyConfig, DisableCLI()
    ] = Field(
        default=ZMQIPCProxyConfig(name="dataset_manager_proxy"),
        description="Configuration for the ZMQ Dealer Router Proxy. If provided, the proxy will be created and started.",
    )
    event_bus_proxy_config: Annotated[  # type: ignore
        ZMQIPCProxyConfig, DisableCLI()
    ] = Field(
        default=ZMQIPCProxyConfig(name="event_bus_proxy"),
        description="Configuration for the ZMQ XPUB/XSUB Proxy. If provided, the proxy will be created and started.",
    )
    raw_inference_proxy_config: Annotated[  # type: ignore
        ZMQIPCProxyConfig, DisableCLI()
    ] = Field(
        default=ZMQIPCProxyConfig(name="raw_inference_proxy"),
        description="Configuration for the ZMQ Push/Pull Proxy. If provided, the proxy will be created and started.",
    )

    @property
    def records_push_pull_address(self) -> str:
        """Get the records push/pull address based on protocol configuration."""
        if not self.path:
            raise ValueError("Path is required for IPC transport")
        return f"ipc://{self.path / 'records_push_pull.ipc'}"

    @property
    def credit_router_address(self) -> str:
        """Get the credit router address for streaming ROUTER-DEALER."""
        if not self.path:
            raise ValueError("Path is required for IPC transport")
        return f"ipc://{self.path / 'credit_router.ipc'}"
