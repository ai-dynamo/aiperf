# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import ClassVar

from pydantic import BaseModel, Field

from aiperf.common.enums import CommAddress


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
    def credit_drop_address(self) -> str:
        """Get the credit drop address based on protocol configuration."""

    @property
    @abstractmethod
    def credit_return_address(self) -> str:
        """Get the credit return address based on protocol configuration."""

    def get_address(self, address_type: CommAddress) -> str:
        """Get the actual address based on the address type."""
        address_map = {
            CommAddress.EVENT_BUS_PROXY_FRONTEND: self.event_bus_proxy_config.frontend_address,
            CommAddress.EVENT_BUS_PROXY_BACKEND: self.event_bus_proxy_config.backend_address,
            CommAddress.DATASET_MANAGER_PROXY_FRONTEND: self.dataset_manager_proxy_config.frontend_address,
            CommAddress.DATASET_MANAGER_PROXY_BACKEND: self.dataset_manager_proxy_config.backend_address,
            CommAddress.CREDIT_DROP: self.credit_drop_address,
            CommAddress.CREDIT_RETURN: self.credit_return_address,
            CommAddress.RECORDS: self.records_push_pull_address,
            CommAddress.RAW_INFERENCE_PROXY_FRONTEND: self.raw_inference_proxy_config.frontend_address,
            CommAddress.RAW_INFERENCE_PROXY_BACKEND: self.raw_inference_proxy_config.backend_address,
        }

        if address_type not in address_map:
            raise ValueError(f"Invalid address type: {address_type}")

        return address_map[address_type]


class ZMQTCPProxyConfig(BaseZMQProxyConfig):
    """Configuration for TCP proxy."""

    host: str = Field(
        default="0.0.0.0",
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
    def frontend_address(self) -> str:
        """Get the frontend address based on protocol configuration."""
        return f"tcp://{self.host}:{self.frontend_port}"

    @property
    def backend_address(self) -> str:
        """Get the backend address based on protocol configuration."""
        return f"tcp://{self.host}:{self.backend_port}"

    @property
    def control_address(self) -> str | None:
        """Get the control address based on protocol configuration."""
        return f"tcp://{self.host}:{self.control_port}" if self.control_port else None

    @property
    def capture_address(self) -> str | None:
        """Get the capture address based on protocol configuration."""
        return f"tcp://{self.host}:{self.capture_port}" if self.capture_port else None


class ZMQIPCProxyConfig(BaseZMQProxyConfig):
    """Configuration for IPC proxy."""

    path: str = Field(default="/tmp/aiperf", description="Path for IPC sockets")
    name: str = Field(default="proxy", description="Name for IPC sockets")
    enable_control: bool = Field(default=False, description="Enable control socket")
    enable_capture: bool = Field(default=False, description="Enable capture socket")

    @property
    def frontend_address(self) -> str:
        """Get the frontend address based on protocol configuration."""
        return f"ipc://{self.path}/{self.name}_frontend.ipc"

    @property
    def backend_address(self) -> str:
        """Get the backend address based on protocol configuration."""
        return f"ipc://{self.path}/{self.name}_backend.ipc"

    @property
    def control_address(self) -> str | None:
        """Get the control address based on protocol configuration."""
        return (
            f"ipc://{self.path}/{self.name}_control.ipc"
            if self.enable_control
            else None
        )

    @property
    def capture_address(self) -> str | None:
        """Get the capture address based on protocol configuration."""
        return (
            f"ipc://{self.path}/{self.name}_capture.ipc"
            if self.enable_capture
            else None
        )


class ZMQTCPConfig(BaseZMQCommunicationConfig):
    """Configuration for TCP transport."""

    host: str = Field(
        default="0.0.0.0",
        description="Host address for TCP connections",
    )
    records_push_pull_port: int = Field(
        default=5557, description="Port for inference push/pull messages"
    )
    credit_drop_port: int = Field(
        default=5562, description="Port for credit drop operations"
    )
    credit_return_port: int = Field(
        default=5563, description="Port for credit return operations"
    )
    dataset_manager_proxy_config: ZMQTCPProxyConfig = Field(  # type: ignore
        default=ZMQTCPProxyConfig(
            frontend_port=5661,
            backend_port=5662,
        ),
        description="Configuration for the ZMQ Proxy. If provided, the proxy will be created and started.",
    )
    event_bus_proxy_config: ZMQTCPProxyConfig = Field(  # type: ignore
        default=ZMQTCPProxyConfig(
            frontend_port=5663,
            backend_port=5664,
        ),
        description="Configuration for the ZMQ Proxy. If provided, the proxy will be created and started.",
    )
    raw_inference_proxy_config: ZMQTCPProxyConfig = Field(  # type: ignore
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
    def credit_drop_address(self) -> str:
        """Get the credit drop address based on protocol configuration."""
        return f"tcp://{self.host}:{self.credit_drop_port}"

    @property
    def credit_return_address(self) -> str:
        """Get the credit return address based on protocol configuration."""
        return f"tcp://{self.host}:{self.credit_return_port}"


class ZMQIPCConfig(BaseZMQCommunicationConfig):
    """Configuration for IPC transport."""

    path: str = Field(default="/tmp/aiperf", description="Path for IPC sockets")
    dataset_manager_proxy_config: ZMQIPCProxyConfig = Field(  # type: ignore
        default=ZMQIPCProxyConfig(name="dataset_manager_proxy"),
        description="Configuration for the ZMQ Dealer Router Proxy. If provided, the proxy will be created and started.",
    )
    event_bus_proxy_config: ZMQIPCProxyConfig = Field(  # type: ignore
        default=ZMQIPCProxyConfig(name="event_bus_proxy"),
        description="Configuration for the ZMQ XPUB/XSUB Proxy. If provided, the proxy will be created and started.",
    )
    raw_inference_proxy_config: ZMQIPCProxyConfig = Field(  # type: ignore
        default=ZMQIPCProxyConfig(name="raw_inference_proxy"),
        description="Configuration for the ZMQ Push/Pull Proxy. If provided, the proxy will be created and started.",
    )

    @property
    def records_push_pull_address(self) -> str:
        """Get the records push/pull address based on protocol configuration."""
        return f"ipc://{self.path}/records_push_pull.ipc"

    @property
    def credit_drop_address(self) -> str:
        """Get the credit drop address based on protocol configuration."""
        return f"ipc://{self.path}/credit_drop.ipc"

    @property
    def credit_return_address(self) -> str:
        """Get the credit return address based on protocol configuration."""
        return f"ipc://{self.path}/credit_return.ipc"
