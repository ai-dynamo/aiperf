# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Annotated, ClassVar

from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self

from aiperf.common.config.cli_parameter import CLIParameter
from aiperf.common.config.groups import Groups
from aiperf.common.enums import CommAddress, CommunicationBackend


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
            description="Host address for TCP connections",
        ),
        CLIParameter(
            name=("--zmq-host"),
            group=_CLI_GROUP,
        ),
    ] = "0.0.0.0"
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

    _CLI_GROUP = Groups.ZMQ_COMMUNICATION
    comm_backend: ClassVar[CommunicationBackend] = CommunicationBackend.ZMQ_IPC

    path: Annotated[
        str,
        Field(
            description="Path for IPC sockets",
        ),
        CLIParameter(
            name=("--zmq-ipc-path"),
            group=_CLI_GROUP,
        ),
    ] = "/tmp/aiperf"

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
