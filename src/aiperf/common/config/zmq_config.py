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

    @property
    def additional_frontend_bind_address(self) -> str | None:
        """Additional frontend address for dual-bind proxies. None by default."""
        return None

    @property
    def additional_backend_bind_address(self) -> str | None:
        """Additional backend address for dual-bind proxies. None by default."""
        return None

    def resolve_frontend(self, remote_host: str | None = None) -> str:
        """Resolve the frontend address. Subclasses may use remote_host for dual-bind."""
        return self.frontend_address

    def resolve_backend(self, remote_host: str | None = None) -> str:
        """Resolve the backend address. Subclasses may use remote_host for dual-bind."""
        return self.backend_address


class BaseZMQCommunicationConfig(BaseModel, ABC):
    """Configuration for ZMQ communication."""

    comm_backend: ClassVar[CommunicationBackend]

    # Proxy config options to be overridden by subclasses
    event_bus_proxy_config: ClassVar[BaseZMQProxyConfig]
    dataset_manager_proxy_config: ClassVar[BaseZMQProxyConfig]
    raw_inference_proxy_config: ClassVar[BaseZMQProxyConfig]

    ipc_path: Annotated[Path | None, DisableCLI()] = Field(
        default=None,
        description="IPC socket directory path. None for non-IPC transports (e.g., TCP).",
    )

    @property
    def _remote_host(self) -> str | None:
        """Remote host for address resolution. None means use local addresses."""
        return None

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
        host = self._remote_host
        match address_type:
            case CommAddress.EVENT_BUS_PROXY_FRONTEND:
                return self.event_bus_proxy_config.resolve_frontend(host)
            case CommAddress.EVENT_BUS_PROXY_BACKEND:
                return self.event_bus_proxy_config.resolve_backend(host)
            case CommAddress.DATASET_MANAGER_PROXY_FRONTEND:
                return self.dataset_manager_proxy_config.resolve_frontend(host)
            case CommAddress.DATASET_MANAGER_PROXY_BACKEND:
                return self.dataset_manager_proxy_config.resolve_backend(host)
            case CommAddress.RAW_INFERENCE_PROXY_FRONTEND:
                return self.raw_inference_proxy_config.resolve_frontend(host)
            case CommAddress.RAW_INFERENCE_PROXY_BACKEND:
                return self.raw_inference_proxy_config.resolve_backend(host)
            case CommAddress.CREDIT_ROUTER:
                return self.credit_router_address
            case CommAddress.RECORDS:
                return self.records_push_pull_address
            case _:
                raise ValueError(f"Invalid address type: {address_type}")


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


class ZMQIPCProxyConfig(BaseZMQProxyConfig):
    """Configuration for IPC proxy."""

    path: Path | None = Field(default=None, description="Path for IPC sockets")
    name: str = Field(default="proxy", description="Name for IPC sockets")
    enable_control: bool = Field(default=False, description="Enable control socket")
    enable_capture: bool = Field(default=False, description="Enable capture socket")

    def _addr(self, endpoint: str) -> str:
        """Build an IPC address for the given endpoint."""
        if self.path is None:
            raise ValueError("Path is required for IPC transport")
        return f"ipc://{self.path / self.name}_{endpoint}.ipc"

    @property
    def frontend_address(self) -> str:
        """Get the frontend address based on protocol configuration."""
        return self._addr("frontend")

    @property
    def backend_address(self) -> str:
        """Get the backend address based on protocol configuration."""
        return self._addr("backend")

    @property
    def control_address(self) -> str | None:
        """Get the control address based on protocol configuration."""
        return self._addr("control") if self.enable_control else None

    @property
    def capture_address(self) -> str | None:
        """Get the capture address based on protocol configuration."""
        return self._addr("capture") if self.enable_capture else None


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
        """Set default IPC path and propagate to proxy configs."""
        if self.path is None:
            self.path = Path(tempfile.mkdtemp()) / "aiperf"
        self.ipc_path = self.path
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


class ZMQDualBindProxyConfig(BaseZMQProxyConfig):
    """Configuration for dual-bind proxy (IPC + TCP).

    Supports binding a single proxy to both IPC (for local services) and TCP
    (for remote services). Used in Kubernetes deployments where controller
    services connect via IPC and worker pods connect via TCP.
    """

    # IPC settings
    ipc_path: Path | None = Field(default=None, description="Path for IPC sockets")
    name: str = Field(default="proxy", description="Name for IPC sockets")

    # TCP settings
    tcp_host: str = Field(
        default="127.0.0.1",
        description="TCP bind host (use 0.0.0.0 for all interfaces)",
    )
    tcp_frontend_port: int = Field(default=15555, description="TCP port for frontend")
    tcp_backend_port: int = Field(default=15556, description="TCP port for backend")

    # Control/capture (optional, IPC only)
    enable_control: bool = Field(default=False, description="Enable control socket")
    enable_capture: bool = Field(default=False, description="Enable capture socket")

    def _ipc_addr(self, endpoint: str) -> str:
        """Build an IPC address for the given endpoint."""
        if self.ipc_path is None:
            raise ValueError("IPC path is required for dual-bind transport")
        return f"ipc://{self.ipc_path / self.name}_{endpoint}.ipc"

    def _tcp_addr(self, port: int) -> str:
        """Build a TCP address for the given port (bind-side)."""
        return f"tcp://{self.tcp_host}:{port}"

    def _resolve(
        self, remote_host: str | None, tcp_port: int, ipc_endpoint: str
    ) -> str:
        """Resolve address: TCP with remote_host if set, otherwise IPC."""
        if remote_host:
            return f"tcp://{remote_host}:{tcp_port}"
        return self._ipc_addr(ipc_endpoint)

    def resolve_frontend(self, remote_host: str | None) -> str:
        """Get frontend address: TCP with remote_host if set, otherwise IPC."""
        return self._resolve(remote_host, self.tcp_frontend_port, "frontend")

    def resolve_backend(self, remote_host: str | None) -> str:
        """Get backend address: TCP with remote_host if set, otherwise IPC."""
        return self._resolve(remote_host, self.tcp_backend_port, "backend")

    @property
    def frontend_address(self) -> str:
        """Get the primary frontend address (IPC)."""
        return self._ipc_addr("frontend")

    @property
    def frontend_tcp_address(self) -> str:
        """Get the TCP frontend address for remote connections."""
        return self._tcp_addr(self.tcp_frontend_port)

    @property
    def additional_frontend_bind_address(self) -> str | None:
        """TCP frontend address for dual-bind proxy binding."""
        return self.frontend_tcp_address

    @property
    def backend_address(self) -> str:
        """Get the primary backend address (IPC)."""
        return self._ipc_addr("backend")

    @property
    def backend_tcp_address(self) -> str:
        """Get the TCP backend address for remote connections."""
        return self._tcp_addr(self.tcp_backend_port)

    @property
    def additional_backend_bind_address(self) -> str | None:
        """TCP backend address for dual-bind proxy binding."""
        return self.backend_tcp_address

    @property
    def control_address(self) -> str | None:
        """Get the control address (IPC only)."""
        return self._ipc_addr("control") if self.enable_control else None

    @property
    def capture_address(self) -> str | None:
        """Get the capture address (IPC only)."""
        return self._ipc_addr("capture") if self.enable_capture else None


class ZMQDualBindConfig(BaseZMQCommunicationConfig):
    """Configuration for dual-bind (IPC + TCP) Kubernetes deployments.

    This config enables proxies to bind to both IPC (for co-located services in
    the controller pod) and TCP (for remote worker pods). Services select their
    transport based on the `controller_host` setting:
    - If controller_host is None: use IPC (local deployment)
    - If controller_host is set: use TCP to connect to that host (remote deployment)
    """

    _CLI_GROUP = Groups.ZMQ_COMMUNICATION
    comm_backend: ClassVar[CommunicationBackend] = CommunicationBackend.ZMQ_DUAL_BIND

    @property
    def proxy_configs(self) -> list[ZMQDualBindProxyConfig]:
        """All proxy configs for iteration."""
        return [
            self.event_bus_proxy_config,
            self.dataset_manager_proxy_config,
            self.raw_inference_proxy_config,
        ]

    @model_validator(mode="after")
    def validate_paths(self) -> Self:
        """Set default IPC path and propagate settings to proxy configs."""
        if self.ipc_path is None:
            self.ipc_path = Path(tempfile.mkdtemp()) / "aiperf"
        for proxy_config in self.proxy_configs:
            if proxy_config.ipc_path is None:
                proxy_config.ipc_path = self.ipc_path
            proxy_config.tcp_host = self.tcp_host
        return self

    ipc_path: Annotated[
        Path | None,
        Field(
            description="Directory path for IPC socket files.",
        ),
    ] = None

    tcp_host: Annotated[
        str,
        Field(
            description="TCP bind host for proxies (Defaults to 127.0.0.1 for localhost, use 0.0.0.0 for all interfaces).",
        ),
    ] = "127.0.0.1"

    controller_host: Annotated[
        str | None,
        Field(
            description="Controller host for remote TCP connections. When set, services "
            "connect via TCP to this host instead of IPC. Set via JobSet DNS in Kubernetes.",
        ),
    ] = None

    records_push_pull_tcp_port: int = Field(
        default=5557,
        description="TCP port for records push/pull communication with remote workers.",
    )
    credit_router_tcp_port: int = Field(
        default=5564,
        description="TCP port for credit router communication with remote workers.",
    )

    event_bus_proxy_config: ZMQDualBindProxyConfig = Field(  # type: ignore
        default=ZMQDualBindProxyConfig(
            name="event_bus_proxy",
            tcp_frontend_port=5663,
            tcp_backend_port=5664,
        ),
        description="Event bus proxy configuration (XPUB/XSUB).",
    )
    dataset_manager_proxy_config: ZMQDualBindProxyConfig = Field(  # type: ignore
        default=ZMQDualBindProxyConfig(
            name="dataset_manager_proxy",
            tcp_frontend_port=5661,
            tcp_backend_port=5662,
        ),
        description="Dataset manager proxy configuration (DEALER/ROUTER).",
    )
    raw_inference_proxy_config: ZMQDualBindProxyConfig = Field(  # type: ignore
        default=ZMQDualBindProxyConfig(
            name="raw_inference_proxy",
            tcp_frontend_port=5665,
            tcp_backend_port=5666,
        ),
        description="Raw inference proxy configuration (PUSH/PULL).",
    )

    def _ipc_addr(self, name: str) -> str:
        """Build an IPC address for the given endpoint name."""
        if not self.ipc_path:
            raise ValueError("IPC path is required")
        return f"ipc://{self.ipc_path / name}.ipc"

    @property
    def records_push_pull_address(self) -> str:
        """Get records push/pull address based on deployment mode."""
        if self.controller_host:
            return f"tcp://{self.controller_host}:{self.records_push_pull_tcp_port}"
        return self._ipc_addr("records_push_pull")

    @property
    def credit_router_address(self) -> str:
        """Get credit router address based on deployment mode."""
        if self.controller_host:
            return f"tcp://{self.controller_host}:{self.credit_router_tcp_port}"
        return self._ipc_addr("credit_router")

    @property
    def credit_router_tcp_bind_address(self) -> str:
        """Get TCP bind address for credit router dual binding (controller-side)."""
        return f"tcp://{self.tcp_host}:{self.credit_router_tcp_port}"

    @property
    def records_push_pull_tcp_bind_address(self) -> str:
        """Get TCP bind address for records push/pull dual binding (controller-side)."""
        return f"tcp://{self.tcp_host}:{self.records_push_pull_tcp_port}"

    @property
    def _remote_host(self) -> str | None:
        """Remote host for address resolution. Returns controller_host for TCP connections."""
        return self.controller_host
