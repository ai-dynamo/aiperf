# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import tempfile
from pathlib import Path
from typing import Annotated, ClassVar

from pydantic import Field, model_validator
from typing_extensions import Self

from aiperf.config.comm.base import BaseZMQCommunicationConfig, BaseZMQProxyConfig
from aiperf.plugin.enums import CommunicationBackend


def _event_bus_proxy_default() -> "ZMQDualBindProxyConfig":
    """Default event-bus proxy ports, sourced from ``Environment.ZMQ``.

    Lazy import: see ``aiperf.config.comm.tcp._event_bus_proxy_default``.
    """
    from aiperf.common.environment import Environment

    return ZMQDualBindProxyConfig(
        name="event_bus_proxy",
        tcp_frontend_port=Environment.ZMQ.EVENT_BUS_PROXY_FRONTEND_PORT,
        tcp_backend_port=Environment.ZMQ.EVENT_BUS_PROXY_BACKEND_PORT,
    )


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
    tcp_frontend_port: int = Field(
        default=15555,
        ge=1,
        le=65535,
        description="TCP port for frontend",
    )
    tcp_backend_port: int = Field(
        default=15556,
        ge=1,
        le=65535,
        description="TCP port for backend",
    )

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
        ge=1,
        le=65535,
        description="TCP port for records push/pull communication with remote workers.",
    )
    credit_router_tcp_port: int = Field(
        default=5564,
        ge=1,
        le=65535,
        description="TCP port for credit router communication with remote workers.",
    )
    credit_return_router_tcp_port: int = Field(
        default=5668,
        ge=1,
        le=65535,
        description="TCP port for credit return router communication with remote workers.",
    )
    control_tcp_port: int = Field(
        default=5667,
        ge=1,
        le=65535,
        description="TCP port for control channel (ROUTER-DEALER) with remote workers.",
    )

    event_bus_proxy_config: ZMQDualBindProxyConfig = Field(  # type: ignore
        default_factory=_event_bus_proxy_default,
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
            raise ValueError(
                f"Dual-bind IPC address for endpoint {name!r} requires comm.ipc_path; "
                "set comm.ipc_path or configure controller_host for TCP addresses."
            )
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
    def credit_return_router_address(self) -> str:
        """Get credit return router address based on deployment mode."""
        if self.controller_host:
            return f"tcp://{self.controller_host}:{self.credit_return_router_tcp_port}"
        return self._ipc_addr("credit_return_router")

    @property
    def control_address(self) -> str:
        """Get control channel address based on deployment mode."""
        if self.controller_host:
            return f"tcp://{self.controller_host}:{self.control_tcp_port}"
        return self._ipc_addr("control")

    @property
    def group_lifecycle_address(self) -> str:
        """Get the group-local lifecycle channel address.

        This channel stays local to a single worker group, so it remains on IPC
        even when controller-facing traffic uses TCP.
        """
        return self._ipc_addr("group_lifecycle")

    @property
    def control_tcp_bind_address(self) -> str:
        """Get TCP bind address for control channel (controller-side)."""
        return f"tcp://{self.tcp_host}:{self.control_tcp_port}"

    @property
    def credit_router_tcp_bind_address(self) -> str:
        """Get TCP bind address for credit router dual binding (controller-side)."""
        return f"tcp://{self.tcp_host}:{self.credit_router_tcp_port}"

    @property
    def credit_return_router_tcp_bind_address(self) -> str:
        """Get TCP bind address for credit return router dual binding (controller-side)."""
        return f"tcp://{self.tcp_host}:{self.credit_return_router_tcp_port}"

    @property
    def records_push_pull_tcp_bind_address(self) -> str:
        """Get TCP bind address for records push/pull dual binding (controller-side)."""
        return f"tcp://{self.tcp_host}:{self.records_push_pull_tcp_port}"

    @property
    def _remote_host(self) -> str | None:
        """Remote host for address resolution. Returns controller_host for TCP connections."""
        return self.controller_host
