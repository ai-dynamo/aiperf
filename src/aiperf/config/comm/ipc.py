# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import tempfile
from pathlib import Path
from typing import Annotated, ClassVar

from pydantic import Field, model_validator
from typing_extensions import Self

from aiperf.config.comm.base import BaseZMQCommunicationConfig, BaseZMQProxyConfig
from aiperf.plugin.enums import CommunicationBackend


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


class ZMQIPCConfig(BaseZMQCommunicationConfig):
    """Configuration for IPC transport."""

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
    ] = None

    dataset_manager_proxy_config: Annotated[  # type: ignore
        ZMQIPCProxyConfig,
        Field(
            description="Configuration for the ZMQ Dealer Router Proxy for the dataset manager.",
        ),
    ] = ZMQIPCProxyConfig(name="dataset_manager_proxy")
    event_bus_proxy_config: Annotated[  # type: ignore
        ZMQIPCProxyConfig,
        Field(
            description="Configuration for the ZMQ XPUB/XSUB Proxy for the event bus.",
        ),
    ] = ZMQIPCProxyConfig(name="event_bus_proxy")
    raw_inference_proxy_config: Annotated[  # type: ignore
        ZMQIPCProxyConfig,
        Field(
            description="Configuration for the ZMQ Push/Pull Proxy for raw inference.",
        ),
    ] = ZMQIPCProxyConfig(name="raw_inference_proxy")

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

    @property
    def credit_return_router_address(self) -> str:
        """Get the credit return router address for dedicated return channel."""
        if not self.path:
            raise ValueError("Path is required for IPC transport")
        return f"ipc://{self.path / 'credit_return_router.ipc'}"

    @property
    def control_address(self) -> str:
        """Get the control channel address."""
        if not self.path:
            raise ValueError("Path is required for IPC transport")
        return f"ipc://{self.path / 'control.ipc'}"

    @property
    def group_lifecycle_address(self) -> str:
        """Get the group-local lifecycle channel address."""
        if not self.path:
            raise ValueError("Path is required for IPC transport")
        return f"ipc://{self.path / 'group_lifecycle.ipc'}"
