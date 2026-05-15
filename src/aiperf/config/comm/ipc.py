# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import tempfile
from pathlib import Path
from typing import Annotated, ClassVar

from pydantic import Field, model_validator
from typing_extensions import Self

from aiperf.common.constants import IS_WINDOWS
from aiperf.config.comm.base import BaseZMQCommunicationConfig, BaseZMQProxyConfig
from aiperf.plugin.enums import CommunicationBackend

# Windows fallback: ZMQ does not support ipc:// on Windows. Use TCP loopback
# with a deterministic port derived from a hash of the would-be IPC path, so
# bind and connect sides agree without explicit coordination.
#
# Range chosen to:
#  - stay below the OS ephemeral-port range (49152+ on Linux/macOS/Win10+)
#  - sit above the cluster of common service ports (HTTP/Prometheus/vLLM/
#    Ollama/OTLP/etc.) so a co-running service on localhost is unlikely to
#    have already bound a port we hash to
#  - keep birthday-paradox collision probability low for AIPerf's ~15 sockets:
#    P(collision) ≈ 1 - exp(-n^2 / (2 * RANGE)). At RANGE=20000, n=15 → ~0.56%.
#
# Per-aiperf-run uniqueness is provided by ``tempfile.mkdtemp()`` randomness
# in ``ZMQIPCConfig.validate_path`` — two concurrent aiperf processes get
# different ipc paths, which feed into the salt, which produces different
# port distributions.
_WINDOWS_TCP_BASE_PORT = 28000
_WINDOWS_TCP_PORT_RANGE = 20000


def _build_socket_address(path: Path | None, ipc_filename: str) -> str:
    """Build a ZMQ socket address for an inter-service connection.

    On Linux/macOS: returns ipc://{path}/{ipc_filename} (Unix domain socket).
    On Windows: returns tcp://127.0.0.1:<port> with a deterministic port
    derived from sha256(path/ipc_filename), since Windows ZMQ does not
    support ipc://. Path is required on every platform so callers maintain
    a consistent contract and the hash inputs are stable.
    """
    if path is None:
        raise ValueError("IPC path is required for socket address derivation")
    if IS_WINDOWS:
        salt = f"{path}/{ipc_filename}"
        digest = hashlib.sha256(salt.encode()).hexdigest()
        port_offset = int(digest[:8], 16) % _WINDOWS_TCP_PORT_RANGE
        return f"tcp://127.0.0.1:{_WINDOWS_TCP_BASE_PORT + port_offset}"
    return f"ipc://{path / ipc_filename}"


class ZMQIPCProxyConfig(BaseZMQProxyConfig):
    """Configuration for IPC proxy."""

    path: Path | None = Field(default=None, description="Path for IPC sockets")
    name: str = Field(default="proxy", description="Name for IPC sockets")
    enable_control: bool = Field(default=False, description="Enable control socket")
    enable_capture: bool = Field(default=False, description="Enable capture socket")

    def _addr(self, endpoint: str) -> str:
        """Build an address for the given endpoint (ipc:// on POSIX, tcp:// on Windows)."""
        return _build_socket_address(self.path, f"{self.name}_{endpoint}.ipc")

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
        """Get the records push/pull address (ipc:// on POSIX, tcp:// on Windows)."""
        return _build_socket_address(self.path, "records_push_pull.ipc")

    @property
    def credit_router_address(self) -> str:
        """Get the credit router address (ipc:// on POSIX, tcp:// on Windows)."""
        return _build_socket_address(self.path, "credit_router.ipc")

    @property
    def credit_return_router_address(self) -> str:
        """Get the credit return router address (ipc:// on POSIX, tcp:// on Windows)."""
        return _build_socket_address(self.path, "credit_return_router.ipc")

    @property
    def control_address(self) -> str:
        """Get the control channel address (ipc:// on POSIX, tcp:// on Windows)."""
        return _build_socket_address(self.path, "control.ipc")

    @property
    def group_lifecycle_address(self) -> str:
        """Get the group-local lifecycle channel address (ipc:// on POSIX, tcp:// on Windows)."""
        return _build_socket_address(self.path, "group_lifecycle.ipc")
