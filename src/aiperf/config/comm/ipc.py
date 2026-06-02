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
# bind and connect sides agree without explicit coordination. The port window
# is configurable via ``AIPERF_SERVICE_WINDOWS_TCP_BASE_PORT`` and
# ``AIPERF_SERVICE_WINDOWS_TCP_PORT_RANGE`` (see ``Environment.SERVICE``).
#
# Defaults chosen to:
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
# port distributions. Same-run intra-process collisions (n sockets within
# one window) are caught at config-validation time by
# ``_validate_no_port_collisions``.


def build_socket_address(path: Path | None, ipc_filename: str) -> str:
    """Build a ZMQ socket address for an inter-service connection.

    Used by both ``ZMQIPCConfig`` and ``ZMQDualBindConfig`` — this is the
    canonical cross-module helper for deriving local IPC endpoint addresses,
    so the bind and connect sides agree without explicit coordination.

    On Linux/macOS: returns ipc://{path}/{ipc_filename} (Unix domain socket).
    On Windows: returns tcp://127.0.0.1:<port> with a deterministic port
    derived from sha256(path/ipc_filename), since Windows ZMQ does not
    support ipc://. Path is required on every platform so callers maintain
    a consistent contract and the hash inputs are stable.
    """
    if path is None:
        raise ValueError("IPC path is required for socket address derivation")
    if IS_WINDOWS:
        # Late import: Environment is loaded lazily on first access, and this
        # module is imported during early bootstrap. Inline the import to keep
        # the cycle broken.
        from aiperf.common.environment import Environment

        # Canonicalize the salt before hashing so bind and connect sides
        # always agree on the derived port. ``str(Path)`` uses backslashes on
        # Windows and forward slashes on POSIX; trailing slashes and mixed
        # casing also affect the raw string. Normalize to a single canonical
        # form (forward-slash separators, lowercased) before hashing.
        canonical_path = str(path / ipc_filename).replace("\\", "/").lower()
        digest = hashlib.sha256(canonical_path.encode("utf-8")).hexdigest()
        port_offset = int(digest[:8], 16) % Environment.SERVICE.WINDOWS_TCP_PORT_RANGE
        return (
            f"tcp://127.0.0.1:{Environment.SERVICE.WINDOWS_TCP_BASE_PORT + port_offset}"
        )
    return f"ipc://{path / ipc_filename}"


def _validate_no_port_collisions(addresses: list[tuple[str, str]]) -> None:
    """Validate that no two endpoints hash to the same Windows TCP-fallback port.

    On Windows, ZMQ IPC sockets fall back to ``tcp://127.0.0.1:<port>`` where
    the port is a deterministic hash of the (path, filename) tuple. Two
    endpoints that hash to the same port cause the second ``bind()`` to fail
    with a misleading ``Address already in use``. This helper detects the
    collision at config-validation time and raises a clear, actionable
    ``ValueError`` instead of letting users debug the opaque downstream error.

    No-op on POSIX where ipc:// addresses don't share this constraint.

    Args:
        addresses: ``(endpoint_label, address)`` tuples. ``endpoint_label``
            is the human-readable name (e.g. ``"records_push_pull"``) used
            in the error message; ``address`` is the derived URL.

    Raises:
        ValueError: if two endpoints hash to the same TCP port. Message
            names both colliding endpoints and points the user at the
            ``AIPERF_SERVICE_WINDOWS_TCP_BASE_PORT`` env var.
    """
    if not IS_WINDOWS:
        return
    seen: dict[int, str] = {}
    for label, addr in addresses:
        if not addr.startswith("tcp://127.0.0.1:"):
            continue
        port = int(addr.rsplit(":", 1)[1])
        if port in seen:
            raise ValueError(
                f"Windows IPC TCP-fallback port collision: "
                f"{seen[port]!r} and {label!r} both hash to port {port}. "
                f"Set AIPERF_SERVICE_WINDOWS_TCP_BASE_PORT to relocate the "
                f"port window, or change comm.ipc_path (the path's mkdtemp "
                f"randomness feeds the hash). This constraint is Windows-only "
                f"because pyzmq there lacks ipc:// support."
            )
        seen[port] = label


class ZMQIPCProxyConfig(BaseZMQProxyConfig):
    """Configuration for IPC proxy."""

    path: Path | None = Field(default=None, description="Path for IPC sockets")
    name: str = Field(default="proxy", description="Name for IPC sockets")
    enable_control: bool = Field(default=False, description="Enable control socket")
    enable_capture: bool = Field(default=False, description="Enable capture socket")

    def _addr(self, endpoint: str) -> str:
        """Build an address for the given endpoint (ipc:// on POSIX, tcp:// on Windows)."""
        return build_socket_address(self.path, f"{self.name}_{endpoint}.ipc")

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
        """Set default IPC path, propagate to proxy configs, and check that
        Windows TCP-fallback ports don't collide.
        """
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

        # Detect Windows TCP-fallback port collisions at config time so users
        # get a clear, actionable error instead of an opaque "Address already
        # in use" from a downstream bind(). No-op on POSIX.
        _validate_no_port_collisions(
            [
                ("records_push_pull", self.records_push_pull_address),
                ("credit_router", self.credit_router_address),
                ("credit_return_router", self.credit_return_router_address),
                ("control", self.control_address),
                ("group_lifecycle", self.group_lifecycle_address),
                *self._proxy_endpoint_addresses(),
            ]
        )
        return self

    def _proxy_endpoint_addresses(self) -> list[tuple[str, str]]:
        """Collect (label, address) for every proxy endpoint for collision
        checking. Optional endpoints (control/capture) are included only
        when enabled."""
        pairs: list[tuple[str, str]] = []
        for proxy in (
            self.dataset_manager_proxy_config,
            self.event_bus_proxy_config,
            self.raw_inference_proxy_config,
        ):
            pairs.append((f"{proxy.name}_frontend", proxy.frontend_address))
            pairs.append((f"{proxy.name}_backend", proxy.backend_address))
            if proxy.control_address is not None:
                pairs.append((f"{proxy.name}_control", proxy.control_address))
            if proxy.capture_address is not None:
                pairs.append((f"{proxy.name}_capture", proxy.capture_address))
        return pairs

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
        return build_socket_address(self.path, "records_push_pull.ipc")

    @property
    def credit_router_address(self) -> str:
        """Get the credit router address (ipc:// on POSIX, tcp:// on Windows)."""
        return build_socket_address(self.path, "credit_router.ipc")

    @property
    def credit_return_router_address(self) -> str:
        """Get the credit return router address (ipc:// on POSIX, tcp:// on Windows)."""
        return build_socket_address(self.path, "credit_return_router.ipc")

    @property
    def control_address(self) -> str:
        """Get the control channel address (ipc:// on POSIX, tcp:// on Windows)."""
        return build_socket_address(self.path, "control.ipc")

    @property
    def group_lifecycle_address(self) -> str:
        """Get the group-local lifecycle channel address (ipc:// on POSIX, tcp:// on Windows)."""
        return build_socket_address(self.path, "group_lifecycle.ipc")
