# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, ClassVar

from pydantic import Field, PrivateAttr, model_validator
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


def build_socket_address(
    path: Path | None, ipc_filename: str, collision_salt: str = ""
) -> str:
    """Build a ZMQ socket address for an inter-service connection.

    Used by both ``ZMQIPCConfig`` and ``ZMQDualBindConfig`` — this is the
    canonical cross-module helper for deriving local IPC endpoint addresses,
    so the bind and connect sides agree without explicit coordination.

    On Linux/macOS: returns ipc://{path}/{ipc_filename} (Unix domain socket).
    On Windows: returns tcp://127.0.0.1:<port> with a deterministic port
    derived from sha256(path/ipc_filename + collision_salt), since Windows
    ZMQ does not support ipc://. Path is required on every platform so
    callers maintain a consistent contract and the hash inputs are stable.

    Args:
        path: IPC directory path used as the hash salt on Windows.
        ipc_filename: Per-endpoint filename, gives each endpoint a different
            hash input within the same config.
        collision_salt: Extra string appended to the hash input on Windows.
            Default ``""`` reproduces the pre-retry behavior. The owning
            config (``ZMQIPCConfig`` / ``ZMQDualBindConfig``) sets this to
            a non-empty value when its ``validate_path`` retry loop has
            rotated to a non-zero salt to escape a port collision. Both
            bind and connect sides see the same value because they share
            the same config instance.
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
        # form (forward-slash separators, lowercased) before hashing. The
        # ``collision_salt`` is appended verbatim; the owning config rotates
        # it when the birthday-paradox collision (~0.56% per draw) actually
        # fires for a given ``path``.
        canonical_path = (
            str(path / ipc_filename).replace("\\", "/").lower() + collision_salt
        )
        digest = hashlib.sha256(canonical_path.encode("utf-8")).hexdigest()
        port_offset = int(digest[:8], 16) % Environment.SERVICE.WINDOWS_TCP_PORT_RANGE
        return (
            f"tcp://127.0.0.1:{Environment.SERVICE.WINDOWS_TCP_BASE_PORT + port_offset}"
        )
    return f"ipc://{path / ipc_filename}"


def _find_port_collision(
    addresses: list[tuple[str, str]],
) -> tuple[str, str, int] | None:
    """Return the first ``(label_a, label_b, port)`` collision among Windows
    TCP-loopback addresses, or ``None`` if no collision exists. No-op on
    POSIX (returns ``None``) where ipc:// addresses cannot collide on port.
    """
    if not IS_WINDOWS:
        return None
    seen: dict[int, str] = {}
    for label, addr in addresses:
        if not addr.startswith("tcp://127.0.0.1:"):
            continue
        port = int(addr.rsplit(":", 1)[1])
        if port in seen:
            return (seen[port], label, port)
        seen[port] = label
    return None


def _validate_no_port_collisions(addresses: list[tuple[str, str]]) -> None:
    """Raise ``ValueError`` if any two endpoints hash to the same Windows
    TCP-fallback port. No-op on POSIX.

    Use ``_find_port_collision`` instead when you want to detect-and-retry
    rather than fail-fast. This helper is the last-line guard used after
    the salt-retry loop has exhausted its attempts.
    """
    collision = _find_port_collision(addresses)
    if collision is None:
        return
    label_a, label_b, port = collision
    raise ValueError(
        f"Windows IPC TCP-fallback port collision: "
        f"{label_a!r} and {label_b!r} both hash to port {port}. "
        f"Set AIPERF_SERVICE_WINDOWS_TCP_BASE_PORT to relocate the "
        f"port window, or change comm.ipc_path (the path's mkdtemp "
        f"randomness feeds the hash). This constraint is Windows-only "
        f"because pyzmq there lacks ipc:// support."
    )


_MAX_COLLISION_RETRIES = 20


def _resolve_collision_salt(
    compute_addresses: "Callable[[str], list[tuple[str, str]]]",
) -> str:
    """Find a ``collision_salt`` such that ``compute_addresses(salt)`` has
    no Windows TCP-loopback port collisions.

    Iterates salt values ``""``, ``":1"``, ``":2"``, ... up to
    ``_MAX_COLLISION_RETRIES`` attempts. Each salt is independently
    appended to the hash input by ``build_socket_address``, so each
    attempt produces an independent random draw of ``n`` ports from the
    Windows port range. With ``n=15`` and range ``20_000`` the birthday-
    paradox per-attempt collision probability is ~0.56%; the chance that
    all 20 retries collide is ~5.7e-46 — effectively zero. No-op on
    POSIX, where ``compute_addresses(salt)`` produces ipc:// addresses
    that cannot collide.

    Args:
        compute_addresses: Callable that takes a ``collision_salt`` and
            returns the ``(label, address)`` pairs that the owning config
            would expose at that salt. Used so the retry loop can re-derive
            addresses for each salt attempt without mutating the config.

    Returns:
        The first salt value with no port collision.

    Raises:
        ValueError: if all retry attempts hit a collision. In practice
            this is unreachable; if you see it, suspect that
            ``compute_addresses`` is not actually salt-dependent
            (a bug in the caller, not in this loop).
    """
    if not IS_WINDOWS:
        return ""
    for attempt in range(_MAX_COLLISION_RETRIES):
        salt = "" if attempt == 0 else f":{attempt}"
        if _find_port_collision(compute_addresses(salt)) is None:
            return salt
    _validate_no_port_collisions(compute_addresses(""))
    raise ValueError(
        f"Could not find a collision-free Windows IPC port assignment "
        f"after {_MAX_COLLISION_RETRIES} salt attempts. This is "
        f"statistically near-impossible and indicates the salt is not "
        f"being threaded into the hash input."
    )


class ZMQIPCProxyConfig(BaseZMQProxyConfig):
    """Configuration for IPC proxy."""

    path: Path | None = Field(default=None, description="Path for IPC sockets")
    name: str = Field(default="proxy", description="Name for IPC sockets")
    enable_control: bool = Field(default=False, description="Enable control socket")
    enable_capture: bool = Field(default=False, description="Enable capture socket")

    # Salt rotated by the owning ``ZMQIPCConfig.validate_path`` when its
    # collision-retry loop has to escape a port collision. Propagated from
    # parent so all endpoints sharing a config see the same salt. Empty
    # string means "no rotation needed" — the default-zero-attempt path.
    _collision_salt: str = PrivateAttr(default="")

    def _addr(self, endpoint: str) -> str:
        """Build an address for the given endpoint (ipc:// on POSIX, tcp:// on Windows)."""
        return build_socket_address(
            self.path, f"{self.name}_{endpoint}.ipc", self._collision_salt
        )

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

    # Salt threaded into every endpoint's hash input on Windows when the
    # default-zero-attempt port assignment hits the birthday-paradox
    # collision (~0.56% per draw). Propagated to all proxy configs so a
    # single source of truth fans out. Empty on POSIX (no-op) and on the
    # common Windows case where the first salt attempt has no collision.
    _collision_salt: str = PrivateAttr(default="")

    @model_validator(mode="after")
    def validate_path(self) -> Self:
        """Set default IPC path, propagate to proxy configs, and resolve any
        Windows TCP-fallback port collision via salt rotation. No-op past
        the path-defaulting on POSIX.
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

        # Find a collision-free salt and propagate it to every proxy so all
        # endpoints share the same hash input. On POSIX
        # ``_resolve_collision_salt`` returns ``""`` and the propagation is
        # a no-op. On Windows the first attempt usually wins; the retry
        # loop fires only on the ~0.56% birthday-paradox draws.
        self._collision_salt = _resolve_collision_salt(self._addresses_with_salt)
        for proxy_config in (
            self.dataset_manager_proxy_config,
            self.event_bus_proxy_config,
            self.raw_inference_proxy_config,
        ):
            proxy_config._collision_salt = self._collision_salt
        return self

    def _addresses_with_salt(self, salt: str) -> list[tuple[str, str]]:
        """Compute every endpoint's address using the given salt, for the
        collision-retry loop. Same shape as the addresses the property
        methods expose post-validation, but parameterized on ``salt`` so
        the retry can probe candidate salts without mutating the config.
        """
        ipc_filename = lambda fname: build_socket_address(  # noqa: E731
            self.path, fname, salt
        )
        proxies = (
            self.dataset_manager_proxy_config,
            self.event_bus_proxy_config,
            self.raw_inference_proxy_config,
        )
        pairs: list[tuple[str, str]] = [
            ("records_push_pull", ipc_filename("records_push_pull.ipc")),
            ("credit_router", ipc_filename("credit_router.ipc")),
            ("credit_return_router", ipc_filename("credit_return_router.ipc")),
            ("credit_return_push_pull", ipc_filename("credit_return_push_pull.ipc")),
            ("control", ipc_filename("control.ipc")),
            ("group_lifecycle", ipc_filename("group_lifecycle.ipc")),
        ]
        for proxy in proxies:
            proxy_addr = lambda endpoint, p=proxy: build_socket_address(  # noqa: E731
                p.path, f"{p.name}_{endpoint}.ipc", salt
            )
            pairs.append((f"{proxy.name}_frontend", proxy_addr("frontend")))
            pairs.append((f"{proxy.name}_backend", proxy_addr("backend")))
            if proxy.enable_control:
                pairs.append((f"{proxy.name}_control", proxy_addr("control")))
            if proxy.enable_capture:
                pairs.append((f"{proxy.name}_capture", proxy_addr("capture")))
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
        return build_socket_address(
            self.path, "records_push_pull.ipc", self._collision_salt
        )

    @property
    def credit_return_push_pull_address(self) -> str:
        """Get the credit-return push/pull address (ipc:// on POSIX, tcp:// on Windows)."""
        return build_socket_address(
            self.path, "credit_return_push_pull.ipc", self._collision_salt
        )

    @property
    def credit_router_address(self) -> str:
        """Get the credit router address (ipc:// on POSIX, tcp:// on Windows)."""
        return build_socket_address(
            self.path, "credit_router.ipc", self._collision_salt
        )

    @property
    def credit_return_router_address(self) -> str:
        """Get the credit return router address (ipc:// on POSIX, tcp:// on Windows)."""
        return build_socket_address(
            self.path, "credit_return_router.ipc", self._collision_salt
        )

    @property
    def control_address(self) -> str:
        """Get the control channel address (ipc:// on POSIX, tcp:// on Windows)."""
        return build_socket_address(self.path, "control.ipc", self._collision_salt)

    @property
    def group_lifecycle_address(self) -> str:
        """Get the group-local lifecycle channel address (ipc:// on POSIX, tcp:// on Windows)."""
        return build_socket_address(
            self.path, "group_lifecycle.ipc", self._collision_salt
        )
