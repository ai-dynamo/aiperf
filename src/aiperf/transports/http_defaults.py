# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import errno
import logging
import socket
from dataclasses import dataclass
from typing import Any, ClassVar

from aiperf.common.constants import IS_WINDOWS
from aiperf.common.enums.enums import IPVersion
from aiperf.common.environment import Environment

_logger = logging.getLogger(__name__)
MIN_SOCKET_BUFFER_FALLBACK_BYTES = 1024


def _get_socket_family() -> socket.AddressFamily:
    """Map IP_VERSION setting to socket address family."""
    ip_version = Environment.HTTP.IP_VERSION.lower()
    if ip_version == IPVersion.V6:
        return socket.AF_INET6
    elif ip_version == IPVersion.AUTO:
        return socket.AF_UNSPEC
    return socket.AF_INET  # Default to IPv4


def _get_socket_option_label(option_name: int) -> str:
    """Return a readable socket option label for logging."""
    if option_name == socket.SO_RCVBUF:
        return "SO_RCVBUF"
    if option_name == socket.SO_SNDBUF:
        return "SO_SNDBUF"
    return str(option_name)


@dataclass(frozen=True)
class SocketDefaults:
    """
    Default values for socket options.
    """

    TCP_NODELAY = 1  # Disable Nagle's algorithm
    TCP_QUICKACK = 1  # Quick ACK mode
    SO_KEEPALIVE = 1  # Enable keepalive
    SO_LINGER = 0  # Disable linger
    SO_REUSEADDR = 1  # Enable reuse address
    SO_REUSEPORT = 1  # Enable reuse port
    _logged_buffer_fallbacks: ClassVar[set[tuple[int, int, int]]] = set()

    @classmethod
    def _set_socket_buffer(
        cls, sock: socket.socket, option_name: int, value: int
    ) -> None:
        """Set a socket buffer, reducing it if the OS rejects the requested size."""
        candidate = value
        while True:
            try:
                sock.setsockopt(socket.SOL_SOCKET, option_name, candidate)
                if candidate != value:
                    log_fallback_key = (option_name, value, candidate)
                    if log_fallback_key not in cls._logged_buffer_fallbacks:
                        cls._logged_buffer_fallbacks.add(log_fallback_key)
                        _logger.warning(
                            "%s=%d was rejected by the OS with ENOBUFS; "
                            "using %d instead",
                            _get_socket_option_label(option_name),
                            value,
                            candidate,
                        )
                return
            except OSError as e:
                # Some operating systems (e.g. macOS) may raise ENOBUFS
                # when requested socket buffer sizes exceed kernel limits.
                if e.errno != errno.ENOBUFS:
                    raise
                if candidate <= MIN_SOCKET_BUFFER_FALLBACK_BYTES:
                    _logger.error(
                        "Failed to set %s socket buffer at minimum fallback size %d",
                        _get_socket_option_label(option_name),
                        candidate,
                    )
                    raise
                candidate = max(candidate // 2, MIN_SOCKET_BUFFER_FALLBACK_BYTES)

    @classmethod
    def apply_to_socket(cls, sock: socket.socket) -> None:
        """Apply the default socket options to the given socket."""

        # Low-latency optimizations for streaming
        sock.setsockopt(socket.SOL_TCP, socket.TCP_NODELAY, cls.TCP_NODELAY)

        # Connection keepalive settings for long-lived SSE connections
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, cls.SO_KEEPALIVE)

        # Fine-tune keepalive timing (Linux-specific)
        if hasattr(socket, "TCP_KEEPIDLE"):
            sock.setsockopt(
                socket.SOL_TCP, socket.TCP_KEEPIDLE, Environment.HTTP.TCP_KEEPIDLE
            )
            sock.setsockopt(
                socket.SOL_TCP, socket.TCP_KEEPINTVL, Environment.HTTP.TCP_KEEPINTVL
            )
            sock.setsockopt(
                socket.SOL_TCP, socket.TCP_KEEPCNT, Environment.HTTP.TCP_KEEPCNT
            )

        # Buffer size optimizations for streaming. Windows-only skip: setting
        # SO_SNDBUF/SO_RCVBUF explicitly on Windows disables TCP Auto-Tuning
        # (the OS no longer dynamically sizes the buffer to the
        # bandwidth-delay product) and at large values like 10MB causes
        # head-of-line blocking under concurrency. We've observed aiohttp
        # requests stalling for 9+ minutes at --concurrency 6 on Windows
        # Python 3.13 with these values set; the Linux streaming use case
        # that motivated the explicit sizes doesn't apply on Windows.
        if not IS_WINDOWS:
            cls._set_socket_buffer(sock, socket.SO_RCVBUF, Environment.HTTP.SO_RCVBUF)
            cls._set_socket_buffer(sock, socket.SO_SNDBUF, Environment.HTTP.SO_SNDBUF)

        # Linux-specific TCP optimizations
        if hasattr(socket, "TCP_QUICKACK"):
            sock.setsockopt(socket.SOL_TCP, socket.TCP_QUICKACK, cls.TCP_QUICKACK)

        if hasattr(socket, "TCP_USER_TIMEOUT"):
            sock.setsockopt(
                socket.SOL_TCP,
                socket.TCP_USER_TIMEOUT,
                Environment.HTTP.TCP_USER_TIMEOUT,
            )


@dataclass(frozen=True)
class AioHttpDefaults:
    """Default values for aiohttp.ClientSession."""

    LIMIT = (
        Environment.HTTP.CONNECTION_LIMIT
    )  # Maximum number of concurrent connections
    LIMIT_PER_HOST = (
        0  # Maximum number of concurrent connections per host (0 will set to LIMIT)
    )
    TTL_DNS_CACHE = Environment.HTTP.TTL_DNS_CACHE  # Time to live for DNS cache
    USE_DNS_CACHE = Environment.HTTP.USE_DNS_CACHE  # Enable DNS cache
    ENABLE_CLEANUP_CLOSED = (
        Environment.HTTP.ENABLE_CLEANUP_CLOSED
    )  # Disable cleanup of closed connections
    FORCE_CLOSE = Environment.HTTP.FORCE_CLOSE  # Disable force close connections
    KEEPALIVE_TIMEOUT = Environment.HTTP.KEEPALIVE_TIMEOUT  # Keepalive timeout
    HAPPY_EYEBALLS_DELAY = None  # Happy eyeballs delay (None = disabled)
    SOCKET_FAMILY = _get_socket_family()  # Family of the socket based on IP_VERSION
    SSL_VERIFY = Environment.HTTP.SSL_VERIFY  # Enable SSL certificate verification
    TRUST_ENV = (
        Environment.HTTP.TRUST_ENV
    )  # Trust environment variables for proxy config

    @classmethod
    def get_default_kwargs(cls) -> dict[str, Any]:
        """Get the default keyword arguments for aiohttp.ClientSession."""
        return {
            "limit": cls.LIMIT,
            "limit_per_host": cls.LIMIT_PER_HOST,
            "ttl_dns_cache": cls.TTL_DNS_CACHE or None,
            "use_dns_cache": cls.USE_DNS_CACHE,
            "enable_cleanup_closed": cls.ENABLE_CLEANUP_CLOSED,
            "force_close": cls.FORCE_CLOSE,
            "keepalive_timeout": cls.KEEPALIVE_TIMEOUT or None,
            "happy_eyeballs_delay": cls.HAPPY_EYEBALLS_DELAY,
            "family": cls.SOCKET_FAMILY,
            "ssl": cls.SSL_VERIFY,
        }
