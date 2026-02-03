# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import socket
from dataclasses import dataclass
from typing import Any

from aiperf.common.enums.enums import IPVersion
from aiperf.common.environment import Environment


def _get_socket_family() -> socket.AddressFamily:
    """Map IP_VERSION setting to socket address family."""
    ip_version = Environment.HTTP.IP_VERSION.lower()
    if ip_version == IPVersion.V6:
        return socket.AF_INET6
    elif ip_version == IPVersion.AUTO:
        return socket.AF_UNSPEC
    return socket.AF_INET  # Default to IPv4


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

        # Buffer size optimizations for streaming
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, Environment.HTTP.SO_RCVBUF)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, Environment.HTTP.SO_SNDBUF)

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
