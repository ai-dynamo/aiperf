# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import socket
from dataclasses import dataclass


@dataclass(frozen=True)
class SocketDefaults:
    """
    Default values for socket options.
    """

    TCP_NODELAY = 1  # Disable Nagle's algorithm
    TCP_QUICKACK = 1  # Quick ACK mode

    SO_KEEPALIVE = 1  # Enable keepalive
    TCP_KEEPIDLE = 60  # Start keepalive after 1 min idle
    TCP_KEEPINTVL = 30  # Keepalive interval: 30 seconds
    TCP_KEEPCNT = 1  # 1 failed keepalive probes = dead

    SO_LINGER = 0  # Disable linger
    SO_REUSEADDR = 1  # Enable reuse address
    SO_REUSEPORT = 1  # Enable reuse port

    SO_RCVBUF = 1024 * 1024 * 10  # 10MB receive buffer
    SO_SNDBUF = 1024 * 1024 * 10  # 10MB send buffer

    SO_RCVTIMEO = 30  # 30 second receive timeout
    SO_SNDTIMEO = 30  # 30 second send timeout
    TCP_USER_TIMEOUT = 30000  # 30 sec user timeout

    @classmethod
    def apply_to_socket(cls, sock: socket.socket) -> None:
        """Apply the default socket options to the given socket."""

        # Low-latency optimizations for streaming
        sock.setsockopt(socket.SOL_TCP, socket.TCP_NODELAY, cls.TCP_NODELAY)

        # Connection keepalive settings for long-lived SSE connections
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, cls.SO_KEEPALIVE)

        # Fine-tune keepalive timing (Linux-specific)
        if hasattr(socket, "TCP_KEEPIDLE"):
            sock.setsockopt(socket.SOL_TCP, socket.TCP_KEEPIDLE, cls.TCP_KEEPIDLE)
            sock.setsockopt(socket.SOL_TCP, socket.TCP_KEEPINTVL, cls.TCP_KEEPINTVL)
            sock.setsockopt(socket.SOL_TCP, socket.TCP_KEEPCNT, cls.TCP_KEEPCNT)

        # Buffer size optimizations for streaming
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, cls.SO_RCVBUF)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, cls.SO_SNDBUF)

        # Linux-specific TCP optimizations
        if hasattr(socket, "TCP_QUICKACK"):
            sock.setsockopt(socket.SOL_TCP, socket.TCP_QUICKACK, cls.TCP_QUICKACK)

        if hasattr(socket, "TCP_USER_TIMEOUT"):
            sock.setsockopt(
                socket.SOL_TCP, socket.TCP_USER_TIMEOUT, cls.TCP_USER_TIMEOUT
            )


@dataclass(frozen=True)
class AioHttpDefaults:
    """Default values for aiohttp.ClientSession."""

    LIMIT = os.environ.get(
        "AIPERF_HTTP_CONNECTION_LIMIT", 2500
    )  # Maximum number of concurrent connections
    LIMIT_PER_HOST = 0  # Maximum number of concurrent connections per host
    TTL_DNS_CACHE = 300  # Time to live for DNS cache
    USE_DNS_CACHE = True  # Enable DNS cache
    ENABLE_CLEANUP_CLOSED = False  # Disable cleanup of closed connections
    FORCE_CLOSE = False  # Disable force close connections
    KEEPALIVE_TIMEOUT = 300  # Keepalive timeout
    HAPPY_EYEBALLS_DELAY = None  # Happy eyeballs delay (None = disabled)
    SOCKET_FAMILY = socket.AF_INET  # Family of the socket (IPv4)
