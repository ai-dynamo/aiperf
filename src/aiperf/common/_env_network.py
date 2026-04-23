# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Network-facing environment settings subgroups.

Private module for :mod:`aiperf.common.environment`. Contains the
``_APIServerSettings``, ``_CompressionSettings``, ``_HTTPSettings``,
``_LoggingSettings``, and ``_ZMQSettings`` classes. Split out to keep the
top-level ``environment`` module small.
"""

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class _APIServerSettings(BaseSettings):
    """API server settings.

    Controls the host and port of the API server.
    """

    model_config = SettingsConfigDict(env_prefix="AIPERF_API_SERVER_")

    HOST: str = Field(
        default="127.0.0.1",
        description="Host to bind the API server to",
    )
    PORT: int | None = Field(
        ge=1,
        le=65535,
        default=None,
        description="Port to bind the API server to",
    )
    CORS_ORIGINS: list[str] = Field(
        default=[],
        description="List of CORS origins to allow (empty = no CORS, ['*'] = all origins)",
    )
    SHUTDOWN_TIMEOUT: float = Field(
        ge=1.0,
        le=300.0,
        default=5.0,
        description="Timeout in seconds for graceful API server shutdown before force-cancelling",
    )


class _CompressionSettings(BaseSettings):
    """Compression settings for streaming file transfers.

    Controls chunk size and compression levels for zstd and gzip encodings
    used in dataset and results file transfers.
    """

    model_config = SettingsConfigDict(
        env_prefix="AIPERF_COMPRESSION_",
    )

    CHUNK_SIZE: int = Field(
        ge=1024,
        le=1048576,
        default=65536,
        description="Chunk size in bytes for streaming compressed data (default: 64KB)",
    )
    ZSTD_LEVEL: int = Field(
        ge=1,
        le=22,
        default=3,
        description="Zstandard compression level (1=fastest, 22=best compression, default: 3)",
    )
    GZIP_LEVEL: int = Field(
        ge=1,
        le=9,
        default=6,
        description="Gzip compression level (1=fastest, 9=best compression, default: 6)",
    )


class _HTTPSettings(BaseSettings):
    """HTTP client socket and connection configuration.

    Controls low-level socket options, keepalive settings, DNS caching, and connection
    pooling for HTTP clients. These settings optimize performance for high-throughput
    streaming workloads.

    Video Generation Polling:
        For async video generation APIs that use job polling (e.g., SGLang /v1/videos),
        the poll interval is controlled by AIPERF_HTTP_VIDEO_POLL_INTERVAL. The max poll time uses
        the --request-timeout-seconds CLI argument.
    """

    model_config = SettingsConfigDict(
        env_prefix="AIPERF_HTTP_",
    )

    CONNECTION_LIMIT: int = Field(
        ge=1,
        le=65000,
        default=2500,
        description="Maximum number of concurrent HTTP connections",
    )
    KEEPALIVE_TIMEOUT: int = Field(
        ge=0,
        le=10000,
        default=300,
        description="HTTP connection keepalive timeout in seconds for connection pooling",
    )
    SO_RCVBUF: int = Field(
        ge=1024,
        default=10485760,  # 10MB
        description="Socket receive buffer size in bytes (default: 10MB for high-throughput streaming)",
    )
    SO_RCVTIMEO: int = Field(
        ge=1,
        le=100000,
        default=30,
        description="Socket receive timeout in seconds",
    )
    SO_SNDBUF: int = Field(
        ge=1024,
        default=10485760,  # 10MB
        description="Socket send buffer size in bytes (default: 10MB for high-throughput streaming)",
    )
    SO_SNDTIMEO: int = Field(
        ge=1,
        le=100000,
        default=30,
        description="Socket send timeout in seconds",
    )
    TCP_KEEPCNT: int = Field(
        ge=1,
        le=100,
        default=1,
        description="Maximum number of keepalive probes to send before considering the connection dead",
    )
    TCP_KEEPIDLE: int = Field(
        ge=1,
        le=100000,
        default=60,
        description="Time in seconds before starting TCP keepalive probes on idle connections",
    )
    TCP_KEEPINTVL: int = Field(
        ge=1,
        le=100000,
        default=30,
        description="Interval in seconds between TCP keepalive probes",
    )
    TCP_USER_TIMEOUT: int = Field(
        ge=1,
        le=1000000,
        default=30000,
        description="TCP user timeout in milliseconds (Linux-specific, detects dead connections)",
    )
    TTL_DNS_CACHE: int = Field(
        ge=0,
        le=1000000,
        default=300,
        description="DNS cache TTL in seconds for aiohttp client sessions",
    )
    FORCE_CLOSE: bool = Field(
        default=False,
        description="Force close connections after each request",
    )
    ENABLE_CLEANUP_CLOSED: bool = Field(
        default=False,
        description="Enable cleanup of closed ssl connections",
    )
    USE_DNS_CACHE: bool = Field(
        default=True,
        description="Enable DNS cache",
    )
    SSL_VERIFY: bool = Field(
        default=True,
        description="Enable SSL certificate verification. Set to False to disable verification. "
        "WARNING: Disabling this is insecure and should only be used for testing in a trusted environment.",
    )
    REQUEST_CANCELLATION_SEND_TIMEOUT: float = Field(
        ge=10.0,
        le=3600.0,
        default=300.0,
        description="Safety net timeout in seconds for waiting for HTTP request to be fully sent "
        "when request cancellation is enabled. Used as fallback when no explicit timeout is configured "
        "to prevent hanging indefinitely while waiting for the request to be written to the socket.",
    )
    IP_VERSION: Literal["4", "6", "auto"] = Field(
        default="4",
        description="IP version for HTTP socket connections. "
        "Options: '4' (AF_INET, default), '6' (AF_INET6), or 'auto' (AF_UNSPEC, system chooses).",
    )
    TRUST_ENV: bool = Field(
        default=False,
        description="Trust environment variables for HTTP client configuration. "
        "When enabled, aiohttp will read proxy settings from HTTP_PROXY, HTTPS_PROXY, "
        "and NO_PROXY environment variables.",
    )
    VIDEO_POLL_INTERVAL: float = Field(
        ge=0.001,
        le=10.0,
        default=0.1,
        description="Interval in seconds between status polls for async video generation jobs. "
        "Lower values provide faster completion detection but increase server load. "
        "Applies to the aiohttp transport.",
    )


class _LoggingSettings(BaseSettings):
    """Logging system configuration.

    Controls multiprocessing log queue size and other logging behavior.
    """

    model_config = SettingsConfigDict(
        env_prefix="AIPERF_LOGGING_",
    )

    QUEUE_MAXSIZE: int = Field(
        ge=1,
        le=1000000,
        default=1000,
        description="Maximum size of the multiprocessing logging queue",
    )


class _ZMQSettings(BaseSettings):
    """ZMQ socket and communication configuration.

    Controls ZMQ socket timeouts, keepalive settings, retry behavior, and concurrency limits.
    These settings affect reliability and performance of the internal message bus.
    """

    model_config = SettingsConfigDict(
        env_prefix="AIPERF_ZMQ_",
    )

    CONTEXT_TERM_TIMEOUT: float = Field(
        ge=1.0,
        le=100000.0,
        default=10.0,
        description="Timeout in seconds for terminating the ZMQ context during shutdown",
    )
    PULL_YIELD_INTERVAL: int = Field(
        ge=0,
        le=1_000_000,
        default=10,
        description="Yield to the event loop after every N received messages from ZMQ PULL clients. "
        "Prevents event loop starvation during message bursts. "
        "0 disables yielding, 1 yields after every message, 10 yields every 10 messages, etc.",
    )
    REPLY_YIELD_INTERVAL: int = Field(
        ge=0,
        le=1_000_000,
        default=10,
        description="Yield to the event loop after every N received requests from ZMQ ROUTER reply clients. "
        "Prevents event loop starvation during request bursts. "
        "0 disables yielding, 1 yields after every request, 10 yields every 10 requests, etc.",
    )
    REQUEST_YIELD_INTERVAL: int = Field(
        ge=0,
        le=1_000_000,
        default=10,
        description="Yield to the event loop after every N received responses from ZMQ DEALER request clients. "
        "Prevents event loop starvation during response bursts. "
        "0 disables yielding, 1 yields after every response, 10 yields every 10 responses, etc.",
    )
    STREAMING_DEALER_YIELD_INTERVAL: int = Field(
        ge=0,
        le=1_000_000,
        default=10,
        description="Yield to the event loop after every N received messages from ZMQ streaming DEALER clients. "
        "Prevents event loop starvation during message bursts. "
        "0 disables yielding, 1 yields after every message, 10 yields every 10 messages, etc.",
    )
    STREAMING_ROUTER_YIELD_INTERVAL: int = Field(
        ge=0,
        le=1_000_000,
        default=10,
        description="Yield to the event loop after every N received messages from ZMQ streaming ROUTER clients. "
        "Prevents event loop starvation during message bursts. "
        "0 disables yielding, 1 yields after every message, 10 yields every 10 messages, etc.",
    )
    SUB_YIELD_INTERVAL: int = Field(
        ge=0,
        le=1_000_000,
        default=10,
        description="Yield to the event loop after every N received messages from ZMQ SUB clients. "
        "Prevents event loop starvation during message bursts. "
        "0 disables yielding, 1 yields after every message, 10 yields every 10 messages, etc.",
    )
    PULL_MAX_CONCURRENCY: int = Field(
        ge=1,
        le=10000000,
        default=10,
        description="Maximum concurrency for ZMQ PULL clients",
    )
    PUSH_MAX_RETRIES: int = Field(
        ge=1,
        le=100,
        default=2,
        description="Maximum number of retry attempts when pushing messages to ZMQ PUSH socket",
    )
    PUSH_RETRY_DELAY: float = Field(
        ge=0.1,
        le=1000.0,
        default=0.1,
        description="Delay in seconds between retry attempts for ZMQ PUSH operations",
    )
    RCVTIMEO: int = Field(
        ge=1,
        le=10000000,
        default=300000,  # 5 minutes
        description="Socket receive timeout in milliseconds (default: 5 minutes)",
    )
    SNDTIMEO: int = Field(
        ge=1,
        le=10000000,
        default=300000,  # 5 minutes
        description="Socket send timeout in milliseconds (default: 5 minutes)",
    )
    TCP_KEEPALIVE_IDLE: int = Field(
        ge=1,
        le=100000,
        default=10,
        description="Time in seconds before starting TCP keepalive probes on idle ZMQ connections",
    )
    TCP_KEEPALIVE_INTVL: int = Field(
        ge=1,
        le=100000,
        default=10,
        description="Interval in seconds between TCP keepalive probes for ZMQ connections",
    )
