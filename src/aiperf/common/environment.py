# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Environment Configuration Module

Provides a hierarchical, type-safe configuration system using Pydantic BaseSettings.
All settings can be configured via environment variables with the AIPERF_ prefix.

Structure:
    Environment.DATASET.*        - Dataset management
    Environment.DEV.*            - Development and debugging settings
    Environment.GPU.*            - GPU telemetry collection
    Environment.HTTP.*           - HTTP client socket and connection settings
    Environment.LOGGING.*        - Logging configuration
    Environment.METRICS.*        - Metrics collection and storage
    Environment.RECORD.*         - Record processing
    Environment.SERVER_METRICS.* - Server metrics collection
    Environment.SERVICE.*        - Service lifecycle and communication
    Environment.TIMING.*         - Timing manager settings
    Environment.UI.*             - User interface settings
    Environment.WORKER.*         - Worker management and scaling
    Environment.ZMQ.*            - ZMQ communication settings

Examples:
    # Via environment variables:
    AIPERF_HTTP_SO_RCVBUF=20971520
    AIPERF_WORKER_CPU_UTILIZATION_FACTOR=0.8

    # In code:
    print(f"Buffer: {Environment.HTTP.SO_RCVBUF}")
    print(f"Workers: {Environment.WORKER.CPU_UTILIZATION_FACTOR}")
"""

import platform
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BeforeValidator, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing_extensions import Self

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.config.config_validators import (
    parse_service_types,
    parse_str_or_csv_list,
)
from aiperf.plugin.enums import ServiceType

_logger = AIPerfLogger(__name__)

__all__ = ["Environment"]


class _DatasetSettings(BaseSettings):
    """Dataset loading and configuration.

    Controls timeouts and behavior for dataset loading operations,
    as well as memory-mapped dataset storage settings.
    """

    model_config = SettingsConfigDict(
        env_prefix="AIPERF_DATASET_",
    )

    CONFIGURATION_TIMEOUT: float = Field(
        ge=1.0,
        le=100000.0,
        default=300.0,
        description="Timeout in seconds for dataset configuration operations",
    )
    MMAP_BASE_PATH: Path | None = Field(
        default=None,
        description="Base path for memory-mapped dataset files. If None, uses system temp directory. "
        "Set to a shared filesystem path for Kubernetes mounted volumes. "
        "Example: AIPERF_DATASET_MMAP_BASE_PATH=/mnt/shared-pvc "
        "creates files at /mnt/shared-pvc/aiperf_mmap_{benchmark_id}/",
    )
    PUBLIC_DATASET_TIMEOUT: float = Field(
        ge=1.0,
        le=100000.0,
        default=300.0,
        description="Timeout in seconds for public dataset loading operations",
    )


class _DeveloperSettings(BaseSettings):
    """Development and debugging configuration.

    Controls developer-focused features like debug logging, profiling, and internal metrics.
    These settings are typically disabled in production environments.
    """

    model_config = SettingsConfigDict(
        env_prefix="AIPERF_DEV_",
    )

    DEBUG_SERVICES: Annotated[
        set[ServiceType] | None,
        BeforeValidator(parse_service_types),
    ] = Field(
        default=None,
        description="List of services to enable DEBUG logging for (comma-separated or multiple flags)",
    )
    ENABLE_YAPPI: bool = Field(
        default=False,
        description="Enable yappi profiling (Yet Another Python Profiler) for performance analysis. "
        "Requires 'pip install yappi snakeviz'",
    )
    MODE: bool = Field(
        default=False,
        description="Enable AIPerf Developer mode for internal metrics and debugging",
    )
    SHOW_EXPERIMENTAL_METRICS: bool = Field(
        default=False,
        description="[Developer use only] Show experimental metrics in output (requires DEV_MODE)",
    )
    SHOW_INTERNAL_METRICS: bool = Field(
        default=False,
        description="[Developer use only] Show internal and hidden metrics in output (requires DEV_MODE)",
    )
    TRACE_SERVICES: Annotated[
        set[ServiceType] | None,
        BeforeValidator(parse_service_types),
    ] = Field(
        default=None,
        description="List of services to enable TRACE logging for (comma-separated or multiple flags)",
    )


class _GPUSettings(BaseSettings):
    """GPU telemetry collection configuration.

    Controls GPU metrics collection frequency, endpoint detection, and shutdown behavior.
    Metrics are collected from DCGM endpoints at the specified interval.
    """

    model_config = SettingsConfigDict(
        env_prefix="AIPERF_GPU_",
        env_parse_enums=True,
    )

    COLLECTION_INTERVAL: float = Field(
        ge=0.01,
        le=300.0,
        default=0.333,
        description="GPU telemetry metrics collection interval in seconds (default: 333ms, ~3Hz)",
    )
    DEFAULT_DCGM_ENDPOINTS: Annotated[
        str | list[str],
        BeforeValidator(parse_str_or_csv_list),
    ] = Field(
        default=["http://localhost:9400/metrics", "http://localhost:9401/metrics"],
        description="Default DCGM endpoint URLs to check for GPU telemetry (comma-separated string or JSON array)",
    )
    EXPORT_BATCH_SIZE: int = Field(
        ge=1,
        le=1000000,
        default=100,
        description="Batch size for telemetry record export results processor",
    )
    REACHABILITY_TIMEOUT: int = Field(
        ge=1,
        le=300,
        default=10,
        description="Timeout in seconds for checking GPU telemetry endpoint reachability during init",
    )
    SHUTDOWN_DELAY: float = Field(
        ge=1.0,
        le=300.0,
        default=5.0,
        description="Delay in seconds before shutting down GPU telemetry service to allow command response transmission",
    )
    THREAD_JOIN_TIMEOUT: float = Field(
        ge=1.0,
        le=300.0,
        default=5.0,
        description="Timeout in seconds for joining GPU telemetry collection threads during shutdown",
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


class _MetricsSettings(BaseSettings):
    """Metrics collection and storage configuration.

    Controls metrics storage allocation and collection behavior.
    """

    model_config = SettingsConfigDict(
        env_prefix="AIPERF_METRICS_",
    )

    ARRAY_INITIAL_CAPACITY: int = Field(
        ge=100,
        le=1000000,
        default=10000,
        description="Initial array capacity for metric storage dictionaries to minimize reallocation",
    )
    USAGE_PCT_DIFF_THRESHOLD: float = Field(
        ge=0.0,
        le=100.0,
        default=10.0,
        description="Percentage difference threshold for flagging discrepancies between API usage and client token counts (default: 10%)",
    )
    OSL_MISMATCH_PCT_THRESHOLD: float = Field(
        ge=0.0,
        le=100.0,
        default=5.0,
        description="Percentage difference threshold for flagging discrepancies between requested and actual output sequence length (default: 5%)",
    )
    OSL_MISMATCH_MAX_TOKEN_THRESHOLD: int = Field(
        ge=1,
        default=50,
        description="Maximum absolute token threshold for OSL mismatch. The effective threshold is min(requested_osl * pct_threshold, this value). Makes threshold tighter for large OSL values (default: 50 tokens)",
    )


class _RecordSettings(BaseSettings):
    """Record processing and export configuration.

    Controls batch sizes, processor scaling, and progress reporting for record processing.
    """

    model_config = SettingsConfigDict(
        env_prefix="AIPERF_RECORD_",
    )

    EXPORT_BATCH_SIZE: int = Field(
        ge=1,
        le=1000000,
        default=100,
        description="Batch size for record export results processor",
    )
    RAW_EXPORT_BATCH_SIZE: int = Field(
        ge=1,
        le=1000000,
        default=10,
        description="Batch size for raw record writer processor",
    )
    PROCESSOR_SCALE_FACTOR: int = Field(
        ge=1,
        le=100,
        default=4,
        description="Scale factor for number of record processors to spawn based on worker count. "
        "Formula: 1 record processor for every X workers",
    )
    PROGRESS_REPORT_INTERVAL: float = Field(
        ge=0.1,
        le=600.0,
        default=2.0,
        description="Interval in seconds between records progress report messages",
    )
    PROCESS_RECORDS_TIMEOUT: float = Field(
        ge=1.0,
        le=100000.0,
        default=300.0,
        description="Timeout in seconds for processing record results",
    )


class _ServerMetricsSettings(BaseSettings):
    """Server metrics collection configuration.

    Controls server metrics collection frequency, endpoint detection, and shutdown behavior.
    Metrics are collected from Prometheus-compatible endpoints at the specified interval.
    Use `--no-server-metrics` CLI flag to disable collection.
    """

    model_config = SettingsConfigDict(
        env_prefix="AIPERF_SERVER_METRICS_",
        env_parse_enums=True,
    )

    COLLECTION_FLUSH_PERIOD: float = Field(
        ge=0.0,
        le=30.0,
        default=2.0,
        description="Time in seconds to continue collecting metrics after profiling completes, "
        "allowing server-side metrics to flush/finalize before shutting down (default: 2.0s)",
    )
    COLLECTION_INTERVAL: float = Field(
        ge=0.001,
        le=300.0,
        default=0.333,
        description="Server metrics collection interval in seconds (default: 333ms, ~3Hz)",
    )
    EXPORT_BATCH_SIZE: int = Field(
        ge=1,
        le=1000000,
        default=100,
        description="Batch size for server metrics jsonl writer export results processor",
    )
    REACHABILITY_TIMEOUT: int = Field(
        ge=1,
        le=300,
        default=10,
        description="Timeout in seconds for checking server metrics endpoint reachability during init",
    )
    SHUTDOWN_DELAY: float = Field(
        ge=1.0,
        le=300.0,
        default=5.0,
        description="Delay in seconds before shutting down server metrics service to allow command response transmission",
    )


class _TimingSettings(BaseSettings):
    """Timing manager configuration.

    Controls timing-related settings for credit phase execution and scheduling.
    """

    model_config = SettingsConfigDict(
        env_prefix="AIPERF_TIMING_",
    )

    CANCEL_DRAIN_TIMEOUT: float = Field(
        ge=1.0,
        le=300.0,
        default=10.0,
        description="Timeout in seconds for waiting for cancelled credits to drain after phase timeout",
    )
    RATE_RAMP_UPDATE_INTERVAL: float = Field(
        ge=0.01,
        le=10.0,
        default=0.1,
        description="Update interval in seconds for continuous rate ramping (default 0.1s = 100ms)",
    )


class _ServiceSettings(BaseSettings):
    """Service lifecycle and inter-service communication configuration.

    Controls timeouts for service registration, startup, shutdown, command handling,
    connection probing, heartbeats, and profile operations.
    """

    model_config = SettingsConfigDict(
        env_prefix="AIPERF_SERVICE_",
    )

    COMMAND_RESPONSE_TIMEOUT: float = Field(
        ge=1.0,
        le=1000.0,
        default=30.0,
        description="Timeout in seconds for command responses",
    )
    COMMS_REQUEST_TIMEOUT: float = Field(
        ge=1.0,
        le=1000.0,
        default=90.0,
        description="Timeout in seconds for requests from req_clients to rep_clients",
    )
    CONNECTION_PROBE_INTERVAL: float = Field(
        ge=0.1,
        le=600.0,
        default=0.1,
        description="Interval in seconds for connection probes while waiting for initial connection to the zmq message bus",
    )
    CONNECTION_PROBE_TIMEOUT: float = Field(
        ge=1.0,
        le=100000.0,
        default=90.0,
        description="Maximum time in seconds to wait for connection probe response while waiting for initial connection to the zmq message bus",
    )
    CREDIT_PROGRESS_REPORT_INTERVAL: float = Field(
        ge=1,
        le=100000.0,
        default=2.0,
        description="Interval in seconds between credit progress report messages",
    )
    DISABLE_UVLOOP: bool = Field(
        default=False,
        description="Disable uvloop and use default asyncio event loop instead",
    )
    HEARTBEAT_INTERVAL: float = Field(
        ge=1.0,
        le=100000.0,
        default=5.0,
        description="Interval in seconds between heartbeat messages for component services",
    )
    PROFILE_CONFIGURE_TIMEOUT: float = Field(
        ge=1.0,
        le=100000.0,
        default=300.0,
        description="Timeout in seconds for profile configure command",
    )
    PROFILE_START_TIMEOUT: float = Field(
        ge=1.0,
        le=100000.0,
        default=60.0,
        description="Timeout in seconds for profile start command",
    )
    PROFILE_CANCEL_TIMEOUT: float = Field(
        ge=1.0,
        le=100000.0,
        default=10.0,
        description="Timeout in seconds for profile cancel command",
    )
    REGISTRATION_INTERVAL: float = Field(
        ge=1.0,
        le=100000.0,
        default=1.0,
        description="Interval in seconds between registration attempts for component services",
    )
    REGISTRATION_MAX_ATTEMPTS: int = Field(
        ge=1,
        le=100000,
        default=10,
        description="Maximum number of registration attempts before giving up",
    )
    REGISTRATION_TIMEOUT: float = Field(
        ge=1.0,
        le=100000.0,
        default=30.0,
        description="Timeout in seconds for service registration",
    )
    START_TIMEOUT: float = Field(
        ge=1.0,
        le=100000.0,
        default=30.0,
        description="Timeout in seconds for service start operations",
    )
    TASK_CANCEL_TIMEOUT_SHORT: float = Field(
        ge=1.0,
        le=100000.0,
        default=2.0,
        description="Maximum time in seconds to wait for simple tasks to complete when cancelling",
    )
    # Event loop health monitoring settings
    EVENT_LOOP_HEALTH_ENABLED: bool = Field(
        default=True,
        description="Enable event loop health monitoring to detect blocked event loops. "
        "When enabled, TimingManager and Worker services periodically check if the event loop is responsive "
        "and log warnings when latency exceeds the threshold.",
    )
    EVENT_LOOP_HEALTH_INTERVAL: float = Field(
        ge=0.05,
        le=10.0,
        default=0.25,
        description="Interval in seconds between event loop health checks (default: 250ms). "
        "The monitor sleeps for this duration and measures actual elapsed time to detect blocking.",
    )
    EVENT_LOOP_HEALTH_WARN_THRESHOLD_MS: float = Field(
        gt=1.0,
        le=10000.0,
        default=10.0,
        description="Warning threshold in milliseconds for event loop latency (default: 10ms). "
        "If the actual sleep duration exceeds the expected duration by this amount, a warning is logged.",
    )

    @model_validator(mode="after")
    def auto_disable_uvloop_on_windows(self) -> Self:
        """Automatically disable uvloop on Windows as it's not supported."""
        if platform.system() == "Windows" and not self.DISABLE_UVLOOP:
            _logger.info(
                "Windows detected: automatically disabling uvloop (not supported on Windows)"
            )
            self.DISABLE_UVLOOP = True
        return self


class _UISettings(BaseSettings):
    """User interface and dashboard configuration.

    Controls refresh rates, update thresholds, and notification behavior for the
    various UI modes (dashboard, tqdm, etc.).
    """

    model_config = SettingsConfigDict(
        env_prefix="AIPERF_UI_",
    )

    LOG_REFRESH_INTERVAL: float = Field(
        ge=0.01,
        le=100000.0,
        default=0.1,
        description="Log viewer refresh interval in seconds (default: 10 FPS)",
    )
    MIN_UPDATE_PERCENT: float = Field(
        ge=0.01,
        le=100.0,
        default=1.0,
        description="Minimum percentage difference from last update to trigger a UI update (for non-dashboard UIs)",
    )
    NOTIFICATION_TIMEOUT: int = Field(
        ge=1,
        le=100000,
        default=3,
        description="Duration in seconds to display UI notifications before auto-dismissing",
    )
    REALTIME_METRICS_INTERVAL: float = Field(
        ge=1.0,
        le=1000.0,
        default=5.0,
        description="Interval in seconds between real-time metrics messages",
    )
    REALTIME_METRICS_ENABLED: bool = Field(
        default=False,
        description="Enable real-time metrics collection and reporting despite UI type",
    )
    SPINNER_REFRESH_RATE: float = Field(
        ge=0.1,
        le=100.0,
        default=0.1,
        description="Progress spinner refresh rate in seconds (default: 10 FPS)",
    )


class _WorkerSettings(BaseSettings):
    """Worker management and auto-scaling configuration.

    Controls worker pool sizing, health monitoring, load detection, and recovery behavior.
    The CPU_UTILIZATION_FACTOR is used in the auto-scaling formula:
    max_workers = max(1, min(int(cpu_count * factor) - 1, MAX_WORKERS_CAP))
    """

    model_config = SettingsConfigDict(
        env_prefix="AIPERF_WORKER_",
    )

    CHECK_INTERVAL: float = Field(
        ge=0.1,
        le=100000.0,
        default=1.0,
        description="Interval in seconds between worker status checks by WorkerManager",
    )
    CPU_UTILIZATION_FACTOR: float = Field(
        ge=0.1,
        le=1.0,
        default=0.75,
        description="Factor multiplied by CPU count to determine default max workers (0.0-1.0). "
        "Formula: max(1, min(int(cpu_count * factor) - 1, MAX_WORKERS_CAP))",
    )
    ERROR_RECOVERY_TIME: float = Field(
        ge=0.1,
        le=1000.0,
        default=3.0,
        description="Time in seconds from last error before worker is considered healthy again",
    )
    HEALTH_CHECK_INTERVAL: float = Field(
        ge=0.1,
        le=1000.0,
        default=2.0,
        description="Interval in seconds between worker health check messages",
    )
    HIGH_LOAD_CPU_USAGE: float = Field(
        ge=50.0,
        le=100.0,
        default=85.0,
        description="CPU usage percentage threshold for considering a worker under high load",
    )
    HIGH_LOAD_RECOVERY_TIME: float = Field(
        ge=0.1,
        le=1000.0,
        default=5.0,
        description="Time in seconds from last high load before worker is considered recovered",
    )
    MAX_WORKERS_CAP: int = Field(
        ge=1,
        le=10000,
        default=32,
        description="Absolute maximum number of workers to spawn, regardless of CPU count",
    )
    STALE_TIME: float = Field(
        ge=0.1,
        le=1000.0,
        default=10.0,
        description="Time in seconds from last status report before worker is considered stale",
    )
    STATUS_SUMMARY_INTERVAL: float = Field(
        ge=0.1,
        le=1000.0,
        default=0.5,
        description="Interval in seconds between worker status summary messages",
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
        default=100_000,
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
        default=60,
        description="Time in seconds before starting TCP keepalive probes on idle ZMQ connections",
    )
    TCP_KEEPALIVE_INTVL: int = Field(
        ge=1,
        le=100000,
        default=10,
        description="Interval in seconds between TCP keepalive probes for ZMQ connections",
    )


class _Environment(BaseSettings):
    """
    Root environment configuration with nested subsystem settings.

    This is a singleton instance that loads configuration from environment variables
    with the AIPERF_ prefix. Settings are organized into logical subsystems for
    better discoverability and maintainability.

    All nested settings can be configured via environment variables using the pattern:
    AIPERF_{SUBSYSTEM}_{SETTING_NAME}

    Example:
        AIPERF_HTTP_CONNECTION_LIMIT=5000
        AIPERF_WORKER_CPU_UTILIZATION_FACTOR=0.8
        AIPERF_ZMQ_RCVTIMEO=600000
    """

    model_config = SettingsConfigDict(
        env_prefix="AIPERF_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow",
    )

    # Nested subsystem settings (alphabetically ordered)
    DATASET: _DatasetSettings = Field(
        default_factory=_DatasetSettings,
        description="Dataset loading and configuration settings",
    )
    DEV: _DeveloperSettings = Field(
        default_factory=_DeveloperSettings,
        description="Development and debugging settings",
    )
    GPU: _GPUSettings = Field(
        default_factory=_GPUSettings,
        description="GPU telemetry collection settings",
    )
    HTTP: _HTTPSettings = Field(
        default_factory=_HTTPSettings,
        description="HTTP client socket and connection settings",
    )
    LOGGING: _LoggingSettings = Field(
        default_factory=_LoggingSettings,
        description="Logging system settings",
    )
    METRICS: _MetricsSettings = Field(
        default_factory=_MetricsSettings,
        description="Metrics collection and storage settings",
    )
    RECORD: _RecordSettings = Field(
        default_factory=_RecordSettings,
        description="Record processing and export settings",
    )
    SERVER_METRICS: _ServerMetricsSettings = Field(
        default_factory=_ServerMetricsSettings,
        description="Server metrics collection settings",
    )
    SERVICE: _ServiceSettings = Field(
        default_factory=_ServiceSettings,
        description="Service lifecycle and communication settings",
    )
    TIMING: _TimingSettings = Field(
        default_factory=_TimingSettings,
        description="Timing manager settings",
    )
    UI: _UISettings = Field(
        default_factory=_UISettings,
        description="User interface and dashboard settings",
    )
    WORKER: _WorkerSettings = Field(
        default_factory=_WorkerSettings,
        description="Worker management and scaling settings",
    )
    ZMQ: _ZMQSettings = Field(
        default_factory=_ZMQSettings,
        description="ZMQ communication settings",
    )

    @model_validator(mode="after")
    def validate_dev_mode(self) -> Self:
        """Validate that developer mode is enabled for features that require it."""
        if self.DEV.SHOW_INTERNAL_METRICS and not self.DEV.MODE:
            _logger.warning(
                "Developer mode is not enabled, disabling AIPERF_DEV_SHOW_INTERNAL_METRICS"
            )
            self.DEV.SHOW_INTERNAL_METRICS = False

        if self.DEV.SHOW_EXPERIMENTAL_METRICS and not self.DEV.MODE:
            _logger.warning(
                "Developer mode is not enabled, disabling AIPERF_DEV_SHOW_EXPERIMENTAL_METRICS"
            )
            self.DEV.SHOW_EXPERIMENTAL_METRICS = False

        return self

    @model_validator(mode="after")
    def validate_profile_configure_timeout(self) -> Self:
        """Validate that the profile configure timeout is at least as long as the dataset configuration timeout."""
        if self.SERVICE.PROFILE_CONFIGURE_TIMEOUT < self.DATASET.CONFIGURATION_TIMEOUT:
            raise ValueError(
                f"AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT: {self.SERVICE.PROFILE_CONFIGURE_TIMEOUT} must be greater than or equal to AIPERF_DATASET_CONFIGURATION_TIMEOUT: {self.DATASET.CONFIGURATION_TIMEOUT}"
            )
        return self


# Global singleton instance
Environment = _Environment()
