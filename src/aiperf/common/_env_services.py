# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Service/runtime environment settings subgroups.

Private module for :mod:`aiperf.common.environment`. Contains the
``_ConfigSettings``, ``_RecordSettings``, ``_ServiceSettings``,
``_TimingSettings``, and ``_WorkerSettings`` classes.
"""

import platform
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing_extensions import Self

from aiperf.common.aiperf_logger import AIPerfLogger

_logger = AIPerfLogger(__name__)


class _ConfigSettings(BaseSettings):
    """Configuration file paths for distributed deployments.

    Controls paths to configuration files loaded by services running in containers.
    These are primarily used by `aiperf service` when running in Kubernetes.
    """

    model_config = SettingsConfigDict(
        env_prefix="AIPERF_CONFIG_",
    )

    SERVICE_FILE: Path | None = Field(
        default=None,
        description="Path to service configuration JSON/YAML file. "
        "Default: /etc/aiperf/service_config.json in Kubernetes deployments.",
    )
    USER_FILE: Path | None = Field(
        default=None,
        description="Path to user configuration JSON/YAML file. "
        "Default: /etc/aiperf/user_config.json in Kubernetes deployments.",
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
    EXPORT_FLUSH_INTERVAL: float = Field(
        ge=0.1,
        le=300.0,
        default=2.0,
        description="Maximum seconds record JSONL data may remain buffered before being flushed to disk",
    )
    RAW_EXPORT_BATCH_SIZE: int = Field(
        ge=1,
        le=1000000,
        default=10,
        description="Batch size for raw record writer processor",
    )
    INGEST_BATCH_SIZE: int = Field(
        ge=1,
        le=1000000,
        default=64,
        description="Batch size for record-processor to records-manager ingestion",
    )
    INGEST_BATCH_FLUSH_INTERVAL: float = Field(
        ge=0.001,
        le=300.0,
        default=0.01,
        description="Maximum seconds metric records may remain buffered before ingestion flush",
    )
    PROCESSOR_SCALE_FACTOR: int = Field(
        ge=1,
        le=100,
        default=4,
        description="Scale factor for number of record processors to spawn based on worker count. "
        "Formula: 1 record processor for every X workers. "
        "Default: 1 record processor for every 4 workers.",
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
    CHECKPOINT_INTERVAL: float = Field(
        ge=1.0,
        le=3600.0,
        default=30.0,
        description="Interval in seconds between controller-side partial checkpoint writes",
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
    RECONCILIATION_INTERVAL: float = Field(
        ge=1.0,
        le=300.0,
        default=5.0,
        description="Interval in seconds between credit reconciliation cycles. "
        "The router periodically checks that workers agree on which credits are in-flight. "
        "Credits missing for two consecutive cycles are treated as orphaned.",
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
    DEFAULT_WORKERS_PER_POD: int = Field(
        ge=1,
        le=100,
        default=10,
        description="Default number of worker subprocesses per Kubernetes worker pod. "
        "Each pod downloads the dataset once and shares it across workers via mmap.",
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
        ge=0.01,
        le=600.0,
        default=0.01,
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
    MULTIPROCESSING_START_METHOD: Literal["spawn", "fork", "forkserver"] | None = Field(
        default=None,
        description="Multiprocessing start method. 'spawn' is safest (default on macOS/Windows), "
        "'fork' is faster but unsafe with threads, 'forkserver' is a compromise. "
        "None uses the platform default.",
    )
    HEARTBEAT_INTERVAL: float = Field(
        ge=1.0,
        le=100000.0,
        default=5.0,
        description="Interval in seconds between heartbeat messages for component services",
    )
    HEARTBEAT_MISSED_THRESHOLD: int = Field(
        ge=1,
        le=100,
        default=3,
        description="Number of missed heartbeat intervals before a service is considered stale",
    )
    POD_FAILURE_ABORT_THRESHOLD_PERCENT: int = Field(
        ge=0,
        le=100,
        default=100,
        description="Percentage of worker pods that must fail before aborting the benchmark. "
        "For example, 50 means abort when 50%+ of worker pods have failed. "
        "Set to 100 to abort only when all workers are gone. Set to 0 to disable pod failure abort.",
    )
    PROCESS_MONITOR_INTERVAL: float = Field(
        ge=0.1,
        le=30.0,
        default=0.5,
        description="Interval in seconds between process liveness checks in MultiProcessServiceManager",
    )
    SHUTDOWN_PROPAGATION_DELAY: float = Field(
        ge=0.0,
        le=10.0,
        default=0.5,
        description="Delay in seconds after broadcasting shutdown command to allow message propagation before stopping services",
    )
    FAILURE_SHUTDOWN_TIMEOUT: float = Field(
        ge=1.0,
        le=300.0,
        default=30.0,
        description="Wall-clock cap on the shutdown path inside AIPerfLifecycleMixin._fail. "
        "If cleanup (on_stop hooks, task cancellation) does not complete within this window "
        "after a failed on_init/on_start transition, the process hard-exits via os._exit(1). "
        "Prevents silent zombie containers when cleanup blocks on a cancelled C-ext call.",
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
        default=300.0,
        description="Timeout in seconds for waiting for workers to become ready and for profile start commands",
    )
    PROFILE_CANCEL_TIMEOUT: float = Field(
        ge=1.0,
        le=100000.0,
        default=10.0,
        description="Timeout in seconds for profile cancel command",
    )
    RAW_RECORD_UPLOAD_TIMEOUT: float = Field(
        ge=1.0,
        le=600.0,
        default=60.0,
        description="Timeout in seconds to wait for worker pods to upload raw record files "
        "to the controller API after benchmark completion.",
    )
    REGISTRATION_INTERVAL: float = Field(
        ge=0.001,
        le=100000.0,
        default=0.1,
        description="Interval in seconds between registration attempts for component services",
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
        default=25.0,
        description="Warning threshold in milliseconds for event loop latency (default: 25ms). "
        "If the actual sleep duration exceeds the expected duration by this amount, a warning is logged.",
    )
    EVENT_LOOP_HEALTH_STACKTRACE: bool = Field(
        default=False,
        description="Enable watchdog thread that captures event loop thread stack traces when blocked. "
        "A daemon thread pings the event loop and captures sys._current_frames() when it fails to "
        "respond within the warning threshold. Adds minimal overhead (one thread per monitored service).",
    )
    # Health server settings for Kubernetes probes
    HEALTH_ENABLED: bool = Field(
        default=False,
        description="Enable the lightweight health server for Kubernetes liveness/readiness probes. "
        "When enabled, non-API services will start an HTTP server serving /healthz and /readyz endpoints.",
    )
    HEALTH_HOST: str = Field(
        default="127.0.0.1",
        description="Host to bind the health server to. Use '0.0.0.0' for Kubernetes deployments.",
    )
    HEALTH_PORT: int = Field(
        ge=1,
        le=65535,
        default=8080,
        description="Port for the health server HTTP endpoints (/healthz, /readyz).",
    )
    HEALTH_REQUEST_TIMEOUT: float = Field(
        ge=0.1,
        le=60.0,
        default=5.0,
        description="Timeout in seconds for reading health check HTTP requests.",
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
