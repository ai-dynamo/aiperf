# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Annotated, Any

from pydantic import BeforeValidator, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing_extensions import Self

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.enums.service_enums import ServiceType

_logger = AIPerfLogger(__name__)


def parse_str_or_csv_list(input: Any) -> list[Any]:
    """
    Parses the input to ensure it is either a string or a list. If the input is a string,
    it splits the string by commas and trims any whitespace around each element, returning
    the result as a list. If the input is already a list, it will split each item by commas
    and trim any whitespace around each element, returning the combined result as a list.
    If the input is neither a string nor a list, a ValueError is raised.

    [1, 2, 3] -> [1, 2, 3]
    "1,2,3" -> ["1", "2", "3"]
    ["1,2,3", "4,5,6"] -> ["1", "2", "3", "4", "5", "6"]
    ["1,2,3", 4, 5] -> ["1", "2", "3", 4, 5]
    """
    if isinstance(input, str):
        output = [item.strip() for item in input.split(",")]
    elif isinstance(input, list):
        output = []
        for item in input:
            if isinstance(item, str):
                output.extend([token.strip() for token in item.split(",")])
            else:
                output.append(item)
    else:
        raise ValueError(f"User Config: {input} - must be a string or list")

    return output


def parse_service_types(input: Any | None) -> set[ServiceType] | None:
    """Parses the input to ensure it is a set of service types.
    Will replace hyphens with underscores for user convenience."""
    if input is None:
        return None

    return {
        ServiceType(service_type.replace("-", "_"))
        for service_type in parse_str_or_csv_list(input)
    }


class _Environment(BaseSettings):
    """
    Singleton environment configuration loaded from environment variables.

    All environment variables should be prefixed with AIPERF_.
    Example: AIPERF_LOG_LEVEL=DEBUG
    """

    model_config = SettingsConfigDict(
        env_prefix="AIPERF_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow",
    )

    @model_validator(mode="after")
    def validate_dev_mode(self) -> Self:
        """Validate that developer mode is enabled for features that require it."""
        if self.SHOW_INTERNAL_METRICS and not self.DEV_MODE:
            _logger.warning(
                "Developer mode is not enabled, disabling AIPERF_SHOW_INTERNAL_METRICS"
            )
            self.SHOW_INTERNAL_METRICS = False

        if self.SHOW_EXPERIMENTAL_METRICS and not self.DEV_MODE:
            _logger.warning(
                "Developer mode is not enabled, disabling AIPERF_SHOW_EXPERIMENTAL_METRICS"
            )
            self.SHOW_EXPERIMENTAL_METRICS = False

        return self

    COMMAND_RESPONSE_TIMEOUT: float = Field(
        default=30.0,
        description="Default timeout for command responses in seconds",
    )

    COMMS_REQUEST_TIMEOUT: float = Field(
        default=90.0,
        description="Default timeout for requests from req_clients to rep_clients in seconds",
    )

    CONNECTION_PROBE_INTERVAL: float = Field(
        default=0.1,
        description="Default interval for connection probes in seconds until a response is received",
    )

    CONNECTION_PROBE_TIMEOUT: float = Field(
        default=30.0,
        description="Maximum amount of time to wait for connection probe response",
    )

    CREDIT_PROGRESS_REPORT_INTERVAL: float = Field(
        default=2.0,
        description="Default interval in seconds between credit progress report messages",
    )

    DEV_MODE: bool = Field(
        default=False,
        description="Enable AIPerf Developer mode",
    )

    DEBUG_SERVICES: Annotated[
        set[ServiceType] | None,
        BeforeValidator(parse_service_types),
    ] = Field(
        default=None,
        description="List of services to enable debug logging for. Can be a comma-separated list, a single service type, "
        "or the cli flag can be used multiple times.",
    )

    DISABLE_UVLOOP: bool = Field(
        default=False,
        description="Disable the use of uvloop, and use the default asyncio event loop instead.",
    )

    ENABLE_YAPPI: bool = Field(
        default=False,
        description="Enable yappi profiling (Yet Another Python Profiler) to profile AIPerf's internal python code. "
        "This can be used in the development of AIPerf in order to find performance bottlenecks across the various services. "
        "The output '.prof' files can be viewed with snakeviz. Requires yappi and snakeviz to be installed. "
        "Run 'pip install yappi snakeviz' to install them.",
    )

    HEARTBEAT_INTERVAL: float = Field(
        default=5.0,
        description="Default interval between heartbeat messages in seconds for component services",
    )

    HTTP_CONNECTION_LIMIT: int = Field(
        default=2500,
        description="Maximum number of concurrent connections for HTTP clients",
    )

    MAX_REGISTRATION_ATTEMPTS: int = Field(
        default=10,
        description="Default maximum number of registration attempts for component services before giving up",
    )

    MAX_WORKERS_CAP: int = Field(
        default=32,
        description="Default absolute maximum number of workers to spawn, regardless of the number "
        "of CPU cores. Only applies if the user does not specify a max workers value",
    )

    PROFILE_CONFIGURE_TIMEOUT: float = Field(
        default=300.0,
        description="Default timeout for profile configure command in seconds",
    )

    PROFILE_START_TIMEOUT: float = Field(
        default=60.0,
        description="Default timeout for profile start command in seconds",
    )

    PULL_CLIENT_MAX_CONCURRENCY: int = Field(
        default=100_000,
        description="Default maximum concurrency for pull clients",
    )

    REALTIME_METRICS_INTERVAL: float = Field(
        default=5.0,
        description="Default interval in seconds between real-time metrics messages",
    )

    RECORD_EXPORT_BATCH_SIZE: int = Field(
        default=100,
        description="Default batch size for record export results processor",
    )

    RECORD_PROCESSOR_SCALE_FACTOR: int = Field(
        default=4,
        description="Default scale factor for the number of record processors to spawn based on the "
        "number of workers. This will spawn 1 record processor for every X workers",
    )

    RECORDS_PROGRESS_REPORT_INTERVAL: float = Field(
        default=2.0,
        description="Default interval in seconds between records progress report messages",
    )

    REGISTRATION_INTERVAL: float = Field(
        default=1.0,
        description="Default interval between registration attempts in seconds for component services",
    )

    SERVICE_REGISTRATION_TIMEOUT: float = Field(
        default=30.0,
        description="Default timeout for service registration in seconds",
    )

    SERVICE_START_TIMEOUT: float = Field(
        default=30.0,
        description="Default timeout for service start in seconds",
    )

    SHOW_INTERNAL_METRICS: bool = Field(
        default=False,
        description="[Developer use only] Whether to show internal and hidden metrics in the output",
    )

    SHOW_EXPERIMENTAL_METRICS: bool = Field(
        default=False,
        description="[Developer use only] Whether to show experimental metrics in the output",
    )

    TASK_CANCEL_TIMEOUT_SHORT: float = Field(
        default=2.0,
        description="Maximum time to wait for simple tasks to complete when cancelling them",
    )

    TRACE_SERVICES: Annotated[
        set[ServiceType] | None,
        BeforeValidator(parse_service_types),
    ] = Field(
        default=None,
        description="List of services to enable trace logging for. Can be a comma-separated list, a single service type, "
        "or the cli flag can be used multiple times.",
    )

    UI_MIN_UPDATE_PERCENT: float = Field(
        default=1.0,
        description="Default minimum percentage difference from the last update to trigger a UI"
        " update (for non-dashboard UIs)",
    )

    WORKER_CHECK_INTERVAL: float = Field(
        default=1.0,
        description="Default interval between worker checks in seconds for the WorkerManager",
    )

    WORKER_ERROR_RECOVERY_TIME: float = Field(
        default=3.0,
        description="Default time in seconds from the last time a worker had an error before it is "
        "considered healthy again",
    )

    WORKER_HEALTH_CHECK_INTERVAL: float = Field(
        default=2.0,
        description="Default interval in seconds between worker health check messages",
    )

    WORKER_HIGH_LOAD_CPU_USAGE: float = Field(
        default=75.0,
        description="Default CPU usage threshold for a worker to be considered high load",
    )

    WORKER_HIGH_LOAD_RECOVERY_TIME: float = Field(
        default=5.0,
        description="Default time in seconds from the last time a worker was in high load before it is "
        "considered healthy again",
    )

    WORKER_STALE_TIME: float = Field(
        default=10.0,
        description="Default time in seconds from the last time a worker reported any status before it is "
        "considered stale",
    )

    WORKER_STATUS_SUMMARY_INTERVAL: float = Field(
        default=0.5,
        description="Default interval in seconds between worker status summary messages",
    )

    ZMQ_CONTEXT_TERM_TIMEOUT: float = Field(
        default=10.0,
        description="Default timeout for terminating the ZMQ context in seconds",
    )


# Global singleton instance
Environment = _Environment()
