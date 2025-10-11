# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
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


@dataclass(frozen=True)
class EnvironmentDefaults:
    """Default values for environment variables. All environment variables
    should be prefixed with AIPERF_."""

    COMMAND_RESPONSE_TIMEOUT = 30.0
    COMMS_REQUEST_TIMEOUT = 90.0
    CONNECTION_PROBE_INTERVAL = 0.1
    CONNECTION_PROBE_TIMEOUT = 30.0
    CREDIT_PROGRESS_REPORT_INTERVAL = 2.0
    DEV_MODE = False
    DEBUG_SERVICES = None
    DISABLE_UVLOOP = False
    ENABLE_YAPPI = False
    HEARTBEAT_INTERVAL = 5.0
    HTTP_CONNECTION_LIMIT = 2500
    MAX_REGISTRATION_ATTEMPTS = 10
    MAX_WORKERS_CAP = 32
    PROFILE_CONFIGURE_TIMEOUT = 300.0
    PROFILE_START_TIMEOUT = 60.0
    PULL_CLIENT_MAX_CONCURRENCY = 100_000
    REALTIME_METRICS_INTERVAL = 5.0
    RECORD_EXPORT_BATCH_SIZE = 100
    RECORD_PROCESSOR_SCALE_FACTOR = 4
    RECORDS_PROGRESS_REPORT_INTERVAL = 2.0
    REGISTRATION_INTERVAL = 1.0
    SERVICE_REGISTRATION_TIMEOUT = 30.0
    SERVICE_START_TIMEOUT = 30.0
    SHOW_INTERNAL_METRICS = False
    SHOW_EXPERIMENTAL_METRICS = False
    TASK_CANCEL_TIMEOUT_SHORT = 2.0
    TRACE_SERVICES = None
    UI_MIN_UPDATE_PERCENT = 1.0
    WORKER_CHECK_INTERVAL = 1.0
    WORKER_ERROR_RECOVERY_TIME = 3.0
    WORKER_HEALTH_CHECK_INTERVAL = 2.0
    WORKER_HIGH_LOAD_CPU_USAGE = 75.0
    WORKER_HIGH_LOAD_RECOVERY_TIME = 5.0
    WORKER_STALE_TIME = 10.0
    WORKER_STATUS_SUMMARY_INTERVAL = 0.5
    ZMQ_CONTEXT_TERM_TIMEOUT = 10.0


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

    COMMAND_RESPONSE_TIMEOUT: Annotated[
        float,
        Field(
            description="Default timeout for command responses in seconds",
        ),
    ] = EnvironmentDefaults.COMMAND_RESPONSE_TIMEOUT

    COMMS_REQUEST_TIMEOUT: Annotated[
        float,
        Field(
            description="Default timeout for requests from req_clients to rep_clients in seconds",
        ),
    ] = EnvironmentDefaults.COMMS_REQUEST_TIMEOUT

    CONNECTION_PROBE_INTERVAL: Annotated[
        float,
        Field(
            description="Default interval for connection probes in seconds until a response is received",
        ),
    ] = EnvironmentDefaults.CONNECTION_PROBE_INTERVAL

    CONNECTION_PROBE_TIMEOUT: Annotated[
        float,
        Field(
            description="Maximum amount of time to wait for connection probe response",
        ),
    ] = EnvironmentDefaults.CONNECTION_PROBE_TIMEOUT

    CREDIT_PROGRESS_REPORT_INTERVAL: Annotated[
        float,
        Field(
            description="Default interval in seconds between credit progress report messages",
        ),
    ] = EnvironmentDefaults.CREDIT_PROGRESS_REPORT_INTERVAL

    DEV_MODE: Annotated[
        bool,
        Field(
            description="Enable AIPerf Developer mode",
        ),
    ] = EnvironmentDefaults.DEV_MODE

    DEBUG_SERVICES: Annotated[
        set[ServiceType] | None,
        Field(
            description="List of services to enable debug logging for. Can be a comma-separated list, a single service type, "
            "or the cli flag can be used multiple times.",
        ),
        BeforeValidator(parse_service_types),
    ] = EnvironmentDefaults.DEBUG_SERVICES

    DISABLE_UVLOOP: Annotated[
        bool,
        Field(
            description="Disable the use of uvloop, and use the default asyncio event loop instead.",
        ),
    ] = EnvironmentDefaults.DISABLE_UVLOOP

    ENABLE_YAPPI: Annotated[
        bool,
        Field(
            description="Enable yappi profiling (Yet Another Python Profiler) to profile AIPerf's internal python code. "
            "This can be used in the development of AIPerf in order to find performance bottlenecks across the various services. "
            "The output '.prof' files can be viewed with snakeviz. Requires yappi and snakeviz to be installed. "
            "Run 'pip install yappi snakeviz' to install them.",
        ),
    ] = EnvironmentDefaults.ENABLE_YAPPI

    HEARTBEAT_INTERVAL: Annotated[
        float,
        Field(
            description="Default interval between heartbeat messages in seconds for component services",
        ),
    ] = EnvironmentDefaults.HEARTBEAT_INTERVAL

    HTTP_CONNECTION_LIMIT: Annotated[
        int,
        Field(
            description="Maximum number of concurrent connections for HTTP clients",
        ),
    ] = EnvironmentDefaults.HTTP_CONNECTION_LIMIT

    MAX_REGISTRATION_ATTEMPTS: Annotated[
        int,
        Field(
            description="Default maximum number of registration attempts for component services before giving up",
        ),
    ] = EnvironmentDefaults.MAX_REGISTRATION_ATTEMPTS

    MAX_WORKERS_CAP: Annotated[
        int,
        Field(
            description="Default absolute maximum number of workers to spawn, regardless of the number "
            "of CPU cores. Only applies if the user does not specify a max workers value",
        ),
    ] = EnvironmentDefaults.MAX_WORKERS_CAP

    PROFILE_CONFIGURE_TIMEOUT: Annotated[
        float,
        Field(
            description="Default timeout for profile configure command in seconds",
        ),
    ] = EnvironmentDefaults.PROFILE_CONFIGURE_TIMEOUT

    PROFILE_START_TIMEOUT: Annotated[
        float,
        Field(
            description="Default timeout for profile start command in seconds",
        ),
    ] = EnvironmentDefaults.PROFILE_START_TIMEOUT

    PULL_CLIENT_MAX_CONCURRENCY: Annotated[
        int,
        Field(
            description="Default maximum concurrency for pull clients",
        ),
    ] = EnvironmentDefaults.PULL_CLIENT_MAX_CONCURRENCY

    REALTIME_METRICS_INTERVAL: Annotated[
        float,
        Field(
            description="Default interval in seconds between real-time metrics messages",
        ),
    ] = EnvironmentDefaults.REALTIME_METRICS_INTERVAL

    RECORD_EXPORT_BATCH_SIZE: Annotated[
        int,
        Field(
            description="Default batch size for record export results processor",
        ),
    ] = EnvironmentDefaults.RECORD_EXPORT_BATCH_SIZE

    RECORD_PROCESSOR_SCALE_FACTOR: Annotated[
        int,
        Field(
            description="Default scale factor for the number of record processors to spawn based on the "
            "number of workers. This will spawn 1 record processor for every X workers",
        ),
    ] = EnvironmentDefaults.RECORD_PROCESSOR_SCALE_FACTOR

    RECORDS_PROGRESS_REPORT_INTERVAL: Annotated[
        float,
        Field(
            description="Default interval in seconds between records progress report messages",
        ),
    ] = EnvironmentDefaults.RECORDS_PROGRESS_REPORT_INTERVAL

    REGISTRATION_INTERVAL: Annotated[
        float,
        Field(
            description="Default interval between registration attempts in seconds for component services",
        ),
    ] = EnvironmentDefaults.REGISTRATION_INTERVAL

    SERVICE_REGISTRATION_TIMEOUT: Annotated[
        float,
        Field(
            description="Default timeout for service registration in seconds",
        ),
    ] = EnvironmentDefaults.SERVICE_REGISTRATION_TIMEOUT

    SERVICE_START_TIMEOUT: Annotated[
        float,
        Field(
            description="Default timeout for service start in seconds",
        ),
    ] = EnvironmentDefaults.SERVICE_START_TIMEOUT

    SHOW_INTERNAL_METRICS: Annotated[
        bool,
        Field(
            description="[Developer use only] Whether to show internal and hidden metrics in the output",
        ),
    ] = EnvironmentDefaults.SHOW_INTERNAL_METRICS

    SHOW_EXPERIMENTAL_METRICS: Annotated[
        bool,
        Field(
            description="[Developer use only] Whether to show experimental metrics in the output",
        ),
    ] = EnvironmentDefaults.SHOW_EXPERIMENTAL_METRICS

    TASK_CANCEL_TIMEOUT_SHORT: Annotated[
        float,
        Field(
            description="Maximum time to wait for simple tasks to complete when cancelling them",
        ),
    ] = EnvironmentDefaults.TASK_CANCEL_TIMEOUT_SHORT

    TRACE_SERVICES: Annotated[
        set[ServiceType] | None,
        Field(
            description="List of services to enable trace logging for. Can be a comma-separated list, a single service type, "
            "or the cli flag can be used multiple times.",
        ),
        BeforeValidator(parse_service_types),
    ] = EnvironmentDefaults.TRACE_SERVICES

    UI_MIN_UPDATE_PERCENT: Annotated[
        float,
        Field(
            description="Default minimum percentage difference from the last update to trigger a UI"
            " update (for non-dashboard UIs)",
        ),
    ] = EnvironmentDefaults.UI_MIN_UPDATE_PERCENT

    WORKER_CHECK_INTERVAL: Annotated[
        float,
        Field(
            description="Default interval between worker checks in seconds for the WorkerManager",
        ),
    ] = EnvironmentDefaults.WORKER_CHECK_INTERVAL

    WORKER_ERROR_RECOVERY_TIME: Annotated[
        float,
        Field(
            description="Default time in seconds from the last time a worker had an error before it is "
            "considered healthy again",
        ),
    ] = EnvironmentDefaults.WORKER_ERROR_RECOVERY_TIME

    WORKER_HEALTH_CHECK_INTERVAL: Annotated[
        float,
        Field(
            description="Default interval in seconds between worker health check messages",
        ),
    ] = EnvironmentDefaults.WORKER_HEALTH_CHECK_INTERVAL

    WORKER_HIGH_LOAD_CPU_USAGE: Annotated[
        float,
        Field(
            description="Default CPU usage threshold for a worker to be considered high load",
        ),
    ] = EnvironmentDefaults.WORKER_HIGH_LOAD_CPU_USAGE

    WORKER_HIGH_LOAD_RECOVERY_TIME: Annotated[
        float,
        Field(
            description="Default time in seconds from the last time a worker was in high load before it is "
            "considered healthy again",
        ),
    ] = EnvironmentDefaults.WORKER_HIGH_LOAD_RECOVERY_TIME

    WORKER_STALE_TIME: Annotated[
        float,
        Field(
            description="Default time in seconds from the last time a worker reported any status before it is "
            "considered stale",
        ),
    ] = EnvironmentDefaults.WORKER_STALE_TIME

    WORKER_STATUS_SUMMARY_INTERVAL: Annotated[
        float,
        Field(
            description="Default interval in seconds between worker status summary messages",
        ),
    ] = EnvironmentDefaults.WORKER_STATUS_SUMMARY_INTERVAL

    ZMQ_CONTEXT_TERM_TIMEOUT: Annotated[
        float,
        Field(
            description="Default timeout for terminating the ZMQ context in seconds",
        ),
    ] = EnvironmentDefaults.ZMQ_CONTEXT_TERM_TIMEOUT


# Global singleton instance
Environment = _Environment()
