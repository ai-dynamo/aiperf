# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Annotated

from cyclopts import Parameter
from pydantic import BeforeValidator, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing_extensions import Self

from aiperf.common.config.base_config import ADD_TO_TEMPLATE
from aiperf.common.config.config_defaults import ServiceDefaults
from aiperf.common.config.config_validators import parse_service_types
from aiperf.common.config.groups import Groups
from aiperf.common.config.worker_config import WorkersConfig
from aiperf.common.config.zmq_config import (
    BaseZMQCommunicationConfig,
    ZMQIPCConfig,
    ZMQTCPConfig,
)
from aiperf.common.enums import (
    AIPerfLogLevel,
    CommunicationBackend,
    ServiceRunType,
)
from aiperf.common.enums.service_enums import ServiceType


class ServiceConfig(BaseSettings):
    """Base configuration for all services. It will be provided to all services during their __init__ function."""

    model_config = SettingsConfigDict(
        env_prefix="AIPERF_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow",
    )

    _CLI_GROUP = Groups.SERVICE

    @model_validator(mode="after")
    def validate_log_level_from_verbose_flags(self) -> Self:
        """Set log level based on verbose flags."""
        if self.extra_verbose:
            self.log_level = AIPerfLogLevel.TRACE
        elif self.verbose:
            self.log_level = AIPerfLogLevel.DEBUG
        return self

    @model_validator(mode="after")
    def validate_comm_config(self) -> Self:
        """Initialize the comm_config if it is not provided, based on the comm_backend."""
        if self.comm_config is None:
            if self.comm_backend == CommunicationBackend.ZMQ_IPC:
                self.comm_config = ZMQIPCConfig()
            elif self.comm_backend == CommunicationBackend.ZMQ_TCP:
                self.comm_config = ZMQTCPConfig()
            else:
                raise ValueError(f"Invalid communication backend: {self.comm_backend}")
        return self

    service_run_type: Annotated[
        ServiceRunType,
        Field(
            description="Type of service run (process, k8s)",
        ),
        Parameter(
            name=("--service-run-type", "--run-type"),
            group=_CLI_GROUP,
        ),
    ] = ServiceDefaults.SERVICE_RUN_TYPE

    comm_backend: Annotated[
        CommunicationBackend,
        Field(
            description="Communication backend to use",
        ),
        Parameter(
            name=("--comm-backend"),
            group=_CLI_GROUP,
        ),
    ] = ServiceDefaults.COMM_BACKEND

    comm_config: Annotated[
        BaseZMQCommunicationConfig | None,
        Field(
            description="Communication configuration",
        ),
        Parameter(
            parse=False,  # This is not supported via CLI
        ),
    ] = ServiceDefaults.COMM_CONFIG

    heartbeat_timeout: Annotated[
        float,
        Field(
            description="Time in seconds after which a service is considered dead if no "
            "heartbeat received",
        ),
        Parameter(
            name=("--heartbeat-timeout"),
            group=_CLI_GROUP,
        ),
    ] = ServiceDefaults.HEARTBEAT_TIMEOUT

    registration_timeout: Annotated[
        float,
        Field(
            description="Time in seconds to wait for all required services to register",
        ),
        Parameter(
            name=("--registration-timeout"),
            group=_CLI_GROUP,
        ),
    ] = ServiceDefaults.REGISTRATION_TIMEOUT

    command_timeout: Annotated[
        float,
        Field(
            description="Default timeout for command responses",
        ),
        Parameter(
            name=("--command-timeout", "--command-timeout-seconds"),
            group=_CLI_GROUP,
        ),
    ] = ServiceDefaults.COMMAND_TIMEOUT

    heartbeat_interval_seconds: Annotated[
        float,
        Field(
            description="Interval in seconds between heartbeat messages",
        ),
        Parameter(
            name=("--heartbeat-interval-seconds", "--heartbeat-interval"),
            group=_CLI_GROUP,
        ),
    ] = ServiceDefaults.HEARTBEAT_INTERVAL_SECONDS

    workers: Annotated[
        WorkersConfig,
        Field(
            description="Worker configuration",
        ),
    ] = WorkersConfig()

    log_level: Annotated[
        AIPerfLogLevel,
        Field(
            description="Logging level",
        ),
        Parameter(
            name=("--log-level"),
            group=_CLI_GROUP,
        ),
    ] = ServiceDefaults.LOG_LEVEL

    verbose: Annotated[
        bool,
        Field(
            description="Equivalent to --log-level DEBUG. Enables more verbose logging output, but lacks some raw message logging.",
            json_schema_extra={ADD_TO_TEMPLATE: False},
        ),
        Parameter(
            name=("--verbose", "-v"),
            group=_CLI_GROUP,
        ),
    ] = ServiceDefaults.VERBOSE

    extra_verbose: Annotated[
        bool,
        Field(
            description="Equivalent to --log-level TRACE. Enables the most verbose logging output possible.",
            json_schema_extra={ADD_TO_TEMPLATE: False},
        ),
        Parameter(
            name=("--extra-verbose", "-vv"),
            group=_CLI_GROUP,
        ),
    ] = ServiceDefaults.EXTRA_VERBOSE

    disable_ui: Annotated[
        bool,
        Field(
            description="Disable the UI (prints progress to the console as log messages). This is equivalent to --ui-type none.",
        ),
        Parameter(
            name=("--disable-ui"),
            group=_CLI_GROUP,
        ),
    ] = ServiceDefaults.DISABLE_UI

    enable_uvloop: Annotated[
        bool,
        Field(
            description="Enable the use of uvloop instead of the default asyncio event loop",
        ),
        Parameter(
            name=("--enable-uvloop"),
            group=_CLI_GROUP,
        ),
    ] = ServiceDefaults.ENABLE_UVLOOP

    # TODO: Potentially auto-scale this in the future.
    record_processor_service_count: Annotated[
        int | None,
        Field(
            ge=1,
            description="Number of services to spawn for processing records. The higher the request rate, the more services "
            "should be spawned in order to keep up with the incoming records. If not specified, the number of services will be "
            "automatically determined based on the worker count.",
        ),
        Parameter(
            name=("--record-processor-service-count", "--record-processors"),
            group=_CLI_GROUP,
        ),
    ] = ServiceDefaults.RECORD_PROCESSOR_SERVICE_COUNT

    progress_report_interval: Annotated[
        float,
        Field(
            description="Interval in seconds to report progress. This is used to report the progress of the profile to the user.",
        ),
        Parameter(
            name=("--progress-report-interval-seconds", "--progress-report-interval"),
            group=_CLI_GROUP,
        ),
    ] = ServiceDefaults.PROGRESS_REPORT_INTERVAL

    enable_yappi: Annotated[
        bool,
        Field(
            description="*[Developer use only]* Enable yappi profiling (Yet Another Python Profiler) to profile AIPerf's internal python code. "
            "This can be used in the development of AIPerf in order to find performance bottlenecks across the various services. "
            "The output '.prof' files can be viewed with snakeviz. Requires yappi and snakeviz to be installed. "
            "Run 'pip install yappi snakeviz' to install them.",
        ),
        Parameter(
            name=("--enable-yappi-profiling"),
            group=_CLI_GROUP,
        ),
    ] = ServiceDefaults.ENABLE_YAPPI

    debug_services: Annotated[
        set[ServiceType] | None,
        Field(
            description="List of services to enable debug logging for. Can be a comma-separated list, a single service type, "
            "or the cli flag can be used multiple times.",
        ),
        Parameter(
            name=("--debug-service", "--debug-services"),
            group=_CLI_GROUP,
        ),
        BeforeValidator(parse_service_types),
    ] = ServiceDefaults.DEBUG_SERVICES

    trace_services: Annotated[
        set[ServiceType] | None,
        Field(
            description="List of services to enable trace logging for. Can be a comma-separated list, a single service type, "
            "or the cli flag can be used multiple times.",
        ),
        Parameter(
            name=("--trace-service", "--trace-services"),
            group=_CLI_GROUP,
        ),
        BeforeValidator(parse_service_types),
    ] = ServiceDefaults.TRACE_SERVICES
