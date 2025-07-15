# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Annotated, Literal

import cyclopts
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing_extensions import Self

from aiperf.common.config.base_config import ADD_TO_TEMPLATE
from aiperf.common.config.config_defaults import ServiceDefaults
from aiperf.common.config.zmq_config import (
    BaseZMQCommunicationConfig,
    ZMQIPCConfig,
    ZMQTCPConfig,
)
from aiperf.common.enums import CommunicationBackend, ServiceRunType


class ServiceConfig(BaseSettings):
    """Base configuration for all services. It will be provided to all services during their __init__ function."""

    model_config = SettingsConfigDict(
        env_prefix="AIPERF_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow",
    )

    @model_validator(mode="after")
    def validate_log_level_from_verbose_flags(self) -> Self:
        """Set log level based on verbose flags."""
        if self.extra_verbose:
            self.log_level = "TRACE"
        elif self.verbose:
            self.log_level = "DEBUG"
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
        cyclopts.Parameter(
            name=("--run-type"),
        ),
    ] = ServiceDefaults.SERVICE_RUN_TYPE

    comm_backend: Annotated[
        CommunicationBackend,
        Field(
            description="Communication backend to use",
        ),
        cyclopts.Parameter(
            name=("--comm-backend"),
        ),
    ] = ServiceDefaults.COMM_BACKEND

    comm_config: Annotated[
        BaseZMQCommunicationConfig | None,
        Field(
            description="Communication configuration",
        ),
        # TODO: Figure out if we need to be able to set this from the command line.
        # cyclopts.Parameter(
        #     name=("--comm-config"),
        # ),
    ] = ServiceDefaults.COMM_CONFIG

    heartbeat_timeout: Annotated[
        float,
        Field(
            description="Time in seconds after which a service is considered dead if no "
            "heartbeat received",
        ),
        cyclopts.Parameter(
            name=("--heartbeat-timeout"),
        ),
    ] = ServiceDefaults.HEARTBEAT_TIMEOUT

    registration_timeout: Annotated[
        float,
        Field(
            description="Time in seconds to wait for all required services to register",
        ),
        cyclopts.Parameter(
            name=("--registration-timeout"),
        ),
    ] = ServiceDefaults.REGISTRATION_TIMEOUT

    command_timeout: Annotated[
        float,
        Field(
            description="Default timeout for command responses",
        ),
        cyclopts.Parameter(
            name=("--command-timeout"),
        ),
    ] = ServiceDefaults.COMMAND_TIMEOUT

    heartbeat_interval: Annotated[
        float,
        Field(
            description="Interval in seconds between heartbeat messages",
        ),
        cyclopts.Parameter(
            name=("--heartbeat-interval"),
        ),
    ] = ServiceDefaults.HEARTBEAT_INTERVAL

    min_workers: Annotated[
        int | None,
        Field(
            description="Minimum number of workers to maintain",
        ),
        cyclopts.Parameter(
            name=("--min-workers"),
        ),
    ] = ServiceDefaults.MIN_WORKERS

    max_workers: Annotated[
        int | None,
        Field(
            description="Maximum number of workers to create. If not specified, the number of"
            " workers will be determined by the smaller of (concurrency + 1) and (num CPUs - 1).",
        ),
        cyclopts.Parameter(
            name=("--max-workers"),
        ),
    ] = ServiceDefaults.MAX_WORKERS

    log_level: Annotated[
        Literal[
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
            "CRITICAL",
            "TRACE",
            "NOTICE",
            "SUCCESS",
        ],
        Field(
            description="Logging level",
        ),
        cyclopts.Parameter(
            name=("--log-level"),
        ),
    ] = ServiceDefaults.LOG_LEVEL

    verbose: Annotated[
        bool,
        Field(
            description="Equivalent to --log-level DEBUG. Enables more verbose logging output, but lacks some raw message logging.",
            json_schema_extra={ADD_TO_TEMPLATE: False},
        ),
        cyclopts.Parameter(
            name=("--verbose", "-v"),
        ),
    ] = ServiceDefaults.VERBOSE

    extra_verbose: Annotated[
        bool,
        Field(
            description="Equivalent to --log-level TRACE. Enables the most verbose logging output possible.",
            json_schema_extra={ADD_TO_TEMPLATE: False},
        ),
        cyclopts.Parameter(
            name=("--extra-verbose", "-vv"),
        ),
    ] = ServiceDefaults.EXTRA_VERBOSE

    disable_ui: Annotated[
        bool,
        Field(
            description="Disable the UI",
        ),
        cyclopts.Parameter(
            name=("--disable-ui"),
        ),
    ] = ServiceDefaults.DISABLE_UI

    enable_uvloop: Annotated[
        bool,
        Field(
            description="Enable the use of uvloop instead of the default asyncio event loop",
        ),
        cyclopts.Parameter(
            name=("--enable-uvloop"),
        ),
    ] = ServiceDefaults.ENABLE_UVLOOP

    # TODO: Potentially auto-scale this in the future.
    result_parser_service_count: Annotated[
        int,
        Field(
            description="Number of services to spawn for parsing inference results. The higher the request rate, the more services should be spawned.",
        ),
        cyclopts.Parameter(
            name=("--result-parser-service-count"),
        ),
    ] = ServiceDefaults.RESULT_PARSER_SERVICE_COUNT
