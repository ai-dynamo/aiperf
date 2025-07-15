# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Annotated, Any, Literal

import cyclopts
from pydantic import BeforeValidator, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from aiperf.common.config.config_defaults import ServiceDefaults
from aiperf.common.config.config_validators import parse_service_types
from aiperf.common.config.zmq_config import (
    BaseZMQCommunicationConfig,
    ZMQIPCConfig,
    ZMQTCPConfig,
)
from aiperf.common.enums import CommunicationBackend, ServiceRunType, ServiceType


class ServiceConfig(BaseSettings):
    """Base configuration for all services. It will be provided to all services during their __init__ function."""

    model_config = SettingsConfigDict(
        env_prefix="AIPERF_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow",
    )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # Initialize the comm_config if it is not provided, based on the comm_backend.
        if self.comm_config is None:
            if self.comm_backend == CommunicationBackend.ZMQ_IPC:
                self.comm_config = ZMQIPCConfig()
            elif self.comm_backend == CommunicationBackend.ZMQ_TCP:
                self.comm_config = ZMQTCPConfig()
            else:
                raise ValueError(f"Invalid communication backend: {self.comm_backend}")

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
        Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        Field(
            description="Logging level",
        ),
        cyclopts.Parameter(
            name=("--log-level"),
        ),
    ] = ServiceDefaults.LOG_LEVEL

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

    debug_services: Annotated[
        set[ServiceType] | None,
        Field(
            description="List of services to enable debug logging for. Can be a comma-separated list, a single service type, "
            "or the cli flag can be used multiple times.",
        ),
        cyclopts.Parameter(
            # Note that the name is singular because it can be used multiple times.
            name=("--debug-service"),
        ),
        BeforeValidator(parse_service_types),
    ] = ServiceDefaults.DEBUG_SERVICES
