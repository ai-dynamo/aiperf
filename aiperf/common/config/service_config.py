# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Annotated

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing_extensions import Self

from aiperf.common.config.base_config import ADD_TO_TEMPLATE
from aiperf.common.config.cli_parameter import CLIParameter, DisableCLI
from aiperf.common.config.config_defaults import ServiceDefaults
from aiperf.common.config.dev_config import DeveloperConfig
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
from aiperf.common.enums.ui_enums import AIPerfUIType


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
        DisableCLI(reason="Only single support for now"),
    ] = ServiceDefaults.SERVICE_RUN_TYPE

    comm_backend: Annotated[
        CommunicationBackend,
        Field(
            description="Communication backend to use",
        ),
        DisableCLI(reason="This is not supported via CLI"),
    ] = ServiceDefaults.COMM_BACKEND

    comm_config: Annotated[
        BaseZMQCommunicationConfig | None,
        Field(
            description="Communication configuration",
        ),
        DisableCLI(reason="This is not supported via CLI"),
    ] = ServiceDefaults.COMM_CONFIG

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
        CLIParameter(
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
        CLIParameter(
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
        CLIParameter(
            name=("--extra-verbose", "-vv"),
            group=_CLI_GROUP,
        ),
    ] = ServiceDefaults.EXTRA_VERBOSE

    record_processor_service_count: Annotated[
        int | None,
        Field(
            ge=1,
            description="Number of services to spawn for processing records. The higher the request rate, the more services "
            "should be spawned in order to keep up with the incoming records. If not specified, the number of services will be "
            "automatically determined based on the worker count.",
        ),
        CLIParameter(
            name=("--record-processor-service-count", "--record-processors"),
            group=_CLI_GROUP,
        ),
    ] = ServiceDefaults.RECORD_PROCESSOR_SERVICE_COUNT

    ui_type: Annotated[
        AIPerfUIType,
        Field(
            description="Type of UI to use",
        ),
        CLIParameter(
            name=("--ui-type", "--ui"),
            group=_CLI_GROUP,
        ),
    ] = ServiceDefaults.UI_TYPE

    developer: DeveloperConfig = DeveloperConfig()
