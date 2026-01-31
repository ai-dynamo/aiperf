# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Annotated

from pydantic import Field, model_validator
from typing_extensions import Self

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.config.base_config import ADD_TO_TEMPLATE, BaseConfig
from aiperf.common.config.cli_parameter import CLIParameter, DisableCLI
from aiperf.common.config.config_defaults import ServiceDefaults
from aiperf.common.config.groups import Groups
from aiperf.common.config.worker_config import WorkersConfig
from aiperf.common.config.zmq_config import (
    BaseZMQCommunicationConfig,
    ZMQIPCConfig,
    ZMQTCPConfig,
)
from aiperf.common.enums import AIPerfLogLevel
from aiperf.plugin.enums import ServiceRunType, UIType

_logger = AIPerfLogger(__name__)


class ServiceConfig(BaseConfig):
    """Base configuration for all services. It will be provided to all services during their __init__ function."""

    _CLI_GROUP = Groups.SERVICE
    _comm_config: BaseZMQCommunicationConfig | None = None

    @model_validator(mode="after")
    def validate_log_level_from_verbose_flags(self) -> Self:
        """Set log level based on verbose flags."""
        if self.extra_verbose:
            self.log_level = AIPerfLogLevel.TRACE
        elif self.verbose:
            self.log_level = AIPerfLogLevel.DEBUG
        return self

    @model_validator(mode="after")
    def validate_ui_type_from_verbose_flags(self) -> Self:
        """Set UI type based on verbose flags."""
        # If the user has explicitly set the UI type, use that.
        if "ui_type" in self.model_fields_set:
            return self

        # If the user selected verbose or extra verbose flags, set the UI type to simple.
        # This will allow the user to see the verbose output in the console easier.
        if self.verbose or self.extra_verbose:
            self.ui_type = UIType.SIMPLE
        return self

    @model_validator(mode="after")
    def validate_comm_config(self) -> Self:
        """Initialize the comm_config based on the zmq_tcp or zmq_ipc config."""
        _logger.debug(
            f"Validating comm_config: tcp: {self.zmq_tcp}, ipc: {self.zmq_ipc}"
        )
        if self.zmq_tcp is not None and self.zmq_ipc is not None:
            raise ValueError(
                "Cannot use both ZMQ TCP and ZMQ IPC configuration at the same time"
            )
        elif self.zmq_tcp is not None:
            _logger.info("Using ZMQ TCP configuration")
            self._comm_config = self.zmq_tcp
        elif self.zmq_ipc is not None:
            _logger.info("Using ZMQ IPC configuration")
            self._comm_config = self.zmq_ipc
        else:
            _logger.info("Using default ZMQ IPC configuration")
            self._comm_config = ZMQIPCConfig()
        return self

    service_run_type: Annotated[
        ServiceRunType,
        Field(
            description="Type of service run (process, k8s)",
        ),
        DisableCLI(reason="Only single support for now"),
    ] = ServiceDefaults.SERVICE_RUN_TYPE

    zmq_tcp: Annotated[
        ZMQTCPConfig | None,
        Field(
            description="ZMQ TCP configuration",
        ),
    ] = None

    zmq_ipc: Annotated[
        ZMQIPCConfig | None,
        Field(
            description="ZMQ IPC configuration",
        ),
    ] = None

    workers: Annotated[
        WorkersConfig,
        Field(
            description="Worker configuration",
        ),
    ] = WorkersConfig()

    log_level: Annotated[
        AIPerfLogLevel,
        Field(
            description="Set the logging verbosity level. Controls the amount of output displayed during benchmark execution. "
            "Use `TRACE` for debugging ZMQ messages, `DEBUG` for detailed operation logs, or `INFO` (default) for standard progress updates.",
        ),
        CLIParameter(
            name=("--log-level"),
            group=_CLI_GROUP,
        ),
    ] = ServiceDefaults.LOG_LEVEL

    verbose: Annotated[
        bool,
        Field(
            description="Equivalent to `--log-level DEBUG`. Enables detailed logging output showing function calls and state transitions. "
            "Also automatically switches UI to `simple` mode for better console visibility. Does not include raw ZMQ message logging.",
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
            description="Equivalent to `--log-level TRACE`. Enables the most verbose logging possible, including all ZMQ messages, "
            "internal state changes, and low-level operations. Also switches UI to `simple` mode. Use for deep debugging.",
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
            description="Number of `RecordProcessor` services to spawn for parallel metric computation. "
            "Higher request rates require more processors to keep up with incoming records. "
            "If not specified, automatically determined based on worker count (typically 1-2 processors per 8 workers).",
        ),
        CLIParameter(
            name=("--record-processor-service-count", "--record-processors"),
            group=_CLI_GROUP,
        ),
    ] = ServiceDefaults.RECORD_PROCESSOR_SERVICE_COUNT

    ui_type: Annotated[
        UIType,
        Field(
            description="Select the user interface type for displaying benchmark progress. "
            "`dashboard` (default) shows real-time metrics in a Textual TUI, `simple` uses TQDM progress bars, "
            "`none` disables UI completely. Automatically set to `simple` when using `--verbose` or `--extra-verbose`.",
        ),
        CLIParameter(
            name=("--ui-type", "--ui"),
            group=_CLI_GROUP,
        ),
    ] = ServiceDefaults.UI_TYPE

    @property
    def comm_config(self) -> BaseZMQCommunicationConfig:
        """Get the communication configuration."""
        if not self._comm_config:
            raise ValueError(
                "Communication configuration is not set. Please provide a valid configuration."
            )
        return self._comm_config
