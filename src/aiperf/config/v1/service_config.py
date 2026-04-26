# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""v1 ServiceConfig - CLI-only service-runtime input DTO.

CLI flags that configure the AIPerf service runtime (logging, ZMQ comm, worker
counts, API host/port, UI type) live here. No validators - AIPerfConfig (or
the converter) owns the resolution logic for these knobs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from pydantic import Field

from aiperf.common.enums import AIPerfLogLevel
from aiperf.config._base import BaseConfig
from aiperf.config.cli_parameter import CLIParameter, DisableCLI, Groups
from aiperf.config.defaults import ServiceDefaults
from aiperf.plugin.enums import ServiceRunType, UIType

if TYPE_CHECKING:
    from aiperf.config._zmq_dual_bind import ZMQDualBindConfig
    from aiperf.config._zmq_ipc import ZMQIPCConfig
    from aiperf.config._zmq_tcp import ZMQTCPConfig
    from aiperf.config.v1._workers import WorkersConfig


class ServiceConfig(BaseConfig):
    """v1 service-runtime CLI input.

    CLI-only DTO. Validators are forbidden on this class. Forward-reference
    string annotations on ZMQ and worker nested classes let those concrete
    types swap in via subsequent v1 tasks (or be re-routed to existing v2
    classes by the converter).
    """

    _CLI_GROUP = Groups.SERVICE

    service_run_type: Annotated[
        ServiceRunType,
        Field(description="Type of service run (process, k8s)"),
        DisableCLI(reason="Only single support for now"),
    ] = ServiceDefaults.SERVICE_RUN_TYPE

    zmq_tcp: Annotated[
        ZMQTCPConfig | None,
        Field(default=None, description="ZMQ TCP configuration"),
    ] = None

    zmq_ipc: Annotated[
        ZMQIPCConfig | None,
        Field(default=None, description="ZMQ IPC configuration"),
    ] = None

    zmq_dual_bind: Annotated[
        ZMQDualBindConfig | None,
        Field(default=None, description="ZMQ dual-bind configuration"),
    ] = None

    workers: Annotated[
        WorkersConfig | None,
        Field(default=None, description="Worker configuration"),
    ] = None

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
            "`dashboard` shows real-time metrics in a Textual TUI, `simple` uses TQDM progress bars, "
            "`none` disables UI completely. Defaults to `dashboard` in interactive terminals, "
            "`none` when not a TTY (e.g., piped or redirected output). "
            "Automatically set to `simple` when using `--verbose` or `--extra-verbose` in a TTY.",
        ),
        CLIParameter(
            name=("--ui-type", "--ui"),
            group=_CLI_GROUP,
        ),
    ] = ServiceDefaults.UI_TYPE

    api_port: Annotated[
        int | None,
        Field(
            default=None,
            ge=1,
            le=65535,
            description="AIPerf API port (enables HTTP + WebSocket endpoints)",
        ),
        CLIParameter(
            name="--api-port",
            group=_CLI_GROUP,
        ),
    ] = None

    api_host: Annotated[
        str | None,
        Field(
            default=None,
            description="AIPerf API host (requires --api-port or AIPERF_API_SERVER_PORT to be set)",
        ),
        CLIParameter(
            name="--api-host",
            group=_CLI_GROUP,
        ),
    ] = None
