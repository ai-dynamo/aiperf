# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""v1 UserConfig - CLI-only input DTO.

This module defines the cyclopts-facing input shape. It carries CLI flag
metadata (CLIParameter), field-level documentation (Field), and CLI-input
parsing helpers (BeforeValidator(parse_str_or_list)) - but NO model-level
or field-level domain validators.

Domain validation (e.g. "concurrency cannot exceed request_count") lives on
AIPerfConfig. The converter at aiperf.config.v1.converter translates a
populated UserConfig into the canonical AIPerfConfig.

See aiperf.config.v1.__init__ for the hard rules around adding new fields.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

from pydantic import BeforeValidator, Field

from aiperf.common.enums import GPUTelemetryMode, ServerMetricsFormat
from aiperf.config._base import BaseConfig
from aiperf.config.cli_parameter import CLIParameter, DisableCLI, Groups
from aiperf.config.parsing import parse_str_or_list
from aiperf.plugin.enums import GPUTelemetryCollectorType

# Default server-metrics export formats. Mirrors origin/main's
# ServerMetricsDefaults.DEFAULT_FORMATS, inlined here because that constant was
# removed from aiperf.config.defaults during the v2 refactor.
_DEFAULT_SERVER_METRICS_FORMATS: list[ServerMetricsFormat] = [
    ServerMetricsFormat.JSON,
    ServerMetricsFormat.CSV,
    ServerMetricsFormat.PARQUET,
]

if TYPE_CHECKING:
    from aiperf.config.v1._accuracy import AccuracyConfig
    from aiperf.config.v1._endpoint import EndpointConfig
    from aiperf.config.v1._input import InputConfig
    from aiperf.config.v1._loadgen import LoadGeneratorConfig
    from aiperf.config.v1._output import OutputConfig
    from aiperf.config.v1._tokenizer import TokenizerConfig


class UserConfig(BaseConfig):
    """v1 user-facing CLI input.

    CLI-only DTO. Validators are forbidden on this class - AIPerfConfig is the
    single validation gate. Forward-reference string annotations on nested
    classes let the v1 nested-class files land independently in later tasks.
    """

    endpoint: Annotated[
        EndpointConfig | None,
        Field(default=None, description="Endpoint configuration"),
    ] = None

    input: Annotated[
        InputConfig | None,
        Field(default=None, description="Input configuration"),
    ] = None

    output: Annotated[
        OutputConfig | None,
        Field(default=None, description="Output configuration"),
    ] = None

    tokenizer: Annotated[
        TokenizerConfig | None,
        Field(default=None, description="Tokenizer configuration"),
    ] = None

    loadgen: Annotated[
        LoadGeneratorConfig | None,
        Field(default=None, description="Load Generator configuration"),
    ] = None

    accuracy: Annotated[
        AccuracyConfig | None,
        Field(default=None, description="Accuracy benchmarking configuration"),
    ] = None

    cli_command: Annotated[
        str | None,
        Field(
            default=None,
            description="The CLI command for the user config.",
        ),
        DisableCLI(reason="This is automatically set by the CLI"),
    ] = None

    benchmark_id: Annotated[
        str | None,
        Field(
            default=None,
            description="Unique identifier for this benchmark run (UUID). Generated automatically and shared across all export formats for correlation.",
        ),
        DisableCLI(reason="This is automatically generated at runtime"),
    ] = None

    gpu_telemetry: Annotated[
        list[str] | None,
        Field(
            description=(
                "Enable GPU telemetry console display and optionally specify: "
                "(1) 'pynvml' to use local pynvml library instead of DCGM HTTP endpoints, "
                "(2) 'dashboard' for realtime dashboard mode, "
                "(3) custom DCGM exporter URLs (e.g., http://node1:9401/metrics), "
                "(4) custom metrics CSV file (e.g., custom_gpu_metrics.csv). "
                "Default: DCGM mode with localhost:9400 and localhost:9401 endpoints. "
                "Examples: --gpu-telemetry pynvml | --gpu-telemetry dashboard node1:9400"
            ),
        ),
        BeforeValidator(parse_str_or_list),
        CLIParameter(
            name=("--gpu-telemetry",),
            consume_multiple=True,
            group=Groups.GPU_TELEMETRY,
        ),
    ] = None

    no_gpu_telemetry: Annotated[
        bool,
        Field(
            description="Disable GPU telemetry collection entirely.",
        ),
        CLIParameter(
            name=("--no-gpu-telemetry",),
            group=Groups.GPU_TELEMETRY,
        ),
    ] = False

    # Internal computed fields populated by the converter, not by validators.
    # Kept as PrivateAttr-style underscore fields so the converter can stash the
    # parsed --gpu-telemetry breakdown (mode/collector/URLs/metrics file) for
    # downstream readers without re-parsing the raw list.
    _gpu_telemetry_mode: GPUTelemetryMode = GPUTelemetryMode.SUMMARY
    _gpu_telemetry_collector_type: GPUTelemetryCollectorType = (
        GPUTelemetryCollectorType.DCGM
    )
    _gpu_telemetry_urls: list[str] = []
    _gpu_telemetry_metrics_file: Path | None = None

    server_metrics: Annotated[
        list[str] | None,
        Field(
            description=(
                "Server metrics collection (ENABLED BY DEFAULT). "
                "Automatically collects from inference endpoint base_url + `/metrics`. "
                "Optionally specify additional custom Prometheus-compatible endpoint URLs "
                "(e.g., http://node1:8081/metrics, http://node2:9090/metrics). "
                "Use `--no-server-metrics` to disable collection. "
                "Example: `--server-metrics node1:8081 node2:9090/metrics` for additional endpoints"
            ),
        ),
        BeforeValidator(parse_str_or_list),
        CLIParameter(
            name=("--server-metrics",),
            consume_multiple=True,
            group=Groups.SERVER_METRICS,
        ),
    ] = None

    no_server_metrics: Annotated[
        bool,
        Field(
            description="Disable server metrics collection entirely.",
        ),
        CLIParameter(
            name=("--no-server-metrics",),
            group=Groups.SERVER_METRICS,
        ),
    ] = False

    server_metrics_formats: Annotated[
        list[ServerMetricsFormat],
        Field(
            description=(
                "Specify which output formats to generate for server metrics. "
                "Multiple formats can be specified (e.g., `--server-metrics-formats json csv parquet`)."
            ),
        ),
        BeforeValidator(parse_str_or_list),
        CLIParameter(
            name=("--server-metrics-formats",),
            consume_multiple=True,
            group=Groups.SERVER_METRICS,
        ),
    ] = _DEFAULT_SERVER_METRICS_FORMATS

    _server_metrics_urls: list[str] = []

    config_file: Annotated[
        Path | None,
        Field(
            default=None,
            description=(
                "Path to a YAML configuration file. "
                "CLI flags override values from the config file."
            ),
        ),
        CLIParameter(
            name=("--config", "-f"),
            group=Groups.INPUT,
        ),
    ] = None
