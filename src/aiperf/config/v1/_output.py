# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""v1 OutputConfig - CLI-only input DTO for output settings.

No validators. AIPerfConfig is the sole validation gate.
"""

from pathlib import Path
from typing import Annotated

from pydantic import Field

from aiperf.common.enums import ExportLevel
from aiperf.config._base import BaseConfig
from aiperf.config.cli_parameter import CLIParameter, Groups
from aiperf.config.defaults import OutputDefaults


class OutputConfig(BaseConfig):
    """A configuration class for defining output related settings."""

    _CLI_GROUP = Groups.OUTPUT

    artifact_directory: Annotated[
        Path,
        Field(
            description="Output directory for all benchmark artifacts including metrics (`.csv`, `.json`, `.jsonl`), raw data (`_raw.jsonl`), "
            "GPU telemetry (`_gpu_telemetry.jsonl`), and time-sliced metrics (`_timeslices.csv/json`). Directory created if it doesn't exist. "
            "All output file paths are constructed relative to this directory.",
        ),
        CLIParameter(
            name=(
                "--output-artifact-dir",
                "--artifact-dir",  # GenAI-Perf
            ),
            group=_CLI_GROUP,
        ),
    ] = OutputDefaults.ARTIFACT_DIRECTORY

    profile_export_prefix: Annotated[
        Path | None,
        Field(
            description="Custom prefix for profile export file names. AIPerf generates multiple output files with different formats: "
            "`.csv` (summary metrics), `.json` (summary with metadata), `.jsonl` (per-record metrics), and `_raw.jsonl` (raw request/response data). "
            "If not specified, defaults to `profile_export_aiperf` for summary files and `profile_export` for detailed files.",
        ),
        CLIParameter(
            name=(
                "--profile-export-prefix",
                "--profile-export-file",  # GenAI-Perf
            ),
            group=_CLI_GROUP,
        ),
    ] = None

    export_level: Annotated[
        ExportLevel,
        Field(
            description="Controls which output files are generated. "
            "`summary`: Only aggregate metrics files (`.csv`, `.json`). "
            "`records`: Includes per-request metrics (`.jsonl`). "
            "`raw`: Includes raw request/response data (`_raw.jsonl`).",
        ),
        CLIParameter(
            name=("--export-level", "--profile-export-level"),
            group=_CLI_GROUP,
        ),
    ] = OutputDefaults.EXPORT_LEVEL

    slice_duration: Annotated[
        float | None,
        Field(
            description="Duration in seconds for time-sliced metric analysis. When set, AIPerf divides the benchmark timeline into fixed-length "
            "windows and computes metrics separately for each window. This enables analysis of performance trends and variations over time "
            "(e.g., warmup effects, degradation under sustained load).",
        ),
        CLIParameter(
            name=("--slice-duration"),
            group=_CLI_GROUP,
        ),
    ] = OutputDefaults.SLICE_DURATION

    export_http_trace: Annotated[
        bool,
        Field(
            description="Include HTTP trace data (timestamps, chunks, headers, socket info) in profile_export.jsonl. "
            "Computed metrics (http_req_duration, http_req_waiting, etc.) are always included regardless of this setting. "
            "See the HTTP Trace Metrics guide for details on trace data fields.",
        ),
        CLIParameter(
            name="--export-http-trace",
            group=_CLI_GROUP,
        ),
    ] = OutputDefaults.EXPORT_HTTP_TRACE

    export_per_chunk_data: Annotated[
        bool,
        Field(
            description=(
                "Include per-chunk list data (e.g., inter_chunk_latency arrays) in "
                "per-record exports. These arrays contain one timing value per SSE "
                "chunk and can be very large for long responses."
            ),
        ),
        CLIParameter(
            name=("--export-per-chunk-data",),
            group=_CLI_GROUP,
        ),
    ] = False

    show_trace_timing: Annotated[
        bool,
        Field(
            description="Display HTTP trace timing metrics in the console at the end of the benchmark. "
            "Shows detailed timing breakdown: blocked, DNS, connecting, sending, waiting (TTFB), receiving, "
            "and total duration following k6 naming conventions.",
        ),
        CLIParameter(
            name="--show-trace-timing",
            group=_CLI_GROUP,
        ),
    ] = OutputDefaults.SHOW_TRACE_TIMING
