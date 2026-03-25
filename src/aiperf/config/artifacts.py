# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
AIPerf Configuration v2.0 - Pydantic Models

Artifacts - Export and output settings for benchmark results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    BeforeValidator,
    ConfigDict,
    Field,
    PrivateAttr,
    model_validator,
)

from aiperf.common.enums import (
    ExportLevel,
    GPUTelemetryMode,
    ServerMetricsDiscoveryMode,
    ServerMetricsFormat,
)
from aiperf.config._base import BaseConfig
from aiperf.config.phases import _normalize_duration

__all__ = [
    "ArtifactsConfig",
    "ServerMetricsConfig",
    "ServerMetricsDiscoveryConfig",
    "GpuTelemetryConfig",
]

# Type aliases for format arrays
SummaryExportFormat = Literal["json", "yaml"]
RecordsExportFormat = Literal["jsonl", "csv"]


class ArtifactsConfig(BaseConfig):
    """
    Artifacts configuration for benchmark output.

    Controls where and how benchmark results are exported.
    Uses flat structure with format arrays instead of nested export configs.
    """

    model_config = ConfigDict(extra="forbid", validate_default=True)

    dir: Annotated[
        Path,
        Field(
            default=Path("./artifacts"),
            description="Output directory for all benchmark artifacts. "
            "Created if it doesn't exist.",
        ),
    ]

    prefix: Annotated[
        str,
        Field(
            default="aiperf",
            description="Filename prefix for all exported files. "
            "Example: 'my_run' produces 'my_run_summary.json', 'my_run_records.jsonl'.",
        ),
    ]

    summary: Annotated[
        list[SummaryExportFormat] | Literal[False],
        Field(
            default_factory=lambda: ["json"],
            description="Summary export formats. "
            "Options: json, yaml. Set to false to disable.",
        ),
    ]

    records: Annotated[
        list[RecordsExportFormat] | Literal[False],
        Field(
            default=False,
            description="Per-request records export formats. "
            "Options: jsonl, csv. Set to false to disable.",
        ),
    ]

    raw: Annotated[
        bool,
        Field(
            default=False,
            description="Export raw request/response payloads as JSONL.",
        ),
    ]

    trace: Annotated[
        bool,
        Field(
            default=False,
            description="Export HTTP trace data for debugging.",
        ),
    ]

    per_chunk_data: Annotated[
        bool,
        Field(
            default=False,
            description="Include per-chunk list data (e.g., inter_chunk_latency arrays) "
            "in per-record exports. These arrays contain one timing value per SSE "
            "chunk and can be very large for long responses.",
        ),
    ]

    slice_duration: Annotated[
        float | None,
        BeforeValidator(_normalize_duration),
        Field(
            default=None,
            description="Time slice duration in seconds for trend analysis (must be > 0). "
            "Divides benchmark into windows for per-window statistics. "
            "Supports: 30, '30s', '5m', '2h'.",
        ),
    ]

    show_trace_timing: Annotated[
        bool,
        Field(
            default=False,
            description="Display HTTP trace timing metrics in console output. "
            "Shows detailed timing breakdown: blocked, DNS, connecting, sending, "
            "waiting (TTFB), receiving, and total duration.",
        ),
    ]

    cli_command: Annotated[
        str | None,
        Field(
            default=None,
            description="CLI command used to run the benchmark. "
            "Populated automatically when running via CLI.",
        ),
    ]

    benchmark_id: Annotated[
        str,
        Field(
            default_factory=lambda: __import__("uuid").uuid4().hex,
            description="Unique identifier for this benchmark run. "
            "Used to correlate artifacts across export formats. "
            "Auto-generated if not provided.",
        ),
    ]

    @model_validator(mode="after")
    def validate_artifacts(self) -> ArtifactsConfig:
        """Validate artifact configuration."""
        if isinstance(self.summary, list) and len(self.summary) == 0:
            raise ValueError(
                "summary format list cannot be empty; use false to disable"
            )
        if isinstance(self.records, list) and len(self.records) == 0:
            raise ValueError(
                "records format list cannot be empty; use false to disable"
            )
        if self.slice_duration is not None and self.slice_duration <= 0:
            raise ValueError("slice_duration must be > 0")
        return self

    # ==========================================================================
    # COMPUTED FILE PATH PROPERTIES
    # ==========================================================================

    @property
    def profile_export_csv_file(self) -> Path:
        """Get the path for the CSV summary export file."""
        return self.dir / f"profile_export_{self.prefix}.csv"

    @property
    def profile_export_json_file(self) -> Path:
        """Get the path for the JSON summary export file."""
        return self.dir / f"profile_export_{self.prefix}.json"

    @property
    def checkpoints_dir(self) -> Path:
        """Get the directory used for partial recovery checkpoints."""
        return self.dir / "checkpoints"

    @property
    def profile_export_partial_json_file(self) -> Path:
        """Get the path for the latest partial checkpoint JSON export."""
        return self.checkpoints_dir / f"profile_export_{self.prefix}_partial.json"

    @property
    def profile_export_timeslices_csv_file(self) -> Path:
        """Get the path for the timeslices CSV export file."""
        return self.dir / f"profile_export_{self.prefix}_timeslices.csv"

    @property
    def profile_export_timeslices_json_file(self) -> Path:
        """Get the path for the timeslices JSON export file."""
        return self.dir / f"profile_export_{self.prefix}_timeslices.json"

    @property
    def profile_export_records_csv_file(self) -> Path:
        """Get the path for the per-record CSV export file."""
        return self.dir / "profile_export_records.csv"

    @property
    def profile_export_jsonl_file(self) -> Path:
        """Get the path for the per-record JSONL export file."""
        return self.dir / "profile_export.jsonl"

    @property
    def profile_export_raw_jsonl_file(self) -> Path:
        """Get the path for the raw request/response JSONL export file."""
        return self.dir / "profile_export_raw.jsonl"

    @property
    def profile_export_console_txt_file(self) -> Path:
        """Get the path for the plain-text console export file."""
        return self.dir / "profile_export_console.txt"

    @property
    def profile_export_console_ansi_file(self) -> Path:
        """Get the path for the ANSI-styled console export file."""
        return self.dir / "profile_export_console.ansi"

    @property
    def profile_export_gpu_telemetry_jsonl_file(self) -> Path:
        """Get the path for the GPU telemetry JSONL export file."""
        return self.dir / "gpu_telemetry_export.jsonl"

    @property
    def server_metrics_export_jsonl_file(self) -> Path:
        """Get the path for the server metrics JSONL export file."""
        return self.dir / "server_metrics_export.jsonl"

    @property
    def server_metrics_export_json_file(self) -> Path:
        """Get the path for the server metrics JSON export file."""
        return self.dir / "server_metrics_export.json"

    @property
    def server_metrics_export_csv_file(self) -> Path:
        """Get the path for the server metrics CSV export file."""
        return self.dir / "server_metrics_export.csv"

    @property
    def server_metrics_export_parquet_file(self) -> Path:
        """Get the path for the server metrics Parquet export file."""
        return self.dir / "server_metrics_export.parquet"

    @property
    def export_level(self) -> ExportLevel:
        """Derive ExportLevel from the raw/records fields.

        Backward compatibility for code that checks config.output.export_level.
        """
        if self.raw:
            return ExportLevel.RAW
        if isinstance(self.records, list):
            return ExportLevel.RECORDS
        return ExportLevel.SUMMARY

    @property
    def artifact_directory(self) -> Path:
        """Alias for dir for backward compatibility."""
        return self.dir


class ServerMetricsDiscoveryConfig(BaseConfig):
    """Kubernetes-based auto-discovery of Prometheus /metrics endpoints.

    When mode is 'auto' or 'kubernetes', queries the K8s API for pods with:
    1. Dynamo label: nvidia.com/metrics-enabled=true
    2. Prometheus annotation: prometheus.io/scrape=true
    3. User-provided label_selector (server-side filter)

    Prometheus annotations (prometheus.io/port, prometheus.io/path,
    prometheus.io/scheme) control the constructed scrape URL when present.
    """

    model_config = ConfigDict(extra="forbid", validate_default=True)

    mode: Annotated[
        ServerMetricsDiscoveryMode,
        Field(
            default=ServerMetricsDiscoveryMode.AUTO,
            description="Discovery mode: 'auto' detects environment and tries K8s "
            "if in-cluster, 'kubernetes' forces K8s API discovery, "
            "'disabled' uses only explicit URLs.",
        ),
    ]

    label_selector: Annotated[
        str | None,
        Field(
            default=None,
            description="Kubernetes label selector for discovery. "
            "Example: 'app=vllm,env=prod'. Applied in addition to "
            "built-in Dynamo and Prometheus discovery.",
        ),
    ]

    namespace: Annotated[
        str | None,
        Field(
            default=None,
            description="Kubernetes namespace to search. "
            "If not specified, searches all namespaces.",
        ),
    ]

    @model_validator(mode="after")
    def validate_discovery_options(self) -> ServerMetricsDiscoveryConfig:
        """Validate that K8s-specific options aren't set when discovery is disabled."""
        if self.mode == ServerMetricsDiscoveryMode.DISABLED:
            k8s_options = []
            if self.label_selector is not None:
                k8s_options.append("label_selector")
            if self.namespace is not None:
                k8s_options.append("namespace")
            if k8s_options:
                msg = (
                    f"{', '.join(k8s_options)} can only be used when "
                    "discovery mode is 'auto' or 'kubernetes'."
                )
                raise ValueError(msg)
        return self


class ServerMetricsConfig(BaseConfig):
    """
    Server metrics configuration for Prometheus scraping.

    Collects server-side operational metrics (queue depth, KV cache utilization,
    batch sizes, GPU memory) from Prometheus endpoints exposed by inference servers
    like vLLM, TensorRT-LLM, or Triton.

    Accepts shorthand forms:
        - String URL: "http://localhost:9090/metrics"
          → ServerMetricsConfig(enabled=True, urls=["http://localhost:9090/metrics"])
        - Singular url field: {url: "..."}
          → ServerMetricsConfig(urls=["..."])
    """

    model_config = ConfigDict(extra="forbid", validate_default=True)

    enabled: Annotated[
        bool,
        Field(
            default=True,
            description="Enable Prometheus metrics scraping. Set to false to disable.",
        ),
    ]

    urls: Annotated[
        list[str],
        Field(
            default_factory=list,
            description="Prometheus metrics endpoint URLs to scrape. "
            "Typically the /metrics endpoint on inference servers.",
        ),
    ]

    formats: Annotated[
        list[ServerMetricsFormat],
        Field(
            default_factory=lambda: [
                ServerMetricsFormat.JSON,
                ServerMetricsFormat.CSV,
                ServerMetricsFormat.PARQUET,
            ],
            description="Export formats for scraped metrics. "
            "Options: json, csv, parquet, jsonl.",
        ),
    ]

    discovery: Annotated[
        ServerMetricsDiscoveryConfig,
        Field(
            default_factory=ServerMetricsDiscoveryConfig,
            description="Auto-discovery of Prometheus endpoints in Kubernetes. "
            "Discovers pods via Dynamo labels, Prometheus annotations, "
            "or custom label selectors.",
        ),
    ]

    @model_validator(mode="before")
    @classmethod
    def normalize_before_validation(cls, data: Any) -> Any:
        """Normalize shorthand forms before validation.

        Handles:
            - String URL → full config dict with that URL
            - url → urls (singular to plural)
        """
        # String URL → full config with that URL
        if isinstance(data, str):
            return {"enabled": True, "urls": [data]}

        if not isinstance(data, dict):
            return data

        # url → urls (singular to plural)
        if "url" in data and "urls" not in data:
            url = data.pop("url")
            data["urls"] = [url] if isinstance(url, str) else url

        return data


class GpuTelemetryConfig(BaseConfig):
    """
    GPU telemetry configuration for DCGM metrics collection.

    Collects GPU metrics (power, utilization, temperature, memory usage)
    from NVIDIA DCGM exporter endpoints.

    Accepts shorthand forms:
        - String URL: "http://localhost:9400/metrics"
          → GpuTelemetryConfig(enabled=True, urls=["http://localhost:9400/metrics"])
        - Singular url field: {url: "..."}
          → GpuTelemetryConfig(urls=["..."])
    """

    model_config = ConfigDict(extra="forbid", validate_default=True)

    enabled: Annotated[
        bool,
        Field(
            default=True,
            description="Enable GPU telemetry collection. Set to false to disable.",
        ),
    ]

    urls: Annotated[
        list[str],
        Field(
            default_factory=list,
            description="DCGM exporter endpoint URLs. "
            "Example: http://localhost:9400/metrics",
        ),
    ]

    metrics_file: Annotated[
        Path | None,
        Field(
            default=None,
            description="Path to CSV file with pre-recorded GPU metrics. "
            "Alternative to live DCGM collection.",
        ),
    ]

    # Private attribute for mutable telemetry mode (used by dashboard for dynamic switching)
    _mode: GPUTelemetryMode = PrivateAttr(default=GPUTelemetryMode.SUMMARY)

    @model_validator(mode="before")
    @classmethod
    def normalize_before_validation(cls, data: Any) -> Any:
        """Normalize shorthand forms before validation.

        Handles:
            - String URL → full config dict with that URL
            - url → urls (singular to plural)
        """
        # String URL → full config with that URL
        if isinstance(data, str):
            return {"enabled": True, "urls": [data]}

        if not isinstance(data, dict):
            return data

        # url → urls (singular to plural)
        if "url" in data and "urls" not in data:
            url = data.pop("url")
            data["urls"] = [url] if isinstance(url, str) else url

        return data

    @property
    def mode(self) -> GPUTelemetryMode:
        """Get the GPU telemetry display mode."""
        return self._mode

    @mode.setter
    def mode(self, value: GPUTelemetryMode) -> None:
        """Set the GPU telemetry display mode."""
        self._mode = value
