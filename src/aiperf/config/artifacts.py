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
from aiperf.config.steady_state import SteadyStateConfig
from aiperf.config.user_files import UserFile
from aiperf.plugin.enums import GPUTelemetryCollectorType

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
            default_factory=lambda: ["jsonl"],
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
            description="CLI command used to run the benchmark, recorded in artifacts "
            "for reproducibility. [auto-populated by the CLI runner; do not set in a "
            "CR spec — any user value is overwritten.]",
        ),
    ]

    benchmark_id: Annotated[
        str,
        Field(
            default_factory=lambda: __import__("uuid").uuid4().hex,
            description="Unique identifier for this benchmark run, used to correlate "
            "artifacts across export formats. [auto-generated; do not set in a CR spec "
            "unless you have a specific reason to override the UUID.]",
        ),
    ]

    user_files: Annotated[
        list[UserFile],
        Field(
            default_factory=list,
            description="User-defined templated files materialized into the run directory "
            "before the benchmark begins. See docs/kubernetes/user-files.md.",
        ),
    ]

    steady_state: Annotated[
        SteadyStateConfig,
        Field(
            default_factory=SteadyStateConfig,
            description="Steady-state detection and windowed metric computation. "
            "When enabled, AIPerf detects the steady-state region of a benchmark run "
            "and reports windowed metrics that exclude ramp-up and ramp-down periods.",
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
    def profile_export_steady_state_csv_file(self) -> Path:
        """Get the path for the steady-state windowed metrics CSV file."""
        return self.dir / f"profile_export_{self.prefix}_steady_state.csv"

    @property
    def profile_export_steady_state_json_file(self) -> Path:
        """Get the path for the steady-state windowed metrics JSON file."""
        return self.dir / f"profile_export_{self.prefix}_steady_state.json"

    @property
    def profile_export_energy_efficiency_json_file(self) -> Path:
        """Get the path for the energy efficiency metrics JSON file."""
        return self.dir / f"profile_export_{self.prefix}_energy_efficiency.json"

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
    """Kubernetes-based auto-discovery of inference-server /metrics endpoints.

    When mode is 'auto' or 'kubernetes', queries the K8s API for pods that
    are recognizable inference servers (vLLM, SGLang, Triton Inference Server,
    TensorRT-LLM, NVIDIA Dynamo). Eligibility (any one is enough):
    1. Dynamo opt-in label: nvidia.com/metrics-enabled=true
    2. AIPerf opt-in annotation: aiperf.nvidia.com/metrics-paths=...
    3. A container image matching a known inference-server signature
    4. User-provided label_selector (server-side filter)

    The broad ``prometheus.io/scrape=true`` annotation is intentionally NOT a
    trigger: Loki, Grafana, kube-state-metrics, and many platform components
    set it without being inference servers. ``prometheus.io/{port,path,scheme}``
    are still honored to construct the scrape URL when an eligible pod sets them.
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

    # x-kubernetes-preserve-unknown-fields lets apiserver accept the
    # string-URL shorthand (collapsed to the full object form by
    # normalize_before_validation) which a Kubernetes structural schema
    # cannot express as a string|object union.
    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
        json_schema_extra={"x-kubernetes-preserve-unknown-fields": True},
    )

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
    GPU telemetry configuration for live or replayed GPU metrics collection.

    Collects GPU metrics through DCGM exporter endpoints by default; the
    collector field can switch collection to local PyNVML, and mode controls
    summary vs. realtime dashboard display.

    Accepts shorthand forms:
        - String URL: "http://localhost:9400/metrics"
          → GpuTelemetryConfig(enabled=True, urls=["http://localhost:9400/metrics"])
        - Singular url field: {url: "..."}
          → GpuTelemetryConfig(urls=["..."])
    """

    # x-kubernetes-preserve-unknown-fields lets apiserver accept the
    # string-URL shorthand (collapsed to the full object form by
    # normalize_before_validation) which a Kubernetes structural schema
    # cannot express as a string|object union.
    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
        json_schema_extra={"x-kubernetes-preserve-unknown-fields": True},
    )

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

    collector: Annotated[
        GPUTelemetryCollectorType,
        Field(
            default=GPUTelemetryCollectorType.DCGM,
            description="GPU telemetry collector backend. Use 'dcgm' for DCGM exporter endpoints or 'pynvml' for local PyNVML collection.",
        ),
    ]

    mode: Annotated[
        GPUTelemetryMode,
        Field(
            default=GPUTelemetryMode.SUMMARY,
            description="GPU telemetry display mode. Summary emits aggregate console output; realtime_dashboard enables live dashboard updates.",
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
