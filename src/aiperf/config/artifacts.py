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
from aiperf.config.base import BaseConfig
from aiperf.config.defaults import MLflowDefaults, OutputDefaults
from aiperf.config.phases import _normalize_duration
from aiperf.config.user_files import UserFile
from aiperf.plugin.enums import GPUTelemetryCollectorType

__all__ = [
    "ArtifactsConfig",
    "MLflowConfig",
    "OTelConfig",
    "ServerMetricsConfig",
    "ServerMetricsDiscoveryConfig",
    "GpuTelemetryConfig",
]

# Type aliases for format arrays.
# Narrow to what the codebase actually emits: MetricsJsonExporter writes the
# summary JSON, RecordExportResultsProcessor writes the records JSONL. No YAML
# summary exporter and no records-CSV exporter exist; do not advertise them.
SummaryExportFormat = Literal["json"]
RecordsExportFormat = Literal["jsonl"]


class MLflowConfig(BaseConfig):
    """MLflow tracking and artifact-upload configuration."""

    model_config = ConfigDict(extra="forbid", validate_default=True)

    tracking_uri: Annotated[
        str | None,
        Field(default=MLflowDefaults.TRACKING_URI, description="MLflow tracking URI."),
    ]
    experiment: Annotated[
        str,
        Field(default=MLflowDefaults.EXPERIMENT, description="MLflow experiment name."),
    ]
    run_name: Annotated[
        str | None,
        Field(default=MLflowDefaults.RUN_NAME, description="MLflow run name."),
    ]
    tags: Annotated[
        str | None,
        Field(default=MLflowDefaults.TAGS, description="Comma-separated MLflow tags."),
    ]
    parent_run_id: Annotated[
        str | None,
        Field(default=None, description="Optional MLflow parent run ID."),
    ]
    artifact_globs: Annotated[
        list[str] | None,
        Field(
            default=MLflowDefaults.ARTIFACT_GLOBS,
            description="Artifact glob overrides for MLflow upload.",
        ),
    ]

    @property
    def enabled(self) -> bool:
        """Whether MLflow export/live streaming is enabled."""
        return self.tracking_uri is not None

    @property
    def tags_dict(self) -> dict[str, str]:
        """Parse comma-separated ``key:value`` MLflow tags."""
        if not self.tags:
            return {}
        tags: dict[str, str] = {}
        for item in self.tags.split(","):
            key, sep, value = item.partition(":")
            if sep and key.strip() and value.strip():
                tags[key.strip()] = value.strip()
        return tags

    @property
    def resolved_artifact_globs(self) -> list[str]:
        """Return MLflow artifact globs, applying defaults when unset."""
        return list(self.artifact_globs or MLflowDefaults.DEFAULT_ARTIFACT_GLOBS)


class OTelConfig(BaseConfig):
    """OpenTelemetry metrics streaming configuration."""

    model_config = ConfigDict(extra="forbid", validate_default=True)

    metrics_url: Annotated[
        str | None,
        Field(default=None, description="OTLP/HTTP metrics endpoint URL."),
    ]
    stream_metrics_enabled: Annotated[
        bool,
        Field(default=True, description="Stream metric records to OTel."),
    ]
    stream_timing_enabled: Annotated[
        bool,
        Field(default=True, description="Stream timing records to OTel."),
    ]
    custom_resource_attributes: Annotated[
        dict[str, str],
        Field(default_factory=dict, description="Custom OTel resource attributes."),
    ]
    gen_ai_provider: Annotated[
        str | None,
        Field(default=None, description="GenAI semantic convention provider override."),
    ]

    @property
    def collector_enabled(self) -> bool:
        """Whether OTel metrics streaming is enabled."""
        return self.metrics_url is not None

    @property
    def stream(self) -> str:
        """Human-readable stream selection for diagnostics."""
        if self.stream_metrics_enabled and self.stream_timing_enabled:
            return "default"
        if self.stream_metrics_enabled:
            return "metrics"
        if self.stream_timing_enabled:
            return "timing"
        return "none"


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
        str | None,
        Field(
            default=None,
            description="Base filename override applied to ALL profile and server-metrics "
            "exports. With prefix='foo' every output becomes `foo.csv`, `foo.json`, "
            "`foo_timeslices.{csv,json}`, `foo.jsonl`, `foo_raw.jsonl`, "
            "`foo_gpu_telemetry.jsonl`, `foo_server_metrics.{jsonl,json,csv,parquet}`. "
            "When unset (the default), historical per-file names are used: "
            "`profile_export_aiperf.csv/json`, `profile_export.jsonl`, "
            "`profile_export_raw.jsonl`, `gpu_telemetry_export.jsonl`, "
            "`server_metrics_export.{jsonl,json,csv,parquet}`. Known suffixes "
            "(`_raw.jsonl`, `_timeslices.{csv,json}`, `_gpu_telemetry.jsonl`, "
            "`_server_metrics.{jsonl,json,csv,parquet}`, `.csv`/`.json`/`.jsonl`/`.parquet`) "
            "are stripped from the supplied value so `--profile-export-prefix foo_raw.jsonl` "
            "still yields a clean `foo` base.",
        ),
    ]

    summary: Annotated[
        list[SummaryExportFormat] | Literal[False],
        Field(
            default_factory=lambda: ["json"],
            description="Summary export formats. "
            "Only 'json' is wired up to this field; the CSV summary is "
            "emitted regardless. Set to false to disable the summary JSON "
            "file only.",
        ),
    ]

    records: Annotated[
        list[RecordsExportFormat] | Literal[False],
        Field(
            default_factory=lambda: ["jsonl"],
            description="Per-request records export formats. "
            "Only 'jsonl' is wired up today. Set to false to disable the "
            "per-record JSONL file.",
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

    user_files: Annotated[
        list[UserFile],
        Field(
            default_factory=list,
            description="User-defined templated files materialized into the run directory "
            "before the benchmark begins.",
        ),
    ]

    auto_plot: Annotated[
        bool,
        Field(
            default=False,
            description=(
                "Auto-invoke `aiperf plot` against the artifact directory after the "
                "benchmark completes. Resolved by the CLI converter from the "
                "tri-state CLI flag and the active search recipe's auto_plot_default; "
                "by the time it lands here it is a plain bool."
            ),
        ),
    ]

    plot_required: Annotated[
        bool,
        Field(
            default=False,
            description=(
                "Treat auto-plot failures as fatal: re-raise so `aiperf profile` exits "
                "non-zero. Only meaningful when auto_plot is True. Default False = warn "
                "and continue."
            ),
        ),
    ]

    export_outputs_json: Annotated[
        bool,
        Field(
            default=False,
            description="Export generated response text to outputs.json after the run.",
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

    # Suffixes the user may legitimately tack onto `--profile-export-prefix`.
    # We strip them so `--profile-export-prefix foo_raw.jsonl` produces a
    # clean `foo` base just like `--profile-export-prefix foo`. Order matters:
    # longest match first so `_server_metrics.parquet` wins over `.parquet`.
    _PREFIX_SUFFIXES_TO_STRIP = (
        "_server_metrics.parquet",
        "_server_metrics.jsonl",
        "_server_metrics.json",
        "_server_metrics.csv",
        "_gpu_telemetry.jsonl",
        "_timeslices.csv",
        "_timeslices.json",
        "_raw.jsonl",
        ".parquet",
        ".csv",
        ".json",
        ".jsonl",
    )

    def _base(self) -> str | None:
        """Return the prefix with known export suffixes stripped, or None."""
        if self.prefix is None:
            return None
        base = self.prefix
        for suffix in self._PREFIX_SUFFIXES_TO_STRIP:
            if base.endswith(suffix):
                return base[: -len(suffix)]
        return base

    @property
    def profile_export_csv_file(self) -> Path:
        """Path for the CSV summary export file."""
        base = self._base()
        name = f"{base}.csv" if base else "profile_export_aiperf.csv"
        return self.dir / name

    @property
    def profile_export_json_file(self) -> Path:
        """Path for the JSON summary export file."""
        base = self._base()
        name = f"{base}.json" if base else "profile_export_aiperf.json"
        return self.dir / name

    @property
    def checkpoints_dir(self) -> Path:
        """Directory used for partial recovery checkpoints."""
        return self.dir / "checkpoints"

    @property
    def profile_export_timeslices_csv_file(self) -> Path:
        """Path for the timeslices CSV export file."""
        base = self._base()
        name = (
            f"{base}_timeslices.csv" if base else "profile_export_aiperf_timeslices.csv"
        )
        return self.dir / name

    @property
    def profile_export_timeslices_json_file(self) -> Path:
        """Path for the timeslices JSON export file."""
        base = self._base()
        name = (
            f"{base}_timeslices.json"
            if base
            else "profile_export_aiperf_timeslices.json"
        )
        return self.dir / name

    @property
    def profile_export_jsonl_file(self) -> Path:
        """Path for the per-record JSONL export file."""
        base = self._base()
        name = f"{base}.jsonl" if base else "profile_export.jsonl"
        return self.dir / name

    @property
    def outputs_json_file(self) -> Path:
        """Path for the aggregated generated outputs JSON export file."""
        return self.dir / OutputDefaults.OUTPUTS_JSON_FILE

    @property
    def profile_export_raw_jsonl_file(self) -> Path:
        """Path for the raw request/response JSONL export file."""
        base = self._base()
        name = f"{base}_raw.jsonl" if base else "profile_export_raw.jsonl"
        return self.dir / name

    @property
    def profile_export_gpu_telemetry_jsonl_file(self) -> Path:
        """Path for the GPU telemetry JSONL export file."""
        base = self._base()
        name = f"{base}_gpu_telemetry.jsonl" if base else "gpu_telemetry_export.jsonl"
        return self.dir / name

    @property
    def server_metrics_export_jsonl_file(self) -> Path:
        """Path for the server metrics JSONL export file."""
        base = self._base()
        name = f"{base}_server_metrics.jsonl" if base else "server_metrics_export.jsonl"
        return self.dir / name

    @property
    def server_metrics_export_json_file(self) -> Path:
        """Path for the server metrics JSON export file."""
        base = self._base()
        name = f"{base}_server_metrics.json" if base else "server_metrics_export.json"
        return self.dir / name

    @property
    def server_metrics_export_csv_file(self) -> Path:
        """Path for the server metrics CSV export file."""
        base = self._base()
        name = f"{base}_server_metrics.csv" if base else "server_metrics_export.csv"
        return self.dir / name

    @property
    def server_metrics_export_parquet_file(self) -> Path:
        """Path for the server metrics Parquet export file."""
        base = self._base()
        name = (
            f"{base}_server_metrics.parquet"
            if base
            else "server_metrics_export.parquet"
        )
        return self.dir / name

    @property
    def export_level(self) -> ExportLevel:
        """Derive ExportLevel from the raw/records fields."""
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
