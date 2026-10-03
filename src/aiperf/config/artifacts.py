# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
AIPerf Configuration v2.0 - Pydantic Models

Artifacts - Export and output settings for benchmark results.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    BeforeValidator,
    ConfigDict,
    Field,
    model_validator,
)

from aiperf.common.enums import ExportLevel
from aiperf.config.base import BaseConfig
from aiperf.config.phases import _normalize_duration
from aiperf.config.user_files import UserFile

__all__ = [
    "ArtifactsConfig",
    "OutputDefaults",
]

# Type aliases for format arrays.
# Narrow to what the codebase actually emits: MetricsJsonExporter writes the
# summary JSON, RecordShardJSONLWriter writes the records JSONL. No YAML
# summary exporter and no records-CSV exporter exist; do not advertise them.
SummaryExportFormat = Literal["json"]
RecordsExportFormat = Literal["jsonl"]


@dataclass(frozen=True)
class OutputDefaults:
    ARTIFACT_DIRECTORY = Path("./artifacts")
    RAW_RECORDS_FOLDER = Path("raw_records")
    RECORDS_SHARDS_FOLDER = Path("records_shards")
    OUTPUT_FRAGMENTS_FOLDER = Path("output_fragments")
    LOG_FOLDER = Path("logs")
    LOG_FILE = Path("aiperf.log")
    INPUTS_JSON_FILE = Path("inputs.json")
    OUTPUTS_JSON_FILE = Path("outputs.json")
    PROFILE_EXPORT_AIPERF_CSV_FILE = Path("profile_export_aiperf.csv")
    PROFILE_EXPORT_AIPERF_JSON_FILE = Path("profile_export_aiperf.json")
    PROFILE_EXPORT_AIPERF_TIMESLICES_CSV_FILE = Path(
        "profile_export_aiperf_timeslices.csv"
    )
    PROFILE_EXPORT_AIPERF_TIMESLICES_JSON_FILE = Path(
        "profile_export_aiperf_timeslices.json"
    )
    PROFILE_EXPORT_JSONL_FILE = Path("profile_export.jsonl")
    PROFILE_EXPORT_RAW_JSONL_FILE = Path("profile_export_raw.jsonl")
    PROFILE_EXPORT_CONSOLE_TXT_FILE = Path("profile_export_console.txt")
    PROFILE_EXPORT_GPU_TELEMETRY_JSONL_FILE = Path("gpu_telemetry_export.jsonl")
    SERVER_METRICS_EXPORT_JSONL_FILE = Path("server_metrics_export.jsonl")
    SERVER_METRICS_EXPORT_JSON_FILE = Path("server_metrics_export.json")
    SERVER_METRICS_EXPORT_CSV_FILE = Path("server_metrics_export.csv")
    SERVER_METRICS_EXPORT_PARQUET_FILE = Path("server_metrics_export.parquet")
    NETWORK_LATENCY_EXPORT_JSONL_FILE = Path("profile_export_network_latency.jsonl")
    EXPORT_LEVEL = ExportLevel.RECORDS
    EXPORT_HTTP_TRACE = False
    SHOW_TRACE_TIMING = False
    SLICE_DURATION = None


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
            "`foo_outputs.json`, `foo_gpu_telemetry.jsonl`, "
            "`foo_server_metrics.{jsonl,json,csv,parquet}`. "
            "When unset (the default), historical per-file names are used: "
            "`profile_export_aiperf.csv/json`, `profile_export.jsonl`, "
            "`profile_export_raw.jsonl`, `outputs.json`, `gpu_telemetry_export.jsonl`, "
            "`server_metrics_export.{jsonl,json,csv,parquet}`. Known suffixes "
            "(`_raw.jsonl`, `_outputs.json`, `_timeslices.{csv,json}`, `_gpu_telemetry.jsonl`, "
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
        bool | None,
        Field(
            default=None,
            description=(
                "Auto-invoke `aiperf plot` against the artifact directory after the "
                "benchmark completes. Unset (the default) means a `plot:` section "
                "implies True; set False to suppress plots even with one, True to "
                "force them. `AIPerfConfig` resolves this to a concrete bool, so any "
                "config that came through the loader carries a plain bool."
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
        bool | None,
        Field(
            default=None,
            description="Export generated response text after the run, to "
            "`outputs.json` or `<prefix>_outputs.json` when `prefix` is set. "
            "Unset (the default) means the `raw` export level implies it; set "
            "False to opt out at raw, True to force it at any level. Always a "
            "concrete bool once validation has run.",
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
        self._apply_raw_implies_outputs_json()
        return self

    def _apply_raw_implies_outputs_json(self) -> None:
        """Resolve the tri-state ``export_outputs_json`` to a concrete bool.

        ``None`` means "decide from the export level": raw implies the file,
        because if you asked for the full request/response bodies you want the
        generated text too. An explicit True or False is honored as-is, so
        raw-level debugging can decline the second copy.

        Tri-state rather than ``model_fields_set`` on a ``False`` default: the
        Kubernetes apiserver materializes CRD ``default:`` values on write, so a
        defaulted-to-False field is indistinguishable from a user-authored one
        and would silently defeat the implication for every AIPerfJob.
        """
        if self.export_outputs_json is None:
            self.export_outputs_json = self.raw
            # Resolving is not authoring: keep the field out of
            # ``model_fields_set`` so ``exclude_unset`` dumps (the Kubernetes
            # ConfigMap payload) do not invent a value the user never wrote.
            self.model_fields_set.discard("export_outputs_json")

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
        "_network_latency.jsonl",
        "_gpu_telemetry.jsonl",
        "_timeslices.csv",
        "_timeslices.json",
        "_console.txt",
        "_outputs.json",
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
        base = self._base()
        default = OutputDefaults.OUTPUTS_JSON_FILE.name
        name = f"{base}_outputs.json" if base else default
        return self.dir / name

    @property
    def profile_export_raw_jsonl_file(self) -> Path:
        """Path for the raw request/response JSONL export file."""
        base = self._base()
        name = f"{base}_raw.jsonl" if base else "profile_export_raw.jsonl"
        return self.dir / name

    @property
    def profile_export_console_txt_file(self) -> Path:
        """Path for the plain-text console output capture file."""
        base = self._base()
        name = f"{base}_console.txt" if base else "profile_export_console.txt"
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
    def accuracy_export_jsonl_file(self) -> Path:
        """Path for the per-record accuracy JSONL export file."""
        base = self._base()
        name = f"{base}_accuracy.jsonl" if base else "accuracy_export.jsonl"
        return self.dir / name

    @property
    def network_latency_export_jsonl_file(self) -> Path:
        """Path for the per-sample network latency RTT probe JSONL export file."""
        base = self._base()
        name = (
            f"{base}_network_latency.jsonl"
            if base
            else "profile_export_network_latency.jsonl"
        )
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
