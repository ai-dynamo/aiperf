# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import ConfigDict, Field

from aiperf.common.models.base_models import AIPerfBaseModel
from aiperf.common.models.branch_stats import BranchStats
from aiperf.common.models.error_models import ErrorDetailsCount
from aiperf.config.config import BenchmarkConfig

if TYPE_CHECKING:
    from aiperf.config import BenchmarkRun

# =============================================================================
# JSON Metric Result
# =============================================================================


class JsonMetricResult(AIPerfBaseModel):
    """The result values of a single metric for JSON export.

    NOTE:
    The shape of this model originates from GenAI-Perf JSON output for
    downstream-tool compatibility. New fields may be added when AIPerf
    surfaces information GenAI-Perf does not (e.g. `count`, `sum` added in
    schema 1.1). When adding a field, bump `JsonExportData.SCHEMA_VERSION`
    and document the field in `docs/reference/json-export-schema.md`. Do
    not remove or rename existing fields without a matching schema bump.
    """

    unit: str = Field(description="The unit of the metric, e.g. 'ms' or 'requests/sec'")
    avg: float | None = None
    p1: float | None = None
    p5: float | None = None
    p10: float | None = None
    p25: float | None = None
    p50: float | None = None
    p75: float | None = None
    p90: float | None = None
    p95: float | None = None
    p99: float | None = None
    min: int | float | None = None
    max: int | float | None = None
    std: float | None = None
    count: int | None = Field(
        default=None,
        description=(
            "Number of records contributing to this metric's distribution. "
            "Present only for record-type metrics; omitted for derived/aggregate "
            "scalar metrics where the count would trivially be 1 and risks being "
            "misread as the request count."
        ),
    )
    sum: int | float | None = Field(
        default=None,
        description=(
            "Sum of all metric values across records. Present for record-type "
            "metrics with at least one observation; absent for derived/aggregate "
            "metrics whose value is itself the computed total or rate."
        ),
    )

    @staticmethod
    def project_summary_dict(payload: dict[str, Any]) -> dict[str, JsonMetricResult]:
        return {
            key: JsonMetricResult.model_validate(value)
            for key, value in payload.items()
            if isinstance(value, dict) and "unit" in value
        }


# =============================================================================
# Telemetry Export Data
# =============================================================================


class TelemetrySummary(AIPerfBaseModel):
    """Summary information for telemetry collection."""

    endpoints_configured: list[str] | None = None
    endpoints_successful: list[str] | None = None
    start_time: datetime
    end_time: datetime


class GpuSummary(AIPerfBaseModel):
    """Summary of GPU telemetry data."""

    gpu_index: int
    gpu_name: str
    gpu_uuid: str
    hostname: str | None
    namespace: str | None = None
    pod_name: str | None = None
    metrics: dict[str, JsonMetricResult]  # metric_key -> {stat_key -> value}


class EndpointData(AIPerfBaseModel):
    """Data for a single endpoint."""

    gpus: dict[str, GpuSummary]


class TelemetryExportData(AIPerfBaseModel):
    """Telemetry data structure for JSON export."""

    summary: TelemetrySummary
    endpoints: dict[str, EndpointData]
    error_summary: list[ErrorDetailsCount] | None = Field(
        default=None,
        description="A list of the unique error details and their counts",
    )


# =============================================================================
# Timeslice Export Data
# =============================================================================


class TimesliceData(AIPerfBaseModel):
    """Data for a single timeslice.

    Contains metrics for one time slice with dynamic metric fields
    added via Pydantic's extra="allow" setting.
    """

    model_config = ConfigDict(extra="allow")

    timeslice_index: int


class TimesliceCollectionExportData(AIPerfBaseModel):
    """Export data for all timeslices in a single file.

    Contains an array of timeslice data objects with metadata.
    """

    timeslices: list[TimesliceData]
    input_config: BenchmarkConfig | None = None


# =============================================================================
# Run Metadata
# =============================================================================


class RunInfo(AIPerfBaseModel):
    """Per-run reproducibility metadata.

    Captures the variation/trial coordinates and the actual seed the run used,
    so a downstream reader of ``profile_export_aiperf.json`` alone can locate
    the run's place in a sweep and reproduce its workload deterministically
    without needing the internal ``run_config.json`` handoff file.
    """

    benchmark_id: str | None = Field(
        default=None,
        description=(
            "Unique identifier for this benchmark run "
            "(BenchmarkRun.benchmark_id). Duplicates the top-level "
            "`benchmark_id` for readers that consume `run_info` as a "
            "self-contained reproducibility block."
        ),
    )
    sweep_id: str | None = Field(
        default=None,
        description=(
            "UUID of the outer sweep this run belongs to "
            "(BenchmarkPlan.sweep_id). Stable across every variation and "
            "trial of one plan; lets readers join all per-run JSON exports "
            "from the same sweep without consulting the parent multi-run "
            "artifact directory. None for runs constructed outside the "
            "multi-run orchestrator."
        ),
    )
    random_seed: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Resolved per-run random seed (envelope `random_seed` for "
            "single-run; SHA-derived for adaptive iterations beyond the "
            "plan-time list). None when the user opted out of consistent "
            "seeding and no `--random-seed` was set."
        ),
    )
    trial: int | None = Field(
        default=None,
        ge=0,
        description="Zero-based trial index within this variation.",
    )
    run_label: str | None = Field(
        default=None,
        description=("Human-readable run label (e.g. `concurrency_10`, `run_0001`)."),
    )
    variation_label: str | None = Field(
        default=None,
        description="Sweep variation label, or `base` for non-sweep runs.",
    )
    variation_index: int | None = Field(
        default=None,
        ge=0,
        description="Sweep variation index (0 for non-sweep / first cell).",
    )
    variation_values: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Sweep parameter point as `{path: value}`. Empty dict for "
            "non-sweep runs; populated for grid/zip/scenario/adaptive cells."
        ),
    )
    cli_command: str | None = Field(
        default=None,
        description=(
            "Redacted CLI command that launched this run "
            "(`BenchmarkRun.cli_command`), captured from sys.argv. None when "
            "the run was constructed without a CLI context."
        ),
    )

    @classmethod
    def from_run(cls, run: BenchmarkRun | None) -> RunInfo | None:
        """Project a ``BenchmarkRun`` envelope onto its run-identity shape.

        Single source of truth shared by ``profile_export_aiperf.json`` and the
        live ``GET /api/run`` endpoint, so the on-disk and live shapes cannot
        diverge.
        """
        if run is None:
            return None
        variation = run.variation
        return cls(
            benchmark_id=run.benchmark_id,
            sweep_id=run.sweep_id,
            random_seed=run.random_seed,
            trial=run.trial,
            run_label=run.label or None,
            variation_label=variation.label if variation is not None else None,
            variation_index=variation.index if variation is not None else None,
            variation_values=dict(variation.values) if variation is not None else None,
            cli_command=run.cli_command,
        )


# =============================================================================
# Main JSON Export Data
# =============================================================================


class JsonExportData(AIPerfBaseModel):
    """Summary data to be exported to a JSON file.

    NOTE:
    This model has been designed to mimic the structure of the GenAI-Perf JSON output
    as closely as possible. Be careful when modifying this model to not break the
    compatibility with the GenAI-Perf JSON output.
    """

    # NOTE: The extra="allow" setting is needed to allow additional metrics not defined in this class
    #       to be added to the export data. It is also already set in the AIPerfBaseModel,
    #       but we are setting it here to guard against base model changes.
    model_config = ConfigDict(extra="allow")

    # Increment on breaking changes to the export structure
    SCHEMA_VERSION: ClassVar[str] = "1.3"

    schema_version: str | None = Field(
        default=None,
        description="Schema version for this export format (for backward compatibility)",
    )
    aiperf_version: str | None = Field(
        default=None,
        description="AIPerf version that generated this export (for backward compatibility)",
    )
    benchmark_id: str | None = Field(
        default=None,
        description="Unique identifier for this benchmark run (for backward compatibility)",
    )
    request_throughput: JsonMetricResult | None = None
    request_latency: JsonMetricResult | None = None
    request_count: JsonMetricResult | None = None
    time_to_first_token: JsonMetricResult | None = None
    time_to_second_token: JsonMetricResult | None = None
    inter_token_latency: JsonMetricResult | None = None
    output_token_throughput: JsonMetricResult | None = None
    output_token_throughput_per_user: JsonMetricResult | None = None
    output_sequence_length: JsonMetricResult | None = None
    input_sequence_length: JsonMetricResult | None = None
    goodput: JsonMetricResult | None = None
    good_request_count: JsonMetricResult | None = None
    output_token_count: JsonMetricResult | None = None
    reasoning_token_count: JsonMetricResult | None = None
    min_request_timestamp: JsonMetricResult | None = None
    max_response_timestamp: JsonMetricResult | None = None
    inter_chunk_latency: JsonMetricResult | None = None
    total_output_tokens: JsonMetricResult | None = None
    total_reasoning_tokens: JsonMetricResult | None = None
    benchmark_duration: JsonMetricResult | None = None
    total_isl: JsonMetricResult | None = None
    total_osl: JsonMetricResult | None = None
    error_request_count: JsonMetricResult | None = None
    error_isl: JsonMetricResult | None = None
    total_error_isl: JsonMetricResult | None = None
    telemetry_data: TelemetryExportData | None = None
    input_config: BenchmarkConfig | None = None
    run_info: RunInfo | None = None
    was_cancelled: bool | None = None
    error_summary: list[ErrorDetailsCount] | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    branch_stats: BranchStats | None = Field(
        default=None,
        description=(
            "DAG branch orchestration counters (children spawned/completed/"
            "errored/truncated, parents suspended/resumed). Present only on "
            "DAG-shaped runs; absent for non-DAG benchmarks."
        ),
    )
