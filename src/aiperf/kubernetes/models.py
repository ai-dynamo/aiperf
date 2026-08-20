# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Data models for Kubernetes operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import Field, field_validator

from aiperf.common.endpoint_credentials import redact_sweep_public_data
from aiperf.common.finite import FiniteFloat
from aiperf.common.redact import redact_url
from aiperf.kubernetes.constants import AIPerfLabels, Annotations, ProgressAnnotations
from aiperf.kubernetes.enums import JobSetStatus
from aiperf.kubernetes.k8s_models import K8sCamelModel


@dataclass
class JobSetInfo:
    """Information about a found JobSet.

    Use ``JobSetInfo.from_raw(raw_dict)`` to create from a Kubernetes API
    response dict.  All field extraction and status parsing is handled here.
    """

    name: str
    """Kubernetes JobSet resource name."""

    namespace: str
    """Kubernetes namespace containing the JobSet."""

    jobset: dict[str, Any]
    """Raw JobSet dict from the Kubernetes API."""

    status: str
    """Current status: "Running", "Completed", or "Failed"."""

    custom_name: str | None = None
    """User-provided benchmark name, if set."""

    model: str | None = None
    """Target model name from the endpoint, if set."""

    endpoint: str | None = None
    """Target LLM endpoint URL, if set."""

    def __post_init__(self) -> None:
        """Keep the legacy JobSet display fallback credential-safe."""
        if self.endpoint is not None:
            self.endpoint = redact_url(self.endpoint)

    # -- Factory ----------------------------------------------------------

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> JobSetInfo:
        """Create a JobSetInfo from a raw Kubernetes JobSet dict."""
        metadata = raw.get("metadata", {})
        labels = metadata.get("labels", {})
        annotations = metadata.get("annotations", {})
        return cls(
            name=metadata["name"],
            namespace=metadata["namespace"],
            jobset=raw,
            status=cls._parse_status(raw),
            custom_name=labels.get(AIPerfLabels.NAME),
            model=annotations.get(Annotations.MODEL),
            endpoint=annotations.get(Annotations.ENDPOINT),
        )

    # -- Derived properties -----------------------------------------------

    @property
    def job_id(self) -> str:
        """AIPerf job ID (falls back to the JobSet name)."""
        labels = self.jobset.get("metadata", {}).get("labels", {})
        return labels.get(AIPerfLabels.JOB_ID, self.name)

    @property
    def created(self) -> str:
        """Creation timestamp from the JobSet metadata."""
        return self.jobset.get("metadata", {}).get("creationTimestamp", "")

    @property
    def progress(self) -> str | None:
        """Human-readable progress string, or None if unavailable."""
        annotations = self.jobset.get("metadata", {}).get("annotations", {})
        if not annotations.get(ProgressAnnotations.STATUS):
            return None

        parts: list[str] = []
        phase = annotations.get(ProgressAnnotations.PHASE, "")
        if phase:
            parts.append(phase)
        requests = annotations.get(ProgressAnnotations.REQUESTS)
        if requests:
            parts.append(requests)
        percent = annotations.get(ProgressAnnotations.PERCENT)
        if percent:
            parts.append(f"({percent}%)")
        return " ".join(parts) if parts else annotations.get(ProgressAnnotations.STATUS)

    # -- Private helpers --------------------------------------------------

    @staticmethod
    def _parse_status(raw: dict[str, Any]) -> str:
        """Extract status string from a raw JobSet dict."""
        status = raw.get("status", {})
        conditions = status.get("conditions", [])
        condition_status = {c.get("type"): c.get("status") for c in conditions}
        if condition_status.get("Completed") == "True":
            return JobSetStatus.COMPLETED
        if condition_status.get("Failed") == "True":
            replicated = {
                rj.get("name"): rj for rj in status.get("replicatedJobsStatus", [])
            }
            if replicated.get("controller", {}).get("failed", 0) > 0:
                return JobSetStatus.FAILED
        return JobSetStatus.RUNNING


# =============================================================================
# AIPerfJob CR structure — parsed via AIPerfJobCR.model_validate(raw_dict)
#
# Reuses operator models where they exist:
#   - PhaseProgress (operator/models.py) for status.phases values
#   - MetricsSummary (operator/models.py) for summary extraction
# Defines only what doesn't exist: metadata, spec subset, status envelope.
# =============================================================================

# Sweep-controller-stamped annotation holding the swept parameter values as a
# bounded JSON object string, e.g. ``{"phases.profiling.concurrency":17}``.
# Same key and encoding as
# ``sweep_controller.k8s_executor.VARIATION_VALUES_ANNOTATION`` and
# ``routers/_sweeps_live._VARIATION_VALUES_ANNOTATION``; duplicated as a
# literal rather than imported so this display model stays free of a
# sweep-controller import.
VARIATION_VALUES_ANNOTATION = "aiperf.nvidia.com/variation-values"


class CRMetadata(K8sCamelModel):
    """Kubernetes object metadata (subset relevant to AIPerfJob)."""

    name: str = Field(default="", description="Resource name.")
    namespace: str = Field(default="", description="Resource namespace.")
    creation_timestamp: str = Field(default="", description="Creation timestamp.")
    labels: dict[str, str] = Field(
        default_factory=dict,
        description="K8s labels on the resource. Used to read sweep linkage "
        "(aiperf.nvidia.com/sweep, /variation-index, /variation-label) for "
        "AIPerfJob children of an AIPerfSweep.",
    )
    annotations: dict[str, str] = Field(
        default_factory=dict,
        description="K8s annotations on the resource. Carries the fourth "
        "member of the sweep linkage, aiperf.nvidia.com/variation-values: "
        "label values are capped at 63 characters and forbid JSON "
        "punctuation, so the swept parameter values cannot live in a label.",
    )


class CREndpoint(K8sCamelModel):
    """Endpoint section from AIPerfJob spec."""

    url: str | None = Field(default=None, description="Single endpoint URL.")
    urls: list[str] = Field(default_factory=list, description="List of endpoint URLs.")


class CRBenchmark(K8sCamelModel):
    """Benchmark section from AIPerfJob spec (nested under spec.benchmark)."""

    models: str | list | dict[str, Any] = Field(
        default_factory=list, description="Model name(s) to benchmark."
    )
    endpoint: CREndpoint | dict[str, Any] = Field(
        default_factory=CREndpoint, description="Endpoint configuration."
    )


class CRSpec(K8sCamelModel):
    """AIPerfJob spec (subset relevant for display).

    AIPerfConfig fields are nested under spec.benchmark. Deployment fields
    (image, podTemplate, etc.) live at the spec level.
    """

    benchmark: CRBenchmark = Field(
        default_factory=CRBenchmark, description="Benchmark configuration."
    )


class CRWorkerStatus(K8sCamelModel):
    """Worker readiness counts from status.workers."""

    ready: int = Field(default=0, description="Number of ready workers.")
    total: int = Field(default=0, description="Total number of workers.")


class CRJobStatus(K8sCamelModel):
    """AIPerfJob status subresource.

    Phase progress dicts (status.phases) are written by the operator via
    PhaseProgress.to_k8s_dict() (camelCase keys including
    ``requestsProgressPercent``). Summary dicts are written via
    MetricsSummary.to_status_dict() — a curated nested
    ``{metric_tag: {avg, p50, p99, ...}}`` projection of the AIPerf metrics
    payload, e.g. ``summary["request_throughput"]["avg"]``,
    ``summary["request_latency"]["p99"]``. Both are kept as raw dicts to
    avoid a circular import with the operator package.
    """

    phase: str = Field(default="Pending", description="Current lifecycle phase.")
    job_id: str = Field(default="", description="Operator-assigned job ID.")
    job_set_name: str | None = Field(
        default=None, description="Name of the managed JobSet."
    )
    workers: CRWorkerStatus = Field(
        default_factory=CRWorkerStatus, description="Worker readiness."
    )
    current_phase: str | None = Field(
        default=None, description="Current benchmark phase name."
    )
    error: str | None = Field(default=None, description="Error message if failed.")
    start_time: str | None = Field(default=None, description="Job start timestamp.")
    completion_time: str | None = Field(
        default=None, description="Job completion timestamp."
    )
    live_summary: dict[str, Any] | None = Field(
        default=None,
        description="Live metrics (MetricsSummary.to_status_dict() format).",
    )
    summary: dict[str, Any] | None = Field(
        default=None,
        description="Final metrics (MetricsSummary.to_status_dict() format).",
    )
    phases: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Per-phase progress (PhaseProgress.to_k8s_dict() format).",
    )
    run_epoch: int | None = Field(
        default=None,
        description="Epoch-seconds key of the most recent successful run. Use as "
        "{epoch} in /api/v1/results/<ns>/<name>/runs/<epoch>/ to pin historical artifacts.",
    )

    @field_validator("phase", mode="before")
    @classmethod
    def coerce_none_phase(cls, v: str | None) -> str:
        """Coerce None or empty phase to 'Pending'."""
        return v or "Pending"


def _first_model_name(models: str | list | dict[str, Any]) -> str | None:
    """Extract a display model name from ``spec.benchmark.models``.

    Handles the three YAML shapes: bare string, list of names, and the
    long-form ``{"items": [{"name": ...}]}`` dict.
    """
    if isinstance(models, str):
        return models
    if isinstance(models, dict):
        items = models.get("items", [])
        return items[0].get("name") if items else None
    if models:
        first = models[0]
        return first if isinstance(first, str) else None
    return None


def _endpoint_url(endpoint: CREndpoint | dict[str, Any]) -> str | None:
    """Extract a display URL from ``spec.benchmark.endpoint`` (dict or model form)."""
    if isinstance(endpoint, dict):
        return endpoint.get("url") or (endpoint.get("urls", [None])[0])
    return endpoint.url or (endpoint.urls[0] if endpoint.urls else None)


def _requests_progress_percent(
    phases: dict[str, dict[str, Any]], current_phase: str | None = None
) -> float | None:
    """Read ``requestsProgressPercent`` for the phase the job is actually in.

    ``status.phases`` is a CRD object map, and the apiserver alphabetizes those
    keys on storage -- so "last one wins" over dict order resolved to
    ``warmup``, not the newest phase. A job 20% into profiling reported the
    warmup phase's 100%. ``status.currentPhase`` is the authoritative pointer;
    the last entry is only a fallback for statuses written before it existed.
    """
    if current_phase:
        entry = phases.get(current_phase)
        if isinstance(entry, dict):
            pct = entry.get("requestsProgressPercent")
            if pct is not None:
                return float(pct)

    progress: float | None = None
    for p in phases.values():
        pct = p.get("requestsProgressPercent")
        if pct is not None:
            progress = float(pct)
    return progress


def _summary_stat(summary: dict[str, Any], tag: str, stat: str) -> float | None:
    """Read ``summary[tag][stat]`` as float, tolerating missing/malformed entries."""
    entry = summary.get(tag) if isinstance(summary, dict) else None
    if not isinstance(entry, dict):
        return None
    val = entry.get(stat)
    return float(val) if isinstance(val, (int, float)) else None


def _total_requests(summary: dict[str, Any]) -> int | None:
    """Total request count from a status summary dict.

    Prefers the derived ``total_requests`` scalar that
    MetricsSummary.from_metrics writes alongside the per-tag entries; falls
    back to ``request_count.avg`` for older statuses written before the
    derived scalar landed.
    """
    if isinstance(summary, dict):
        raw_total = summary.get("total_requests")
        if isinstance(raw_total, (int, float)):
            return int(raw_total)
    rc = _summary_stat(summary, "request_count", "avg")
    return int(rc) if rc is not None else None


def _error_rate(summary: dict[str, Any]) -> float | None:
    """Read the derived ``error_rate`` scalar from a status summary dict."""
    if isinstance(summary, dict):
        raw_err = summary.get("error_rate")
        if isinstance(raw_err, (int, float)):
            return float(raw_err)
    return None


def _label_int(labels: dict[str, str], key: str) -> int | None:
    """Parse an integer-valued label, tolerating absent or malformed values."""
    raw = labels.get(key)
    try:
        return int(raw) if raw is not None else None
    except ValueError:
        return None


def _sweep_linkage(
    labels: dict[str, str],
) -> tuple[str | None, int | None, str | None, int | None]:
    """Read sweep linkage labels as
    ``(sweep_name, variation_index, variation_label, trial_index)``.

    The labels are stamped on every AIPerfJob created by the sweep-controller
    (``trial_index`` only for multi-trial sweeps); standalone jobs return
    ``(None, None, None, None)``.
    """
    sweep_name = labels.get("aiperf.nvidia.com/sweep") or None
    variation_index = _label_int(labels, "aiperf.nvidia.com/variation-index")
    variation_label = labels.get("aiperf.nvidia.com/variation-label") or None
    trial_index = _label_int(labels, "aiperf.nvidia.com/trial-index")
    return sweep_name, variation_index, variation_label, trial_index


def _variation_values(annotations: dict[str, str]) -> str | None:
    """Read the swept parameter values from the child's annotations.

    The sweep-controller stamps ``aiperf.nvidia.com/variation-values`` at
    child-create time (``sweep_controller/k8s_executor._build_child_metadata``),
    so this is the earliest and only place the values are observable while a
    child is still running -- ``AIPerfSweep.status.runs[]`` gains an entry only
    once the child reaches a terminal phase, and
    ``status.aggregate.children`` only once the sweep-controller patches it.

    Returns None for standalone jobs and for the empty annotation, so callers
    fall back to ``variation_label`` rather than render an empty descriptor.
    """
    return annotations.get(VARIATION_VALUES_ANNOTATION) or None


class AIPerfJobCR(K8sCamelModel):
    """Parsed AIPerfJob custom resource.

    Use ``AIPerfJobCR.model_validate(raw_dict)`` to parse a raw K8s API
    response dict. Then call ``to_info()`` for a flat CLI display model.
    """

    metadata: CRMetadata = Field(
        default_factory=CRMetadata, description="K8s object metadata."
    )
    spec: CRSpec = Field(default_factory=CRSpec, description="Job specification.")
    status: CRJobStatus = Field(
        default_factory=CRJobStatus, description="Job status subresource."
    )

    def to_info(self) -> AIPerfJobInfo:
        """Convert to flat AIPerfJobInfo for CLI display."""
        # Summary: operator writes nested metric tags via MetricsSummary.from_metrics(),
        # so request_throughput.avg / request_latency.p99 are the canonical reads.
        summary = self.status.live_summary or self.status.summary or {}
        throughput = _summary_stat(summary, "request_throughput", "avg")
        latency = _summary_stat(summary, "request_latency", "p99")
        sweep_name, variation_index, variation_label, trial_index = _sweep_linkage(
            self.metadata.labels
        )

        return AIPerfJobInfo(
            name=self.metadata.name,
            namespace=self.metadata.namespace,
            phase=self.status.phase,
            job_id=self.status.job_id or self.metadata.name,
            jobset_name=self.status.job_set_name,
            workers_ready=self.status.workers.ready,
            workers_total=self.status.workers.total,
            current_phase=self.status.current_phase,
            error=self.status.error,
            start_time=self.status.start_time,
            completion_time=self.status.completion_time,
            created=self.metadata.creation_timestamp,
            progress_percent=_requests_progress_percent(
                self.status.phases, self.status.current_phase
            ),
            throughput_rps=float(throughput) if throughput is not None else None,
            latency_p99_ms=float(latency) if latency is not None else None,
            ttft_ms=_summary_stat(summary, "time_to_first_token", "avg"),
            output_token_throughput_tps=_summary_stat(
                summary, "output_token_throughput", "avg"
            ),
            inter_token_latency_ms=_summary_stat(summary, "inter_token_latency", "avg"),
            total_requests=_total_requests(summary),
            error_rate=_error_rate(summary),
            model=_first_model_name(self.spec.benchmark.models),
            endpoint=_endpoint_url(self.spec.benchmark.endpoint),
            sweep_name=sweep_name,
            variation_index=variation_index,
            variation_label=variation_label,
            variation_values=_variation_values(self.metadata.annotations),
            trial_index=trial_index,
        )


# =============================================================================
# AIPerfJobInfo — flat display model for CLI consumption
# =============================================================================


class AIPerfJobInfo(K8sCamelModel):
    """Flat view of an AIPerfJob for CLI display.

    Constructed via ``AIPerfJobCR.model_validate(raw).to_info()`` for
    data from the K8s API, or directly with kwargs for fallback paths.
    """

    name: str = Field(description="AIPerfJob resource name.")
    namespace: str = Field(description="Kubernetes namespace containing the AIPerfJob.")
    phase: str = Field(description="Current lifecycle phase.")
    job_id: str = Field(description="Operator-assigned job ID.")
    jobset_name: str | None = Field(
        default=None, description="Name of the managed JobSet from .status.jobSetName."
    )
    workers_ready: int = Field(default=0, ge=0, description="Number of ready workers.")
    workers_total: int = Field(default=0, ge=0, description="Total number of workers.")
    current_phase: str | None = Field(
        default=None,
        description="Current benchmark phase name (e.g. warmup, profiling).",
    )
    error: str | None = Field(
        default=None, description="Error message if the job failed."
    )
    start_time: str | None = Field(
        default=None, description="ISO 8601 timestamp when the job started."
    )
    completion_time: str | None = Field(
        default=None, description="ISO 8601 timestamp when the job completed."
    )
    created: str = Field(default="", description="Creation timestamp from metadata.")
    progress_percent: FiniteFloat | None = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Overall progress percentage (0-100).",
    )
    throughput_rps: FiniteFloat | None = Field(
        default=None, description="Live or final throughput in requests per second."
    )
    latency_p99_ms: FiniteFloat | None = Field(
        default=None, description="Live or final p99 latency in milliseconds."
    )
    ttft_ms: FiniteFloat | None = Field(
        default=None,
        description=(
            "Live average time-to-first-token in milliseconds "
            "(time_to_first_token.avg from status.liveSummary). None for "
            "non-streaming endpoints or before any responses arrive."
        ),
    )
    output_token_throughput_tps: FiniteFloat | None = Field(
        default=None,
        description=(
            "Live average output token throughput, tokens per second "
            "(output_token_throughput.avg). None for non-streaming endpoints "
            "or completion-only benchmarks."
        ),
    )
    inter_token_latency_ms: FiniteFloat | None = Field(
        default=None,
        description=(
            "Live average inter-token latency in milliseconds "
            "(inter_token_latency.avg). None until at least two tokens have "
            "been observed on a streaming endpoint."
        ),
    )
    total_requests: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Total successful + failed requests issued so far. Derived as "
            "request_count.avg + error_request_count.avg (successes + errors) "
            "by ``MetricsSummary.from_metrics``, falling back to "
            "request_count.avg on older statuses."
        ),
    )
    error_rate: FiniteFloat | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of requests that errored (0..1). Derived by "
            "``MetricsSummary.from_metrics`` from the authoritative "
            "request_error_rate metric (rate/100) when present, else as "
            "error_count / (successes + errors)."
        ),
    )
    model: str | None = Field(default=None, description="Target model name from spec.")
    endpoint: str | None = Field(
        default=None, description="Credential-redacted target endpoint URL from spec."
    )
    source: Literal["live", "archived", "both"] = Field(
        default="live",
        description=(
            "Provenance: 'live' = CR on cluster only; 'archived' = PVC results "
            "only (CR no longer exists); 'both' = CR + PVC results."
        ),
    )
    sweep_name: str | None = Field(
        default=None,
        description="Parent AIPerfSweep name when this job is a sweep child.",
    )
    variation_index: int | None = Field(
        default=None,
        ge=0,
        description="Variation index from expand_sweep() for sweep children.",
    )
    variation_label: str | None = Field(
        default=None,
        description="Human-readable variation label for sweep children.",
    )
    variation_values: str | None = Field(
        default=None,
        description=(
            "Swept parameter values as a bounded JSON object string, e.g. "
            '{"phases.profiling.concurrency":17} (serializes as '
            "variationValues). Read from the aiperf.nvidia.com/variation-values "
            "annotation. Adaptive planners label variations search_iter_NNNN, "
            "which names the artifact cell but describes nothing, so every "
            "sweep surface needs these values to say what was actually tried. "
            "None for standalone jobs and for archives written before the "
            "children manifest carried the field."
        ),
    )
    trial_index: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Trial index from the aiperf.nvidia.com/trial-index label; only "
            "set on multi-trial sweep children (serializes as trialIndex)."
        ),
    )

    @field_validator("endpoint")
    @classmethod
    def _redact_endpoint(cls, endpoint: str | None) -> str | None:
        """Remove embedded credentials from the public display projection."""
        return redact_url(endpoint) if endpoint is not None else None

    @field_validator("variation_values")
    @classmethod
    def _redact_variation_values(cls, values: str | None) -> str | None:
        """Redact credential-shaped swept parameters at the model boundary.

        A sweep may legitimately sweep ``endpoint.apiKey``. Routers that dump
        this model into a sweep response run the whole payload through
        ``redact_sweep_public_data`` already, but the jobs routes and the CLI
        do not, so the guarantee is anchored here instead of at each caller.
        Idempotent: re-redacting an already-redacted string is a no-op.
        """
        if values is None:
            return None
        return str(redact_sweep_public_data(values, path="variation_values"))

    @property
    def workers_str(self) -> str:
        """Format as 'ready/total'."""
        return f"{self.workers_ready}/{self.workers_total}"


# =============================================================================
# AIPerfSweep CR structure — parsed via AIPerfSweepCR.model_validate(raw_dict)
#
# Mirrors the AIPerfJobCR pattern but for the parent AIPerfSweep CR. The CLI
# resolver builds AIPerfSweepInfo from the raw apiserver response so kube
# commands can decide whether a name refers to a job or a sweep.
# =============================================================================


class CRSweepStatus(K8sCamelModel):
    """AIPerfSweep status subresource (subset relevant for CLI display).

    Authoritative writer is the operator's sweep handler chain — see
    ``operator/handlers/sweep/create.py`` and ``handlers/sweep/child_rollup.py``
    for the canonical field semantics.
    """

    phase: str = Field(default="Pending", description="Current lifecycle phase.")
    run_epoch: int = Field(
        default=0,
        description="Epoch-seconds key of the most recent successful run.",
    )
    total_variations: int = Field(
        default=0,
        description="Total variation cells produced by ``expand_sweep()``.",
    )
    max_total_runs: int = Field(
        default=0,
        description="Upper bound on total child runs (variations * max_trials).",
    )
    completed_runs: int = Field(
        default=0,
        description="Sum of children in a terminal-success phase.",
    )
    failed_runs: int = Field(
        default=0,
        description="Sum of children in a terminal-failure phase.",
    )
    run_states: dict[str, int] = Field(
        default_factory=dict,
        description="Breakdown of child run states: pending, running, completed, failed, cancelled.",
    )
    last_child_event: dict[str, str] | None = Field(
        default=None,
        description="Most recent child phase change: name and phase.",
    )
    current_child_ref: dict[str, Any] | None = Field(
        default=None,
        description="Reference to the currently-active child: name, index, label.",
    )
    api_url: str | None = Field(
        default=None,
        description="API endpoint URL for accessing sweep results and drill-down.",
    )

    @field_validator("phase", mode="before")
    @classmethod
    def coerce_none_phase(cls, v: str | None) -> str:
        """Coerce None or empty phase to 'Pending'."""
        return v or "Pending"


class AIPerfSweepCR(K8sCamelModel):
    """Parsed AIPerfSweep custom resource.

    Use ``AIPerfSweepCR.model_validate(raw_dict)`` to parse a raw K8s API
    response dict, then ``to_info()`` for a flat CLI display model. The
    spec is intentionally not modeled here — sweep spec validation lives
    in :mod:`aiperf.kubernetes.sweep_models` (used by the operator on
    create), and CLI display only needs metadata + status fields.
    """

    metadata: CRMetadata = Field(
        default_factory=CRMetadata, description="K8s object metadata."
    )
    status: CRSweepStatus = Field(
        default_factory=CRSweepStatus, description="Sweep status subresource."
    )

    def to_info(self) -> AIPerfSweepInfo:
        """Convert to flat AIPerfSweepInfo for CLI display."""
        return AIPerfSweepInfo(
            name=self.metadata.name,
            namespace=self.metadata.namespace,
            phase=self.status.phase,
            run_epoch=self.status.run_epoch,
            total_variations=self.status.total_variations,
            max_total_runs=self.status.max_total_runs,
            completed_runs=self.status.completed_runs,
            failed_runs=self.status.failed_runs,
            created=self.metadata.creation_timestamp,
            run_states=self.status.run_states,
            last_child_event=self.status.last_child_event,
            current_child_ref=self.status.current_child_ref,
            api_url=self.status.api_url,
        )


class AIPerfSweepInfo(K8sCamelModel):
    """Flat view of an AIPerfSweep for CLI display.

    Constructed via ``AIPerfSweepCR.model_validate(raw).to_info()`` for
    data from the K8s API, or directly with kwargs for fallback paths.
    """

    name: str = Field(description="AIPerfSweep resource name.")
    namespace: str = Field(
        description="Kubernetes namespace containing the AIPerfSweep."
    )
    phase: str = Field(description="Current lifecycle phase.")
    run_epoch: int = Field(
        default=0,
        description="Epoch-seconds key of the most recent successful run.",
    )
    total_variations: int = Field(
        default=0,
        description="Total variation cells produced by ``expand_sweep()``.",
    )
    max_total_runs: int = Field(
        default=0,
        description="Upper bound on total child runs (variations * max_trials).",
    )
    completed_runs: int = Field(
        default=0,
        description="Sum of children in a terminal-success phase.",
    )
    failed_runs: int = Field(
        default=0,
        description="Sum of children in a terminal-failure phase.",
    )
    created: str = Field(default="", description="Creation timestamp from metadata.")
    run_states: dict[str, int] = Field(
        default_factory=dict,
        description="Breakdown of child run states: pending, running, completed, failed, cancelled.",
    )
    last_child_event: dict[str, str] | None = Field(
        default=None,
        description="Most recent child phase change: name and phase.",
    )
    current_child_ref: dict[str, Any] | None = Field(
        default=None,
        description="Reference to the currently-active child: name, index, label.",
    )
    api_url: str | None = Field(
        default=None,
        description="API endpoint URL for accessing sweep results and drill-down.",
    )


@dataclass
class PodSummary:
    """Summary of pod readiness for a JobSet."""

    ready: int
    """Number of pods with all containers ready and phase Running."""

    total: int
    """Total number of pods belonging to the JobSet."""

    restarts: int
    """Sum of container restart counts across all pods."""

    @property
    def ready_str(self) -> str:
        """Format as 'ready/total'."""
        return f"{self.ready}/{self.total}"
