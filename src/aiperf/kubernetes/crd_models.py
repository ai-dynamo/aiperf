# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pydantic models for AIPerfJob operator.

This module provides validated models for:
- AIPerfJob spec validation
- Metrics summary extraction
- Results TTL configuration
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any

from pydantic import Field, field_validator, model_validator

from aiperf.common.enums import SweepMode
from aiperf.common.finite import FiniteFloat, is_finite_value
from aiperf.common.models import AIPerfBaseModel
from aiperf.common.types import PhaseKind
from aiperf.config import AIPerfConfig
from aiperf.config.deployment import DeploymentConfig
from aiperf.kubernetes.k8s_models import K8sCamelModel
from aiperf.kubernetes.sweep_models import FailurePolicy, ObjectMetaPartial


class OwnerReference(K8sCamelModel):
    """Kubernetes owner reference for cascade deletion."""

    api_version: str = Field(
        description="The API group and version (e.g. 'aiperf.nvidia.com/v1alpha1')"
    )
    kind: str = Field(description="The kind of the owner resource (e.g. 'AIPerfJob')")
    name: str = Field(description="The name of the owner resource")
    uid: str = Field(description="The UID of the owner resource")
    controller: bool = Field(
        default=True,
        description="Whether this reference points to the managing controller",
    )
    block_owner_deletion: bool = Field(
        default=True,
        description="Whether the owner's deletion is blocked until this resource is removed",
    )

    @classmethod
    def for_aiperf_job(cls, name: str, uid: str) -> OwnerReference:
        """Create an owner reference for an AIPerfJob CR."""
        from aiperf.kubernetes.cr_refs import AIPERF_JOB_API_VERSION

        return cls(
            api_version=AIPERF_JOB_API_VERSION,
            kind="AIPerfJob",
            name=name,
            uid=uid,
        )


@dataclass(slots=True)
class EndpointHealthResult:
    """Result of probing the benchmark target endpoint for reachability.

    Distinct from `aiperf.common.mixins.health_check_mixin.HealthCheckResult`,
    which tracks internal service health.
    """

    reachable: bool
    """Whether the endpoint responded to at least one health probe."""

    error: str
    """Error message if unreachable, empty string otherwise."""


@dataclass(slots=True)
class ControllerFetchResult:
    """Result of fetching metrics and files from the AIPerfJob controller pod."""

    metrics: dict[str, Any] | None
    """Full metrics dict from the controller's /api/metrics endpoint."""

    downloaded: list[str]
    """List of file paths successfully downloaded to the results directory."""

    checkpoints: list[str] = dataclasses.field(default_factory=list)
    """Checkpoint artifact paths downloaded for partial recovery."""

    error: str = ""
    """Error message if fetch failed or returned partial results."""


class PhaseProgress(K8sCamelModel):
    """Progress data for a single benchmark phase."""

    phase_name: str = Field(description="User-provided unique phase name")
    phase_kind: PhaseKind = Field(
        description="Semantic phase role: warmup or profiling"
    )
    phase_index: int | None = Field(
        default=None, ge=0, description="Absolute index in the ordered phase list"
    )
    profiling_index: int | None = Field(
        default=None,
        ge=0,
        description="Index among profiling-kind phases; None for warmup",
    )

    requests_completed: int = Field(
        ge=0, description="Number of requests that received a complete response"
    )
    requests_sent: int = Field(
        ge=0, description="Number of requests dispatched to the endpoint"
    )
    requests_total: int = Field(
        ge=0, description="Total number of requests expected for this phase"
    )
    requests_cancelled: int = Field(
        ge=0, description="Number of requests cancelled before completion"
    )
    requests_errors: int = Field(
        ge=0, description="Number of requests that failed with an error"
    )
    requests_in_flight: int = Field(
        ge=0, description="Number of requests currently awaiting a response"
    )
    requests_per_second: FiniteFloat = Field(description="Current request throughput")
    requests_progress_percent: FiniteFloat = Field(
        description="Percentage of total requests completed (0.0 to 100.0)"
    )
    sessions_sent: int = Field(
        ge=0, description="Number of multi-turn sessions dispatched"
    )
    sessions_completed: int = Field(
        ge=0, description="Number of sessions that finished all turns"
    )
    sessions_cancelled: int = Field(
        ge=0, description="Number of sessions cancelled before completion"
    )
    sessions_in_flight: int = Field(
        ge=0, description="Number of sessions currently in progress"
    )
    records_success: int = Field(
        ge=0, description="Number of individual records completed successfully"
    )
    records_error: int = Field(
        ge=0, description="Number of individual records that failed"
    )
    records_per_second: FiniteFloat = Field(description="Current record throughput")
    records_progress_percent: FiniteFloat = Field(
        description="Percentage of total records completed (0.0 to 100.0)"
    )
    sending_complete: bool = Field(
        description="Whether all requests have been dispatched"
    )
    is_requests_complete: bool = Field(
        description=(
            "True once all expected requests have completed for this phase "
            "(`requests_end_ns` set on CombinedPhaseStats — last response "
            "received). Useful for `kubectl wait` and dashboards."
        ),
    )
    is_records_complete: bool = Field(
        description=(
            "True once the record processor has aggregated all records for "
            "this phase (`records_end_ns` set on CombinedPhaseStats)."
        ),
    )
    timeout_triggered: bool = Field(
        description="Whether the phase ended due to a timeout"
    )
    was_cancelled: bool = Field(
        description="Whether the phase was cancelled by the user"
    )
    requests_eta_seconds: int | None = Field(
        default=None, ge=0, description="Estimated seconds until all requests complete"
    )
    records_eta_seconds: int | None = Field(
        default=None, ge=0, description="Estimated seconds until all records complete"
    )
    expected_duration_seconds: FiniteFloat | None = Field(
        default=None, description="Expected total duration of the phase in seconds"
    )
    elapsed_time_seconds: FiniteFloat | None = Field(
        default=None, description="Wall-clock time elapsed since the phase started"
    )


# Derived scalars that are 0..1 fractions rather than human-scale metric
# values. The 2-decimal display rounding applied to every other float would
# destroy them: a 0.4% error rate would round to exactly 0.0 (hiding real
# failures) and a 4.76% rate would inflate to 5%.
_FRACTION_KEYS: frozenset[str] = frozenset({"error_rate"})
_FRACTION_NDIGITS = 9


def _round_summary(obj: Any, ndigits: int = 2) -> Any:
    """Recursively round floats in a summary dict to ndigits decimal places.

    Keys in :data:`_FRACTION_KEYS` are rounded at :data:`_FRACTION_NDIGITS`
    instead, preserving per-request granularity for 0..1 fractions.
    """
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {
            k: _round_summary(v, _FRACTION_NDIGITS if k in _FRACTION_KEYS else ndigits)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_round_summary(v, ndigits) for v in obj]
    return obj


@dataclass(slots=True)
class MetricsSummary:
    """Filtered nested-dict view of metric tags written to ``status.summary``.

    The CR exposes the full per-tag metrics dict at ``status.liveMetrics.metrics``
    (running) and ``status.results.metrics`` (completed) — keys are AIPerf metric
    tags (``output_token_throughput``, ``request_latency``, ...) and values are
    the metric's full sub-dict (``avg``, ``p50``, ``p99``, ``count``, ``unit``,
    ...). ``status.summary`` and ``status.liveSummary`` are a curated subset of
    those tags written verbatim — same shape, fewer keys — so the UI reads
    every metric through the single path ``summary[tag][stat]``.

    Two derived top-level scalars are bolted on: ``total_requests``
    (successful + failed requests) and ``error_rate`` (fraction of that
    total that failed, 0..1). These aren't AIPerf metric tags; they're
    computed from the raw success/error counts.
    """

    data: dict[str, Any] = dataclasses.field(default_factory=dict)
    """Projected ``{metric_tag: metric_dict}`` plus derived scalars."""

    @classmethod
    def from_metrics(cls, metrics: dict[str, Any] | None) -> MetricsSummary:
        """Project a curated nested view from a full metrics payload.

        Accepts three input shapes:
        1. ``{"metrics": {tag: {...}, ...}, ...}`` — live ``/api/metrics``.
        2. ``{"metrics": [{"tag": ..., ...}, ...], ...}`` — legacy results
           list form.
        3. ``{tag: {...}, ...}`` — top-level tag dict, as written by
           ``profile_export_aiperf.json``.
        """
        if not metrics:
            return cls()
        by_tag = _normalize_to_by_tag(metrics)
        out: dict[str, Any] = {
            tag: by_tag[tag]
            for tag in _SUMMARY_TAGS
            if tag in by_tag and _metric_dict_is_finite(by_tag[tag])
        }
        out.update(_derived_scalars(metrics, by_tag))
        return cls(data=out)

    def to_status_dict(self) -> dict[str, Any]:
        """Return the projected dict for writing to CR status (omits empty)."""
        return _round_summary(self.data)


# Metric tags from the AIPerf metrics payload that we mirror verbatim into
# ``status.summary``. Keep this list aligned with the metric tags emitted by
# the controller's ``/api/metrics`` endpoint and the ``profile_export_aiperf.json``
# results format. New metrics surface in summary by adding their tag here.
#
# Excluded by policy (see metrics/types/*.py for the source-of-truth flags):
#   - ``MetricFlags.INTERNAL`` (credit_drop_latency, requested_osl,
#     min_request_timestamp, max_response_timestamp) — implementation
#     detail, not user-facing.
#   - ``MetricFlags.EXPERIMENTAL`` (stream_setup_latency, stream_prefill_latency,
#     thinking_efficiency, overall_thinking_efficiency) — schema not yet
#     stable, gated by --enable-experimental-metrics.
_SUMMARY_TAGS: tuple[str, ...] = (
    # Throughput family
    "request_throughput",
    "output_token_throughput",
    "total_token_throughput",
    "output_token_throughput_per_user",
    "prefill_throughput_per_user",
    "e2e_output_token_throughput",
    "image_throughput",
    "goodput",
    # Latency family
    "request_latency",
    "time_to_first_token",
    "time_to_first_output_token",
    "time_to_second_token",
    "inter_token_latency",
    "inter_chunk_latency",
    "image_latency",
    # Tokens / sequence lengths
    "input_sequence_length",
    "output_sequence_length",
    "output_token_count",
    "reasoning_token_count",
    "error_isl",
    "osl_mismatch_diff_pct",
    "osl_mismatch_count",
    "usage_prompt_tokens",
    "usage_completion_tokens",
    "usage_total_tokens",
    "usage_reasoning_tokens",
    "usage_prompt_tokens_diff_pct",
    "usage_completion_tokens_diff_pct",
    "usage_reasoning_tokens_diff_pct",
    "usage_discrepancy_count",
    # Counts & totals
    "request_count",
    "good_request_count",
    "error_request_count",
    "total_isl",
    "total_osl",
    "total_error_isl",
    "total_output_tokens",
    "total_usage_prompt_tokens",
    "total_usage_completion_tokens",
    "total_usage_total_tokens",
    "total_reasoning_tokens",
    "benchmark_duration",
    "num_images",
    # Video
    "video_inference_time",
    "video_peak_memory",
    # HTTP trace (only present when --collect-http-traces is enabled)
    "http_req_duration",
    "http_req_total",
    "http_req_waiting",
    "http_req_blocked",
    "http_req_connecting",
    "http_req_dns_lookup",
    "http_req_sending",
    "http_req_receiving",
    "http_req_connection_overhead",
    "http_req_connection_reused",
    "http_req_chunks_sent",
    "http_req_chunks_received",
    "http_req_data_sent",
    "http_req_data_received",
)


def _normalize_to_by_tag(metrics: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Coerce any of the three accepted metrics shapes into a tag-keyed dict.

    The live API delivers ``{"metrics": {tag: {...}}}``, the legacy results
    file delivers ``{"metrics": [{"tag": ..., ...}]}``, and
    ``profile_export_aiperf.json`` delivers metric tags at the top level.
    """
    raw: Any = metrics.get("metrics")
    if raw is None:
        raw = metrics
    if isinstance(raw, dict):
        return {k: v for k, v in raw.items() if isinstance(v, dict)}
    if isinstance(raw, list):
        out: dict[str, dict[str, Any]] = {}
        for m in raw:
            if not isinstance(m, dict):
                continue
            tag = m.get("tag")
            if isinstance(tag, str):
                out[tag] = m
        return out
    return {}


def _metric_dict_is_finite(metric: dict[str, Any]) -> bool:
    """Return True unless a numeric stat in the metric dict is NaN/inf.

    A summary tag whose headline stats carry non-finite values must not be
    mirrored into ``status.summary`` — orjson would serialize NaN/inf as
    JSON ``null`` and the dashboard would render garbage. Non-numeric values
    (``unit`` strings, nested dicts) are ignored; only actual float/int stats
    gate the tag. An ``avg`` of ``inf`` drops the whole tag rather than
    writing a half-scrubbed dict.
    """
    for value in metric.values():
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and not is_finite_value(value)
        ):
            return False
    return True


def _derived_scalars(
    metrics: dict[str, Any], by_tag: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Compute the two derived top-level scalars: ``total_requests`` and ``error_rate``.

    ``total_requests`` is *successes + errors*. ``request_count`` counts only
    successful requests (see :class:`RequestCountMetric`) and
    ``error_request_count`` counts only failures, so their sum is the grand
    total the export and console report — using ``request_count`` alone would
    undercount by every failed request.

    ``error_rate`` is the fraction (0..1) of that grand total that failed.
    When the authoritative ``request_error_rate`` metric tag (a percent,
    ``100 * errors / total``) is present it is mirrored verbatim (``rate/100``)
    so ``status.summary`` agrees with the export exactly; otherwise it falls
    back to ``errors / total``.

    Prefers the live per-tag counts; falls back to the top-level scalars a
    round-tripped ``status.summary`` / ``profile_export_aiperf.json`` keeps for
    completed runs. A fully-failed run (``request_count`` absent, only
    ``error_request_count`` present) still reports its total and a 1.0 error
    rate. A counter that is *absent* contributes 0; a counter that is *present
    but non-finite* (e.g. an ``error_request_count`` of NaN) is treated as
    unknown, so ``error_rate`` is dropped rather than fabricated from a
    NaN-as-zero.
    """
    out: dict[str, Any] = {}
    rc = (by_tag.get("request_count") or {}).get("avg")
    ec = (by_tag.get("error_request_count") or {}).get("avg")
    rer = (by_tag.get("request_error_rate") or {}).get("avg")
    # Distinguish "absent" (None -> counts as 0) from "present but non-finite"
    # (NaN/inf -> unknown): a NaN error count must not surface as a bogus 0.0
    # error rate that hides real failures.
    rc_bad = rc is not None and not is_finite_value(rc)
    ec_bad = ec is not None and not is_finite_value(ec)
    successes = float(rc) if is_finite_value(rc) else 0.0
    errors = float(ec) if is_finite_value(ec) else 0.0
    total = successes + errors
    if total > 0:
        out["total_requests"] = int(total)
        if is_finite_value(rer):
            out["error_rate"] = float(rer) / 100.0
        elif not rc_bad and not ec_bad:
            out["error_rate"] = errors / total
    if "error_rate" not in out and is_finite_value(metrics.get("error_rate")):
        out["error_rate"] = float(metrics["error_rate"])
    if "total_requests" not in out and is_finite_value(metrics.get("request_count")):
        out["total_requests"] = int(metrics["request_count"])
    return out


class K8sEndpointConfig(AIPerfBaseModel):
    """Validated endpoint configuration for the AIPerfJob CRD.

    Distinct from :class:`aiperf.config.endpoint.EndpointConfig`, which is the
    full benchmark-side endpoint config. This one models the reduced CRD-spec
    shape the operator validates before creating resources.
    """

    __slots__ = ()

    url: str = Field(description="LLM endpoint URL")
    model: str | None = Field(default=None, description="Model name")
    api_type: str = Field(default="openai", description="API type (openai, triton)")

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Validate URL format."""
        if not v:
            raise ValueError(f"Endpoint URL is required, got {v!r}")
        if not v.startswith(("http://", "https://")):
            raise ValueError(
                f"Endpoint URL must start with http:// or https://, got {v!r}"
            )
        return v


class AIPerfWorkloadSpec(AIPerfConfig, DeploymentConfig):
    """Shared base for AIPerfJob and AIPerfSweep CRD specs.

    Composes the AIPerf YAML envelope (`benchmark`, `sweep`, `multi_run`,
    `variables`, `random_seed`) with the K8s deployment surface
    (image, podTemplate, scheduling, …) and adds the orchestration fields
    used by both kinds (`skip_endpoint_check`, `failure_policy`, `cancel`,
    `ttl_seconds_after_finished`). Subclasses pin the kind-specific
    `spec.sweep` cardinality via a `model_validator(mode="after")`.

    `cancel` and `ttl_seconds_after_finished` are inherited from
    DeploymentConfig; `skip_endpoint_check` and `failure_policy` are added
    here.
    """

    skip_endpoint_check: bool = Field(
        default=False,
        description="Skip the operator-side endpoint reachability probe before deploying.",
    )

    failure_policy: FailurePolicy = Field(
        default_factory=FailurePolicy,
        description=(
            "Failure handling policy. On AIPerfJob this governs the single "
            "benchmark; on AIPerfSweep it governs whether a failed child "
            "aborts the sweep or advances to the next variation."
        ),
    )

    @field_validator("image")
    @classmethod
    def _validate_image_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError(
                f"Image is required (got {v!r}); set image.repository and image.tag or pass --image."
            )
        return v


class AIPerfJobSpec(AIPerfWorkloadSpec):
    """Validated AIPerfJob spec mirroring the full CRD spec.

    Carries a single benchmark (`spec.benchmark`); `spec.sweep` MUST be
    null — for sweeps, use AIPerfSweep. Validate a raw CRD dict via
    ``AIPerfJobSpec.model_validate(spec)``; ``from_crd_spec`` is a thin
    back-compat alias.
    """

    @model_validator(mode="after")
    def _reject_orchestration_on_aiperfjob(self) -> AIPerfJobSpec:
        if self.sweep is not None:
            raise ValueError(
                "AIPerfJob.spec.sweep must be null; sweep CRs are AIPerfSweep. "
                "Move the `sweep:` block to an AIPerfSweep CR or drop it."
            )
        if self.multi_run.num_runs > 1 or self.multi_run.convergence is not None:
            raise ValueError(
                "AIPerfJob.spec.multiRun must describe one run without "
                "convergence; use AIPerfSweep for multi-run orchestration."
            )
        return self

    @classmethod
    def from_crd_spec(cls, spec: dict[str, Any]) -> AIPerfJobSpec:
        """Validate a raw CRD spec dict (back-compat alias for model_validate)."""
        return cls.model_validate(spec)

    def get_endpoint_url(self) -> str | None:
        """Extract primary endpoint URL from benchmark.endpoint."""
        endpoint = self.benchmark.endpoint
        if endpoint is None:
            return None
        url = getattr(endpoint, "url", None)
        if url:
            return url
        urls = getattr(endpoint, "urls", None) or []
        return urls[0] if urls else None


class AIPerfSweepSpec(AIPerfWorkloadSpec):
    """Validated AIPerfSweep spec; requires `spec.sweep` to be set.

    The sweep block drives variation generation; the rest of the workload
    body (`benchmark`, `multi_run`, deployment fields) acts as the per-child
    template. ``from_crd_spec`` is a thin back-compat alias for
    ``model_validate``.
    """

    child_metadata: ObjectMetaPartial | None = Field(
        default=None,
        description=(
            "Optional labels and annotations stamped onto every child "
            "AIPerfJob created by this sweep. Sweep-tracking labels "
            "(aiperf.nvidia.com/sweep, sweep-uid, sweep-run-epoch, "
            "variation-index, variation-label, trial-index) are reserved "
            "and override any user-supplied values with the same key. "
            "Only present on AIPerfSweep — AIPerfJob has no children."
        ),
    )

    @model_validator(mode="after")
    def _require_sweep_on_aiperfsweep(self) -> AIPerfSweepSpec:
        if self.sweep is None:
            raise ValueError(
                "AIPerfSweep.spec.sweep is required; set a `sweep:` block "
                "(grid or scenarios). For a single benchmark, use AIPerfJob."
            )
        return self

    @model_validator(mode="after")
    def _reject_non_finite_sweep_knobs(self) -> AIPerfSweepSpec:
        """Reject NaN/inf on scalar sweep knobs at the CRD-spec boundary.

        The underlying sweep-config models (``aiperf.config.sweep.config``)
        type these as plain ``float`` with ``ge=0``/``gt=0`` bounds, which inf
        satisfies (``inf >= 0`` is True) — so a non-finite ``cooldownSeconds``,
        ``plateauThreshold``, or ``slaWarmupSeconds`` would otherwise survive
        into ``status``/the CRD where orjson coerces it to JSON ``null``.
        Validate here so an AIPerfSweep CR with a non-finite knob is rejected
        before the operator ever acts on it.
        """
        if self.sweep is None:
            return self
        for knob in ("cooldown_seconds", "plateau_threshold", "sla_warmup_seconds"):
            value = getattr(self.sweep, knob, None)
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"sweep.{knob} must be finite, got {value!r}")
        return self

    @model_validator(mode="after")
    def _reject_repeated_iteration_with_convergence(self) -> AIPerfSweepSpec:
        if self.sweep is None or self.multi_run.convergence is None:
            return self
        iteration_order = getattr(self.sweep, "iteration_order", None)
        if iteration_order == SweepMode.REPEATED:
            raise ValueError(
                "iteration_order='repeated' is incompatible with adaptive trial "
                "convergence (multi_run.convergence). Use 'independent' instead."
            )
        return self

    @classmethod
    def from_crd_spec(cls, spec: dict[str, Any]) -> AIPerfSweepSpec:
        """Validate a raw CRD spec dict (back-compat alias for model_validate)."""
        return cls.model_validate(spec)
