# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public data models for steady-state windowed metrics."""

from __future__ import annotations

import dataclasses
from dataclasses import asdict
from typing import Any

from pydantic import Field

from aiperf.common.models import MetricResult
from aiperf.common.models.base_models import AIPerfBaseModel
from aiperf.common.types import MetricTagT


def _to_dict(obj: Any) -> dict[str, Any]:
    """Polymorphic dict-ifier: handles slotted dataclasses and Pydantic models.

    ``to_json_result()`` returns the slotted ``MetricResult`` for fast-path
    metrics but the Pydantic ``JsonMetricResult`` shape for sweep-injected
    metrics. Both need to flatten through ``asdict`` semantics for the JSON
    exporter.
    """
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return dict(obj)


class SteadyStateWindowMetadata(AIPerfBaseModel, frozen=True):
    """Diagnostic metadata about the detected steady-state window."""

    ramp_up_end_ns: float = Field(description="Timestamp (ns) when ramp-up ends")
    ramp_down_start_ns: float = Field(
        description="Timestamp (ns) when ramp-down starts"
    )
    steady_state_duration_ns: float = Field(
        description="Duration of the steady-state window in nanoseconds"
    )
    total_requests: int = Field(description="Total requests in the benchmark")
    steady_state_requests: int = Field(
        description="Requests within the steady-state window"
    )
    detection_method: str = Field(description="Method used to detect steady state")
    fraction_retained: float = Field(
        description="Fraction of total requests retained in the steady-state window"
    )

    # Sample quality
    variance_inflation_factor: float = Field(
        description="Approximate variance inflation from truncation: total_requests / steady_state_requests",
    )
    effective_p99_sample_size: int = Field(
        description="Approximate observations contributing to p99: int(steady_state_requests * 0.01)",
    )
    sample_size_warning: bool = Field(
        default=False,
        description="True if effective p99 sample size < 10 (p99 estimate unreliable)",
    )

    # Stationarity validation
    trend_correlation: float | None = Field(
        default=None,
        description="Spearman rank correlation of batch means (latency trend test)",
    )
    trend_p_value: float | None = Field(
        default=None,
        description="P-value of the batch means trend test",
    )
    stationarity_warning: bool = Field(
        default=False,
        description="True if windowed latency shows a statistically significant trend",
    )

    # Per-signal boundaries (diagnostic)
    cusum_ramp_up_end_ns: float | None = Field(
        default=None, description="CUSUM-detected ramp-up end timestamp (ns)"
    )
    cusum_ramp_down_start_ns: float | None = Field(
        default=None, description="CUSUM-detected ramp-down start timestamp (ns)"
    )
    mser5_latency_ramp_up_end_ns: float | None = Field(
        default=None,
        description="MSER-5 latency-detected ramp-up end timestamp (ns)",
    )
    mser5_latency_ramp_down_start_ns: float | None = Field(
        default=None,
        description="MSER-5 latency-detected ramp-down start timestamp (ns)",
    )
    mser5_ttft_ramp_up_end_ns: float | None = Field(
        default=None,
        description="MSER-5 TTFT-detected ramp-up end timestamp (ns)",
    )
    mser5_ttft_ramp_down_start_ns: float | None = Field(
        default=None,
        description="MSER-5 TTFT-detected ramp-down start timestamp (ns)",
    )
    cusum_throughput_ramp_up_end_ns: float | None = Field(
        default=None,
        description="CUSUM throughput-detected ramp-up end timestamp (ns)",
    )
    cusum_throughput_ramp_down_start_ns: float | None = Field(
        default=None,
        description="CUSUM throughput-detected ramp-down start timestamp (ns)",
    )

    # Bootstrap confidence intervals (optional, only when bootstrap_iterations is set)
    bootstrap_ci_ramp_up_ns: tuple[float, float] | None = Field(
        default=None,
        description="Bootstrap confidence interval for ramp-up boundary (ns)",
    )
    bootstrap_ci_ramp_down_ns: tuple[float, float] | None = Field(
        default=None,
        description="Bootstrap confidence interval for ramp-down boundary (ns)",
    )
    bootstrap_ci_mean_latency: tuple[float, float] | None = Field(
        default=None,
        description="Bootstrap confidence interval for mean latency within window",
    )
    bootstrap_ci_p99_latency: tuple[float, float] | None = Field(
        default=None,
        description="Bootstrap confidence interval for p99 latency within window",
    )
    bootstrap_n_iterations: int | None = Field(
        default=None,
        description="Number of bootstrap iterations performed",
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to structured JSON-ready dictionary."""
        data: dict[str, Any] = {
            "detection_method": self.detection_method,
            "ramp_up_end_ns": self.ramp_up_end_ns,
            "ramp_down_start_ns": self.ramp_down_start_ns,
            "steady_state_duration_ns": self.steady_state_duration_ns,
            "total_requests": self.total_requests,
            "steady_state_requests": self.steady_state_requests,
            "quality": {
                "fraction_retained": self.fraction_retained,
                "variance_inflation_factor": self.variance_inflation_factor,
                "effective_p99_sample_size": self.effective_p99_sample_size,
                "sample_size_warning": self.sample_size_warning,
            },
            "stationarity": {
                "trend_correlation": self.trend_correlation,
                "trend_p_value": self.trend_p_value,
                "stationarity_warning": self.stationarity_warning,
            },
            "cross_validation": {
                "cusum_ramp_up_end_ns": self.cusum_ramp_up_end_ns,
                "cusum_ramp_down_start_ns": self.cusum_ramp_down_start_ns,
                "mser5_latency_ramp_up_end_ns": self.mser5_latency_ramp_up_end_ns,
                "mser5_latency_ramp_down_start_ns": self.mser5_latency_ramp_down_start_ns,
                "mser5_ttft_ramp_up_end_ns": self.mser5_ttft_ramp_up_end_ns,
                "mser5_ttft_ramp_down_start_ns": self.mser5_ttft_ramp_down_start_ns,
                "cusum_throughput_ramp_up_end_ns": self.cusum_throughput_ramp_up_end_ns,
                "cusum_throughput_ramp_down_start_ns": self.cusum_throughput_ramp_down_start_ns,
            },
        }
        if self.bootstrap_n_iterations is not None:
            data["bootstrap"] = {
                "n_iterations": self.bootstrap_n_iterations,
                "ci_ramp_up_ns": self.bootstrap_ci_ramp_up_ns,
                "ci_ramp_down_ns": self.bootstrap_ci_ramp_down_ns,
                "ci_mean_latency": self.bootstrap_ci_mean_latency,
                "ci_p99_latency": self.bootstrap_ci_p99_latency,
            }
        return data


class SteadyStateSummary(AIPerfBaseModel):
    """Typed result from SteadyStateAnalyzer.summarize()."""

    results: dict[MetricTagT, MetricResult] = Field(
        description="Metric results within the steady-state window"
    )
    effective_concurrency: MetricResult = Field(
        description="Time-weighted concurrency statistics during steady state"
    )
    effective_throughput: MetricResult = Field(
        description="Time-weighted throughput statistics during steady state"
    )
    effective_prefill_throughput: MetricResult = Field(
        description="Time-weighted prefill throughput statistics during steady state"
    )
    effective_generation_concurrency: MetricResult = Field(
        description="Time-weighted generation-phase concurrency statistics during steady state"
    )
    effective_prefill_concurrency: MetricResult = Field(
        description="Time-weighted prefill-phase concurrency statistics during steady state"
    )
    effective_total_throughput: MetricResult = Field(
        description="Time-weighted total throughput (prefill + generation) during steady state"
    )
    effective_throughput_per_user: MetricResult = Field(
        description="Time-weighted per-user throughput statistics during steady state"
    )
    effective_prefill_throughput_per_user: MetricResult = Field(
        description="Time-weighted per-user prefill throughput statistics during steady state"
    )
    tokens_in_flight: MetricResult = Field(
        description="Time-weighted tokens in flight (GPU memory/compute pressure) during steady state"
    )
    window_metadata: SteadyStateWindowMetadata = Field(
        description="Metadata about the detected steady-state window"
    )

    @property
    def sweep_metrics(self) -> dict[str, MetricResult]:
        """Return all sweep MetricResults keyed by tag."""
        return {
            self.effective_concurrency.tag: self.effective_concurrency,
            self.effective_throughput.tag: self.effective_throughput,
            self.effective_prefill_throughput.tag: self.effective_prefill_throughput,
            self.effective_generation_concurrency.tag: self.effective_generation_concurrency,
            self.effective_prefill_concurrency.tag: self.effective_prefill_concurrency,
            self.effective_total_throughput.tag: self.effective_total_throughput,
            self.effective_throughput_per_user.tag: self.effective_throughput_per_user,
            self.effective_prefill_throughput_per_user.tag: self.effective_prefill_throughput_per_user,
            self.tokens_in_flight.tag: self.tokens_in_flight,
        }

    def to_json(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "results": [_to_dict(r.to_json_result()) for r in self.results.values()],
            "effective_concurrency": _to_dict(
                self.effective_concurrency.to_json_result()
            ),
            "effective_throughput": _to_dict(
                self.effective_throughput.to_json_result()
            ),
            "effective_prefill_throughput": _to_dict(
                self.effective_prefill_throughput.to_json_result()
            ),
            "effective_generation_concurrency": _to_dict(
                self.effective_generation_concurrency.to_json_result()
            ),
            "effective_prefill_concurrency": _to_dict(
                self.effective_prefill_concurrency.to_json_result()
            ),
            "effective_total_throughput": _to_dict(
                self.effective_total_throughput.to_json_result()
            ),
            "effective_throughput_per_user": _to_dict(
                self.effective_throughput_per_user.to_json_result()
            ),
            "effective_prefill_throughput_per_user": _to_dict(
                self.effective_prefill_throughput_per_user.to_json_result()
            ),
            "tokens_in_flight": _to_dict(self.tokens_in_flight.to_json_result()),
            "window_metadata": self.window_metadata.to_dict(),
        }
        return data

    def to_csv(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for r in self.results.values():
            row = _to_dict(r)
            row.pop("current", None)
            rows.append(row)
        return rows
