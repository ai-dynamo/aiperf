# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Energy efficiency metrics — controller-side cross-input analysis.

Computes energy-per-token / per-request / per-watt metrics from already-summarized
GPU telemetry and inference profile results. Runs in SystemController, not in
RecordsManager: the underlying accumulators (GPUTelemetryAccumulator,
MetricsAccumulator) live in separate container processes on K8s and in separate
subprocesses in local multi-process mode, so an in-process AnalyzerProtocol
implementation cannot reach the GPU telemetry data. SystemController is the
fan-in point that already receives both summary payloads via
ProcessTelemetryResultMessage and ProcessRecordsResultMessage.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.models import MetricResult

if TYPE_CHECKING:
    from aiperf.common.models.export_models import TelemetryExportData
    from aiperf.common.models.metric_result_models import ProfileResults

logger = logging.getLogger(__name__)


# Wire-contract tag names. Inference tags are owned by MetricRegistry; the GPU
# telemetry tags are owned by gpu_telemetry/constants.py:41 (energy_consumption
# is also pinned to EnergyMetricUnit.MEGAJOULE there). Hard-coded here so both
# sides break together if either source ever renames; tests pin the expected
# behavior.
_GPU_TELEM_ENERGY_TAG = "energy_consumption"
_GPU_TELEM_POWER_TAG = "gpu_power_usage"
_MJ_TO_J = 1e6


class EnergySource(str, enum.Enum):
    """How total GPU energy was determined."""

    DCGM_COUNTER = "dcgm_counter"
    POWER_INTEGRATION = "power_integration"
    UNAVAILABLE = "unavailable"


def _safe_div(numerator: float | None, denominator: float | None) -> float | None:
    """Return numerator/denominator, or None if either operand is None or denominator <= 0."""
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


@dataclass
class EnergyEfficiencySummary:
    """Typed result from compute_energy_efficiency_from_summaries()."""

    # Source data
    total_gpu_energy_j: float = 0.0
    average_gpu_power_w: float = 0.0
    gpu_count: int = 0
    energy_source: EnergySource = EnergySource.UNAVAILABLE

    # Tier 1
    energy_per_output_token_mj: float | None = None
    energy_per_request_j: float | None = None

    # Tier 2
    energy_per_total_token_mj: float | None = None
    performance_per_watt: float | None = None
    output_tps_per_watt: float | None = None
    goodput_per_watt: float | None = None

    # Metric results for export pipeline
    metric_results: dict[str, MetricResult] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        """Serialize to JSON-compatible structure."""
        data: dict[str, Any] = {
            "source": {
                "total_gpu_energy_j": self.total_gpu_energy_j,
                "average_gpu_power_w": self.average_gpu_power_w,
                "gpu_count": self.gpu_count,
                "energy_source": self.energy_source.value,
            },
            "metrics": {},
        }
        metrics = data["metrics"]
        if self.energy_per_output_token_mj is not None:
            metrics["energy_per_output_token_mj"] = self.energy_per_output_token_mj
        if self.energy_per_request_j is not None:
            metrics["energy_per_request_j"] = self.energy_per_request_j
        if self.energy_per_total_token_mj is not None:
            metrics["energy_per_total_token_mj"] = self.energy_per_total_token_mj
        if self.performance_per_watt is not None:
            metrics["performance_per_watt"] = self.performance_per_watt
        if self.output_tps_per_watt is not None:
            metrics["output_tps_per_watt"] = self.output_tps_per_watt
        if self.goodput_per_watt is not None:
            metrics["goodput_per_watt"] = self.goodput_per_watt
        if self.metric_results:
            data["results"] = [
                asdict(r.to_json_result()) for r in self.metric_results.values()
            ]
        return data

    def to_csv(self) -> list[dict[str, Any]]:
        """Serialize to list of CSV-compatible row dicts."""
        rows: list[dict[str, Any]] = []
        for r in self.metric_results.values():
            row = asdict(r)
            row.pop("current", None)
            rows.append(row)
        return rows


def _get_avg(profile_results: ProfileResults, tag: str) -> float | None:
    """Look up a metric tag's avg from ProfileResults; None if missing or non-positive."""
    result = profile_results.get(tag)
    if result is None or result.avg is None or result.avg <= 0:
        return None
    return result.avg


def compute_energy_efficiency_from_summaries(
    *,
    telemetry: TelemetryExportData | None,
    profile_results: ProfileResults | None,
) -> EnergyEfficiencySummary | None:
    """Compute energy efficiency from already-summarized telemetry + profile results.

    Returns None when inputs are insufficient (no telemetry, no profile results,
    no energy/power readings) — callers write the result to
    ``self._energy_efficiency_results`` only when non-None. This is the single
    compute path; it runs controller-side because the underlying accumulators
    live in separate processes.
    """
    if telemetry is None or profile_results is None:
        return None

    duration_s = (
        (profile_results.end_ns - profile_results.start_ns) / NANOS_PER_SECOND
        if profile_results.end_ns > profile_results.start_ns
        else 0.0
    )
    total_energy_j, avg_power_w, gpu_count, source = _extract_energy_from_summary(
        telemetry, duration_s
    )
    if source is EnergySource.UNAVAILABLE or total_energy_j <= 0:
        return None

    derived = _compute_derived_from_profile(
        profile_results,
        total_energy_j=total_energy_j,
        avg_power_w=avg_power_w,
    )
    metric_results = _build_metric_results(
        total_energy_j=total_energy_j, avg_power_w=avg_power_w, **derived
    )
    return EnergyEfficiencySummary(
        total_gpu_energy_j=total_energy_j,
        average_gpu_power_w=avg_power_w,
        gpu_count=gpu_count,
        energy_source=source,
        metric_results=metric_results,
        **derived,
    )


def _extract_energy_from_summary(
    telemetry: TelemetryExportData,
    duration_s: float,
) -> tuple[float, float, int, EnergySource]:
    """Sum energy + power across published GPUs, with power-integration fallback.

    Prefers the DCGM ``energy_consumption`` counter delta (in MJ, summed across
    GPUs and converted to J) when at least one GPU reports it. Falls back to
    ``total_power_w * duration_s`` when no counter is present but power is —
    same fallback policy the deleted in-process EnergyEfficiencyAnalyzer used.

    The MJ -> J multiplier is hard-coded; the unit is fixed at
    gpu_telemetry/constants.py:41 (EnergyMetricUnit.MEGAJOULE). Both sides
    break together if the published unit ever changes — that's intentional.
    """
    total_energy_mj = 0.0
    total_power_w = 0.0
    gpu_count = 0
    has_counter = False

    for endpoint in telemetry.endpoints.values():
        for gpu in endpoint.gpus.values():
            gpu_count += 1
            energy = gpu.metrics.get(_GPU_TELEM_ENERGY_TAG)
            if energy is not None and energy.avg is not None and energy.avg > 0:
                total_energy_mj += energy.avg
                has_counter = True
            power = gpu.metrics.get(_GPU_TELEM_POWER_TAG)
            if power is not None and power.avg is not None and power.avg > 0:
                total_power_w += power.avg

    if has_counter:
        total_energy_j = total_energy_mj * _MJ_TO_J
        avg_power_w = total_energy_j / max(duration_s, 1e-9)
        return total_energy_j, avg_power_w, gpu_count, EnergySource.DCGM_COUNTER
    if total_power_w > 0 and duration_s > 0:
        return (
            total_power_w * duration_s,
            total_power_w,
            gpu_count,
            EnergySource.POWER_INTEGRATION,
        )
    return 0.0, 0.0, gpu_count, EnergySource.UNAVAILABLE


def _compute_derived_from_profile(
    profile_results: ProfileResults,
    *,
    total_energy_j: float,
    avg_power_w: float,
) -> dict[str, float | None]:
    """Derive per-token / per-watt metrics from inference summary scalars.

    Reads via ``profile_results.get(tag)`` (returns ``MetricResult | None``)
    instead of the ``AccumulatorMetricsSummary.results[tag]`` lookup the
    deleted in-process path used. Same math, different source.
    """
    total_output_tokens = _get_avg(profile_results, "total_osl")
    total_input_tokens = _get_avg(profile_results, "total_isl")
    request_count = _get_avg(profile_results, "request_count")
    request_throughput = _get_avg(profile_results, "request_throughput")
    output_token_throughput = _get_avg(profile_results, "output_token_throughput")
    goodput = _get_avg(profile_results, "goodput")

    energy_per_output_token_mj = _safe_div(total_energy_j * 1000, total_output_tokens)
    energy_per_request_j = _safe_div(total_energy_j, request_count)
    total_tokens = (total_input_tokens or 0) + (total_output_tokens or 0)
    energy_per_total_token_mj = (
        _safe_div(total_energy_j * 1000, total_tokens) if total_tokens > 0 else None
    )
    return {
        "energy_per_output_token_mj": energy_per_output_token_mj,
        "energy_per_request_j": energy_per_request_j,
        "energy_per_total_token_mj": energy_per_total_token_mj,
        "performance_per_watt": (
            _safe_div(request_throughput, avg_power_w)
            if request_throughput is not None and avg_power_w > 0
            else None
        ),
        "output_tps_per_watt": (
            _safe_div(output_token_throughput, avg_power_w)
            if output_token_throughput is not None and avg_power_w > 0
            else None
        ),
        "goodput_per_watt": (
            _safe_div(goodput, avg_power_w)
            if goodput is not None and avg_power_w > 0
            else None
        ),
    }


def _build_metric_results(
    *,
    total_energy_j: float,
    avg_power_w: float,
    energy_per_output_token_mj: float | None,
    energy_per_request_j: float | None,
    energy_per_total_token_mj: float | None,
    performance_per_watt: float | None,
    output_tps_per_watt: float | None,
    goodput_per_watt: float | None,
) -> dict[str, MetricResult]:
    """Build MetricResult objects for each energy metric."""
    results: dict[str, MetricResult] = {}

    def _add(tag: str, header: str, unit: str, value: float | None) -> None:
        if value is None:
            return
        results[tag] = MetricResult(tag=tag, header=header, unit=unit, avg=value)

    _add(
        "energy_per_output_token",
        "Energy Per Output Token",
        "mJ/token",
        energy_per_output_token_mj,
    )
    _add("energy_per_request", "Energy Per Request", "J/req", energy_per_request_j)
    _add("total_gpu_energy", "Total GPU Energy", "J", total_energy_j)
    _add("average_gpu_power", "Average GPU Power", "W", avg_power_w)
    _add(
        "energy_per_total_token",
        "Energy Per Total Token",
        "mJ/token",
        energy_per_total_token_mj,
    )
    _add(
        "performance_per_watt",
        "Performance Per Watt",
        "req/s/W",
        performance_per_watt,
    )
    _add(
        "output_tps_per_watt",
        "Output Tokens Per Second Per Watt",
        "tps/W",
        output_tps_per_watt,
    )
    _add(
        "goodput_per_watt",
        "Goodput Per Watt",
        "good-req/s/W",
        goodput_per_watt,
    )
    return results
