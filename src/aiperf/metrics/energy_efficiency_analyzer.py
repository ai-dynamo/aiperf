# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Energy-efficiency analyzer: the first cross-accumulator ``analyzer`` plugin.

Joins GPU-telemetry energy/power (a live query on the ``GPUTelemetryAccumulator``
over the profiling window) with inference token/throughput/latency totals (read
off the ``MetricsAccumulator`` summary) to emit the energy-efficiency metric
family. Runs at summarize time via the SummaryContext; skipped by RecordsManager
when GPU telemetry is not collected. See design doc ``0005-energy-efficiency-metrics.md``.

Energy source: prefers the DCGM ``energy_consumption`` counter delta; falls back
to power-integration (``fleet_power * duration``) when only the power gauge is
available (e.g. pynvml/amdsmi collectors that expose no energy counter).
"""

from __future__ import annotations

import enum
from typing import TYPE_CHECKING, Any

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.logging import AIPerfLogger
from aiperf.common.models import MetricResult
from aiperf.metrics.types.power_efficiency_metrics import (
    AverageGpuPowerMetric,
    EnergyDelayProductMetric,
    EnergyPerOutputTokenMetric,
    EnergyPerRequestMetric,
    EnergyPerTotalTokenMetric,
    EnergyPerUserMetric,
    GoodputPerWattMetric,
    OutputTokensPerJouleMetric,
    OutputTokensPerSecondPerWattMetric,
    PerformancePerWattMetric,
    TotalGpuEnergyMetric,
    TotalGpuPowerMetric,
)
from aiperf.plugin.enums import AccumulatorType

if TYPE_CHECKING:
    from collections.abc import Callable

    from aiperf.common.accumulator_protocols import SummaryContext
    from aiperf.config.resolution.plan import BenchmarkRun

_logger = AIPerfLogger(__name__)

_MS_PER_SECOND = 1000.0


class EnergySource(str, enum.Enum):
    """How total GPU energy was determined for this run."""

    DCGM_COUNTER = "dcgm_counter"
    POWER_INTEGRATION = "power_integration"
    UNAVAILABLE = "unavailable"


def _result(metric_cls: type, value: float) -> MetricResult:
    """Build the injected MetricResult for an energy metric class."""
    return MetricResult(
        tag=metric_cls.tag,
        header=metric_cls.header,
        unit=str(metric_cls.unit),
        avg=value,
        count=None,
    )


class EnergyEfficiencyAnalyzer:
    """Summarize-time analyzer emitting GPU energy-efficiency metrics.

    Reads the live ``GPUTelemetryAccumulator`` (windowed energy/power) and the
    ``MetricsAccumulator`` summary (token/throughput/latency totals) from the
    SummaryContext, then computes the energy metric family. Each metric is
    emitted only when its inputs are available.
    """

    def __init__(
        self,
        service_id: str | None = None,
        run: BenchmarkRun | None = None,
        pub_client: Any = None,
        **kwargs: Any,
    ) -> None:
        self.run = run

    def _concurrency(self) -> int | None:
        """Positive integer profiling concurrency, or None (rate-only runs)."""
        if self.run is None:
            return None
        phases = self.run.cfg.get_profiling_phases()
        raw = phases[0].concurrency if phases else None
        if isinstance(raw, int) and not isinstance(raw, bool) and raw > 0:
            return raw
        return None

    async def analyze(self, ctx: SummaryContext) -> list[MetricResult]:
        gpu = ctx.get_accumulator(AccumulatorType.GPU_TELEMETRY)
        summary = ctx.get_output(AccumulatorType.METRIC_RESULTS)
        if gpu is None or summary is None:
            return []
        results_by_tag = getattr(summary, "results", None)
        if not isinstance(results_by_tag, dict):
            return []

        def metric(tag: str) -> float | None:
            r = results_by_tag.get(tag)
            return r.avg if r is not None and r.avg is not None else None

        start_ns = ctx.start_ns or None
        end_ns = ctx.end_ns or None
        duration_s = (
            (ctx.end_ns - ctx.start_ns) / NANOS_PER_SECOND
            if ctx.end_ns > ctx.start_ns
            else 0.0
        )
        energy = gpu.total_energy_joules(start_ns, end_ns)
        power = gpu.total_power_watts(start_ns, end_ns)
        total_energy_j, avg_power_w, source = self._resolve_energy(
            energy, power, duration_s
        )

        out: list[MetricResult] = []
        if power[1] > 0:
            out.append(_result(TotalGpuPowerMetric, power[0]))
        if source is EnergySource.UNAVAILABLE or total_energy_j <= 0:
            return out

        out.append(_result(AverageGpuPowerMetric, avg_power_w))
        out.append(_result(TotalGpuEnergyMetric, total_energy_j))
        out += self._energy_ratio_metrics(total_energy_j, metric, self._concurrency())
        out += self._per_watt_metrics(avg_power_w, metric)

        _logger.debug(
            lambda: (
                f"EnergyEfficiencyAnalyzer emitted {len(out)} metrics "
                f"(source={source.value}, energy={total_energy_j:.2f}J, "
                f"avg_power={avg_power_w:.2f}W)"
            )
        )
        return out

    @staticmethod
    def _resolve_energy(
        energy: tuple[float, int],
        power: tuple[float, int],
        duration_s: float,
    ) -> tuple[float, float, EnergySource]:
        """Total energy (J) + average fleet power (W) + how they were determined.

        ``energy``/``power`` are ``(value, gpu_count)`` from the telemetry
        accumulator. Prefers the DCGM energy counter (average power = energy /
        duration); falls back to power-integration (energy = fleet_power *
        duration, average power = fleet_power) when only the power gauge is
        available.
        """
        energy_j, energy_count = energy
        power_w, power_count = power
        if energy_count > 0 and energy_j > 0:
            avg = energy_j / duration_s if duration_s > 0 else 0.0
            return energy_j, avg, EnergySource.DCGM_COUNTER
        if power_count > 0 and power_w > 0 and duration_s > 0:
            return power_w * duration_s, power_w, EnergySource.POWER_INTEGRATION
        return 0.0, 0.0, EnergySource.UNAVAILABLE

    def _energy_ratio_metrics(
        self,
        total_energy_j: float,
        metric: Callable[[str], float | None],
        concurrency: int | None,
    ) -> list[MetricResult]:
        out: list[MetricResult] = []
        output_tokens = metric("total_osl")
        if output_tokens:
            out.append(
                _result(OutputTokensPerJouleMetric, output_tokens / total_energy_j)
            )
            out.append(
                _result(
                    EnergyPerOutputTokenMetric,
                    total_energy_j * _MS_PER_SECOND / output_tokens,
                )
            )
        total_tokens = (metric("total_isl") or 0.0) + (output_tokens or 0.0)
        if total_tokens > 0:
            out.append(
                _result(
                    EnergyPerTotalTokenMetric,
                    total_energy_j * _MS_PER_SECOND / total_tokens,
                )
            )
        energy_per_request_j = None
        if request_count := metric("request_count"):
            energy_per_request_j = total_energy_j / request_count
            out.append(_result(EnergyPerRequestMetric, energy_per_request_j))
        if concurrency is not None:
            out.append(_result(EnergyPerUserMetric, total_energy_j / concurrency))
        latency_ms = metric("request_latency")
        if energy_per_request_j is not None and latency_ms:
            out.append(
                _result(
                    EnergyDelayProductMetric,
                    energy_per_request_j * (latency_ms / _MS_PER_SECOND),
                )
            )
        return out

    @staticmethod
    def _per_watt_metrics(
        avg_power_w: float, metric: Callable[[str], float | None]
    ) -> list[MetricResult]:
        if avg_power_w <= 0:
            return []
        out: list[MetricResult] = []
        if throughput := metric("request_throughput"):
            out.append(_result(PerformancePerWattMetric, throughput / avg_power_w))
        if output_tps := metric("output_token_throughput"):
            out.append(
                _result(OutputTokensPerSecondPerWattMetric, output_tps / avg_power_w)
            )
        if goodput := metric("goodput"):
            out.append(_result(GoodputPerWattMetric, goodput / avg_power_w))
        return out
