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

Vendor fan-out: the analyzer calls ``gpu.available_platforms()`` and emits
``nvidia_*`` metrics for NVIDIA GPUs and ``amd_*`` metrics for AMD GPUs.
Mixed-vendor runs emit both families independently.
"""

from __future__ import annotations

import enum
from typing import TYPE_CHECKING, Any

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.logging import AIPerfLogger
from aiperf.common.models import MetricResult
from aiperf.gpu_telemetry.constants import (
    AMD_GPU_TELEMETRY_PLATFORM,
    NVIDIA_GPU_TELEMETRY_PLATFORM,
)
from aiperf.metrics.types.power_efficiency_metrics import (
    AmdAverageGpuPowerMetric,
    AmdEnergyDelayProductMetric,
    AmdEnergyPerOutputTokenMetric,
    AmdEnergyPerRequestMetric,
    AmdEnergyPerTotalTokenMetric,
    AmdEnergyPerUserMetric,
    AmdGoodputPerWattMetric,
    AmdOutputTokensPerJouleMetric,
    AmdOutputTokensPerSecondPerWattMetric,
    AmdPerformancePerWattMetric,
    AmdTotalGpuEnergyMetric,
    AmdTotalGpuPowerMetric,
    NvidiaAverageGpuPowerMetric,
    NvidiaEnergyDelayProductMetric,
    NvidiaEnergyPerOutputTokenMetric,
    NvidiaEnergyPerRequestMetric,
    NvidiaEnergyPerTotalTokenMetric,
    NvidiaEnergyPerUserMetric,
    NvidiaGoodputPerWattMetric,
    NvidiaOutputTokensPerJouleMetric,
    NvidiaOutputTokensPerSecondPerWattMetric,
    NvidiaPerformancePerWattMetric,
    NvidiaTotalGpuEnergyMetric,
    NvidiaTotalGpuPowerMetric,
)
from aiperf.plugin.enums import AccumulatorType

if TYPE_CHECKING:
    from collections.abc import Callable

    from aiperf.common.accumulator_protocols import SummaryContext
    from aiperf.config.resolution.plan import BenchmarkRun

_logger = AIPerfLogger(__name__)

_MS_PER_SECOND = 1000.0

_VENDOR_METRICS: dict[str, dict[str, type]] = {
    NVIDIA_GPU_TELEMETRY_PLATFORM: {
        "total_power": NvidiaTotalGpuPowerMetric,
        "average_power": NvidiaAverageGpuPowerMetric,
        "total_energy": NvidiaTotalGpuEnergyMetric,
        "output_tokens_per_joule": NvidiaOutputTokensPerJouleMetric,
        "energy_per_output_token": NvidiaEnergyPerOutputTokenMetric,
        "energy_per_total_token": NvidiaEnergyPerTotalTokenMetric,
        "energy_per_request": NvidiaEnergyPerRequestMetric,
        "energy_per_user": NvidiaEnergyPerUserMetric,
        "energy_delay_product": NvidiaEnergyDelayProductMetric,
        "performance_per_watt": NvidiaPerformancePerWattMetric,
        "output_tps_per_watt": NvidiaOutputTokensPerSecondPerWattMetric,
        "goodput_per_watt": NvidiaGoodputPerWattMetric,
    },
    AMD_GPU_TELEMETRY_PLATFORM: {
        "total_power": AmdTotalGpuPowerMetric,
        "average_power": AmdAverageGpuPowerMetric,
        "total_energy": AmdTotalGpuEnergyMetric,
        "output_tokens_per_joule": AmdOutputTokensPerJouleMetric,
        "energy_per_output_token": AmdEnergyPerOutputTokenMetric,
        "energy_per_total_token": AmdEnergyPerTotalTokenMetric,
        "energy_per_request": AmdEnergyPerRequestMetric,
        "energy_per_user": AmdEnergyPerUserMetric,
        "energy_delay_product": AmdEnergyDelayProductMetric,
        "performance_per_watt": AmdPerformancePerWattMetric,
        "output_tps_per_watt": AmdOutputTokensPerSecondPerWattMetric,
        "goodput_per_watt": AmdGoodputPerWattMetric,
    },
}


class EnergySource(str, enum.Enum):
    """How total GPU energy was determined for this run."""

    DCGM_COUNTER = "dcgm_counter"
    POWER_INTEGRATION = "power_integration"
    UNAVAILABLE = "unavailable"


def _result(metric_cls: type, value: float) -> MetricResult:
    """Build the injected MetricResult for an energy metric class.

    ``value`` is emitted in ``metric_cls.unit`` verbatim: analyzer output bypasses
    ``MetricsAccumulator._convert_display_units``, so any ``display_unit`` on one of
    these classes would be silently ignored. Keep ``display_unit == unit`` (i.e.
    don't declare a ``display_unit``) on the power_efficiency_metrics classes, or
    route this through unit conversion first.
    """
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
    SummaryContext, then computes the energy metric family per vendor. Each metric
    is emitted only when its inputs are available; vendors with no telemetry data
    contribute nothing.
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
        # DCGM average power = energy / duration. The energy numerator integrates
        # over [start, end + FINAL_SCRAPE_GRACE_NS) to capture the trailing scrape,
        # while this denominator is the exact profiling window, so avg power is
        # biased slightly high (~grace/duration; negligible past a few seconds, up
        # to a few percent on very short runs). Exact alignment would require the
        # accumulator to expose the actual first/last scrape timestamps.
        duration_s = (
            (ctx.end_ns - ctx.start_ns) / NANOS_PER_SECOND
            if ctx.end_ns > ctx.start_ns
            else 0.0
        )
        # An unbounded (full-range) window leaves start==end==0 -> duration_s==0,
        # which would drop average power and every per-watt metric while total
        # energy still emits (a half-populated family). Recover a real duration
        # from the observed telemetry scrape span so the whole family stays
        # consistent. A genuinely empty accumulator still yields 0.0.
        if duration_s <= 0.0:
            span = gpu.scrape_span_ns()
            if span is not None and span[1] > span[0]:
                duration_s = (span[1] - span[0]) / NANOS_PER_SECOND

        platforms = gpu.available_platforms()
        concurrency = self._concurrency()
        out: list[MetricResult] = []
        for platform in sorted(platforms):
            vendor_metrics = _VENDOR_METRICS.get(platform)
            if vendor_metrics is None:
                _logger.debug(
                    lambda p=platform: f"EnergyEfficiencyAnalyzer: skipping unmapped platform '{p}' — no entry in _VENDOR_METRICS"
                )
                continue
            vendor_out = self._analyze_vendor(
                gpu=gpu,
                start_ns=start_ns,
                end_ns=end_ns,
                duration_s=duration_s,
                metric=metric,
                concurrency=concurrency,
                vendor_metrics=vendor_metrics,
                platform=platform,
            )
            out.extend(vendor_out)
        return out

    def _analyze_vendor(
        self,
        *,
        gpu: Any,
        start_ns: int | None,
        end_ns: int | None,
        duration_s: float,
        metric: Callable[[str], float | None],
        concurrency: int | None,
        vendor_metrics: dict[str, type],
        platform: str,
    ) -> list[MetricResult]:
        energy = gpu.total_energy_joules(start_ns, end_ns, platform=platform)
        power = gpu.total_power_watts(start_ns, end_ns, platform=platform)
        total_energy_j, avg_power_w, source = self._resolve_energy(
            energy, power, duration_s
        )

        out: list[MetricResult] = []
        if power[1] > 0:
            out.append(_result(vendor_metrics["total_power"], power[0]))
        if source is EnergySource.UNAVAILABLE or total_energy_j <= 0:
            return out

        # A degenerate/empty profiling window yields duration_s == 0, so the DCGM
        # branch cannot derive an average power; emit total energy but skip the
        # misleading 0 W average (and _per_watt_metrics likewise returns []).
        if avg_power_w > 0:
            out.append(_result(vendor_metrics["average_power"], avg_power_w))
        out.append(_result(vendor_metrics["total_energy"], total_energy_j))
        out += self._energy_ratio_metrics(
            total_energy_j, metric, concurrency, vendor_metrics
        )
        out += self._per_watt_metrics(avg_power_w, metric, vendor_metrics)

        _logger.debug(
            lambda: (
                f"EnergyEfficiencyAnalyzer emitted {len(out)} metrics for {platform} "
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

    @staticmethod
    def _energy_ratio_metrics(
        total_energy_j: float,
        metric: Callable[[str], float | None],
        concurrency: int | None,
        vendor_metrics: dict[str, type],
    ) -> list[MetricResult]:
        out: list[MetricResult] = []
        output_tokens = metric("total_osl")
        if output_tokens:
            out.append(
                _result(
                    vendor_metrics["output_tokens_per_joule"],
                    output_tokens / total_energy_j,
                )
            )
            out.append(
                _result(
                    vendor_metrics["energy_per_output_token"],
                    total_energy_j * _MS_PER_SECOND / output_tokens,
                )
            )
        total_tokens = (metric("total_isl") or 0.0) + (output_tokens or 0.0)
        if total_tokens > 0:
            out.append(
                _result(
                    vendor_metrics["energy_per_total_token"],
                    total_energy_j * _MS_PER_SECOND / total_tokens,
                )
            )
        energy_per_request_j = None
        if request_count := metric("request_count"):
            energy_per_request_j = total_energy_j / request_count
            out.append(
                _result(vendor_metrics["energy_per_request"], energy_per_request_j)
            )
        if concurrency is not None:
            out.append(
                _result(vendor_metrics["energy_per_user"], total_energy_j / concurrency)
            )
        latency_ms = metric("request_latency")
        if energy_per_request_j is not None and latency_ms:
            out.append(
                _result(
                    vendor_metrics["energy_delay_product"],
                    energy_per_request_j * (latency_ms / _MS_PER_SECOND),
                )
            )
        return out

    @staticmethod
    def _per_watt_metrics(
        avg_power_w: float,
        metric: Callable[[str], float | None],
        vendor_metrics: dict[str, type],
    ) -> list[MetricResult]:
        if avg_power_w <= 0:
            return []
        out: list[MetricResult] = []
        # Compare to None, not truthiness: a genuine 0.0 throughput/goodput is a
        # valid per-watt result (e.g. goodput_per_watt == 0 when no request met the
        # SLO) and must be emitted, not silently dropped. avg_power_w > 0 is
        # guaranteed above, so none of these divisions is degenerate.
        throughput = metric("request_throughput")
        if throughput is not None:
            out.append(
                _result(
                    vendor_metrics["performance_per_watt"], throughput / avg_power_w
                )
            )
        output_tps = metric("output_token_throughput")
        if output_tps is not None:
            out.append(
                _result(vendor_metrics["output_tps_per_watt"], output_tps / avg_power_w)
            )
        goodput = metric("goodput")
        if goodput is not None:
            out.append(
                _result(vendor_metrics["goodput_per_watt"], goodput / avg_power_w)
            )
        return out
