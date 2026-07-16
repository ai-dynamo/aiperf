# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Energy-efficiency analyzer: the first cross-accumulator ``analyzer`` plugin.

Joins GPU-telemetry energy/power (a live query on the ``GPUTelemetryAccumulator``
over the profiling window) with inference token/throughput/latency totals (read
off the ``MetricsAccumulator`` summary) to emit the energy-efficiency metric
family. Runs at summarize time via the SummaryContext; skipped by RecordsManager
when GPU telemetry is not collected. See design doc ``0005-energy-efficiency-metrics.md``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aiperf.common.logging import AIPerfLogger
from aiperf.common.models import MetricResult
from aiperf.metrics.types.power_efficiency_metrics import (
    AverageGpuPowerMetric,
    EnergyDelayProductMetric,
    EnergyPerOutputTokenMetric,
    EnergyPerRequestMetric,
    EnergyPerTotalTokenMetric,
    EnergyPerUserMetric,
    OutputTokensPerJouleMetric,
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
    emitted only when its inputs are available, mirroring the per-signal omission
    of the pipeline it replaces.
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
        energy_j, energy_count = gpu.total_energy_joules(start_ns, end_ns)
        power_w, power_count = gpu.total_power_watts(start_ns, end_ns)

        out = self._power_metrics(power_w, power_count)
        out += self._energy_metrics(energy_j, energy_count, metric)
        if (
            power_count > 0
            and power_w > 0
            and (throughput := metric("request_throughput"))
        ):
            out.append(_result(PerformancePerWattMetric, throughput / power_w))

        _logger.debug(
            lambda: (
                f"EnergyEfficiencyAnalyzer emitted {len(out)} metrics "
                f"(energy={energy_j:.2f}J/{energy_count}gpu, "
                f"power={power_w:.2f}W/{power_count}gpu)"
            )
        )
        return out

    @staticmethod
    def _power_metrics(power_w: float, power_count: int) -> list[MetricResult]:
        if power_count <= 0:
            return []
        return [
            _result(TotalGpuPowerMetric, power_w),
            _result(AverageGpuPowerMetric, power_w / power_count),
        ]

    def _energy_metrics(
        self,
        energy_j: float,
        energy_count: int,
        metric: Callable[[str], float | None],
    ) -> list[MetricResult]:
        if energy_count <= 0 or energy_j <= 0:
            return []
        out: list[MetricResult] = [_result(TotalGpuEnergyMetric, energy_j)]

        output_tokens = metric("total_output_tokens")
        if output_tokens:
            out.append(_result(OutputTokensPerJouleMetric, output_tokens / energy_j))
            out.append(
                _result(
                    EnergyPerOutputTokenMetric,
                    energy_j * _MS_PER_SECOND / output_tokens,
                )
            )
        total_tokens = (metric("total_isl") or 0.0) + (output_tokens or 0.0)
        if total_tokens > 0:
            out.append(
                _result(
                    EnergyPerTotalTokenMetric, energy_j * _MS_PER_SECOND / total_tokens
                )
            )

        energy_per_request_j = None
        if request_count := metric("request_count"):
            energy_per_request_j = energy_j / request_count
            out.append(_result(EnergyPerRequestMetric, energy_per_request_j))
        if (concurrency := self._concurrency()) is not None:
            out.append(_result(EnergyPerUserMetric, energy_j / concurrency))

        # Energy-delay product: J/request * mean request latency (s).
        latency_ms = metric("request_latency")
        if energy_per_request_j is not None and latency_ms:
            out.append(
                _result(
                    EnergyDelayProductMetric,
                    energy_per_request_j * (latency_ms / _MS_PER_SECOND),
                )
            )
        return out
