# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Formula + dependency-gating tests for the EnergyEfficiencyAnalyzer.

The analyzer joins a live GPU-telemetry accumulator (windowed energy/power) with
the metrics-accumulator summary (token/throughput/latency totals) via a
SummaryContext, and emits the energy metric family per vendor. Values are pinned
to the formulas in design doc ``0005-energy-efficiency-metrics.md``.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from aiperf.common.accumulator_protocols import SummaryContext
from aiperf.common.models import MetricResult
from aiperf.metrics.accumulator_models import AccumulatorMetricsSummary
from aiperf.metrics.energy_efficiency_analyzer import (
    EnergyEfficiencyAnalyzer,
    EnergySource,
)
from aiperf.plugin.enums import AccumulatorType

# 10-second profiling window in nanoseconds (round numbers for the formulas).
_START_NS = 0
_END_NS = 10 * 1_000_000_000
_DURATION_S = 10.0


class _StubGpu:
    """Minimal GPUTelemetryAccumulator query surface for the analyzer."""

    def __init__(
        self,
        energy: tuple[float, int],
        power: tuple[float, int],
        platforms: set[str] | None = None,
    ) -> None:
        self._energy = energy
        self._power = power
        self._platforms = platforms if platforms is not None else {"nvidia"}

    def available_platforms(self) -> set[str]:
        return self._platforms

    def total_energy_joules(
        self, start_ns: int | None, end_ns: int | None, platform: str | None = None
    ) -> tuple[float, int]:
        return self._energy

    def total_power_watts(
        self, start_ns: int | None, end_ns: int | None, platform: str | None = None
    ) -> tuple[float, int]:
        return self._power

    def scrape_span_ns(self) -> tuple[int, int] | None:
        return None


def _metrics_summary(**avgs: float) -> AccumulatorMetricsSummary:
    return AccumulatorMetricsSummary(
        results={
            tag: MetricResult(tag=tag, header=tag, unit="x", avg=avg)
            for tag, avg in avgs.items()
        }
    )


def _analyzer(concurrency: int | None = 8) -> EnergyEfficiencyAnalyzer:
    run = MagicMock()
    run.cfg.get_profiling_phases.return_value = [
        SimpleNamespace(concurrency=concurrency)
    ]
    return EnergyEfficiencyAnalyzer(service_id="t", run=run, pub_client=None)


def _ctx(gpu, summary) -> SummaryContext:
    return SummaryContext(
        accumulators={AccumulatorType.GPU_TELEMETRY: gpu} if gpu else {},
        accumulator_outputs=(
            {AccumulatorType.METRIC_RESULTS: summary} if summary else {}
        ),
        start_ns=_START_NS,
        end_ns=_END_NS,
    )


class TestEnergyEfficiencyAnalyzer:
    @pytest.mark.asyncio
    async def test_full_metric_set_matches_doc_formulas_nvidia(self) -> None:
        gpu = _StubGpu(energy=(1000.0, 2), power=(200.0, 2), platforms={"nvidia"})
        summary = _metrics_summary(
            total_osl=5000.0,
            total_isl=3000.0,
            request_count=100.0,
            request_throughput=10.0,
            output_token_throughput=50.0,
            goodput=8.0,
            request_latency=500.0,  # ms
        )

        results = await _analyzer(concurrency=8).analyze(_ctx(gpu, summary))
        by_tag = {r.tag: r.avg for r in results}

        assert by_tag["nvidia_total_gpu_power"] == pytest.approx(200.0)
        # DCGM counter path: avg power = energy / duration = 1000 / 10s.
        assert by_tag["nvidia_average_gpu_power"] == pytest.approx(100.0)
        assert by_tag["nvidia_total_gpu_energy"] == pytest.approx(1000.0)
        assert by_tag["nvidia_output_tokens_per_joule"] == pytest.approx(
            5.0
        )  # 5000 / 1000
        assert by_tag["nvidia_energy_per_output_token"] == pytest.approx(
            200.0
        )  # 1000*1000/5000
        assert by_tag["nvidia_energy_per_total_token"] == pytest.approx(
            125.0
        )  # 1e6/(3000+5000)
        assert by_tag["nvidia_energy_per_request"] == pytest.approx(10.0)  # 1000 / 100
        assert by_tag["nvidia_energy_per_user"] == pytest.approx(125.0)  # 1000 / 8
        assert by_tag["nvidia_energy_delay_product"] == pytest.approx(5.0)  # 10 * 0.5s
        assert by_tag["nvidia_performance_per_watt"] == pytest.approx(0.1)  # 10 / 100
        assert by_tag["nvidia_output_tps_per_watt"] == pytest.approx(0.5)  # 50 / 100
        assert by_tag["nvidia_goodput_per_watt"] == pytest.approx(0.08)  # 8 / 100

    @pytest.mark.asyncio
    async def test_full_metric_set_matches_doc_formulas_amd(self) -> None:
        gpu = _StubGpu(energy=(1000.0, 2), power=(200.0, 2), platforms={"amd"})
        summary = _metrics_summary(
            total_osl=5000.0,
            total_isl=3000.0,
            request_count=100.0,
            request_throughput=10.0,
            output_token_throughput=50.0,
            goodput=8.0,
            request_latency=500.0,
        )

        results = await _analyzer(concurrency=8).analyze(_ctx(gpu, summary))
        by_tag = {r.tag: r.avg for r in results}

        assert by_tag["amd_total_gpu_power"] == pytest.approx(200.0)
        assert by_tag["amd_average_gpu_power"] == pytest.approx(100.0)
        assert by_tag["amd_total_gpu_energy"] == pytest.approx(1000.0)
        assert by_tag["amd_output_tokens_per_joule"] == pytest.approx(5.0)
        assert by_tag["amd_energy_per_output_token"] == pytest.approx(200.0)
        assert by_tag["amd_energy_per_total_token"] == pytest.approx(125.0)
        assert by_tag["amd_energy_per_request"] == pytest.approx(10.0)
        assert by_tag["amd_energy_per_user"] == pytest.approx(125.0)
        assert by_tag["amd_energy_delay_product"] == pytest.approx(5.0)
        assert by_tag["amd_performance_per_watt"] == pytest.approx(0.1)
        assert by_tag["amd_output_tps_per_watt"] == pytest.approx(0.5)
        assert by_tag["amd_goodput_per_watt"] == pytest.approx(0.08)

    @pytest.mark.asyncio
    async def test_mixed_vendor_emits_both_families(self) -> None:
        gpu = _StubGpu(
            energy=(1000.0, 2), power=(200.0, 2), platforms={"nvidia", "amd"}
        )
        summary = _metrics_summary(total_osl=5000.0, request_count=100.0)

        results = await _analyzer(concurrency=None).analyze(_ctx(gpu, summary))
        tags = {r.tag for r in results}

        assert "nvidia_total_gpu_energy" in tags
        assert "amd_total_gpu_energy" in tags
        assert "nvidia_output_tokens_per_joule" in tags
        assert "amd_output_tokens_per_joule" in tags

    @pytest.mark.asyncio
    async def test_unknown_platform_is_skipped(self) -> None:
        gpu = _StubGpu(energy=(1000.0, 2), power=(200.0, 2), platforms={"intel"})
        summary = _metrics_summary(total_osl=5000.0)
        results = await _analyzer().analyze(_ctx(gpu, summary))
        assert results == []

    @pytest.mark.asyncio
    async def test_zero_valued_per_watt_metrics_are_emitted_not_dropped(self) -> None:
        # A genuine 0.0 goodput (no request met the SLO) is a valid measurement:
        # goodput_per_watt must be emitted as 0.0, not silently omitted.
        gpu = _StubGpu(energy=(1000.0, 2), power=(200.0, 2), platforms={"nvidia"})
        summary = _metrics_summary(
            total_osl=5000.0,
            request_count=100.0,
            request_throughput=10.0,
            output_token_throughput=50.0,
            goodput=0.0,
        )

        results = await _analyzer(concurrency=8).analyze(_ctx(gpu, summary))
        by_tag = {r.tag: r.avg for r in results}

        assert "nvidia_goodput_per_watt" in by_tag
        assert by_tag["nvidia_goodput_per_watt"] == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_no_gpu_accumulator_returns_empty(self) -> None:
        summary = _metrics_summary(total_osl=5000.0)
        results = await _analyzer().analyze(_ctx(None, summary))
        assert results == []

    @pytest.mark.asyncio
    async def test_no_metrics_summary_returns_empty(self) -> None:
        gpu = _StubGpu(energy=(1000.0, 2), power=(200.0, 2))
        results = await _analyzer().analyze(_ctx(gpu, None))
        assert results == []

    @pytest.mark.asyncio
    async def test_power_integration_fallback_when_no_energy_counter(self) -> None:
        # No energy counter (count 0); fall back to power * duration.
        gpu = _StubGpu(energy=(0.0, 0), power=(150.0, 3), platforms={"nvidia"})
        summary = _metrics_summary(request_throughput=6.0)
        results = await _analyzer(concurrency=None).analyze(_ctx(gpu, summary))
        by_tag = {r.tag: r.avg for r in results}

        assert by_tag["nvidia_total_gpu_power"] == pytest.approx(150.0)
        # Integrated energy = 150 W * 10 s = 1500 J; avg power = fleet power.
        assert by_tag["nvidia_total_gpu_energy"] == pytest.approx(1500.0)
        assert by_tag["nvidia_average_gpu_power"] == pytest.approx(150.0)
        assert by_tag["nvidia_performance_per_watt"] == pytest.approx(0.04)  # 6 / 150
        assert "nvidia_energy_per_output_token" not in by_tag  # no tokens
        assert "nvidia_energy_per_user" not in by_tag  # concurrency None

    @pytest.mark.asyncio
    async def test_no_energy_signal_at_all_returns_empty(self) -> None:
        # Neither energy counter nor power gauge: source UNAVAILABLE, nothing emitted.
        gpu = _StubGpu(energy=(0.0, 0), power=(0.0, 0), platforms={"nvidia"})
        summary = _metrics_summary(request_throughput=6.0)
        results = await _analyzer().analyze(_ctx(gpu, summary))
        assert results == []

    @pytest.mark.asyncio
    async def test_no_platforms_returns_empty(self) -> None:
        gpu = _StubGpu(energy=(1000.0, 2), power=(200.0, 2), platforms=set())
        summary = _metrics_summary(total_osl=5000.0)
        results = await _analyzer().analyze(_ctx(gpu, summary))
        assert results == []

    @pytest.mark.asyncio
    async def test_resolve_energy_source_detection(self) -> None:
        _, _, counter = EnergyEfficiencyAnalyzer._resolve_energy(
            (1000.0, 2), (200.0, 2), _DURATION_S
        )
        assert counter is EnergySource.DCGM_COUNTER
        _, _, integ = EnergyEfficiencyAnalyzer._resolve_energy(
            (0.0, 0), (150.0, 3), _DURATION_S
        )
        assert integ is EnergySource.POWER_INTEGRATION
        _, _, none = EnergyEfficiencyAnalyzer._resolve_energy((0.0, 0), (0.0, 0), 0.0)
        assert none is EnergySource.UNAVAILABLE

    @pytest.mark.asyncio
    async def test_concurrency_gates_energy_per_user(self) -> None:
        gpu = _StubGpu(energy=(1000.0, 1), power=(100.0, 1), platforms={"nvidia"})
        summary = _metrics_summary(total_osl=1000.0)
        results = await _analyzer(concurrency=None).analyze(_ctx(gpu, summary))
        assert "nvidia_energy_per_user" not in {r.tag for r in results}
