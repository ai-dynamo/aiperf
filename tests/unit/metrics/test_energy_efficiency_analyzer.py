# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Formula + dependency-gating tests for the EnergyEfficiencyAnalyzer.

The analyzer joins a live GPU-telemetry accumulator (windowed energy/power) with
the metrics-accumulator summary (token/throughput/latency totals) via a
SummaryContext, and emits the energy metric family. Values are pinned to the
formulas in design doc ``0005-energy-efficiency-metrics.md``.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from aiperf.common.accumulator_protocols import SummaryContext
from aiperf.common.models import MetricResult
from aiperf.metrics.accumulator_models import AccumulatorMetricsSummary
from aiperf.metrics.energy_efficiency_analyzer import EnergyEfficiencyAnalyzer
from aiperf.plugin.enums import AccumulatorType


class _StubGpu:
    """Minimal GPUTelemetryAccumulator query surface for the analyzer."""

    def __init__(self, energy: tuple[float, int], power: tuple[float, int]) -> None:
        self._energy = energy
        self._power = power

    def total_energy_joules(self, start_ns, end_ns) -> tuple[float, int]:
        return self._energy

    def total_power_watts(self, start_ns, end_ns) -> tuple[float, int]:
        return self._power


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
        start_ns=1_000,
        end_ns=2_000,
    )


class TestEnergyEfficiencyAnalyzer:
    @pytest.mark.asyncio
    async def test_full_metric_set_matches_doc_formulas(self) -> None:
        gpu = _StubGpu(energy=(1000.0, 2), power=(200.0, 2))
        summary = _metrics_summary(
            total_output_tokens=5000.0,
            total_isl=3000.0,
            request_count=100.0,
            request_throughput=10.0,
            request_latency=500.0,  # ms
        )

        results = await _analyzer(concurrency=8).analyze(_ctx(gpu, summary))
        by_tag = {r.tag: r.avg for r in results}

        assert by_tag["total_gpu_power"] == pytest.approx(200.0)
        assert by_tag["average_gpu_power"] == pytest.approx(100.0)  # 200 / 2 GPUs
        assert by_tag["total_gpu_energy"] == pytest.approx(1000.0)
        assert by_tag["output_tokens_per_joule"] == pytest.approx(5.0)  # 5000 / 1000
        assert by_tag["energy_per_output_token"] == pytest.approx(
            200.0
        )  # 1000*1000/5000
        assert by_tag["energy_per_total_token"] == pytest.approx(
            125.0
        )  # 1e6/(3000+5000)
        assert by_tag["energy_per_request"] == pytest.approx(10.0)  # 1000 / 100
        assert by_tag["energy_per_user"] == pytest.approx(125.0)  # 1000 / 8
        assert by_tag["energy_delay_product"] == pytest.approx(5.0)  # 10 * 0.5s
        assert by_tag["performance_per_watt"] == pytest.approx(0.05)  # 10 / 200

    @pytest.mark.asyncio
    async def test_no_gpu_accumulator_returns_empty(self) -> None:
        summary = _metrics_summary(total_output_tokens=5000.0)
        results = await _analyzer().analyze(_ctx(None, summary))
        assert results == []

    @pytest.mark.asyncio
    async def test_no_metrics_summary_returns_empty(self) -> None:
        gpu = _StubGpu(energy=(1000.0, 2), power=(200.0, 2))
        results = await _analyzer().analyze(_ctx(gpu, None))
        assert results == []

    @pytest.mark.asyncio
    async def test_missing_signals_omit_dependent_metrics(self) -> None:
        # No energy signal (count 0) and no tokens: only power-derived metrics.
        gpu = _StubGpu(energy=(0.0, 0), power=(150.0, 3))
        summary = _metrics_summary(request_throughput=6.0)
        results = await _analyzer(concurrency=None).analyze(_ctx(gpu, summary))
        by_tag = {r.tag: r.avg for r in results}

        assert by_tag["total_gpu_power"] == pytest.approx(150.0)
        assert by_tag["average_gpu_power"] == pytest.approx(50.0)
        assert by_tag["performance_per_watt"] == pytest.approx(0.04)  # 6 / 150
        assert "total_gpu_energy" not in by_tag
        assert "energy_per_output_token" not in by_tag
        assert "energy_per_user" not in by_tag

    @pytest.mark.asyncio
    async def test_concurrency_gates_energy_per_user(self) -> None:
        gpu = _StubGpu(energy=(1000.0, 1), power=(100.0, 1))
        summary = _metrics_summary(total_output_tokens=1000.0)
        results = await _analyzer(concurrency=None).analyze(_ctx(gpu, summary))
        assert "energy_per_user" not in {r.tag for r in results}
