# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for compute_energy_efficiency_from_summaries.

The cross-input energy analyzer runs controller-side as a plain function (not
an AnalyzerProtocol plugin) because the underlying GPU telemetry accumulator
lives in a separate process. These tests construct synthetic
``TelemetryExportData`` and ``ProfileResults`` directly and assert the math /
branching of the compute function.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from aiperf.analysis.energy_analyzer import (
    EnergyEfficiencySummary,
    EnergySource,
    _safe_div,
    compute_energy_efficiency_from_summaries,
)
from aiperf.common.models.export_models import (
    EndpointData,
    GpuSummary,
    JsonMetricResult,
    TelemetryExportData,
    TelemetrySummary,
)
from aiperf.common.models.metric_result_models import MetricResult, ProfileResults

# ============================================================
# Fixtures
# ============================================================


def _gpu(
    *,
    energy_mj: float | None = None,
    power_w: float | None = None,
    index: int = 0,
    uuid: str = "GPU-00000000",
) -> GpuSummary:
    """Build a single GpuSummary with optional energy_consumption / gpu_power_usage."""
    metrics: dict[str, JsonMetricResult] = {}
    if energy_mj is not None:
        metrics["energy_consumption"] = JsonMetricResult(unit="MJ", avg=energy_mj)
    if power_w is not None:
        metrics["gpu_power_usage"] = JsonMetricResult(unit="W", avg=power_w)
    return GpuSummary(
        gpu_index=index,
        gpu_name="Test-GPU",
        gpu_uuid=uuid,
        hostname=None,
        metrics=metrics,
    )


def _telemetry(*gpus: GpuSummary, endpoint: str = "dcgm-0") -> TelemetryExportData:
    """Wrap GpuSummary instances into a TelemetryExportData with one endpoint."""
    return TelemetryExportData(
        summary=TelemetrySummary(
            start_time=datetime(2026, 5, 2, 0, 0, 0),
            end_time=datetime(2026, 5, 2, 0, 1, 0),
        ),
        endpoints={
            endpoint: EndpointData(
                gpus={g.gpu_uuid: g for g in gpus},
            )
        },
    )


def _profile(
    *,
    duration_s: float = 60.0,
    request_count: float | None = 1000.0,
    output_token_throughput: float | None = 500.0,
    request_throughput: float | None = 16.6,
    total_osl: float | None = 30000.0,
    total_isl: float | None = 70000.0,
    goodput: float | None = 14.2,
    was_cancelled: bool = False,
) -> ProfileResults:
    """Build a ProfileResults with the standard inference summary metrics."""
    records: list[MetricResult] = []

    def _add(tag: str, header: str, unit: str, avg: float | None) -> None:
        if avg is not None:
            records.append(MetricResult(tag=tag, header=header, unit=unit, avg=avg))

    _add("request_count", "Request Count", "req", request_count)
    _add(
        "output_token_throughput",
        "Output Token Throughput",
        "tps",
        output_token_throughput,
    )
    _add("request_throughput", "Request Throughput", "req/s", request_throughput)
    _add("total_osl", "Total OSL", "tokens", total_osl)
    _add("total_isl", "Total ISL", "tokens", total_isl)
    _add("goodput", "Goodput", "good-req/s", goodput)
    return ProfileResults(
        completed=int(request_count) if request_count else 0,
        start_ns=1_000_000_000,
        end_ns=1_000_000_000 + int(duration_s * 1e9),
        records=records,
        was_cancelled=was_cancelled,
    )


# ============================================================
# Branch coverage
# ============================================================


class TestEarlyReturns:
    def test_returns_none_when_telemetry_missing(self) -> None:
        result = compute_energy_efficiency_from_summaries(
            telemetry=None, profile_results=_profile()
        )
        assert result is None

    def test_returns_none_when_profile_results_missing(self) -> None:
        result = compute_energy_efficiency_from_summaries(
            telemetry=_telemetry(_gpu(energy_mj=1.0)),
            profile_results=None,
        )
        assert result is None

    def test_returns_none_when_no_endpoints(self) -> None:
        empty = TelemetryExportData(
            summary=TelemetrySummary(
                start_time=datetime(2026, 5, 2),
                end_time=datetime(2026, 5, 2),
            ),
            endpoints={},
        )
        result = compute_energy_efficiency_from_summaries(
            telemetry=empty, profile_results=_profile()
        )
        assert result is None

    def test_returns_none_when_no_metrics(self) -> None:
        # GPU present but no energy or power readings
        result = compute_energy_efficiency_from_summaries(
            telemetry=_telemetry(_gpu()),
            profile_results=_profile(),
        )
        assert result is None


class TestEnergySources:
    def test_dcgm_counter_path(self) -> None:
        # 1.5 MJ counter delta on a 60 s window
        result = compute_energy_efficiency_from_summaries(
            telemetry=_telemetry(_gpu(energy_mj=1.5, power_w=20000.0)),
            profile_results=_profile(duration_s=60.0),
        )
        assert result is not None
        assert result.energy_source is EnergySource.DCGM_COUNTER
        assert result.gpu_count == 1
        assert result.total_gpu_energy_j == pytest.approx(1.5e6)
        # avg_power = energy / duration = 1.5e6 J / 60 s = 25000 W
        assert result.average_gpu_power_w == pytest.approx(25000.0)

    def test_power_integration_fallback(self) -> None:
        # No counter, power-only — energy = power * duration
        result = compute_energy_efficiency_from_summaries(
            telemetry=_telemetry(_gpu(power_w=300.0)),
            profile_results=_profile(duration_s=60.0),
        )
        assert result is not None
        assert result.energy_source is EnergySource.POWER_INTEGRATION
        assert result.average_gpu_power_w == pytest.approx(300.0)
        assert result.total_gpu_energy_j == pytest.approx(300.0 * 60.0)

    def test_mj_to_j_conversion(self) -> None:
        """1.5 MJ on the wire → total_gpu_energy_j == 1.5e6."""
        result = compute_energy_efficiency_from_summaries(
            telemetry=_telemetry(_gpu(energy_mj=1.5)),
            profile_results=_profile(),
        )
        assert result is not None
        assert result.total_gpu_energy_j == pytest.approx(1.5e6)


class TestMultiGPU:
    def test_dual_gpu_sums_energy_and_power(self) -> None:
        result = compute_energy_efficiency_from_summaries(
            telemetry=_telemetry(
                _gpu(energy_mj=1.0, power_w=200.0, index=0, uuid="GPU-A"),
                _gpu(energy_mj=2.0, power_w=300.0, index=1, uuid="GPU-B"),
            ),
            profile_results=_profile(duration_s=60.0),
        )
        assert result is not None
        assert result.gpu_count == 2
        # 3.0 MJ total → 3e6 J
        assert result.total_gpu_energy_j == pytest.approx(3.0e6)
        # avg_power from energy/duration: 3e6 / 60 = 50000 W
        assert result.average_gpu_power_w == pytest.approx(50000.0)


class TestDerivedMetrics:
    def test_missing_optional_inference_metrics(self) -> None:
        # goodput absent → goodput_per_watt is None; others computed
        result = compute_energy_efficiency_from_summaries(
            telemetry=_telemetry(_gpu(energy_mj=1.0)),
            profile_results=_profile(goodput=None),
        )
        assert result is not None
        assert result.goodput_per_watt is None
        assert result.energy_per_output_token_mj is not None
        assert result.energy_per_request_j is not None
        assert result.performance_per_watt is not None

    def test_energy_per_request_math(self) -> None:
        # 1.0 MJ = 1e6 J / 1000 requests = 1000 J/req
        result = compute_energy_efficiency_from_summaries(
            telemetry=_telemetry(_gpu(energy_mj=1.0)),
            profile_results=_profile(request_count=1000.0),
        )
        assert result is not None
        assert result.energy_per_request_j == pytest.approx(1000.0)

    def test_energy_per_output_token_math(self) -> None:
        # 1.0 MJ = 1e6 J = 1e9 mJ / 30000 output tokens = ~33333 mJ/token
        result = compute_energy_efficiency_from_summaries(
            telemetry=_telemetry(_gpu(energy_mj=1.0)),
            profile_results=_profile(total_osl=30000.0),
        )
        assert result is not None
        assert result.energy_per_output_token_mj == pytest.approx(1e9 / 30000.0)


class TestContractBehavior:
    def test_cancelled_run_still_produces_results(self) -> None:
        """Cancelled runs with valid telemetry still produce a summary.

        Behavioral contract test — the compute function does not branch on
        was_cancelled. Documents the chosen policy: partial-run energy data
        is still informative.
        """
        result = compute_energy_efficiency_from_summaries(
            telemetry=_telemetry(_gpu(energy_mj=1.0, power_w=200.0)),
            profile_results=_profile(was_cancelled=True),
        )
        assert result is not None
        assert isinstance(result, EnergyEfficiencySummary)


# ============================================================
# Pre-existing helper coverage retained
# ============================================================


class TestSafeDiv:
    @pytest.mark.parametrize(
        "num, denom, expected",
        [
            (10.0, 2.0, 5.0),
            (1.0, 4.0, 0.25),
            (100.0, 100.0, 1.0),
        ],
    )
    def test_safe_div_normal(self, num: float, denom: float, expected: float) -> None:
        assert _safe_div(num, denom) == expected

    def test_safe_div_zero_denominator_returns_none(self) -> None:
        assert _safe_div(10.0, 0.0) is None

    def test_safe_div_negative_denominator_returns_none(self) -> None:
        assert _safe_div(10.0, -1.0) is None

    def test_safe_div_none_denominator_returns_none(self) -> None:
        assert _safe_div(10.0, None) is None

    def test_safe_div_none_numerator_returns_none(self) -> None:
        assert _safe_div(None, 10.0) is None


class TestSummarySerialization:
    def test_to_json_includes_source_and_metrics(self) -> None:
        result = compute_energy_efficiency_from_summaries(
            telemetry=_telemetry(_gpu(energy_mj=1.5, power_w=300.0)),
            profile_results=_profile(),
        )
        assert result is not None
        data = result.to_json()
        assert data["source"]["energy_source"] == "dcgm_counter"
        assert data["source"]["total_gpu_energy_j"] == pytest.approx(1.5e6)
        assert data["source"]["gpu_count"] == 1
        assert "energy_per_request_j" in data["metrics"]
        assert "results" in data  # MetricResult list for the export pipeline

    def test_to_csv_returns_metric_result_rows(self) -> None:
        result = compute_energy_efficiency_from_summaries(
            telemetry=_telemetry(_gpu(energy_mj=1.0)),
            profile_results=_profile(),
        )
        assert result is not None
        rows = result.to_csv()
        # One row per non-None derived metric + total_gpu_energy + average_gpu_power.
        # At minimum: total_gpu_energy + average_gpu_power.
        assert len(rows) >= 2
        for row in rows:
            assert "tag" in row
            assert "current" not in row  # CSV strips "current"
