# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for SystemController cross-input analyzer fan-in.

Verifies that ``SystemController._compute_cross_input_analyzers`` correctly
populates ``self._energy_efficiency_results`` from ``self._telemetry_results``
+ ``self._profile_results`` when GPU telemetry is enabled, and is a no-op
otherwise.

The full ``_export_results_data`` integration (stops services, writes K8s
markers, runs ExporterManager) is not exercised here — the spec calls for an
e2e integration test as a separate piece of work once telemetry test infra
exists.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from aiperf.analysis.energy_analyzer import EnergyEfficiencySummary, EnergySource
from aiperf.common.models.export_models import (
    EndpointData,
    GpuSummary,
    JsonMetricResult,
    TelemetryExportData,
    TelemetrySummary,
)
from aiperf.common.models.metric_result_models import (
    MetricResult,
    ProcessRecordsResult,
    ProfileResults,
)
from aiperf.controller.system_controller import SystemController


def _telemetry_with_energy(
    energy_mj: float = 1.0, power_w: float = 200.0
) -> TelemetryExportData:
    """Synthetic TelemetryExportData with a single GPU reporting energy + power."""
    return TelemetryExportData(
        summary=TelemetrySummary(
            start_time=datetime(2026, 5, 2),
            end_time=datetime(2026, 5, 2),
        ),
        endpoints={
            "dcgm-0": EndpointData(
                gpus={
                    "GPU-0": GpuSummary(
                        gpu_index=0,
                        gpu_name="Test-GPU",
                        gpu_uuid="GPU-0",
                        hostname=None,
                        metrics={
                            "energy_consumption": JsonMetricResult(
                                unit="MJ", avg=energy_mj
                            ),
                            "gpu_power_usage": JsonMetricResult(unit="W", avg=power_w),
                        },
                    ),
                },
            ),
        },
    )


def _profile_with_inference_metrics() -> ProcessRecordsResult:
    """Synthetic ProcessRecordsResult with the inference scalars the analyzer reads."""
    profile = ProfileResults(
        completed=1000,
        start_ns=1_000_000_000,
        end_ns=1_000_000_000 + 60 * 10**9,
        records=[
            MetricResult(
                tag="request_count", header="Request Count", unit="req", avg=1000.0
            ),
            MetricResult(
                tag="output_token_throughput",
                header="Output Token Throughput",
                unit="tps",
                avg=500.0,
            ),
            MetricResult(
                tag="request_throughput",
                header="Request Throughput",
                unit="req/s",
                avg=16.6,
            ),
            MetricResult(
                tag="total_osl", header="Total OSL", unit="tokens", avg=30000.0
            ),
            MetricResult(
                tag="total_isl", header="Total ISL", unit="tokens", avg=70000.0
            ),
        ],
    )
    return ProcessRecordsResult(results=profile, errors=[])


class TestCrossInputAnalyzerFanIn:
    """`_compute_cross_input_analyzers` populates `_energy_efficiency_results`."""

    def test_populates_when_inputs_present(
        self, system_controller: SystemController
    ) -> None:
        system_controller._telemetry_results = _telemetry_with_energy()
        system_controller._profile_results = _profile_with_inference_metrics()
        system_controller._energy_efficiency_results = None

        system_controller._compute_cross_input_analyzers()

        result = system_controller._energy_efficiency_results
        assert result is not None
        assert isinstance(result, EnergyEfficiencySummary)
        assert result.energy_source is EnergySource.DCGM_COUNTER
        assert result.gpu_count == 1
        # 1.0 MJ -> 1e6 J
        assert result.total_gpu_energy_j == pytest.approx(1.0e6)

    def test_no_op_when_telemetry_disabled(
        self, system_controller: SystemController
    ) -> None:
        # Even with valid inputs, gpu_telemetry_disabled short-circuits.
        system_controller.run.cfg.gpu_telemetry.enabled = False
        system_controller._telemetry_results = _telemetry_with_energy()
        system_controller._profile_results = _profile_with_inference_metrics()
        system_controller._energy_efficiency_results = None

        system_controller._compute_cross_input_analyzers()

        assert system_controller._energy_efficiency_results is None

    def test_no_op_when_already_populated(
        self, system_controller: SystemController
    ) -> None:
        # If a prior path populated the field, do not overwrite it.
        sentinel = EnergyEfficiencySummary(
            total_gpu_energy_j=42.0,
            average_gpu_power_w=1.0,
            gpu_count=99,
            energy_source=EnergySource.DCGM_COUNTER,
        )
        system_controller._telemetry_results = _telemetry_with_energy()
        system_controller._profile_results = _profile_with_inference_metrics()
        system_controller._energy_efficiency_results = sentinel

        system_controller._compute_cross_input_analyzers()

        assert system_controller._energy_efficiency_results is sentinel

    def test_no_op_when_telemetry_results_missing(
        self, system_controller: SystemController
    ) -> None:
        # Telemetry message never arrived (e.g., GPUTelemetryManager crashed).
        system_controller._telemetry_results = None
        system_controller._profile_results = _profile_with_inference_metrics()
        system_controller._energy_efficiency_results = None

        system_controller._compute_cross_input_analyzers()

        assert system_controller._energy_efficiency_results is None

    def test_no_op_when_profile_results_missing(
        self, system_controller: SystemController
    ) -> None:
        # Profile results message never arrived — extremely unlikely path,
        # but the hook should not crash.
        system_controller._telemetry_results = _telemetry_with_energy()
        system_controller._profile_results = None
        system_controller._energy_efficiency_results = None

        system_controller._compute_cross_input_analyzers()

        assert system_controller._energy_efficiency_results is None
