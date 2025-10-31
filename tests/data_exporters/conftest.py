# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared test fixtures for data exporters."""

import pytest

from aiperf.common.config import ServiceConfig
from aiperf.common.models import MetricResult
from aiperf.common.models.telemetry_models import (
    TelemetryHierarchy,
    TelemetryMetrics,
    TelemetryRecord,
    TelemetryResults,
)
from aiperf.exporters.exporter_config import ExporterConfig


@pytest.fixture
def sample_telemetry_record():
    """Create a sample TelemetryRecord for testing."""
    return TelemetryRecord(
        timestamp_ns=1000000000,
        dcgm_url="http://localhost:9400/metrics",
        gpu_index=0,
        gpu_model_name="NVIDIA H100",
        gpu_uuid="GPU-12345678-1234-1234-1234-123456789abc",
        pci_bus_id="00000000:01:00.0",
        device="nvidia0",
        hostname="test-node-01",
        telemetry_data=TelemetryMetrics(
            gpu_power_usage=300.0,
            energy_consumption=1000.5,
            gpu_utilization=85.0,
            gpu_memory_used=72.5,
            sm_clock_frequency=1410.0,
            memory_clock_frequency=1215.0,
            memory_temperature=65.0,
            gpu_temperature=70.0,
        ),
    )


@pytest.fixture
def sample_telemetry_results():
    """Create a sample TelemetryResults with realistic multi-GPU, multi-endpoint data."""
    hierarchy = TelemetryHierarchy()

    # Endpoint 1: localhost with 2 GPUs
    for gpu_idx in range(2):
        for time_offset in range(5):  # 5 time samples per GPU
            record = TelemetryRecord(
                timestamp_ns=1000000000 + time_offset * 1000000000,
                dcgm_url="http://localhost:9400/metrics",
                gpu_index=gpu_idx,
                gpu_model_name="NVIDIA H100",
                gpu_uuid=f"GPU-12345678-1234-1234-1234-12345678{gpu_idx:04d}",
                pci_bus_id=f"00000000:0{gpu_idx + 1}:00.0",
                device=f"nvidia{gpu_idx}",
                hostname="test-node-01",
                telemetry_data=TelemetryMetrics(
                    gpu_power_usage=280.0 + gpu_idx * 20 + time_offset * 5,
                    energy_consumption=1000.0 + time_offset * 100,
                    gpu_utilization=80.0 + time_offset * 2,
                    gpu_memory_used=70.0 + gpu_idx * 5 + time_offset * 1,
                    sm_clock_frequency=1410.0,
                    memory_clock_frequency=1215.0,
                    memory_temperature=60.0 + time_offset,
                    gpu_temperature=65.0 + time_offset * 2,
                ),
            )
            hierarchy.add_record(record)

    # Endpoint 2: remote node with 1 GPU
    for time_offset in range(5):
        record = TelemetryRecord(
            timestamp_ns=1000000000 + time_offset * 1000000000,
            dcgm_url="http://remote-node:9400/metrics",
            gpu_index=0,
            gpu_model_name="NVIDIA A100",
            gpu_uuid="GPU-abcdef01-2345-6789-abcd-ef0123456789",
            pci_bus_id="00000000:01:00.0",
            device="nvidia0",
            hostname="test-node-02",
            telemetry_data=TelemetryMetrics(
                gpu_power_usage=250.0 + time_offset * 10,
                energy_consumption=800.0 + time_offset * 80,
                gpu_utilization=75.0 + time_offset * 3,
                gpu_memory_used=60.0 + time_offset * 2,
                sm_clock_frequency=1350.0,
                memory_clock_frequency=1200.0,
                memory_temperature=58.0 + time_offset,
                gpu_temperature=63.0 + time_offset * 2,
            ),
        )
        hierarchy.add_record(record)

    return TelemetryResults(
        telemetry_data=hierarchy,
        start_ns=1000000000,
        end_ns=6000000000,
        endpoints_configured=[
            "http://localhost:9400/metrics",
            "http://remote-node:9400/metrics",
        ],
        endpoints_successful=[
            "http://localhost:9400/metrics",
            "http://remote-node:9400/metrics",
        ],
        error_summary=[],
    )


@pytest.fixture
def sample_telemetry_results_with_failures():
    """Create TelemetryResults with some failed endpoints."""
    hierarchy = TelemetryHierarchy()

    # Only one successful endpoint
    for time_offset in range(3):
        record = TelemetryRecord(
            timestamp_ns=1000000000 + time_offset * 1000000000,
            dcgm_url="http://localhost:9400/metrics",
            gpu_index=0,
            gpu_model_name="NVIDIA H100",
            gpu_uuid="GPU-12345678-1234-1234-1234-123456789abc",
            telemetry_data=TelemetryMetrics(
                gpu_power_usage=300.0 + time_offset * 10,
                gpu_utilization=85.0,
                gpu_memory_used=72.5,
                gpu_temperature=70.0,
            ),
        )
        hierarchy.add_record(record)

    return TelemetryResults(
        telemetry_data=hierarchy,
        start_ns=1000000000,
        end_ns=4000000000,
        endpoints_configured=[
            "http://localhost:9400/metrics",
            "http://unreachable-node:9400/metrics",
            "http://failed-node:9400/metrics",
        ],
        endpoints_successful=["http://localhost:9400/metrics"],
        error_summary=[],
    )


@pytest.fixture
def empty_telemetry_results():
    """Create TelemetryResults with no GPU data (all endpoints failed)."""
    return TelemetryResults(
        telemetry_data=TelemetryHierarchy(),
        start_ns=1000000000,
        end_ns=2000000000,
        endpoints_configured=[
            "http://unreachable-1:9400/metrics",
            "http://unreachable-2:9400/metrics",
        ],
        endpoints_successful=[],
        error_summary=[],
    )


def create_exporter_config(
    profile_results, user_config, telemetry_results=None, verbose=True
):
    """Helper to create ExporterConfig with common defaults."""
    return ExporterConfig(
        results=profile_results,
        user_config=user_config,
        service_config=ServiceConfig(verbose=verbose),
        telemetry_results=telemetry_results,
    )


@pytest.fixture
def sample_timeslice_metric_results():
    """Create sample timeslice metric results for testing."""
    return {
        0: [
            MetricResult(
                tag="time_to_first_token",
                header="Time to First Token",
                unit="ms",
                avg=45.2,
                min=12.1,
                max=89.3,
                p50=44.0,
                p90=78.0,
                p99=88.0,
                std=15.2,
            ),
            MetricResult(
                tag="inter_token_latency",
                header="Inter Token Latency",
                unit="ms",
                avg=5.1,
                min=2.3,
                max=12.4,
                p50=4.8,
                p90=9.2,
                p99=11.8,
                std=2.1,
            ),
        ],
        1: [
            MetricResult(
                tag="time_to_first_token",
                header="Time to First Token",
                unit="ms",
                avg=48.5,
                min=15.2,
                max=92.1,
                p50=47.3,
                p90=82.4,
                p99=90.5,
                std=16.1,
            ),
            MetricResult(
                tag="inter_token_latency",
                header="Inter Token Latency",
                unit="ms",
                avg=5.4,
                min=2.5,
                max=13.1,
                p50=5.1,
                p90=9.8,
                p99=12.3,
                std=2.3,
            ),
        ],
    }


@pytest.fixture
def mock_results_with_timeslices(sample_timeslice_metric_results):
    """Create mock results with timeslice data."""

    class MockResultsWithTimeslices:
        def __init__(self):
            self.timeslice_metric_results = sample_timeslice_metric_results
            self.records = []
            self.start_ns = None
            self.end_ns = None
            self.has_results = True
            self.was_cancelled = False
            self.error_summary = []

    return MockResultsWithTimeslices()


@pytest.fixture
def mock_results_without_timeslices():
    """Create mock results without timeslice data."""

    class MockResultsNoTimeslices:
        def __init__(self):
            self.timeslice_metric_results = None
            self.records = []
            self.start_ns = None
            self.end_ns = None
            self.has_results = False
            self.was_cancelled = False
            self.error_summary = []

    return MockResultsNoTimeslices()
