# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared test fixtures for data exporters."""

import pytest

from aiperf.common.models.telemetry_models import (
    TelemetryHierarchy,
    TelemetryRecord,
    TelemetryResults,
)


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
        gpu_power_usage=300.0,
        energy_consumption=1000.5,
        gpu_utilization=85.0,
        gpu_memory_used=72.5,
        sm_clock_frequency=1410.0,
        memory_clock_frequency=1215.0,
        memory_temperature=65.0,
        gpu_temperature=70.0,
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
                gpu_power_usage=280.0 + gpu_idx * 20 + time_offset * 5,
                energy_consumption=1000.0 + time_offset * 100,
                gpu_utilization=80.0 + time_offset * 2,
                gpu_memory_used=70.0 + gpu_idx * 5 + time_offset * 1,
                sm_clock_frequency=1410.0,
                memory_clock_frequency=1215.0,
                memory_temperature=60.0 + time_offset,
                gpu_temperature=65.0 + time_offset * 2,
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
            gpu_power_usage=250.0 + time_offset * 10,
            energy_consumption=800.0 + time_offset * 80,
            gpu_utilization=75.0 + time_offset * 3,
            gpu_memory_used=60.0 + time_offset * 2,
            sm_clock_frequency=1350.0,
            memory_clock_frequency=1200.0,
            memory_temperature=58.0 + time_offset,
            gpu_temperature=63.0 + time_offset * 2,
        )
        hierarchy.add_record(record)

    return TelemetryResults(
        telemetry_data=hierarchy,
        start_ns=1000000000,
        end_ns=6000000000,
        endpoints_tested=[
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
            gpu_power_usage=300.0 + time_offset * 10,
            gpu_utilization=85.0,
            gpu_memory_used=72.5,
            gpu_temperature=70.0,
        )
        hierarchy.add_record(record)

    return TelemetryResults(
        telemetry_data=hierarchy,
        start_ns=1000000000,
        end_ns=4000000000,
        endpoints_tested=[
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
        endpoints_tested=[
            "http://unreachable-1:9400/metrics",
            "http://unreachable-2:9400/metrics",
        ],
        endpoints_successful=[],
        error_summary=[],
    )
