# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Shared fixtures for testing GPU telemetry components.
"""

import pytest

from aiperf.common.models.telemetry_models import TelemetryMetrics, TelemetryRecord
from tests.aiperf_mock_server.dcgm_faker import DCGMFaker


@pytest.fixture
def sample_dcgm_data():
    """Sample DCGM metrics from DCGMFaker (single GPU)."""

    faker = DCGMFaker(
        gpu_name="rtx6000",
        num_gpus=1,
        seed=42,
        hostname="ed7e7a5e585f",
        initial_load=0.1,
    )
    return faker.generate()


@pytest.fixture
def multi_gpu_dcgm_data():
    """Multi-GPU DCGM metrics from DCGMFaker (3 GPUs, mixed types)."""

    faker = DCGMFaker(
        gpu_name="rtx6000",
        num_gpus=3,
        seed=42,
        hostname="ed7e7a5e585f",
        initial_load=0.3,
    )
    return faker.generate()


@pytest.fixture
def sample_telemetry_records():
    """Sample TelemetryRecord objects for testing."""

    return [
        TelemetryRecord(
            timestamp_ns=1000000000,
            dcgm_url="http://localhost:9401/metrics",
            gpu_index=0,
            gpu_model_name="NVIDIA RTX 6000 Ada Generation",
            gpu_uuid="GPU-ef6ef310-f8e2-cef9-036e-8f12d59b5ffc",
            pci_bus_id="00000000:02:00.0",
            device="nvidia0",
            hostname="ed7e7a5e585f",
            telemetry_data=TelemetryMetrics(
                gpu_power_usage=22.582,
                energy_consumption=955.287014,
                gpu_utilization=1.0,
                gpu_memory_used=45.521,  # 46614 MiB / 1024 ≈ 45.521 GB
            ),
        ),
    ]


@pytest.fixture
def multi_gpu_telemetry_records():
    """Multiple GPU records for batch processing tests."""

    records = []

    # Generate 50 samples for each GPU with realistic patterns
    for i in range(50):
        timestamp = 1000000000 + (i * 33000000)  # 33ms intervals

        # GPU 0 - Active workload (RTX 6000)
        records.append(
            TelemetryRecord(
                timestamp_ns=timestamp,
                dcgm_url="http://localhost:9401/metrics",
                gpu_index=0,
                gpu_model_name="NVIDIA RTX 6000 Ada Generation",
                gpu_uuid="GPU-ef6ef310-f8e2-cef9-036e-8f12d59b5ffc",
                pci_bus_id="00000000:02:00.0",
                device="nvidia0",
                hostname="ed7e7a5e585f",
                telemetry_data=TelemetryMetrics(
                    gpu_power_usage=70.0 + (i % 30),  # Varying power 70-99W
                    energy_consumption=(280000000 + (i * 2000000))
                    / 1e6,  # Increasing energy
                    gpu_utilization=float(80 + (i % 20)),  # 80-99%
                    gpu_memory_used=15.0 + (i % 5),  # 15-19 GB
                ),
            )
        )

        # GPU 1 - Idle (RTX 6000)
        records.append(
            TelemetryRecord(
                timestamp_ns=timestamp + 1000,
                dcgm_url="http://localhost:9401/metrics",
                gpu_index=1,
                gpu_model_name="NVIDIA RTX 6000 Ada Generation",
                gpu_uuid="GPU-12345678-1234-1234-1234-123456789abc",
                pci_bus_id="00000000:03:00.0",
                device="nvidia1",
                hostname="ed7e7a5e585f",
                telemetry_data=TelemetryMetrics(
                    gpu_power_usage=42.0 + (i % 3),  # Idle power 42-44W
                    energy_consumption=(230000000 + (i * 500000))
                    / 1e6,  # Slower energy growth
                    gpu_utilization=0.0,
                    gpu_memory_used=0.0,
                ),
            )
        )

        # GPU 2 - Moderate workload (H100)
        records.append(
            TelemetryRecord(
                timestamp_ns=timestamp + 2000,
                dcgm_url="http://localhost:9401/metrics",
                gpu_index=2,
                gpu_model_name="NVIDIA H100 PCIe",
                gpu_uuid="GPU-87654321-4321-4321-4321-cba987654321",
                pci_bus_id="00000000:04:00.0",
                device="nvidia2",
                hostname="ed7e7a5e585f",
                telemetry_data=TelemetryMetrics(
                    gpu_power_usage=200.0 + (i % 50),  # Higher power 200-249W
                    energy_consumption=(250000000 + (i * 3000000)) / 1e6,
                    gpu_utilization=float(50 + (i % 30)),  # 50-79%
                    gpu_memory_used=40.0 + (i % 10),  # 40-49 GB
                ),
            )
        )

    return records
