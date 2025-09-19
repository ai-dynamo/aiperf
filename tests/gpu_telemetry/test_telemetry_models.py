# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from aiperf.common.models.telemetry_models import TelemetryRecord


class TestTelemetryRecord:
    """Test TelemetryRecord model validation and data structure integrity.

    This test class focuses on Pydantic model validation, field requirements,
    and data structure correctness. It does NOT test parsing logic or metric
    extraction - those belong in other test files.
    """

    def test_telemetry_record_complete_creation(self):
        """Test creating a TelemetryRecord with all fields populated.

        Verifies that a fully-populated TelemetryRecord stores all fields correctly
        including both required fields (timestamp, dcgm_url, gpu_index, etc.) and
        optional metadata fields (pci_bus_id, device, hostname).
        """

        record = TelemetryRecord(
            timestamp_ns=1000000000,
            dcgm_url="http://localhost:9401/metrics",
            gpu_index=0,
            gpu_model_name="NVIDIA RTX 6000 Ada Generation",
            gpu_uuid="GPU-ef6ef310-f8e2-cef9-036e-8f12d59b5ffc",
            pci_bus_id="00000000:02:00.0",
            device="nvidia0",
            hostname="ed7e7a5e585f",
            gpu_power_usage=75.5,
            gpu_power_limit=300.0,
            energy_consumption=1000000000,
            gpu_utilization=85.0,
            gpu_memory_used=15.26,
            total_gpu_memory=48.0,
        )

        assert record.timestamp_ns == 1000000000
        assert record.dcgm_url == "http://localhost:9401/metrics"
        assert record.gpu_index == 0
        assert record.gpu_model_name == "NVIDIA RTX 6000 Ada Generation"
        assert record.gpu_uuid == "GPU-ef6ef310-f8e2-cef9-036e-8f12d59b5ffc"

        assert record.pci_bus_id == "00000000:02:00.0"
        assert record.device == "nvidia0"
        assert record.hostname == "ed7e7a5e585f"

        assert record.gpu_power_usage == 75.5
        assert record.gpu_power_limit == 300.0
        assert record.energy_consumption == 1000000000
        assert record.gpu_utilization == 85.0
        assert record.gpu_memory_used == 15.26
        assert record.total_gpu_memory == 48.0

    def test_telemetry_record_minimal_creation(self):
        """Test creating a TelemetryRecord with only required fields.

        Verifies that TelemetryRecord can be created with minimal required fields
        and that optional fields default to None. This tests the flexibility
        needed for varying DCGM response completeness.
        """

        record = TelemetryRecord(
            timestamp_ns=1000000000,
            dcgm_url="http://node2:9401/metrics",
            gpu_index=1,
            gpu_model_name="NVIDIA H100",
            gpu_uuid="GPU-00000000-0000-0000-0000-000000000001",
        )

        # Verify required fields are set
        assert record.timestamp_ns == 1000000000
        assert record.dcgm_url == "http://node2:9401/metrics"
        assert record.gpu_index == 1
        assert record.gpu_model_name == "NVIDIA H100"
        assert record.gpu_uuid == "GPU-00000000-0000-0000-0000-000000000001"

        assert record.pci_bus_id is None
        assert record.device is None
        assert record.hostname is None
        assert record.gpu_power_usage is None
        assert record.gpu_power_limit is None
        assert record.energy_consumption is None
        assert record.gpu_utilization is None
        assert record.gpu_memory_used is None
        assert record.total_gpu_memory is None

    def test_telemetry_record_field_validation(self):
        """Test Pydantic validation of required fields.

        Verifies that TelemetryRecord enforces required field validation
        and raises appropriate validation errors when required fields
        are missing. Tests the data integrity guarantees.
        """

        record = TelemetryRecord(
            timestamp_ns=1000000000,
            dcgm_url="http://localhost:9401/metrics",
            gpu_index=0,
            gpu_model_name="NVIDIA RTX 6000",
            gpu_uuid="GPU-test-uuid",
        )
        assert record.timestamp_ns == 1000000000

        with pytest.raises(ValidationError):  # Pydantic validation error
            TelemetryRecord()  # No fields provided

    def test_telemetry_record_metadata_structure(self):
        """Test the hierarchical metadata structure for GPU identification.

        Verifies that TelemetryRecord properly supports the hierarchical
        identification structure needed for telemetry organization:
        dcgm_url -> gpu_uuid -> metadata. This structure enables proper
        grouping and filtering in the dashboard.
        """

        record = TelemetryRecord(
            timestamp_ns=1000000000,
            dcgm_url="http://gpu-node-01:9401/metrics",
            gpu_index=0,
            gpu_model_name="NVIDIA RTX 6000 Ada Generation",
            gpu_uuid="GPU-ef6ef310-f8e2-cef9-036e-8f12d59b5ffc",
            pci_bus_id="00000000:02:00.0",
            device="nvidia0",
            hostname="gpu-node-01",
        )

        # Verify hierarchical identification works
        # Level 1: DCGM endpoint identification
        assert record.dcgm_url == "http://gpu-node-01:9401/metrics"

        # Level 2: Unique GPU identification
        assert record.gpu_uuid == "GPU-ef6ef310-f8e2-cef9-036e-8f12d59b5ffc"

        # Level 3: Human-readable metadata
        assert record.gpu_index == 0  # For display ordering
        assert record.gpu_model_name == "NVIDIA RTX 6000 Ada Generation"
        assert record.hostname == "gpu-node-01"

        # Level 4: Hardware-specific metadata
        assert record.pci_bus_id == "00000000:02:00.0"
        assert record.device == "nvidia0"
