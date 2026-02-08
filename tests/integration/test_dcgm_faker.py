# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for DCGMFaker using real telemetry parsing logic."""

import pytest
from aiperf_mock_server.dcgm_faker import GPU_CONFIGS, DCGMFaker
from pytest import approx

from aiperf.gpu_telemetry.dcgm_collector import DCGMTelemetryCollector


class TestDCGMFaker:
    """Test DCGMFaker by parsing output with actual TelemetryDataCollector."""

    @pytest.mark.parametrize("gpu_name", GPU_CONFIGS.keys())
    def test_faker_output_parsed_by_real_telemetry_collector(self, gpu_name):
        """Test that faker output is parsed correctly by actual TelemetryDataCollector."""
        faker = DCGMFaker(gpu_name=gpu_name, num_gpus=2, seed=42, hostname="testnode")
        metrics_text = faker.generate()
        print(metrics_text)

        # Use real TelemetryDataCollector to parse the output
        collector = DCGMTelemetryCollector(dcgm_url="http://fake")
        records = collector._parse_metrics_to_records(metrics_text)

        # Should get 2 TelemetryRecord objects (one per GPU)
        assert len(records) == 2
        assert all(record is not None for record in records)

        # Verify GPU indices
        gpu_indices = {record.gpu_index for record in records}
        assert gpu_indices == {0, 1}

        # Verify metadata fields (TelemetryRecord inherits from GpuMetadata)
        for i, record in enumerate(records):
            gpu = faker.gpus[i]
            assert record.gpu_model_name == gpu.cfg.model
            assert record.hostname == faker.hostname
            assert record.gpu_uuid == gpu.uuid
            assert record.pci_bus_id == gpu.pci_bus_id
            assert record.device == gpu.device

            # Verify TelemetryMetrics are correctly scaled
            telemetry = record.telemetry_data
            assert telemetry is not None
            assert telemetry.gpu_power_usage == approx(gpu.power, abs=0.01)
            assert telemetry.gpu_utilization == approx(gpu.util, abs=0.01)
            assert telemetry.gpu_temperature == approx(gpu.temp, abs=0.01)
            assert telemetry.energy_consumption == approx(
                gpu.energy * 1e-9, abs=0.01
            )  # mJ to MJ
            assert telemetry.gpu_memory_used == approx(
                gpu.mem_used * 1.048576 * 1e-3, abs=0.01
            )  # MiB to GB
            assert telemetry.xid_errors == approx(gpu.xid, abs=0.01)
            assert telemetry.power_violation == approx(gpu.power_viol, abs=0.01)

            # Verify values are in reasonable ranges
            assert 0 <= telemetry.gpu_utilization <= 100
            assert 0 < telemetry.gpu_power_usage <= gpu.cfg.max_power_w
            assert 0 < telemetry.gpu_temperature <= 100

    def test_load_affects_telemetry_records(self):
        """Test that load changes affect TelemetryRecords when parsed by real collector."""
        faker = DCGMFaker(gpu_name="b200", num_gpus=1, seed=42)
        collector = DCGMTelemetryCollector(dcgm_url="http://fake")

        # Low load
        faker.set_load(0.1)
        low_metrics = faker.generate()
        low_records = collector._parse_metrics_to_records(low_metrics)
        low_telemetry = low_records[0].telemetry_data

        # High load
        faker.set_load(0.9)
        high_metrics = faker.generate()
        high_records = collector._parse_metrics_to_records(high_metrics)
        high_telemetry = high_records[0].telemetry_data

        # High load should produce higher values
        assert high_telemetry.gpu_power_usage > low_telemetry.gpu_power_usage
        assert high_telemetry.gpu_temperature > low_telemetry.gpu_temperature
        assert high_telemetry.gpu_utilization > low_telemetry.gpu_utilization
        assert high_telemetry.gpu_memory_used > low_telemetry.gpu_memory_used
