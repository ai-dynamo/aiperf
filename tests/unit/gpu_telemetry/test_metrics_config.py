# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for GPU telemetry metrics configuration."""

import tempfile
from pathlib import Path

import pytest

from aiperf.common.enums import (
    EnergyMetricUnit,
    FrequencyMetricUnit,
    GenericMetricUnit,
    MetricSizeUnit,
    MetricTimeUnit,
    PowerMetricUnit,
    TemperatureMetricUnit,
)
from aiperf.gpu_telemetry.constants import (
    DCGM_TO_FIELD_MAPPING,
    GPU_TELEMETRY_METRICS_CONFIG,
)
from aiperf.gpu_telemetry.metrics_config import MetricsConfigLoader


class TestMetricsConfigLoader:
    """Tests for MetricsConfigLoader class."""

    @pytest.fixture(autouse=True)
    def reset_global_state(self):
        """Reset global state before each test for isolation."""
        original_mapping = DCGM_TO_FIELD_MAPPING.copy()
        original_config = GPU_TELEMETRY_METRICS_CONFIG.copy()
        yield
        DCGM_TO_FIELD_MAPPING.clear()
        DCGM_TO_FIELD_MAPPING.update(original_mapping)
        GPU_TELEMETRY_METRICS_CONFIG.clear()
        GPU_TELEMETRY_METRICS_CONFIG.extend(original_config)

    def test_parse_valid_csv(self):
        """Test parsing a valid DCGM metrics CSV file."""
        csv_content = """# Test CSV
# Comment line
DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL, gauge, NVLink bandwidth (in KB/s)
DCGM_FI_PROF_PIPE_TENSOR_ACTIVE, gauge, Tensor core active (in %)
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            csv_path = Path(f.name)

        try:
            loader = MetricsConfigLoader()
            metrics = loader.parse_custom_metrics_csv(csv_path)

            assert len(metrics) == 2
            assert metrics[0] == (
                "DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL",
                "gauge",
                "NVLink bandwidth (in KB/s)",
            )
            assert metrics[1] == (
                "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE",
                "gauge",
                "Tensor core active (in %)",
            )
        finally:
            csv_path.unlink()

    def test_parse_csv_skips_comments_and_empty_lines(self):
        """Test that parser skips comment lines and empty lines."""
        csv_content = """# Comment 1

DCGM_FI_DEV_POWER_USAGE, gauge, Power (in W)

# Comment 2
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            csv_path = Path(f.name)

        try:
            loader = MetricsConfigLoader()
            metrics = loader.parse_custom_metrics_csv(csv_path)

            assert len(metrics) == 1
            assert metrics[0][0] == "DCGM_FI_DEV_POWER_USAGE"
        finally:
            csv_path.unlink()

    def test_parse_csv_validates_dcgm_field_prefix(self):
        """Test that parser warns on invalid DCGM field names."""
        csv_content = """INVALID_FIELD, gauge, Should be skipped
DCGM_FI_DEV_GPU_UTIL, gauge, Valid field (in %)
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            csv_path = Path(f.name)

        try:
            loader = MetricsConfigLoader()
            metrics = loader.parse_custom_metrics_csv(csv_path)

            # Only the valid DCGM field should be included
            assert len(metrics) == 1
            assert metrics[0][0] == "DCGM_FI_DEV_GPU_UTIL"
        finally:
            csv_path.unlink()

    def test_parse_csv_validates_metric_type(self):
        """Test that parser warns on invalid metric types."""
        csv_content = """DCGM_FI_DEV_GPU_UTIL, invalid_type, Invalid metric type
DCGM_FI_DEV_POWER_USAGE, gauge, Valid metric (in W)
DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION, counter, Valid counter (in MJ)
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            csv_path = Path(f.name)

        try:
            loader = MetricsConfigLoader()
            metrics = loader.parse_custom_metrics_csv(csv_path)

            # Only valid metric types should be included
            assert len(metrics) == 2
            assert metrics[0][1] == "gauge"
            assert metrics[1][1] == "counter"
        finally:
            csv_path.unlink()

    def test_infer_unit_from_help_message(self):
        """Test unit inference from help messages."""
        loader = MetricsConfigLoader()

        # Test various units
        assert loader._infer_unit_from_help("Power (in W)") == PowerMetricUnit.WATT
        assert (
            loader._infer_unit_from_help("Utilization (in %)")
            == GenericMetricUnit.PERCENT
        )
        assert (
            loader._infer_unit_from_help("Memory (in GB)") == MetricSizeUnit.GIGABYTES
        )
        assert (
            loader._infer_unit_from_help("Memory (in MB)") == MetricSizeUnit.MEGABYTES
        )
        assert (
            loader._infer_unit_from_help("Memory (in KB)") == MetricSizeUnit.KILOBYTES
        )
        assert (
            loader._infer_unit_from_help("Frequency (in MHz)")
            == FrequencyMetricUnit.MEGAHERTZ
        )
        assert (
            loader._infer_unit_from_help("Temperature (in °C)")
            == TemperatureMetricUnit.CELSIUS
        )
        assert (
            loader._infer_unit_from_help("Temperature (in C)")
            == TemperatureMetricUnit.CELSIUS
        )
        assert (
            loader._infer_unit_from_help("Errors (in count)") == GenericMetricUnit.COUNT
        )
        assert (
            loader._infer_unit_from_help("Time (in us)") == MetricTimeUnit.MICROSECONDS
        )
        assert (
            loader._infer_unit_from_help("Time (in ms)") == MetricTimeUnit.MILLISECONDS
        )
        assert (
            loader._infer_unit_from_help("Energy (in MJ)") == EnergyMetricUnit.MEGAJOULE
        )

        # Test no unit
        assert loader._infer_unit_from_help("No unit here") == GenericMetricUnit.COUNT

    def test_build_custom_metrics_deduplication(self):
        """Test that metrics already in display config are skipped."""
        # CSV with mix of metrics: some in display defaults, some not
        csv_content = """DCGM_FI_DEV_GPU_TEMP, gauge, GPU temperature (in °C)
DCGM_FI_DEV_POWER_USAGE, gauge, Power draw (in W)
DCGM_FI_DEV_SM_CLOCK, gauge, SM clock frequency (in MHz)
DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL, gauge, NVLink bandwidth (in KB/s)
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            csv_path = Path(f.name)

        try:
            loader = MetricsConfigLoader()

            # Build custom metrics - loader accesses constants directly and returns new mappings
            custom_metrics, new_dcgm_mappings = loader.build_custom_metrics_from_csv(
                custom_csv_path=csv_path
            )

            # Should return 2 metrics:
            # - DCGM_FI_DEV_GPU_TEMP → gpu_temperature (in 7 defaults, SKIPPED)
            # - DCGM_FI_DEV_POWER_USAGE → gpu_power_usage (in 7 defaults, SKIPPED)
            # - DCGM_FI_DEV_SM_CLOCK → sm_clock (NOT in 7 defaults, ADDED with auto-generated name)
            # - DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL → nvlink_bandwidth_total (new field, ADDED)
            assert len(custom_metrics) == 2
            custom_field_names = {m[1] for m in custom_metrics}
            assert "sm_clock" in custom_field_names
            assert "nvlink_bandwidth_total" in custom_field_names

            # Verify the new DCGM mappings were returned
            assert "DCGM_FI_DEV_SM_CLOCK" in new_dcgm_mappings
            assert new_dcgm_mappings["DCGM_FI_DEV_SM_CLOCK"] == "sm_clock"
            assert "DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL" in new_dcgm_mappings
            assert (
                new_dcgm_mappings["DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL"]
                == "nvlink_bandwidth_total"
            )

            # Apply mappings (simulating what cli_runner does)
            DCGM_TO_FIELD_MAPPING.update(new_dcgm_mappings)

            # Verify existing mappings were NOT changed
            assert DCGM_TO_FIELD_MAPPING["DCGM_FI_DEV_GPU_TEMP"] == "gpu_temperature"
            assert DCGM_TO_FIELD_MAPPING["DCGM_FI_DEV_POWER_USAGE"] == "gpu_power_usage"
        finally:
            csv_path.unlink()

    def test_build_custom_metrics_field_name_generation(self):
        """Test that field names are correctly generated from DCGM field names."""
        csv_content = """DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL, gauge, NVLink bandwidth (in KB/s)
DCGM_FI_PROF_PIPE_TENSOR_ACTIVE, gauge, Tensor active (in %)
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            csv_path = Path(f.name)

        try:
            loader = MetricsConfigLoader()

            custom_metrics, new_dcgm_mappings = loader.build_custom_metrics_from_csv(
                custom_csv_path=csv_path
            )

            # Verify field names
            assert len(custom_metrics) == 2
            assert custom_metrics[0][1] == "nvlink_bandwidth_total"
            assert custom_metrics[1][1] == "dcgm_fi_prof_pipe_tensor_active"

            # Verify display names extracted from help messages (title cased with acronyms)
            assert custom_metrics[0][0] == "NVLINK Bandwidth"
            assert custom_metrics[1][0] == "Tensor Active"

            # Verify units
            # Note: "KB/s" is not mapped, falls back to COUNT
            assert custom_metrics[0][2] == GenericMetricUnit.COUNT
            assert custom_metrics[1][2] == GenericMetricUnit.PERCENT
        finally:
            csv_path.unlink()

    def test_metrics_removed_from_defaults_are_added_from_csv(self):
        """Test that metrics in DCGM mapping but removed from display defaults are added."""
        # These are in DCGM_TO_FIELD_MAPPING but were removed from the 7 display defaults
        csv_content = """DCGM_FI_DEV_SM_CLOCK, gauge, SM clock (in MHz)
DCGM_FI_DEV_MEM_CLOCK, gauge, Memory clock (in MHz)
DCGM_FI_DEV_MEMORY_TEMP, gauge, Memory temp (in °C)
DCGM_FI_DEV_POWER_MGMT_LIMIT, gauge, Power limit (in W)
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            csv_path = Path(f.name)

        try:
            loader = MetricsConfigLoader()

            # Get existing field names from the 7 display defaults
            existing_field_names = {cfg[1] for cfg in GPU_TELEMETRY_METRICS_CONFIG}

            # These fields were removed from DCGM mapping and are not in the 7 display defaults
            # When added via CSV, they'll get auto-generated field names
            assert "sm_clock" not in existing_field_names
            assert "mem_clock" not in existing_field_names
            assert "memory_temp" not in existing_field_names
            assert "power_mgmt_limit" not in existing_field_names

            custom_metrics, new_dcgm_mappings = loader.build_custom_metrics_from_csv(
                custom_csv_path=csv_path
            )

            # All 4 should be ADDED with auto-generated field names
            assert len(custom_metrics) == 4
            custom_field_names = {m[1] for m in custom_metrics}
            assert custom_field_names == {
                "sm_clock",
                "mem_clock",
                "memory_temp",
                "power_mgmt_limit",
            }

            # Verify new DCGM mappings (only if they didn't already exist from previous tests)
            # Note: Tests may run in any order, so some may already be in mapping
            for dcgm_field, field_name in [
                ("DCGM_FI_DEV_SM_CLOCK", "sm_clock"),
                ("DCGM_FI_DEV_MEM_CLOCK", "mem_clock"),
                ("DCGM_FI_DEV_MEMORY_TEMP", "memory_temp"),
                ("DCGM_FI_DEV_POWER_MGMT_LIMIT", "power_mgmt_limit"),
            ]:
                if dcgm_field in new_dcgm_mappings:
                    assert new_dcgm_mappings[dcgm_field] == field_name

            # Apply mappings
            DCGM_TO_FIELD_MAPPING.update(new_dcgm_mappings)

            # After applying, all should be in the global mapping
            assert DCGM_TO_FIELD_MAPPING["DCGM_FI_DEV_SM_CLOCK"] == "sm_clock"
            assert DCGM_TO_FIELD_MAPPING["DCGM_FI_DEV_MEM_CLOCK"] == "mem_clock"
            assert DCGM_TO_FIELD_MAPPING["DCGM_FI_DEV_MEMORY_TEMP"] == "memory_temp"
            assert (
                DCGM_TO_FIELD_MAPPING["DCGM_FI_DEV_POWER_MGMT_LIMIT"]
                == "power_mgmt_limit"
            )

            # Verify the auto-generated field names are correct
            for metric in custom_metrics:
                display_name, field_name, unit = metric
                assert field_name in [
                    "sm_clock",
                    "mem_clock",
                    "memory_temp",
                    "power_mgmt_limit",
                ]
        finally:
            csv_path.unlink()

    def test_actual_custom_gpu_metrics_csv(self):
        """Test with the actual custom_gpu_metrics.csv file if it exists."""
        csv_path = Path("custom_gpu_metrics.csv")

        if not csv_path.exists():
            pytest.skip("custom_gpu_metrics.csv not found in current directory")

        loader = MetricsConfigLoader()

        # Get existing field names from defaults (7 metrics)
        initial_count = len(GPU_TELEMETRY_METRICS_CONFIG)

        # Build custom metrics
        custom_metrics, new_dcgm_mappings = loader.build_custom_metrics_from_csv(
            custom_csv_path=csv_path
        )

        # Verify we got some custom metrics
        assert len(custom_metrics) > 0, "Should have parsed some custom metrics"

        # Verify no duplicates with defaults
        custom_field_names = {m[1] for m in custom_metrics}
        default_field_names = {cfg[1] for cfg in GPU_TELEMETRY_METRICS_CONFIG}
        overlaps = custom_field_names & default_field_names
        assert len(overlaps) == 0, (
            f"Custom metrics should not overlap with defaults. Found: {overlaps}"
        )

        # Verify all custom metrics have valid structure
        for display_name, field_name, unit in custom_metrics:
            assert isinstance(display_name, str) and len(display_name) > 0
            assert isinstance(field_name, str) and len(field_name) > 0
            assert unit is not None

        # Verify new DCGM mappings were returned
        assert len(new_dcgm_mappings) > 0, "Should have some new DCGM mappings"

        # Apply mappings
        DCGM_TO_FIELD_MAPPING.update(new_dcgm_mappings)

        # Log results for debugging
        print("\nTest Results:")
        print(f"  Default metrics: {initial_count}")
        print(f"  Custom metrics parsed: {len(custom_metrics)}")
        print(f"  Total after extension: {initial_count + len(custom_metrics)}")
        print(f"  Custom field names: {sorted(custom_field_names)}")

    def test_parse_csv_with_invalid_column_count(self):
        """Test that parser skips lines with invalid column count."""
        csv_content = """DCGM_FI_DEV_POWER_USAGE, gauge
DCGM_FI_DEV_GPU_TEMP, gauge, Valid field (in C)
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            csv_path = Path(f.name)

        try:
            loader = MetricsConfigLoader()
            metrics = loader.parse_custom_metrics_csv(csv_path)

            assert len(metrics) == 1
            assert metrics[0][0] == "DCGM_FI_DEV_GPU_TEMP"
        finally:
            csv_path.unlink()

    def test_build_custom_metrics_from_csv_file_error(self):
        """Test error handling when CSV file cannot be parsed."""

        loader = MetricsConfigLoader()
        non_existent_path = Path("/nonexistent/path/metrics.csv")

        custom_metrics, new_dcgm_mappings = loader.build_custom_metrics_from_csv(
            custom_csv_path=non_existent_path
        )

        assert custom_metrics == []
        assert new_dcgm_mappings == {}

    def test_build_custom_metrics_from_csv_empty_result(self):
        """Test handling of CSV file that results in no valid metrics."""
        csv_content = """# All comments
# No valid metrics
INVALID_FIELD, gauge, Should be skipped
DCGM_FI_DEV_GPU_UTIL, invalid_type, Should be skipped
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            csv_path = Path(f.name)

        try:
            loader = MetricsConfigLoader()

            custom_metrics, new_dcgm_mappings = loader.build_custom_metrics_from_csv(
                custom_csv_path=csv_path
            )

            assert custom_metrics == []
            assert new_dcgm_mappings == {}
        finally:
            csv_path.unlink()
