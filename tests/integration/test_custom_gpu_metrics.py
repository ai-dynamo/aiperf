# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for custom GPU metrics CSV loading functionality."""

import platform
from pathlib import Path

import pytest

from aiperf.common.enums import (
    FrequencyMetricUnit,
    GenericMetricUnit,
    TemperatureMetricUnit,
)
from tests.harness.utils import AIPerfCLI, AIPerfMockServer

# DCGMFaker provides 8 of the 12 default metrics defined in GPU_TELEMETRY_METRICS_CONFIG.
# Missing from DCGMFaker: encoder_utilization, decoder_utilization, sm_utilization, jpg_utilization
DCGM_FAKER_DEFAULT_METRIC_COUNT = 8


@pytest.mark.skipif(
    platform.system() == "Darwin",
    reason="Requires NVIDIA GPUs for DCGM telemetry (only available on Linux CI).",
)
@pytest.mark.integration
@pytest.mark.asyncio
class TestCustomGpuMetrics:
    """Integration tests for custom GPU metrics CSV loading."""

    @pytest.fixture
    def custom_gpu_metrics_csv(self, tmp_path: Path) -> Path:
        """Create a custom GPU metrics CSV file for testing.

        Note: Only includes metrics that DCGMFaker actually returns.
        """
        csv_path = tmp_path / "custom_gpu_metrics.csv"
        csv_content = """# Custom GPU Metrics Test File
# Format: DCGM_FIELD, metric_type, help_message

# Custom clock metrics (DCGMFaker returns these)
DCGM_FI_DEV_SM_CLOCK, gauge, SM clock frequency (in MHz)
DCGM_FI_DEV_MEM_CLOCK, gauge, Memory clock frequency (in MHz)

# Custom temperature metrics (DCGMFaker returns this)
DCGM_FI_DEV_MEMORY_TEMP, gauge, Memory temperature (in °C)

# This is already a default metric (maps to mem_utilization), included to test deduplication
DCGM_FI_DEV_MEM_COPY_UTIL, gauge, Memory copy utilization (in %)
"""
        csv_path.write_text(csv_content)
        return csv_path

    @pytest.fixture
    def custom_gpu_metrics_csv_with_defaults(self, tmp_path: Path) -> Path:
        """Create a CSV with mix of default and custom metrics."""
        csv_path = tmp_path / "custom_gpu_metrics.csv"
        csv_content = """# Mix of default and custom metrics
# This should deduplicate the default metrics

# Default metrics (should be skipped to avoid duplicates)
DCGM_FI_DEV_GPU_UTIL, gauge, GPU utilization (in %)
DCGM_FI_DEV_POWER_USAGE, gauge, Power draw (in W)

# Custom metrics (should be added - DCGMFaker returns these)
DCGM_FI_DEV_SM_CLOCK, gauge, SM clock frequency (in MHz)
DCGM_FI_DEV_MEM_CLOCK, gauge, Memory clock frequency (in MHz)
"""
        csv_path.write_text(csv_content)
        return csv_path

    @pytest.fixture
    def custom_gpu_metrics_csv_invalid(self, tmp_path: Path) -> Path:
        """Create a CSV with some invalid entries."""
        csv_path = tmp_path / "custom_gpu_metrics.csv"
        csv_content = """# CSV with invalid entries for error handling tests

# Invalid entries (should be skipped)
INVALID_FIELD, gauge, Invalid field name
DCGM_FI_DEV_GPU_UTIL, invalid_type, Invalid metric type

# Valid entries (should be processed)
DCGM_FI_DEV_SM_CLOCK, gauge, SM clock frequency (in MHz)
"""
        csv_path.write_text(csv_content)
        return csv_path

    async def test_custom_metrics_csv_loading_basic(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        custom_gpu_metrics_csv: Path,
    ):
        """Test loading custom metrics from CSV and verifying they appear in output."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model nvidia/llama-3.1-nemotron-70b-instruct \
                --url {aiperf_mock_server.url} \
                --tokenizer gpt2 \
                --endpoint-type chat \
                --gpu-telemetry {custom_gpu_metrics_csv} {" ".join(aiperf_mock_server.dcgm_urls)} \
                --benchmark-duration 2 \
                --concurrency 2 \
                --workers-max 2
            """
        )

        assert result.request_count > 0
        assert result.has_gpu_telemetry
        assert result.json.telemetry_data.endpoints is not None
        assert len(result.json.telemetry_data.endpoints) > 0

        for dcgm_url in result.json.telemetry_data.endpoints:
            endpoint_data = result.json.telemetry_data.endpoints[dcgm_url]
            assert endpoint_data.gpus is not None
            assert len(endpoint_data.gpus) > 0

            for gpu_data in endpoint_data.gpus.values():
                assert gpu_data.metrics is not None

                # 8 defaults from DCGMFaker + 3 custom (sm_clock, mem_clock, memory_temp)
                # Note: DCGM_FI_DEV_MEM_COPY_UTIL maps to default "mem_utilization", not added as custom
                expected_min_metrics = DCGM_FAKER_DEFAULT_METRIC_COUNT + 3

                assert len(gpu_data.metrics) >= expected_min_metrics, (
                    f"Expected at least {expected_min_metrics} metrics, "
                    f"got {len(gpu_data.metrics)}"
                )

                # These are the actual custom metrics added (mem_copy_util is a default as mem_utilization)
                custom_metric_names = [
                    "sm_clock",
                    "mem_clock",
                    "memory_temp",
                ]
                for metric_name in custom_metric_names:
                    assert metric_name in gpu_data.metrics, (
                        f"Missing {metric_name}. Available metrics: {list(gpu_data.metrics.keys())}"
                    )

                for metric_name, metric_value in gpu_data.metrics.items():
                    assert metric_value is not None, (
                        f"Metric {metric_name} has None value"
                    )
                    assert metric_value.unit is not None, (
                        f"Metric {metric_name} has None unit"
                    )

                assert (
                    gpu_data.metrics["sm_clock"].unit
                    == FrequencyMetricUnit.MEGAHERTZ.value
                ), (
                    f"sm_clock unit is {gpu_data.metrics['sm_clock'].unit}, expected {FrequencyMetricUnit.MEGAHERTZ.value}"
                )
                assert (
                    gpu_data.metrics["mem_clock"].unit
                    == FrequencyMetricUnit.MEGAHERTZ.value
                ), (
                    f"mem_clock unit is {gpu_data.metrics['mem_clock'].unit}, expected {FrequencyMetricUnit.MEGAHERTZ.value}"
                )
                assert (
                    gpu_data.metrics["memory_temp"].unit
                    == TemperatureMetricUnit.CELSIUS.value
                ), (
                    f"memory_temp unit is {gpu_data.metrics['memory_temp'].unit}, expected {TemperatureMetricUnit.CELSIUS.value}"
                )
                # DCGM_FI_DEV_MEM_COPY_UTIL maps to default "mem_utilization" (not "mem_copy_util")
                assert (
                    gpu_data.metrics["mem_utilization"].unit
                    == GenericMetricUnit.PERCENT.value
                ), (
                    f"mem_utilization unit is {gpu_data.metrics['mem_utilization'].unit}, expected {GenericMetricUnit.PERCENT.value}"
                )

    async def test_custom_metrics_deduplication(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        custom_gpu_metrics_csv_with_defaults: Path,
    ):
        """Test that metrics already in defaults are not duplicated."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model nvidia/llama-3.1-nemotron-70b-instruct \
                --url {aiperf_mock_server.url} \
                --tokenizer gpt2 \
                --endpoint-type chat \
                --gpu-telemetry {custom_gpu_metrics_csv_with_defaults} {" ".join(aiperf_mock_server.dcgm_urls)} \
                --benchmark-duration 2 \
                --concurrency 2 \
                --workers-max 2
            """
        )

        assert result.has_gpu_telemetry
        assert result.json.telemetry_data.endpoints is not None

        for dcgm_url in result.json.telemetry_data.endpoints:
            endpoint_data = result.json.telemetry_data.endpoints[dcgm_url]
            for gpu_data in endpoint_data.gpus.values():
                metric_names = list(gpu_data.metrics.keys())
                unique_metric_names = set(metric_names)

                assert len(metric_names) == len(unique_metric_names), (
                    f"Found duplicate metrics. Metrics list: {metric_names}"
                )

                assert "gpu_utilization" in gpu_data.metrics
                assert "gpu_power_usage" in gpu_data.metrics

                assert "sm_clock" in gpu_data.metrics
                assert "mem_clock" in gpu_data.metrics

                # 8 defaults from DCGMFaker + 2 custom (sm_clock, mem_clock)
                # GPU_UTIL and POWER_USAGE from CSV are already defaults, so not added as custom
                expected_min_metrics = DCGM_FAKER_DEFAULT_METRIC_COUNT + 2

                assert len(gpu_data.metrics) >= expected_min_metrics

    async def test_invalid_csv_fallback_to_defaults(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        custom_gpu_metrics_csv_invalid: Path,
    ):
        """Test that invalid CSV entries are skipped gracefully."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model nvidia/llama-3.1-nemotron-70b-instruct \
                --url {aiperf_mock_server.url} \
                --tokenizer gpt2 \
                --endpoint-type chat \
                --gpu-telemetry {custom_gpu_metrics_csv_invalid} {" ".join(aiperf_mock_server.dcgm_urls)} \
                --benchmark-duration 2 \
                --concurrency 2 \
                --workers-max 2
            """
        )

        assert result.request_count > 0
        assert result.has_gpu_telemetry

        for dcgm_url in result.json.telemetry_data.endpoints:
            endpoint_data = result.json.telemetry_data.endpoints[dcgm_url]
            for gpu_data in endpoint_data.gpus.values():
                assert "sm_clock" in gpu_data.metrics

                # 8 defaults from DCGMFaker + 1 valid custom (sm_clock)
                expected_min_metrics = DCGM_FAKER_DEFAULT_METRIC_COUNT + 1
                assert len(gpu_data.metrics) >= expected_min_metrics

    async def test_nonexistent_csv_file_error(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer, tmp_path: Path
    ):
        """Test that nonexistent CSV file produces appropriate error."""
        nonexistent_csv = tmp_path / "nonexistent_custom_gpu_metrics.csv"

        result = await cli.run(
            f"""
            aiperf profile \
                --model nvidia/llama-3.1-nemotron-70b-instruct \
                --url {aiperf_mock_server.url} \
                --tokenizer gpt2 \
                --endpoint-type chat \
                --gpu-telemetry {nonexistent_csv} {" ".join(aiperf_mock_server.dcgm_urls)} \
                --request-count 10 \
                --concurrency 2
            """,
            assert_success=False,
        )

        assert result.exit_code != 0
