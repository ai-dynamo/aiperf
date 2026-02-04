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
from aiperf.gpu_telemetry.constants import (
    GPU_TELEMETRY_METRICS_CONFIG,
)
from tests.harness.utils import AIPerfCLI, AIPerfMockServer


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

# Custom utilization metric (DCGMFaker returns this)
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

                default_metric_count = len(GPU_TELEMETRY_METRICS_CONFIG)

                assert len(gpu_data.metrics) >= default_metric_count, (
                    f"Expected at least {default_metric_count} default metrics, "
                    f"got {len(gpu_data.metrics)}"
                )

                custom_metric_names = [
                    "sm_clock",
                    "mem_clock",
                    "memory_temp",
                    "mem_copy_util",
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
                assert (
                    gpu_data.metrics["mem_copy_util"].unit
                    == GenericMetricUnit.PERCENT.value
                ), (
                    f"mem_copy_util unit is {gpu_data.metrics['mem_copy_util'].unit}, expected {GenericMetricUnit.PERCENT.value}"
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

                default_metric_count = len(GPU_TELEMETRY_METRICS_CONFIG)
                custom_metrics_added = 2

                assert (
                    len(gpu_data.metrics) >= default_metric_count + custom_metrics_added
                )

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

                default_metric_count = len(GPU_TELEMETRY_METRICS_CONFIG)
                assert len(gpu_data.metrics) >= default_metric_count

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
