# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from rich.text import Text

from aiperf.common.models import MetricResult
from aiperf.ui.dashboard.realtime_telemetry_dashboard import (
    GPUMetricsTable,
    SingleNodeView,
)


class TestGPUMetricsTable:
    """Test utility methods in GPUMetricsTable."""

    @pytest.fixture
    def gpu_metrics_table(self):
        """Create a GPUMetricsTable instance for testing."""
        return GPUMetricsTable(
            endpoint="localhost:9400",
            gpu_uuid="GPU-12345678-90ab",
            gpu_index=0,
            model_name="NVIDIA RTX 4090",
        )

    def test_is_for_this_gpu_matching(self, gpu_metrics_table):
        """Test that _is_for_this_gpu returns True for matching GPU."""
        metric = MetricResult(
            tag="gpu_util_dcgm_http___localhost_9400_metrics_gpu0_GPU-12345678",
            header="GPU Utilization | localhost:9400 | GPU 0 | NVIDIA RTX 4090",
            unit="%",
            avg=75.0,
        )

        assert gpu_metrics_table._is_for_this_gpu(metric) is True

    def test_is_for_this_gpu_non_matching_index(self, gpu_metrics_table):
        """Test that _is_for_this_gpu returns False for different GPU index."""
        metric = MetricResult(
            tag="gpu_util_dcgm_http___localhost_9400_metrics_gpu1_GPU-12345678",
            header="GPU Utilization | localhost:9400 | GPU 1 | NVIDIA RTX 4090",
            unit="%",
            avg=75.0,
        )

        assert gpu_metrics_table._is_for_this_gpu(metric) is False

    def test_is_for_this_gpu_non_matching_uuid(self, gpu_metrics_table):
        """Test that _is_for_this_gpu returns False for different UUID."""
        metric = MetricResult(
            tag="gpu_util_dcgm_http___localhost_9400_metrics_gpu0_GPU-99999999",
            header="GPU Utilization | localhost:9400 | GPU 0 | NVIDIA RTX 4090",
            unit="%",
            avg=75.0,
        )

        assert gpu_metrics_table._is_for_this_gpu(metric) is False

    def test_format_metric_row_with_all_stats(self, gpu_metrics_table):
        """Test _format_metric_row formats all statistics correctly."""
        metric = MetricResult(
            tag="gpu_util_dcgm_http___localhost_9400_metrics_gpu0_GPU-12345678",
            header="GPU Power Usage | localhost:9400 | GPU 0 | NVIDIA RTX 4090",
            unit="W",
            current=250.5,
            avg=245.0,
            min=200.0,
            max=300.0,
            p99=290.0,
            p90=280.0,
            p50=245.0,
            std=15.5,
        )

        row_cells = gpu_metrics_table._format_metric_row(metric)

        # Should have 9 cells: metric name + 8 stats
        assert len(row_cells) == 9

        # First cell should be the metric name (before |)
        assert row_cells[0].plain == "GPU Power Usage"

        # All cells should be Text objects
        assert all(isinstance(cell, Text) for cell in row_cells)

    def test_format_metric_row_simple_header(self, gpu_metrics_table):
        """Test _format_metric_row with simple header (no | separator)."""
        metric = MetricResult(
            tag="simple_metric",
            header="Simple Metric",
            unit="ms",
            avg=10.0,
        )

        row_cells = gpu_metrics_table._format_metric_row(metric)

        # First cell should be the full header
        assert row_cells[0].plain == "Simple Metric"

    def test_format_value_none(self, gpu_metrics_table):
        """Test _format_value with None returns 'N/A'."""
        result = gpu_metrics_table._format_value(None)

        assert isinstance(result, Text)
        assert result.plain == "N/A"
        assert result.style == "dim"

    def test_format_value_small_number(self, gpu_metrics_table):
        """Test _format_value with small number (< 1,000,000)."""
        result = gpu_metrics_table._format_value(1234.567)

        assert isinstance(result, Text)
        assert result.plain == "1,234.57"
        assert result.style == "green"

    def test_format_value_large_number(self, gpu_metrics_table):
        """Test _format_value with large number (>= 1,000,000) uses scientific notation."""
        result = gpu_metrics_table._format_value(1234567.89)

        assert isinstance(result, Text)
        assert result.plain == "1.23e+06"
        assert result.style == "green"

    def test_format_value_zero(self, gpu_metrics_table):
        """Test _format_value with zero."""
        result = gpu_metrics_table._format_value(0.0)

        assert isinstance(result, Text)
        assert result.plain == "0.00"

    def test_format_value_negative(self, gpu_metrics_table):
        """Test _format_value with negative number."""
        result = gpu_metrics_table._format_value(-123.45)

        assert isinstance(result, Text)
        assert result.plain == "-123.45"

    def test_format_value_non_numeric(self, gpu_metrics_table):
        """Test _format_value with non-numeric value."""
        result = gpu_metrics_table._format_value("text_value")

        assert isinstance(result, Text)
        assert result.plain == "text_value"


class TestSingleNodeView:
    """Test utility methods in SingleNodeView."""

    @pytest.fixture
    def single_node_view(self):
        """Create a SingleNodeView instance for testing."""
        return SingleNodeView()

    def test_group_metrics_by_gpu_single_gpu(self, single_node_view):
        """Test _group_metrics_by_gpu with metrics from a single GPU."""
        metrics = [
            MetricResult(
                tag="gpu_util_dcgm_http___localhost_9400_metrics_gpu0_GPU-12345678",
                header="GPU Utilization | localhost:9400 | GPU 0 | Model",
                unit="%",
                avg=75.0,
            ),
            MetricResult(
                tag="gpu_memory_dcgm_http___localhost_9400_metrics_gpu0_GPU-12345678",
                header="GPU Memory | localhost:9400 | GPU 0 | Model",
                unit="GB",
                avg=8.5,
            ),
        ]

        grouped = single_node_view._group_metrics_by_gpu(metrics)

        # Should have 1 GPU group
        assert len(grouped) == 1

        # Both metrics should be in the same group
        gpu_key = list(grouped.keys())[0]
        assert len(grouped[gpu_key]) == 2

    def test_group_metrics_by_gpu_multiple_gpus(self, single_node_view):
        """Test _group_metrics_by_gpu with metrics from multiple GPUs."""
        metrics = [
            MetricResult(
                tag="gpu_util_dcgm_http___localhost_9400_metrics_gpu0_GPU-12345678",
                header="GPU Utilization | localhost:9400 | GPU 0 | Model",
                unit="%",
                avg=75.0,
            ),
            MetricResult(
                tag="gpu_util_dcgm_http___localhost_9400_metrics_gpu1_GPU-87654321",
                header="GPU Utilization | localhost:9400 | GPU 1 | Model",
                unit="%",
                avg=80.0,
            ),
            MetricResult(
                tag="gpu_memory_dcgm_http___localhost_9400_metrics_gpu0_GPU-12345678",
                header="GPU Memory | localhost:9400 | GPU 0 | Model",
                unit="GB",
                avg=8.5,
            ),
        ]

        grouped = single_node_view._group_metrics_by_gpu(metrics)

        # Should have 2 GPU groups
        assert len(grouped) == 2

        # Verify metrics are grouped correctly
        all_metrics = [m for group in grouped.values() for m in group]
        assert len(all_metrics) == 3

    def test_group_metrics_by_gpu_empty_list(self, single_node_view):
        """Test _group_metrics_by_gpu with empty metrics list."""
        grouped = single_node_view._group_metrics_by_gpu([])

        assert grouped == {}

    def test_extract_gpu_key_from_tag_valid(self, single_node_view):
        """Test _extract_gpu_key_from_tag with valid tag."""
        tag = "gpu_util_dcgm_http___localhost_9400_metrics_gpu0_GPU-12345678"

        gpu_key = single_node_view._extract_gpu_key_from_tag(tag)

        # Should extract everything after _dcgm_ with _gpu replaced by _
        assert "http___localhost_9400_metrics" in gpu_key
        assert "0_GPU-12345678" in gpu_key

    def test_extract_gpu_key_from_tag_no_dcgm(self, single_node_view):
        """Test _extract_gpu_key_from_tag with tag missing _dcgm_."""
        tag = "simple_metric_tag"

        gpu_key = single_node_view._extract_gpu_key_from_tag(tag)

        assert gpu_key == "unknown"

    def test_extract_gpu_key_from_tag_no_gpu(self, single_node_view):
        """Test _extract_gpu_key_from_tag with tag missing _gpu."""
        tag = "metric_dcgm_http___localhost_9400_metrics"

        gpu_key = single_node_view._extract_gpu_key_from_tag(tag)

        assert gpu_key == "unknown"

    def test_extract_gpu_info_full_header(self, single_node_view):
        """Test _extract_gpu_info with complete header and tag."""
        metric = MetricResult(
            tag="gpu_util_dcgm_http___localhost_9400_metrics_gpu0_GPU-12345678",
            header="GPU Power Usage | localhost:9400 | GPU 0 | NVIDIA RTX 4090",
            unit="W",
            avg=250.0,
        )

        endpoint, gpu_index, gpu_uuid, model_name = single_node_view._extract_gpu_info(
            metric
        )

        assert endpoint == "localhost:9400"
        assert gpu_index == 0
        assert gpu_uuid == "GPU-12345678"
        assert model_name == "NVIDIA RTX 4090"

    def test_extract_gpu_info_incomplete_header(self, single_node_view):
        """Test _extract_gpu_info with incomplete header uses defaults."""
        metric = MetricResult(
            tag="simple_metric_tag",
            header="Simple Metric",
            unit="ms",
            avg=10.0,
        )

        endpoint, gpu_index, gpu_uuid, model_name = single_node_view._extract_gpu_info(
            metric
        )

        # Should use default values
        assert endpoint == "unknown"
        assert gpu_index == 0
        assert model_name == "GPU"

    def test_extract_gpu_info_different_gpu_index(self, single_node_view):
        """Test _extract_gpu_info correctly parses different GPU indices."""
        metric = MetricResult(
            tag="gpu_util_dcgm_http___localhost_9401_metrics_gpu7_UUID-999",
            header="GPU Utilization | localhost:9401 | GPU 7 | Tesla V100",
            unit="%",
            avg=85.0,
        )

        endpoint, gpu_index, gpu_uuid, model_name = single_node_view._extract_gpu_info(
            metric
        )

        assert endpoint == "localhost:9401"
        assert gpu_index == 7
        assert gpu_uuid == "UUID-999"
        assert model_name == "Tesla V100"

    def test_extract_gpu_info_uuid_from_tag(self, single_node_view):
        """Test _extract_gpu_info extracts UUID from tag (last part)."""
        metric = MetricResult(
            tag="metric_name_dcgm_endpoint_gpu0_my-custom-uuid",
            header="Metric | endpoint | GPU 0 | Model",
            unit="ms",
            avg=10.0,
        )

        endpoint, gpu_index, gpu_uuid, model_name = single_node_view._extract_gpu_info(
            metric
        )

        assert gpu_uuid == "my-custom-uuid"

    def test_group_metrics_preserves_order(self, single_node_view):
        """Test that _group_metrics_by_gpu preserves metric order within groups."""
        metrics = [
            MetricResult(
                tag="metric1_dcgm_endpoint_gpu0_uuid",
                header="Metric 1 | endpoint | GPU 0 | Model",
                unit="ms",
                avg=10.0,
            ),
            MetricResult(
                tag="metric2_dcgm_endpoint_gpu0_uuid",
                header="Metric 2 | endpoint | GPU 0 | Model",
                unit="ms",
                avg=20.0,
            ),
            MetricResult(
                tag="metric3_dcgm_endpoint_gpu0_uuid",
                header="Metric 3 | endpoint | GPU 0 | Model",
                unit="ms",
                avg=30.0,
            ),
        ]

        grouped = single_node_view._group_metrics_by_gpu(metrics)

        # All metrics should be in one group and maintain order
        gpu_key = list(grouped.keys())[0]
        assert [m.header.split(" | ")[0] for m in grouped[gpu_key]] == [
            "Metric 1",
            "Metric 2",
            "Metric 3",
        ]
