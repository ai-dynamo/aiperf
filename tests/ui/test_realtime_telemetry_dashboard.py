# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

import pytest
from rich.text import Text
from textual.widgets.data_table import RowDoesNotExist

from aiperf.common.config.service_config import ServiceConfig
from aiperf.common.models import MetricResult
from aiperf.ui.dashboard.realtime_telemetry_dashboard import (
    GPUMetricsTable,
    RealtimeTelemetryDashboard,
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

        # First cell should be the metric name with unit
        assert row_cells[0].plain == "GPU Power Usage (W)"

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

        # First cell should be the full header with unit
        assert row_cells[0].plain == "Simple Metric (ms)"

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


class TestGPUMetricsTableLifecycle:
    """Test lifecycle methods of GPUMetricsTable."""

    @pytest.fixture
    def gpu_metrics_table(self):
        """Create a GPUMetricsTable instance for testing."""
        return GPUMetricsTable(
            endpoint="localhost:9400",
            gpu_uuid="GPU-12345678-90ab",
            gpu_index=0,
            model_name="NVIDIA RTX 4090",
        )

    def test_compose_creates_widgets(self, gpu_metrics_table):
        """Test that compose yields the correct widgets."""
        widgets = list(gpu_metrics_table.compose())

        assert len(widgets) == 2
        assert widgets[0].__class__.__name__ == "Static"
        assert widgets[1].__class__.__name__ == "NonFocusableDataTable"

    def test_initialize_columns(self, gpu_metrics_table):
        """Test that _initialize_columns sets up all columns correctly."""
        mock_table = Mock()
        mock_column_key = Mock()
        mock_table.add_column.return_value = mock_column_key

        gpu_metrics_table.data_table = mock_table
        gpu_metrics_table._initialize_columns()

        assert mock_table.add_column.call_count == len(gpu_metrics_table.COLUMNS)
        assert gpu_metrics_table._columns_initialized is True
        assert len(gpu_metrics_table._column_keys) == len(gpu_metrics_table.COLUMNS)

    def test_update_with_no_data_table(self, gpu_metrics_table):
        """Test update method when data_table is None."""
        gpu_metrics_table.data_table = None
        metrics = [
            MetricResult(
                tag="gpu_util_dcgm_http___localhost_9400_metrics_gpu0_GPU-12345678",
                header="GPU Utilization | localhost:9400 | GPU 0 | NVIDIA RTX 4090",
                unit="%",
                avg=75.0,
            )
        ]

        gpu_metrics_table.update(metrics)

    def test_update_with_unmounted_table(self, gpu_metrics_table):
        """Test update method when data_table is not mounted."""
        mock_table = Mock()
        mock_table.is_mounted = False
        gpu_metrics_table.data_table = mock_table

        metrics = [
            MetricResult(
                tag="gpu_util_dcgm_http___localhost_9400_metrics_gpu0_GPU-12345678",
                header="GPU Utilization | localhost:9400 | GPU 0 | NVIDIA RTX 4090",
                unit="%",
                avg=75.0,
            )
        ]

        gpu_metrics_table.update(metrics)
        mock_table.add_row.assert_not_called()

    def test_update_adds_new_row(self, gpu_metrics_table):
        """Test that update adds a new row for a new metric."""
        mock_table = Mock()
        mock_table.is_mounted = True
        mock_column_key = Mock()
        mock_row_key = Mock()
        mock_table.add_column.return_value = mock_column_key
        mock_table.add_row.return_value = mock_row_key

        gpu_metrics_table.data_table = mock_table
        gpu_metrics_table._initialize_columns()

        metrics = [
            MetricResult(
                tag="gpu_util_dcgm_http___localhost_9400_metrics_gpu0_GPU-12345678",
                header="GPU Utilization | localhost:9400 | GPU 0 | NVIDIA RTX 4090",
                unit="%",
                current=75.0,
                avg=75.0,
                min=70.0,
                max=80.0,
                p99=79.0,
                p90=78.0,
                p50=75.0,
                std=2.5,
            )
        ]

        gpu_metrics_table.update(metrics)

        mock_table.add_row.assert_called_once()
        assert (
            gpu_metrics_table._metric_row_keys[
                "gpu_util_dcgm_http___localhost_9400_metrics_gpu0_GPU-12345678"
            ]
            == mock_row_key
        )

    def test_update_single_row_with_exception(self, gpu_metrics_table):
        """Test that _update_single_row handles exceptions gracefully."""
        mock_table = Mock()
        mock_table.update_cell.side_effect = Exception("Update failed")
        gpu_metrics_table.data_table = mock_table

        mock_row_key = Mock()
        mock_column_key = Mock()
        gpu_metrics_table._column_keys = {
            col: mock_column_key for col in gpu_metrics_table.COLUMNS
        }

        row_cells = [Text("Test", justify="left") for _ in gpu_metrics_table.COLUMNS]

        gpu_metrics_table._update_single_row(row_cells, mock_row_key)

    def test_update_handles_row_does_not_exist(self, gpu_metrics_table):
        """Test that update handles RowDoesNotExist exception and re-adds row."""
        mock_table = Mock()
        mock_table.is_mounted = True
        mock_column_key = Mock()
        mock_row_key = Mock()
        mock_table.add_column.return_value = mock_column_key
        mock_table.add_row.return_value = mock_row_key

        mock_table.get_row_index.side_effect = RowDoesNotExist("Row not found")

        gpu_metrics_table.data_table = mock_table
        gpu_metrics_table._initialize_columns()

        metrics = [
            MetricResult(
                tag="gpu_util_dcgm_http___localhost_9400_metrics_gpu0_GPU-12345678",
                header="GPU Utilization | localhost:9400 | GPU 0 | NVIDIA RTX 4090",
                unit="%",
                current=75.0,
                avg=75.0,
                min=70.0,
                max=80.0,
                p99=79.0,
                p90=78.0,
                p50=75.0,
                std=2.5,
            )
        ]

        gpu_metrics_table._metric_row_keys[
            "gpu_util_dcgm_http___localhost_9400_metrics_gpu0_GPU-12345678"
        ] = Mock()

        gpu_metrics_table.update(metrics)

        assert mock_table.add_row.call_count == 1


class TestSingleNodeViewLifecycle:
    """Test lifecycle methods of SingleNodeView."""

    @pytest.fixture
    def single_node_view(self):
        """Create a SingleNodeView instance for testing."""
        return SingleNodeView()

    def test_compose_yields_nothing(self, single_node_view):
        """Test that compose yields nothing initially (GPU tables added dynamically)."""
        widgets = list(single_node_view.compose())
        assert len(widgets) == 0

    def test_update_creates_gpu_table_when_mounted(self, single_node_view):
        """Test that update creates GPU tables for new GPUs."""
        with (
            patch.object(
                type(single_node_view),
                "is_mounted",
                new_callable=lambda: property(lambda self: True),
            ),
            patch.object(single_node_view, "mount") as mock_mount,
        ):
            metrics = [
                MetricResult(
                    tag="gpu_util_dcgm_http___localhost_9400_metrics_gpu0_GPU-12345678",
                    header="GPU Utilization | localhost:9400 | GPU 0 | NVIDIA RTX 4090",
                    unit="%",
                    avg=75.0,
                )
            ]

            single_node_view.update(metrics)

            assert len(single_node_view.gpu_tables) == 1
            mock_mount.assert_called_once()

    def test_update_skips_creation_when_unmounted(self, single_node_view):
        """Test that update doesn't create GPU tables when not mounted."""
        with patch.object(
            type(single_node_view),
            "is_mounted",
            new_callable=lambda: property(lambda self: False),
        ):
            metrics = [
                MetricResult(
                    tag="gpu_util_dcgm_http___localhost_9400_metrics_gpu0_GPU-12345678",
                    header="GPU Utilization | localhost:9400 | GPU 0 | NVIDIA RTX 4090",
                    unit="%",
                    avg=75.0,
                )
            ]

            single_node_view.update(metrics)

            assert len(single_node_view.gpu_tables) == 0

    def test_update_updates_existing_gpu_table(self, single_node_view):
        """Test that update calls update on existing GPU tables."""
        mock_gpu_table = Mock()
        gpu_key = "http___localhost_9400_metrics_0_GPU-12345678"
        single_node_view.gpu_tables[gpu_key] = mock_gpu_table

        with patch.object(
            type(single_node_view),
            "is_mounted",
            new_callable=lambda: property(lambda self: True),
        ):
            metrics = [
                MetricResult(
                    tag="gpu_util_dcgm_http___localhost_9400_metrics_gpu0_GPU-12345678",
                    header="GPU Utilization | localhost:9400 | GPU 0 | NVIDIA RTX 4090",
                    unit="%",
                    avg=75.0,
                )
            ]

            single_node_view.update(metrics)

            mock_gpu_table.update.assert_called_once()


class TestRealtimeTelemetryDashboard:
    """Test RealtimeTelemetryDashboard widget."""

    @pytest.fixture
    def service_config(self):
        """Create a mock ServiceConfig."""
        return Mock(spec=ServiceConfig)

    @pytest.fixture
    def dashboard(self, service_config):
        """Create a RealtimeTelemetryDashboard instance for testing."""
        return RealtimeTelemetryDashboard(service_config=service_config)

    def test_init(self, dashboard, service_config):
        """Test dashboard initialization."""
        assert dashboard.service_config == service_config
        assert dashboard.all_nodes_view is None
        assert dashboard.metrics == []
        assert dashboard.border_title == "Real-Time GPU Telemetry"

    def test_compose_creates_widgets(self, dashboard):
        """Test that compose yields the correct widgets."""
        widgets = list(dashboard.compose())

        assert len(widgets) == 2
        assert widgets[0].__class__.__name__ == "Static"
        assert widgets[1].__class__.__name__ == "SingleNodeView"

    def test_set_status_message(self, dashboard):
        """Test set_status_message updates the status widget."""
        mock_status = Mock()
        mock_all_nodes = Mock()

        with patch.object(dashboard, "query_one", return_value=mock_status):
            dashboard.all_nodes_view = mock_all_nodes

            dashboard.set_status_message("Test message")

            mock_status.update.assert_called_once_with("Test message")
            mock_status.remove_class.assert_called_once_with("hidden")
            mock_all_nodes.add_class.assert_called_once_with("hidden")

    def test_set_status_message_handles_exception(self, dashboard):
        """Test set_status_message handles exceptions gracefully."""
        with patch.object(
            dashboard, "query_one", side_effect=Exception("Widget not found")
        ):
            dashboard.set_status_message("Test message")

    def test_on_realtime_telemetry_metrics_first_update(self, dashboard):
        """Test on_realtime_telemetry_metrics on first metrics update."""
        mock_all_nodes = Mock()
        mock_status = Mock()
        dashboard.all_nodes_view = mock_all_nodes

        with patch.object(dashboard, "query_one", return_value=mock_status):
            metrics = [
                MetricResult(
                    tag="gpu_util_dcgm_http___localhost_9400_metrics_gpu0_GPU-12345678",
                    header="GPU Utilization | localhost:9400 | GPU 0 | NVIDIA RTX 4090",
                    unit="%",
                    avg=75.0,
                )
            ]

            dashboard.on_realtime_telemetry_metrics(metrics)

            assert dashboard.metrics == metrics
            mock_status.add_class.assert_called_once_with("hidden")
            mock_all_nodes.remove_class.assert_called_once_with("hidden")
            mock_all_nodes.update.assert_called_once_with(metrics)

    def test_on_realtime_telemetry_metrics_subsequent_update(self, dashboard):
        """Test on_realtime_telemetry_metrics on subsequent updates."""
        mock_all_nodes = Mock()
        dashboard.all_nodes_view = mock_all_nodes
        dashboard.metrics = [Mock()]

        metrics = [
            MetricResult(
                tag="gpu_util_dcgm_http___localhost_9400_metrics_gpu0_GPU-12345678",
                header="GPU Utilization | localhost:9400 | GPU 0 | NVIDIA RTX 4090",
                unit="%",
                avg=80.0,
            )
        ]

        dashboard.on_realtime_telemetry_metrics(metrics)

        assert dashboard.metrics == metrics
        mock_all_nodes.update.assert_called_once_with(metrics)
