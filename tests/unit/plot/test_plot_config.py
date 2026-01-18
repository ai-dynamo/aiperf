# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for PlotConfig class.

Tests YAML configuration loading, validation, and conversion to PlotSpec objects.
"""

from pathlib import Path

import pytest

from aiperf.plot.config import PlotConfig, _parse_and_validate_metric_name
from aiperf.plot.core.plot_specs import (
    DataSource,
    MetricSpec,
    PlotSpec,
    PlotType,
    TimeSlicePlotSpec,
)


class TestPlotConfigLoading:
    """Tests for PlotConfig loading and priority."""

    def test_auto_create_user_config(self, tmp_path, monkeypatch):
        """Test that user config is auto-created on first access if it doesn't exist."""
        # Create fake home directory
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setattr(Path, "home", lambda: fake_home)

        # User config should not exist yet
        user_config = fake_home / ".aiperf" / "plot_config.yaml"
        assert not user_config.exists()

        # Instantiate PlotConfig - should auto-create user config
        config = PlotConfig()

        # Verify user config was created
        assert user_config.exists()
        assert config.resolved_path == user_config
        assert config.config is not None
        assert "visualization" in config.config

    def test_load_custom_config(self, tmp_path):
        """Test loading from a custom config path."""
        custom_config = tmp_path / "custom_config.yaml"
        custom_config.write_text(
            """
visualization:
  multi_run:
    - name: test_plot
      plot_type: scatter
      metrics:
        - name: x_metric
          source: aggregated
          axis: x
        - name: y_metric
          source: aggregated
          axis: y
  single_run: []
"""
        )

        config = PlotConfig(custom_config)

        assert config.resolved_path == custom_config
        assert config.config["visualization"]["multi_run"][0]["name"] == "test_plot"

    def test_load_user_config(self, tmp_path, monkeypatch):
        """Test loading from user home config (~/.aiperf/plot_config.yaml)."""
        # Create fake home directory
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setenv("HOME", str(fake_home))

        # Create user config
        user_config_dir = fake_home / ".aiperf"
        user_config_dir.mkdir()
        user_config = user_config_dir / "plot_config.yaml"
        user_config.write_text(
            """
visualization:
  multi_run:
    - name: user_plot
      plot_type: pareto
      metrics:
        - name: x
          source: aggregated
          axis: x
        - name: y
          source: aggregated
          axis: y
  single_run: []
"""
        )

        # Monkeypatch Path.home() to return fake_home
        monkeypatch.setattr(Path, "home", lambda: fake_home)

        config = PlotConfig()

        assert config.resolved_path == user_config
        assert config.config["visualization"]["multi_run"][0]["name"] == "user_plot"

    def test_config_priority(self, tmp_path, monkeypatch):
        """Test that CLI config takes priority over user config."""
        # Create fake home
        fake_home = tmp_path / "home"
        fake_home.mkdir()

        # Create user config
        user_config_dir = fake_home / ".aiperf"
        user_config_dir.mkdir()
        user_config = user_config_dir / "plot_config.yaml"
        user_config.write_text(
            """
visualization:
  multi_run:
    - name: user_plot
      plot_type: scatter
      metrics:
        - name: x
          source: aggregated
          axis: x
        - name: y
          source: aggregated
          axis: y
  single_run: []
"""
        )

        # Create CLI config
        cli_config = tmp_path / "cli_config.yaml"
        cli_config.write_text(
            """
visualization:
  multi_run:
    - name: cli_plot
      plot_type: pareto
      metrics:
        - name: x
          source: aggregated
          axis: x
        - name: y
          source: aggregated
          axis: y
  single_run: []
"""
        )

        # Monkeypatch Path.home()
        monkeypatch.setattr(Path, "home", lambda: fake_home)

        # CLI config should take priority
        config = PlotConfig(cli_config)

        assert config.resolved_path == cli_config
        assert config.config["visualization"]["multi_run"][0]["name"] == "cli_plot"

    def test_missing_config_file(self, tmp_path):
        """Test error when custom config file doesn't exist."""
        missing_config = tmp_path / "missing.yaml"

        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            PlotConfig(missing_config)

    def test_invalid_yaml_syntax(self, tmp_path):
        """Test error handling for invalid YAML syntax."""
        invalid_config = tmp_path / "invalid.yaml"
        invalid_config.write_text("{ invalid: yaml: syntax")

        with pytest.raises(ValueError, match="Failed to load YAML config"):
            PlotConfig(invalid_config)

    def test_missing_visualization_key(self, tmp_path):
        """Test error when YAML is missing 'visualization' top-level key."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("other_key: value")

        with pytest.raises(ValueError, match="missing 'visualization' top-level key"):
            PlotConfig(config_file)


class TestPlotSpecConversion:
    """Tests for converting YAML to PlotSpec objects."""

    def test_get_multi_run_plot_specs(self, tmp_path):
        """Test getting multi-run plot specs from config."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
visualization:
  multi_run_defaults:
    - test_plot
  multi_run_plots:
    test_plot:
      type: scatter_line
      x: request_latency_p50
      y: request_throughput_avg
      title: "Test Plot"
      labels: [concurrency]
      groups: [model]
  single_run_defaults: []
  single_run_plots: {}
"""
        )

        config = PlotConfig(config_file)
        specs = config.get_multi_run_plot_specs()

        assert len(specs) == 1
        assert isinstance(specs[0], PlotSpec)
        assert specs[0].name == "test_plot"
        assert specs[0].plot_type == PlotType.SCATTER_LINE
        assert specs[0].title == "Test Plot"
        assert specs[0].filename == "test_plot.png"
        assert specs[0].label_by == "concurrency"
        assert specs[0].group_by == "model"

        # Check metrics
        assert len(specs[0].metrics) == 2
        assert isinstance(specs[0].metrics[0], MetricSpec)
        assert specs[0].metrics[0].name == "request_latency"
        assert specs[0].metrics[0].source == DataSource.AGGREGATED
        assert specs[0].metrics[0].axis == "x"
        assert specs[0].metrics[0].stat == "p50"

    def test_get_single_run_plot_specs(self, tmp_path):
        """Test getting single-run plot specs from config."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
visualization:
  multi_run_defaults: []
  multi_run_plots: {}
  single_run_defaults:
    - ttft_plot
  single_run_plots:
    ttft_plot:
      type: scatter
      x: request_number
      y: time_to_first_token
      title: "TTFT Over Time"
"""
        )

        config = PlotConfig(config_file)
        specs = config.get_single_run_plot_specs()

        assert len(specs) == 1
        assert isinstance(specs[0], PlotSpec)
        assert specs[0].name == "ttft_plot"
        assert specs[0].plot_type == PlotType.SCATTER
        assert len(specs[0].metrics) == 2
        assert specs[0].metrics[0].source == DataSource.REQUESTS

    def test_timeslice_plot_spec_conversion(self, tmp_path):
        """Test conversion of TimeSlicePlotSpec with use_slice_duration field."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
visualization:
  multi_run_defaults: []
  multi_run_plots: {}
  single_run_defaults:
    - timeslice_plot
  single_run_plots:
    timeslice_plot:
      type: histogram
      x: Timeslice
      y: Time to First Token
      stat: avg
      source: timeslices
      use_slice_duration: true
"""
        )

        config = PlotConfig(config_file)
        specs = config.get_single_run_plot_specs()

        assert len(specs) == 1
        assert isinstance(specs[0], TimeSlicePlotSpec)
        assert specs[0].use_slice_duration is True

    def test_multiple_plot_specs(self, tmp_path):
        """Test loading multiple plot specs."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
visualization:
  multi_run_defaults:
    - plot1
    - plot2
  multi_run_plots:
    plot1:
      type: scatter
      x: request_latency_p50
      y: request_throughput_avg
    plot2:
      type: pareto
      x: request_latency_avg
      y: output_token_throughput_per_gpu_avg
  single_run_defaults: []
  single_run_plots: {}
"""
        )

        config = PlotConfig(config_file)
        specs = config.get_multi_run_plot_specs()

        assert len(specs) == 2
        assert specs[0].name == "plot1"
        assert specs[1].name == "plot2"

    def test_missing_required_field(self, tmp_path):
        """Test error when required field is missing in plot spec."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
visualization:
  multi_run_defaults:
    - incomplete_plot
  multi_run_plots:
    incomplete_plot:
      # Missing type field
      x: request_latency_p50
      y: request_throughput_avg
  single_run_defaults: []
  single_run_plots: {}
"""
        )

        config = PlotConfig(config_file)

        with pytest.raises(
            ValueError,
            match="Config validation failed for multi_run plot 'incomplete_plot'",
        ):
            config.get_multi_run_plot_specs()

    def test_invalid_enum_value(self, tmp_path):
        """Test error when enum value is invalid."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
visualization:
  multi_run_defaults:
    - bad_plot
  multi_run_plots:
    bad_plot:
      type: invalid_type
      x: request_latency_p50
      y: request_throughput_avg
  single_run_defaults: []
  single_run_plots: {}
"""
        )

        config = PlotConfig(config_file)

        with pytest.raises(ValueError):
            config.get_multi_run_plot_specs()

    def test_empty_multi_run_list(self, tmp_path):
        """Test handling of empty multi_run defaults."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
visualization:
  multi_run_defaults: []
  multi_run_plots: {}
  single_run_defaults: []
  single_run_plots: {}
"""
        )

        config = PlotConfig(config_file)
        specs = config.get_multi_run_plot_specs()

        assert specs == []

    def test_default_config_structure(self, tmp_path, monkeypatch):
        """Test that default config has expected structure and valid specs."""
        # Create fake home directory for auto-creation
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setattr(Path, "home", lambda: fake_home)

        config = PlotConfig()

        # Test multi-run specs
        multi_run_specs = config.get_multi_run_plot_specs()
        assert len(multi_run_specs) > 0
        for spec in multi_run_specs:
            assert isinstance(spec, PlotSpec)
            assert spec.name
            assert spec.plot_type
            assert len(spec.metrics) > 0

        # Test single-run specs
        single_run_specs = config.get_single_run_plot_specs()
        assert len(single_run_specs) > 0
        for spec in single_run_specs:
            assert isinstance(spec, PlotSpec)
            assert spec.name
            assert spec.plot_type
            assert len(spec.metrics) > 0


class TestExperimentClassificationOverride:
    """Tests for automatic groups override when experiment classification is enabled."""

    def test_classification_enabled_overrides_groups_to_experiment_type(
        self, tmp_path: Path
    ) -> None:
        """Test that when classification is enabled, groups are overridden to experiment_group."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
experiment_classification:
  baselines:
    - "*baseline*"
  treatments:
    - "*treatment*"
  default: treatment

visualization:
  multi_run_defaults:
    - test_plot
  multi_run_plots:
    test_plot:
      type: scatter_line
      x: request_latency_avg
      y: request_throughput_avg
      groups: [model]
  single_run_defaults: []
  single_run_plots: {}
"""
        )

        config = PlotConfig(config_file)
        specs = config.get_multi_run_plot_specs()

        assert len(specs) == 1
        # Should override to experiment_group
        assert specs[0].group_by == "experiment_group"

    def test_classification_disabled_respects_original_groups(
        self, tmp_path: Path
    ) -> None:
        """Test that without classification, original groups setting is used."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
# experiment_classification commented out (disabled)

visualization:
  multi_run_defaults:
    - test_plot
  multi_run_plots:
    test_plot:
      type: scatter_line
      x: request_latency_avg
      y: request_throughput_avg
      groups: [model]
  single_run_defaults: []
  single_run_plots: {}
"""
        )

        config = PlotConfig(config_file)
        specs = config.get_multi_run_plot_specs()

        assert len(specs) == 1
        # Should keep original groups setting
        assert specs[0].group_by == "model"

    def test_classification_overrides_all_group_types(self, tmp_path: Path) -> None:
        """Test that classification overrides groups regardless of original value."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
experiment_classification:
  baselines:
    - "*baseline*"
  treatments:
    - "*treatment*"
  default: treatment

visualization:
  multi_run_defaults:
    - plot1
    - plot2
    - plot3
  multi_run_plots:
    plot1:
      type: scatter_line
      x: request_latency_avg
      y: request_throughput_avg
      groups: [model]
    plot2:
      type: scatter_line
      x: request_latency_avg
      y: request_throughput_avg
      groups: [run_name]
    plot3:
      type: scatter_line
      x: request_latency_avg
      y: request_throughput_avg
      groups: [concurrency]
  single_run_defaults: []
  single_run_plots: {}
"""
        )

        config = PlotConfig(config_file)
        specs = config.get_multi_run_plot_specs()

        assert len(specs) == 3
        # All should be overridden to experiment_group
        for spec in specs:
            assert spec.group_by == "experiment_group"

    def test_classification_with_nested_directory_structure(
        self, tmp_path: Path
    ) -> None:
        """Test that classification works with nested baseline/treatment directories."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
experiment_classification:
  baselines:
    - "*baseline*"
  treatments:
    - "*treatment*"
  default: treatment

visualization:
  multi_run_defaults:
    - test_plot
  multi_run_plots:
    test_plot:
      type: scatter_line
      x: request_latency_avg
      y: request_throughput_avg
      groups: [model]
  single_run_defaults: []
  single_run_plots: {}
"""
        )

        config = PlotConfig(config_file)
        specs = config.get_multi_run_plot_specs()

        # Verify override happens
        assert specs[0].group_by == "experiment_group"

        # Verify classification config is accessible
        classification = config.get_experiment_classification_config()
        assert classification is not None
        assert "*baseline*" in classification.baselines
        assert "*treatment*" in classification.treatments


class TestPlotSpecDetails:
    """Tests for detailed PlotSpec field handling."""

    def test_optional_fields(self, tmp_path):
        """Test that optional fields are properly handled."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
visualization:
  multi_run_defaults:
    - minimal_plot
  multi_run_plots:
    minimal_plot:
      type: scatter
      x: request_latency_p50
      y: request_throughput_avg
      # Only required fields, no optional ones
  single_run_defaults: []
  single_run_plots: {}
"""
        )

        config = PlotConfig(config_file)
        specs = config.get_multi_run_plot_specs()

        assert specs[0].title is None
        assert specs[0].filename == "minimal_plot.png"  # Auto-generated
        assert specs[0].label_by is None
        assert specs[0].group_by == "run_name"  # Smart default when groups omitted

    def test_dual_axis_plot_spec(self, tmp_path):
        """Test dual-axis plot spec with y2 axis."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
visualization:
  multi_run_defaults: []
  multi_run_plots: {}
  single_run_defaults:
    - dual_axis_plot
  single_run_plots:
    dual_axis_plot:
      type: dual_axis
      x: timestamp_s
      y: throughput_tokens_per_sec
      y2: gpu_utilization
      primary_style:
        mode: lines
        line_shape: hv
      secondary_style:
        mode: lines
        fill: tozeroy
      supplementary_col: active_requests
"""
        )

        config = PlotConfig(config_file)
        specs = config.get_single_run_plot_specs()

        assert len(specs) == 1
        assert specs[0].plot_type == PlotType.DUAL_AXIS
        assert len(specs[0].metrics) == 3
        assert specs[0].metrics[2].axis == "y2"
        assert specs[0].primary_style.mode == "lines"
        assert specs[0].primary_style.line_shape == "hv"
        assert specs[0].secondary_style.fill == "tozeroy"
        assert specs[0].supplementary_col == "active_requests"


class TestMultiValueGrouping:
    """Tests for multi-value label_by and group_by support."""

    def test_single_value_list_label_by(self, tmp_path):
        """Test single value in list format for label_by is converted to string."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
visualization:
  multi_run_defaults:
    - test_plot
  multi_run_plots:
    test_plot:
      type: scatter_line
      x: request_latency_p50
      y: request_throughput_avg
      labels: [concurrency]
  single_run_defaults: []
  single_run_plots: {}
"""
        )

        config = PlotConfig(config_file)
        specs = config.get_multi_run_plot_specs()

        assert len(specs) == 1
        assert specs[0].label_by == "concurrency"

    def test_single_value_list_group_by(self, tmp_path):
        """Test single value in list format for group_by is converted to string."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
visualization:
  multi_run_defaults:
    - test_plot
  multi_run_plots:
    test_plot:
      type: pareto
      x: request_latency_avg
      y: output_token_throughput_per_gpu_avg
      groups: [model]
  single_run_defaults: []
  single_run_plots: {}
"""
        )

        config = PlotConfig(config_file)
        specs = config.get_multi_run_plot_specs()

        assert len(specs) == 1
        assert specs[0].group_by == "model"

    def test_multi_element_list_raises_error(self, tmp_path):
        """Test that multi-element lists in groups raise validation error."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
visualization:
  multi_run_defaults:
    - test_plot
  multi_run_plots:
    test_plot:
      type: pareto
      x: request_latency_avg
      y: request_throughput_avg
      groups: [model, endpoint_type]
      labels: [concurrency]
  single_run_defaults: []
  single_run_plots: {}
"""
        )

        config = PlotConfig(config_file)
        with pytest.raises(ValueError, match="Config validation failed"):
            config.get_multi_run_plot_specs()

    def test_omitted_fields_for_auto_detection(self, tmp_path):
        """Test that omitted label_by and group_by default to None for auto-detection."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
visualization:
  multi_run_defaults:
    - test_plot
  multi_run_plots:
    test_plot:
      type: pareto
      x: request_latency_avg
      y: output_token_throughput_per_gpu_avg
      # label_by and group_by omitted for auto-detection
  single_run_defaults: []
  single_run_plots: {}
"""
        )

        config = PlotConfig(config_file)
        specs = config.get_multi_run_plot_specs()

        assert len(specs) == 1
        assert specs[0].label_by is None
        assert specs[0].group_by == "run_name"  # Smart default when groups omitted

    def test_explicit_null_for_auto_detection(self, tmp_path):
        """Test explicit null/None values for auto-detection."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
visualization:
  multi_run_defaults:
    - test_plot
  multi_run_plots:
    test_plot:
      type: scatter_line
      x: request_latency_p50
      y: request_throughput_avg
      labels: null
      groups: null
  single_run_defaults: []
  single_run_plots: {}
"""
        )

        config = PlotConfig(config_file)
        specs = config.get_multi_run_plot_specs()

        assert len(specs) == 1
        assert specs[0].label_by is None
        assert specs[0].group_by == "run_name"  # Smart default when groups omitted


class TestDynamicMetricShortcuts:
    """Tests for dynamic metric shortcut resolution."""

    def test_aggregated_metric_with_avg(self, tmp_path):
        """Test dynamic resolution of aggregated metric with avg stat."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
visualization:
  multi_run_defaults:
    - test_plot
  multi_run_plots:
    test_plot:
      type: scatter_line
      x: time_to_first_token_avg
      y: request_throughput_avg
  single_run_defaults: []
  single_run_plots: {}
"""
        )

        config = PlotConfig(config_file)
        specs = config.get_multi_run_plot_specs()

        assert len(specs) == 1
        assert specs[0].metrics[0].name == "time_to_first_token"
        assert specs[0].metrics[0].stat == "avg"
        assert specs[0].metrics[0].source == DataSource.AGGREGATED

    def test_aggregated_metric_with_percentiles(self, tmp_path):
        """Test dynamic resolution with different percentiles."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
visualization:
  multi_run_defaults:
    - p50_plot
    - p90_plot
    - p99_plot
  multi_run_plots:
    p50_plot:
      type: scatter
      x: request_latency_p50
      y: request_throughput_avg
    p90_plot:
      type: scatter
      x: request_latency_p90
      y: request_throughput_avg
    p99_plot:
      type: scatter
      x: request_latency_p99
      y: request_throughput_avg
  single_run_defaults: []
  single_run_plots: {}
"""
        )

        config = PlotConfig(config_file)
        specs = config.get_multi_run_plot_specs()

        assert len(specs) == 3
        assert specs[0].metrics[0].stat == "p50"
        assert specs[1].metrics[0].stat == "p90"
        assert specs[2].metrics[0].stat == "p99"

    def test_all_stat_types(self, tmp_path):
        """Test that all stat types work: avg, min, max, std, p1-p99."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
visualization:
  multi_run_defaults:
    - test_avg
    - test_min
    - test_max
    - test_std
    - test_p1
    - test_p95
  multi_run_plots:
    test_avg:
      type: scatter
      x: request_latency_avg
      y: request_throughput_avg
    test_min:
      type: scatter
      x: request_latency_min
      y: request_throughput_avg
    test_max:
      type: scatter
      x: request_latency_max
      y: request_throughput_avg
    test_std:
      type: scatter
      x: request_latency_std
      y: request_throughput_avg
    test_p1:
      type: scatter
      x: request_latency_p1
      y: request_throughput_avg
    test_p95:
      type: scatter
      x: request_latency_p95
      y: request_throughput_avg
  single_run_defaults: []
  single_run_plots: {}
"""
        )

        config = PlotConfig(config_file)
        specs = config.get_multi_run_plot_specs()

        assert len(specs) == 6
        assert specs[0].metrics[0].stat == "avg"
        assert specs[1].metrics[0].stat == "min"
        assert specs[2].metrics[0].stat == "max"
        assert specs[3].metrics[0].stat == "std"
        assert specs[4].metrics[0].stat == "p1"
        assert specs[5].metrics[0].stat == "p95"

    def test_request_metrics_without_stat(self, tmp_path):
        """Test request metrics without stat suffix."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
visualization:
  multi_run_defaults: []
  multi_run_plots: {}
  single_run_defaults:
    - test_plot
  single_run_plots:
    test_plot:
      type: scatter
      x: request_number
      y: time_to_first_token
"""
        )

        config = PlotConfig(config_file)
        specs = config.get_single_run_plot_specs()

        assert len(specs) == 1
        assert specs[0].metrics[0].name == "request_number"
        assert specs[0].metrics[0].stat is None
        assert specs[0].metrics[0].source == DataSource.REQUESTS
        assert specs[0].metrics[1].name == "time_to_first_token"
        assert specs[0].metrics[1].stat is None

    def test_invalid_metric_name(self, tmp_path):
        """Test that invalid metric names raise helpful errors."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
visualization:
  multi_run_defaults:
    - bad_plot
  multi_run_plots:
    bad_plot:
      type: scatter
      x: nonexistent_metric_avg
      y: request_throughput_avg
  single_run_defaults: []
  single_run_plots: {}
"""
        )

        config = PlotConfig(config_file)

        with pytest.raises(
            ValueError, match="Config validation failed for multi_run plot 'bad_plot'"
        ):
            config.get_multi_run_plot_specs()

    def test_invalid_stat_type(self, tmp_path):
        """Test that invalid stat types raise helpful errors."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
visualization:
  multi_run_defaults:
    - bad_plot
  multi_run_plots:
    bad_plot:
      type: scatter
      x: request_latency_p999
      y: request_throughput_avg
  single_run_defaults: []
  single_run_plots: {}
"""
        )

        config = PlotConfig(config_file)

        with pytest.raises(
            ValueError, match="Config validation failed for multi_run plot 'bad_plot'"
        ):
            config.get_multi_run_plot_specs()


class TestMetricNameValidation:
    """Tests for _parse_and_validate_metric_name function."""

    def test_parse_metric_with_avg_stat(self):
        """Test parsing metric with avg stat suffix."""
        base, stat = _parse_and_validate_metric_name("request_latency_avg")
        assert base == "request_latency"
        assert stat == "avg"

    def test_parse_metric_with_percentile_stats(self):
        """Test parsing metrics with valid percentile suffixes."""
        for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
            base, stat = _parse_and_validate_metric_name(f"metric_p{p}")
            assert base == "metric"
            assert stat == f"p{p}"

    def test_parse_metric_with_all_basic_stats(self):
        """Test parsing metrics with min, max, std stats."""
        for stat_type in ["min", "max", "std"]:
            base, stat = _parse_and_validate_metric_name(f"latency_{stat_type}")
            assert base == "latency"
            assert stat == stat_type

    def test_parse_metric_without_stat(self):
        """Test parsing simple metric name without stat suffix."""
        base, stat = _parse_and_validate_metric_name("request_number")
        assert base == "request_number"
        assert stat is None

    def test_invalid_percentile_p100_raises_error(self):
        """Test that p100 raises an error with suggestions."""
        with pytest.raises(ValueError) as exc_info:
            _parse_and_validate_metric_name("metric_p100")

        error_msg = str(exc_info.value)
        assert "Invalid stat suffix 'p100'" in error_msg
        assert "Valid stat suffixes are:" in error_msg
        assert "p99" in error_msg

    def test_invalid_percentile_p999_raises_error(self):
        """Test that p999 raises an error."""
        with pytest.raises(ValueError) as exc_info:
            _parse_and_validate_metric_name("metric_p999")

        assert "Invalid stat suffix 'p999'" in str(exc_info.value)

    def test_metric_with_underscore_in_name(self):
        """Test metric with underscores in base name."""
        base, stat = _parse_and_validate_metric_name("time_to_first_token_p50")
        assert base == "time_to_first_token"
        assert stat == "p50"

    def test_invalid_percentile_p42_raises_error(self):
        """Test that p42 raises an error with suggestions."""
        with pytest.raises(ValueError) as exc_info:
            _parse_and_validate_metric_name("metric_p42")

        error_msg = str(exc_info.value)
        assert "Invalid stat suffix 'p42'" in error_msg

    def test_invalid_stat_suffix_raises_helpful_error(self):
        """Test that invalid stat suffixes like p67 raise helpful errors."""
        with pytest.raises(ValueError) as exc_info:
            _parse_and_validate_metric_name("latency_p67")

        error_msg = str(exc_info.value)
        assert "Invalid stat suffix 'p67'" in error_msg
        assert "Valid stat suffixes are:" in error_msg
        assert "p50" in error_msg
        assert "Did you mean" in error_msg
        assert "latency_p75" in error_msg or "latency_p50" in error_msg

    def test_fuzzy_matching_suggests_close_percentiles(self):
        """Test that fuzzy matching suggests numerically close percentiles."""
        with pytest.raises(ValueError) as exc_info:
            _parse_and_validate_metric_name("latency_p92")

        error_msg = str(exc_info.value)
        assert "p90" in error_msg or "p95" in error_msg

    def test_valid_percentiles_do_not_raise(self):
        """Test that valid percentiles don't raise errors."""
        base, stat = _parse_and_validate_metric_name("metric_p50")
        assert base == "metric"
        assert stat == "p50"

        base, stat = _parse_and_validate_metric_name("metric_p95")
        assert base == "metric"
        assert stat == "p95"
