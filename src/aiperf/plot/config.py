# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Plot configuration loader for YAML-based plot definitions.

Loads plot specifications from YAML files with the following priority:
1. Custom path (if provided via --config flag)
2. User home config (~/.aiperf/plot_config.yaml) - auto-created on first run
3. Default shipped config (src/aiperf/plot/default_plot_config.yaml)
"""

import logging
import shutil
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

from aiperf.plot._metrics import (
    expand_metric_shortcut,
    parse_and_validate_metric_name,
)
from aiperf.plot.core.plot_specs import (
    ExperimentClassificationConfig,
    MetricSpec,
    PlotSpec,
    PlotType,
    Style,
    TimeSlicePlotSpec,
)

# Back-compat re-export: tests and external callers import this from
# `aiperf.plot.config`. The implementation lives in `_metrics.py`.
_parse_and_validate_metric_name = parse_and_validate_metric_name

_logger = logging.getLogger(__name__)


class PlotConfig:
    """
    Load and manage plot configuration from YAML.

    Supports loading from multiple sources with priority:
    1. Custom config path (CLI override)
    2. User home config (~/.aiperf/plot_config.yaml)
    3. Default shipped config

    Args:
        config_path: Optional custom path to YAML config file
    """

    def __init__(self, config_path: Path | None = None, verbose: bool = False) -> None:
        """
        Initialize plot configuration loader.

        Args:
            config_path: Optional custom path to YAML config file
            verbose: Show detailed error tracebacks in console
        """
        self.custom_path = config_path
        self.verbose = verbose
        self.resolved_path = self._resolve_config_path()
        self.config = self._load_yaml()

    def _resolve_config_path(self) -> Path:
        """
        Resolve which config file to use based on priority.

        Priority:
        1. Custom path via --config flag (explicit override)
        2. ~/.aiperf/plot_config.yaml (auto-created from default on first run)
        3. System default (fallback only, indicates package issue)

        Console messages:
        - Shows "Using config: <path>" when using customized config (Priority 1 or 2)
        - Shows creation message when auto-creating config on first run
        - Silent when using system defaults

        Returns:
            Path to the configuration file to use

        Raises:
            FileNotFoundError: If custom path is specified but doesn't exist
        """
        # Priority 1: Custom path via CLI
        if self.custom_path:
            if not self.custom_path.exists():
                raise FileNotFoundError(
                    f"Configuration file not found: {self.custom_path}"
                )
            print(f"Using config: {self.custom_path}")
            return self.custom_path

        # Priority 2: User home config (auto-create if missing)
        user_config = Path.home() / ".aiperf" / "plot_config.yaml"
        if not user_config.exists():
            default_config = Path(__file__).parent / "default_plot_config.yaml"
            if not default_config.exists():
                raise FileNotFoundError(
                    f"Default plot config not found at {default_config}. "
                    "This indicates a package installation issue."
                )

            user_config.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(default_config, user_config)

            print(f"\nCreated plot configuration: {user_config}")
            print(
                "   Edit this file to customize plots (changes take effect on next run)\n"
            )
        else:
            print(f"Using config: {user_config}")

        return user_config

    def _load_yaml(self) -> dict:
        """
        Load and parse YAML configuration file.

        Returns:
            Dictionary containing the parsed YAML configuration

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If YAML is invalid or malformed
        """
        if not self.resolved_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {self.resolved_path}"
            )

        try:
            yaml = YAML(typ="safe")
            with open(self.resolved_path, encoding="utf-8") as f:
                config = yaml.load(f)

            if not isinstance(config, dict):
                raise ValueError(
                    f"Invalid YAML config: expected dictionary, got {type(config).__name__}"
                )

            if "visualization" not in config:
                raise ValueError(
                    "Invalid YAML config: missing 'visualization' top-level key"
                )

            return config

        except Exception as e:
            raise ValueError(
                f"Failed to load YAML config from {self.resolved_path}: {e}"
            ) from e

    def get_multi_run_plot_specs(self) -> list[PlotSpec]:
        """
        Get plot specifications for multi-run comparison plots.

        Returns:
            List of PlotSpec objects for multi-run visualizations

        Raises:
            ValueError: If multi_run section is missing or invalid
        """
        viz_config = self.config.get("visualization", {})

        defaults = viz_config.get("multi_run_defaults", [])
        if not isinstance(defaults, list):
            raise ValueError(
                f"Invalid config: 'visualization.multi_run_defaults' must be a list, "
                f"got {type(defaults).__name__}"
            )

        presets = viz_config.get("multi_run_plots", {})
        if not isinstance(presets, dict):
            raise ValueError(
                f"Invalid config: 'visualization.multi_run_plots' must be a dict, "
                f"got {type(presets).__name__}"
            )

        plot_specs = []
        for plot_name in defaults:
            try:
                if plot_name not in presets:
                    raise ValueError(
                        f"Plot '{plot_name}' listed in multi_run_defaults but not found in multi_run_plots"
                    )

                preset = presets[plot_name]
                plot_spec = self._preset_to_plot_spec(plot_name, preset)
                plot_specs.append(plot_spec)
            except Exception as e:
                error_context = (
                    f"Failed to parse multi_run plot preset '{plot_name}'\n"
                    f"Config file: {self.resolved_path}\n"
                    f"Preset: {preset if plot_name in presets else '<not found>'}\n"
                    f"Error: {e}"
                )
                _logger.error(error_context, exc_info=True)

                raise ValueError(
                    f"Config validation failed for multi_run plot '{plot_name}'. "
                    f"Check the configuration file at {self.resolved_path}"
                ) from e

        return plot_specs

    def get_single_run_plot_specs(self) -> list[PlotSpec]:
        """
        Get plot specifications for single-run time series plots.

        Returns:
            List of PlotSpec objects for single-run visualizations

        Raises:
            ValueError: If single_run section is missing or invalid
        """
        viz_config = self.config.get("visualization", {})

        defaults = viz_config.get("single_run_defaults", [])
        if not isinstance(defaults, list):
            raise ValueError(
                f"Invalid config: 'visualization.single_run_defaults' must be a list, "
                f"got {type(defaults).__name__}"
            )

        presets = viz_config.get("single_run_plots", {})
        if not isinstance(presets, dict):
            raise ValueError(
                f"Invalid config: 'visualization.single_run_plots' must be a dict, "
                f"got {type(presets).__name__}"
            )

        plot_specs = []
        for plot_name in defaults:
            try:
                if plot_name not in presets:
                    raise ValueError(
                        f"Plot '{plot_name}' listed in single_run_defaults but not found in single_run_plots"
                    )

                preset = presets[plot_name]
                plot_spec = self._preset_to_plot_spec(plot_name, preset)
                plot_specs.append(plot_spec)
            except Exception as e:
                error_context = (
                    f"Failed to parse single_run plot preset '{plot_name}'\n"
                    f"Config file: {self.resolved_path}\n"
                    f"Preset: {preset if plot_name in presets else '<not found>'}\n"
                    f"Error: {e}"
                )
                _logger.error(error_context, exc_info=True)

                raise ValueError(
                    f"Config validation failed for single_run plot '{plot_name}'. "
                    f"Check the configuration file at {self.resolved_path}"
                ) from e

        return plot_specs

    def get_experiment_classification_config(
        self,
    ) -> ExperimentClassificationConfig | None:
        """
        Get experiment classification configuration for baseline/treatment assignment.

        Returns:
            ExperimentClassificationConfig object if section exists, None otherwise

        Raises:
            ValueError: If experiment_classification section is invalid
        """
        exp_class_config = self.config.get("experiment_classification")

        if exp_class_config is None:
            return None

        if not isinstance(exp_class_config, dict):
            raise ValueError(
                f"Invalid config: 'experiment_classification' must be a dict, "
                f"got {type(exp_class_config).__name__}"
            )

        try:
            return ExperimentClassificationConfig(**exp_class_config)
        except Exception as e:
            raise ValueError(
                f"Failed to parse experiment_classification config: {e}"
            ) from e

    def get_downsampling_config(self) -> dict:
        """
        Get server metrics downsampling configuration.

        Returns:
            Dictionary with downsampling configuration:
            {
                "enabled": bool,
                "window_size_seconds": float,
                "aggregation_method": str
            }
            Returns defaults if settings section is missing.
        """
        settings = self.config.get("settings", {})
        downsampling = settings.get("server_metrics_downsampling", {})

        # Provide sensible defaults
        return {
            "enabled": downsampling.get("enabled", True),
            "window_size_seconds": downsampling.get("window_size_seconds", 5.0),
            "aggregation_method": downsampling.get("aggregation_method", "mean"),
        }

    def _preset_to_plot_spec(
        self, name: str, preset: dict
    ) -> PlotSpec | TimeSlicePlotSpec:
        """
        Convert preset dictionary to PlotSpec object.

        Args:
            name: Plot name/key from YAML
            preset: Preset dictionary with simplified format

        Returns:
            PlotSpec or TimeSlicePlotSpec object

        Raises:
            ValueError: If preset is invalid
        """
        if not isinstance(preset, dict):
            raise ValueError(
                f"Expected dictionary for preset, got {type(preset).__name__}"
            )

        plot_type_str = preset.get("type")
        if not plot_type_str:
            raise ValueError(f"Missing 'type' field in preset '{name}'")
        plot_type = PlotType(plot_type_str)

        metrics = _build_preset_metrics(name, preset)
        groups = self._resolve_preset_groups(name, preset)
        spec_kwargs = _build_spec_kwargs(
            name=name,
            preset=preset,
            plot_type=plot_type,
            metrics=metrics,
            groups=groups,
        )

        if "use_slice_duration" in preset:
            spec_kwargs["use_slice_duration"] = preset["use_slice_duration"]
            return TimeSlicePlotSpec(**spec_kwargs)

        return PlotSpec(**spec_kwargs)

    def _resolve_preset_groups(self, name: str, preset: dict) -> str | list[str]:
        """Pick the grouping strategy based on experiment classification state."""
        exp_class_config = self.get_experiment_classification_config()
        if exp_class_config is not None:
            # When experiment classification is enabled, ALWAYS use experiment_group
            groups: str | list[str] = "experiment_group"
            _logger.info(
                f"Classification enabled for plot '{name}': forcing groups={groups}"
            )
            return groups

        # When classification disabled, use explicit YAML setting or default
        groups = preset.get("groups")
        if groups is None or groups == []:
            groups = ["run_name"]
        _logger.info(
            f"Classification disabled for plot '{name}': using groups={groups}"
        )
        return groups


def _build_preset_metrics(name: str, preset: dict) -> list[MetricSpec]:
    """Expand x/y/y2 shortcut fields from a preset into MetricSpec objects."""
    metrics: list[MetricSpec] = []

    x_metric = preset.get("x")
    if x_metric:
        metrics.append(expand_metric_shortcut(x_metric, "x", preset.get("source")))

    y_metric = preset.get("y")
    if y_metric:
        y_stat = preset.get("stat")
        metrics.append(
            expand_metric_shortcut(y_metric, "y", preset.get("source"), y_stat)
        )

    y2_metric = preset.get("y2")
    if y2_metric:
        metrics.append(expand_metric_shortcut(y2_metric, "y2", None))

    if not metrics:
        raise ValueError(f"No metrics defined in preset '{name}'")

    return metrics


def _build_spec_kwargs(
    *,
    name: str,
    preset: dict,
    plot_type: PlotType,
    metrics: list[MetricSpec],
    groups: str | list[str],
) -> dict[str, Any]:
    """Assemble the kwargs dict used to construct PlotSpec / TimeSlicePlotSpec."""
    spec_kwargs: dict[str, Any] = {
        "name": name,
        "plot_type": plot_type,
        "metrics": metrics,
        "title": preset.get("title"),
        "filename": f"{name}.png",
        "description": preset.get("description"),
        "label_by": preset.get("labels"),
        "group_by": groups,
    }

    if "primary_style" in preset:
        spec_kwargs["primary_style"] = Style(**preset["primary_style"])
    if "secondary_style" in preset:
        spec_kwargs["secondary_style"] = Style(**preset["secondary_style"])
    if "supplementary_col" in preset:
        spec_kwargs["supplementary_col"] = preset["supplementary_col"]
    if "autoscale" in preset:
        spec_kwargs["autoscale"] = preset["autoscale"]

    return spec_kwargs
