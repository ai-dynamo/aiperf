# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Plot configuration loader for YAML-based plot definitions.

Loads plot specifications from YAML files with the following priority:
1. Custom path (if provided via --config flag)
2. User home config (~/.aiperf/plot_config.yaml) - auto-created on first run
3. Default shipped config (src/aiperf/plot/default_plot_config.yaml)
"""

import difflib
import logging
import re
import shutil
from pathlib import Path

from ruamel.yaml import YAML

from aiperf.plot.constants import ALL_STAT_KEYS
from aiperf.plot.core.plot_specs import (
    DataSource,
    ExperimentClassificationConfig,
    MetricSpec,
    PlotSpec,
    PlotType,
    Style,
    TimeSlicePlotSpec,
)
from aiperf.plot.metric_names import (
    get_aggregated_metrics,
    get_gpu_metrics,
    get_request_metrics,
    get_timeslice_metrics,
)

_logger = logging.getLogger(__name__)


def _detect_invalid_stat_pattern(metric_name: str) -> str | None:
    """
    Detect if metric name has an invalid stat-like suffix pattern.

    Args:
        metric_name: Full metric name

    Returns:
        The invalid stat suffix if detected (e.g., "p67"), None otherwise
    """
    if "_" not in metric_name:
        return None

    _, potential_stat = metric_name.rsplit("_", 1)

    if potential_stat in ["avg", "min", "max", "std"]:
        return None

    if (
        potential_stat.startswith("p")
        and potential_stat[1:].isdigit()
        and potential_stat not in ALL_STAT_KEYS
    ):
        return potential_stat

    return None


def _parse_and_validate_metric_name(metric_name: str) -> tuple[str, str | None]:
    """
    Parse and validate metric name format.

    Supports two formats:
    1. {metric_name}_{stat} - e.g., "request_latency_p50"
    2. {metric_name} - e.g., "request_number"

    Args:
        metric_name: Metric shortcut name

    Returns:
        Tuple of (base_metric_name, stat) where stat is None if no suffix

    Raises:
        ValueError: If metric name has invalid stat suffix pattern
    """
    if "_" not in metric_name:
        return (metric_name, None)

    base_name, potential_stat = metric_name.rsplit("_", 1)

    if potential_stat in ALL_STAT_KEYS:
        return (base_name, potential_stat)

    invalid_stat = _detect_invalid_stat_pattern(metric_name)
    if invalid_stat:
        close_matches = difflib.get_close_matches(
            invalid_stat, ALL_STAT_KEYS, n=3, cutoff=0.6
        )

        error_msg = (
            f"Invalid stat suffix '{invalid_stat}' in metric '{metric_name}'.\n\n"
        )
        error_msg += "Valid stat suffixes are:\n"
        error_msg += f"  {', '.join(ALL_STAT_KEYS)}\n"

        if close_matches:
            error_msg += "\nDid you mean one of these?\n"
            for match in close_matches:
                error_msg += f"  - {base_name}_{match}\n"

        raise ValueError(error_msg)

    return (metric_name, None)


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

        metrics = []

        x_metric = preset.get("x")
        if x_metric:
            metrics.append(
                self._expand_metric_shortcut(x_metric, "x", preset.get("source"))
            )

        y_metric = preset.get("y")
        if y_metric:
            y_stat = preset.get("stat")
            metrics.append(
                self._expand_metric_shortcut(
                    y_metric, "y", preset.get("source"), y_stat
                )
            )

        y2_metric = preset.get("y2")
        if y2_metric:
            metrics.append(self._expand_metric_shortcut(y2_metric, "y2", None))

        if not metrics:
            raise ValueError(f"No metrics defined in preset '{name}'")

        exp_class_config = self.get_experiment_classification_config()
        if exp_class_config is not None:
            # When experiment classification is enabled, ALWAYS use experiment_group
            groups = "experiment_group"
            _logger.info(
                f"Classification enabled for plot '{name}': forcing groups={groups}"
            )
        else:
            # When classification disabled, use explicit YAML setting or default
            groups = preset.get("groups")
            if groups is None or groups == []:
                groups = ["run_name"]
            _logger.info(
                f"Classification disabled for plot '{name}': using groups={groups}"
            )

        spec_kwargs = {
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

        if "use_slice_duration" in preset:
            spec_kwargs["use_slice_duration"] = preset["use_slice_duration"]
            return TimeSlicePlotSpec(**spec_kwargs)

        return PlotSpec(**spec_kwargs)

    def _is_server_metric(self, metric_name: str) -> bool:
        """
        Check if a metric name appears to be a server metric.

        This is a heuristic-based detection used during config parsing to determine
        the data source for metrics. The actual metric data comes from export files,
        so this is only used for automatic source inference in plot specifications.

        Server metrics typically follow Prometheus naming conventions:
        - Contains colon separator (e.g., "vllm:metric_name", "triton:metric")
        - Common prefixes: vllm, triton, http, dynamo, nvidia, nv
        - May include endpoint/label filters: metric[endpoint], metric{labels}

        Note: If you have custom Prometheus metrics that don't match these patterns,
        explicitly set `source: server_metrics` in your plot specification.

        Args:
            metric_name: Metric name to check

        Returns:
            True if likely a server metric, False otherwise
        """
        # Strip endpoint/label filters first
        base_name = re.sub(r"\[.*?\]|\{.*?\}", "", metric_name).strip()

        # Check for Prometheus namespace convention (most reliable indicator)
        # Format: namespace:metric_name (e.g., "vllm:kv_cache_usage")
        if ":" in base_name:
            return True

        # Check for common Prometheus/server metric prefixes
        # Includes standard patterns from vLLM, Triton, HTTP, DCGM, NVIDIA
        prometheus_prefixes = [
            "vllm_",
            "sglang_",
            "trtllm_",
            "nv_inference_",  # Triton Inference Server (most specific)
            "nv_gpu_",  # Triton GPU metrics
            "nv_",  # Generic Triton/NVIDIA
            "http_",
            "https_",
            "dynamo_",
            "nvidia_",
            "dcgm_",
            "gpu_",
            "process_",
            "node_",
            "container_",
        ]
        if any(base_name.startswith(prefix) for prefix in prometheus_prefixes):
            return True

        # Check for common Prometheus suffixes (counter/gauge indicators)
        prometheus_suffixes = [
            "_total",
            "_count",
            "_sum",
            "_bucket",
            "_seconds",
            "_milliseconds",
            "_microseconds",
            "_us",  # Triton microseconds
            "_ms",  # Triton milliseconds
            "_ns",
            "_bytes",
        ]
        return any(base_name.endswith(suffix) for suffix in prometheus_suffixes)

    def _expand_metric_shortcut(
        self,
        metric_value: str | dict,
        axis: str,
        source_override: str | None = None,
        stat_override: str | None = None,
    ) -> MetricSpec:
        """
        Expand metric shortcut to full MetricSpec using dynamic pattern matching.

        Supports two formats:
        1. Dict format: {"metric": "request_latency", "stat": "avg"}
        2. String format (legacy): "request_latency_avg" or "request_number"

        Args:
            metric_value: Metric as dict with 'metric' and 'stat' keys, or string shortcut
            axis: Axis assignment ("x", "y", "y2")
            source_override: Override data source (for timeslice plots)
            stat_override: Override stat (for timeslice plots)

        Returns:
            MetricSpec object

        Raises:
            ValueError: If metric name or stat is not recognized
        """
        if isinstance(metric_value, dict):
            base_name = metric_value["metric"]
            stat = metric_value.get("stat")
            # Extract source from dict if present (overrides source_override)
            if "source" in metric_value and not source_override:
                source_override = metric_value["source"]
        else:
            base_name, stat = _parse_and_validate_metric_name(metric_value)

        # If source is explicitly specified, use it and skip validation
        # This allows users to specify server metrics that don't match heuristic patterns
        if source_override:
            source = DataSource(source_override)
        else:
            # Auto-detect source from metric name
            if base_name in get_aggregated_metrics():
                source = DataSource.AGGREGATED
            elif base_name in get_request_metrics():
                source = DataSource.REQUESTS
            elif base_name in get_timeslice_metrics():
                source = DataSource.TIMESLICES
            elif base_name in get_gpu_metrics():
                source = DataSource.GPU_TELEMETRY
            elif self._is_server_metric(base_name):
                # Server metrics (Prometheus-style names like "vllm:kv_cache_usage_perc")
                source = DataSource.SERVER_METRICS
            else:
                all_known = (
                    get_aggregated_metrics()
                    + get_request_metrics()
                    + get_timeslice_metrics()
                    + get_gpu_metrics()
                )
                raise ValueError(
                    f"Unknown metric: '{base_name}' (from shortcut '{metric_value}'). "
                    f"Known metrics: {all_known}. For server metrics, use Prometheus-style names like 'vllm:metric_name'."
                )
        if stat_override:
            stat = stat_override

        return MetricSpec(name=base_name, source=source, axis=axis, stat=stat)
