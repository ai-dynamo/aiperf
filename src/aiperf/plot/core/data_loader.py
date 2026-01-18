# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Data loading functionality for visualization.

This module provides classes and functions to load AIPerf profiling data
from various output files (JSONL, JSON) and parse them into structured
formats suitable for visualization and analysis.
"""

from fnmatch import fnmatch
from pathlib import Path
from typing import Any

import numpy as np
import orjson
import pandas as pd
from pydantic import Field

from aiperf.common.constants import STAT_KEYS
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.models import AIPerfBaseModel
from aiperf.common.models.record_models import MetricRecordInfo, MetricResult
from aiperf.common.models.server_metrics_models import (
    CounterTimeslice,
    GaugeTimeslice,
    HistogramTimeslice,
    ServerMetricsExportData,
)
from aiperf.plot.constants import (
    NON_METRIC_KEYS,
    PROFILE_EXPORT_AIPERF_JSON,
    PROFILE_EXPORT_GPU_TELEMETRY_JSONL,
    PROFILE_EXPORT_JSONL,
    PROFILE_EXPORT_TIMESLICES_CSV,
    SERVER_METRICS_EXPORT_JSON,
    SERVER_METRICS_EXPORT_PARQUET,
)
from aiperf.plot.core.plot_specs import ExperimentClassificationConfig
from aiperf.plot.exceptions import DataLoadError
from aiperf.plot.metric_names import get_metric_display_name


class RunMetadata(AIPerfBaseModel):
    """Metadata for a single profiling run."""

    run_name: str = Field(description="Name of the run (typically directory name)")
    run_path: Path = Field(description="Path to the run directory")
    model: str | None = Field(default=None, description="Model name used in the run")
    concurrency: int | None = Field(default=None, description="Concurrency level used")
    request_count: int | None = Field(
        default=None, description="Total number of requests"
    )
    duration_seconds: float | None = Field(
        default=None, description="Duration of the run in seconds"
    )
    endpoint_type: str | None = Field(
        default=None, description="Type of endpoint (e.g., 'chat', 'completions')"
    )
    start_time: str | None = Field(
        default=None, description="ISO timestamp when the profiling run started"
    )
    end_time: str | None = Field(
        default=None, description="ISO timestamp when the profiling run ended"
    )
    was_cancelled: bool = Field(
        default=False, description="Whether the profiling run was cancelled early"
    )
    experiment_type: str = Field(
        default="treatment",
        description="Classification of run as 'baseline' or 'treatment' for visualization",
    )
    experiment_group: str = Field(
        default="",
        description="Experiment group identifier extracted from run name or path for grouping variants",
    )


class RunData(AIPerfBaseModel):
    """Complete data for a single profiling run."""

    model_config = {"arbitrary_types_allowed": True}

    metadata: RunMetadata = Field(description="Metadata for the run")
    requests: pd.DataFrame | None = Field(
        description="DataFrame containing per-request data, or None if not loaded"
    )
    aggregated: dict[str, Any] = Field(
        description="Dictionary containing aggregated statistics. The 'metrics' key "
        "contains a dict mapping metric tags to MetricResult objects"
    )
    timeslices: pd.DataFrame | None = Field(
        default=None,
        description="DataFrame containing timeslice data in tidy format with columns: "
        "[Timeslice, Metric, Unit, Stat, Value], or None if not loaded",
    )
    slice_duration: float | None = Field(
        default=None,
        description="Duration of each time slice in seconds, or None if not available",
    )
    gpu_telemetry: pd.DataFrame | None = Field(
        default=None,
        description="DataFrame containing GPU telemetry time series data, or None if not loaded",
    )
    server_metrics: pd.DataFrame | None = Field(
        default=None,
        description="DataFrame containing server metrics time series data in tidy format with columns: "
        "[timestamp_ns, endpoint_url, metric_name, metric_type, value, histogram_count, histogram_sum, "
        "labels_json, unit], or None if not loaded",
    )
    server_metrics_aggregated: dict[str, Any] = Field(
        default_factory=dict,
        description="Dictionary containing aggregated server metrics statistics by metric name. "
        "Structure: {metric_name: {endpoint_url: {labels_key: {type, stats, unit, description, timeslices}}}}",
    )

    def get_metric(self, metric_name: str) -> MetricResult | dict[str, Any] | None:
        """Get a metric from aggregated data."""
        if not self.aggregated:
            return None

        if "metrics" in self.aggregated:
            return self.aggregated["metrics"].get(metric_name)

        return self.aggregated.get(metric_name)


class DerivedMetricCalculator:
    """
    Registry for derived metric calculations.

    Provides a centralized registry of functions that compute derived metrics
    from base metrics when GPU telemetry data is available. New derived metrics
    can be added by registering additional calculator functions.
    """

    @staticmethod
    def per_gpu_throughput(
        aggregated: dict[str, Any], gpu_count: int
    ) -> dict[str, Any] | None:
        """
        Calculate per-GPU throughput by dividing total throughput by GPU count.

        Args:
            aggregated: Aggregated metrics dictionary
            gpu_count: Total number of GPUs

        Returns:
            Dictionary with per-GPU throughput stats and unit, or None if base metric not found
        """
        throughput_data = None

        if (
            "metrics" in aggregated
            and "output_token_throughput" in aggregated["metrics"]
        ):
            throughput_data = aggregated["metrics"]["output_token_throughput"]
        elif "output_token_throughput" in aggregated:
            throughput_data = aggregated["output_token_throughput"]

        if throughput_data is None:
            return None

        per_gpu_data = {"unit": "tokens/sec/gpu"}

        if isinstance(throughput_data, dict):
            for key, value in throughput_data.items():
                if key == "unit":
                    continue
                if isinstance(value, int | float):
                    per_gpu_data[key] = value / gpu_count
        else:
            for stat_name in STAT_KEYS:
                stat_value = getattr(throughput_data, stat_name, None)
                if stat_value is not None and isinstance(stat_value, int | float):
                    per_gpu_data[stat_name] = stat_value / gpu_count

        return per_gpu_data


DERIVED_METRICS_REGISTRY: dict[str, callable] = {
    "output_token_throughput_per_gpu": DerivedMetricCalculator.per_gpu_throughput,
}


class DataLoader(AIPerfLoggerMixin):
    """
    Loader for AIPerf profiling data.

    This class provides methods to load profiling data from various files
    and parse them into structured formats for visualization.
    """

    def __init__(
        self,
        classification_config: ExperimentClassificationConfig | None = None,
        downsampling_config: dict | None = None,
    ):
        """
        Initialize DataLoader.

        Args:
            classification_config: Configuration for baseline/treatment classification
            downsampling_config: Configuration for server metrics downsampling
                Dictionary with keys: enabled (bool), window_size_seconds (float),
                aggregation_method (str). If None, uses defaults.
        """
        super().__init__()
        self.classification_config = classification_config
        self.downsampling_config = downsampling_config or {
            "enabled": True,
            "window_size_seconds": 5.0,
            "aggregation_method": "mean",
        }

    def load_run(self, run_path: Path, load_per_request_data: bool = True) -> RunData:
        """
        Load data from a single profiling run.

        Args:
            run_path: Path to the run directory.
            load_per_request_data: Whether to load per-request data from JSONL.
                Defaults to True. Set to False for multi-run comparisons where
                per-request data is not needed.

        Returns:
            RunData object containing metadata, per-request data, and aggregated
            statistics.

        Raises:
            DataLoadError: If data cannot be loaded from the run directory.
        """
        if not run_path.exists():
            raise DataLoadError("Run path does not exist", path=str(run_path))

        if not run_path.is_dir():
            raise DataLoadError("Run path is not a directory", path=str(run_path))

        self.info(f"Loading run from: {run_path}")

        jsonl_path = run_path / PROFILE_EXPORT_JSONL
        if not jsonl_path.exists():
            raise DataLoadError("Required JSONL file not found", path=str(jsonl_path))

        requests_df = self._load_jsonl(jsonl_path) if load_per_request_data else None

        aggregated = self._load_aggregated_json(run_path / PROFILE_EXPORT_AIPERF_JSON)

        self._add_all_derived_metrics(aggregated)

        timeslices_path = run_path / PROFILE_EXPORT_TIMESLICES_CSV
        timeslices_df = None
        slice_duration = None

        if timeslices_path.exists():
            try:
                timeslices_df = self._load_timeslices_csv(timeslices_path)
            except DataLoadError as e:
                self.warning(f"Failed to load timeslice CSV data: {e}")

        if "input_config" in aggregated:
            input_config = aggregated["input_config"]
            if "output" in input_config and "slice_duration" in input_config["output"]:
                slice_duration = input_config["output"]["slice_duration"]
                self.info(f"Extracted slice_duration: {slice_duration}s")

        metadata = self._extract_metadata(run_path, requests_df, aggregated)

        gpu_telemetry_path = run_path / PROFILE_EXPORT_GPU_TELEMETRY_JSONL
        gpu_telemetry_df = None

        run_start_time_ns = None
        if (
            requests_df is not None
            and not requests_df.empty
            and "request_start_ns" in requests_df.columns
        ):
            start_times = requests_df["request_start_ns"].dropna()
            if not start_times.empty:
                first_start = start_times.min()
                if isinstance(first_start, pd.Timestamp):
                    run_start_time_ns = int(first_start.value)
                else:
                    run_start_time_ns = int(first_start)

        if gpu_telemetry_path.exists():
            try:
                gpu_telemetry_df = self._load_gpu_telemetry_jsonl(
                    gpu_telemetry_path, run_start_time_ns
                )
            except DataLoadError as e:
                self.warning(f"Failed to load GPU telemetry data: {e}")

        # Load server metrics - load BOTH Parquet (time-series) AND JSON (aggregated stats)
        server_metrics_df = None
        server_metrics_aggregated = {}

        server_metrics_parquet_path = run_path / SERVER_METRICS_EXPORT_PARQUET
        server_metrics_json_path = run_path / SERVER_METRICS_EXPORT_JSON

        # Try Parquet first (for time-series data)
        if server_metrics_parquet_path.exists():
            try:
                df_parquet, agg_parquet = self._load_server_metrics_parquet(
                    server_metrics_parquet_path
                )
                server_metrics_df = df_parquet
                server_metrics_aggregated = agg_parquet
            except DataLoadError as e:
                self.warning(f"Failed to load server metrics from Parquet: {e}")

        # Load JSON (for aggregated stats and metadata)
        if server_metrics_json_path.exists():
            try:
                df_json, agg_json = self._load_server_metrics_json(
                    server_metrics_json_path
                )

                # If Parquet provided time-series, use JSON only for aggregated stats
                if server_metrics_df is not None:
                    if agg_json:
                        server_metrics_aggregated = agg_json
                        self.info(
                            "Loaded server metrics: time-series from Parquet, "
                            "aggregated stats from JSON"
                        )
                else:
                    # No Parquet - use JSON for both
                    server_metrics_df = df_json
                    server_metrics_aggregated = agg_json
                    self.info("Loaded server metrics from JSON (Parquet not available)")
            except DataLoadError as e:
                self.warning(f"Failed to load server metrics from JSON: {e}")

        # If we have time-series but no aggregated stats, compute them
        if server_metrics_df is not None and not server_metrics_aggregated:
            self.info("Computing aggregated stats from time-series data...")
            server_metrics_aggregated = self._compute_aggregated_from_timeseries(
                server_metrics_df
            )

        return RunData(
            metadata=metadata,
            requests=requests_df,
            aggregated=aggregated,
            timeslices=timeslices_df,
            slice_duration=slice_duration,
            gpu_telemetry=gpu_telemetry_df,
            server_metrics=server_metrics_df,
            server_metrics_aggregated=server_metrics_aggregated,
        )

    def load_multiple_runs(self, run_paths: list[Path]) -> list[RunData]:
        """
        Load data from multiple profiling runs.

        This method also detects swept parameters across runs. Per-request
        data (JSONL) is not loaded for multi-run comparisons as only aggregated
        statistics are needed.

        Args:
            run_paths: List of paths to run directories.

        Returns:
            List of RunData objects, one for each run. The requests field will
            be None for all runs.

        Raises:
            DataLoadError: If any run cannot be loaded.
        """
        if not run_paths:
            raise DataLoadError("No run paths provided")

        runs = []
        for path in run_paths:
            try:
                run = self.load_run(path, load_per_request_data=False)
                runs.append(run)
            except DataLoadError as e:
                self.error(f"Failed to load run from {path}: {e}")
                raise

        return runs

    def reload_with_details(self, run_path: Path) -> RunData:
        """
        Reload a run with full per-request data.

        This method is useful in interactive mode (HTML or hosted dashboard) where a run was initially
        loaded as part of a multi-run comparison (without per-request data), but
        now detailed analysis is needed for a specific run.

        Args:
            run_path: Path to the run directory to reload.

        Returns:
            RunData object with full per-request data loaded.

        Raises:
            DataLoadError: If data cannot be loaded from the run directory.
        """
        return self.load_run(run_path, load_per_request_data=True)

    def extract_telemetry_data(
        self, aggregated: dict[str, Any]
    ) -> dict[str, Any] | None:
        """
        Extract telemetry data from aggregated statistics.

        Args:
            aggregated: The aggregated data dictionary from profile_export_aiperf.json

        Returns:
            Telemetry data dictionary with 'summary' and 'endpoints' keys, or None if
            telemetry data is not available.
        """
        if not aggregated or "telemetry_data" not in aggregated:
            self.debug("No telemetry data found in aggregated statistics")
            return None

        telemetry = aggregated.get("telemetry_data")

        if not isinstance(telemetry, dict):
            self.warning("Telemetry data exists but has unexpected structure")
            return None

        if "summary" not in telemetry or "endpoints" not in telemetry:
            self.warning("Telemetry data missing expected keys (summary, endpoints)")
            return None

        self.info(
            f"Extracted telemetry data with {len(telemetry.get('endpoints', {}))} endpoints"
        )
        return telemetry

    def get_telemetry_summary(
        self, aggregated: dict[str, Any]
    ) -> dict[str, Any] | None:
        """
        Get telemetry summary information (start_time, end_time, endpoints).

        Args:
            aggregated: The aggregated data dictionary

        Returns:
            Dictionary with keys: start_time, end_time, endpoints_configured,
            endpoints_successful, or None if not available.
        """
        telemetry = self.extract_telemetry_data(aggregated)
        return telemetry.get("summary") if telemetry else None

    def calculate_gpu_count_from_telemetry(
        self, aggregated: dict[str, Any]
    ) -> int | None:
        """
        Calculate total GPU count from telemetry data.

        Counts unique GPUs across all endpoints in the telemetry data structure.

        Args:
            aggregated: The aggregated data dictionary from profile_export_aiperf.json

        Returns:
            Total GPU count across all endpoints, or None if telemetry data is not
            available.
        """
        telemetry = self.extract_telemetry_data(aggregated)
        if not telemetry:
            return None

        endpoints = telemetry.get("endpoints", {})
        if not isinstance(endpoints, dict):
            self.warning("Telemetry endpoints data has unexpected structure")
            return None

        gpu_count = 0
        for _endpoint_name, endpoint_data in endpoints.items():
            if not isinstance(endpoint_data, dict):
                continue

            gpus = endpoint_data.get("gpus", {})
            if isinstance(gpus, dict):
                gpu_count += len(gpus)

        if gpu_count == 0:
            self.debug("No GPUs found in telemetry data")
            return None

        self.info(f"Calculated GPU count from telemetry: {gpu_count}")
        return gpu_count

    def _add_all_derived_metrics(self, aggregated: dict[str, Any]) -> None:
        """
        Add all registered derived GPU metrics to aggregated data when telemetry is available.

        Iterates through the DERIVED_METRICS_REGISTRY and applies each calculator function
        to compute derived metrics from base metrics and GPU telemetry data. Metrics are
        added in-place to the aggregated dictionary.

        Args:
            aggregated: The aggregated data dictionary (will be modified in-place)
        """
        gpu_count = self.calculate_gpu_count_from_telemetry(aggregated)

        if gpu_count is None or gpu_count == 0:
            self.debug(
                "Skipping derived GPU metrics: telemetry data not available or no GPUs found"
            )
            return

        metrics_added = []
        for metric_name, calculator_func in DERIVED_METRICS_REGISTRY.items():
            try:
                result = calculator_func(aggregated, gpu_count)
                if result is not None:
                    aggregated[metric_name] = result
                    metrics_added.append(metric_name)
            except Exception as e:
                self.warning(f"Failed to calculate derived metric '{metric_name}': {e}")

        if metrics_added:
            self.info(
                f"Added {len(metrics_added)} derived metric(s): {', '.join(metrics_added)} "
                f"(using {gpu_count} GPUs)"
            )

    def get_available_metrics(self, run_data: RunData) -> dict[str, dict[str, str]]:
        """
        Get metrics available in the loaded data.

        Extracts metric information from the aggregated data, which has a flat structure
        where metrics are stored at the top level (not nested under a "metrics" key).

        Args:
            run_data: RunData object with loaded aggregated data.

        Returns:
            Dictionary with two keys:
                - "display_names": dict mapping metric tag to display name
                - "units": dict mapping metric tag to unit string
        """
        if not run_data.aggregated:
            self.warning("No aggregated data available")
            return {"display_names": {}, "units": {}}

        display_names = {}
        units = {}

        for key, value in run_data.aggregated.items():
            if key in NON_METRIC_KEYS:
                continue

            if isinstance(value, dict) and "unit" in value and value is not None:
                display_names[key] = get_metric_display_name(key)
                units[key] = value["unit"]

        if not display_names:
            self.warning("No metrics found in aggregated data")
        else:
            self.info(
                f"Found {len(display_names)} available metrics: {sorted(display_names.keys())}"
            )

        return {"display_names": display_names, "units": units}

    def _read_jsonl_with_error_handling(
        self,
        jsonl_path: Path,
        parse_func: callable,
        raise_on_empty: bool = True,
        file_description: str = "JSONL",
    ) -> list[dict] | None:
        """
        Common utility for reading JSONL files with error handling.

        Args:
            jsonl_path: Path to JSONL file
            parse_func: Function to parse each line string into a dict
            raise_on_empty: Whether to raise error if no records found
            file_description: Description for log messages

        Returns:
            List of parsed records, or None if no records found and raise_on_empty=False

        Raises:
            DataLoadError: If file cannot be read or no records found when raise_on_empty=True
        """
        records = []
        corrupted_lines = 0

        try:
            with open(jsonl_path, encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        record = parse_func(line)
                        records.append(record)
                    except (orjson.JSONDecodeError, Exception) as e:
                        corrupted_lines += 1
                        self.warning(
                            f"Skipping invalid line {line_num} in {jsonl_path}: {e}"
                        )
                        continue

            if corrupted_lines > 0:
                self.warning(
                    f"Skipped {corrupted_lines} corrupted lines in {jsonl_path}"
                )

            if not records:
                if raise_on_empty:
                    raise DataLoadError(
                        f"No valid records found in {file_description} file",
                        path=str(jsonl_path),
                    )
                else:
                    self.warning(
                        f"No valid records found in {file_description} file: {jsonl_path}"
                    )
                    return None

            return records

        except OSError as e:
            raise DataLoadError(
                f"Failed to read {file_description} file: {e}", path=str(jsonl_path)
            ) from e

    @staticmethod
    def _calculate_relative_timestamp_seconds(
        timestamp_ns: int, run_start_time_ns: int | None = None
    ) -> float:
        """
        Convert nanosecond timestamp to relative seconds.

        Args:
            timestamp_ns: Absolute timestamp in nanoseconds
            run_start_time_ns: Optional reference start time in nanoseconds.
                If provided, returns relative seconds from this start time.
                If None, returns absolute seconds.

        Returns:
            Timestamp in seconds (relative or absolute)
        """
        if run_start_time_ns:
            return (timestamp_ns - run_start_time_ns) / 1e9
        return timestamp_ns / 1e9

    def _load_jsonl(self, jsonl_path: Path) -> pd.DataFrame:
        """
        Load per-request data from JSONL file.

        Args:
            jsonl_path: Path to the profile_export.jsonl file.

        Returns:
            DataFrame containing per-request data with flattened metrics.

        Raises:
            DataLoadError: If file cannot be read or parsed.
        """
        if not jsonl_path.exists():
            raise DataLoadError("JSONL file not found", path=str(jsonl_path))

        def parse_line(line: str) -> dict:
            metric_record = MetricRecordInfo.model_validate_json(line)
            return self._convert_to_flat_dict(metric_record)

        records = self._read_jsonl_with_error_handling(
            jsonl_path, parse_line, raise_on_empty=True, file_description="JSONL"
        )

        df = pd.DataFrame(records)
        self.info(f"Loaded {len(df)} records from {jsonl_path}")
        return df

    @staticmethod
    def _compute_inter_chunk_latency_stats(values: list[float]) -> dict[str, float]:
        """
        Compute per-request statistics from inter_chunk_latency array.

        These statistics are useful for analyzing stream health, jitter,
        and stability on a per-request basis.

        Args:
            values: List of inter-chunk latency values (in milliseconds).

        Returns:
            Dictionary mapping statistic names to computed values.
            Returns empty dict if values list is empty.
        """
        if not values:
            return {}

        arr = np.array(values)
        return {
            "inter_chunk_latency_avg": float(np.mean(arr)),
            "inter_chunk_latency_p50": float(np.percentile(arr, 50)),
            "inter_chunk_latency_p95": float(np.percentile(arr, 95)),
            "inter_chunk_latency_std": float(np.std(arr)),
            "inter_chunk_latency_min": float(np.min(arr)),
            "inter_chunk_latency_max": float(np.max(arr)),
            "inter_chunk_latency_range": float(np.max(arr) - np.min(arr)),
        }

    def _convert_to_flat_dict(self, record: MetricRecordInfo) -> dict:
        """
        Convert a MetricRecordInfo Pydantic model to a flat dictionary for DataFrame.

        Args:
            record: Pydantic model from JSONL line.

        Returns:
            Flattened dictionary with metrics and metadata at top level.
        """
        flat = {}

        flat.update(record.metadata.model_dump())

        for key, metric_value in record.metrics.items():
            if key == "inter_chunk_latency" and isinstance(metric_value.value, list):
                stats = self._compute_inter_chunk_latency_stats(metric_value.value)
                flat.update(stats)
                continue
            flat[key] = metric_value.value

        if record.error:
            flat["error"] = record.error.model_dump()

        return flat

    def _load_aggregated_json(self, json_path: Path) -> dict[str, Any]:
        """
        Load aggregated statistics from JSON file.

        Parses the metrics in the JSON file into MetricResult objects for type-safe
        access to metric fields.

        Args:
            json_path: Path to the profile_export_aiperf.json file.

        Returns:
            Dictionary containing aggregated statistics. The "metrics" key contains
            a dict mapping metric tags to MetricResult objects.

        Raises:
            DataLoadError: If file cannot be read or parsed.
        """
        if not json_path.exists():
            raise DataLoadError("Required JSON file not found", path=str(json_path))

        try:
            with open(json_path, "rb") as f:
                data = orjson.loads(f.read())

            if "metrics" in data and isinstance(data["metrics"], dict):
                parsed_metrics = {}
                for tag, metric_data in data["metrics"].items():
                    try:
                        parsed_metrics[tag] = MetricResult(**metric_data)
                    except Exception as e:
                        self.warning(f"Failed to parse metric {tag}: {e}")
                        parsed_metrics[tag] = metric_data
                data["metrics"] = parsed_metrics

            self.info(f"Loaded aggregated data from {json_path}")
            return data
        except orjson.JSONDecodeError as e:
            raise DataLoadError(
                f"Failed to parse JSON file: {e}", path=str(json_path)
            ) from e
        except OSError as e:
            raise DataLoadError(
                f"Failed to read JSON file: {e}", path=str(json_path)
            ) from e

    def _load_timeslices_csv(self, csv_path: Path) -> pd.DataFrame:
        """
        Load timeslice data from CSV file.

        Args:
            csv_path: Path to the profile_export_aiperf_timeslices.csv file.

        Returns:
            DataFrame containing timeslice data in tidy/long format with columns:
            [Timeslice, Metric, Unit, Stat, Value]

        Raises:
            DataLoadError: If file cannot be read or parsed.
        """
        if not csv_path.exists():
            raise DataLoadError("Timeslices CSV file not found", path=str(csv_path))

        try:
            df = pd.read_csv(csv_path)

            expected_columns = ["Timeslice", "Metric", "Unit", "Stat", "Value"]
            if not all(col in df.columns for col in expected_columns):
                raise DataLoadError(
                    f"CSV file missing expected columns. Expected: {expected_columns}, "
                    f"Found: {list(df.columns)}",
                    path=str(csv_path),
                )

            self.info(
                f"Loaded timeslice data from {csv_path} ({len(df)} rows, "
                f"{df['Timeslice'].nunique()} timeslices)"
            )
            return df
        except pd.errors.ParserError as e:
            raise DataLoadError(
                f"Failed to parse CSV file: {e}", path=str(csv_path)
            ) from e
        except OSError as e:
            raise DataLoadError(
                f"Failed to read CSV file: {e}", path=str(csv_path)
            ) from e

    def _load_gpu_telemetry_jsonl(
        self, jsonl_path: Path, run_start_time_ns: int | None = None
    ) -> pd.DataFrame | None:
        """
        Load GPU telemetry time series data from JSONL file.

        Args:
            jsonl_path: Path to the gpu_telemetry_export.jsonl file.
            run_start_time_ns: Optional run start time in nanoseconds for relative timestamps.
                If not provided, timestamps will be kept as absolute values.

        Returns:
            DataFrame containing GPU telemetry data with flattened metrics,
            or None if the file doesn't exist.

        Raises:
            DataLoadError: If file exists but cannot be read or parsed.
        """
        if not jsonl_path.exists():
            self.debug(f"GPU telemetry file not found: {jsonl_path}")
            return None

        def parse_line(line: str) -> dict:
            data = orjson.loads(line.encode("utf-8"))

            telemetry_data = data.pop("telemetry_data", {})
            flat_record = {**data, **telemetry_data}

            if "timestamp_ns" in flat_record:
                flat_record["timestamp_s"] = self._calculate_relative_timestamp_seconds(
                    flat_record["timestamp_ns"], run_start_time_ns
                )

            return flat_record

        records = self._read_jsonl_with_error_handling(
            jsonl_path,
            parse_line,
            raise_on_empty=False,
            file_description="GPU telemetry",
        )

        if records is None:
            return None

        df = pd.DataFrame(records)
        self.info(
            f"Loaded {len(df)} GPU telemetry records from {jsonl_path} "
            f"({df['gpu_index'].nunique() if 'gpu_index' in df.columns else 0} GPUs)"
        )
        return df

    def _load_server_metrics_json(
        self, json_path: Path
    ) -> tuple[pd.DataFrame | None, dict[str, Any]]:
        """
        Load server metrics from JSON export file (preferred format).

        Parses ServerMetricsExportData structure and extracts both time series
        data (from timeslices) and aggregated statistics. Handles all metric types
        (GAUGE, COUNTER, HISTOGRAM) and multi-endpoint configurations.

        Args:
            json_path: Path to the server_metrics_export.json file

        Returns:
            Tuple of (timeseries_df, aggregated_dict):
            - timeseries_df: Tidy DataFrame for time series plots, or None if no timeslices
            - aggregated_dict: Nested dict for multi-run comparison

        Raises:
            DataLoadError: If file exists but cannot be read or parsed
        """
        if not json_path.exists():
            self.debug(f"Server metrics JSON file not found: {json_path}")
            return None, {}

        try:
            with open(json_path, "rb") as f:
                data = orjson.loads(f.read())

            export_data = ServerMetricsExportData.model_validate(data)

            # Build aggregated dict
            aggregated: dict[str, Any] = {}
            rows: list[dict[str, Any]] = []

            for metric_name, metric_data in export_data.metrics.items():
                aggregated[metric_name] = {}

                for series in metric_data.series:
                    endpoint_url = series.endpoint_url or "unknown"
                    labels_key = (
                        orjson.dumps(
                            series.labels, option=orjson.OPT_SORT_KEYS
                        ).decode()
                        if series.labels
                        else "{}"
                    )

                    if endpoint_url not in aggregated[metric_name]:
                        aggregated[metric_name][endpoint_url] = {}

                    # Store aggregated stats
                    stats_value = (
                        series.stats if series.stats is not None else series.value
                    )
                    aggregated[metric_name][endpoint_url][labels_key] = {
                        "type": metric_data.type.value,
                        "stats": stats_value,
                        "unit": metric_data.unit,
                        "description": metric_data.description,
                        "timeslices": series.timeslices,
                    }

                    # Build time series rows from timeslices (if present)
                    if series.timeslices:
                        for ts in series.timeslices:
                            # Use midpoint of timeslice as timestamp
                            timestamp_ns = (ts.start_ns + ts.end_ns) // 2

                            row = {
                                "timestamp_ns": timestamp_ns,
                                "endpoint_url": endpoint_url,
                                "metric_name": metric_name,
                                "metric_type": metric_data.type.value,
                                "labels_json": labels_key,
                                "unit": metric_data.unit or "",
                            }

                            # Add type-specific fields
                            if isinstance(ts, GaugeTimeslice):
                                row["value"] = ts.avg
                                row["histogram_count"] = None
                                row["histogram_sum"] = None
                            elif isinstance(ts, CounterTimeslice):
                                row["value"] = ts.rate
                                row["histogram_count"] = None
                                row["histogram_sum"] = None
                            elif isinstance(ts, HistogramTimeslice):
                                row["value"] = ts.avg
                                row["histogram_count"] = ts.count
                                row["histogram_sum"] = ts.sum

                            rows.append(row)

            # Create DataFrame from rows
            df = pd.DataFrame(rows) if rows else None

            self.info(
                f"Loaded {len(export_data.metrics)} server metrics from {json_path} "
                f"({len(rows)} timeslice data points)"
            )
            return df, aggregated

        except orjson.JSONDecodeError as e:
            raise DataLoadError(
                f"Failed to parse server metrics JSON: {e}", path=str(json_path)
            ) from e
        except Exception as e:
            raise DataLoadError(
                f"Failed to load server metrics from JSON: {e}", path=str(json_path)
            ) from e

    def _load_server_metrics_parquet(
        self, parquet_path: Path
    ) -> tuple[pd.DataFrame | None, dict[str, Any]]:
        """
        Load server metrics from Parquet export file (fast binary format).

        Parquet format stores server metrics in wide format with all labels
        as separate columns. This is the most efficient format for large
        datasets with high query performance.

        Args:
            parquet_path: Path to the server_metrics_export.parquet file

        Returns:
            Tuple of (timeseries_df, aggregated_dict):
            - timeseries_df: Tidy DataFrame for time series plots
            - aggregated_dict: Nested dict for multi-run comparison

        Raises:
            DataLoadError: If file exists but cannot be read or parsed
        """
        if not parquet_path.exists():
            self.debug(f"Server metrics Parquet file not found: {parquet_path}")
            return None, {}

        try:
            import pyarrow.parquet as pq

            # Read Parquet file
            table = pq.read_table(parquet_path)
            df_wide = table.to_pandas()

            # Identify label columns (columns that aren't core metrics)
            core_columns = {
                "endpoint_url",
                "metric_name",
                "metric_type",
                "description",
                "timestamp_ns",
                "value",
                "sum",
                "count",
                "bucket_le",
                "bucket_count",
            }
            label_columns = [c for c in df_wide.columns if c not in core_columns]

            # Parquet contains CUMULATIVE data - need to compute deltas
            # For histograms: buckets are in separate rows, use +Inf bucket for totals
            # Filter histogram data to only aggregate rows (bucket_le == '+Inf')
            is_histogram = df_wide["metric_type"] == "histogram"
            is_aggregate_bucket = df_wide["bucket_le"] == "+Inf"

            # For histograms, keep only +Inf bucket rows (total aggregates)
            # For other types, keep all rows
            df_filtered = df_wide[~is_histogram | is_aggregate_bucket].copy()

            self.debug(
                f"Filtered Parquet: {len(df_wide)} rows → {len(df_filtered)} rows "
                f"(removed per-bucket histogram rows)"
            )

            # Build series key for grouping
            df_filtered["labels_json"] = df_filtered.apply(
                lambda row: orjson.dumps(
                    {
                        k: row[k]
                        for k in label_columns
                        if pd.notna(row[k]) and row[k] != ""
                    },
                    option=orjson.OPT_SORT_KEYS,
                ).decode()
                if any(pd.notna(row[k]) and row[k] != "" for k in label_columns)
                else "{}",
                axis=1,
            )

            # Group by metric + endpoint + labels to compute deltas
            rows = []
            grouped = df_filtered.groupby(
                ["metric_name", "endpoint_url", "labels_json", "metric_type"]
            )

            for (metric_name, endpoint_url, labels_json, metric_type), group in grouped:
                # Sort by timestamp
                group = group.sort_values("timestamp_ns")

                # Compute deltas for cumulative metrics (COUNTER and HISTOGRAM)
                if metric_type in ["counter", "histogram"]:
                    # For each timestamp, compute delta from previous
                    group["delta_count"] = group["count"].diff()
                    group["delta_sum"] = group["sum"].diff()

                    # First row has no previous - skip or use cumulative
                    # Skip first row to avoid large initial values
                    group = group[1:]

                    # Compute rate or average from deltas
                    if metric_type == "counter":
                        # For counters: use delta as the value (rate will be computed later)
                        computed_values = group["delta_count"]
                    else:
                        # For histograms: avg = delta_sum / delta_count
                        computed_values = (
                            group["delta_sum"] / group["delta_count"]
                        ).where(group["delta_count"] > 0, 0)

                else:
                    # GAUGE - use value directly (not cumulative)
                    computed_values = group["value"]

                # Create tidy rows
                for idx, computed_value in zip(
                    group.index, computed_values, strict=False
                ):
                    row_wide = group.loc[idx]
                    tidy_row = {
                        "timestamp_ns": row_wide["timestamp_ns"],
                        "endpoint_url": endpoint_url,
                        "metric_name": metric_name,
                        "metric_type": metric_type,
                        "labels_json": labels_json,
                        "unit": row_wide.get("unit", "")
                        if pd.notna(row_wide.get("unit"))
                        else "",
                        "value": computed_value if pd.notna(computed_value) else None,
                        "histogram_count": row_wide.get("delta_count")
                        if metric_type == "histogram"
                        else None,
                        "histogram_sum": row_wide.get("delta_sum")
                        if metric_type == "histogram"
                        else None,
                    }
                    rows.append(tidy_row)

            # Create tidy DataFrame
            df_tidy = pd.DataFrame(rows) if rows else None

            # Apply time-window aggregation to reduce data density
            # Parquet has 100-200x more points than JSON - downsample based on config
            if df_tidy is not None and not df_tidy.empty:
                if self.downsampling_config["enabled"]:
                    window_size_seconds = self.downsampling_config[
                        "window_size_seconds"
                    ]
                    aggregation_method = self.downsampling_config["aggregation_method"]
                    df_tidy = self._downsample_server_metrics_to_windows(
                        df_tidy,
                        window_size_ns=int(window_size_seconds * 1e9),
                        aggregation_method=aggregation_method,
                    )
                else:
                    self.info("Server metrics downsampling disabled by configuration")

            # Build aggregated dict (empty for Parquet - compute on demand)
            aggregated = {}

            unique_metrics = len(df_wide["metric_name"].unique())
            self.info(
                f"Loaded server metrics from Parquet: {unique_metrics} metrics, "
                f"{len(rows)} raw points → {len(df_tidy) if df_tidy is not None else 0} "
                f"windowed points (5s aggregation)"
            )

            return df_tidy, aggregated

        except ImportError as e:
            raise DataLoadError(
                "pyarrow is required to load Parquet files. Install with: pip install pyarrow",
                path=str(parquet_path),
            ) from e
        except Exception as e:
            raise DataLoadError(
                f"Failed to load server metrics from Parquet: {e}",
                path=str(parquet_path),
            ) from e

    def _downsample_server_metrics_to_windows(
        self,
        df: pd.DataFrame,
        window_size_ns: int = 5_000_000_000,
        aggregation_method: str = "mean",
    ) -> pd.DataFrame:
        """
        Downsample server metrics to time windows for efficient plotting.

        Aggregates high-frequency Parquet data into time windows (default 5s)
        to match JSON timeslice granularity and improve rendering performance.
        Reduces data points by ~100x while preserving visual fidelity.

        Args:
            df: Tidy server metrics DataFrame
            window_size_ns: Window size in nanoseconds (default: 5 seconds)
            aggregation_method: Method for aggregating values ("mean", "max", "min", "median")

        Returns:
            Downsampled DataFrame with same schema
        """
        if df.empty:
            return df

        # Validate aggregation method
        valid_methods = ["mean", "max", "min", "median"]
        if aggregation_method not in valid_methods:
            self.warning(
                f"Invalid aggregation method '{aggregation_method}', using 'mean'. "
                f"Valid options: {valid_methods}"
            )
            aggregation_method = "mean"

        # Create window bins
        min_ts = df["timestamp_ns"].min()
        df["window"] = ((df["timestamp_ns"] - min_ts) // window_size_ns).astype(int)

        # Group by (metric, endpoint, labels, window) and aggregate
        agg_funcs = {
            "timestamp_ns": "mean",  # Use window midpoint
            "value": aggregation_method,  # User-configurable aggregation
            "histogram_count": "sum",  # Sum counts (for histograms)
            "histogram_sum": "sum",  # Sum sums (for histograms)
            "metric_type": "first",  # Metadata (same for all in group)
            "unit": "first",
        }

        grouped = df.groupby(
            ["metric_name", "endpoint_url", "labels_json", "window"],
            dropna=False,
        ).agg(agg_funcs)

        # Reset index to get back to flat DataFrame
        df_downsampled = grouped.reset_index(drop=False)
        df_downsampled = df_downsampled.drop(columns=["window"])

        # Convert timestamp back to int
        df_downsampled["timestamp_ns"] = df_downsampled["timestamp_ns"].astype("int64")

        self.debug(
            f"Downsampled server metrics: {len(df)} → {len(df_downsampled)} rows "
            f"({len(df) / len(df_downsampled):.1f}x reduction, {window_size_ns / 1e9:.1f}s windows, "
            f"{aggregation_method} aggregation)"
        )

        return df_downsampled

    def _compute_aggregated_from_timeseries(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Compute aggregated statistics from time-series DataFrame.

        Used when Parquet is loaded but JSON is not available. Computes
        basic statistics (avg, min, max, p50, p95, p99) from time-series data.

        Args:
            df: Time-series DataFrame with columns: timestamp_ns, endpoint_url,
                metric_name, metric_type, value, labels_json, etc.

        Returns:
            Aggregated dict matching JSON format:
            {metric_name: {endpoint_url: {labels_key: {type, stats, unit, description}}}}
        """
        if df is None or df.empty:
            return {}

        aggregated: dict[str, Any] = {}

        # Group by metric, endpoint, labels
        grouped = df.groupby(["metric_name", "endpoint_url", "labels_json"])

        for (metric_name, endpoint_url, labels_json), group in grouped:
            # Initialize nested structure
            if metric_name not in aggregated:
                aggregated[metric_name] = {}
            if endpoint_url not in aggregated[metric_name]:
                aggregated[metric_name][endpoint_url] = {}

            # Compute statistics from values
            values = group["value"].dropna()

            if len(values) == 0:
                continue

            stats = {
                "avg": float(values.mean()),
                "min": float(values.min()),
                "max": float(values.max()),
                "p50": float(values.quantile(0.5)),
                "p95": float(values.quantile(0.95)),
                "p99": float(values.quantile(0.99)),
            }

            # Get metric type, unit, and description from first row
            metric_type = (
                group["metric_type"].iloc[0] if "metric_type" in group else None
            )
            unit = group["unit"].iloc[0] if "unit" in group else ""
            description = (
                group["description"].iloc[0]
                if "description" in group and pd.notna(group["description"].iloc[0])
                else ""
            )

            aggregated[metric_name][endpoint_url][labels_json] = {
                "type": metric_type,
                "stats": stats,
                "unit": unit,
                "description": description,
                "timeslices": None,  # Not computed here
            }

        self.info(
            f"Computed aggregated stats for {len(aggregated)} metrics from time-series data"
        )
        return aggregated

    def _classify_experiment_type(self, run_path: Path, run_name: str) -> str:
        """
        Classify run as baseline or treatment.

        Priority (highest to lowest):
        1. Pattern matching from plot_config.yaml
        2. Default from plot_config.yaml (or "treatment" if no config)

        Args:
            run_path: Path to the run directory
            run_name: Name of the run (typically directory name)

        Returns:
            "baseline" or "treatment"
        """
        if self.classification_config:
            for pattern in self.classification_config.baselines:
                if fnmatch(run_name, pattern) or fnmatch(str(run_path), pattern):
                    return "baseline"

            for pattern in self.classification_config.treatments:
                if fnmatch(run_name, pattern) or fnmatch(str(run_path), pattern):
                    return "treatment"

            return self.classification_config.default

        return "treatment"

    def _extract_experiment_group(self, run_path: Path, run_name: str) -> str:
        """
        Extract experiment group identifier from run path.

        If experiment classification is configured and the parent directory matches
        any baseline or treatment pattern, uses parent directory name.
        Otherwise uses run directory name.

        Args:
            run_path: Path to the run directory
            run_name: Name of the run (typically directory name)

        Returns:
            Experiment group identifier for grouping runs
        """
        # Try parent-based grouping if classification config exists
        if self.classification_config:
            parent = run_path.parent
            if parent and parent.name:
                parent_name = parent.name

                # Check if parent matches any baseline pattern
                for pattern in self.classification_config.baselines:
                    if fnmatch(parent_name, pattern):
                        return parent_name

                # Check if parent matches any treatment pattern
                for pattern in self.classification_config.treatments:
                    if fnmatch(parent_name, pattern):
                        return parent_name

        # Fallback: use run_name
        result = run_name if run_name else str(run_path.name)

        if not result:
            self.warning(
                f"Could not extract experiment_group from {run_path}, using full path"
            )
            result = str(run_path)

        return result

    def _extract_metadata(
        self,
        run_path: Path,
        requests_df: pd.DataFrame | None,
        aggregated: dict[str, Any],
    ) -> RunMetadata:
        """
        Extract metadata from loaded data.

        Args:
            run_path: Path to the run directory.
            requests_df: DataFrame with per-request data, or None if not loaded.
            aggregated: Aggregated statistics dictionary containing input_config.

        Returns:
            RunMetadata object with extracted information.
        """
        run_name = run_path.name
        model = None
        concurrency = None
        request_count = None
        endpoint_type = None
        start_time = None
        end_time = None
        was_cancelled = False

        if aggregated and "input_config" in aggregated:
            config = aggregated["input_config"]

            if "endpoint" in config and "model_names" in config["endpoint"]:
                models = config["endpoint"]["model_names"]
                if models:
                    model = models[0]

            if "loadgen" in config and "concurrency" in config["loadgen"]:
                concurrency = config["loadgen"]["concurrency"]

            if "loadgen" in config and "request_count" in config["loadgen"]:
                request_count = config["loadgen"]["request_count"]

            if "endpoint" in config and "type" in config["endpoint"]:
                endpoint_type = config["endpoint"]["type"]

        if aggregated:
            start_time = aggregated.get("start_time")
            end_time = aggregated.get("end_time")
            was_cancelled = aggregated.get("was_cancelled", False)

        duration_seconds = None
        if (
            requests_df is not None
            and not requests_df.empty
            and "request_start_ns" in requests_df.columns
            and "request_end_ns" in requests_df.columns
        ):
            start_times = requests_df["request_start_ns"].dropna()
            end_times = requests_df["request_end_ns"].dropna()
            if not start_times.empty and not end_times.empty:
                duration = end_times.max() - start_times.min()
                if isinstance(duration, pd.Timedelta):
                    duration_seconds = duration.total_seconds()
                else:
                    duration_seconds = duration / 1e9

        experiment_type = self._classify_experiment_type(run_path, run_name)

        experiment_group = self._extract_experiment_group(run_path, run_name)

        return RunMetadata(
            run_name=run_name,
            run_path=run_path,
            model=model,
            concurrency=concurrency,
            request_count=request_count,
            duration_seconds=duration_seconds,
            endpoint_type=endpoint_type,
            start_time=start_time,
            end_time=end_time,
            was_cancelled=was_cancelled,
            experiment_type=experiment_type,
            experiment_group=experiment_group,
        )
