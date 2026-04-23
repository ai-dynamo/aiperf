# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Data loading functionality for visualization.

Orchestrates per-run loading from JSONL/JSON/CSV/Parquet artifacts and produces
:class:`RunData`. File-format readers live in ``file_readers.py``; server
metrics parsing lives in ``server_metrics_loader.py``; Pydantic models live in
``run_models.py``; derived metric registry lives in ``derived_metrics.py``.
Those symbols are re-exported here so the existing import path continues to work.
"""

from pathlib import Path
from typing import Any

import pandas as pd

from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.plot.constants import (
    NON_METRIC_KEYS,
    PROFILE_EXPORT_AIPERF_JSON,
    PROFILE_EXPORT_GPU_TELEMETRY_JSONL,
    PROFILE_EXPORT_JSONL,
    PROFILE_EXPORT_TIMESLICES_CSV,
    SERVER_METRICS_EXPORT_JSON,
    SERVER_METRICS_EXPORT_PARQUET,
)
from aiperf.plot.core.derived_metrics import (
    DERIVED_METRICS_REGISTRY,
    DerivedMetricCalculator,
)
from aiperf.plot.core.experiment_classifier import (
    ExperimentClassifierMixin,
    TelemetryMixin,
)
from aiperf.plot.core.file_readers import FileReaderMixin
from aiperf.plot.core.plot_specs import ExperimentClassificationConfig
from aiperf.plot.core.run_models import RunData, RunMetadata
from aiperf.plot.core.server_metrics_loader import ServerMetricsLoaderMixin
from aiperf.plot.exceptions import DataLoadError
from aiperf.plot.metric_names import get_metric_display_name

__all__ = [
    "DERIVED_METRICS_REGISTRY",
    "DataLoader",
    "DerivedMetricCalculator",
    "RunData",
    "RunMetadata",
]


class DataLoader(
    FileReaderMixin,
    ServerMetricsLoaderMixin,
    ExperimentClassifierMixin,
    TelemetryMixin,
    AIPerfLoggerMixin,
):
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
        """Load data from a single profiling run directory into :class:`RunData`.

        Raises:
            DataLoadError: If required artifacts are missing or cannot be parsed.
        """
        self._validate_run_path(run_path)
        self.info(f"Loading run from: {run_path}")

        requests_df = self._load_requests(run_path, load_per_request_data)
        aggregated = self._load_aggregated(run_path)
        self._add_all_derived_metrics(aggregated)

        timeslices_df, slice_duration = self._load_timeslices(run_path, aggregated)
        metadata = self._extract_metadata(run_path, requests_df, aggregated)
        gpu_telemetry_df = self._load_gpu_telemetry(run_path, requests_df)
        server_metrics_df, server_metrics_aggregated = self._load_server_metrics(
            run_path
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

    @staticmethod
    def _validate_run_path(run_path: Path) -> None:
        """Ensure the run path exists and is a directory."""
        if not run_path.exists():
            raise DataLoadError("Run path does not exist", path=str(run_path))
        if not run_path.is_dir():
            raise DataLoadError("Run path is not a directory", path=str(run_path))

    def _load_requests(
        self, run_path: Path, load_per_request_data: bool
    ) -> pd.DataFrame | None:
        """Locate and optionally load the per-request JSONL into a DataFrame."""
        jsonl_path = self._resolve_file(run_path, PROFILE_EXPORT_JSONL)
        if jsonl_path is None:
            raise DataLoadError(
                "Required JSONL file not found",
                path=str(run_path / PROFILE_EXPORT_JSONL),
            )
        return self._load_jsonl(jsonl_path) if load_per_request_data else None

    def _load_aggregated(self, run_path: Path) -> dict[str, Any]:
        """Locate and parse the aggregated JSON export."""
        agg_path = self._resolve_file(run_path, PROFILE_EXPORT_AIPERF_JSON)
        if agg_path is None:
            raise DataLoadError(
                "Required JSON file not found",
                path=str(run_path / PROFILE_EXPORT_AIPERF_JSON),
            )
        return self._load_aggregated_json(agg_path)

    def _load_timeslices(
        self, run_path: Path, aggregated: dict[str, Any]
    ) -> tuple[pd.DataFrame | None, float | None]:
        """Load timeslice CSV and extract slice_duration from aggregated config."""
        timeslices_path = self._resolve_file(run_path, PROFILE_EXPORT_TIMESLICES_CSV)
        timeslices_df = None
        if timeslices_path is not None:
            try:
                timeslices_df = self._load_timeslices_csv(timeslices_path)
            except DataLoadError as e:
                self.warning(f"Failed to load timeslice CSV data: {e}")

        slice_duration = None
        if "input_config" in aggregated:
            input_config = aggregated["input_config"]
            if "output" in input_config and "slice_duration" in input_config["output"]:
                slice_duration = input_config["output"]["slice_duration"]
                self.info(f"Extracted slice_duration: {slice_duration}s")

        return timeslices_df, slice_duration

    def _load_gpu_telemetry(
        self, run_path: Path, requests_df: pd.DataFrame | None
    ) -> pd.DataFrame | None:
        """Load GPU telemetry JSONL keyed off of the first request's start time."""
        gpu_telemetry_path = self._resolve_file(
            run_path, PROFILE_EXPORT_GPU_TELEMETRY_JSONL
        )
        if gpu_telemetry_path is None:
            return None

        run_start_time_ns = self._infer_run_start_time_ns(requests_df)
        try:
            return self._load_gpu_telemetry_jsonl(gpu_telemetry_path, run_start_time_ns)
        except DataLoadError as e:
            self.warning(f"Failed to load GPU telemetry data: {e}")
            return None

    @staticmethod
    def _infer_run_start_time_ns(requests_df: pd.DataFrame | None) -> int | None:
        """Pull the earliest request_start_ns from requests_df, if available."""
        if (
            requests_df is None
            or requests_df.empty
            or "request_start_ns" not in requests_df.columns
        ):
            return None
        start_times = requests_df["request_start_ns"].dropna()
        if start_times.empty:
            return None
        first_start = start_times.min()
        if isinstance(first_start, pd.Timestamp):
            return int(first_start.value)
        return int(first_start)

    def _load_server_metrics(
        self, run_path: Path
    ) -> tuple[pd.DataFrame | None, dict[str, Any]]:
        """Load server metrics from Parquet (time-series) and JSON (aggregates).

        Policy: Parquet supplies the tidy time-series when present; JSON supplies
        aggregated stats (and also time-series, if Parquet is missing). If only
        Parquet loaded, aggregates are computed from the time-series.
        """
        server_metrics_df: pd.DataFrame | None = None
        server_metrics_aggregated: dict[str, Any] = {}

        parquet_path = self._resolve_file(run_path, SERVER_METRICS_EXPORT_PARQUET)
        json_path = self._resolve_file(run_path, SERVER_METRICS_EXPORT_JSON)

        if parquet_path is not None:
            try:
                df_parquet, agg_parquet = self._load_server_metrics_parquet(
                    parquet_path
                )
                server_metrics_df = df_parquet
                server_metrics_aggregated = agg_parquet
            except DataLoadError as e:
                self.warning(f"Failed to load server metrics from Parquet: {e}")

        if json_path is not None:
            try:
                df_json, agg_json = self._load_server_metrics_json(json_path)
                server_metrics_df, server_metrics_aggregated = (
                    self._merge_server_metrics_json(
                        server_metrics_df,
                        server_metrics_aggregated,
                        df_json,
                        agg_json,
                    )
                )
            except DataLoadError as e:
                self.warning(f"Failed to load server metrics from JSON: {e}")

        if server_metrics_df is not None and not server_metrics_aggregated:
            self.info("Computing aggregated stats from time-series data...")
            server_metrics_aggregated = self._compute_aggregated_from_timeseries(
                server_metrics_df
            )

        return server_metrics_df, server_metrics_aggregated

    def _merge_server_metrics_json(
        self,
        current_df: pd.DataFrame | None,
        current_agg: dict[str, Any],
        df_json: pd.DataFrame | None,
        agg_json: dict[str, Any],
    ) -> tuple[pd.DataFrame | None, dict[str, Any]]:
        """Merge JSON-loaded server metrics with any Parquet results already in hand."""
        if current_df is not None:
            if agg_json:
                self.info(
                    "Loaded server metrics: time-series from Parquet, "
                    "aggregated stats from JSON"
                )
                return current_df, agg_json
            return current_df, current_agg

        self.info("Loaded server metrics from JSON (Parquet not available)")
        return df_json, agg_json

    def load_multiple_runs(self, run_paths: list[Path]) -> list[RunData]:
        """Load data from multiple profiling runs (no per-request data)."""
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
        """Reload a single run with full per-request data (for interactive drill-down)."""
        return self.load_run(run_path, load_per_request_data=True)

    def _add_all_derived_metrics(self, aggregated: dict[str, Any]) -> None:
        """Apply every calculator in DERIVED_METRICS_REGISTRY; mutates ``aggregated``."""
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
            except (ValueError, TypeError, KeyError, ZeroDivisionError) as e:
                self.warning(f"Failed to calculate derived metric '{metric_name}': {e}")

        if metrics_added:
            self.info(
                f"Added {len(metrics_added)} derived metric(s): {', '.join(metrics_added)} "
                f"(using {gpu_count} GPUs)"
            )

    def get_available_metrics(self, run_data: RunData) -> dict[str, dict[str, str]]:
        """Return {'display_names': {...}, 'units': {...}} for metrics in aggregated."""
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

    def _extract_metadata(
        self,
        run_path: Path,
        requests_df: pd.DataFrame | None,
        aggregated: dict[str, Any],
    ) -> RunMetadata:
        """Build RunMetadata from run_path + requests_df + aggregated input_config."""
        run_name = run_path.name
        config_fields = self._extract_input_config_fields(aggregated)
        timing_fields = self._extract_run_timing_fields(aggregated)
        duration_seconds = self._compute_duration_seconds(requests_df)

        return RunMetadata(
            run_name=run_name,
            run_path=run_path,
            duration_seconds=duration_seconds,
            experiment_type=self._classify_experiment_type(run_path, run_name),
            experiment_group=self._extract_experiment_group(run_path, run_name),
            **config_fields,
            **timing_fields,
        )

    @staticmethod
    def _extract_input_config_fields(aggregated: dict[str, Any]) -> dict[str, Any]:
        """Pull model/concurrency/request_count/endpoint_type from input_config."""
        fields: dict[str, Any] = {
            "model": None,
            "concurrency": None,
            "request_count": None,
            "endpoint_type": None,
        }
        if not aggregated or "input_config" not in aggregated:
            return fields

        config = aggregated["input_config"]
        endpoint = config.get("endpoint", {}) if isinstance(config, dict) else {}
        loadgen = config.get("loadgen", {}) if isinstance(config, dict) else {}

        models = endpoint.get("model_names")
        if models:
            fields["model"] = models[0]
        if "concurrency" in loadgen:
            fields["concurrency"] = loadgen["concurrency"]
        if "request_count" in loadgen:
            fields["request_count"] = loadgen["request_count"]
        if "type" in endpoint:
            fields["endpoint_type"] = endpoint["type"]

        return fields

    @staticmethod
    def _extract_run_timing_fields(aggregated: dict[str, Any]) -> dict[str, Any]:
        """Pull start_time/end_time/was_cancelled off the aggregated top level."""
        if not aggregated:
            return {"start_time": None, "end_time": None, "was_cancelled": False}
        return {
            "start_time": aggregated.get("start_time"),
            "end_time": aggregated.get("end_time"),
            "was_cancelled": aggregated.get("was_cancelled", False),
        }

    @staticmethod
    def _compute_duration_seconds(
        requests_df: pd.DataFrame | None,
    ) -> float | None:
        """Wall-clock duration from earliest request_start_ns to latest request_end_ns."""
        if (
            requests_df is None
            or requests_df.empty
            or "request_start_ns" not in requests_df.columns
            or "request_end_ns" not in requests_df.columns
        ):
            return None
        start_times = requests_df["request_start_ns"].dropna()
        end_times = requests_df["request_end_ns"].dropna()
        if start_times.empty or end_times.empty:
            return None
        duration = end_times.max() - start_times.min()
        if isinstance(duration, pd.Timedelta):
            return duration.total_seconds()
        return duration / 1e9
