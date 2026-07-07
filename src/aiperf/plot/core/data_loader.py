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

from dataclasses import asdict
from pathlib import Path
from typing import Any

import orjson
import pandas as pd
from pydantic import ValidationError

from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.models.export_models import JsonMetricResult
from aiperf.plot.constants import (
    NON_METRIC_KEYS,
    PROFILE_EXPORT_AIPERF_AGGREGATE_JSON,
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


# Stat-key suffixes the confidence-aggregate exporter flattens onto metric names
# (one entry per ``(metric, stat)`` pair, e.g. ``request_latency_p99``).
# ``DataLoader._unflatten_confidence_metrics`` rpartitions each flat key on the
# last underscore and groups by the head when the tail is in this set; tails not
# in the set are treated as a metric name with no stat suffix (bucketed to avg).
_KNOWN_STAT_SUFFIXES: frozenset[str] = frozenset(
    (
        "avg",
        "p1",
        "p5",
        "p10",
        "p25",
        "p50",
        "p75",
        "p90",
        "p95",
        "p99",
        "min",
        "max",
        "std",
        "count",
        "sum",
    )
)


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

        # Per-cell confidence-aggregate cells (trials>1 sweep cells) carry
        # ``profile_export_aiperf_aggregate.json`` and no JSONL/single-run JSON.
        # Route them through the aggregate-only loader, which re-shapes the
        # confidence metrics into the single-run shape the pipeline expects.
        if self._resolve_file(run_path, PROFILE_EXPORT_JSONL) is None:
            aggregate_path = self._resolve_file(
                run_path, PROFILE_EXPORT_AIPERF_AGGREGATE_JSON
            )
            if aggregate_path is not None:
                return self._load_aggregate_only(run_path, aggregate_path)

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
            if isinstance(input_config, dict):
                output_config = input_config.get("output")
                artifacts_config = input_config.get("artifacts")
                output_config = output_config if isinstance(output_config, dict) else {}
                artifacts_config = (
                    artifacts_config if isinstance(artifacts_config, dict) else {}
                )
                slice_duration = output_config.get("slice_duration")
                if slice_duration is None:
                    slice_duration = artifacts_config.get("slice_duration")
                if slice_duration is not None:
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

    def _load_aggregate_only(self, run_path: Path, aggregate_path: Path) -> RunData:
        """Load a per-cell confidence-aggregate dir as a pseudo-run.

        Reads ``profile_export_aiperf_aggregate.json`` (no JSONL exists for
        aggregate cells) and re-shapes the confidence-aggregate metrics into the
        single-run format so the rest of the plot pipeline operates uniformly.
        Returns ``RunData`` with ``requests=None`` and only ``metadata`` +
        ``aggregated`` populated.
        """
        raw = self._read_aggregate_json(aggregate_path)
        unflattened = self._unflatten_confidence_metrics(raw.get("metrics", {}) or {})

        aggregated: dict[str, Any] = dict(raw)
        aggregated["metrics"] = unflattened
        aggregated.setdefault("aggregation_type", "confidence")

        self._mirror_metrics_to_top_level(aggregated, unflattened)
        self._plumb_variation_values_into_input_config(aggregated, raw)

        self._add_all_derived_metrics(aggregated)
        metadata = self._extract_metadata(
            run_path, requests_df=None, aggregated=aggregated
        )
        self.info(f"Loaded aggregate-only run from {run_path}")
        return RunData(metadata=metadata, requests=None, aggregated=aggregated)

    def _read_aggregate_json(self, aggregate_path: Path) -> dict[str, Any]:
        """Read+parse a confidence-aggregate JSON, raising DataLoadError on failure."""
        try:
            return orjson.loads(self._read_bytes(aggregate_path))
        except orjson.JSONDecodeError as e:
            raise DataLoadError(
                f"Failed to parse aggregate JSON: {e}", path=str(aggregate_path)
            ) from e
        except OSError as e:
            raise DataLoadError(
                f"Failed to read aggregate JSON: {e}", path=str(aggregate_path)
            ) from e

    def _mirror_metrics_to_top_level(
        self,
        aggregated: dict[str, Any],
        unflattened: dict[str, JsonMetricResult | dict[str, Any]],
    ) -> None:
        """Copy un-flattened metrics to top-level keys for plot discovery.

        Single-run metrics live as TOP-LEVEL fields in the export JSON (one per
        metric tag), and ``get_available_metrics`` iterates the top level while
        skipping the nested ``metrics`` key. Mirroring keeps aggregate cells
        discoverable the same way. Reserved keys already present are not clobbered.
        """
        for metric_name, parsed in unflattened.items():
            if metric_name in aggregated:
                continue
            if hasattr(parsed, "model_dump"):
                aggregated[metric_name] = parsed.model_dump(
                    mode="json", exclude_none=True
                )
            elif isinstance(parsed, JsonMetricResult):
                aggregated[metric_name] = {
                    key: value
                    for key, value in asdict(parsed).items()
                    if value is not None
                }
            elif isinstance(parsed, dict):
                aggregated[metric_name] = parsed

    def _plumb_variation_values_into_input_config(
        self, aggregated: dict[str, Any], raw: dict[str, Any]
    ) -> None:
        """Surface swept ``concurrency`` from the aggregate metadata into
        ``input_config.loadgen`` so per-cell concurrency labels resolve.

        The aggregate file carries ``variation_values`` in its metadata block but
        no ``input_config``; without this plumb ``RunMetadata.concurrency`` is
        None. Only leaf ``concurrency`` dims are handled.
        """
        meta = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
        variation_values = (
            meta.get("variation_values")
            if isinstance(meta.get("variation_values"), dict)
            else {}
        )
        if not variation_values:
            return
        for key, value in variation_values.items():
            leaf = key.rsplit(".", 1)[-1]
            if leaf == "concurrency" and isinstance(value, int) and value >= 1:
                # The aggregate JSON may carry ``input_config: null``; coalesce to
                # a fresh dict before plumbing rather than crashing on None.
                input_config = aggregated.get("input_config")
                if not isinstance(input_config, dict):
                    input_config = {}
                    aggregated["input_config"] = input_config
                loadgen = input_config.get("loadgen")
                if not isinstance(loadgen, dict):
                    loadgen = {}
                    input_config["loadgen"] = loadgen
                loadgen["concurrency"] = value
                return

    def _unflatten_confidence_metrics(
        self, flat: dict[str, Any]
    ) -> dict[str, JsonMetricResult | dict[str, Any]]:
        """Reverse the ``f"{metric_name}_{stat_key}"`` flattening into one
        ``JsonMetricResult`` per metric.

        Confidence aggregate JSON stores one entry per ``(metric, stat)`` pair
        (e.g. ``request_latency_p99``) carrying ``{mean, std, ..., unit}``. We map
        ``payload["mean"]`` onto the matching stat slot and drop CI/cv/se fields
        (``JsonMetricResult`` has no place for them). Keys whose suffix is not a
        known stat are bucketed under ``avg``.
        """
        nested: dict[str, dict[str, Any]] = {}
        for flat_key, payload in flat.items():
            if not isinstance(payload, dict):
                continue
            head, _, tail = flat_key.rpartition("_")
            if tail in _KNOWN_STAT_SUFFIXES and head:
                metric_name, stat_key = head, tail
            else:
                metric_name, stat_key = flat_key, "avg"

            bucket = nested.setdefault(metric_name, {"unit": ""})
            unit = payload.get("unit")
            if unit and not bucket["unit"]:
                bucket["unit"] = unit
            mean_value = payload.get("mean")
            if mean_value is not None:
                bucket[stat_key] = mean_value

        parsed: dict[str, JsonMetricResult | dict[str, Any]] = {}
        for name, fields in nested.items():
            try:
                parsed[name] = JsonMetricResult(**fields)
            except (ValidationError, TypeError, ValueError) as e:
                self.warning(
                    f"Failed to parse aggregate metric {name} as JsonMetricResult: {e}"
                )
                parsed[name] = fields
        return parsed

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

    @classmethod
    def _extract_input_config_fields(cls, aggregated: dict[str, Any]) -> dict[str, Any]:
        """Pull model/concurrency/request_count/endpoint_type from aggregated data.

        Handles two artifact shapes:
        - YAML v2: model at ``input_config.models.items[].name`` and
          concurrency/request_count on the profiling phase in
          ``input_config.phases[]``.
        - Legacy: model at ``input_config.endpoint.model_names[]`` and
          concurrency/request_count on ``input_config.loadgen``.

        v2 shapes win when both are present. Aggregate-only runs (no
        input_config) fall back to ``aggregated["metadata"]["model"]``.
        """
        fields: dict[str, Any] = {
            "model": None,
            "concurrency": None,
            "request_count": None,
            "endpoint_type": None,
        }
        if not aggregated:
            return fields

        config = aggregated.get("input_config")
        config = config if isinstance(config, dict) else {}
        endpoint = config.get("endpoint") or {}
        loadgen = config.get("loadgen") or {}
        phases = config.get("phases") or []

        fields["model"] = cls._resolve_model_name(endpoint, config, aggregated)

        concurrency, request_count = cls._resolve_load_fields(phases, loadgen)
        fields["concurrency"] = concurrency
        fields["request_count"] = request_count

        if "type" in endpoint:
            fields["endpoint_type"] = endpoint["type"]

        return fields

    @staticmethod
    def _resolve_model_name(
        endpoint: dict[str, Any],
        config: dict[str, Any],
        aggregated: dict[str, Any],
    ) -> str | None:
        """Resolve the model name across v2, legacy, and aggregate-metadata shapes."""
        models = config.get("models") if isinstance(config.get("models"), dict) else {}
        items = models.get("items") or []
        if items and isinstance(items[0], dict) and items[0].get("name"):
            return items[0]["name"]

        legacy_names = endpoint.get("model_names")
        if legacy_names:
            return legacy_names[0]

        metadata = aggregated.get("metadata")
        if isinstance(metadata, dict) and metadata.get("model"):
            return metadata["model"]

        return None

    @staticmethod
    def _resolve_load_fields(
        phases: list[Any],
        loadgen: dict[str, Any],
    ) -> tuple[Any, Any]:
        """Resolve (concurrency, request_count), preferring the v2 profiling phase."""
        valid_phases = [p for p in phases if isinstance(p, dict)]
        if valid_phases:
            phase = next(
                (p for p in valid_phases if p.get("name") == "profiling"),
                valid_phases[0],
            )
            return phase.get("concurrency"), phase.get("requests")

        return loadgen.get("concurrency"), loadgen.get("request_count")

    @staticmethod
    def _extract_run_timing_fields(aggregated: dict[str, Any]) -> dict[str, Any]:
        """Pull start_time/end_time/was_cancelled off the aggregated top level."""
        if not aggregated:
            return {"start_time": None, "end_time": None, "was_cancelled": False}
        return {
            "start_time": aggregated.get("start_time"),
            "end_time": aggregated.get("end_time"),
            # Coalesce explicit null (present in confidence-aggregate JSON) to
            # False; RunMetadata.was_cancelled is a non-nullable bool.
            "was_cancelled": bool(aggregated.get("was_cancelled") or False),
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
