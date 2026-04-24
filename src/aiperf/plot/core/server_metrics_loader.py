# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Server metrics loaders (JSON and Parquet) for the plot DataLoader.

Isolated here so ``data_loader.py`` can stay within ergonomics limits; kept as
a mixin so the subclass can share `_read_bytes`, `downsampling_config`, and the
`AIPerfLoggerMixin` logging helpers.
"""

import io
from pathlib import Path
from typing import Any

import orjson
import pandas as pd

from aiperf.common.enums import PrometheusMetricType
from aiperf.common.models.server_metrics_models import ServerMetricsExportData
from aiperf.plot.exceptions import DataLoadError


class ServerMetricsLoaderMixin:
    """Mixin providing server metrics Parquet/JSON loading on a DataLoader.

    The host class must provide:
      - ``_read_bytes(path)`` static helper
      - ``downsampling_config: dict``
      - the ``AIPerfLoggerMixin`` logging methods (``info``/``debug``/``warning``)
    """

    downsampling_config: dict

    # ----- JSON loader -----

    def _load_server_metrics_json(
        self, json_path: Path
    ) -> tuple[pd.DataFrame | None, dict[str, Any]]:
        """Load server metrics from JSON export file (preferred format).

        Parses ServerMetricsExportData structure and extracts both time series
        data (from timeslices) and aggregated statistics. Handles all metric types
        (GAUGE, COUNTER, HISTOGRAM) and multi-endpoint configurations.
        """
        try:
            data = orjson.loads(self._read_bytes(json_path))
            export_data = ServerMetricsExportData.model_validate(data)

            aggregated, rows = self._collect_server_metrics_json(export_data)
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
        except Exception as e:  # noqa: BLE001 - ServerMetricsExportData.model_validate + orjson across many shapes; wrap once into DataLoadError
            raise DataLoadError(
                f"Failed to load server metrics from JSON: {e}", path=str(json_path)
            ) from e

    def _collect_server_metrics_json(
        self, export_data: ServerMetricsExportData
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Walk ``export_data`` and build the aggregated dict + tidy rows."""
        aggregated: dict[str, Any] = {}
        rows: list[dict[str, Any]] = []

        for metric_name, metric_data in export_data.metrics.items():
            aggregated[metric_name] = {}

            for series in metric_data.series:
                endpoint_url = series.endpoint_url or "unknown"
                labels_key = (
                    orjson.dumps(series.labels, option=orjson.OPT_SORT_KEYS).decode()
                    if series.labels
                    else "{}"
                )

                aggregated[metric_name].setdefault(endpoint_url, {})

                stats_value = series.stats if series.stats is not None else series.value
                aggregated[metric_name][endpoint_url][labels_key] = {
                    "type": metric_data.type.value,
                    "stats": stats_value,
                    "unit": metric_data.unit,
                    "description": metric_data.description,
                    "timeslices": series.timeslices,
                }

                if series.timeslices:
                    rows.extend(
                        self._timeslice_to_row(
                            ts,
                            endpoint_url=endpoint_url,
                            metric_name=metric_name,
                            metric_type=metric_data.type,
                            labels_key=labels_key,
                            unit=metric_data.unit or "",
                        )
                        for ts in series.timeslices
                    )

        return aggregated, rows

    @staticmethod
    def _timeslice_to_row(
        ts: Any,
        *,
        endpoint_url: str,
        metric_name: str,
        metric_type: PrometheusMetricType,
        labels_key: str,
        unit: str,
    ) -> dict[str, Any]:
        """Convert a ServerTimeslice into a tidy DataFrame row dict.

        Dispatches on metric_type because GaugeTimeslice/CounterTimeslice/HistogramTimeslice
        now alias a single unified ``ServerTimeslice`` dataclass.
        """
        timestamp_ns = (ts.start_ns + ts.end_ns) // 2
        row: dict[str, Any] = {
            "timestamp_ns": timestamp_ns,
            "endpoint_url": endpoint_url,
            "metric_name": metric_name,
            "metric_type": metric_type.value,
            "labels_json": labels_key,
            "unit": unit,
            "histogram_count": None,
            "histogram_sum": None,
        }
        if metric_type == PrometheusMetricType.GAUGE:
            row["value"] = ts.avg
        elif metric_type == PrometheusMetricType.COUNTER:
            row["value"] = ts.rate
        elif metric_type == PrometheusMetricType.HISTOGRAM:
            row["value"] = ts.avg
            row["histogram_count"] = ts.count
            row["histogram_sum"] = ts.sum
        return row

    # ----- Parquet loader -----

    def _load_server_metrics_parquet(
        self, parquet_path: Path
    ) -> tuple[pd.DataFrame | None, dict[str, Any]]:
        """Load server metrics from Parquet export file (fast binary format)."""
        try:
            import pyarrow.parquet as pq

            table = pq.read_table(io.BytesIO(self._read_bytes(parquet_path)))
            df_wide = table.to_pandas()

            label_columns = self._parquet_label_columns(df_wide)
            df_filtered = self._parquet_filter_histogram_aggregates(df_wide)
            df_filtered["labels_json"] = self._parquet_build_labels_json(
                df_filtered, label_columns
            )

            rows = self._parquet_rows_with_deltas(df_filtered)
            df_tidy = pd.DataFrame(rows) if rows else None
            df_tidy = self._maybe_downsample_parquet(df_tidy)

            aggregated: dict[str, Any] = {}
            unique_metrics = len(df_wide["metric_name"].unique())
            self.info(
                f"Loaded server metrics from Parquet: {unique_metrics} metrics, "
                f"{len(rows)} raw points → {len(df_tidy) if df_tidy is not None else 0} "
                f"windowed points (5s aggregation)"
            )

            return df_tidy, aggregated

        except ImportError as e:
            raise DataLoadError(
                "pyarrow is required to load Parquet files. Install with: uv add pyarrow",
                path=str(parquet_path),
            ) from e
        except Exception as e:
            raise DataLoadError(
                f"Failed to load server metrics from Parquet: {e}",
                path=str(parquet_path),
            ) from e

    @staticmethod
    def _parquet_label_columns(df_wide: pd.DataFrame) -> list[str]:
        """Return label columns (columns that aren't part of the core metric schema)."""
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
        return [c for c in df_wide.columns if c not in core_columns]

    def _parquet_filter_histogram_aggregates(
        self, df_wide: pd.DataFrame
    ) -> pd.DataFrame:
        """Keep only +Inf bucket rows for histograms; keep everything else."""
        is_histogram = df_wide["metric_type"] == "histogram"
        is_aggregate_bucket = df_wide["bucket_le"] == "+Inf"
        df_filtered = df_wide[~is_histogram | is_aggregate_bucket].copy()

        self.debug(
            f"Filtered Parquet: {len(df_wide)} rows → {len(df_filtered)} rows "
            f"(removed per-bucket histogram rows)"
        )
        return df_filtered

    @staticmethod
    def _parquet_build_labels_json(
        df_filtered: pd.DataFrame, label_columns: list[str]
    ) -> pd.Series:
        """Build canonical labels_json string per row from label columns."""
        return df_filtered.apply(
            lambda row: orjson.dumps(
                {k: row[k] for k in label_columns if pd.notna(row[k]) and row[k] != ""},
                option=orjson.OPT_SORT_KEYS,
            ).decode()
            if any(pd.notna(row[k]) and row[k] != "" for k in label_columns)
            else "{}",
            axis=1,
        )

    @staticmethod
    def _parquet_rows_with_deltas(df_filtered: pd.DataFrame) -> list[dict[str, Any]]:
        """Group by series and emit tidy rows, computing deltas for cumulative types."""
        rows: list[dict[str, Any]] = []
        grouped = df_filtered.groupby(
            ["metric_name", "endpoint_url", "labels_json", "metric_type"]
        )

        for (metric_name, endpoint_url, labels_json, metric_type), group in grouped:
            group = group.sort_values("timestamp_ns")

            if metric_type in ["counter", "histogram"]:
                group["delta_count"] = group["count"].diff()
                group["delta_sum"] = group["sum"].diff()
                # Skip first row — no previous sample to diff against.
                group = group[1:]

                if metric_type == "counter":
                    computed_values = group["delta_count"]
                else:
                    computed_values = (group["delta_sum"] / group["delta_count"]).where(
                        group["delta_count"] > 0, 0
                    )
            else:
                computed_values = group["value"]

            for idx, computed_value in zip(group.index, computed_values, strict=False):
                row_wide = group.loc[idx]
                rows.append(
                    {
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
                )
        return rows

    def _maybe_downsample_parquet(
        self, df_tidy: pd.DataFrame | None
    ) -> pd.DataFrame | None:
        """Apply configured downsampling if enabled and data is present."""
        if df_tidy is None or df_tidy.empty:
            return df_tidy
        if not self.downsampling_config["enabled"]:
            self.info("Server metrics downsampling disabled by configuration")
            return df_tidy

        window_size_seconds = self.downsampling_config["window_size_seconds"]
        aggregation_method = self.downsampling_config["aggregation_method"]
        return self._downsample_server_metrics_to_windows(
            df_tidy,
            window_size_ns=int(window_size_seconds * 1e9),
            aggregation_method=aggregation_method,
        )

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
        """
        if df.empty:
            return df

        valid_methods = ["mean", "max", "min", "median"]
        if aggregation_method not in valid_methods:
            self.warning(
                f"Invalid aggregation method '{aggregation_method}', using 'mean'. "
                f"Valid options: {valid_methods}"
            )
            aggregation_method = "mean"

        min_ts = df["timestamp_ns"].min()
        df["window"] = ((df["timestamp_ns"] - min_ts) // window_size_ns).astype(int)

        agg_funcs = {
            "timestamp_ns": "mean",
            "value": aggregation_method,
            "histogram_count": "sum",
            "histogram_sum": "sum",
            "metric_type": "first",
            "unit": "first",
        }

        grouped = df.groupby(
            ["metric_name", "endpoint_url", "labels_json", "window"],
            dropna=False,
        ).agg(agg_funcs)

        df_downsampled = grouped.reset_index(drop=False)
        df_downsampled = df_downsampled.drop(columns=["window"])
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
        """
        if df is None or df.empty:
            return {}

        aggregated: dict[str, Any] = {}
        grouped = df.groupby(["metric_name", "endpoint_url", "labels_json"])

        for (metric_name, endpoint_url, labels_json), group in grouped:
            aggregated.setdefault(metric_name, {}).setdefault(endpoint_url, {})

            values = group["value"].dropna()
            if len(values) == 0:
                continue

            aggregated[metric_name][endpoint_url][labels_json] = (
                self._summarise_timeseries_group(group, values)
            )

        self.info(
            f"Computed aggregated stats for {len(aggregated)} metrics from time-series data"
        )
        return aggregated

    @staticmethod
    def _summarise_timeseries_group(
        group: pd.DataFrame, values: pd.Series
    ) -> dict[str, Any]:
        """Summarise one (metric, endpoint, labels) group into the aggregated-dict shape."""
        stats = {
            "avg": float(values.mean()),
            "min": float(values.min()),
            "max": float(values.max()),
            "p50": float(values.quantile(0.5)),
            "p95": float(values.quantile(0.95)),
            "p99": float(values.quantile(0.99)),
        }
        metric_type = group["metric_type"].iloc[0] if "metric_type" in group else None
        unit = group["unit"].iloc[0] if "unit" in group else ""
        description = (
            group["description"].iloc[0]
            if "description" in group and pd.notna(group["description"].iloc[0])
            else ""
        )
        return {
            "type": metric_type,
            "stats": stats,
            "unit": unit,
            "description": description,
            "timeslices": None,
        }
