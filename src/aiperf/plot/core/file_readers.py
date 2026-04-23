# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""File-format readers for the plot ``DataLoader``.

Split out of ``data_loader.py`` so it stays within the ergonomics file-size
limit. Exposed as a mixin because every reader shares the same zst-aware
byte/text helpers and logger.
"""

import io
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import msgspec
import numpy as np
import orjson
import pandas as pd

from aiperf.common.models.record_models import (
    MetricRecordInfo,
    MetricResult,
    decode_metric_record_info_json,
)
from aiperf.plot.exceptions import DataLoadError


class FileReaderMixin:
    """Mixin providing JSONL/JSON/CSV readers, shared by DataLoader."""

    @staticmethod
    def _resolve_file(directory: Path, name: str) -> Path | None:
        """Resolve a file, checking raw then .zst variant."""
        raw = directory / name
        if raw.exists():
            return raw
        zst = directory / f"{name}.zst"
        if zst.exists():
            return zst
        return None

    @staticmethod
    def _read_bytes(path: Path) -> bytes:
        """Read file bytes, decompressing if .zst.

        Uses stream_reader to handle files compressed without content size
        in the frame header (e.g. streaming compression).
        """
        raw = path.read_bytes()
        if path.suffix == ".zst":
            import zstandard

            dctx = zstandard.ZstdDecompressor()
            return dctx.stream_reader(raw).read()
        return raw

    @staticmethod
    @contextmanager
    def _open_text(path: Path) -> Generator[io.StringIO | io.TextIOWrapper, None, None]:
        """Open a file as a text stream, decompressing .zst transparently.

        Uses stream_reader to handle files compressed without content size
        in the frame header (e.g. streaming compression).
        """
        if path.suffix == ".zst":
            import zstandard

            dctx = zstandard.ZstdDecompressor()
            data = dctx.stream_reader(path.read_bytes()).read()
            yield io.StringIO(data.decode("utf-8"))
        else:
            with open(path, encoding="utf-8") as f:
                yield f

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
            with self._open_text(jsonl_path) as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        record = parse_func(line)
                        records.append(record)
                    except (
                        orjson.JSONDecodeError,
                        ValueError,
                        TypeError,
                        KeyError,
                    ) as e:
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
                self.warning(
                    f"No valid records found in {file_description} file: {jsonl_path}"
                )
                return None

            return records

        except OSError as e:
            raise DataLoadError(
                f"Failed to read {file_description} file: {e}", path=str(jsonl_path)
            ) from e

    def _load_jsonl(self, jsonl_path: Path) -> pd.DataFrame:
        """Load per-request data from JSONL file into a flattened DataFrame."""
        if not jsonl_path.exists():
            raise DataLoadError("JSONL file not found", path=str(jsonl_path))

        def parse_line(line: str) -> dict:
            metric_record = decode_metric_record_info_json(line)
            return self._convert_to_flat_dict(metric_record)

        records = self._read_jsonl_with_error_handling(
            jsonl_path, parse_line, raise_on_empty=True, file_description="JSONL"
        )

        df = pd.DataFrame(records)
        self.info(f"Loaded {len(df)} records from {jsonl_path}")
        return df

    @staticmethod
    def _compute_inter_chunk_latency_stats(values: list[float]) -> dict[str, float]:
        """Compute per-request statistics from an inter_chunk_latency array.

        Useful for analyzing stream health, jitter, and stability per request.
        Returns empty dict if ``values`` is empty.
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
        """Flatten a MetricRecordInfo artifact record into a DataFrame-row dict."""
        flat: dict[str, Any] = {}

        flat.update(record.metadata.model_dump())

        for key, metric_value in record.metrics.items():
            if key == "inter_chunk_latency" and isinstance(metric_value.value, list):
                stats = self._compute_inter_chunk_latency_stats(metric_value.value)
                flat.update(stats)
                continue
            flat[key] = metric_value.value

        if record.error:
            flat["error"] = msgspec.to_builtins(record.error)

        return flat

    def _load_aggregated_json(self, json_path: Path) -> dict[str, Any]:
        """Load aggregated statistics JSON, parsing metrics into MetricResult objects."""
        try:
            data = orjson.loads(self._read_bytes(json_path))

            if "metrics" in data and isinstance(data["metrics"], dict):
                parsed_metrics: dict[str, Any] = {}
                for tag, metric_data in data["metrics"].items():
                    try:
                        parsed_metrics[tag] = MetricResult(**metric_data)
                    except (ValueError, TypeError, KeyError) as e:
                        self.warning(f"Failed to parse metric {tag}: {e}")
                        parsed_metrics[tag] = metric_data
                data["metrics"] = parsed_metrics

            self.info(f"Loaded aggregated data from {json_path}")
            return data
        except orjson.JSONDecodeError as e:
            raise DataLoadError(
                f"Failed to parse JSON file: {e}", path=str(json_path)
            ) from e
        except FileNotFoundError as e:
            raise DataLoadError(
                "Required JSON file not found", path=str(json_path)
            ) from e
        except OSError as e:
            raise DataLoadError(
                f"Failed to read JSON file: {e}", path=str(json_path)
            ) from e

    def _load_timeslices_csv(self, csv_path: Path) -> pd.DataFrame:
        """Load timeslice data from CSV into tidy/long DataFrame."""
        try:
            df = pd.read_csv(io.BytesIO(self._read_bytes(csv_path)))

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
        """Load GPU telemetry JSONL into a flat DataFrame (relative-time seconds)."""
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
