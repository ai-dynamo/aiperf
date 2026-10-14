# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Detect whether a trace's first record carries timestamps (fixed-schedule auto-promotion)."""

from __future__ import annotations

import gzip
import zlib
from pathlib import Path

from aiperf.common.aiperf_logger import AIPerfLogger

_logger = AIPerfLogger(__name__)


def _columnar_file_has_timestamp(path: Path) -> bool | None:
    """Probe Arrow IPC files, or return None for another format."""
    if path.suffix.lower() in {".arrow", ".ipc"}:
        from aiperf.dataset.loader.baseten_trace import BasetenTraceDatasetLoader

        return BasetenTraceDatasetLoader.can_load(filename=path)
    return None


def _parquet_has_timestamp(path: Path) -> bool:
    """Baseten's time column, else the first row, read like a JSONL first line."""
    from aiperf.common.record_files import first_record, parquet_files

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        return False
    try:
        if "timestamp_start_unix_ms" in set(
            pq.read_schema(parquet_files(path)[0]).names
        ):
            return True
        record = first_record(path)
    except (OSError, ValueError, pa.ArrowException):
        return False
    return isinstance(record, dict) and record_has_timestamp(record)


def _has_timing_events_timestamp(data: dict) -> bool:
    events = data.get("timing_events")
    return bool(
        events
        and isinstance(events, list)
        and isinstance(events[0], dict)
        and events[0].get("timestamp") is not None
    )


def record_has_timestamp(data: dict) -> bool:
    """Whether one parsed trace record carries timestamp data."""
    return data.get("timestamp") is not None or _has_timing_events_timestamp(data)


def first_record_has_timestamp(file_path: object) -> bool:
    """Return True when a trace file carries timestamp data."""
    from aiperf.common.record_files import is_parquet_input
    from aiperf.common.utils import load_json_str, open_text_maybe_gzip

    path = Path(file_path)
    if is_parquet_input(path):
        return _parquet_has_timestamp(path)
    if not path.is_file():
        return False
    if (columnar_result := _columnar_file_has_timestamp(path)) is not None:
        return columnar_result
    try:
        with open_text_maybe_gzip(path) as f:
            for line in f:
                if not (stripped := line.strip()):
                    continue
                try:
                    data = load_json_str(stripped)
                except (ValueError, TypeError):
                    return False
                if not isinstance(data, dict):
                    return False
                return record_has_timestamp(data)
    except (EOFError, gzip.BadGzipFile, zlib.error) as e:
        _logger.warning(f"Truncated or corrupt gzip in '{file_path}': {e}")
        return False
    except (OSError, UnicodeDecodeError):
        return False
    return False
