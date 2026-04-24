# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared JSONL loader for reading per-request profiling records."""

import logging
from collections.abc import Generator
from pathlib import Path

import orjson

from aiperf.common.config.config_defaults import OutputDefaults

logger = logging.getLogger(__name__)

DEFAULT_JSONL_FILENAME = str(OutputDefaults.PROFILE_EXPORT_JSONL_FILE)


def iter_profiling_records(
    artifacts_path: Path,
    jsonl_filename: str = DEFAULT_JSONL_FILENAME,
) -> Generator[dict, None, None]:
    """Yield validated profiling-phase metric records from a run's JSONL export.

    Filters to profiling phase, skips error records, malformed lines, and
    records with missing/invalid structure. Each yielded dict is the
    ``"metrics"`` mapping from a valid record.

    Args:
        artifacts_path: Path to the run's artifacts directory.
        jsonl_filename: JSONL filename within the artifacts directory.

    Yields:
        The ``metrics`` dict from each valid profiling-phase record.
    """
    jsonl_path = artifacts_path / jsonl_filename
    if not jsonl_path.exists():
        return

    try:
        with open(jsonl_path, "rb") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = orjson.loads(line)
                except orjson.JSONDecodeError:
                    logger.warning("Skipping malformed JSONL line in %s", jsonl_path)
                    continue

                if not isinstance(record, dict):
                    logger.warning("Skipping non-dict JSONL record in %s", jsonl_path)
                    continue

                metadata = record.get("metadata", {})
                if not isinstance(metadata, dict):
                    continue
                if metadata.get("benchmark_phase") != "profiling":
                    continue
                if record.get("error") is not None:
                    continue

                metrics = record.get("metrics", {})
                if not isinstance(metrics, dict):
                    continue

                yield metrics
    except OSError:
        logger.exception("I/O error reading %s", jsonl_path)


def load_single_metric(
    artifacts_path: Path,
    metric_name: str,
    jsonl_filename: str = DEFAULT_JSONL_FILENAME,
) -> list[float]:
    """Extract values for a single metric from a run's JSONL export.

    Args:
        artifacts_path: Path to the run's artifacts directory.
        metric_name: Name of the metric to extract (e.g. "time_to_first_token").
        jsonl_filename: JSONL filename within the artifacts directory.

    Returns:
        List of float metric values. Empty if file is missing or has no matches.
    """
    values: list[float] = []
    jsonl_path = artifacts_path / jsonl_filename
    for metrics in iter_profiling_records(artifacts_path, jsonl_filename):
        metric_entry = metrics.get(metric_name)
        if metric_entry is None or not isinstance(metric_entry, dict):
            continue
        value = metric_entry.get("value")
        if value is None:
            continue
        try:
            values.append(float(value))
        except (ValueError, TypeError):
            logger.warning(
                "Skipping non-numeric value for %s in %s",
                metric_name,
                jsonl_path,
            )
    return values


def load_all_metrics(
    artifacts_path: Path,
    jsonl_filename: str = DEFAULT_JSONL_FILENAME,
) -> dict[str, list[float]]:
    """Extract all metric values from a run's JSONL export.

    Args:
        artifacts_path: Path to the run's artifacts directory.
        jsonl_filename: JSONL filename within the artifacts directory.

    Returns:
        Dict mapping metric name to list of float values.
        Empty dict if the file is missing, empty, or unreadable.
    """
    result: dict[str, list[float]] = {}
    jsonl_path = artifacts_path / jsonl_filename
    for metrics in iter_profiling_records(artifacts_path, jsonl_filename):
        for metric_name, metric_entry in metrics.items():
            value = (
                metric_entry.get("value") if isinstance(metric_entry, dict) else None
            )
            if value is None:
                continue
            try:
                float_value = float(value)
            except (ValueError, TypeError):
                logger.warning(
                    "Skipping non-numeric metric value for %s in %s",
                    metric_name,
                    jsonl_path,
                )
                continue
            if metric_name not in result:
                result[metric_name] = []
            result[metric_name].append(float_value)
    return result
