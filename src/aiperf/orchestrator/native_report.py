# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Projection of a Rust native-v2 report into outer-orchestrator metrics.

The Rust runner owns metric accumulation and writes the authoritative native-v2
report.  Python's multi-run/search layer consumes only the compact
``JsonMetricResult`` projection produced here; it never recomputes timing facts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import orjson

from aiperf.common.models.export_models import JsonMetricResult

NATIVE_REPORT_SCHEMA_VERSION = "2.0"

_PERCENTILE_FIELDS = frozenset(
    {"p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99"}
)


class NativeReportError(ValueError):
    """Raised when a Rust report cannot satisfy the orchestration contract."""


def load_native_summary(path: Path) -> dict[str, JsonMetricResult]:
    """Load one authoritative Rust report and project its flat run summary."""
    try:
        payload = orjson.loads(path.read_bytes())
    except OSError as error:
        raise NativeReportError(
            f"failed to read native report {path}: {error}"
        ) from error
    except orjson.JSONDecodeError as error:
        raise NativeReportError(
            f"native report {path} is invalid JSON: {error}"
        ) from error
    return project_native_summary(payload)


def project_native_summary(payload: Any) -> dict[str, JsonMetricResult]:
    """Project native-v2 metric entries without changing their numeric values.

    The outer orchestrator has a deliberately flat metric vocabulary. Native
    inference metrics contain one unlabeled series, while telemetry may contain
    several labeled series. A labeled metric enters the flat projection only
    when it has exactly one series or an explicit unlabeled aggregate series.
    The complete labeled data remains in the native report.
    """
    root = _mapping(payload, "native report")
    schema_version = root.get("schema_version")
    if schema_version != NATIVE_REPORT_SCHEMA_VERSION:
        raise NativeReportError(
            "unsupported native report schema_version "
            f"{schema_version!r}; expected {NATIVE_REPORT_SCHEMA_VERSION!r}"
        )
    metrics = _mapping(root.get("metrics"), "native report metrics")
    projected: dict[str, JsonMetricResult] = {}
    for name, authored_entry in metrics.items():
        if not isinstance(name, str) or not name:
            raise NativeReportError(
                "native report metric names must be non-empty strings"
            )
        entry = _mapping(authored_entry, f"metric {name!r}")
        series = _summary_series(entry, name)
        if series is None:
            continue
        stats = _mapping(series.get("stats"), f"metric {name!r} stats")
        metric_type = entry.get("type")
        unit = entry.get("unit")
        if not isinstance(unit, str):
            raise NativeReportError(f"metric {name!r} unit must be a string")
        values = _legacy_stats(metric_type, stats, name)
        projected[name] = JsonMetricResult(unit=unit, **values)
    _project_accuracy(root.get("accuracy"), projected)
    return projected


def _project_accuracy(authored: Any, projected: dict[str, JsonMetricResult]) -> None:
    """Expose native accuracy analysis to sweep/search metric consumers."""
    if authored is None:
        return
    accuracy = _mapping(authored, "native report accuracy")
    summary = _mapping(accuracy.get("summary"), "native report accuracy summary")
    overall = _mapping(summary.get("overall"), "accuracy overall")
    projected["accuracy.overall"] = _accuracy_rollup(overall, "accuracy overall")

    per_task = _mapping(summary.get("per_task"), "accuracy per_task")
    for task, value in per_task.items():
        if not isinstance(task, str) or not task:
            raise NativeReportError("accuracy task names must be non-empty strings")
        rollup = _mapping(value, f"accuracy task {task!r}")
        projected[f"accuracy.task.{task}"] = _accuracy_rollup(
            rollup, f"accuracy task {task!r}"
        )
        projected[f"accuracy.unparsed.task.{task}"] = _unparsed_rollup(
            rollup, f"accuracy task {task!r}"
        )
    projected["accuracy.unparsed"] = _unparsed_rollup(overall, "accuracy overall")

    at_load = accuracy.get("accuracy_at_load")
    if at_load is not None:
        joined = _mapping(at_load, "accuracy_at_load")
        value = joined.get("correct_answers_per_second")
        if value is not None:
            number = _required_number(
                value, "accuracy.correct_answers_per_second", "value"
            )
            projected["accuracy.correct_answers_per_second"] = JsonMetricResult(
                unit="answers/sec", avg=number, min=number, max=number
            )
    value = accuracy.get("correct_answers_per_kwh")
    if value is not None:
        number = _required_number(value, "accuracy.correct_answers_per_kwh", "value")
        projected["accuracy.correct_answers_per_kwh"] = JsonMetricResult(
            unit="answers/kWh", avg=number, min=number, max=number
        )


def _accuracy_rollup(rollup: dict[str, Any], label: str) -> JsonMetricResult:
    count = _required_int(rollup.get("n"), label, "n")
    correct = _required_int(rollup.get("correct_count"), label, "correct_count")
    value = _optional_number(rollup.get("accuracy"), label, "accuracy")
    return JsonMetricResult(
        unit="ratio",
        count=count,
        sum=correct,
        avg=value,
        min=value,
        max=value,
    )


def _unparsed_rollup(rollup: dict[str, Any], label: str) -> JsonMetricResult:
    count = _required_int(rollup.get("n"), label, "n")
    unparsed = _required_int(rollup.get("unparsed_count"), label, "unparsed_count")
    value = _optional_number(rollup.get("unparsed_rate"), label, "unparsed_rate")
    return JsonMetricResult(
        unit="ratio",
        count=count,
        sum=unparsed,
        avg=value,
        min=value,
        max=value,
    )


def _summary_series(entry: dict[str, Any], name: str) -> dict[str, Any] | None:
    authored = entry.get("series")
    if not isinstance(authored, list) or not authored:
        raise NativeReportError(f"metric {name!r} must contain at least one series")
    series = [
        _mapping(value, f"metric {name!r} series[{index}]")
        for index, value in enumerate(authored)
    ]
    if len(series) == 1:
        return series[0]
    aggregate = [value for value in series if value.get("labels") is None]
    if len(aggregate) > 1:
        raise NativeReportError(
            f"metric {name!r} contains multiple unlabeled aggregate series"
        )
    return aggregate[0] if aggregate else None


def _legacy_stats(
    metric_type: Any, stats: dict[str, Any], name: str
) -> dict[str, int | float | None]:
    if metric_type == "distribution":
        values: dict[str, int | float | None] = {
            "count": _optional_int(stats.get("count"), name, "count"),
            "avg": _optional_number(stats.get("avg"), name, "avg"),
            "min": _optional_number(stats.get("min"), name, "min"),
            "max": _optional_number(stats.get("max"), name, "max"),
            "std": _optional_number(stats.get("std"), name, "std"),
        }
        values.update(_percentiles(stats.get("percentiles"), name))
        return values
    if metric_type == "scalar":
        value = _required_number(stats.get("value"), name, "value")
        return {"avg": value, "min": value, "max": value}
    if metric_type == "counter":
        total = _required_number(stats.get("total"), name, "total")
        return {"avg": total, "min": total, "max": total, "sum": total}
    if metric_type == "histogram":
        count = _required_int(stats.get("count"), name, "count")
        total = _required_number(stats.get("sum"), name, "sum")
        values = {
            "count": count,
            "sum": total,
            "avg": _optional_number(stats.get("avg"), name, "avg"),
        }
        values.update(_percentiles(stats.get("percentiles"), name))
        return values
    raise NativeReportError(
        f"metric {name!r} has unsupported native metric type {metric_type!r}"
    )


def _percentiles(authored: Any, name: str) -> dict[str, float | None]:
    percentiles = _mapping(authored, f"metric {name!r} percentiles")
    projected: dict[str, float | None] = {}
    for key, value in percentiles.items():
        if key not in _PERCENTILE_FIELDS:
            raise NativeReportError(
                f"metric {name!r} percentile {key!r} is not representable by "
                "the outer-orchestrator metric contract"
            )
        projected[key] = _optional_number(value, name, key)
    return projected


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise NativeReportError(f"{label} must be a JSON object")
    return value


def _optional_number(value: Any, name: str, field: str) -> float | None:
    if value is None:
        return None
    return _required_number(value, name, field)


def _required_number(value: Any, name: str, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise NativeReportError(f"metric {name!r} {field} must be numeric or null")
    return float(value)


def _optional_int(value: Any, name: str, field: str) -> int | None:
    if value is None:
        return None
    return _required_int(value, name, field)


def _required_int(value: Any, name: str, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise NativeReportError(
            f"metric {name!r} {field} must be a non-negative integer"
        )
    return value
