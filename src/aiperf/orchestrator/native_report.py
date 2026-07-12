# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Projection of a Rust native-v2 report into outer-orchestrator metrics.

The Rust runner owns metric accumulation and writes the authoritative native-v2
report.  Python's multi-run/search layer consumes only the compact
``JsonMetricResult`` projection produced here; it never recomputes timing facts.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import orjson

from aiperf.common.models.export_models import JsonMetricResult

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun

NATIVE_REPORT_SCHEMA_VERSION = "2.0"

_PERCENTILE_FIELDS = frozenset(
    {"p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99"}
)


class NativeReportError(ValueError):
    """Raised when a Rust report cannot satisfy the orchestration contract."""


def load_native_summary(path: Path) -> dict[str, JsonMetricResult]:
    """Load one authoritative Rust report and project its flat run summary."""
    return project_native_summary(load_native_report(path))


def load_native_report(path: Path) -> dict[str, Any]:
    """Read one native report without weakening its schema checks."""
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
    root = _mapping(payload, "native report")
    if root.get("schema_version") != NATIVE_REPORT_SCHEMA_VERSION:
        raise NativeReportError(
            "unsupported native report schema_version "
            f"{root.get('schema_version')!r}; expected {NATIVE_REPORT_SCHEMA_VERSION!r}"
        )
    return root


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
    projected = _project_metric_entries(
        root.get("metrics"), label="native report metrics"
    )
    _project_accuracy(root.get("accuracy"), projected)
    return projected


def _project_metric_entries(
    authored: Any, *, label: str
) -> dict[str, JsonMetricResult]:
    metrics = _mapping(authored, label)
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
    return projected


def export_python_compatibility_reports(
    payload: dict[str, Any],
    summary_metrics: dict[str, JsonMetricResult],
    run: BenchmarkRun,
) -> None:
    """Run the canonical Python JSON/CSV generators over native results.

    Rust remains authoritative for request facts and aggregate values. Python
    only serializes those values into the established Config-v2 report files so
    plotters and upload extensions keep working during the native transition.
    """
    from aiperf.common.models import MetricResult, ProfileResults
    from aiperf.common.models.error_models import ErrorDetails, ErrorDetailsCount
    from aiperf.exporters.exporter_config import ExporterConfig
    from aiperf.exporters.metrics_csv_exporter import MetricsCsvExporter
    from aiperf.exporters.metrics_json_exporter import MetricsJsonExporter
    from aiperf.metrics.metric_registry import MetricRegistry

    native_summary = _mapping(payload.get("summary"), "native report summary")
    warmup_authored = payload.get("warmup_metrics")
    warmup_metrics = (
        _project_metric_entries(warmup_authored, label="native warmup metrics")
        if warmup_authored is not None
        else {}
    )

    def metric_result(tag: str, value: JsonMetricResult) -> MetricResult:
        metric_class = MetricRegistry.get_class_or_none(tag)
        return MetricResult(
            tag=tag,
            header=metric_class.header if metric_class is not None else tag,
            **value.model_dump(),
        )

    successful = _counter_value(summary_metrics.get("request_count"))
    failed = _counter_value(summary_metrics.get("error_request_count"))
    errors = []
    for index, authored_error in enumerate(payload.get("errors", [])):
        error = _mapping(authored_error, f"native report errors[{index}]")
        count = _required_int(error.get("count"), "native report error", "count")
        errors.append(
            ErrorDetailsCount(
                error_details=ErrorDetails(
                    code=error.get("code"),
                    type=error.get("type"),
                    message=str(error.get("message", "native request failed")),
                ),
                count=count,
            )
        )
    results = ProfileResults(
        records=[metric_result(tag, value) for tag, value in summary_metrics.items()],
        warmup_records=[
            metric_result(tag, value) for tag, value in warmup_metrics.items()
        ]
        or None,
        completed=successful + failed,
        start_ns=0,
        end_ns=0,
        was_cancelled=bool(native_summary.get("was_cancelled", False)),
        successful_request_count=successful,
        error_request_count=failed,
        error_summary=errors,
    )
    telemetry_results = _project_gpu_telemetry(payload, native_summary)
    server_metrics_results = _project_server_metrics(payload, native_summary, run)
    exporter_config = ExporterConfig(
        results=results,
        cfg=run.cfg,
        telemetry_results=telemetry_results,
        server_metrics_results=server_metrics_results,
        run=run,
    )

    exporters = [MetricsCsvExporter(exporter_config)]
    authored_summary = run.cfg.artifacts.summary
    if authored_summary is not False and "json" in authored_summary:
        exporters.append(MetricsJsonExporter(exporter_config))
    for exporter in exporters:
        destination = exporter.get_export_info().file_path
        _atomic_write_text(destination, exporter.render())

    if server_metrics_results is not None:
        from aiperf.server_metrics.csv_exporter import ServerMetricsCsvExporter
        from aiperf.server_metrics.json_exporter import ServerMetricsJsonExporter

        formats = {str(value) for value in run.cfg.server_metrics.formats}
        server_exporters = []
        if "json" in formats:
            server_exporters.append(ServerMetricsJsonExporter(exporter_config))
        if "csv" in formats:
            server_exporters.append(ServerMetricsCsvExporter(exporter_config))
        for exporter in server_exporters:
            destination = exporter.get_export_info().file_path
            _atomic_write_text(destination, exporter.render())
        if "parquet" in formats:
            _render_server_metrics_parquet(run, server_metrics_results)


def _project_server_metrics(
    payload: dict[str, Any], native_summary: dict[str, Any], run: BenchmarkRun
) -> Any | None:
    """Validate native server series into the canonical Python export models."""
    from aiperf.common.models.server_metrics_models import (
        CounterMetricData,
        CounterSeries,
        CounterStats,
        GaugeMetricData,
        GaugeSeries,
        GaugeStats,
        HistogramMetricData,
        HistogramSeries,
        HistogramStats,
        ServerMetricsEndpointInfo,
        ServerMetricsEndpointSummary,
        ServerMetricsResults,
        TimeRangeFilter,
        UnknownMetricData,
    )
    from aiperf.exporters.utils import normalize_endpoint_display

    authored_metadata = native_summary.get("server_metrics")
    if authored_metadata is None:
        return None
    metadata = _mapping(authored_metadata, "native server metrics metadata")
    profiling_metrics = _mapping(
        payload.get("server_metrics", {}), "native profiling server metrics"
    )
    if not profiling_metrics:
        return None
    descriptions = _mapping(
        metadata.get("descriptions"), "native server metric descriptions"
    )
    metric_types = _mapping(metadata.get("metric_types"), "native server metric types")
    endpoint_info_authored = _mapping(
        metadata.get("endpoint_info"), "native server endpoint info"
    )
    endpoint_info = {
        endpoint: ServerMetricsEndpointInfo.model_validate(info)
        for endpoint, info in endpoint_info_authored.items()
    }

    metric_classes = {
        "gauge": GaugeMetricData,
        "unknown": UnknownMetricData,
        "counter": CounterMetricData,
        "histogram": HistogramMetricData,
    }
    series_classes = {
        "gauge": GaugeSeries,
        "unknown": GaugeSeries,
        "counter": CounterSeries,
        "histogram": HistogramSeries,
    }

    def project_phase(authored: dict[str, Any]) -> dict[str, Any] | None:
        grouped: dict[str, dict[str, list[Any]]] = {}
        metric_meta: dict[str, tuple[str, str, str | None]] = {}
        for name, authored_entry in authored.items():
            if name not in metric_types:
                raise NativeReportError(
                    f"native server metric {name!r} has no Prometheus type metadata"
                )
            metric_type = metric_types[name]
            if metric_type not in metric_classes:
                continue
            entry = _mapping(authored_entry, f"native server metric {name!r}")
            description = descriptions.get(name, "")
            if not isinstance(description, str):
                raise NativeReportError(
                    f"native server metric {name!r} description must be a string"
                )
            unit = entry.get("unit")
            if not isinstance(unit, str):
                raise NativeReportError(
                    f"native server metric {name!r} unit must be a string"
                )
            metric_meta[name] = (metric_type, description, unit or None)
            series_list = entry.get("series")
            if not isinstance(series_list, list):
                raise NativeReportError(
                    f"native server metric {name!r} series must be an array"
                )
            for index, authored_series in enumerate(series_list):
                series = _mapping(
                    authored_series,
                    f"native server metric {name!r} series[{index}]",
                )
                endpoint = series.get("endpoint_url")
                if not isinstance(endpoint, str) or not endpoint:
                    raise NativeReportError(
                        f"native server metric {name!r} series omitted endpoint_url"
                    )
                if endpoint not in endpoint_info:
                    raise NativeReportError(
                        f"native server endpoint {endpoint!r} omitted collection metadata"
                    )
                labels = series.get("labels")
                if labels is not None and not isinstance(labels, dict):
                    raise NativeReportError(
                        f"native server metric {name!r} labels must be an object or null"
                    )
                stats = _mapping(
                    series.get("stats"), f"native server metric {name!r} stats"
                )
                timeslices = _server_metric_timeslices(
                    metric_type, series.get("timeslices", []), name
                )
                common = {"labels": labels, "timeslices": timeslices or None}
                if metric_type in {"gauge", "unknown"}:
                    percentiles = _mapping(
                        stats.get("percentiles", {}),
                        f"native server metric {name!r} percentiles",
                    )
                    gauge_stats = {
                        key: _optional_number(stats.get(key), name, key)
                        for key in ("avg", "min", "max", "std")
                    }
                    gauge_stats.update(
                        {
                            key: _optional_number(percentiles.get(key), name, key)
                            for key in _PERCENTILE_FIELDS
                        }
                    )
                    projected_series = series_classes[metric_type](
                        **common,
                        stats=GaugeStats(**gauge_stats),
                    )
                elif metric_type == "counter":
                    projected_series = CounterSeries(
                        **common,
                        stats=CounterStats(
                            total=_optional_number(stats.get("total"), name, "total"),
                            rate=_optional_number(stats.get("rate"), name, "rate"),
                        ),
                    )
                else:
                    percentiles = _mapping(
                        stats.get("percentiles", {}),
                        f"native server metric {name!r} percentiles",
                    )
                    histogram_stats = {
                        key: _optional_number(stats.get(key), name, key)
                        for key in ("sum", "avg", "count_rate", "sum_rate")
                    }
                    histogram_stats["count"] = _optional_int(
                        stats.get("count"), name, "count"
                    )
                    histogram_stats.update(
                        {
                            f"{key}_estimate": _optional_number(
                                percentiles.get(key), name, key
                            )
                            for key in _PERCENTILE_FIELDS
                        }
                    )
                    buckets = _mapping(
                        stats.get("buckets", {}),
                        f"native server metric {name!r} buckets",
                    )
                    projected_series = HistogramSeries(
                        **common,
                        stats=HistogramStats(**histogram_stats),
                        buckets={
                            key: _required_int(value, name, f"bucket {key}")
                            for key, value in buckets.items()
                        }
                        or None,
                    )
                grouped.setdefault(endpoint, {}).setdefault(name, []).append(
                    projected_series
                )

        summaries = {}
        for endpoint, grouped_metrics in grouped.items():
            metrics = {}
            for name, series in grouped_metrics.items():
                metric_type, description, unit = metric_meta[name]
                metrics[name] = metric_classes[metric_type](
                    description=description,
                    unit=unit,
                    series=series,
                )
            display = normalize_endpoint_display(endpoint)
            if display in summaries:
                raise NativeReportError(
                    f"server metrics endpoints collide after normalization: {display!r}"
                )
            summaries[display] = ServerMetricsEndpointSummary(
                endpoint_url=endpoint,
                info=endpoint_info[endpoint],
                metrics=metrics,
            )
        return summaries or None

    profiling = _mapping(metadata.get("profiling"), "native server profiling boundary")
    start_ns = _required_int(profiling.get("start_ns"), "server metrics", "start_ns")
    end_ns = _required_int(profiling.get("end_ns"), "server metrics", "end_ns")
    warmup_boundary = metadata.get("warmup")
    warmup_start_ns = None
    warmup_end_ns = None
    if warmup_boundary is not None:
        warmup_boundary = _mapping(warmup_boundary, "native server warmup boundary")
        warmup_start_ns = _required_int(
            warmup_boundary.get("start_ns"), "server metrics warmup", "start_ns"
        )
        warmup_end_ns = _required_int(
            warmup_boundary.get("end_ns"), "server metrics warmup", "end_ns"
        )
    warmup_metrics = _mapping(
        payload.get("warmup_server_metrics", {}), "native warmup server metrics"
    )
    endpoints_configured = metadata.get("endpoints_configured")
    endpoints_successful = metadata.get("endpoints_successful")
    if not isinstance(endpoints_configured, list) or not all(
        isinstance(value, str) for value in endpoints_configured
    ):
        raise NativeReportError("server metrics configured endpoints must be strings")
    if not isinstance(endpoints_successful, list) or not all(
        isinstance(value, str) for value in endpoints_successful
    ):
        raise NativeReportError("server metrics successful endpoints must be strings")
    return ServerMetricsResults(
        benchmark_id=run.benchmark_id,
        endpoint_summaries=project_phase(profiling_metrics),
        warmup_endpoint_summaries=project_phase(warmup_metrics),
        start_ns=start_ns,
        end_ns=end_ns,
        endpoints_configured=endpoints_configured,
        endpoints_successful=endpoints_successful,
        warmup_start_ns=warmup_start_ns,
        warmup_end_ns=warmup_end_ns,
        aggregation_time_filter=TimeRangeFilter(start_ns=start_ns, end_ns=end_ns)
        if start_ns < end_ns
        else None,
    )


def _server_metric_timeslices(
    metric_type: str, authored: Any, metric_name: str
) -> list[Any]:
    """Translate typed native timeslices without recomputing their statistics."""
    from aiperf.common.models.server_metrics_models import (
        CounterTimeslice,
        GaugeTimeslice,
        HistogramTimeslice,
    )

    if not isinstance(authored, list):
        raise NativeReportError(
            f"native server metric {metric_name!r} timeslices must be an array"
        )
    output = []
    for index, authored_slice in enumerate(authored):
        timeslice = _mapping(
            authored_slice,
            f"native server metric {metric_name!r} timeslice[{index}]",
        )
        start_ns = _required_int(timeslice.get("start_ns"), metric_name, "start_ns")
        end_ns = _required_int(timeslice.get("end_ns"), metric_name, "end_ns")
        complete = timeslice.get("complete")
        if not isinstance(complete, bool):
            raise NativeReportError(
                f"native server metric {metric_name!r} timeslice complete must be boolean"
            )
        stats = _mapping(
            timeslice.get("stats"),
            f"native server metric {metric_name!r} timeslice stats",
        )
        common = {
            "start_ns": start_ns,
            "end_ns": end_ns,
            "is_complete": None if complete else False,
        }
        if metric_type in {"gauge", "unknown"}:
            values = {
                key: _optional_number(stats.get(key), metric_name, key)
                for key in ("avg", "min", "max")
            }
            if all(value is not None for value in values.values()):
                output.append(GaugeTimeslice(**common, **values))
        elif metric_type == "counter":
            total = _optional_number(stats.get("total"), metric_name, "total")
            rate = _optional_number(stats.get("rate"), metric_name, "rate")
            if total is not None and rate is not None:
                output.append(CounterTimeslice(**common, total=total, rate=rate))
        elif metric_type == "histogram":
            count = _optional_int(stats.get("count"), metric_name, "count")
            total = _optional_number(stats.get("sum"), metric_name, "sum")
            avg = _optional_number(stats.get("avg"), metric_name, "avg")
            buckets = _mapping(
                stats.get("buckets", {}),
                f"native server metric {metric_name!r} timeslice buckets",
            )
            if count is not None and total is not None and avg is not None:
                output.append(
                    HistogramTimeslice(
                        **common,
                        count=count,
                        sum=total,
                        avg=avg,
                        buckets={
                            key: _required_int(value, metric_name, f"bucket {key}")
                            for key, value in buckets.items()
                        }
                        or None,
                    )
                )
    return output


def _render_server_metrics_parquet(run: BenchmarkRun, results: Any) -> None:
    """Feed Rust-owned raw records into Python's canonical Parquet renderer."""
    import asyncio

    from aiperf.common.models.server_metrics_models import (
        ServerMetricsRecord,
        TimeRangeFilter,
    )
    from aiperf.orchestrator.rust_wire import SERVER_METRICS_PARQUET_WIRE_PATH
    from aiperf.server_metrics.parquet_exporter import ServerMetricsParquetExporter
    from aiperf.server_metrics.storage import ServerMetricsHierarchy

    wire_path = run.artifact_dir / SERVER_METRICS_PARQUET_WIRE_PATH
    hierarchy = ServerMetricsHierarchy()
    try:
        with wire_path.open("rb") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    record = ServerMetricsRecord.model_validate(orjson.loads(line))
                except Exception as error:
                    raise NativeReportError(
                        f"invalid Rust server-metrics Parquet wire record at line "
                        f"{line_number}: {error}"
                    ) from error
                hierarchy.add_record(record)
    except OSError as error:
        raise NativeReportError(
            f"failed to read Rust server-metrics Parquet wire {wire_path}: {error}"
        ) from error

    class NativeServerMetricsHierarchyAdapter:
        def __init__(self) -> None:
            self.run = run

        def get_hierarchy_for_export(self) -> Any:
            return hierarchy

    if results.start_ns >= results.end_ns:
        raise NativeReportError(
            "cannot render server-metrics Parquet without a positive profiling boundary"
        )
    exporter = ServerMetricsParquetExporter(
        NativeServerMetricsHierarchyAdapter(),
        TimeRangeFilter(start_ns=results.start_ns, end_ns=results.end_ns),
    )
    asyncio.run(exporter.export())
    wire_path.unlink()


def _project_gpu_telemetry(
    payload: dict[str, Any], native_summary: dict[str, Any]
) -> Any | None:
    """Serialize Rust-labeled GPU series through the established Python model."""
    from datetime import datetime

    from aiperf.common.models import (
        EndpointData,
        GpuSummary,
        TelemetryExportData,
        TelemetrySummary,
    )
    from aiperf.exporters.utils import normalize_endpoint_display

    metrics = _mapping(payload.get("metrics"), "native report metrics")
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    endpoint_order: list[str] = []
    for name, authored_entry in metrics.items():
        entry = _mapping(authored_entry, f"metric {name!r}")
        metric_type = entry.get("type")
        unit = entry.get("unit")
        if not isinstance(unit, str):
            raise NativeReportError(f"metric {name!r} unit must be a string")
        authored_series = entry.get("series")
        if not isinstance(authored_series, list) or not authored_series:
            raise NativeReportError(f"metric {name!r} must contain at least one series")
        for index, authored in enumerate(authored_series):
            series = _mapping(authored, f"metric {name!r} series[{index}]")
            labels = series.get("labels")
            endpoint_url = series.get("endpoint_url")
            if not isinstance(labels, dict) or not isinstance(endpoint_url, str):
                continue
            required = ("gpu", "gpu_uuid", "model_name")
            if not all(isinstance(labels.get(field), str) for field in required):
                continue
            try:
                gpu_index = int(labels["gpu"])
            except ValueError as error:
                raise NativeReportError(
                    f"metric {name!r} GPU index must be an integer string"
                ) from error
            if gpu_index < 0:
                raise NativeReportError(
                    f"metric {name!r} GPU index must be non-negative"
                )
            stats = _mapping(series.get("stats"), f"metric {name!r} stats")
            values = JsonMetricResult(
                unit=unit,
                **_legacy_stats(metric_type, stats, name),
            )
            if endpoint_url not in endpoint_order:
                endpoint_order.append(endpoint_url)
            endpoint = grouped.setdefault(endpoint_url, {})
            gpu_uuid = labels["gpu_uuid"]
            gpu = endpoint.setdefault(
                gpu_uuid,
                {
                    "gpu_index": gpu_index,
                    "gpu_name": labels["model_name"],
                    "gpu_uuid": gpu_uuid,
                    "hostname": labels.get("hostname"),
                    "namespace": labels.get("namespace"),
                    "pod_name": labels.get("pod"),
                    "metrics": {},
                },
            )
            identity = (gpu["gpu_index"], gpu["gpu_name"], gpu["gpu_uuid"])
            if identity != (gpu_index, labels["model_name"], gpu_uuid):
                raise NativeReportError(
                    f"GPU metadata changed across native series for {gpu_uuid!r}"
                )
            gpu["metrics"][name] = values

    if not grouped:
        return None
    endpoints = {
        normalize_endpoint_display(endpoint_url): EndpointData(
            gpus={
                f"gpu_{gpu['gpu_index']}": GpuSummary.model_validate(gpu)
                for gpu in gpus.values()
            }
        )
        for endpoint_url, gpus in grouped.items()
    }

    def native_time(field: str) -> datetime:
        value = native_summary.get(field)
        if isinstance(value, bool) or not isinstance(value, int | float) or value < 0:
            value = 0
        return datetime.fromtimestamp(float(value) / 1_000_000_000)

    return TelemetryExportData(
        summary=TelemetrySummary(
            endpoints_configured=endpoint_order,
            endpoints_successful=endpoint_order,
            start_time=native_time("start_time"),
            end_time=native_time("end_time"),
        ),
        endpoints=endpoints,
        error_summary=None,
    )


def _atomic_write_text(path: Path, content: str) -> None:
    """Commit one rendered compatibility report without a partial-file window."""
    import os
    from uuid import uuid4

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _counter_value(metric: JsonMetricResult | None) -> int:
    if metric is None:
        return 0
    value = metric.sum if metric.sum is not None else metric.avg
    if value is None:
        return 0
    if not float(value).is_integer() or value < 0:
        raise NativeReportError(
            f"native counter value must be a non-negative integer: {value!r}"
        )
    return int(value)


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
