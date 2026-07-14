# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import orjson
import pytest

from aiperf.config import AIPerfConfig, BenchmarkRun
from aiperf.orchestrator.native_report import (
    NativeReportError,
    export_python_compatibility_reports,
    load_native_report,
    load_native_summary,
    project_native_summary,
)


@pytest.fixture(autouse=True)
def _legacy_python_export(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the legacy Python ExporterManager rendering path.

    ``export_python_compatibility_reports`` is a no-op on the default native
    path (the Rust ``aiperf::export`` sinks are the sole emitter). These tests
    pin the legacy Python renderer that remains reachable via
    ``AIPERF_RUNTIME_NATIVE_EXPORT=0`` and the ``AIPERF_RUNTIME_ENGINE=python``
    service mesh, so they run with the native plane disabled.
    """
    from aiperf.common.environment import Environment

    monkeypatch.setattr(Environment.RUNTIME, "NATIVE_EXPORT", False)


def _entry(metric_type: str, unit: str, stats: dict, *, labels=None) -> dict:
    return {
        "type": metric_type,
        "unit": unit,
        "group": "latency",
        "higher_is_better": False,
        "series": [{"labels": labels, "stats": stats}],
    }


def test_projects_every_native_stats_shape_without_recomputing_values(tmp_path) -> None:
    report = {
        "schema_version": "2.0",
        "metrics": {
            "request_latency": _entry(
                "distribution",
                "ms",
                {
                    "count": 3,
                    "avg": 4.5,
                    "min": 1.0,
                    "max": 8.0,
                    "std": 2.25,
                    "percentiles": {"p50": 4.0, "p95": 7.5, "p99": 8.0},
                },
            ),
            "request_throughput": _entry("scalar", "requests/sec", {"value": 12.25}),
            "request_count": _entry(
                "counter", "requests", {"total": 3.0, "rate": 12.25}
            ),
            "server_histogram": _entry(
                "histogram",
                "ms",
                {
                    "count": 4,
                    "sum": 18.0,
                    "avg": 4.5,
                    "count_rate": 2.0,
                    "sum_rate": 9.0,
                    "percentiles": {"p50": 4.0, "p99": 9.0},
                    "buckets": {"5": 3, "+Inf": 1},
                },
            ),
        },
    }

    result = project_native_summary(report)
    assert result["request_latency"].model_dump(exclude_none=True) == {
        "unit": "ms",
        "avg": 4.5,
        "p50": 4.0,
        "p95": 7.5,
        "p99": 8.0,
        "min": 1.0,
        "max": 8.0,
        "std": 2.25,
        "count": 3,
    }
    assert result["request_throughput"].avg == 12.25
    assert result["request_count"].avg == 3.0
    assert result["request_count"].sum == 3.0
    assert result["server_histogram"].count == 4
    assert result["server_histogram"].sum == 18.0

    path = tmp_path / "native.json"
    path.write_bytes(orjson.dumps(report))
    assert load_native_summary(path) == result


def test_selects_explicit_unlabeled_aggregate_from_labeled_series() -> None:
    entry = _entry("scalar", "watts", {"value": 200.0})
    entry["series"] = [
        {"labels": {"gpu": "0"}, "stats": {"value": 90.0}},
        {"labels": None, "stats": {"value": 200.0}},
        {"labels": {"gpu": "1"}, "stats": {"value": 110.0}},
    ]
    result = project_native_summary(
        {"schema_version": "2.0", "metrics": {"gpu_power": entry}}
    )
    assert result["gpu_power"].avg == 200.0


def test_omits_multi_series_metric_without_flat_aggregate() -> None:
    entry = _entry("scalar", "watts", {"value": 90.0}, labels={"gpu": "0"})
    entry["series"].append({"labels": {"gpu": "1"}, "stats": {"value": 110.0}})
    assert (
        project_native_summary(
            {"schema_version": "2.0", "metrics": {"gpu_power": entry}}
        )
        == {}
    )


def test_projects_accuracy_analysis_for_sweeps_and_search() -> None:
    report = {
        "schema_version": "2.0",
        "metrics": {"request_count": _entry("counter", "requests", {"total": 4.0})},
        "accuracy": {
            "summary": {
                "overall": {
                    "n": 4,
                    "correct_count": 2,
                    "unparsed_count": 1,
                    "accuracy": 0.5,
                    "unparsed_rate": 0.25,
                    "mean_confidence": 0.5,
                    "ci": {"low": 0.15, "high": 0.85},
                },
                "per_task": {
                    "math": {
                        "n": 2,
                        "correct_count": 2,
                        "unparsed_count": 0,
                        "accuracy": 1.0,
                        "unparsed_rate": 0.0,
                        "mean_confidence": 1.0,
                        "ci": {"low": 0.34, "high": 1.0},
                    }
                },
            },
            "accuracy_at_load": {
                "accuracy": 0.5,
                "goodput": 10.0,
                "request_throughput": 12.0,
                "correct_answers_per_second": 5.0,
            },
            "correct_answers_per_kwh": 720.0,
        },
    }

    projected = project_native_summary(report)

    assert projected["accuracy.overall"].avg == 0.5
    assert projected["accuracy.overall"].count == 4
    assert projected["accuracy.overall"].sum == 2
    assert projected["accuracy.task.math"].avg == 1.0
    assert projected["accuracy.unparsed"].avg == 0.25
    assert projected["accuracy.unparsed"].sum == 1
    assert projected["accuracy.unparsed.task.math"].avg == 0.0
    assert projected["accuracy.correct_answers_per_second"].avg == 5.0
    assert projected["accuracy.correct_answers_per_kwh"].avg == 720.0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda report: report.update(schema_version="1.4"), "schema_version"),
        (
            lambda report: report["metrics"]["latency"]["series"][0]["stats"][
                "percentiles"
            ].update(p42=1.0),
            "not representable",
        ),
        (
            lambda report: report["metrics"]["latency"].update(type="unknown"),
            "unsupported native metric type",
        ),
    ],
)
def test_rejects_contract_drift(mutation, message: str) -> None:
    report = {
        "schema_version": "2.0",
        "metrics": {
            "latency": _entry(
                "distribution",
                "ms",
                {
                    "count": 1,
                    "avg": 1.0,
                    "min": 1.0,
                    "max": 1.0,
                    "std": 0.0,
                    "percentiles": {"p50": 1.0},
                },
            )
        },
    }
    mutation(report)
    with pytest.raises(NativeReportError, match=message):
        project_native_summary(report)


def test_python_report_generators_serialize_native_values_without_recomputing(
    tmp_path,
) -> None:
    gpu_power = _entry(
        "distribution",
        "W",
        {
            "count": 3,
            "avg": 250.0,
            "min": 240.0,
            "max": 260.0,
            "std": 10.0,
            "percentiles": {"p50": 250.0, "p99": 260.0},
        },
        labels={
            "gpu": "0",
            "gpu_uuid": "GPU-native",
            "model_name": "H100",
            "hostname": "node-a",
        },
    )
    gpu_power["series"][0]["endpoint_url"] = "http://dcgm:9400/metrics"
    report = {
        "schema_version": "2.0",
        "aiperf_version": "0.0.0",
        "run": {"mode": "online", "model": "mock-model"},
        "summary": {"was_cancelled": False},
        "metrics": {
            "request_latency": _entry(
                "distribution",
                "ms",
                {
                    "count": 2,
                    "avg": 12.5,
                    "min": 10.0,
                    "max": 15.0,
                    "std": 2.5,
                    "percentiles": {"p50": 12.5, "p99": 15.0},
                },
            ),
            "request_count": _entry("counter", "requests", {"total": 2.0, "rate": 4.0}),
            "gpu_power_usage": gpu_power,
        },
        "warmup_metrics": {
            "request_count": _entry("counter", "requests", {"total": 1.0, "rate": 1.0})
        },
        "errors": [],
    }
    report["metrics"]["request_latency"]["series"][0]["timeslices"] = [
        {
            "start_ns": 0,
            "end_ns": 1_000_000_000,
            "complete": True,
            "stats": {
                "count": 1,
                "avg": 10.0,
                "min": 10.0,
                "max": 10.0,
                "std": 0.0,
                "percentiles": {"p50": 10.0, "p99": 10.0},
            },
        },
        {
            "start_ns": 1_000_000_000,
            "end_ns": 1_500_000_000,
            "complete": False,
            "stats": {
                "count": 1,
                "avg": 15.0,
                "min": 15.0,
                "max": 15.0,
                "std": 0.0,
                "percentiles": {"p50": 15.0, "p99": 15.0},
            },
        },
    ]
    envelope = AIPerfConfig.model_validate(
        {
            "benchmark": {
                "models": ["mock-model"],
                "endpoint": {"urls": ["http://127.0.0.1:8000"]},
                "dataset": {"type": "synthetic"},
                "profiling": {
                    "type": "concurrency",
                    "requests": 2,
                    "concurrency": 1,
                },
                "artifacts": {"dir": str(tmp_path), "slice_duration": 1.0},
                "gpu_telemetry": {"enabled": False},
                "server_metrics": {"enabled": False},
            }
        }
    )
    run = BenchmarkRun(
        benchmark_id="compatibility-proof",
        cfg=envelope.benchmark,
        artifact_dir=tmp_path,
        label="native",
        random_seed=7,
    )
    projected = project_native_summary(report)

    export_python_compatibility_reports(report, projected, run)

    exported = orjson.loads((tmp_path / "profile_export_aiperf.json").read_bytes())
    assert exported["schema_version"] == "1.4"
    assert exported["benchmark_id"] == "compatibility-proof"
    assert exported["request_latency"] == {
        "unit": "ms",
        "avg": 12.5,
        "p50": 12.5,
        "p99": 15.0,
        "min": 10.0,
        "max": 15.0,
        "std": 2.5,
        "count": 2,
    }
    assert exported["warmup_metrics"]["request_count"]["avg"] == 1.0
    assert exported["run_info"]["random_seed"] == 7
    assert exported["input_config"]["endpoint"]["urls"] == ["http://127.0.0.1:8000"]
    assert exported["telemetry_data"]["summary"]["endpoints_successful"] == [
        "http://dcgm:9400/metrics"
    ]
    gpu = exported["telemetry_data"]["endpoints"]["dcgm:9400"]["gpus"]["gpu_0"]
    assert gpu["gpu_uuid"] == "GPU-native"
    assert gpu["hostname"] == "node-a"
    assert gpu["metrics"]["gpu_power_usage"] == {
        "unit": "W",
        "avg": 250.0,
        "p50": 250.0,
        "p99": 260.0,
        "min": 240.0,
        "max": 260.0,
        "std": 10.0,
        "count": 3,
    }
    csv = (tmp_path / "profile_export_aiperf.csv").read_text()
    assert "Request Latency (ms)" in csv
    assert "12.50" in csv
    assert "GPU Power Usage (W)" in csv
    assert "GPU-native" in csv
    timeslices = orjson.loads(
        (tmp_path / "profile_export_aiperf_timeslices.json").read_bytes()
    )
    assert [item["request_latency"]["avg"] for item in timeslices["timeslices"]] == [
        10.0,
        15.0,
    ]
    assert timeslices["timeslices"][1]["is_complete"] is False
    assert (
        "Request Latency"
        in (tmp_path / "profile_export_aiperf_timeslices.csv").read_text()
    )
    assert (tmp_path / "profile_export_console.txt").exists()
    (tmp_path / "profile_export.jsonl").write_text("{}\n")
    from aiperf.plot.core.mode_detector import ModeDetector, VisualizationMode

    mode, run_directories = ModeDetector().detect_mode([tmp_path])
    assert mode == VisualizationMode.SINGLE_RUN
    assert run_directories == [tmp_path]

    native_path = tmp_path / "native-v2.json"
    native_path.write_bytes(orjson.dumps(report))
    assert load_native_report(native_path) == report


def test_python_export_plugin_extensions_receive_native_results(
    tmp_path, monkeypatch
) -> None:
    from types import SimpleNamespace

    from aiperf.plugin.enums import PluginType

    marker = tmp_path / "custom-extension.txt"

    class CustomExporter:
        def __init__(self, exporter_config) -> None:
            assert exporter_config.run.benchmark_id == "plugin-proof"
            assert exporter_config.results.records[0].tag == "request_count"

        async def export(self) -> None:
            marker.write_text("native results reached Python extension")

        def get_export_info(self):
            return SimpleNamespace(file_path=marker)

    def iter_all(plugin_type):
        if plugin_type == PluginType.DATA_EXPORTER:
            return [(SimpleNamespace(name="custom-native-proof"), CustomExporter)]
        return []

    monkeypatch.setattr("aiperf.exporters.exporter_manager.plugins.iter_all", iter_all)
    report = {
        "schema_version": "2.0",
        "aiperf_version": "0.0.0",
        "run": {"mode": "online", "model": "mock-model"},
        "summary": {"was_cancelled": False},
        "metrics": {
            "request_count": _entry("counter", "requests", {"total": 1.0, "rate": 1.0})
        },
        "errors": [],
    }
    envelope = AIPerfConfig.model_validate(
        {
            "benchmark": {
                "models": ["mock-model"],
                "endpoint": {"urls": ["http://127.0.0.1:8000"]},
                "dataset": {"type": "synthetic"},
                "profiling": {
                    "type": "concurrency",
                    "requests": 1,
                    "concurrency": 1,
                },
                "artifacts": {"dir": str(tmp_path)},
                "gpu_telemetry": {"enabled": False},
                "server_metrics": {"enabled": False},
            }
        }
    )
    run = BenchmarkRun(
        benchmark_id="plugin-proof",
        cfg=envelope.benchmark,
        artifact_dir=tmp_path,
        label="plugin-proof",
    )

    export_python_compatibility_reports(report, project_native_summary(report), run)

    assert marker.read_text() == "native results reached Python extension"


def test_python_server_exporters_render_typed_native_server_series(tmp_path) -> None:
    gauge = _entry(
        "distribution",
        "%",
        {
            "count": 3,
            "avg": 0.5,
            "min": 0.25,
            "max": 0.75,
            "std": 0.25,
            "percentiles": {"p50": 0.5, "p99": 0.75},
        },
        labels={"model": "mock-model"},
    )
    gauge["series"][0].update(
        endpoint_url="http://127.0.0.1:8000/metrics",
        timeslices=[
            {
                "start_ns": 1_000,
                "end_ns": 2_000,
                "complete": True,
                "stats": {
                    "count": 2,
                    "avg": 0.375,
                    "min": 0.25,
                    "max": 0.5,
                    "std": 0.125,
                    "percentiles": {},
                },
            }
        ],
    )
    counter = _entry(
        "counter",
        "requests",
        {"total": 8.0, "rate": 4.0},
        labels={"status": "ok"},
    )
    counter["series"][0]["endpoint_url"] = "http://127.0.0.1:8000/metrics"
    histogram = _entry(
        "histogram",
        "s",
        {
            "count": 4,
            "sum": 1.0,
            "avg": 0.25,
            "count_rate": 2.0,
            "sum_rate": 0.5,
            "percentiles": {"p50": 0.2, "p99": 0.5},
            "buckets": {"0.1": 1, "0.5": 3, "+Inf": 4},
        },
    )
    histogram["series"][0]["endpoint_url"] = "http://127.0.0.1:8000/metrics"
    report = {
        "schema_version": "2.0",
        "aiperf_version": "0.0.0",
        "run": {"mode": "online", "model": "mock-model"},
        "summary": {
            "was_cancelled": False,
            "server_metrics": {
                "endpoints_configured": ["http://127.0.0.1:8000/metrics"],
                "endpoints_successful": ["http://127.0.0.1:8000/metrics"],
                "descriptions": {
                    "kv_cache_usage": "KV cache usage",
                    "requests": "Completed requests",
                    "request_latency": "Request latency",
                },
                "metric_types": {
                    "kv_cache_usage": "gauge",
                    "requests": "counter",
                    "request_latency": "histogram",
                },
                "endpoint_info": {
                    "http://127.0.0.1:8000/metrics": {
                        "total_fetches": 3,
                        "first_fetch_ns": 1_000,
                        "last_fetch_ns": 3_000,
                        "avg_fetch_latency_ms": 0.01,
                        "unique_updates": 3,
                        "first_update_ns": 1_000,
                        "last_update_ns": 3_000,
                        "duration_seconds": 0.000002,
                        "avg_update_interval_ms": 0.001,
                        "median_update_interval_ms": 0.001,
                    }
                },
                "profiling": {"start_ns": 1_000, "end_ns": 3_000},
            },
        },
        "metrics": {
            "request_count": _entry("counter", "requests", {"total": 1.0, "rate": 1.0})
        },
        "server_metrics": {
            "kv_cache_usage": gauge,
            "requests": counter,
            "request_latency": histogram,
        },
        "errors": [],
    }
    envelope = AIPerfConfig.model_validate(
        {
            "benchmark": {
                "models": ["mock-model"],
                "endpoint": {"urls": ["http://127.0.0.1:8000"]},
                "dataset": {"type": "synthetic"},
                "profiling": {
                    "type": "concurrency",
                    "requests": 1,
                    "concurrency": 1,
                },
                "artifacts": {"dir": str(tmp_path)},
                "gpu_telemetry": {"enabled": False},
                "server_metrics": {"formats": ["json", "csv"]},
            }
        }
    )
    run = BenchmarkRun(
        benchmark_id="server-compatibility-proof",
        cfg=envelope.benchmark,
        artifact_dir=tmp_path,
        label="native-server",
    )

    export_python_compatibility_reports(report, project_native_summary(report), run)

    exported = orjson.loads((tmp_path / "server_metrics_export.json").read_bytes())
    assert exported["summary"]["endpoints_successful"] == [
        "http://127.0.0.1:8000/metrics"
    ]
    assert exported["metrics"]["kv_cache_usage"]["type"] == "gauge"
    assert exported["metrics"]["kv_cache_usage"]["series"][0]["stats"] == {
        "avg": 0.5,
        "min": 0.25,
        "max": 0.75,
        "std": 0.25,
        "p50": 0.5,
        "p99": 0.75,
    }
    assert exported["metrics"]["requests"]["series"][0]["stats"] == {
        "total": 8.0,
        "rate": 4.0,
    }
    histogram_stats = exported["metrics"]["request_latency"]["series"][0]
    assert histogram_stats["stats"]["p50_estimate"] == 0.2
    assert histogram_stats["buckets"] == {"0.1": 1, "0.5": 3, "+Inf": 4}
    csv = (tmp_path / "server_metrics_export.csv").read_text()
    assert "kv_cache_usage" in csv
    assert "requests" in csv
    assert "request_latency" in csv
