# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
import types
from pathlib import Path
from queue import Empty
from typing import Any

import orjson
import pytest

from aiperf.post_processors.otel_streaming_fanout import (
    OTelStreamingFanoutConfig,
    run_otel_streaming_fanout,
)
from tests.unit.post_processors.conftest import install_fake_otel_modules


class _SequenceQueue:
    def __init__(self, events: list[dict[str, Any]]) -> None:
        self._events = list(events)

    def get(self, timeout: float | None = None) -> dict[str, Any]:
        if not self._events:
            raise Empty
        return self._events.pop(0)


class _ObservedSequenceQueue(_SequenceQueue):
    def __init__(
        self,
        events: list[dict[str, Any]],
        *,
        before_get: dict[int, Any],
    ) -> None:
        super().__init__(events)
        self._before_get = before_get
        self._get_calls = 0

    def get(self, timeout: float | None = None) -> dict[str, Any]:
        callback = self._before_get.get(self._get_calls)
        if callback is not None:
            callback()
        self._get_calls += 1
        return super().get(timeout)


def _install_fake_mlflow_modules(
    monkeypatch: pytest.MonkeyPatch,
    state: dict[str, Any],
) -> None:
    mlflow_module = types.ModuleType("mlflow")
    entities_module = types.ModuleType("mlflow.entities")
    tracking_module = types.ModuleType("mlflow.tracking")

    class FakeMetric:
        def __init__(self, key: str, value: float, timestamp: int, step: int) -> None:
            self.key = key
            self.value = value
            self.timestamp = timestamp
            self.step = step

    class FakeClient:
        def log_batch(
            self,
            *,
            run_id: str,
            metrics: list[FakeMetric],
            params: list[Any],
            tags: list[Any],
        ) -> None:
            state["log_batch_calls"].append(
                {"run_id": run_id, "metrics": metrics, "params": params, "tags": tags}
            )

    class FakeRun:
        def __init__(self, run_id: str) -> None:
            self.info = types.SimpleNamespace(run_id=run_id)

    def set_tracking_uri(uri: str) -> None:
        state["tracking_uri"] = uri

    def set_experiment(name: str) -> None:
        state["experiment"] = name

    def start_run(run_name: str | None = None) -> FakeRun:
        state["run_name"] = run_name
        return FakeRun("live-run-1")

    def set_tags(tags: dict[str, str]) -> None:
        state["tags"] = tags

    def set_tag(key: str, value: str) -> None:
        state.setdefault("single_tags", {})[key] = value

    def end_run() -> None:
        state["end_run_called"] = True

    mlflow_module.set_tracking_uri = set_tracking_uri  # type: ignore[attr-defined]
    mlflow_module.set_experiment = set_experiment  # type: ignore[attr-defined]
    mlflow_module.start_run = start_run  # type: ignore[attr-defined]
    mlflow_module.set_tags = set_tags  # type: ignore[attr-defined]
    mlflow_module.set_tag = set_tag  # type: ignore[attr-defined]
    mlflow_module.end_run = end_run  # type: ignore[attr-defined]
    entities_module.Metric = FakeMetric  # type: ignore[attr-defined]
    tracking_module.MlflowClient = FakeClient  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "mlflow", mlflow_module)
    monkeypatch.setitem(sys.modules, "mlflow.entities", entities_module)
    monkeypatch.setitem(sys.modules, "mlflow.tracking", tracking_module)


def _build_config(
    tmp_path: Path,
    *,
    endpoint_url: str | None,
    max_batch_records: int = 500,
) -> OTelStreamingFanoutConfig:
    return OTelStreamingFanoutConfig(
        endpoint_url=endpoint_url,
        request_timeout_seconds=5.0,
        export_interval_millis=100,
        export_timeout_millis=1000,
        max_batch_records=max_batch_records,
        resource_attributes={"service.name": "aiperf"},
        mlflow_tracking_uri="http://mlflow:5000",
        mlflow_experiment="aiperf-tests",
        mlflow_run_name="live-test",
        mlflow_tags={"team": "perf"},
        mlflow_parent_run_id=None,
        benchmark_id="bench-1",
        metadata_file=tmp_path / "mlflow_export.json",
    )


def test_run_fanout_processes_events_for_otel_and_mlflow(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    otel_state: dict[str, Any] = {}
    mlflow_state: dict[str, Any] = {"log_batch_calls": []}
    install_fake_otel_modules(monkeypatch, otel_state)
    _install_fake_mlflow_modules(monkeypatch, mlflow_state)

    queue = _SequenceQueue(
        [
            {
                "type": "histogram_record",
                "payload": {
                    "metric_name": "aiperf.request_latency_ns",
                    "unit": "ns",
                    "description": "latency",
                    "value": 123.0,
                    "attributes": {"aiperf.worker.id": "worker-1"},
                },
            },
            {
                "type": "counter_add",
                "payload": {
                    "metric_name": "aiperf.requests.completed",
                    "unit": "1",
                    "description": "completed",
                    "value": 1.0,
                    "attributes": {"aiperf.worker.id": "worker-1"},
                },
            },
            {"type": "flush", "payload": {}},
            {"type": "shutdown", "payload": {}},
        ]
    )
    config = _build_config(tmp_path, endpoint_url="http://collector:4318/v1/metrics")

    run_otel_streaming_fanout(queue, config)

    meter = otel_state["meter"]
    assert "aiperf.request_latency_ns" in meter.histograms
    assert "aiperf.requests.completed" in meter.counters
    assert otel_state["force_flush_calls"]
    assert otel_state["shutdown_calls"] == 1

    assert mlflow_state["log_batch_calls"]
    logged_metric_keys = [
        metric.key
        for call in mlflow_state["log_batch_calls"]
        for metric in call["metrics"]
    ]
    assert "live.aiperf.request_latency_ns" in logged_metric_keys
    assert "live.aiperf.requests.completed" in logged_metric_keys
    assert mlflow_state["end_run_called"] is True

    metadata = orjson.loads((tmp_path / "mlflow_export.json").read_bytes())
    assert metadata["run_id"] == "live-run-1"
    assert metadata["live_streaming"] is True


def test_run_fanout_without_otel_sink_still_logs_mlflow(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mlflow_state: dict[str, Any] = {"log_batch_calls": []}
    _install_fake_mlflow_modules(monkeypatch, mlflow_state)

    queue = _SequenceQueue(
        [
            {
                "type": "counter_add",
                "payload": {
                    "metric_name": "aiperf.requests.completed",
                    "unit": "1",
                    "description": "completed",
                    "value": 1.0,
                    "attributes": {},
                },
            },
            {"type": "shutdown", "payload": {}},
        ]
    )
    config = _build_config(tmp_path, endpoint_url=None)

    run_otel_streaming_fanout(queue, config)

    assert mlflow_state["log_batch_calls"]
    assert mlflow_state["end_run_called"] is True


def test_run_fanout_logs_timing_gauge_snapshots_to_mlflow(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mlflow_state: dict[str, Any] = {"log_batch_calls": []}
    _install_fake_mlflow_modules(monkeypatch, mlflow_state)

    queue = _SequenceQueue(
        [
            {
                "type": "up_down_counter_add",
                "payload": {
                    "metric_name": "aiperf.timing.requests.in_flight",
                    "unit": "1",
                    "description": "in-flight requests",
                    "value": 2.0,
                    "attributes": {"aiperf.benchmark_phase": "profiling"},
                },
            },
            {
                "type": "up_down_counter_add",
                "payload": {
                    "metric_name": "aiperf.timing.requests.in_flight",
                    "unit": "1",
                    "description": "in-flight requests",
                    "value": -1.0,
                    "attributes": {"aiperf.benchmark_phase": "profiling"},
                },
            },
            {
                "type": "up_down_counter_add",
                "payload": {
                    "metric_name": "aiperf.timing.requests.in_flight",
                    "unit": "1",
                    "description": "in-flight requests",
                    "value": -1.0,
                    "attributes": {"aiperf.benchmark_phase": "profiling"},
                },
            },
            {"type": "shutdown", "payload": {}},
        ]
    )
    config = _build_config(tmp_path, endpoint_url=None)

    run_otel_streaming_fanout(queue, config)

    assert len(mlflow_state["log_batch_calls"]) == 1
    logged_metrics = mlflow_state["log_batch_calls"][0]["metrics"]
    assert [metric.key for metric in logged_metrics] == [
        "live.aiperf.timing.requests.in_flight.profiling",
        "live.aiperf.timing.requests.in_flight.profiling",
        "live.aiperf.timing.requests.in_flight.profiling",
    ]
    assert [metric.value for metric in logged_metrics] == [2.0, 1.0, 0.0]


def test_run_fanout_flushes_mlflow_batches_when_max_batch_size_is_reached(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mlflow_state: dict[str, Any] = {"log_batch_calls": []}
    _install_fake_mlflow_modules(monkeypatch, mlflow_state)

    def assert_threshold_flush_happened() -> None:
        assert len(mlflow_state["log_batch_calls"]) == 1
        first_batch = mlflow_state["log_batch_calls"][0]["metrics"]
        assert [metric.value for metric in first_batch] == [1.0, 3.0]

    queue = _ObservedSequenceQueue(
        [
            {
                "type": "counter_add",
                "payload": {
                    "metric_name": "aiperf.requests.completed",
                    "unit": "1",
                    "description": "completed",
                    "value": 1.0,
                    "attributes": {},
                },
            },
            {
                "type": "counter_add",
                "payload": {
                    "metric_name": "aiperf.requests.completed",
                    "unit": "1",
                    "description": "completed",
                    "value": 2.0,
                    "attributes": {},
                },
            },
            {
                "type": "counter_add",
                "payload": {
                    "metric_name": "aiperf.requests.completed",
                    "unit": "1",
                    "description": "completed",
                    "value": 3.0,
                    "attributes": {},
                },
            },
            {"type": "shutdown", "payload": {}},
        ],
        before_get={2: assert_threshold_flush_happened},
    )
    config = _build_config(tmp_path, endpoint_url=None, max_batch_records=2)

    run_otel_streaming_fanout(queue, config)

    assert len(mlflow_state["log_batch_calls"]) == 2
    first_batch = mlflow_state["log_batch_calls"][0]["metrics"]
    second_batch = mlflow_state["log_batch_calls"][1]["metrics"]
    assert [metric.value for metric in first_batch] == [1.0, 3.0]
    assert [metric.value for metric in second_batch] == [6.0]
    assert [metric.step for metric in first_batch] == [0, 1]
    assert [metric.step for metric in second_batch] == [2]
    assert mlflow_state["end_run_called"] is True


def test_run_fanout_invalid_payload_logs_warning_and_continues(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    otel_state: dict[str, Any] = {}
    install_fake_otel_modules(monkeypatch, otel_state)

    queue = _SequenceQueue(
        [
            {
                "type": "histogram_record",
                "payload": {
                    # Missing required fields on purpose.
                    "metric_name": "aiperf.invalid_payload",
                },
            },
            {"type": "shutdown", "payload": {}},
        ]
    )
    config = OTelStreamingFanoutConfig(
        endpoint_url="http://collector:4318/v1/metrics",
        request_timeout_seconds=5.0,
        export_interval_millis=100,
        export_timeout_millis=1000,
        max_batch_records=500,
        resource_attributes={"service.name": "aiperf"},
        mlflow_tracking_uri=None,
        mlflow_experiment="aiperf-tests",
        mlflow_run_name=None,
        mlflow_tags={},
        mlflow_parent_run_id=None,
        benchmark_id=None,
        metadata_file=tmp_path / "mlflow_export.json",
    )

    run_otel_streaming_fanout(queue, config)

    assert "Invalid histogram fanout payload received" in caplog.text
    assert otel_state["shutdown_calls"] == 1


def test_run_fanout_redacts_tracking_uri_userinfo_in_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Regression: mlflow_export.json is uploaded as a run artifact by the
    deferred exporter, so credentials in --mlflow-tracking-uri must not be
    written verbatim to the on-disk metadata file."""
    mlflow_state: dict[str, Any] = {"log_batch_calls": []}
    _install_fake_mlflow_modules(monkeypatch, mlflow_state)

    queue = _SequenceQueue([{"type": "shutdown", "payload": {}}])
    config = OTelStreamingFanoutConfig(
        endpoint_url=None,
        request_timeout_seconds=5.0,
        export_interval_millis=100,
        export_timeout_millis=1000,
        max_batch_records=500,
        resource_attributes={"service.name": "aiperf"},
        mlflow_tracking_uri="postgresql://dbuser:s3cret@db:5432/mlflow",
        mlflow_experiment="aiperf-tests",
        mlflow_run_name="live-test",
        mlflow_tags={},
        mlflow_parent_run_id=None,
        benchmark_id="bench-1",
        metadata_file=tmp_path / "mlflow_export.json",
    )

    run_otel_streaming_fanout(queue, config)

    metadata_path = tmp_path / "mlflow_export.json"
    metadata = orjson.loads(metadata_path.read_bytes())
    assert metadata["tracking_uri"] == "postgresql://<redacted>@db:5432/mlflow"
    raw = metadata_path.read_text(encoding="utf-8")
    assert "dbuser" not in raw
    assert "s3cret" not in raw
    # The fanout still configures mlflow with the real URI so it can authenticate.
    assert mlflow_state["tracking_uri"] == "postgresql://dbuser:s3cret@db:5432/mlflow"
