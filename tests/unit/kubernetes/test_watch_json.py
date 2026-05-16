# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the JSON renderer."""

from __future__ import annotations

import io
from datetime import datetime, timezone

import orjson

from aiperf.kubernetes.watch_models import (
    DiagnosisIssue,
    DiagnosisResult,
    EventSnapshot,
    MetricsSnapshot,
    PodSnapshot,
    ProgressSnapshot,
    WatchSnapshot,
    WorkersSnapshot,
)
from aiperf.kubernetes.watch_render_json import JsonRenderer


def _snap(**kwargs) -> WatchSnapshot:
    defaults = dict(
        timestamp=datetime(2026, 3, 18, 12, 0, 0, tzinfo=timezone.utc),
        job_id="test-job",
        namespace="aiperf-benchmarks",
        phase="Running",
    )
    defaults.update(kwargs)
    return WatchSnapshot(**defaults)


class TestJsonRendererOutput:
    def test_renders_single_ndjson_line(self) -> None:
        buf = io.StringIO()
        renderer = JsonRenderer(output=buf)
        renderer.render(_snap())
        output = buf.getvalue()
        lines = output.strip().split("\n")
        assert len(lines) == 1

    def test_output_is_valid_json(self) -> None:
        buf = io.StringIO()
        renderer = JsonRenderer(output=buf)
        renderer.render(_snap())
        data = orjson.loads(buf.getvalue())
        assert isinstance(data, dict)

    def test_contains_required_fields(self) -> None:
        buf = io.StringIO()
        renderer = JsonRenderer(output=buf)
        renderer.render(_snap())
        data = orjson.loads(buf.getvalue())
        assert data["job_id"] == "test-job"
        assert data["namespace"] == "aiperf-benchmarks"
        assert data["phase"] == "Running"
        assert "timestamp" in data
        assert "diagnosis" in data
        assert "conditions" in data

    def test_metrics_serialized(self) -> None:
        buf = io.StringIO()
        renderer = JsonRenderer(output=buf)
        renderer.render(
            _snap(
                metrics=MetricsSnapshot(
                    request_throughput_rps=1500.5,
                    request_latency_avg_ms=3.14,
                    request_count=50000,
                ),
            )
        )
        data = orjson.loads(buf.getvalue())
        assert data["metrics"]["request_throughput_rps"] == 1500.5
        assert data["metrics"]["request_latency_avg_ms"] == 3.14
        assert data["metrics"]["request_count"] == 50000

    def test_workers_serialized(self) -> None:
        buf = io.StringIO()
        renderer = JsonRenderer(output=buf)
        renderer.render(_snap(workers=WorkersSnapshot(ready=3, total=5)))
        data = orjson.loads(buf.getvalue())
        assert data["workers"]["ready"] == 3
        assert data["workers"]["total"] == 5

    def test_pods_serialized(self) -> None:
        buf = io.StringIO()
        renderer = JsonRenderer(output=buf)
        renderer.render(
            _snap(
                pods=[
                    PodSnapshot(
                        name="ctrl-0", role="controller", status="Running", ready=True
                    ),
                    PodSnapshot(
                        name="work-0",
                        role="worker",
                        status="Running",
                        restarts=1,
                        oom_killed=True,
                    ),
                ]
            )
        )
        data = orjson.loads(buf.getvalue())
        assert len(data["pods"]) == 2
        assert data["pods"][0]["role"] == "controller"
        assert data["pods"][1]["oom_killed"] is True

    def test_events_serialized(self) -> None:
        buf = io.StringIO()
        renderer = JsonRenderer(output=buf)
        renderer.render(
            _snap(
                events=[
                    EventSnapshot(
                        timestamp="2026-03-18T12:00:00Z",
                        type="Warning",
                        reason="OOMKilled",
                        object="Pod/work-0",
                        message="OOM",
                        count=1,
                    ),
                ]
            )
        )
        data = orjson.loads(buf.getvalue())
        assert len(data["events"]) == 1
        assert data["events"][0]["reason"] == "OOMKilled"

    def test_diagnosis_serialized(self) -> None:
        buf = io.StringIO()
        renderer = JsonRenderer(output=buf)
        renderer.render(
            _snap(
                diagnosis=DiagnosisResult(
                    health="degraded",
                    issues=[
                        DiagnosisIssue(
                            id="oom_restart",
                            severity="warning",
                            title="OOM",
                            detail="killed",
                            impact="progress reset",
                            suggested_fix="increase memory",
                            runbook="aiperf kube profile --env ...",
                        )
                    ],
                    error_rate=0.05,
                ),
            )
        )
        data = orjson.loads(buf.getvalue())
        assert data["diagnosis"]["health"] == "degraded"
        assert data["diagnosis"]["issues"][0]["id"] == "oom_restart"
        assert data["diagnosis"]["issues"][0]["runbook"] is not None
        assert data["diagnosis"]["error_rate"] == 0.05

    def test_progress_serialized(self) -> None:
        buf = io.StringIO()
        renderer = JsonRenderer(output=buf)
        renderer.render(
            _snap(
                progress=ProgressSnapshot(
                    percent=78.3,
                    requests_completed=84150,
                    requests_total=107000,
                    eta_seconds=13.0,
                    duration_target_seconds=60.0,
                ),
            )
        )
        data = orjson.loads(buf.getvalue())
        assert data["progress"]["percent"] == 78.3
        assert data["progress"]["eta_seconds"] == 13.0

    def test_results_included_on_completion(self) -> None:
        buf = io.StringIO()
        renderer = JsonRenderer(output=buf)
        renderer.render(
            _snap(
                phase="Completed",
                results={"request_throughput": {"avg": 1500.0, "unit": "requests/sec"}},
            )
        )
        data = orjson.loads(buf.getvalue())
        assert data["phase"] == "Completed"
        assert data["results"]["request_throughput"]["avg"] == 1500.0

    def test_multiple_renders_produce_multiple_lines(self) -> None:
        buf = io.StringIO()
        renderer = JsonRenderer(output=buf)
        renderer.render(_snap(phase="Pending"))
        renderer.render(_snap(phase="Running"))
        renderer.render(_snap(phase="Completed"))
        lines = buf.getvalue().strip().split("\n")
        assert len(lines) == 3
        assert orjson.loads(lines[0])["phase"] == "Pending"
        assert orjson.loads(lines[2])["phase"] == "Completed"

    def test_start_and_stop_are_noop(self) -> None:
        renderer = JsonRenderer()
        renderer.start()
        renderer.stop()
