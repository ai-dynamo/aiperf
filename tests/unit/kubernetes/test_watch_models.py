# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for watch snapshot data models."""

from datetime import datetime, timezone

import orjson

from aiperf.kubernetes.watch_models import (
    DiagnosisIssue,
    DiagnosisResult,
    MetricsSnapshot,
    PodSnapshot,
    WatchSnapshot,
    WorkersSnapshot,
)


class TestWatchSnapshot:
    def test_create_minimal_snapshot(self) -> None:
        snap = WatchSnapshot(
            timestamp=datetime(2026, 3, 18, tzinfo=timezone.utc),
            job_id="test-job",
            namespace="aiperf-benchmarks",
            phase="Pending",
        )
        assert snap.phase == "Pending"
        assert snap.metrics is None
        assert snap.events == []

    def test_to_dict_serializable(self) -> None:
        snap = WatchSnapshot(
            timestamp=datetime(2026, 3, 18, tzinfo=timezone.utc),
            job_id="test-job",
            namespace="aiperf-benchmarks",
            phase="Running",
            metrics=MetricsSnapshot(request_throughput_rps=1500.0),
            workers=WorkersSnapshot(ready=1, total=1),
        )
        d = snap.to_dict()
        assert d["job_id"] == "test-job"
        assert d["metrics"]["request_throughput_rps"] == 1500.0
        # Must be JSON-serializable
        orjson.dumps(d)

    def test_diagnosis_included_in_dict(self) -> None:
        snap = WatchSnapshot(
            timestamp=datetime(2026, 3, 18, tzinfo=timezone.utc),
            job_id="test-job",
            namespace="aiperf-benchmarks",
            phase="Running",
            diagnosis=DiagnosisResult(
                health="degraded",
                issues=[
                    DiagnosisIssue(
                        id="oom_restart",
                        severity="warning",
                        title="Worker pod OOM restart",
                        detail="OOMKilled",
                        impact="Progress reset",
                        suggested_fix="Increase memory",
                    )
                ],
            ),
        )
        d = snap.to_dict()
        assert d["diagnosis"]["health"] == "degraded"
        assert len(d["diagnosis"]["issues"]) == 1


class TestPodSnapshot:
    def test_from_raw_pod(self) -> None:
        raw = {
            "metadata": {"name": "aiperf-test-controller-0-0-abc"},
            "status": {
                "phase": "Running",
                "containerStatuses": [
                    {
                        "name": "control-plane",
                        "ready": True,
                        "restartCount": 0,
                        "state": {"running": {}},
                    }
                ],
            },
        }
        pod = PodSnapshot.from_raw(raw)
        assert pod.name == "aiperf-test-controller-0-0-abc"
        assert pod.status == "Running"
        assert pod.restarts == 0
        assert pod.role == "controller"
