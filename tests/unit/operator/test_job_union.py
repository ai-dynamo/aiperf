# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the unified-jobs-source helpers."""

from __future__ import annotations

import pytest

from aiperf.kubernetes.models import AIPerfJobInfo


def test_aiperfjobinfo_source_defaults_to_live():
    info = AIPerfJobInfo(
        name="j1", namespace="ns", phase="Running", job_id="j1",
    )
    assert info.source == "live"


def test_aiperfjobinfo_source_accepts_archived_and_both():
    for s in ("archived", "both"):
        info = AIPerfJobInfo(
            name="j1", namespace="ns", phase="Succeeded", job_id="j1", source=s,
        )
        assert info.source == s


def test_aiperfjobinfo_source_rejects_unknown():
    with pytest.raises(ValueError):
        AIPerfJobInfo(
            name="j1", namespace="ns", phase="Running", job_id="j1", source="bogus",
        )


from pathlib import Path
import orjson


def _write_summary(base: Path, ns: str, name: str, **extra) -> None:
    d = base / ns / name
    d.mkdir(parents=True, exist_ok=True)
    body = {
        "status": "Succeeded",
        "start_time": "2026-04-22T10:00:00Z",
        "end_time": "2026-04-22T10:45:00Z",
        "request_throughput": {"avg": 42.1, "unit": "requests/sec"},
        "request_latency": {"p99": 390.0, "unit": "ms"},
        "input_config": {
            "models": {"items": [{"name": "llama3-8b"}]},
            "endpoint": {"urls": ["http://llama3.svc:8000/v1"], "type": "chat"},
        },
    }
    body.update(extra)
    (d / "profile_export_aiperf.json").write_bytes(orjson.dumps(body))


def test_scan_pvc_jobs_empty_dir(tmp_path):
    from aiperf.operator.job_union import _scan_pvc_jobs
    assert _scan_pvc_jobs(tmp_path) == []


def test_scan_pvc_jobs_finds_summary_files(tmp_path):
    from aiperf.operator.job_union import _scan_pvc_jobs
    _write_summary(tmp_path, "aiperf-bench", "run-a")
    _write_summary(tmp_path, "ml-lab", "run-b", **{"status": "Failed"})
    entries = _scan_pvc_jobs(tmp_path)
    by_key = {(e.namespace, e.name): e for e in entries}
    assert (("aiperf-bench", "run-a")) in by_key
    assert (("ml-lab", "run-b")) in by_key
    a = by_key[("aiperf-bench", "run-a")]
    assert a.source == "archived"
    assert a.phase == "Succeeded"
    assert a.throughput_rps == 42.1
    assert a.latency_p99_ms == 390.0
    assert a.model == "llama3-8b"
    assert a.endpoint == "http://llama3.svc:8000/v1"
    assert a.progress_percent == 100.0
    b = by_key[("ml-lab", "run-b")]
    assert b.phase == "Failed"


def test_scan_pvc_jobs_skips_dirs_without_summary(tmp_path):
    from aiperf.operator.job_union import _scan_pvc_jobs
    (tmp_path / "aiperf-bench" / "no-summary").mkdir(parents=True)
    (tmp_path / "aiperf-bench" / "no-summary" / "other.txt").write_bytes(b"x")
    assert _scan_pvc_jobs(tmp_path) == []


def test_scan_pvc_jobs_missing_status_defaults_to_archived(tmp_path):
    from aiperf.operator.job_union import _scan_pvc_jobs
    d = tmp_path / "ns" / "job"
    d.mkdir(parents=True)
    (d / "profile_export_aiperf.json").write_bytes(orjson.dumps({
        "request_throughput": {"avg": 10.0, "unit": "requests/sec"},
    }))
    [e] = _scan_pvc_jobs(tmp_path)
    assert e.phase == "Archived"


def test_scan_pvc_jobs_filters_by_namespace(tmp_path):
    from aiperf.operator.job_union import _scan_pvc_jobs
    _write_summary(tmp_path, "ns-a", "j1")
    _write_summary(tmp_path, "ns-b", "j2")
    entries = _scan_pvc_jobs(tmp_path, namespace="ns-a")
    assert [e.namespace for e in entries] == ["ns-a"]
