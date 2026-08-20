# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the unified-jobs-source helpers."""

from __future__ import annotations

from pathlib import Path

import orjson
import pytest
from aiperf.operator.results_layout import write_latest

from aiperf.kubernetes.models import AIPerfJobInfo

_TEST_EPOCH = "1714064523"


def test_aiperfjobinfo_source_defaults_to_live():
    info = AIPerfJobInfo(
        name="j1",
        namespace="ns",
        phase="Running",
        job_id="j1",
    )
    assert info.source == "live"


def test_aiperfjobinfo_source_accepts_archived_and_both():
    for s in ("archived", "both"):
        info = AIPerfJobInfo(
            name="j1",
            namespace="ns",
            phase="Succeeded",
            job_id="j1",
            source=s,
        )
        assert info.source == s


def test_aiperfjobinfo_source_rejects_unknown():
    with pytest.raises(ValueError):
        AIPerfJobInfo(
            name="j1",
            namespace="ns",
            phase="Running",
            job_id="j1",
            source="bogus",
        )


def _write_summary(base: Path, ns: str, name: str, **extra) -> None:
    d = base / ns / name / _TEST_EPOCH
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
    write_latest(base, ns, name, _TEST_EPOCH)


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
    assert a.phase == "Archived"
    assert a.throughput_rps == 42.1
    assert a.latency_p99_ms == 390.0
    assert a.model == "llama3-8b"
    assert a.endpoint == "http://llama3.svc:8000/v1"
    assert a.progress_percent == 100.0
    b = by_key[("ml-lab", "run-b")]
    assert b.phase == "Archived"


def test_scan_pvc_jobs_normalizes_timezone_less_summary_timestamps(tmp_path):
    from aiperf.operator.job_union import _scan_pvc_jobs

    _write_summary(
        tmp_path,
        "ns",
        "archived-naive-time",
        start_time="2026-08-05T10:17:45.222190",
        end_time="2026-08-05T10:17:50.292792",
    )

    [entry] = _scan_pvc_jobs(tmp_path)

    assert entry.start_time == "2026-08-05T10:17:45.222190Z"
    assert entry.completion_time == "2026-08-05T10:17:50.292792Z"
    assert entry.created == "2026-08-05T10:17:45.222190Z"


def test_scan_pvc_jobs_skips_dirs_without_summary(tmp_path):
    from aiperf.operator.job_union import _scan_pvc_jobs

    run = tmp_path / "aiperf-bench" / "no-summary" / _TEST_EPOCH
    run.mkdir(parents=True)
    (run / "other.txt").write_bytes(b"x")
    write_latest(tmp_path, "aiperf-bench", "no-summary", _TEST_EPOCH)
    assert _scan_pvc_jobs(tmp_path) == []


def test_scan_pvc_jobs_finds_custom_prefixed_summary(tmp_path):
    from aiperf.operator.job_union import _scan_pvc_jobs

    run = tmp_path / "ns" / "custom" / _TEST_EPOCH
    run.mkdir(parents=True)
    (run / "job_spec.json").write_bytes(
        orjson.dumps({"benchmark": {"artifacts": {"prefix": "nightly"}}})
    )
    (run / "nightly.json").write_bytes(
        orjson.dumps({"request_throughput": {"avg": 9.0}})
    )
    write_latest(tmp_path, "ns", "custom", _TEST_EPOCH)

    [entry] = _scan_pvc_jobs(tmp_path)

    assert entry.name == "custom"
    assert entry.throughput_rps == 9.0


def test_scan_pvc_jobs_keeps_ready_csv_only_run_as_stub(tmp_path):
    from aiperf.kubernetes.results_sidecar import write_ready_marker
    from aiperf.operator.job_union import _scan_pvc_jobs

    run = tmp_path / "ns" / "csv-only" / _TEST_EPOCH
    run.mkdir(parents=True)
    (run / "profile_export_aiperf.csv").write_text("Metric,Value\nRequests,1\n")
    write_ready_marker(run)
    write_latest(tmp_path, "ns", "csv-only", _TEST_EPOCH)

    [entry] = _scan_pvc_jobs(tmp_path)

    assert entry.name == "csv-only"
    assert entry.phase == "Unknown"


def test_scan_pvc_jobs_missing_status_defaults_to_archived(tmp_path):
    from aiperf.operator.job_union import _scan_pvc_jobs

    d = tmp_path / "ns" / "job" / _TEST_EPOCH
    d.mkdir(parents=True)
    (d / "profile_export_aiperf.json").write_bytes(
        orjson.dumps(
            {
                "request_throughput": {"avg": 10.0, "unit": "requests/sec"},
            }
        )
    )
    write_latest(tmp_path, "ns", "job", _TEST_EPOCH)
    [e] = _scan_pvc_jobs(tmp_path)
    assert e.phase == "Archived"


def test_scan_pvc_jobs_filters_by_namespace(tmp_path):
    from aiperf.operator.job_union import _scan_pvc_jobs

    _write_summary(tmp_path, "ns-a", "j1")
    _write_summary(tmp_path, "ns-b", "j2")
    entries = _scan_pvc_jobs(tmp_path, namespace="ns-a")
    assert [e.namespace for e in entries] == ["ns-a"]


class _FakeCR:
    """Minimal stand-in for AIPerfJobInfo returned by list_aiperf_jobs."""


@pytest.mark.asyncio
async def test_list_all_jobs_cr_only(tmp_path, monkeypatch):
    from aiperf.operator import job_union

    async def fake_list(api, *, all_namespaces=True, namespace=None, **_):
        return [
            AIPerfJobInfo(
                name="live-a",
                namespace="ns",
                phase="Running",
                job_id="live-a",
            ),
        ]

    monkeypatch.setattr(job_union, "list_aiperf_jobs", fake_list)

    out = await job_union.list_all_jobs(api=None, results_dir=tmp_path)
    assert [e.name for e in out] == ["live-a"]
    assert out[0].source == "live"


@pytest.mark.asyncio
async def test_list_all_jobs_pvc_only(tmp_path, monkeypatch):
    from aiperf.operator import job_union

    async def fake_list(api, *, all_namespaces=True, namespace=None, **_):
        return []

    monkeypatch.setattr(job_union, "list_aiperf_jobs", fake_list)
    _write_summary(tmp_path, "ns", "archive-a")
    out = await job_union.list_all_jobs(api=None, results_dir=tmp_path)
    assert [e.name for e in out] == ["archive-a"]
    assert out[0].source == "archived"


@pytest.mark.asyncio
async def test_list_all_jobs_orders_archived_records_by_newest_timestamp(
    tmp_path,
    monkeypatch,
):
    from aiperf.operator import job_union

    async def fake_list(api, *, all_namespaces=True, namespace=None, **_):
        return []

    monkeypatch.setattr(job_union, "list_aiperf_jobs", fake_list)
    _write_summary(
        tmp_path,
        "ns",
        "a-old-archive",
        start_time="2026-04-22T10:00:00Z",
        end_time="2026-04-22T10:01:00Z",
    )
    _write_summary(
        tmp_path,
        "ns",
        "z-new-archive",
        start_time="2026-04-22T11:00:00Z",
        end_time="2026-04-22T11:01:00Z",
    )

    out = await job_union.list_all_jobs(api=None, results_dir=tmp_path)

    assert [e.name for e in out] == ["z-new-archive", "a-old-archive"]


@pytest.mark.asyncio
async def test_list_all_jobs_overlap_marks_both(tmp_path, monkeypatch):
    from aiperf.operator import job_union

    async def fake_list(api, *, all_namespaces=True, namespace=None, **_):
        return [
            AIPerfJobInfo(
                name="run-1",
                namespace="ns",
                phase="Succeeded",
                job_id="run-1",
                throughput_rps=77.7,
            ),
        ]

    monkeypatch.setattr(job_union, "list_aiperf_jobs", fake_list)
    _write_summary(tmp_path, "ns", "run-1")
    out = await job_union.list_all_jobs(api=None, results_dir=tmp_path)
    [entry] = out
    assert entry.source == "both"
    # CR wins for live fields
    assert entry.phase == "Succeeded"
    assert entry.throughput_rps == 77.7


@pytest.mark.asyncio
async def test_list_all_jobs_filters_both_sides_by_namespace(
    tmp_path,
    monkeypatch,
):
    from aiperf.operator import job_union

    async def fake_list(api, *, all_namespaces=True, namespace=None, **_):
        assert all_namespaces is False
        assert namespace == "ns-a"
        return [
            AIPerfJobInfo(
                name="live-a",
                namespace="ns-a",
                phase="Running",
                job_id="live-a",
            )
        ]

    monkeypatch.setattr(job_union, "list_aiperf_jobs", fake_list)
    _write_summary(tmp_path, "ns-a", "archive-a")
    _write_summary(tmp_path, "ns-b", "archive-b")
    out = await job_union.list_all_jobs(
        api=None,
        results_dir=tmp_path,
        all_namespaces=False,
        namespace="ns-a",
    )
    names = {e.name for e in out}
    assert names == {"live-a", "archive-a"}


@pytest.mark.asyncio
async def test_find_any_job_prefers_cr(tmp_path, monkeypatch):
    from aiperf.operator import job_union

    async def fake_find(api, name, namespace):
        return AIPerfJobInfo(
            name=name,
            namespace=namespace,
            phase="Running",
            job_id=name,
            throughput_rps=123.0,
        )

    monkeypatch.setattr(job_union, "find_aiperf_job", fake_find)
    _write_summary(tmp_path, "ns", "j1")
    info = await job_union.find_any_job(None, tmp_path, "ns", "j1")
    assert info is not None
    assert info.source == "both"
    assert info.throughput_rps == 123.0


@pytest.mark.asyncio
async def test_find_any_job_falls_back_to_pvc(tmp_path, monkeypatch):
    from aiperf.operator import job_union

    async def fake_find(api, name, namespace):
        return None

    monkeypatch.setattr(job_union, "find_aiperf_job", fake_find)
    _write_summary(tmp_path, "ns", "j1")
    info = await job_union.find_any_job(None, tmp_path, "ns", "j1")
    assert info is not None
    assert info.source == "archived"


@pytest.mark.asyncio
async def test_find_any_job_returns_none_when_neither(tmp_path, monkeypatch):
    from aiperf.operator import job_union

    async def fake_find(api, name, namespace):
        return None

    monkeypatch.setattr(job_union, "find_aiperf_job", fake_find)
    info = await job_union.find_any_job(None, tmp_path, "ns", "missing")
    assert info is None


# =============================================================================
# synthesize_status_from_summary
# =============================================================================


def _full_summary() -> dict:
    """A profile_export-shaped summary with every nested metric populated."""
    return {
        "status": "Succeeded",
        "start_time": "2026-04-22T10:00:00Z",
        "end_time": "2026-04-22T10:45:00Z",
        "request_count": 10000,
        "request_throughput": {"avg": 42.1, "unit": "requests/sec"},
        "request_latency": {"avg": 180.0, "p99": 390.0, "unit": "ms"},
        "time_to_first_token": {"avg": 45.5, "p99": 120.0, "unit": "ms"},
        "inter_token_latency": {"avg": 12.3, "p99": 30.0, "unit": "ms"},
        "output_token_throughput": {"avg": 2048.5, "unit": "tokens/sec"},
        "error_rate": 0.005,
    }


def test_synthesize_status_full_summary():
    from aiperf.operator.job_union import synthesize_status_from_summary

    status = synthesize_status_from_summary(
        "ns", "j1", _full_summary(), phase="Succeeded"
    )

    assert status["jobId"] == "j1"
    assert status["phase"] == "Succeeded"
    assert status["startTime"] == "2026-04-22T10:00:00Z"
    assert status["completionTime"] == "2026-04-22T10:45:00Z"
    assert status["currentPhase"] == "completed"
    assert status["workers"] == {"ready": 0, "total": 0}

    summary = status["summary"]
    # Each tag passes through as the full sub-dict — UI reads ``[tag][stat]``.
    assert summary["request_throughput"]["avg"] == 42.1
    assert summary["request_latency"]["avg"] == 180.0
    assert summary["request_latency"]["p99"] == 390.0
    assert summary["time_to_first_token"]["avg"] == 45.5
    assert summary["time_to_first_token"]["p99"] == 120.0
    assert summary["inter_token_latency"]["avg"] == 12.3
    assert summary["inter_token_latency"]["p99"] == 30.0
    assert summary["output_token_throughput"]["avg"] == 2048.5
    assert summary["total_requests"] == 10000
    assert summary["error_rate"] == 0.005

    phases = status["phases"]
    assert phases["benchmark"]["requestsCompleted"] == 10000
    assert phases["benchmark"]["requestsTotal"] == 10000
    assert phases["benchmark"]["requestsProgressPercent"] == 100


def test_synthesize_status_missing_metric_skipped():
    from aiperf.operator.job_union import synthesize_status_from_summary

    partial = {
        "status": "Succeeded",
        "request_throughput": {"avg": 42.1},
        # No request_latency, no ttft, no itl, no output_token_throughput
    }
    status = synthesize_status_from_summary("ns", "j1", partial)
    summary = status["summary"]
    assert summary["request_throughput"]["avg"] == 42.1
    # Absent metric tags must be OMITTED (not present as empty dicts).
    assert "request_latency" not in summary
    assert "time_to_first_token" not in summary
    assert "inter_token_latency" not in summary
    assert "output_token_throughput" not in summary


def test_synthesize_status_archived_phase_fallback():
    from aiperf.operator.job_union import synthesize_status_from_summary

    status = synthesize_status_from_summary(
        "ns", "j1", {"request_throughput": {"avg": 1.0}}
    )
    assert status["phase"] == "Archived"


def test_synthesize_status_conditions_passthrough():
    from aiperf.operator.job_union import synthesize_status_from_summary

    conds = [
        {"type": "Progressing", "status": "False", "reason": "Completed"},
        {"type": "Ready", "status": "True"},
    ]
    status = synthesize_status_from_summary(
        "ns", "j1", _full_summary(), conditions=conds
    )
    assert status["conditions"] == conds


def test_synthesize_status_conditions_synthesized():
    from aiperf.operator.job_union import synthesize_status_from_summary

    status = synthesize_status_from_summary(
        "ns", "j1", {"status": "Failed"}, phase="Failed"
    )
    [cond] = status["conditions"]
    assert cond["type"] == "Failed"
    assert cond["status"] == "True"
