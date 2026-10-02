# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import orjson
from fastapi import FastAPI
from fastapi.testclient import TestClient

from aiperf.common.enums import OptimizationDirection, SweepType
from aiperf.operator.routers.sweeps import create_sweeps_router
from aiperf.operator.sweep_union import SweepRecord


def _client_with(api: object | None, base_dir: Path) -> TestClient:
    holder: list = [api]
    app = FastAPI()
    app.include_router(create_sweeps_router(holder, base_dir))
    return TestClient(app)


def _seed_sweep_epoch(base_dir: Path, namespace: str, name: str, epoch: str) -> Path:
    from aiperf.operator.results_layout import write_sweep_latest

    sweep_dir = base_dir / namespace / "sweeps" / name / epoch
    sweep_dir.mkdir(parents=True, exist_ok=True)
    write_sweep_latest(base_dir, namespace, name, epoch)
    return sweep_dir


def _live_record(name: str = "s1") -> SweepRecord:
    return SweepRecord(
        namespace="bench",
        name=name,
        source="live",
        phase="Running",
        total_variations=4,
        completed_runs=1,
        failed_runs=0,
        age_seconds=10,
        model="m",
        raw_spec={
            "benchmark": {
                "models": {"items": [{"name": "m"}]},
                "endpoint": {"urls": ["http://x:8000/v1/chat/completions"]},
                "datasets": [{"name": "main", "type": "synthetic"}],
                "phases": [
                    {
                        "name": "profiling",
                        "type": "concurrency",
                        "concurrency": 1,
                        "requests": 1,
                    }
                ],
            },
            "sweep": {
                "type": SweepType.GRID,
                "parameters": {"phases.profiling.concurrency": [1, 2, 4, 8]},
            },
        },
        raw_status={
            "phase": "Running",
            "totalVariations": 4,
            "completedRuns": 1,
            "failedRuns": 0,
        },
    )


def test_list_returns_503_when_api_missing(tmp_path: Path) -> None:
    c = _client_with(None, tmp_path)
    r = c.get("/api/v1/sweeps")
    assert r.status_code == 503


def test_list_returns_records(tmp_path: Path) -> None:
    api = MagicMock()
    with patch(
        "aiperf.operator.routers.sweeps.list_all_sweeps",
        AsyncMock(return_value=[_live_record()]),
    ):
        c = _client_with(api, tmp_path)
        r = c.get("/api/v1/sweeps")
    assert r.status_code == 200
    body = r.json()
    assert len(body["sweeps"]) == 1
    assert body["sweeps"][0]["name"] == "s1"
    assert body["sweeps"][0]["source"] == "live"


def test_detail_404_when_missing(tmp_path: Path) -> None:
    api = MagicMock()
    with patch(
        "aiperf.operator.routers.sweeps.find_any_sweep",
        AsyncMock(return_value=None),
    ):
        c = _client_with(api, tmp_path)
        r = c.get("/api/v1/sweeps/bench/nope")
    assert r.status_code == 404


def test_detail_returns_adaptive_search_spec_summary_from_live(tmp_path: Path) -> None:
    api = MagicMock()
    rec = _live_record("adaptive-search-smoke")
    rec.raw_spec = {
        "benchmark": {
            "models": {"items": [{"name": "mock"}]},
            "endpoint": {"urls": ["http://x:8000/v1/chat/completions"]},
            "datasets": [{"name": "main", "type": "synthetic"}],
            "phases": [
                {
                    "name": "profiling",
                    "type": "concurrency",
                    "concurrency": 1,
                    "requests": 1,
                }
            ],
        },
        "sweep": {
            "type": SweepType.ADAPTIVE_SEARCH,
            "searchSpace": [
                {
                    "path": "phases.profiling.concurrency",
                    "lo": 1,
                    "hi": 40,
                    "kind": "int",
                }
            ],
            "objectives": [
                {
                    "metric": "output_token_throughput",
                    "stat": "avg",
                    "direction": OptimizationDirection.MAXIMIZE,
                }
            ],
            "maxIterations": 5,
            "nInitialPoints": 2,
        },
    }
    with (
        patch(
            "aiperf.operator.routers.sweeps.find_any_sweep",
            AsyncMock(return_value=rec),
        ),
        patch(
            "aiperf.operator.routers.sweeps.list_all_jobs",
            AsyncMock(return_value=[]),
        ),
    ):
        c = _client_with(api, tmp_path)
        r = c.get("/api/v1/sweeps/bench/adaptive-search-smoke")
    assert r.status_code == 200
    body = r.json()
    assert body["spec_summary"]["sweep_type"] == "adaptive_search"


def test_detail_returns_spec_summary_from_live(tmp_path: Path) -> None:
    api = MagicMock()
    rec = _live_record()
    with (
        patch(
            "aiperf.operator.routers.sweeps.find_any_sweep",
            AsyncMock(return_value=rec),
        ),
        patch(
            "aiperf.operator.routers.sweeps.list_all_jobs",
            AsyncMock(return_value=[]),
        ),
    ):
        c = _client_with(api, tmp_path)
        r = c.get("/api/v1/sweeps/bench/s1")
    assert r.status_code == 200
    body = r.json()
    assert body["sweep"]["name"] == "s1"
    assert body["spec_summary"]["sweep_type"] == "grid"
    dim_names = [d["name"] for d in body["spec_summary"]["dimensions"]]
    assert "concurrency" in dim_names


def test_historical_detail_never_attaches_current_live_children(tmp_path: Path) -> None:
    """A pinned sweep must fetch its child manifest from that sweep epoch only."""
    api = MagicMock()
    historical = _live_record()
    historical.source = "archived"
    historical.aggregate_doc = {
        "phase": "Succeeded",
        "specSnapshot": historical.raw_spec,
    }
    current_child = MagicMock()
    current_child.namespace = "bench"
    current_child.sweep_name = "s1"
    current_child.model_dump.return_value = {
        "name": "s1-v00",
        "namespace": "bench",
        "summary": {"request_throughput": {"avg": 999}},
    }
    with (
        patch(
            "aiperf.operator.routers.sweeps.find_any_sweep",
            AsyncMock(return_value=historical),
        ),
        patch(
            "aiperf.operator.routers.sweeps.list_all_jobs",
            AsyncMock(return_value=[current_child]),
        ) as list_jobs,
        patch(
            "aiperf.operator.routers.sweeps.fetch_sweep_pod_summaries",
            AsyncMock(return_value=[]),
        ),
    ):
        c = _client_with(api, tmp_path)
        response = c.get("/api/v1/sweeps/bench/s1?epoch=1714069323")

    assert response.status_code == 200
    assert response.json()["children"] == []
    list_jobs.assert_not_awaited()


def test_historical_sweep_config_uses_snapshot_not_current_cr(tmp_path: Path) -> None:
    """Relaunching a pinned epoch must never silently load the current spec."""
    api = MagicMock()
    historical = _live_record()
    historical.source = "archived"
    historical.raw_spec = {"benchmark": {"models": {"items": [{"name": "CURRENT"}]}}}
    historical.aggregate_doc = {
        "phase": "Succeeded",
        "specSnapshot": {"benchmark": {"models": {"items": [{"name": "OLD"}]}}},
    }
    with patch(
        "aiperf.operator.routers.sweeps.find_any_sweep",
        AsyncMock(return_value=historical),
    ) as find:
        c = _client_with(api, tmp_path)
        response = c.get("/api/v1/sweeps/bench/s1/config?epoch=1714069323")

    assert response.status_code == 200
    assert response.json()["spec"]["benchmark"]["models"]["items"][0]["name"] == "OLD"
    assert find.await_args.kwargs["epoch"] == "1714069323"


def test_historical_sweep_config_without_snapshot_never_uses_current_cr(
    tmp_path: Path,
) -> None:
    api = MagicMock()
    historical = _live_record()
    historical.source = "archived"
    historical.raw_spec = {"benchmark": {"models": {"items": [{"name": "CURRENT"}]}}}
    historical.aggregate_doc = {"phase": "Succeeded"}
    with patch(
        "aiperf.operator.routers.sweeps.find_any_sweep",
        AsyncMock(return_value=historical),
    ):
        c = _client_with(api, tmp_path)
        response = c.get("/api/v1/sweeps/bench/s1/config?epoch=1714069323")

    assert response.status_code == 404


def test_detail_lists_children_from_requested_namespace(tmp_path: Path) -> None:
    api = MagicMock()
    rec = _live_record()
    list_jobs = AsyncMock(return_value=[])
    with (
        patch(
            "aiperf.operator.routers.sweeps.find_any_sweep",
            AsyncMock(return_value=rec),
        ),
        patch("aiperf.operator.routers.sweeps.list_all_jobs", list_jobs),
    ):
        c = _client_with(api, tmp_path)
        r = c.get("/api/v1/sweeps/bench/s1")

    assert r.status_code == 200
    list_jobs.assert_awaited_once_with(
        api, tmp_path, all_namespaces=False, namespace="bench"
    )


def test_detail_archived_uses_synthesized_status(tmp_path: Path) -> None:
    sweep_dir = tmp_path / "bench" / "sweeps" / "s1"
    sweep_dir.mkdir(parents=True)
    (sweep_dir / "aggregate.json").write_text(
        json.dumps(
            {
                "phase": "Succeeded",
                "totalVariations": 4,
                "completedRuns": 4,
                "failedRuns": 0,
                "completedAt": "2026-04-25T01:00:00Z",
                "specSummary": {
                    "sweep_type": "grid",
                    "dimensions": [{"name": "concurrency", "values": [1, 2, 4, 8]}],
                },
                "model": "m",
            }
        )
    )
    api = MagicMock()
    rec = SweepRecord(
        namespace="bench",
        name="s1",
        source="archived",
        phase="Succeeded",
        total_variations=4,
        completed_runs=4,
        failed_runs=0,
        age_seconds=999,
        model="m",
        aggregate_path=str(sweep_dir / "aggregate.json"),
        aggregate_doc={
            "phase": "Succeeded",
            "totalVariations": 4,
            "completedRuns": 4,
            "failedRuns": 0,
            "completedAt": "2026-04-25T01:00:00Z",
            "specSummary": {
                "sweep_type": "grid",
                "dimensions": [{"name": "concurrency", "values": [1, 2, 4, 8]}],
            },
            "model": "m",
        },
    )
    with (
        patch(
            "aiperf.operator.routers.sweeps.find_any_sweep",
            AsyncMock(return_value=rec),
        ),
        patch(
            "aiperf.operator.routers.sweeps.list_all_jobs",
            AsyncMock(return_value=[]),
        ),
    ):
        c = _client_with(api, tmp_path)
        r = c.get("/api/v1/sweeps/bench/s1")
    assert r.status_code == 200
    body = r.json()
    assert body["sweep"]["source"] == "archived"
    assert body["status"]["phase"] == "Succeeded"
    assert body["status"]["completedAt"] == "2026-04-25T01:00:00Z"


def test_cells_archived_reads_per_cell_aggregates(tmp_path: Path) -> None:
    sweep_dir = tmp_path / "bench" / "sweeps" / "s1"
    sweep_dir.mkdir(parents=True)
    (sweep_dir / "aggregate.json").write_text(
        json.dumps(
            {
                "phase": "Succeeded",
                "totalVariations": 2,
                "completedRuns": 4,
                "failedRuns": 0,
                "completedAt": "2026-04-25T01:00:00Z",
                "specSummary": {
                    "sweep_type": "grid",
                    "dimensions": [{"name": "concurrency", "values": [8, 32]}],
                },
                "per_cell_aggregates": [
                    {
                        "variation_index": 0,
                        "variation_label": "concurrency-8",
                        "values": {"concurrency": 8},
                        "trials_completed": 2,
                        "trials_failed": 0,
                        "metrics": {"request_throughput": {"avg": 100.0, "p99": 110.0}},
                        "children": [
                            {
                                "namespace": "bench",
                                "name": "ch-0-0",
                                "trial_index": 0,
                                "phase": "Succeeded",
                            },
                            {
                                "namespace": "bench",
                                "name": "ch-0-1",
                                "trial_index": 1,
                                "phase": "Succeeded",
                            },
                        ],
                    },
                    {
                        "variation_index": 1,
                        "variation_label": "concurrency-32",
                        "values": {"concurrency": 32},
                        "trials_completed": 2,
                        "trials_failed": 0,
                        "metrics": {"request_throughput": {"avg": 280.0, "p99": 300.0}},
                        "children": [
                            {
                                "namespace": "bench",
                                "name": "ch-1-0",
                                "trial_index": 0,
                                "phase": "Succeeded",
                            },
                            {
                                "namespace": "bench",
                                "name": "ch-1-1",
                                "trial_index": 1,
                                "phase": "Succeeded",
                            },
                        ],
                    },
                ],
            }
        )
    )
    api = MagicMock()
    rec = SweepRecord(
        namespace="bench",
        name="s1",
        source="archived",
        phase="Succeeded",
        total_variations=2,
        completed_runs=4,
        failed_runs=0,
        age_seconds=999,
        model="m",
        aggregate_path=str(sweep_dir / "aggregate.json"),
        aggregate_doc=json.loads((sweep_dir / "aggregate.json").read_text()),
    )
    with patch(
        "aiperf.operator.routers.sweeps.find_any_sweep",
        AsyncMock(return_value=rec),
    ):
        c = _client_with(api, tmp_path)
        r = c.get("/api/v1/sweeps/bench/s1/cells")
    assert r.status_code == 200
    body = r.json()
    assert body["source"] == "archived"
    assert len(body["cells"]) == 2
    assert body["cells"][0]["metrics"]["request_throughput"]["avg"] == 100.0
    assert body["cells"][1]["values"]["concurrency"] == 32


def test_cells_live_no_aggregate_returns_empty_with_dimensions(tmp_path: Path) -> None:
    api = MagicMock()
    rec = _live_record()
    with (
        patch(
            "aiperf.operator.routers.sweeps.find_any_sweep",
            AsyncMock(return_value=rec),
        ),
        patch(
            "aiperf.operator.routers.sweeps._cells_from_live_children",
            AsyncMock(return_value=[]),
        ),
    ):
        c = _client_with(api, tmp_path)
        r = c.get("/api/v1/sweeps/bench/s1/cells")
    assert r.status_code == 200
    body = r.json()
    assert body["source"] == "live"
    assert body["cells"] == []
    assert [d["name"] for d in body["dimensions"]] == ["concurrency"]


def test_cells_live_child_lookup_scopes_to_requested_namespace(tmp_path: Path) -> None:
    api = MagicMock()
    rec = _live_record()
    list_jobs = AsyncMock(return_value=[])
    with (
        patch(
            "aiperf.operator.routers.sweeps.find_any_sweep",
            AsyncMock(return_value=rec),
        ),
        patch("aiperf.operator.routers.sweeps.list_all_jobs", list_jobs),
    ):
        c = _client_with(api, tmp_path)
        r = c.get("/api/v1/sweeps/bench/s1/cells")

    assert r.status_code == 200
    list_jobs.assert_awaited_once_with(
        api, tmp_path, all_namespaces=False, namespace="bench"
    )


def test_cells_404_when_neither_present(tmp_path: Path) -> None:
    api = MagicMock()
    with patch(
        "aiperf.operator.routers.sweeps.find_any_sweep",
        AsyncMock(return_value=None),
    ):
        c = _client_with(api, tmp_path)
        r = c.get("/api/v1/sweeps/bench/nope/cells")
    assert r.status_code == 404


def test_sweep_artifacts_list_scopes_to_epoch_root_and_aggregate_dir(
    tmp_path: Path,
) -> None:
    api = MagicMock()
    epoch = "1714150923"
    sweep_dir = _seed_sweep_epoch(tmp_path, "bench", "s1", epoch)
    (sweep_dir / "aggregate.json").write_bytes(b'{"ok": true}')
    (sweep_dir / "children.json").write_bytes(b'{"children": []}')
    aggregate_dir = sweep_dir / "sweep_aggregate"
    aggregate_dir.mkdir()
    (aggregate_dir / "sweep_results.csv").write_bytes(b"variation,throughput\n0,10\n")
    child_dir = sweep_dir / "s1-v00"
    child_dir.mkdir()
    (child_dir / "profile_export_aiperf.json").write_bytes(b'{"child": true}')

    c = _client_with(api, tmp_path)
    r = c.get(f"/api/v1/sweeps/bench/s1/epochs/{epoch}/artifacts")

    assert r.status_code == 200
    body = r.json()
    assert body["namespace"] == "bench"
    assert body["job_id"] == "s1"
    assert [f["name"] for f in body["files"]] == [
        "aggregate.json",
        "children.json",
        "sweep_aggregate/sweep_results.csv",
    ]


def test_sweep_artifacts_zip_matches_list_scope(tmp_path: Path) -> None:
    api = MagicMock()
    epoch = "1714150923"
    sweep_dir = _seed_sweep_epoch(tmp_path, "bench", "s1", epoch)
    (sweep_dir / "aggregate.json").write_bytes(b'{"ok": true}')
    aggregate_dir = sweep_dir / "sweep_aggregate"
    aggregate_dir.mkdir()
    (aggregate_dir / "sweep_results.csv").write_bytes(b"variation,throughput\n0,10\n")
    child_dir = sweep_dir / "s1-v00"
    child_dir.mkdir()
    (child_dir / "profile_export_aiperf.json").write_bytes(b'{"child": true}')

    c = _client_with(api, tmp_path)
    r = c.get(f"/api/v1/sweeps/bench/s1/epochs/{epoch}/artifacts.zip")

    assert r.status_code == 200
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        assert sorted(zf.namelist()) == [
            "aggregate.json",
            "sweep_aggregate/sweep_results.csv",
        ]


def test_sweep_artifacts_list_and_zip_skip_symlink_escape(tmp_path: Path) -> None:
    api = MagicMock()
    epoch = "1714150923"
    sweep_dir = _seed_sweep_epoch(tmp_path, "bench", "s1", epoch)
    (sweep_dir / "aggregate.json").write_bytes(b'{"ok": true}')
    outside_file = tmp_path / "bench" / "outside.json"
    outside_file.write_bytes(b'{"secret": true}')
    (sweep_dir / "outside-link.json").symlink_to(outside_file)

    c = _client_with(api, tmp_path)
    list_resp = c.get(f"/api/v1/sweeps/bench/s1/epochs/{epoch}/artifacts")
    zip_resp = c.get(f"/api/v1/sweeps/bench/s1/epochs/{epoch}/artifacts.zip")

    assert list_resp.status_code == 200
    assert [f["name"] for f in list_resp.json()["files"]] == ["aggregate.json"]
    assert zip_resp.status_code == 200
    with zipfile.ZipFile(io.BytesIO(zip_resp.content)) as zf:
        assert zf.namelist() == ["aggregate.json"]


def test_sweep_artifact_download_serves_root_and_aggregate_files(
    tmp_path: Path,
) -> None:
    api = MagicMock()
    epoch = "1714150923"
    sweep_dir = _seed_sweep_epoch(tmp_path, "bench", "s1", epoch)
    (sweep_dir / "aggregate.json").write_bytes(b'{"root": true}')
    aggregate_dir = sweep_dir / "sweep_aggregate"
    aggregate_dir.mkdir()
    (aggregate_dir / "sweep_results.csv").write_bytes(b"variation,throughput\n0,10\n")

    c = _client_with(api, tmp_path)
    root = c.get(f"/api/v1/sweeps/bench/s1/epochs/{epoch}/artifacts/aggregate.json")
    nested = c.get(
        f"/api/v1/sweeps/bench/s1/epochs/{epoch}/artifacts/"
        "sweep_aggregate/sweep_results.csv"
    )

    assert root.status_code == 200
    assert root.content == b'{"root": true}'
    assert root.headers["x-filename"] == "aggregate.json"
    assert nested.status_code == 200
    assert nested.content == b"variation,throughput\n0,10\n"
    assert nested.headers["x-filename"] == "sweep_results.csv"


def test_sweep_artifact_download_rejects_traversal(tmp_path: Path) -> None:
    api = MagicMock()
    epoch = "1714150923"
    _seed_sweep_epoch(tmp_path, "bench", "s1", epoch)
    (tmp_path / "bench" / "secret.json").write_bytes(b'{"secret": true}')

    c = _client_with(api, tmp_path)
    r = c.get(f"/api/v1/sweeps/bench/s1/epochs/{epoch}/artifacts/..%2Fsecret.json")

    assert r.status_code in (404, 422)


def test_sweep_artifacts_missing_epoch_and_file_return_404(tmp_path: Path) -> None:
    api = MagicMock()
    epoch = "1714150923"
    _seed_sweep_epoch(tmp_path, "bench", "s1", epoch)

    c = _client_with(api, tmp_path)
    missing_epoch = c.get("/api/v1/sweeps/bench/s1/epochs/1714064523/artifacts")
    missing_file = c.get(f"/api/v1/sweeps/bench/s1/epochs/{epoch}/artifacts/nope.json")

    assert missing_epoch.status_code == 404
    assert missing_file.status_code == 404


def test_sweep_artifacts_invalid_epoch_returns_422(tmp_path: Path) -> None:
    api = MagicMock()
    c = _client_with(api, tmp_path)

    r = c.get("/api/v1/sweeps/bench/s1/epochs/not-an-epoch/artifacts")

    assert r.status_code == 422


def test_sweep_profile_export_quick_route_supports_json_and_csv(tmp_path: Path) -> None:
    api = MagicMock()
    epoch = "1714150923"
    sweep_dir = _seed_sweep_epoch(tmp_path, "bench", "s1", epoch)
    aggregate_dir = sweep_dir / "sweep_aggregate"
    aggregate_dir.mkdir()
    (aggregate_dir / "profile_export_aiperf_aggregate.json").write_bytes(
        orjson.dumps({"sweep": True})
    )
    (aggregate_dir / "profile_export_aiperf_aggregate.csv").write_bytes(
        b"metric,value\nthroughput,10\n"
    )

    c = _client_with(api, tmp_path)
    json_resp = c.get(
        f"/api/v1/sweeps/bench/s1/epochs/{epoch}/artifacts/profile_export?format=json"
    )
    csv_resp = c.get(
        f"/api/v1/sweeps/bench/s1/epochs/{epoch}/artifacts/profile_export?format=csv"
    )

    assert json_resp.status_code == 200
    assert json_resp.headers["content-type"].startswith("application/json")
    assert orjson.loads(json_resp.content) == {"sweep": True}
    assert csv_resp.status_code == 200
    assert csv_resp.headers["content-type"].startswith("text/csv")
    assert csv_resp.content == b"metric,value\nthroughput,10\n"


def test_sweep_profile_export_quick_route_rejects_csv_symlink_escape(
    tmp_path: Path,
) -> None:
    api = MagicMock()
    epoch = "1714150923"
    sweep_dir = _seed_sweep_epoch(tmp_path, "bench", "s1", epoch)
    aggregate_dir = sweep_dir / "sweep_aggregate"
    aggregate_dir.mkdir()
    outside_file = tmp_path / "outside-profile.csv"
    outside_file.write_bytes(b"secret,value\nleaked,1\n")
    (aggregate_dir / "profile_export_aiperf_aggregate.csv").symlink_to(outside_file)

    c = _client_with(api, tmp_path)
    response = c.get(
        f"/api/v1/sweeps/bench/s1/epochs/{epoch}/artifacts/profile_export?format=csv"
    )

    assert response.status_code == 404
