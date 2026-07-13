# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit gates for the thin Python-to-runner telemetry-watch surface."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import orjson
import pytest
import yaml

from aiperf.cli_commands.watch import (
    _load_archive_report,
    _parse_watch_terminal,
    _terminal_failure,
    build_watch_request,
)


class _Installation:
    distribution_id = "blake3:" + "a" * 64
    capabilities = {
        "protocol_versions": [2],
        "supported_pairs": [["http", "telemetry_watch"]],
    }

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    def preflight_request(self, request: dict[str, Any]) -> None:
        assert request["run"]["workload"]["type"] == "telemetry_watch"
        self.requests.append(request)


def _document() -> dict[str, Any]:
    return {
        "schema_version": "2.0",
        "variables": {"cadence": 250_000_000},
        "run": {
            "identity": {"benchmark_id": "watch-test"},
            "artifact_target": "artifacts/watch-test",
            "transport": {"type": "http", "config": {}},
            "workload": {
                "type": "telemetry_watch",
                "config": {
                    "mode": "collect",
                    "duration_ns": 1_000_000_000,
                    "shutdown_timeout_ns": 2_000_000_000,
                    "sources": [
                        {
                            "id": "node-a",
                            "type": "prometheus_http",
                            "interval_ns": "{{ cadence }}",
                            "request_timeout_ns": 100_000_000,
                            "config": {
                                "url": "${WATCH_URL}",
                                "connect_timeout_ns": 50_000_000,
                                "redirects": "disabled",
                                "proxy": "disabled",
                                "accepted_formats": ["prometheus_text_0_0_4"],
                                "max_compressed_bytes": 1024,
                                "max_decompressed_bytes": 4096,
                            },
                        }
                    ],
                    "archive": {
                        "target": "archives/watch-test",
                        "local_spool": "spool/watch-test",
                        "spool_quota_bytes": 1_000_000,
                        "spool_quota_files": 1000,
                        "required": True,
                        "writer": {"type": "parquet_archive_v1", "config": {}},
                        "store_access": {
                            "type": "local_filesystem",
                            "config": {},
                        },
                        "rotation": {"type": "rows_bytes_age", "config": {}},
                        "admission": {"type": "primary_durable", "config": {}},
                        "recovery": {"type": "create_new", "config": {}},
                        "archive_key": {"type": "test_key", "config": {}},
                        "raw_body": {"type": "none", "config": {}},
                    },
                },
            },
            "resources": {},
        },
    }


def test_watch_config_expands_and_projects_exact_v2_envelope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WATCH_URL", "http://127.0.0.1:9090/metrics")
    path = tmp_path / "watch.yaml"
    path.write_text(yaml.safe_dump(_document()), encoding="utf-8")
    installation = _Installation()

    request = build_watch_request(
        path, installation, operation="execute"  # type: ignore[arg-type]
    )

    assert request["protocol_version"] == 2
    assert request["operation"] == "execute"
    assert request["expected_distribution_id"] == installation.distribution_id
    run = request["run"]
    assert Path(run["artifact_target"]).is_absolute()
    source = run["workload"]["config"]["sources"][0]
    assert source["interval_ns"] == 250_000_000
    assert source["config"]["url"] == "http://127.0.0.1:9090/metrics"
    archive = run["workload"]["config"]["archive"]
    assert Path(archive["local_spool"]).is_absolute()
    assert archive["target"] == (tmp_path / "archives/watch-test").as_uri()
    assert installation.requests == [request]


def test_watch_config_rejects_non_watch_and_unknown_outer_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WATCH_URL", "http://127.0.0.1:9090/metrics")
    document = _document()
    document["unexpected"] = True
    path = tmp_path / "watch.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown watch config fields"):
        build_watch_request(path, _Installation(), operation="validate")  # type: ignore[arg-type]

    document = _document()
    document["run"]["workload"]["type"] = "scheduled"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="workload.type='telemetry_watch'"):
        build_watch_request(path, _Installation(), operation="validate")  # type: ignore[arg-type]


def test_watch_terminal_is_bound_to_execution_identity() -> None:
    terminal = {
        "protocol_version": 2,
        "event": "run_terminal",
        "distribution_id": _Installation.distribution_id,
        "benchmark_id": "watch-test",
        "success": True,
        "report_path": "/tmp/native-v2.json",
    }
    completed = subprocess.CompletedProcess(
        ["aiperf-runner"], 0, stdout=orjson.dumps(terminal) + b"\n", stderr=b""
    )
    assert (
        _parse_watch_terminal(
            completed,
            benchmark_id="watch-test",
            distribution_id=_Installation.distribution_id,
        )
        == terminal
    )
    terminal["distribution_id"] = "blake3:" + "b" * 64
    completed = subprocess.CompletedProcess(
        ["aiperf-runner"], 0, stdout=orjson.dumps(terminal), stderr=b""
    )
    with pytest.raises(ValueError, match="distribution_id"):
        _parse_watch_terminal(
            completed,
            benchmark_id="watch-test",
            distribution_id=_Installation.distribution_id,
        )


def test_watch_loads_only_typed_report_inside_artifact_target(tmp_path: Path) -> None:
    report = tmp_path / "native-v2.json"
    report.write_bytes(
        orjson.dumps(
            {
                "schema_version": "2.0",
                "telemetry_archive": {
                    "schema_version": "1.0",
                    "archive_id": "018f84a7-1f3c-7c21-8be2-7e8dbf9536b1",
                    "state": "locally_finalized",
                },
            }
        )
    )
    archive = _load_archive_report({"report_path": str(report)}, tmp_path)
    assert archive["state"] == "locally_finalized"

    outside = tmp_path.parent / "native-v2.json"
    with pytest.raises(ValueError, match="escaped"):
        _load_archive_report({"report_path": str(outside)}, tmp_path)


def test_watch_failure_surfaces_typed_diagnostic_without_a_report() -> None:
    terminal = {
        "success": False,
        "stage": "reporting",
        "errors": [
            {
                "code": "archive_remote_finalization_failed",
                "message": "remote archive unavailable",
            }
        ],
        "diagnostic_artifacts": [
            {
                "kind": "archive_failure_diagnostic",
                "relative_path": "archive-failure-diagnostic.json",
                "content_hash": "blake3:" + "a" * 64,
            }
        ],
    }

    detail = _terminal_failure(terminal, b"")

    assert "remote archive unavailable" in detail
    assert "archive-failure-diagnostic.json" in detail
    assert "native-v2.json" not in detail
