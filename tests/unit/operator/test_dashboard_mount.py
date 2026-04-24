# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for operator dashboard mount (Dash inside results server)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import orjson
import zstandard

from aiperf.operator.dashboard_mount import DashboardProxy, build_dashboard


def _create_zst_run(base: Path, namespace: str, job_id: str) -> Path:
    """Create a minimal zst run directory on a mock PVC."""
    run_dir = base / namespace / job_id
    run_dir.mkdir(parents=True)
    cctx = zstandard.ZstdCompressor()

    jsonl_line = orjson.dumps(
        {
            "metadata": {
                "session_num": 0,
                "x_request_id": "req-1",
                "x_correlation_id": None,
                "conversation_id": None,
                "turn_index": None,
                "request_start_ns": 1000000000,
                "request_end_ns": 2000000000,
                "worker_id": "worker-0",
                "record_processor_id": "rp-0",
                "benchmark_phase": "main",
            },
            "metrics": {
                "ttft": {"value": 0.1, "unit": "s"},
                "itl_avg": {"value": 0.05, "unit": "s"},
                "e2e_latency": {"value": 1.0, "unit": "s"},
            },
            "trace_data": None,
            "error": None,
        }
    )
    (run_dir / "profile_export.jsonl.zst").write_bytes(
        cctx.compress(jsonl_line + b"\n")
    )
    agg = orjson.dumps(
        {
            "metrics": {
                "request_throughput": {
                    "avg": 100.0,
                    "min": 90.0,
                    "max": 110.0,
                    "p50": 100.0,
                    "p90": 108.0,
                    "p95": 109.0,
                    "p99": 110.0,
                    "unit": "req/s",
                    "tag": "request_throughput",
                }
            },
            "input_config": {},
        }
    )
    (run_dir / "profile_export_aiperf.json.zst").write_bytes(cctx.compress(agg))
    return run_dir


class TestDashboardProxy:
    """Tests for the DashboardProxy WSGI wrapper."""

    def test_proxy_delegates_to_inner_app(self) -> None:
        inner = MagicMock()
        inner.return_value = [b"response"]
        proxy = DashboardProxy(inner)
        environ = {"REQUEST_METHOD": "GET"}
        start_response = MagicMock()
        result = proxy(environ, start_response)
        inner.assert_called_once_with(environ, start_response)
        assert result == [b"response"]

    def test_proxy_swap(self) -> None:
        old_app = MagicMock(return_value=[b"old"])
        new_app = MagicMock(return_value=[b"new"])
        proxy = DashboardProxy(old_app)
        proxy.app = new_app
        result = proxy({}, MagicMock())
        new_app.assert_called_once()
        old_app.assert_not_called()
        assert result == [b"new"]


class TestBuildDashboard:
    """Tests for the build_dashboard factory function."""

    def test_returns_none_for_empty_pvc(self, tmp_path: Path) -> None:
        dash_app, run_count = build_dashboard(tmp_path)
        assert dash_app is None
        assert run_count == 0

    def test_returns_dash_app_for_valid_runs(self, tmp_path: Path) -> None:
        _create_zst_run(tmp_path, "default", "job-1")
        _create_zst_run(tmp_path, "default", "job-2")
        dash_app, run_count = build_dashboard(tmp_path)
        assert dash_app is not None
        assert run_count == 2
        assert hasattr(dash_app, "server")

    def test_returns_dash_app_for_single_run(self, tmp_path: Path) -> None:
        _create_zst_run(tmp_path, "default", "job-1")
        dash_app, run_count = build_dashboard(tmp_path)
        assert dash_app is not None
        assert run_count == 1
