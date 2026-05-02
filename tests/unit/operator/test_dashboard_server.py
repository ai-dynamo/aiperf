# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the standalone dashboard sidecar HTTP app."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient


def test_healthz_returns_200(tmp_path: Path) -> None:
    from aiperf.operator.dashboard_server import create_app

    app = create_app(results_dir=tmp_path)
    with TestClient(app) as client:
        resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
