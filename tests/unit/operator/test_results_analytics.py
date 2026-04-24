# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the operator web UI results_analytics router.

Covers the ``/api/v1/config/{namespace}/{job_id}`` fallback chain — specifically
the live-CR spec fallback that keeps the dashboard hero's SLO chips working
for running jobs with no on-disk artifacts.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from aiperf.operator.results_db import ResultsDB
from aiperf.operator.routers import results_analytics as mod
from aiperf.operator.routers.results_analytics import create_results_analytics_router


@pytest.mark.asyncio
async def test_get_job_config_falls_back_to_live_cr_spec(tmp_path, monkeypatch):
    """When no file + no summary, config endpoint returns the live CR spec."""
    fake_cr = {
        "apiVersion": "aiperf.nvidia.com/v1alpha1",
        "metadata": {"name": "live-job", "namespace": "aiperf-bench"},
        "spec": {
            "benchmark": {
                "models": {"items": [{"name": "llama3-8b"}]},
                "endpoint": {"urls": ["http://llama3.svc:8000/v1"], "type": "chat"},
                "slos": {"time_to_first_token": 500},
            },
        },
    }

    async def fake_get_raw(api, namespace, name):
        if namespace == "aiperf-bench" and name == "live-job":
            return fake_cr
        return None

    monkeypatch.setattr(mod, "get_raw_aiperfjob", fake_get_raw, raising=False)

    api_holder: list = [object()]  # sentinel so the None-guard passes
    db = ResultsDB(tmp_path)
    router = create_results_analytics_router(lambda: db, tmp_path, api_holder)

    app = FastAPI()
    app.include_router(router)
    try:
        with TestClient(app) as client:
            resp = client.get("/api/v1/config/aiperf-bench/live-job")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["source"] == "cr"
        assert body["spec"]["benchmark"]["slos"] == {"time_to_first_token": 500}
    finally:
        db.close()


@pytest.mark.asyncio
async def test_get_job_config_returns_404_when_cr_missing(tmp_path, monkeypatch):
    """When all fallbacks miss (no file, no summary, no live CR), still 404."""

    async def fake_get_raw(api, namespace, name):
        return None

    monkeypatch.setattr(mod, "get_raw_aiperfjob", fake_get_raw, raising=False)

    api_holder: list = [object()]
    db = ResultsDB(tmp_path)
    router = create_results_analytics_router(lambda: db, tmp_path, api_holder)

    app = FastAPI()
    app.include_router(router)
    try:
        with TestClient(app) as client:
            resp = client.get("/api/v1/config/aiperf-bench/missing-job")
        assert resp.status_code == 404
    finally:
        db.close()
