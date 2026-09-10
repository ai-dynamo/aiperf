# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the operator's /api/v1/config/* endpoints."""

from __future__ import annotations

import importlib

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _client(monkeypatch: pytest.MonkeyPatch, **env: str) -> TestClient:
    for k in ("AIPERF_DASHBOARD_PROXY_ENABLED", "AIPERF_DASHBOARD_PORT"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)

    from aiperf.operator import environment as env_mod

    importlib.reload(env_mod)
    from aiperf.operator.routers import config as config_mod

    importlib.reload(config_mod)
    app = FastAPI()
    app.include_router(config_mod.create_config_router())
    return TestClient(app)


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({}, False),
        ({"AIPERF_DASHBOARD_PROXY_ENABLED": "1"}, True),
        ({"AIPERF_DASHBOARD_PROXY_ENABLED": "0"}, False),
    ],
)
def test_features_endpoint_reflects_dashboard_proxy_env(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, str], expected: bool
) -> None:
    client = _client(monkeypatch, **env)
    resp = client.get("/api/v1/config/features")
    assert resp.status_code == 200
    body = resp.json()
    assert body["dashboard_enabled"] is expected
