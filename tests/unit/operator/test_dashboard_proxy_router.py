# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the dashboard-proxy reverse-proxy router mounted in results-server."""

from __future__ import annotations

import importlib

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _make_app(monkeypatch: pytest.MonkeyPatch, *, enabled: bool, port: int) -> FastAPI:
    if enabled:
        monkeypatch.setenv("AIPERF_DASHBOARD_PROXY_ENABLED", "1")
    else:
        monkeypatch.delenv("AIPERF_DASHBOARD_PROXY_ENABLED", raising=False)
    monkeypatch.setenv("AIPERF_DASHBOARD_PORT", str(port))

    from aiperf.operator import environment as env_mod

    importlib.reload(env_mod)
    from aiperf.operator.routers import dashboard_proxy

    importlib.reload(dashboard_proxy)

    app = FastAPI()
    app.include_router(dashboard_proxy.create_dashboard_proxy_router())
    return app


def test_proxy_returns_503_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _make_app(monkeypatch, enabled=False, port=0)
    with TestClient(app) as client:
        resp = client.get("/dashboard/")
    assert resp.status_code == 503
    assert b"disabled" in resp.content.lower()


def test_proxy_forwards_to_localhost_port(monkeypatch: pytest.MonkeyPatch) -> None:
    """Proxy passes method, path, and body to localhost:<port> and streams the response back."""
    app = _make_app(monkeypatch, enabled=True, port=8082)

    captured: dict[str, object] = {}

    class _FakeResp:
        status_code = 200
        headers = {"content-type": "application/json"}

        async def aiter_raw(self):
            yield b'{"hello": "world"}'

    class _FakeStream:
        async def __aenter__(self):
            captured["entered"] = True
            return _FakeResp()

        async def __aexit__(self, *_a):
            return None

    def _fake_stream(self, method, url, **kwargs):
        captured["method"] = method
        captured["url"] = url
        return _FakeStream()

    monkeypatch.setattr(httpx.AsyncClient, "stream", _fake_stream)

    with TestClient(app) as client:
        resp = client.get("/dashboard/foo/bar?x=1")

    assert resp.status_code == 200
    assert captured["method"] == "GET"
    assert captured["url"] == "http://localhost:8082/dashboard/foo/bar?x=1"
    assert b'"hello": "world"' in resp.content


def test_proxy_returns_503_when_upstream_unreachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _make_app(monkeypatch, enabled=True, port=8082)

    def _fake_stream(self, *_a, **_kw):
        raise httpx.ConnectError("upstream down")

    monkeypatch.setattr(httpx.AsyncClient, "stream", _fake_stream)

    with TestClient(app) as client:
        resp = client.get("/dashboard/")
    assert resp.status_code == 503
    assert b"unreachable" in resp.content.lower()
