# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for the operator dashboard proxy and results-server mount.

Focuses on:
- upstream failure contracts for dashboard-sidecar proxying
- request/response header filtering across the reverse-proxy boundary
- URL encoding, query preservation, and encoded traversal-shaped paths
- results-server route ordering so dashboard paths do not fall through to static UI

Out of scope: Dash plot rendering internals, covered by
``tests/unit/operator/test_dashboard_mount.py`` and dashboard-sidecar tests.
"""

from __future__ import annotations

import importlib
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import cast

import aiohttp
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pytest import param

# ================================================================
# Helpers
# ================================================================


@dataclass(slots=True)
class _CapturedProxyRequest:
    """Request observed at the fake dashboard sidecar boundary."""

    method: str | None = None
    url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    content: bytes | None = None
    closed: bool = False


@dataclass(slots=True)
class _FakeUpstreamResponse:
    """Minimal async-streaming response returned by the fake sidecar."""

    status: int = 200
    headers: dict[str, str] = field(default_factory=dict)
    chunks: list[bytes] = field(default_factory=lambda: [b"dashboard-ok"])

    @property
    def content(self) -> _FakeUpstreamContent:
        return _FakeUpstreamContent(self.chunks)


@dataclass(slots=True)
class _FakeUpstreamContent:
    """Stand-in for ``aiohttp.ClientResponse.content``."""

    chunks: list[bytes]

    async def iter_any(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            yield chunk


@dataclass(slots=True)
class _FakeUpstreamStream:
    """Async context manager matching ``aiohttp.ClientSession.request``."""

    response: _FakeUpstreamResponse

    async def __aenter__(self) -> _FakeUpstreamResponse:
        return self.response

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        del exc_type, exc, tb


def _make_dashboard_proxy_app(
    monkeypatch: pytest.MonkeyPatch,
    *,
    enabled: bool = True,
    port: int = 8082,
) -> FastAPI:
    """Create a minimal app with env-backed dashboard-proxy settings reloaded."""
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


def _install_fake_upstream(
    monkeypatch: pytest.MonkeyPatch,
    *,
    response: _FakeUpstreamResponse | None = None,
    error: Exception | None = None,
) -> _CapturedProxyRequest:
    """Capture proxied requests and return a deterministic fake sidecar response."""
    captured = _CapturedProxyRequest()
    upstream_response = response or _FakeUpstreamResponse()

    def _fake_request(
        self: aiohttp.ClientSession,
        method: str,
        url: str,
        **kwargs: object,
    ) -> _FakeUpstreamStream:
        del self
        if error is not None:
            raise error
        captured.method = method
        captured.url = url
        captured.headers = dict(cast(Mapping[str, str], kwargs["headers"]))
        captured.content = cast(bytes, kwargs["data"])
        return _FakeUpstreamStream(upstream_response)

    async def _fake_close(self: aiohttp.ClientSession) -> None:
        del self
        captured.closed = True

    monkeypatch.setattr(aiohttp.ClientSession, "request", _fake_request)
    monkeypatch.setattr(aiohttp.ClientSession, "close", _fake_close)
    return captured


def _make_results_server_app(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    enabled: bool = True,
) -> FastAPI:
    """Create the real results-server app without entering lifespan."""
    if enabled:
        monkeypatch.setenv("AIPERF_DASHBOARD_PROXY_ENABLED", "1")
    else:
        monkeypatch.delenv("AIPERF_DASHBOARD_PROXY_ENABLED", raising=False)
    monkeypatch.setenv("AIPERF_DASHBOARD_PORT", "8082")

    from aiperf.operator import environment as env_mod

    importlib.reload(env_mod)
    from aiperf.operator.routers import dashboard_proxy

    from aiperf.operator import results_server

    importlib.reload(dashboard_proxy)
    importlib.reload(results_server)
    return results_server.create_app(results_dir=tmp_path)


# ================================================================
# Upstream failures and disabled states
# ================================================================


class TestDashboardProxyUpstreamFailures:
    """Dashboard sidecar failures return a stable user-facing 503."""

    @pytest.mark.parametrize(
        "error",
        [
            param(aiohttp.ClientConnectionError("connection refused"), id="connect-error"),
            param(TimeoutError("dashboard sidecar timed out"), id="read-timeout"),
            param(aiohttp.ClientPayloadError("malformed upstream response"), id="remote-protocol"),
        ],
    )  # fmt: skip
    def test_proxy_client_error_returns_503_and_closes_client(
        self,
        monkeypatch: pytest.MonkeyPatch,
        error: Exception,
    ) -> None:
        app = _make_dashboard_proxy_app(monkeypatch)
        captured = _install_fake_upstream(monkeypatch, error=error)

        with TestClient(app) as client:
            response = client.get("/dashboard/assets/index.js")

        assert response.status_code == 503
        assert response.text == "Dashboard sidecar is unreachable."
        assert captured.closed is True

    def test_proxy_enabled_with_nonpositive_port_returns_503_without_upstream_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app = _make_dashboard_proxy_app(monkeypatch, enabled=True, port=0)

        def _raise_if_called(
            self: aiohttp.ClientSession,
            method: str,
            url: str,
            **kwargs: object,
        ) -> _FakeUpstreamStream:
            raise AssertionError(f"unexpected dashboard proxy call {method} {url}")

        monkeypatch.setattr(aiohttp.ClientSession, "request", _raise_if_called)

        with TestClient(app) as client:
            response = client.get("/dashboard/")

        assert response.status_code == 503
        assert response.text == "Dashboard is disabled on this cluster."


# ================================================================
# Reverse-proxy wire contract
# ================================================================


class TestDashboardProxyWireContract:
    """Proxy behavior at the HTTP trust boundary."""

    def test_proxy_post_filters_hop_by_hop_headers_and_preserves_body(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app = _make_dashboard_proxy_app(monkeypatch)
        captured = _install_fake_upstream(monkeypatch)
        payload = b'{"job":"aiperf-bench-7f2a","action":"refresh"}'

        with TestClient(app) as client:
            response = client.post(
                "/dashboard/api/refresh",
                content=payload,
                headers={
                    "Connection": "keep-alive",
                    "Content-Length": str(len(payload)),
                    "Host": "evil.example.invalid",
                    "Transfer-Encoding": "chunked",
                    "X-AIPerf-Trace": "conv-2026-04-21-9c3a",
                },
            )

        assert response.status_code == 200
        assert captured.method == "POST"
        assert captured.url == "http://localhost:8082/dashboard/api/refresh"
        assert captured.content == payload
        lowered = {key.lower() for key in captured.headers}
        assert "connection" not in lowered
        assert "content-length" not in lowered
        assert "host" not in lowered
        assert "transfer-encoding" not in lowered
        assert captured.headers["x-aiperf-trace"] == "conv-2026-04-21-9c3a"

    def test_proxy_response_filters_hop_by_hop_headers_and_preserves_chunks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app = _make_dashboard_proxy_app(monkeypatch)
        _install_fake_upstream(
            monkeypatch,
            response=_FakeUpstreamResponse(
                status=206,
                headers={
                    "Connection": "close",
                    "Content-Type": "text/javascript; charset=utf-8",
                    "Transfer-Encoding": "chunked",
                    "X-Dashboard-Build": "ui-2026-05-18",
                },
                chunks=[b"window.", b"AIPERF_DASHBOARD", b" = true;"],
            ),
        )

        with TestClient(app) as client:
            response = client.get("/dashboard/assets/app.js")

        assert response.status_code == 206
        assert response.content == b"window.AIPERF_DASHBOARD = true;"
        assert response.headers["content-type"] == "text/javascript; charset=utf-8"
        assert response.headers["x-dashboard-build"] == "ui-2026-05-18"
        assert "connection" not in response.headers
        assert "transfer-encoding" not in response.headers

    def test_proxy_options_request_forwards_to_sidecar_not_cors_stub(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app = _make_dashboard_proxy_app(monkeypatch)
        captured = _install_fake_upstream(
            monkeypatch,
            response=_FakeUpstreamResponse(status=204, chunks=[b""]),
        )

        with TestClient(app) as client:
            response = client.options(
                "/dashboard/assets/app.js",
                headers={
                    "Origin": "https://operator.example.invalid",
                    "Access-Control-Request-Method": "GET",
                },
            )

        assert response.status_code == 204
        assert captured.method == "OPTIONS"
        assert captured.url == "http://localhost:8082/dashboard/assets/app.js"


# ================================================================
# URL encoding and route-mount adversaries
# ================================================================


class TestDashboardProxyUrlAndMounting:
    """Encoded paths stay under the dashboard route; API routes stay API routes."""

    def test_proxy_preserves_query_encoding_without_rewriting_dashboard_url(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app = _make_dashboard_proxy_app(monkeypatch)
        captured = _install_fake_upstream(monkeypatch)

        with TestClient(app) as client:
            response = client.get(
                "/dashboard/assets/app.js?next=%2Fapi%2Fv1%2Fjobs&space=a+b&literal=%252Fdashboard"
            )

        assert response.status_code == 200
        assert captured.url == (
            "http://localhost:8082/dashboard/assets/app.js?"
            "next=%2Fapi%2Fv1%2Fjobs&space=a+b&literal=%252Fdashboard"
        )

    def test_proxy_encoded_traversal_shape_does_not_fall_through_to_local_api(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app = _make_dashboard_proxy_app(monkeypatch)

        @app.get("/api/v1/results")
        async def local_results_route() -> dict[str, bool]:
            return {"local_results_route": True}

        captured = _install_fake_upstream(
            monkeypatch,
            response=_FakeUpstreamResponse(chunks=[b"proxied-dashboard-path"]),
        )

        with TestClient(app) as client:
            response = client.get("/dashboard/%2E%2E/api/v1/results")

        assert response.status_code == 200
        assert response.content == b"proxied-dashboard-path"
        assert captured.url == "http://localhost:8082/dashboard/../api/v1/results"

    def test_results_server_mount_routes_dashboard_before_static_root(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        app = _make_results_server_app(monkeypatch, tmp_path)
        captured = _install_fake_upstream(
            monkeypatch,
            response=_FakeUpstreamResponse(chunks=[b"dashboard-sidecar-asset"]),
        )

        client = TestClient(app)
        dashboard_response = client.get("/dashboard/assets/index.js")
        api_response = client.get("/api/v1/results")

        assert dashboard_response.status_code == 200
        assert dashboard_response.content == b"dashboard-sidecar-asset"
        assert captured.url == "http://localhost:8082/dashboard/assets/index.js"
        assert api_response.status_code == 200
        assert api_response.json() == {"jobs": []}
