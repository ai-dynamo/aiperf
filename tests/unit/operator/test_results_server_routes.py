# Copyright 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Smoke tests for FastAPI route registration in `results_server.create_app`."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from tests.harness.operator import collect_app_paths


async def _post(app, path: str, *, token: str | None = None, json: dict | None = None):
    headers = {"Authorization": f"Bearer {token}"} if token is not None else {}
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        return await client.post(path, headers=headers, json=json)


def test_create_app_includes_sweeps_router(tmp_path: Path) -> None:
    """`/api/v1/sweeps` endpoints must be registered alongside jobs."""
    from aiperf.operator.results_server import create_app

    app = create_app(results_dir=tmp_path)
    routes = collect_app_paths(app)
    assert "/api/v1/sweeps" in routes
    assert "/api/v1/sweeps/{namespace}/{name}" in routes
    assert "/api/v1/sweeps/{namespace}/{name}/cells" in routes


def test_create_app_mounts_packaged_ui_when_override_env_is_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The results server ignores runtime UI override env and serves bundled UI."""
    from aiperf.operator.results_server import create_app

    monkeypatch.setenv("AIPERF_DEV_UI_OVERRIDE_DIR", str(tmp_path / "override"))

    app = create_app(results_dir=tmp_path)
    ui_route = next(r for r in app.routes if getattr(r, "name", None) == "ui")

    assert Path(ui_route.app.directory).name == "ui"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/api/v1/jobs", {"manifest": {"metadata": {"name": "bench"}}}),
        ("/api/v1/jobs/default/bench/cancel", None),
        ("/admin/index/rebuild", None),
    ],
)
async def test_mutating_routes_default_deny(
    tmp_path: Path, path: str, body: dict | None
) -> None:
    """Cluster-mutating API routes fail closed unless explicitly enabled."""
    from aiperf.operator.results_server import create_app

    app = create_app(results_dir=tmp_path)

    response = await _post(app, path, json=body)

    assert response.status_code == 403
    assert "disabled" in response.json()["detail"]


@pytest.mark.asyncio
async def test_mutating_route_rejects_missing_or_wrong_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Enabled mutating routes still require the configured bearer token."""
    from aiperf.operator.results_server import create_app

    monkeypatch.setenv("AIPERF_OPERATOR_MUTATING_ROUTES_ENABLED", "true")
    monkeypatch.setenv("AIPERF_OPERATOR_MUTATING_ROUTES_TOKEN", "correct-token")
    app = create_app(results_dir=tmp_path)

    missing = await _post(app, "/api/v1/jobs/default/bench/cancel")
    wrong = await _post(app, "/api/v1/jobs/default/bench/cancel", token="wrong-token")

    assert missing.status_code == 401
    assert wrong.status_code == 401


@pytest.mark.asyncio
async def test_mutating_route_allows_configured_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A valid token passes auth, but the read-only sidecar still disables rebuild.

    The results-server sidecar mounts the admin router with ``allow_rebuild=False``
    because it opens the runs index read-only (the kopf operator process is the
    single SQLite writer). So even an authenticated rebuild is denied with 503
    before ``runs_index.bootstrap`` (a ``DELETE``-issuing writer call) runs — the
    same contract enforced by ``admin.py``'s 503 message and the adversarial
    ``test_rebuild_on_explicit_read_only_admin_router_returns_503_json``.
    """
    from aiperf.operator.results_server import create_app

    monkeypatch.setenv("AIPERF_OPERATOR_MUTATING_ROUTES_ENABLED", "true")
    monkeypatch.setenv("AIPERF_OPERATOR_MUTATING_ROUTES_TOKEN", "correct-token")
    app = create_app(results_dir=tmp_path)

    bootstrap_calls: list[Path] = []

    async def fake_bootstrap(base_dir: Path, *, force: bool = False) -> SimpleNamespace:
        del force
        bootstrap_calls.append(base_dir)
        return SimpleNamespace(
            runs_indexed=1, sweep_variations_indexed=2, duration_seconds=0.5
        )

    with patch("aiperf.operator.runs_index.bootstrap", fake_bootstrap):
        response = await _post(app, "/admin/index/rebuild", token="correct-token")

    assert response.status_code == 503
    assert "disabled" in response.json()["detail"]
    assert bootstrap_calls == []


@pytest.mark.asyncio
async def test_create_job_route_allows_configured_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The configured bearer token authorizes job creation before handler logic runs."""
    from aiperf.operator.routers.jobs import create_jobs_router
    from aiperf.operator.routers.mutating_auth import mutating_route_dependencies

    monkeypatch.setenv("AIPERF_OPERATOR_MUTATING_ROUTES_ENABLED", "true")
    monkeypatch.setenv("AIPERF_OPERATOR_MUTATING_ROUTES_TOKEN", "correct-token")
    app = FastAPI()
    app.include_router(
        create_jobs_router([object()], tmp_path, mutating_route_dependencies())
    )
    create_impl = AsyncMock(
        return_value={"namespace": "default", "name": "bench", "uid": "uid-123"}
    )

    with patch("aiperf.operator.routers.jobs._create_job_impl", create_impl):
        response = await _post(
            app,
            "/api/v1/jobs",
            token="correct-token",
            json={"manifest": {"metadata": {"name": "bench"}}},
        )

    assert response.status_code == 201
    assert response.status_code not in {401, 403}
    create_impl.assert_awaited_once()


@pytest.mark.asyncio
async def test_cancel_job_route_allows_configured_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The configured bearer token authorizes job cancellation before handler logic runs."""
    from aiperf.operator.routers.jobs import create_jobs_router
    from aiperf.operator.routers.mutating_auth import mutating_route_dependencies

    monkeypatch.setenv("AIPERF_OPERATOR_MUTATING_ROUTES_ENABLED", "true")
    monkeypatch.setenv("AIPERF_OPERATOR_MUTATING_ROUTES_TOKEN", "correct-token")
    app = FastAPI()
    app.include_router(
        create_jobs_router([object()], tmp_path, mutating_route_dependencies())
    )
    cancel_impl = AsyncMock(return_value={"cancelled": True})

    with patch("aiperf.operator.routers.jobs._cancel_job_impl", cancel_impl):
        response = await _post(
            app, "/api/v1/jobs/default/bench/cancel", token="correct-token"
        )

    assert response.status_code == 200
    assert response.status_code not in {401, 403}
    cancel_impl.assert_awaited_once()


@pytest.mark.asyncio
async def test_read_only_results_route_unaffected_by_default_deny(
    tmp_path: Path,
) -> None:
    """Read-only results endpoints remain available without auth by default."""
    from aiperf.operator.results_server import create_app

    app = create_app(results_dir=tmp_path)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/api/v1/results")

    assert response.status_code == 200


# ============================================================
# GZip Compression Middleware
# ============================================================


def test_gzip_middleware_present(tmp_path: Path) -> None:
    """GZipMiddleware must be installed on the results-server app."""
    from starlette.middleware.gzip import GZipMiddleware

    from aiperf.operator.results_server import create_app

    app = create_app(results_dir=tmp_path)
    middleware_classes = [m.cls for m in app.user_middleware]
    assert GZipMiddleware in middleware_classes, (
        f"GZipMiddleware not found in app middleware stack: {middleware_classes}"
    )


def test_gzip_middleware_minimum_size_is_500(tmp_path: Path) -> None:
    """Tiny responses (health checks) must skip compression overhead."""
    from starlette.middleware.gzip import GZipMiddleware

    from aiperf.operator.results_server import create_app

    app = create_app(results_dir=tmp_path)
    gzip_mw = next(m for m in app.user_middleware if m.cls is GZipMiddleware)
    assert gzip_mw.kwargs["minimum_size"] == 500


@pytest.mark.asyncio
async def test_gzip_compresses_large_json_response(tmp_path: Path) -> None:
    """A >500-byte JSON response is returned with Content-Encoding: gzip."""
    from aiperf.operator.results_server import create_app

    app = create_app(results_dir=tmp_path)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get(
            "/openapi.json", headers={"Accept-Encoding": "gzip"}
        )

    assert response.status_code == 200
    assert len(response.content) > 500
    assert response.headers.get("content-encoding") == "gzip"


@pytest.mark.asyncio
async def test_gzip_skips_small_health_response(tmp_path: Path) -> None:
    """The /healthz payload is far under minimum_size and stays uncompressed."""
    from aiperf.operator.results_server import create_app

    app = create_app(results_dir=tmp_path)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/healthz", headers={"Accept-Encoding": "gzip"})

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert "content-encoding" not in response.headers


@pytest.mark.asyncio
async def test_gzip_does_not_corrupt_streaming_zip_bundle(tmp_path: Path) -> None:
    """The ZIP bundle is a StreamingResponse; gzip must not truncate it.

    Streaming responses take the middleware's chunked branch, which drops
    Content-Length and compresses incrementally. Guard the round-trip by
    verifying the zip is structurally intact, not just that a filename
    appears in the raw bytes (which passes even on a truncated archive).
    """
    import io
    import zipfile

    from aiperf.kubernetes.results_sidecar import write_ready_marker
    from aiperf.operator.results_layout import run_dir, write_latest
    from aiperf.operator.results_server import create_app

    epoch = "1714150923"
    d = run_dir(tmp_path, "ns", "job", epoch)
    d.mkdir(parents=True, exist_ok=True)
    (d / "old.json").write_bytes(b'{"v":1}')
    write_ready_marker(d)
    write_latest(tmp_path, "ns", "job", epoch)

    app = create_app(results_dir=tmp_path)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get(
            f"/api/v1/results/ns/job/runs/{epoch}.zip",
            headers={"Accept-Encoding": "gzip"},
        )

    assert response.status_code == 200
    # Verify the zip is structurally valid and contains the expected file.
    # httpx transparently decodes any Content-Encoding, so response.content
    # is the raw zip bytes; zipfile.ZipFile proves the archive is intact.
    zf = zipfile.ZipFile(io.BytesIO(response.content))
    assert "old.json" in zf.namelist()
    assert zf.testzip() is None  # None = no bad CRC/file found
