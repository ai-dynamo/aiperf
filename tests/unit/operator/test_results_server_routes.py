# Copyright 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Smoke tests for FastAPI route registration in `results_server.create_app`."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient


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
    routes = {r.path for r in app.routes if hasattr(r, "path")}
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
