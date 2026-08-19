# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for operator results-server app assembly.

Focuses on:
- route-mount boundaries that keep every /api/v1 surface on results-server
- /healthz liveness before heavyweight startup dependencies are available
- dashboard proxy and packaged static UI ordering at the root catch-all
- environment/default result-directory selection without crossing into the operator port
- read-only admin guardrails and startup holder state transitions

Out of scope: per-router business behavior; see sibling router-specific tests such as
``test_results_server.py``, ``test_jobs_router.py``, and ``test_sweeps_router.py``.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from pytest import param
from starlette.routing import Mount

from tests.harness.operator import collect_app_paths

# ============================================================
# Helpers
# ============================================================


class _FakeResultsDB:
    """Minimal ResultsDB stand-in that records the sidecar startup directory."""

    opened_dirs: list[Path] = []
    close_count: int = 0

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        type(self).opened_dirs.append(base_dir)

    def close(self) -> None:
        type(self).close_count += 1


class _FakeApiClient:
    """Minimal kubernetes_asyncio ApiClient stand-in with observable close state."""

    instances: list[_FakeApiClient] = []

    def __init__(self) -> None:
        self.closed = False
        type(self).instances.append(self)

    async def close(self) -> None:
        self.closed = True


def _reset_fakes() -> None:
    _FakeResultsDB.opened_dirs = []
    _FakeResultsDB.close_count = 0
    _FakeApiClient.instances = []


def _route_paths(results_dir: Path) -> list[str]:
    from aiperf.operator.results_server import create_app
    from tests.harness.operator import collect_app_paths

    return collect_app_paths(create_app(results_dir=results_dir))


def _path_index(paths: list[str], path: str) -> int:
    return paths.index(path)


# ============================================================
# Results-server route surface
# ============================================================


class TestResultsServerRouteSurface:
    """Verify the app assembly owns the complete public HTTP surface."""

    @pytest.mark.parametrize(
        "path",
        [
            "/api/v1/jobs",
            "/api/v1/jobs/{namespace}/{name}/ws",
            "/api/v1/cluster",
            "/api/v1/sweeps",
            "/api/v1/sweeps/{namespace}/{name}/cells",
            "/api/v1/sweeps/{namespace}/{name}/children",
            "/api/v1/results",
            "/api/v1/results/{namespace}/{job_id}/runs/{epoch}/profile_export",
            "/api/v1/config/retention",
            "/api/v1/config/features",
            "/api/v1/analytics/leaderboard",
            "/api/v1/analytics/summary/{namespace}/{job_id}",
            "/api/v1/index",
            "/api/v1/config/{namespace}/{job_id}",
            "/admin/index/stats",
            "/admin/index/rebuild",
            "/healthz",
            "/dashboard/{path:path}",
        ],
    )  # fmt: skip
    def test_create_app_results_server_mounts_expected_route(
        self, tmp_path: Path, path: str
    ) -> None:
        paths = set(_route_paths(tmp_path))

        assert path in paths

    @pytest.mark.parametrize(
        "path",
        [
            param("/metrics", id="prometheus-metrics-stays-on-operator-container"),
            param("/api/metrics", id="controller-api-metrics-not-results-server"),
            param("/api/progress", id="controller-progress-not-results-server"),
            param("/api/shutdown", id="controller-shutdown-not-results-server"),
            param("/api/results/list", id="controller-results-list-not-results-server"),
        ],
    )  # fmt: skip
    def test_create_app_operator_and_controller_paths_are_not_mounted(
        self, tmp_path: Path, path: str
    ) -> None:
        paths = set(_route_paths(tmp_path))

        assert path not in paths

    def test_create_app_static_root_is_last_after_api_health_and_dashboard(
        self, tmp_path: Path
    ) -> None:
        paths = _route_paths(tmp_path)

        from aiperf.operator.results_server import create_app

        ui_route = next(
            route
            for route in create_app(results_dir=tmp_path).routes
            if getattr(route, "name", None) == "ui"
        )

        assert isinstance(ui_route, Mount)
        assert paths[-1] == ""
        static_index = len(paths) - 1
        for path in (
            "/api/v1/jobs",
            "/api/v1/sweeps",
            "/api/v1/results",
            "/api/v1/config/features",
            "/healthz",
            "/dashboard/{path:path}",
        ):
            assert _path_index(paths, path) < static_index


# ============================================================
# Boundary requests
# ============================================================


class TestResultsServerBoundaryRequests:
    """Route ordering must be observable at HTTP boundaries, not just in tables."""

    def test_healthz_before_lifespan_returns_liveness_payload(
        self, tmp_path: Path
    ) -> None:
        from aiperf.operator.results_server import create_app

        client = TestClient(create_app(results_dir=tmp_path))

        response = client.get("/healthz")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_dashboard_proxy_route_wins_over_packaged_static_root(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from aiperf.operator.environment import OperatorEnvironment
        from aiperf.operator.results_server import create_app

        monkeypatch.setattr(OperatorEnvironment.DASHBOARD, "PROXY_ENABLED", False)
        monkeypatch.setattr(OperatorEnvironment.DASHBOARD, "PORT", 0)
        client = TestClient(create_app(results_dir=tmp_path))

        response = client.get("/dashboard/")

        assert response.status_code == 503
        assert "disabled" in response.text.lower()
        assert "<!doctype html" not in response.text.lower()

    @pytest.mark.parametrize(
        "path,expected_status",
        [
            ("/api/v1/results", 200),
            param(
                "/api/v1/results/acme-bench/vllm-bench",
                409,
                id="results-router-before-static-root",
            ),
            param(
                "/api/v1/jobs/acme-bench/vllm-bench/epochs",
                200,
                id="jobs-epochs-degrades-without-k8s-client",
            ),
            param(
                "/api/v1/jobs/acme-bench/vllm-bench",
                503,
                id="jobs-live-route-not-static-root",
            ),
            param(
                "/api/v1/sweeps/acme-bench/latency-sweep/epochs",
                200,
                id="sweep-epochs-disk-route-not-static-root",
            ),
            param(
                "/api/v1/sweeps/acme-bench/latency-sweep",
                503,
                id="sweeps-live-route-not-static-root",
            ),
            param(
                "/admin/index/rebuild",
                404,
                id="admin-get-does-not-fall-through-to-static-root",
            ),
        ],
    )  # fmt: skip
    def test_api_boundary_paths_do_not_fall_through_to_spa_index(
        self, tmp_path: Path, path: str, expected_status: int
    ) -> None:
        from aiperf.operator.results_server import create_app

        client = TestClient(create_app(results_dir=tmp_path))

        response = client.get(path)

        assert response.status_code == expected_status
        assert "AIPerf Kubernetes Operator" not in response.text

    def test_static_root_serves_packaged_spa_for_non_api_navigation(
        self, tmp_path: Path
    ) -> None:
        from aiperf.operator.results_server import create_app

        client = TestClient(create_app(results_dir=tmp_path))

        response = client.get("/")

        assert response.status_code == 200
        assert "AIPerf Kubernetes Operator" in response.text


# ============================================================
# Environment and startup state
# ============================================================


class TestResultsServerEnvironmentAndStartup:
    """Validate base directory selection and lifespan-owned app state."""

    def test_create_app_without_results_dir_uses_aiperf_results_dir_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AIPERF_RESULTS_DIR", str(tmp_path / "env-results"))
        from aiperf.operator import results_server

        importlib.reload(results_server)

        ui_mount = next(
            route
            for route in results_server.create_app().routes
            if getattr(route, "name", None) == "ui"
        )

        assert tmp_path / "env-results" == results_server.RESULTS_DIR
        assert Path(ui_mount.app.directory).name == "ui"

    def test_create_app_explicit_results_dir_overrides_env_constant(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AIPERF_RESULTS_DIR", str(tmp_path / "env-results"))
        from aiperf.operator import results_server

        importlib.reload(results_server)
        explicit_dir = tmp_path / "explicit-results"
        app = results_server.create_app(results_dir=explicit_dir)
        routes = collect_app_paths(app)

        assert "/api/v1/results" in routes
        assert tmp_path / "env-results" == results_server.RESULTS_DIR

    @pytest.mark.asyncio
    async def test_lifespan_initializes_readonly_db_and_kubernetes_holder_then_closes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from kubernetes_asyncio import config

        from aiperf.kubernetes import client as aiperf_k8s_client
        from aiperf.operator import results_db, runs_index
        from aiperf.operator.results_server import create_app

        _reset_fakes()
        calls: list[str] = []

        async def fake_open_readonly(path: Path) -> None:
            calls.append(f"open_readonly:{path.name}")

        async def fake_close_index() -> None:
            calls.append("close_index")

        def fake_load_incluster_config() -> None:
            calls.append("load_incluster_config")

        async def fake_load_kube_config(**kwargs: object) -> None:
            del kwargs
            calls.append("load_kube_config")

        monkeypatch.setattr(runs_index, "is_open", lambda: False)
        monkeypatch.setattr(runs_index, "open_readonly", fake_open_readonly)
        monkeypatch.setattr(runs_index, "close", fake_close_index)
        monkeypatch.setattr(results_db, "ResultsDB", _FakeResultsDB)
        monkeypatch.setattr(config, "load_incluster_config", fake_load_incluster_config)
        monkeypatch.setattr(config, "load_kube_config", fake_load_kube_config)
        monkeypatch.setattr(aiperf_k8s_client, "ApiClient", _FakeApiClient)

        app = create_app(results_dir=tmp_path)
        ctx = app.router.lifespan_context(app)

        await ctx.__aenter__()
        await ctx.__aexit__(None, None, None)

        assert calls == [
            "open_readonly:.aiperf_index.sqlite",
            "load_incluster_config",
            "close_index",
        ]
        assert _FakeResultsDB.opened_dirs == [tmp_path]
        assert _FakeResultsDB.close_count == 1
        assert len(_FakeApiClient.instances) == 1
        assert _FakeApiClient.instances[0].closed is True

    @pytest.mark.asyncio
    async def test_lifespan_without_kubernetes_keeps_live_routes_unavailable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from kubernetes_asyncio import config

        from aiperf.operator import results_db, runs_index
        from aiperf.operator.results_server import create_app

        _reset_fakes()

        async def fake_open_readonly(path: Path) -> None:
            del path

        async def fake_close_index() -> None:
            return None

        def fake_load_incluster_config() -> None:
            raise config.ConfigException("no in-cluster token")

        async def fake_load_kube_config(**kwargs: object) -> None:
            del kwargs
            raise config.ConfigException("no kubeconfig for unit test")

        monkeypatch.setattr(runs_index, "is_open", lambda: False)
        monkeypatch.setattr(runs_index, "open_readonly", fake_open_readonly)
        monkeypatch.setattr(runs_index, "close", fake_close_index)
        monkeypatch.setattr(results_db, "ResultsDB", _FakeResultsDB)
        monkeypatch.setattr(config, "load_incluster_config", fake_load_incluster_config)
        monkeypatch.setattr(config, "load_kube_config", fake_load_kube_config)

        app = create_app(results_dir=tmp_path)
        ctx = app.router.lifespan_context(app)
        await ctx.__aenter__()
        transport = httpx.ASGITransport(app=app)

        async with httpx.AsyncClient(
            transport=transport, base_url="http://aiperf-operator.local"
        ) as client:
            response = await client.get("/api/v1/jobs/acme-bench/vllm-bench")

        await ctx.__aexit__(None, None, None)

        assert response.status_code == 503
        assert "Kubernetes API client" in response.json()["detail"]
        assert _FakeResultsDB.opened_dirs == [tmp_path]


# ============================================================
# Admin guardrails and port ownership
# ============================================================


class TestResultsServerAdminAndPortOwnership:
    """Read-only results-server must not expose writer or operator-only surfaces."""

    @pytest.mark.asyncio
    async def test_admin_rebuild_route_fails_closed_without_token(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unauthenticated rebuild POST is denied before any writer side effect.

        ``/admin/index/rebuild`` on the results-server is a mutating route gated
        by the mutating-route auth dependency, disabled by default. An
        unauthenticated POST is rejected with 403 ("disabled") before the
        handler body runs, so ``runs_index.bootstrap`` is never invoked.
        """
        from aiperf.operator import runs_index
        from aiperf.operator.results_server import create_app

        bootstrap_calls: list[Path] = []

        async def fake_bootstrap(
            base_dir: Path, *, force: bool = False
        ) -> SimpleNamespace:
            del force
            bootstrap_calls.append(base_dir)
            return SimpleNamespace(
                runs_indexed=1,
                sweep_variations_indexed=0,
                duration_seconds=0.01,
            )

        monkeypatch.setattr(runs_index, "bootstrap", fake_bootstrap)
        transport = httpx.ASGITransport(app=create_app(results_dir=tmp_path))

        async with httpx.AsyncClient(
            transport=transport, base_url="http://aiperf-operator.local"
        ) as client:
            response = await client.post("/admin/index/rebuild")

        assert response.status_code == 403
        assert "disabled" in response.json()["detail"]
        assert bootstrap_calls == []

    def test_results_server_main_runs_uvicorn_on_results_server_port(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from aiperf.operator import results_server

        captured: dict[str, object] = {}

        def fake_run(app: object, **kwargs: object) -> None:
            captured["app"] = app
            captured.update(kwargs)

        monkeypatch.setattr(results_server.uvicorn, "run", fake_run)
        monkeypatch.setattr(results_server, "SERVER_PORT", 19081)

        results_server.main()

        assert captured["host"] == "0.0.0.0"
        assert captured["port"] == 19081
        assert captured["access_log"] is False
        assert captured["log_level"] == "info"

    def test_operator_main_does_not_define_results_server_fastapi_app(self) -> None:
        from aiperf.operator import main as operator_main

        assert not hasattr(operator_main, "create_app")
        assert not hasattr(operator_main, "app")
