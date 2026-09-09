# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for operator admin API security guardrails.

Focuses on:
- method restrictions for the narrow ``/admin/index`` recovery surface
- strict request-body rejection for routes whose schema is path-only or bodyless
- read-only results-server sidecar guards for the rebuild mutation endpoint
- static-UI fallthrough not turning admin-shaped paths into public HTML
- stable JSON error schemas for admin misses and disabled mutations
- no accidental exposure of admin mutation endpoints under ``/api/v1``

Out of scope: index rebuild concurrency and CLI behavior; see sibling
``tests/unit/operator/test_index_admin_adversarial.py``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from pytest import param

from aiperf.operator import runs_index
from aiperf.operator.results_server import create_app
from aiperf.operator.routers.admin import create_admin_router

# ============================================================================
# Helpers
# ============================================================================


def _admin_app(base_dir: Path, *, allow_rebuild: bool = True) -> FastAPI:
    """Build only the admin router so method tests avoid unrelated side effects."""
    app = FastAPI()
    app.include_router(
        create_admin_router(
            base_dir,
            base_dir / ".aiperf_index.sqlite",
            allow_rebuild=allow_rebuild,
        )
    )
    return app


@pytest.fixture
async def admin_client(tmp_path: Path) -> AsyncIterator[httpx.AsyncClient]:
    """HTTP client backed by a temp operator PVC-like results directory."""
    transport = httpx.ASGITransport(app=_admin_app(tmp_path))
    async with httpx.AsyncClient(
        transport=transport, base_url="http://aiperf-operator.local"
    ) as client:
        yield client


# ============================================================================
# Method restrictions
# ============================================================================


class TestAdminMethodRestrictions:
    """Only documented methods are accepted on the admin recovery surface."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "method,path,expected_status",
        [
            ("POST", "/admin/index/stats", 405),
            ("PUT", "/admin/index/stats", 405),
            ("DELETE", "/admin/index/stats", 405),
            ("GET", "/admin/index/rebuild", 405),
            ("PUT", "/admin/index/rebuild", 405),
            ("PATCH", "/admin/index/rebuild", 405),
            ("DELETE", "/admin/index/rebuild", 405),
            param("POST", "/admin/index/run/bench-prod/llama-3-70b", 405, id="run-row-post"),
            param("DELETE", "/admin/index/run/bench-prod/llama-3-70b", 405, id="run-row-delete"),
        ],
    )  # fmt: skip
    async def test_admin_routes_unsupported_methods_return_json_error_without_mutation(
        self,
        admin_client: httpx.AsyncClient,
        monkeypatch: pytest.MonkeyPatch,
        method: str,
        path: str,
        expected_status: int,
    ) -> None:
        bootstrap_calls: list[Path] = []

        async def fake_bootstrap(base_dir: Path, *, force: bool = False) -> object:
            del force
            bootstrap_calls.append(base_dir)
            raise AssertionError("unsupported admin method must not rebuild the index")

        monkeypatch.setattr(runs_index, "bootstrap", fake_bootstrap)

        response = await admin_client.request(method, path)

        assert response.status_code == expected_status
        assert response.headers["content-type"].startswith("application/json")
        assert set(response.json()) == {"detail"}
        assert bootstrap_calls == []


# ============================================================================
# Body schema strictness
# ============================================================================


class TestAdminBodySchemaStrictness:
    """Bodyless admin endpoints must reject smuggled JSON instead of ignoring it."""

    @pytest.mark.asyncio
    async def test_stats_request_body_rejected_before_index_stats_lookup(
        self,
        admin_client: httpx.AsyncClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        stats_calls: list[Path] = []

        async def fake_stats(db_path: Path) -> dict[str, int | None]:
            stats_calls.append(db_path)
            return {
                "runs_count": 3,
                "sweep_variations_count": 1,
                "db_bytes": 4096,
                "last_bootstrap_unix": 1_714_150_923,
                "schema_version": 1,
            }

        monkeypatch.setattr(runs_index, "stats", fake_stats)

        response = await admin_client.request(
            "GET",
            "/admin/index/stats",
            json={"namespace": "bench-prod", "drop_index": True},
        )

        assert response.status_code == 422
        assert stats_calls == []
        assert "namespace" in response.text or "body" in response.text

    @pytest.mark.asyncio
    async def test_run_row_request_body_rejected_before_index_lookup(
        self,
        admin_client: httpx.AsyncClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        lookup_calls: list[tuple[str, str]] = []

        async def fake_get_run_narrow_metrics(
            namespace: str, job_id: str
        ) -> dict[str, str] | None:
            lookup_calls.append((namespace, job_id))
            return {"epoch": "1714150923", "phase": "Succeeded"}

        monkeypatch.setattr(
            runs_index, "get_run_narrow_metrics", fake_get_run_narrow_metrics
        )

        response = await admin_client.request(
            "GET",
            "/admin/index/run/bench-prod/llama-3-70b",
            json={"include_metrics_blob": True},
        )

        assert response.status_code == 422
        assert lookup_calls == []
        assert "include_metrics_blob" in response.text or "body" in response.text


# ============================================================================
# Read-only sidecar guards
# ============================================================================


class TestAdminReadOnlySidecarGuards:
    """The public results-server sidecar must expose no writer-capable admin path."""

    @pytest.mark.asyncio
    async def test_rebuild_without_token_fails_closed_before_bootstrap(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unauthenticated rebuild POST must fail closed before any writer side effect.

        ``/admin/index/rebuild`` is a mutating route gated by the
        mutating-route auth dependency. With mutating routes disabled by
        default, an unauthenticated POST is rejected with 403 ("disabled")
        before the handler body runs — so ``runs_index.bootstrap`` is never
        invoked. The security invariant is "no unauthorized rebuild", not a
        specific status code.
        """
        bootstrap_calls: list[Path] = []

        async def fake_bootstrap(base_dir: Path, *, force: bool = False) -> object:
            del force
            bootstrap_calls.append(base_dir)
            raise AssertionError(
                "unauthenticated rebuild must not invoke writer bootstrap"
            )

        monkeypatch.setattr(runs_index, "bootstrap", fake_bootstrap)
        transport = httpx.ASGITransport(app=create_app(tmp_path))

        async with httpx.AsyncClient(
            transport=transport, base_url="http://aiperf-results.local"
        ) as client:
            response = await client.post("/admin/index/rebuild")

        assert response.status_code == 403
        assert "disabled" in response.json()["detail"]
        assert bootstrap_calls == []

    @pytest.mark.asyncio
    async def test_rebuild_on_explicit_read_only_admin_router_returns_503_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bootstrap_calls: list[Path] = []

        async def fake_bootstrap(base_dir: Path, *, force: bool = False) -> object:
            del force
            bootstrap_calls.append(base_dir)
            raise AssertionError("disabled admin router must not invoke bootstrap")

        monkeypatch.setattr(runs_index, "bootstrap", fake_bootstrap)
        transport = httpx.ASGITransport(app=_admin_app(tmp_path, allow_rebuild=False))

        async with httpx.AsyncClient(
            transport=transport, base_url="http://aiperf-operator.local"
        ) as client:
            response = await client.post("/admin/index/rebuild")

        assert response.status_code == 503
        assert response.headers["content-type"].startswith("application/json")
        assert "read-only results-server sidecar" in response.json()["detail"]
        assert bootstrap_calls == []


# ============================================================================
# Static fallthrough and /api/v1 exposure
# ============================================================================


class TestAdminStaticFallthroughAndApiV1Exposure:
    """Admin-shaped misses must stay JSON misses, not UI HTML or API aliases."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "path",
        [
            "/admin/index/not-a-real-admin-route",
            "/admin/index/rebuild/extra-segment",
            "/api/v1/admin/index/rebuild",
            "/api/v1/index/rebuild",
        ],
    )  # fmt: skip
    async def test_admin_shaped_unknown_paths_do_not_fall_through_to_static_ui(
        self, tmp_path: Path, path: str
    ) -> None:
        transport = httpx.ASGITransport(app=create_app(tmp_path))

        async with httpx.AsyncClient(
            transport=transport, base_url="http://aiperf-results.local"
        ) as client:
            response = await client.get(path, headers={"Accept": "text/html"})

        assert response.status_code in {404, 405}
        assert response.headers["content-type"].startswith("application/json")
        assert set(response.json()) == {"detail"}
        assert "<!doctype" not in response.text.lower()
        assert "AIPerf Operator" not in response.text

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "method,path",
        [
            ("POST", "/api/v1/admin/index/rebuild"),
            ("PUT", "/api/v1/admin/index/rebuild"),
            ("PATCH", "/api/v1/admin/index/rebuild"),
            param("DELETE", "/api/v1/admin/index/rebuild", id="delete-api-v1-admin-rebuild"),
            param("POST", "/api/v1/index/rebuild", id="post-api-v1-index-rebuild"),
        ],
    )  # fmt: skip
    async def test_api_v1_admin_mutation_aliases_do_not_invoke_rebuild(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        method: str,
        path: str,
    ) -> None:
        bootstrap_calls: list[Path] = []

        async def fake_bootstrap(base_dir: Path, *, force: bool = False) -> object:
            del force
            bootstrap_calls.append(base_dir)
            raise AssertionError("/api/v1 must not alias admin rebuild")

        monkeypatch.setattr(runs_index, "bootstrap", fake_bootstrap)
        transport = httpx.ASGITransport(app=create_app(tmp_path))

        async with httpx.AsyncClient(
            transport=transport, base_url="http://aiperf-results.local"
        ) as client:
            response = await client.request(method, path)

        assert response.status_code in {404, 405}
        assert response.headers["content-type"].startswith("application/json")
        assert bootstrap_calls == []

    def test_openapi_schema_has_no_admin_mutation_under_api_v1(
        self, tmp_path: Path
    ) -> None:
        app = create_app(tmp_path)
        schema = app.openapi()
        api_v1_admin_paths = [
            path for path in schema["paths"] if path.startswith("/api/v1/admin")
        ]
        api_v1_rebuild_paths = [
            path
            for path in schema["paths"]
            if path.startswith("/api/v1") and "rebuild" in path
        ]

        assert api_v1_admin_paths == []
        assert api_v1_rebuild_paths == []
        assert schema["paths"]["/admin/index/rebuild"].keys() == {"post"}


# ============================================================================
# Error schemas
# ============================================================================


class TestAdminErrorSchemas:
    """Admin errors return minimal JSON with operation context, not raw traces."""

    @pytest.mark.asyncio
    async def test_run_row_missing_index_entry_returns_json_detail_with_namespace_and_job(
        self,
        admin_client: httpx.AsyncClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        async def fake_get_run_narrow_metrics(namespace: str, job_id: str) -> None:
            assert namespace == "bench-prod"
            assert job_id == "missing-llama-3-70b"
            return None

        monkeypatch.setattr(
            runs_index, "get_run_narrow_metrics", fake_get_run_narrow_metrics
        )

        response = await admin_client.get(
            "/admin/index/run/bench-prod/missing-llama-3-70b"
        )

        assert response.status_code == 404
        assert response.headers["content-type"].startswith("application/json")
        assert response.json() == {
            "detail": "No index row for bench-prod/missing-llama-3-70b"
        }
        assert "Traceback" not in response.text

    @pytest.mark.asyncio
    async def test_method_not_allowed_error_schema_is_detail_only(
        self, admin_client: httpx.AsyncClient
    ) -> None:
        response = await admin_client.post("/admin/index/stats")

        assert response.status_code == 405
        assert response.headers["content-type"].startswith("application/json")
        assert response.json() == {"detail": "Method Not Allowed"}
