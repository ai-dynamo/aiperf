# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for operator admin index rebuild APIs and CLI.

Focuses on:
- concurrent ``POST /admin/index/rebuild`` requests against the writer process
- rebuild failure propagation without success-shaped stale payloads
- request and response schema boundaries for the manual recovery hatch
- missing operator API auto-discovery diagnostics when ``--api-url`` is omitted

Out of scope: runs_index filesystem-walk correctness; see sibling
``tests/unit/operator/test_runs_index_adversarial.py`` and
``tests/unit/operator/test_runs_index_edge_cases.py``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from aiperf.operator import runs_index
from aiperf.operator.routers.admin import create_admin_router


def _admin_app(base_dir: Path, *, allow_rebuild: bool = True) -> FastAPI:
    """Build a narrow FastAPI app containing only the admin index router."""
    app = FastAPI()
    app.include_router(
        create_admin_router(
            base_dir,
            base_dir / ".aiperf_index.sqlite",
            allow_rebuild=allow_rebuild,
        )
    )
    return app


# ============================================================================
# Helpers
# ============================================================================


# ============================================================================


class TestAdminIndexRebuildApi:
    """Exercise the writer-side ``/admin/index/rebuild`` trust boundary."""

    @pytest.mark.asyncio
    async def test_rebuild_concurrent_requests_reject_second_without_second_bootstrap(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A force rebuild is destructive; overlapping requests must not both run."""
        started = asyncio.Event()
        release = asyncio.Event()
        bootstrap_calls: list[Path] = []

        async def fake_bootstrap(
            base_dir: Path, *, force: bool = False
        ) -> SimpleNamespace:
            assert force is True
            bootstrap_calls.append(base_dir)
            started.set()
            await release.wait()
            return SimpleNamespace(
                runs_indexed=3,
                sweep_variations_indexed=1,
                duration_seconds=0.25,
            )

        monkeypatch.setattr(runs_index, "bootstrap", fake_bootstrap)
        transport = httpx.ASGITransport(app=_admin_app(tmp_path))

        async with httpx.AsyncClient(
            transport=transport, base_url="http://aiperf-operator.local"
        ) as client:
            first = asyncio.create_task(client.post("/admin/index/rebuild"))
            await started.wait()
            second = asyncio.create_task(client.post("/admin/index/rebuild"))
            await asyncio.sleep(0)
            release.set()
            first_response, second_response = await asyncio.gather(first, second)

        assert first_response.status_code == 200
        assert second_response.status_code == 409
        assert second_response.json()["detail"] == "Index rebuild already in progress"
        assert bootstrap_calls == [tmp_path]

    @pytest.mark.asyncio
    async def test_rebuild_bootstrap_exception_returns_500_without_success_schema(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failed rebuild must not be reported as a stale successful rebuild."""

        async def fake_bootstrap(
            base_dir: Path, *, force: bool = False
        ) -> SimpleNamespace:
            del base_dir, force
            raise RuntimeError("sqlite disk I/O error while rebuilding aiperf index")

        monkeypatch.setattr(runs_index, "bootstrap", fake_bootstrap)
        transport = httpx.ASGITransport(
            app=_admin_app(tmp_path), raise_app_exceptions=False
        )

        async with httpx.AsyncClient(
            transport=transport, base_url="http://aiperf-operator.local"
        ) as client:
            response = await client.post("/admin/index/rebuild")

        assert response.status_code == 500
        assert "runs_indexed" not in response.text
        assert "sweep_variations_indexed" not in response.text

    @pytest.mark.asyncio
    async def test_rebuild_success_response_schema_has_only_contract_fields(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The admin response schema is the CLI contract; extra attrs must not leak."""

        async def fake_bootstrap(
            base_dir: Path, *, force: bool = False
        ) -> SimpleNamespace:
            assert base_dir == tmp_path
            assert force is True
            return SimpleNamespace(
                runs_indexed=11,
                sweep_variations_indexed=4,
                duration_seconds=0.125,
                stale_status="from-prior-rebuild",
            )

        monkeypatch.setattr(runs_index, "bootstrap", fake_bootstrap)
        transport = httpx.ASGITransport(app=_admin_app(tmp_path))

        async with httpx.AsyncClient(
            transport=transport, base_url="http://aiperf-operator.local"
        ) as client:
            response = await client.post("/admin/index/rebuild")

        assert response.status_code == 200
        assert response.json() == {
            "runs_indexed": 11,
            "sweep_variations_indexed": 4,
            "duration_seconds": 0.125,
        }

    @pytest.mark.asyncio
    async def test_rebuild_request_body_with_partial_scope_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The endpoint has no partial-rebuild schema; bodies must fail closed."""
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
        transport = httpx.ASGITransport(app=_admin_app(tmp_path))

        async with httpx.AsyncClient(
            transport=transport, base_url="http://aiperf-operator.local"
        ) as client:
            response = await client.post(
                "/admin/index/rebuild",
                json={"namespace": "bench-prod", "job_id": "llama-bench-7f2a"},
            )

        assert response.status_code == 422
        assert bootstrap_calls == []

    @pytest.mark.asyncio
    async def test_rebuild_disabled_returns_503_without_touching_index(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Read-only sidecars expose the route but must not call the writer path."""
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
        transport = httpx.ASGITransport(app=_admin_app(tmp_path, allow_rebuild=False))

        async with httpx.AsyncClient(
            transport=transport, base_url="http://aiperf-operator.local"
        ) as client:
            response = await client.post("/admin/index/rebuild")

        assert response.status_code == 503
        assert "read-only results-server sidecar" in response.json()["detail"]
        assert bootstrap_calls == []
