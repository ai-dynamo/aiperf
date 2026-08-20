# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial HTTP strictness tests for operator routers.

Focuses on:
- GET route tolerance for unexpected request bodies at the browser/API boundary
- unsupported method rejection before JSON body parsing or handler execution
- content-type edge cases for body-bearing mutation endpoints
- explicit POST-only semantics for job creation, cancellation, and index rebuilds

Out of scope: per-router payload schemas and path traversal, covered by sibling
operator router adversarial tests.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from pytest import param

from aiperf.operator.results_db import ResultsDB
from aiperf.operator.routers.admin import create_admin_router
from aiperf.operator.routers.config import create_config_router
from aiperf.operator.routers.jobs import create_jobs_router
from aiperf.operator.routers.results_analytics import create_results_analytics_router
from aiperf.operator.routers.results_files import create_results_files_router
from aiperf.operator.routers.sweeps import create_sweeps_router

# ============================================================
# Helpers
# ============================================================


_MUTATION_JOB_PATH = "/api/v1/jobs/aiperf-benchmarks/llama-3-8b-load/cancel"
_CREATE_JOB_PATH = "/api/v1/jobs"
_REBUILD_PATH = "/admin/index/rebuild"


class _ClosedResultsDB:
    """DB provider sentinel for routes that must not reach analytics handlers."""

    def __call__(self) -> ResultsDB:
        raise HTTPException(503, "analytics engine intentionally unavailable")


def _strictness_app(tmp_path: Path, *, api: object | None = object()) -> FastAPI:
    """Build the operator HTTP routers without lifespan side effects."""
    app = FastAPI()
    api_holder = [api]
    app.include_router(create_jobs_router(api_holder, tmp_path))
    app.include_router(create_sweeps_router(api_holder, tmp_path))
    app.include_router(create_results_files_router(tmp_path))
    app.include_router(create_config_router())
    app.include_router(create_admin_router(tmp_path, tmp_path / ".aiperf_index.sqlite"))
    app.include_router(
        create_results_analytics_router(_ClosedResultsDB(), tmp_path, api_holder)
    )
    return app


@pytest.fixture
async def client(tmp_path: Path) -> AsyncIterator[httpx.AsyncClient]:
    """HTTP client mounted on all operator routers with temp PVC state."""
    transport = httpx.ASGITransport(app=_strictness_app(tmp_path))
    async with httpx.AsyncClient(
        transport=transport, base_url="http://aiperf.operator.local"
    ) as c:
        yield c


def _install_read_route_fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make read-only list routes deterministic without Kubernetes or DuckDB."""
    from aiperf.operator.routers import jobs as jobs_module
    from aiperf.operator.routers import sweeps as sweeps_module

    monkeypatch.setattr(jobs_module, "list_all_jobs", AsyncMock(return_value=[]))
    monkeypatch.setattr(sweeps_module, "list_all_sweeps", AsyncMock(return_value=[]))


def _create_patch_api(return_uid: str = "uid-aiperf-7f2a") -> MagicMock:
    """Return a CustomObjectsApi fake for mutation endpoint wire-shape assertions."""
    return MagicMock(
        create_namespaced_custom_object=AsyncMock(
            return_value={"metadata": {"uid": return_uid}}
        ),
        patch_namespaced_custom_object=AsyncMock(return_value={}),
    )


# ============================================================
# GET and DELETE body handling
# ============================================================


class TestReadRoutesIgnoreBodies:
    """Bodyless read routes do not parse attacker-supplied JSON bodies."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "path,expected_fragment",
        [
            ("/api/v1/config/features", '"dashboard_enabled"'),
            ("/api/v1/jobs", '"jobs"'),
            ("/api/v1/sweeps", '"sweeps"'),
            ("/api/v1/results", '"jobs"'),
        ],
    )  # fmt: skip
    async def test_get_route_unexpected_json_body_is_ignored_and_returns_resource(
        self,
        client: httpx.AsyncClient,
        monkeypatch: pytest.MonkeyPatch,
        path: str,
        expected_fragment: str,
    ) -> None:
        _install_read_route_fakes(monkeypatch)

        response = await client.request(
            "GET",
            path,
            content=b'{"manifest": {"metadata": "should-not-be-parsed"',
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 200, response.text
        assert expected_fragment in response.text
        assert "should-not-be-parsed" not in response.text

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "path",
        [
            param("/api/v1/config/features", id="config-features"),
            param("/api/v1/jobs", id="jobs-list"),
            param("/api/v1/sweeps", id="sweeps-list"),
            param("/api/v1/results", id="results-list"),
        ],
    )  # fmt: skip
    async def test_delete_route_json_body_is_rejected_by_method_not_body_parser(
        self,
        client: httpx.AsyncClient,
        monkeypatch: pytest.MonkeyPatch,
        path: str,
    ) -> None:
        _install_read_route_fakes(monkeypatch)

        response = await client.request(
            "DELETE",
            path,
            content=b'{"confirm": true, "job": "llama-3-8b-load"',
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 405
        assert response.json()["detail"] == "Method Not Allowed"


# ============================================================
# Unsupported methods and explicit mutation semantics
# ============================================================


class TestUnsupportedMethods:
    """Unsupported verbs fail before reaching route handlers or parsing bodies."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "method,path",
        [
            param("PATCH", _CREATE_JOB_PATH, id="patch-create-job"),
            param("PUT", _CREATE_JOB_PATH, id="put-create-job"),
            param("GET", _MUTATION_JOB_PATH, id="get-cancel-job"),
            param("PATCH", _MUTATION_JOB_PATH, id="patch-cancel-job"),
            param("DELETE", _MUTATION_JOB_PATH, id="delete-cancel-job"),
            param("GET", _REBUILD_PATH, id="get-index-rebuild"),
            param("PATCH", _REBUILD_PATH, id="patch-index-rebuild"),
            param("DELETE", _REBUILD_PATH, id="delete-index-rebuild"),
        ],
    )  # fmt: skip
    async def test_mutation_route_unsupported_method_returns_405_without_mutation(
        self,
        tmp_path: Path,
        method: str,
        path: str,
    ) -> None:
        mock_custom = _create_patch_api()
        transport = httpx.ASGITransport(app=_strictness_app(tmp_path))

        with patch(
            "aiperf.kubernetes.client.client.CustomObjectsApi",
            return_value=mock_custom,
        ):
            async with httpx.AsyncClient(
                transport=transport, base_url="http://aiperf.operator.local"
            ) as c:
                response = await c.request(
                    method,
                    path,
                    content=b'{"manifest": {"metadata": {"name": "mutate-me"}}',
                    headers={"Content-Type": "application/json"},
                )

        assert response.status_code == 405
        assert response.json()["detail"] == "Method Not Allowed"
        mock_custom.create_namespaced_custom_object.assert_not_awaited()
        mock_custom.patch_namespaced_custom_object.assert_not_awaited()


class TestMutationContentTypes:
    """Body-bearing mutation endpoints distinguish absent JSON from malformed JSON."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "headers,content,expected_status,expected_detail_fragment",
        [
            param(
                {"Content-Type": "application/json"},
                b'{"manifest": {"metadata": {"name": "unterminated-job"}}',
                422,
                "json_invalid",
                id="malformed-json-rejected-before-create",
            ),
            param(
                {"Content-Type": "text/plain"},
                b'{"manifest": {"metadata": {"name": "plain-text-job"}}}',
                422,
                "model_attributes_type",
                id="text-plain-json-not-treated-as-create-body",
            ),
        ],
    )  # fmt: skip
    async def test_create_job_invalid_content_type_or_json_does_not_create_resource(
        self,
        tmp_path: Path,
        headers: dict[str, str],
        content: bytes,
        expected_status: int,
        expected_detail_fragment: str,
    ) -> None:
        mock_custom = _create_patch_api()
        transport = httpx.ASGITransport(app=_strictness_app(tmp_path))

        with patch(
            "aiperf.kubernetes.client.client.CustomObjectsApi",
            return_value=mock_custom,
        ):
            async with httpx.AsyncClient(
                transport=transport, base_url="http://aiperf.operator.local"
            ) as c:
                response = await c.post(
                    _CREATE_JOB_PATH, content=content, headers=headers
                )

        assert response.status_code == expected_status
        assert expected_detail_fragment in response.text
        mock_custom.create_namespaced_custom_object.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_create_job_valid_json_body_creates_resource_only_with_post(
        self, tmp_path: Path
    ) -> None:
        mock_custom = _create_patch_api(return_uid="uid-post-only-7f2a")
        transport = httpx.ASGITransport(app=_strictness_app(tmp_path))
        payload = {
            "manifest": {
                "metadata": {
                    "namespace": "aiperf-benchmarks",
                    "name": "llama-3-8b-post-only",
                },
                "spec": {"benchmark": {"model": "meta-llama/Llama-3-8B"}},
            }
        }

        with patch(
            "aiperf.kubernetes.client.client.CustomObjectsApi",
            return_value=mock_custom,
        ):
            async with httpx.AsyncClient(
                transport=transport, base_url="http://aiperf.operator.local"
            ) as c:
                response = await c.post(_CREATE_JOB_PATH, json=payload)

        assert response.status_code == 201, response.text
        assert response.json() == {
            "namespace": "aiperf-benchmarks",
            "name": "llama-3-8b-post-only",
            "uid": "uid-post-only-7f2a",
        }
        mock_custom.create_namespaced_custom_object.assert_awaited_once()
        assert (
            mock_custom.create_namespaced_custom_object.call_args.kwargs["body"]["kind"]
            == "AIPerfJob"
        )


class TestMutationBodyIndependence:
    """POST-only mutation routes ignore extraneous bodies when the path owns identity."""

    @pytest.mark.asyncio
    async def test_cancel_job_post_ignores_json_body_and_patches_path_job_only(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from aiperf.operator.routers import jobs as jobs_module

        async def fake_find_any_job(
            api: object,
            results_dir: Path,
            namespace: str,
            name: str,
            *,
            epoch: str | None = None,
        ) -> object:
            del api, results_dir, epoch
            from aiperf.kubernetes.models import AIPerfJobInfo

            return AIPerfJobInfo(
                name=name,
                namespace=namespace,
                phase="Running",
                job_id=name,
                jobset_name=f"aiperf-{name}",
                source="live",
            )

        monkeypatch.setattr(jobs_module, "find_any_job", fake_find_any_job)
        mock_custom = _create_patch_api()
        transport = httpx.ASGITransport(app=_strictness_app(tmp_path))

        with patch(
            "aiperf.kubernetes.client.client.CustomObjectsApi",
            return_value=mock_custom,
        ):
            async with httpx.AsyncClient(
                transport=transport, base_url="http://aiperf.operator.local"
            ) as c:
                response = await c.post(
                    _MUTATION_JOB_PATH,
                    json={
                        "namespace": "attacker-namespace",
                        "name": "attacker-job",
                        "cancel": False,
                    },
                )

        assert response.status_code == 200, response.text
        assert response.json() == {"cancelled": True}
        kwargs = mock_custom.patch_namespaced_custom_object.call_args.kwargs
        assert kwargs["namespace"] == "aiperf-benchmarks"
        assert kwargs["name"] == "llama-3-8b-load"
        assert kwargs["body"] == {"spec": {"cancel": True}}
        mock_custom.create_namespaced_custom_object.assert_not_awaited()
