# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for operator config HTTP routes.

Focuses on:
- live-CR fallback behavior for ``/api/v1/config/{namespace}/{job_id}``
- missing CRs, malformed live CR shapes, and Kubernetes API error boundaries
- namespace/job path encoding and stable response schemas for dashboard clients
- secret-redaction and no-mutation guarantees when exposing CR specs

Out of scope: analytics SQL behavior and results-file download boundaries, covered by
sibling operator router tests.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from kubernetes_asyncio.client.exceptions import ApiException

from aiperf.common.redact import REDACTED_VALUE
from aiperf.operator import runs_index
from aiperf.operator.results_db import ResultsDB
from aiperf.operator.routers import results_analytics as mod
from aiperf.operator.routers.config import create_config_router
from aiperf.operator.routers.results_analytics import create_results_analytics_router

# ============================================================
# Helpers
# ============================================================


@dataclass(slots=True)
class _ConfigRouteSubject:
    """Mounted config routes plus the DB object that owns their fallback state."""

    app: FastAPI
    db: ResultsDB


def _make_config_route_subject(
    tmp_path: Path,
    *,
    api_holder: list[object | None] | None = None,
) -> _ConfigRouteSubject:
    """Build only the config-related routers for route-boundary tests."""
    db = ResultsDB(tmp_path)
    app = FastAPI()
    app.include_router(create_config_router())
    app.include_router(
        create_results_analytics_router(
            lambda: db,
            tmp_path,
            api_holder if api_holder is not None else [object()],
        )
    )
    return _ConfigRouteSubject(app=app, db=db)


@pytest.fixture
async def open_runs_index(tmp_path: Path) -> AsyncIterator[None]:
    """Open an empty runs index so config lookup misses are deterministic."""
    await runs_index.close()
    await runs_index.open(tmp_path / ".aiperf_index.sqlite")
    try:
        yield
    finally:
        await runs_index.close()


# ============================================================
# Missing CRs and malformed live payloads
# ============================================================


class TestConfigRouterLiveCrFallback:
    """Live CR fallback returns only valid spec data or a stable not-found response."""

    @pytest.mark.asyncio
    async def test_get_job_config_missing_cr_returns_404_with_namespace_and_job(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        open_runs_index: None,
    ) -> None:
        async def fake_get_raw_aiperfjob(
            api: object, namespace: str, name: str
        ) -> dict[str, object] | None:
            del api, namespace, name
            return None

        monkeypatch.setattr(mod, "get_raw_aiperfjob", fake_get_raw_aiperfjob)
        subject = _make_config_route_subject(tmp_path)
        try:
            with TestClient(subject.app) as client:
                response = client.get("/api/v1/config/aiperf-bench/missing-gpt-oss")

            assert response.status_code == 404
            assert response.json()["detail"] == (
                "No config found for aiperf-bench/missing-gpt-oss"
            )
        finally:
            subject.db.close()

    @pytest.mark.asyncio
    async def test_get_job_config_malformed_live_spec_returns_404_not_scalar_spec(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        open_runs_index: None,
    ) -> None:
        async def fake_get_raw_aiperfjob(
            api: object, namespace: str, name: str
        ) -> dict[str, object]:
            del api, namespace, name
            return {
                "metadata": {"namespace": "aiperf-bench", "name": "malformed-spec"},
                "spec": "benchmark: should-have-been-a-mapping",
            }

        monkeypatch.setattr(mod, "get_raw_aiperfjob", fake_get_raw_aiperfjob)
        subject = _make_config_route_subject(tmp_path)
        try:
            with TestClient(subject.app) as client:
                response = client.get("/api/v1/config/aiperf-bench/malformed-spec")

            assert response.status_code == 404
            assert "malformed-spec" in response.json()["detail"]
        finally:
            subject.db.close()

    @pytest.mark.asyncio
    async def test_get_job_config_malformed_status_is_ignored_and_not_exposed(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        open_runs_index: None,
    ) -> None:
        async def fake_get_raw_aiperfjob(
            api: object, namespace: str, name: str
        ) -> dict[str, object]:
            del api, namespace, name
            return {
                "metadata": {"namespace": "aiperf-bench", "name": "live-llama-70b"},
                "status": "operator-status-should-not-leak",
                "spec": {"benchmark": {"models": ["meta-llama/Llama-3.1-70B"]}},
            }

        monkeypatch.setattr(mod, "get_raw_aiperfjob", fake_get_raw_aiperfjob)
        subject = _make_config_route_subject(tmp_path)
        try:
            with TestClient(subject.app) as client:
                response = client.get("/api/v1/config/aiperf-bench/live-llama-70b")

            assert response.status_code == 200
            body = response.json()
            assert set(body) == {"source", "spec"}
            assert body["source"] == "cr"
            assert body["spec"] == {
                "benchmark": {"models": ["meta-llama/Llama-3.1-70B"]}
            }
            assert "operator-status-should-not-leak" not in response.text
        finally:
            subject.db.close()


# ============================================================
# Kubernetes API error boundaries
# ============================================================


class TestConfigRouterKubernetesApiErrors:
    """Kubernetes 404s are misses; non-404 API failures are not mislabeled as misses."""

    @pytest.mark.asyncio
    async def test_get_job_config_kube_api_404_returns_missing_config_404(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        open_runs_index: None,
    ) -> None:
        class FakeCustomObjectsApi:
            def __init__(self, api: object) -> None:
                del api

            async def get_namespaced_custom_object(
                self,
                **kwargs: object,
            ) -> dict[str, object]:
                assert kwargs["namespace"] == "aiperf-bench"
                assert kwargs["name"] == "deleted-job"
                raise ApiException(status=404, reason="Not Found")

        monkeypatch.setattr(
            "aiperf.kubernetes.client.client.CustomObjectsApi", FakeCustomObjectsApi
        )
        subject = _make_config_route_subject(tmp_path)
        try:
            with TestClient(subject.app, raise_server_exceptions=False) as client:
                response = client.get("/api/v1/config/aiperf-bench/deleted-job")

            assert response.status_code == 404
            assert (
                response.json()["detail"]
                == "No config found for aiperf-bench/deleted-job"
            )
        finally:
            subject.db.close()

    @pytest.mark.asyncio
    async def test_historical_config_miss_never_falls_back_to_live_cr(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, open_runs_index: None
    ) -> None:
        class FakeCustomObjectsApi:
            def __init__(self, api: object) -> None:
                del api

            async def get_namespaced_custom_object(
                self, **kwargs: object
            ) -> dict[str, object]:
                raise AssertionError(f"historical request reached live CR: {kwargs}")

        monkeypatch.setattr(
            "aiperf.kubernetes.client.client.CustomObjectsApi", FakeCustomObjectsApi
        )
        subject = _make_config_route_subject(tmp_path)
        try:
            with TestClient(subject.app, raise_server_exceptions=False) as client:
                response = client.get(
                    "/api/v1/config/aiperf-bench/deleted-job?epoch=1700000000"
                )
            assert response.status_code == 404
        finally:
            subject.db.close()

    @pytest.mark.asyncio
    async def test_get_job_config_kube_api_500_returns_unavailable_not_missing(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        open_runs_index: None,
    ) -> None:
        class FakeCustomObjectsApi:
            def __init__(self, api: object) -> None:
                del api

            async def get_namespaced_custom_object(
                self,
                **kwargs: object,
            ) -> dict[str, object]:
                assert kwargs["namespace"] == "aiperf-bench"
                assert kwargs["name"] == "live-job"
                raise ApiException(status=500, reason="apiserver overloaded")

        monkeypatch.setattr(
            "aiperf.kubernetes.client.client.CustomObjectsApi", FakeCustomObjectsApi
        )
        subject = _make_config_route_subject(tmp_path)
        try:
            with TestClient(subject.app, raise_server_exceptions=False) as client:
                response = client.get("/api/v1/config/aiperf-bench/live-job")

            assert response.status_code == 503
            assert response.json()["detail"] == (
                "Could not read live AIPerfJob config for aiperf-bench/live-job: "
                "Kubernetes API returned 500 apiserver overloaded"
            )
        finally:
            subject.db.close()


# ============================================================
# Path encoding and response schemas
# ============================================================


class TestConfigRouterEncodingAndSchema:
    """Path parameters and response shapes remain stable for the operator SPA."""

    @pytest.mark.asyncio
    async def test_get_job_config_url_encoded_namespace_and_job_fetches_exact_cr(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        open_runs_index: None,
    ) -> None:
        captured: list[tuple[str, str]] = []

        async def fake_get_raw_aiperfjob(
            api: object, namespace: str, name: str
        ) -> dict[str, object]:
            del api
            captured.append((namespace, name))
            return {
                "metadata": {"namespace": namespace, "name": name},
                "spec": {"benchmark": {"models": ["nvidia/llama-3.1-nemotron"]}},
            }

        monkeypatch.setattr(mod, "get_raw_aiperfjob", fake_get_raw_aiperfjob)
        subject = _make_config_route_subject(tmp_path)
        try:
            with TestClient(subject.app) as client:
                # Valid Kubernetes identifiers; the encoded dash/dot still
                # exercises percent-decoding while passing path validation.
                response = client.get(
                    "/api/v1/config/team%2Dalpha/llama-3%2E1-8b-draft"
                )

            assert response.status_code == 200
            assert captured == [("team-alpha", "llama-3.1-8b-draft")]
            assert response.json()["source"] == "cr"
        finally:
            subject.db.close()

    @pytest.mark.asyncio
    async def test_get_job_config_live_cr_response_schema_excludes_metadata_status_and_kind(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        open_runs_index: None,
    ) -> None:
        async def fake_get_raw_aiperfjob(
            api: object, namespace: str, name: str
        ) -> dict[str, object]:
            del api
            return {
                "apiVersion": "aiperf.nvidia.com/v1alpha1",
                "kind": "AIPerfJob",
                "metadata": {"namespace": namespace, "name": name, "uid": "uid-7f2a"},
                "status": {"phase": "Running", "apiUrl": "http://operator:8081"},
                "spec": {"benchmark": {"slos": {"time_to_first_token": 750}}},
            }

        monkeypatch.setattr(mod, "get_raw_aiperfjob", fake_get_raw_aiperfjob)
        subject = _make_config_route_subject(tmp_path)
        try:
            with TestClient(subject.app) as client:
                response = client.get("/api/v1/config/aiperf-bench/live-slo-job")

            assert response.status_code == 200
            assert response.json() == {
                "source": "cr",
                "spec": {"benchmark": {"slos": {"time_to_first_token": 750}}},
            }
        finally:
            subject.db.close()

    def test_operator_settings_config_routes_have_stable_schema_keys(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "aiperf.operator.environment.OperatorEnvironment.RESULTS.RETAIN_RUNS", 31
        )
        monkeypatch.setattr(
            "aiperf.operator.environment.OperatorEnvironment.RESULTS.RETAIN_DAYS", 14
        )
        monkeypatch.setattr(
            "aiperf.operator.environment.OperatorEnvironment.DASHBOARD.PROXY_ENABLED",
            True,
        )
        subject = _make_config_route_subject(tmp_path, api_holder=[None])
        try:
            with TestClient(subject.app) as client:
                retention = client.get("/api/v1/config/retention")
                features = client.get("/api/v1/config/features")

            assert retention.status_code == 200
            assert retention.json() == {"retain_runs": 31, "retain_days": 14}
            assert features.status_code == 200
            assert features.json() == {"dashboard_enabled": True}
        finally:
            subject.db.close()


# ============================================================
# Config exposure boundaries
# ============================================================


class TestConfigRouterExposureBoundaries:
    """Exposed specs are safe for browsers and do not mutate operator-owned CR data."""

    @pytest.mark.asyncio
    async def test_get_job_config_live_cr_redacts_endpoint_secrets_without_mutating_raw_cr(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        open_runs_index: None,
    ) -> None:
        raw_cr: dict[str, Any] = {
            "metadata": {"namespace": "aiperf-bench", "name": "secret-bearing-job"},
            "spec": {
                "benchmark": {
                    "endpoint": {
                        "urls": [
                            "https://user:pass@api.example.invalid/v1"
                            "?api_key=query-secret&model=m"
                        ],
                        "apiKey": "sk-live-should-not-reach-browser",
                        "headers": {
                            "Authorization": "Bearer operator-service-account-token",
                            "X-API-Key": "provider-key-2026-05-18",
                            "X-AIPerf-Trace": "conv-2026-04-21-9c3a",
                        },
                    }
                }
            },
        }
        original_raw_cr = deepcopy(raw_cr)

        async def fake_get_raw_aiperfjob(
            api: object, namespace: str, name: str
        ) -> dict[str, Any]:
            del api, namespace, name
            return raw_cr

        monkeypatch.setattr(mod, "get_raw_aiperfjob", fake_get_raw_aiperfjob)
        subject = _make_config_route_subject(tmp_path)
        try:
            with TestClient(subject.app) as client:
                response = client.get("/api/v1/config/aiperf-bench/secret-bearing-job")

            assert response.status_code == 200
            body = response.json()
            endpoint = body["spec"]["benchmark"]["endpoint"]
            assert endpoint["apiKey"] == REDACTED_VALUE
            assert endpoint["urls"] == [
                f"https://{REDACTED_VALUE}@api.example.invalid/v1"
                f"?api_key={REDACTED_VALUE}&model=m"
            ]
            assert endpoint["headers"]["Authorization"] == REDACTED_VALUE
            assert endpoint["headers"]["X-API-Key"] == REDACTED_VALUE
            assert endpoint["headers"]["X-AIPerf-Trace"] == "conv-2026-04-21-9c3a"
            assert "sk-live-should-not-reach-browser" not in response.text
            assert "operator-service-account-token" not in response.text
            assert "user:pass" not in response.text
            assert "query-secret" not in response.text
            assert raw_cr == original_raw_cr
        finally:
            subject.db.close()

    @pytest.mark.asyncio
    async def test_get_job_config_live_cr_uses_redaction_copy_for_nested_headers(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        open_runs_index: None,
    ) -> None:
        raw_spec: dict[str, object] = {
            "benchmark": {
                "endpoint": {
                    "headers": {"api-key": "provider-key", "X-Trace-ID": "trace-7f2a"}
                }
            }
        }

        async def fake_get_raw_aiperfjob(
            api: object, namespace: str, name: str
        ) -> dict[str, object]:
            del api, namespace, name
            return {"spec": raw_spec}

        monkeypatch.setattr(mod, "get_raw_aiperfjob", fake_get_raw_aiperfjob)
        subject = _make_config_route_subject(tmp_path)
        try:
            with TestClient(subject.app) as client:
                first = client.get("/api/v1/config/aiperf-bench/header-redaction-job")
                second = client.get("/api/v1/config/aiperf-bench/header-redaction-job")

            def exposed_headers(response: object) -> dict[str, str]:
                json_response = response.json()
                return json_response["spec"]["benchmark"]["endpoint"]["headers"]

            assert exposed_headers(first) == {
                "api-key": REDACTED_VALUE,
                "X-Trace-ID": "trace-7f2a",
            }
            assert exposed_headers(second) == exposed_headers(first)
            assert raw_spec == {
                "benchmark": {
                    "endpoint": {
                        "headers": {
                            "api-key": "provider-key",
                            "X-Trace-ID": "trace-7f2a",
                        }
                    }
                }
            }
        finally:
            subject.db.close()
