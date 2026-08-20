# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for operator pod-log retrieval.

Focuses on:
- required pod selection and owned-pod enforcement before log reads
- tail-line lower/upper bounds and malformed query rejection
- explicit and default container name validation at the router trust boundary
- Kubernetes API 404 vs 500 propagation from the log read path
- raw text/binary log payload preservation without accidental JSON schema drift
- multi-pod rosters where only the requested pod is tailed

Out of scope: UI log-strip rendering and event severity counters, covered by
``tests/unit/ui/test_operator_logs_events_edges.py`` and diagnostics router tests.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from aiperf.operator.routers.jobs import create_jobs_router
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from kubernetes_asyncio.client.exceptions import ApiException
from pytest import param

# ============================================================
# Helpers
# ============================================================


def _app(api: object | None, results_dir: Path) -> FastAPI:
    """Build the jobs router with the production Kubernetes exception shape."""
    app = FastAPI()

    @app.exception_handler(ApiException)
    async def _api_exception_handler(
        request: Request, exc: ApiException
    ) -> JSONResponse:
        del request
        return JSONResponse(
            status_code=exc.status or 500,
            content={"detail": str(exc.body or exc.reason or "Kubernetes API error")},
        )

    app.include_router(create_jobs_router([api], results_dir))
    return app


@pytest.fixture
async def client(tmp_path: Path) -> AsyncIterator[httpx.AsyncClient]:
    """HTTP client for pod-log routes with a live ApiClient token."""
    transport = httpx.ASGITransport(
        app=_app(object(), tmp_path), raise_app_exceptions=False
    )
    async with httpx.AsyncClient(
        transport=transport, base_url="http://aiperf.operator.local"
    ) as c:
        yield c


def _pod(
    name: str,
    *,
    containers: list[str] | None = None,
    default_container: str | None = None,
) -> MagicMock:
    """Build a V1Pod-shaped mock with the metadata/spec used by log helpers."""
    pod = MagicMock()
    pod.metadata = MagicMock()
    pod.metadata.name = name
    pod.metadata.annotations = {}
    if default_container is not None:
        pod.metadata.annotations["kubectl.kubernetes.io/default-container"] = (
            default_container
        )
    pod.spec = MagicMock()
    container_names = ["controller", "event-bus"] if containers is None else containers
    pod.spec.containers = [
        SimpleNamespace(name=container_name) for container_name in container_names
    ]
    pod.status = MagicMock()
    pod.status.phase = "Running"
    pod.status.container_statuses = []
    return pod


async def _get_logs(
    client: httpx.AsyncClient,
    *,
    params: dict[str, object] | None = None,
) -> httpx.Response:
    """Call the canonical pod-log endpoint for the diagnostics benchmark."""
    return await client.get(
        "/api/v1/jobs/aiperf-benchmarks/llama-3-8b-diagnostics/logs",
        params=params,
    )


# ============================================================
# Required pod selection and owned-pod enforcement
# ============================================================


class TestPodLogsPodSelection:
    """The endpoint tails only a selected pod owned by the requested job."""

    @pytest.mark.asyncio
    async def test_get_pod_logs_missing_pod_query_returns_422_before_pod_lookup(
        self, client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from aiperf.operator.routers import jobs_logs

        pod_lookup = AsyncMock()
        monkeypatch.setattr(jobs_logs, "get_pods", pod_lookup)

        response = await _get_logs(client)

        assert response.status_code == 422
        assert response.json()["detail"][0]["loc"] == ["query", "pod"]
        pod_lookup.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_get_pod_logs_unowned_pod_returns_404_and_skips_log_read(
        self, client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from aiperf.operator.routers import jobs_logs

        monkeypatch.setattr(
            jobs_logs,
            "get_pods",
            AsyncMock(return_value=[_pod("llama-3-8b-diagnostics-controller-0")]),
        )
        read_log = AsyncMock(return_value="should not be read\n")
        mock_core = MagicMock(read_namespaced_pod_log=read_log)
        monkeypatch.setattr(
            jobs_logs.client, "CoreV1Api", MagicMock(return_value=mock_core)
        )

        response = await _get_logs(
            client,
            params={"pod": "llama-3-8b-diagnostics-worker-7"},
        )

        assert response.status_code == 404
        assert "llama-3-8b-diagnostics-worker-7" in response.json()["detail"]
        assert "aiperf-benchmarks/llama-3-8b-diagnostics" in response.json()["detail"]
        read_log.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_get_pod_logs_multi_pod_roster_reads_only_requested_pod(
        self, client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from aiperf.operator.routers import jobs_logs

        pods = [
            _pod("llama-3-8b-diagnostics-controller-0"),
            _pod("llama-3-8b-diagnostics-worker-0", containers=["worker"]),
            _pod("llama-3-8b-diagnostics-worker-1", containers=["worker"]),
        ]
        monkeypatch.setattr(jobs_logs, "get_pods", AsyncMock(return_value=pods))
        read_log = AsyncMock(return_value="worker ready\n")
        mock_core = MagicMock(read_namespaced_pod_log=read_log)
        monkeypatch.setattr(
            jobs_logs.client, "CoreV1Api", MagicMock(return_value=mock_core)
        )

        response = await _get_logs(
            client,
            params={"pod": "llama-3-8b-diagnostics-worker-1"},
        )

        assert response.status_code == 200, response.text
        assert response.text == "worker ready\n"
        read_log.assert_awaited_once()
        assert read_log.call_args.kwargs["name"] == "llama-3-8b-diagnostics-worker-1"
        assert read_log.call_args.kwargs["container"] == "worker"


# ============================================================
# Tail-line limits and malformed query values
# ============================================================


class TestPodLogsTailLineLimits:
    """Tail limits are enforced at the API boundary before expensive reads."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "tail_lines",
        [
            param(1, id="min-boundary-accepted"),
            param(10_000, id="max-boundary-accepted"),
        ],
    )  # fmt: skip
    async def test_get_pod_logs_tail_line_boundaries_are_passed_to_apiserver(
        self,
        client: httpx.AsyncClient,
        monkeypatch: pytest.MonkeyPatch,
        tail_lines: int,
    ) -> None:
        from aiperf.operator.routers import jobs_logs

        monkeypatch.setattr(
            jobs_logs,
            "get_pods",
            AsyncMock(return_value=[_pod("llama-3-8b-diagnostics-controller-0")]),
        )
        read_log = AsyncMock(return_value="bounded log tail\n")
        mock_core = MagicMock(read_namespaced_pod_log=read_log)
        monkeypatch.setattr(
            jobs_logs.client, "CoreV1Api", MagicMock(return_value=mock_core)
        )

        response = await _get_logs(
            client,
            params={
                "pod": "llama-3-8b-diagnostics-controller-0",
                "tail_lines": tail_lines,
            },
        )

        assert response.status_code == 200, response.text
        assert read_log.call_args.kwargs["tail_lines"] == tail_lines

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "tail_lines,expected_status,expected_fragment",
        [
            param(0, 400, "tail_lines must be in [1, 10000]", id="zero-rejected"),
            param(-1, 400, "tail_lines must be in [1, 10000]", id="negative-rejected"),
            param(10_001, 400, "tail_lines must be in [1, 10000]", id="above-max-rejected"),
            param("not-a-number", 422, "tail_lines", id="non-integer-rejected-by-fastapi"),
        ],
    )  # fmt: skip
    async def test_get_pod_logs_invalid_tail_lines_return_client_error_before_lookup(
        self,
        client: httpx.AsyncClient,
        monkeypatch: pytest.MonkeyPatch,
        tail_lines: int | str,
        expected_status: int,
        expected_fragment: str,
    ) -> None:
        from aiperf.operator.routers import jobs_logs

        pod_lookup = AsyncMock()
        monkeypatch.setattr(jobs_logs, "get_pods", pod_lookup)

        response = await _get_logs(
            client,
            params={
                "pod": "llama-3-8b-diagnostics-controller-0",
                "tail_lines": tail_lines,
            },
        )

        assert response.status_code == expected_status
        assert expected_fragment in response.text
        pod_lookup.assert_not_awaited()


# ============================================================
# Container name validation and defaults
# ============================================================


class TestPodLogsContainerValidation:
    """Container names are trust-boundary inputs whether explicit or defaulted."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "container_name",
        [
            param("Controller", id="uppercase-rejected"),
            param("controller.0", id="dot-rejected"),
            param("controller/../../secrets", id="slash-smuggling-rejected"),
            param("-controller", id="leading-dash-rejected"),
        ],
    )  # fmt: skip
    async def test_get_pod_logs_invalid_container_query_returns_400_before_pod_lookup(
        self,
        client: httpx.AsyncClient,
        monkeypatch: pytest.MonkeyPatch,
        container_name: str,
    ) -> None:
        from aiperf.operator.routers import jobs_logs

        pod_lookup = AsyncMock()
        monkeypatch.setattr(jobs_logs, "get_pods", pod_lookup)

        response = await _get_logs(
            client,
            params={
                "pod": "llama-3-8b-diagnostics-controller-0",
                "container": container_name,
            },
        )

        assert response.status_code == 400
        assert (
            f"Invalid container name: {container_name!r}" in response.json()["detail"]
        )
        pod_lookup.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_get_pod_logs_invalid_default_container_annotation_returns_400_before_read(
        self, client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from aiperf.operator.routers import jobs_logs

        pod = _pod(
            "llama-3-8b-diagnostics-controller-0",
            containers=["controller", "results"],
            default_container="controller/../../secrets",
        )
        monkeypatch.setattr(jobs_logs, "get_pods", AsyncMock(return_value=[pod]))
        read_log = AsyncMock(return_value="annotation should not be trusted\n")
        mock_core = MagicMock(read_namespaced_pod_log=read_log)
        monkeypatch.setattr(
            jobs_logs.client, "CoreV1Api", MagicMock(return_value=mock_core)
        )

        response = await _get_logs(
            client,
            params={"pod": "llama-3-8b-diagnostics-controller-0"},
        )

        assert response.status_code == 400
        assert "Invalid container name" in response.json()["detail"]
        read_log.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_get_pod_logs_no_container_spec_passes_none_to_apiserver(
        self, client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from aiperf.operator.routers import jobs_logs

        pod = _pod("llama-3-8b-diagnostics-init-0", containers=[])
        monkeypatch.setattr(jobs_logs, "get_pods", AsyncMock(return_value=[pod]))
        read_log = AsyncMock(return_value="pod log from apiserver default\n")
        mock_core = MagicMock(read_namespaced_pod_log=read_log)
        monkeypatch.setattr(
            jobs_logs.client, "CoreV1Api", MagicMock(return_value=mock_core)
        )

        response = await _get_logs(
            client,
            params={"pod": "llama-3-8b-diagnostics-init-0"},
        )

        assert response.status_code == 200, response.text
        assert read_log.call_args.kwargs["container"] is None


# ============================================================
# Kubernetes API error propagation
# ============================================================


class TestPodLogsApiErrors:
    """404 and 500 from Kubernetes remain distinct for operators debugging pods."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "status,detail",
        [
            param(404, "pods 'controller-0' not found", id="read-404-preserved"),
            param(500, "apiserver log backend timeout", id="read-500-preserved"),
        ],
    )  # fmt: skip
    async def test_get_pod_logs_read_api_errors_preserve_status_and_body(
        self,
        client: httpx.AsyncClient,
        monkeypatch: pytest.MonkeyPatch,
        status: int,
        detail: str,
    ) -> None:
        from aiperf.operator.routers import jobs_logs

        api_error = ApiException(status=status, reason="Kubernetes API error")
        api_error.body = detail
        monkeypatch.setattr(
            jobs_logs,
            "get_pods",
            AsyncMock(return_value=[_pod("llama-3-8b-diagnostics-controller-0")]),
        )
        mock_core = MagicMock(read_namespaced_pod_log=AsyncMock(side_effect=api_error))
        monkeypatch.setattr(
            jobs_logs.client, "CoreV1Api", MagicMock(return_value=mock_core)
        )

        response = await _get_logs(
            client,
            params={"pod": "llama-3-8b-diagnostics-controller-0"},
        )

        assert response.status_code == status
        assert detail in response.json()["detail"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "status,detail",
        [
            param(404, "job pod list not found", id="pod-list-404-preserved"),
            param(500, "apiserver selector timeout", id="pod-list-500-preserved"),
        ],
    )  # fmt: skip
    async def test_get_pod_logs_pod_lookup_api_errors_preserve_status_and_body(
        self,
        client: httpx.AsyncClient,
        monkeypatch: pytest.MonkeyPatch,
        status: int,
        detail: str,
    ) -> None:
        from aiperf.operator.routers import jobs_logs

        api_error = ApiException(status=status, reason="Kubernetes API error")
        api_error.body = detail
        monkeypatch.setattr(jobs_logs, "get_pods", AsyncMock(side_effect=api_error))

        response = await _get_logs(
            client,
            params={"pod": "llama-3-8b-diagnostics-controller-0"},
        )

        assert response.status_code == status
        assert detail in response.json()["detail"]


# ============================================================
# Raw content and response schema stability
# ============================================================


class TestPodLogsResponseContent:
    """Log responses stay raw text/plain and preserve diagnostic payload bytes."""

    @pytest.mark.asyncio
    async def test_get_pod_logs_unicode_severity_words_are_raw_text_not_counts(
        self, client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from aiperf.operator.routers import jobs_logs

        log_text = (
            "INFO profiler started for meta-llama/Llama-3-8B\n"
            "WARN GPU temperature 74°C on dgx-node-01\n"
            "ERROR tokenizer retry for café prompt\n"
        )
        monkeypatch.setattr(
            jobs_logs,
            "get_pods",
            AsyncMock(return_value=[_pod("llama-3-8b-diagnostics-controller-0")]),
        )
        read_log = AsyncMock(return_value=log_text)
        mock_core = MagicMock(read_namespaced_pod_log=read_log)
        monkeypatch.setattr(
            jobs_logs.client, "CoreV1Api", MagicMock(return_value=mock_core)
        )

        response = await _get_logs(
            client,
            params={"pod": "llama-3-8b-diagnostics-controller-0"},
        )

        assert response.status_code == 200, response.text
        assert response.headers["content-type"].startswith("text/plain")
        assert response.text == log_text
        assert "warn" not in response.headers
        with pytest.raises(ValueError, match="Expecting value"):
            response.json()

    @pytest.mark.asyncio
    async def test_get_pod_logs_binary_bytes_are_preserved_under_text_plain_contract(
        self, client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from aiperf.operator.routers import jobs_logs

        raw_bytes = b"\xff\xfeERROR binary tokenizer frame\n\x00WARN partial utf8\n"
        monkeypatch.setattr(
            jobs_logs,
            "get_pods",
            AsyncMock(return_value=[_pod("llama-3-8b-diagnostics-controller-0")]),
        )
        read_log = AsyncMock(return_value=raw_bytes)
        mock_core = MagicMock(read_namespaced_pod_log=read_log)
        monkeypatch.setattr(
            jobs_logs.client, "CoreV1Api", MagicMock(return_value=mock_core)
        )

        response = await _get_logs(
            client,
            params={"pod": "llama-3-8b-diagnostics-controller-0"},
        )

        assert response.status_code == 200, response.text
        assert response.headers["content-type"].startswith("text/plain")
        assert response.content == raw_bytes

    @pytest.mark.asyncio
    async def test_get_pod_logs_empty_log_keeps_empty_text_plain_response(
        self, client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from aiperf.operator.routers import jobs_logs

        monkeypatch.setattr(
            jobs_logs,
            "get_pods",
            AsyncMock(return_value=[_pod("llama-3-8b-diagnostics-controller-0")]),
        )
        read_log = AsyncMock(return_value=None)
        mock_core = MagicMock(read_namespaced_pod_log=read_log)
        monkeypatch.setattr(
            jobs_logs.client, "CoreV1Api", MagicMock(return_value=mock_core)
        )

        response = await _get_logs(
            client,
            params={"pod": "llama-3-8b-diagnostics-controller-0"},
        )

        assert response.status_code == 200, response.text
        assert response.headers["content-type"].startswith("text/plain")
        assert response.text == ""
        assert response.content == b""
