# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for operator admin index rebuild APIs and CLI.

Focuses on:
- concurrent ``POST /admin/index/rebuild`` requests against the writer process
- rebuild failure propagation without success-shaped stale payloads
- request and response schema boundaries for the manual recovery hatch
- ``aiperf kube index rebuild`` JSON/text output cleanliness and timeout handling
- missing operator API auto-discovery diagnostics when ``--api-url`` is omitted

Out of scope: runs_index filesystem-walk correctness; see sibling
``tests/unit/operator/test_runs_index_adversarial.py`` and
``tests/unit/operator/test_runs_index_edge_cases.py``.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Literal
from unittest.mock import MagicMock, patch

import httpx
import orjson
import pytest
from fastapi import FastAPI

from aiperf.config.kube import KubeManageOptions
from aiperf.operator import runs_index
from aiperf.operator.routers.admin import create_admin_router

# ============================================================================
# Helpers
# ============================================================================


class _FakeResponse:
    """Minimal HTTP response that exposes bytes content and status errors."""

    def __init__(
        self,
        payload: dict[str, object],
        *,
        status_code: int = 200,
        reason: str = "OK",
    ) -> None:
        self.content = orjson.dumps(payload)
        self.status_code = status_code
        self._reason = reason

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"{self.status_code} {self._reason}",
                request=MagicMock(),
                response=MagicMock(status_code=self.status_code),
            )


class _FakeAsyncClient:
    """HTTP client stand-in that records base URL, timeout, and POST paths."""

    def __init__(
        self,
        *,
        post_payload: dict[str, object] | None = None,
        post_exc: Exception | None = None,
    ) -> None:
        self._post_payload = post_payload or {}
        self._post_exc = post_exc
        self.base_url: str | None = None
        self.timeout: float | None = None
        self.post_calls: list[str] = []
        self.post_headers: list[dict[str, str] | None] = []

    async def __aenter__(self) -> _FakeAsyncClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def post(
        self, url: str, *, headers: dict[str, str] | None = None
    ) -> _FakeResponse:
        self.post_calls.append(url)
        self.post_headers.append(headers)
        if self._post_exc is not None:
            raise self._post_exc
        return _FakeResponse(self._post_payload)


class _RecordingConsole:
    """Console replacement that captures exactly what the CLI would print."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def print(self, message: str) -> None:
        self.lines.append(message)


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


def _patch_httpx(client: _FakeAsyncClient) -> object:
    """Patch the CLI's ``httpx.AsyncClient`` factory and capture constructor args."""

    def _factory(**kwargs: object) -> _FakeAsyncClient:
        base_url = kwargs.get("base_url")
        timeout = kwargs.get("timeout")
        client.base_url = str(base_url) if base_url is not None else None
        client.timeout = float(timeout) if timeout is not None else None
        return client

    return patch("aiperf.cli_commands.kube.index.httpx.AsyncClient", _factory)


@asynccontextmanager
async def _fixed_operator_api_base(url: str) -> AsyncIterator[str]:
    """Yield a deterministic operator API base URL without Kubernetes I/O."""
    yield url


# ============================================================================
# Admin rebuild API
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


# ============================================================================
# CLI rebuild behavior
# ============================================================================


class TestKubeIndexRebuildCli:
    """Validate ``aiperf kube index rebuild`` at its HTTP and output boundaries."""

    @pytest.mark.asyncio
    async def test_rebuild_json_output_is_single_parseable_payload(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """JSON mode must emit machine-readable JSON with no text banner."""
        from aiperf.cli_commands.kube import index

        payload = {
            "runs_indexed": 9,
            "sweep_variations_indexed": 2,
            "duration_seconds": 3.5,
        }
        client = _FakeAsyncClient(post_payload=payload)
        console = _RecordingConsole()
        monkeypatch.setattr(index.kube_console, "console", console)
        monkeypatch.setattr(
            index,
            "_operator_api_base",
            lambda api_url, options, operator_namespace=None: _fixed_operator_api_base(
                api_url or "http://localhost:39081"
            ),
        )

        with _patch_httpx(client):
            await index.rebuild(output="json")

        assert client.post_calls == ["/admin/index/rebuild"]
        assert client.timeout == 300.0
        assert console.lines == [
            orjson.dumps(payload, option=orjson.OPT_INDENT_2).decode()
        ]
        assert orjson.loads(console.lines[0]) == payload
        assert "Indexed" not in console.lines[0]

    @pytest.mark.asyncio
    async def test_rebuild_text_output_is_human_summary_without_json_braces(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Text mode should be concise operator feedback, not mixed JSON."""
        from aiperf.cli_commands.kube import index

        client = _FakeAsyncClient(
            post_payload={
                "runs_indexed": 5,
                "sweep_variations_indexed": 1,
                "duration_seconds": 1.236,
            }
        )
        console = _RecordingConsole()
        monkeypatch.setattr(index.kube_console, "console", console)
        monkeypatch.setattr(
            index,
            "_operator_api_base",
            lambda api_url, options, operator_namespace=None: _fixed_operator_api_base(
                api_url or "http://localhost:39081"
            ),
        )

        with _patch_httpx(client):
            await index.rebuild(output="text")

        assert console.lines == ["Indexed 5 runs and 1 sweep variations in 1.24s"]
        assert "{" not in console.lines[0]
        assert "}" not in console.lines[0]

    @pytest.mark.asyncio
    async def test_rebuild_timeout_propagates_read_timeout_with_rebuild_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A slow rebuild should surface httpx timeout instead of fake success."""
        from aiperf.cli_commands.kube import index

        client = _FakeAsyncClient(post_exc=httpx.ReadTimeout("rebuild timed out"))
        monkeypatch.setattr(
            index,
            "_operator_api_base",
            lambda api_url, options, operator_namespace=None: _fixed_operator_api_base(
                api_url or "http://localhost:39081"
            ),
        )

        with _patch_httpx(client), pytest.raises(httpx.ReadTimeout, match="timed out"):
            await index.rebuild()

        assert client.timeout == 300.0
        assert client.post_calls == ["/admin/index/rebuild"]

    @pytest.mark.asyncio
    async def test_rebuild_json_mode_restores_logger_level_after_timeout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The JSON-output log suppression must not leave later kube logs muted."""
        from aiperf.cli_commands.kube import index

        kube_logger = logging.getLogger("aiperf.kube")
        kube_logger.setLevel(logging.INFO)
        client = _FakeAsyncClient(post_exc=httpx.ReadTimeout("rebuild timed out"))
        monkeypatch.setattr(
            index,
            "_operator_api_base",
            lambda api_url, options, operator_namespace=None: _fixed_operator_api_base(
                api_url or "http://localhost:39081"
            ),
        )

        with _patch_httpx(client), pytest.raises(httpx.ReadTimeout):
            await index.rebuild(output="json")

        assert kube_logger.level == logging.INFO

    @pytest.mark.asyncio
    async def test_operator_api_base_missing_operator_pod_error_names_namespace(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Omitting --api-url should fail with namespace context and bypass guidance."""
        from aiperf.cli_commands.kube import index
        from aiperf.kubernetes import client as kube_client

        @asynccontextmanager
        async def fake_k8s_client(
            *, kubeconfig: str | None = None, context: str | None = None
        ) -> AsyncIterator[Literal["fake-api"]]:
            assert kubeconfig == "/home/bench/.kube/prod.yaml"
            assert context == "kind-aiperf-prod"
            yield "fake-api"

        async def fake_resolve_operator_namespace(
            api: Literal["fake-api"], *, explicit: str | None = None
        ) -> str:
            assert api == "fake-api"
            assert explicit == "bench-operator"
            return "bench-operator"

        async def fake_find_operator_pod(
            api: Literal["fake-api"], *, namespace: str
        ) -> None:
            assert api == "fake-api"
            assert namespace == "bench-operator"
            return None

        monkeypatch.setattr(kube_client, "k8s_client", fake_k8s_client)
        monkeypatch.setattr(
            kube_client, "resolve_operator_namespace", fake_resolve_operator_namespace
        )
        monkeypatch.setattr(kube_client, "find_operator_pod", fake_find_operator_pod)

        options = KubeManageOptions(
            kubeconfig="/home/bench/.kube/prod.yaml",
            kube_context="kind-aiperf-prod",
        )

        with pytest.raises(
            RuntimeError,
            match=r"Operator pod not found in namespace 'bench-operator'.*Pass --api-url",
        ):
            async with index._operator_api_base(
                None, options, operator_namespace="bench-operator"
            ):
                raise AssertionError("missing operator pod must fail before yielding")
