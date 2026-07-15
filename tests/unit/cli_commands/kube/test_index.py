# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for `aiperf kube index` cyclopts subcommand wiring.

Focus is on:
- module exposes `app` cyclopts.App; subcommands `stats` and `rebuild` registered
- end-to-end: monkeypatched httpx.AsyncClient routed through GET /admin/index/stats
  and POST /admin/index/rebuild; happy-path text + json output paths render
- httpx errors raise out of the helper (no exit_on_error wrapper inside this module)
- json mode toggles `aiperf.kube` logger to WARNING for the duration of the call
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Mapping
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import orjson
import pytest

_API_URL = "http://operator"


class _FakeAsyncClient:
    """Minimal stand-in for ``httpx.AsyncClient`` that records calls."""

    def __init__(
        self,
        *,
        get_payload: dict[str, Any] | None = None,
        post_payload: dict[str, Any] | None = None,
        get_exc: Exception | None = None,
        post_exc: Exception | None = None,
        post_status_code: int = 200,
        base_url: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self._get_payload = get_payload or {}
        self._post_payload = post_payload or {}
        self._get_exc = get_exc
        self._post_exc = post_exc
        self._post_status_code = post_status_code
        self.base_url = base_url
        self.timeout = timeout
        self.get_calls: list[str] = []
        self.post_calls: list[str] = []
        self.post_headers: list[dict[str, str]] = []

    async def __aenter__(self) -> _FakeAsyncClient:
        return self

    async def __aexit__(self, *_: Any) -> None:
        return None

    async def get(self, url: str) -> Any:
        self.get_calls.append(url)
        if self._get_exc is not None:
            raise self._get_exc
        return _FakeResponse(self._get_payload)

    async def post(self, url: str, *, headers: Mapping[str, str] | None = None) -> Any:
        self.post_calls.append(url)
        self.post_headers.append(dict(headers or {}))
        if self._post_exc is not None:
            raise self._post_exc
        return _FakeResponse(self._post_payload, status_code=self._post_status_code)


class _FakeResponse:
    """Minimal httpx.Response stand-in with byte content."""

    def __init__(self, payload: dict[str, Any], *, status_code: int = 200) -> None:
        self._payload = payload
        self.content = orjson.dumps(payload)
        self.status_code = status_code

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code < 400:
            return
        request = httpx.Request("POST", "http://operator/admin/index/rebuild")
        response = httpx.Response(self.status_code, request=request)
        raise httpx.HTTPStatusError(
            f"{self.status_code} Error", request=request, response=response
        )


def _patch_httpx(client: _FakeAsyncClient) -> Any:
    """Return a context manager that swaps httpx.AsyncClient with a factory.

    The factory captures ``base_url=`` and ``timeout=`` onto the fake client so
    we can assert the helper used the right URL/timeout.
    """

    def _factory(**kwargs: Any) -> _FakeAsyncClient:
        client.base_url = kwargs.get("base_url")
        client.timeout = kwargs.get("timeout")
        return client

    return patch("aiperf.cli_commands.kube.index.httpx.AsyncClient", _factory)


def test_index_module_importable() -> None:
    from aiperf.cli_commands.kube import index

    assert hasattr(index, "app"), "index.app (cyclopts App) must be defined"


def test_index_registered_in_kube_app() -> None:
    from aiperf.cli_commands.kube._app import app

    assert "index" in set(app)


class TestIndexSubcommandRegistration:
    """Both stats and rebuild must be registered on the index sub-app."""

    @pytest.mark.parametrize(
        "subcommand",
        [
            "stats",
            "rebuild",
        ],
    )  # fmt: skip
    def test_subcommand_registered(self, subcommand: str) -> None:
        from aiperf.cli_commands.kube.index import app

        assert subcommand in set(app)


class TestIndexCallableSignatures:
    """stats and rebuild expose --output, --api-url, options."""

    @pytest.mark.parametrize(
        "func_name,param_name",
        [
            ("stats", "output"),
            ("stats", "api_url"),
            ("stats", "options"),
            ("rebuild", "output"),
            ("rebuild", "api_url"),
            ("rebuild", "options"),
        ],
    )  # fmt: skip
    def test_signature_has_param(self, func_name: str, param_name: str) -> None:
        from aiperf.cli_commands.kube import index

        sig = inspect.signature(getattr(index, func_name))
        assert param_name in sig.parameters

    def test_default_api_url_is_none_for_auto_resolve(self) -> None:
        """Default ``api_url=None`` triggers cluster-wide pod-label discovery
        + auto port-forward to the results-server container — same pattern
        as ``aiperf kube results list-runs``. The pre-collapse default of
        ``http://localhost:38465`` was a magic-port hardcode that silently
        failed when the user didn't have an external port-forward pinned.
        """
        from aiperf.cli_commands.kube import index

        for func_name in ("stats", "rebuild"):
            sig = inspect.signature(getattr(index, func_name))
            assert sig.parameters["api_url"].default is None
            assert sig.parameters["output"].default == "text"


class TestIndexStats:
    """End-to-end: GET /admin/index/stats."""

    @pytest.mark.asyncio
    async def test_text_output_renders_one_line_summary(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from aiperf.cli_commands.kube.index import stats

        payload = {
            "runs_count": 12,
            "sweep_variations_count": 4,
            "db_bytes": 8192,
            "schema_version": 3,
            "last_bootstrap_unix": 1714600000,
        }
        client = _FakeAsyncClient(get_payload=payload)
        with _patch_httpx(client):
            await stats(api_url=_API_URL)

        out = capsys.readouterr().out
        assert client.get_calls == ["/admin/index/stats"]
        assert "runs=12" in out
        assert "sweep_variations=4" in out
        assert "size=8192B" in out
        assert "schema_version=3" in out
        assert "last_bootstrap_unix=1714600000" in out

    @pytest.mark.asyncio
    async def test_json_output_emits_indent2_orjson(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from aiperf.cli_commands.kube.index import stats

        payload = {
            "runs_count": 1,
            "sweep_variations_count": 0,
            "db_bytes": 100,
            "schema_version": 2,
            "last_bootstrap_unix": 1,
        }
        client = _FakeAsyncClient(get_payload=payload)
        with _patch_httpx(client):
            await stats(output="json", api_url=_API_URL)

        out = capsys.readouterr().out
        # orjson with OPT_INDENT_2 emits a 2-space indented dump
        assert orjson.loads(out) == payload
        assert "  " in out  # indented

    @pytest.mark.asyncio
    async def test_api_url_propagates_to_httpx_client(self) -> None:
        from aiperf.cli_commands.kube.index import stats

        payload = {
            "runs_count": 0,
            "sweep_variations_count": 0,
            "db_bytes": 0,
            "schema_version": 1,
            "last_bootstrap_unix": 0,
        }
        client = _FakeAsyncClient(get_payload=payload)
        with _patch_httpx(client):
            await stats(api_url="http://other-host:9999")

        assert client.base_url == "http://other-host:9999"

    @pytest.mark.asyncio
    async def test_http_error_propagates(self) -> None:
        """The stats command does NOT wrap in exit_on_error - errors bubble up."""
        from aiperf.cli_commands.kube.index import stats

        client = _FakeAsyncClient(
            get_exc=httpx.HTTPStatusError(
                "500 Server Error",
                request=MagicMock(),
                response=MagicMock(status_code=500),
            )
        )
        with _patch_httpx(client), pytest.raises(httpx.HTTPStatusError):
            await stats(api_url=_API_URL)

    @pytest.mark.asyncio
    async def test_json_mode_restores_logger_level_on_error(self) -> None:
        """The finally block must restore the original aiperf.kube level."""
        from aiperf.cli_commands.kube.index import stats

        kube_logger = logging.getLogger("aiperf.kube")
        kube_logger.setLevel(logging.ERROR)

        client = _FakeAsyncClient(get_exc=RuntimeError("boom"))
        with _patch_httpx(client), pytest.raises(RuntimeError):
            await stats(output="json", api_url=_API_URL)

        assert kube_logger.level == logging.ERROR


class TestIndexRebuild:
    """End-to-end: POST /admin/index/rebuild."""

    @pytest.mark.asyncio
    async def test_text_output_renders_summary_line(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from aiperf.cli_commands.kube.index import rebuild

        payload = {
            "runs_indexed": 7,
            "sweep_variations_indexed": 2,
            "duration_seconds": 1.234,
        }
        client = _FakeAsyncClient(post_payload=payload)
        with _patch_httpx(client):
            await rebuild(api_url=_API_URL)

        out = capsys.readouterr().out
        assert client.post_calls == ["/admin/index/rebuild"]
        assert client.post_headers == [{}]
        assert "Indexed 7 runs" in out
        assert "2 sweep variations" in out
        assert "1.23s" in out

    @pytest.mark.asyncio
    async def test_rebuild_sends_bearer_token_from_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from aiperf.cli_commands.kube.index import rebuild

        monkeypatch.setenv("AIPERF_OPERATOR_MUTATING_ROUTES_TOKEN", "secret-token")
        client = _FakeAsyncClient(
            post_payload={
                "runs_indexed": 1,
                "sweep_variations_indexed": 0,
                "duration_seconds": 0.0,
            }
        )
        with _patch_httpx(client):
            await rebuild(api_url=_API_URL)

        assert client.post_headers == [{"Authorization": "Bearer secret-token"}]

    @pytest.mark.asyncio
    async def test_json_output_emits_indent2_orjson(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from aiperf.cli_commands.kube.index import rebuild

        payload = {
            "runs_indexed": 0,
            "sweep_variations_indexed": 0,
            "duration_seconds": 0.5,
        }
        client = _FakeAsyncClient(post_payload=payload)
        with _patch_httpx(client):
            await rebuild(output="json", api_url=_API_URL)

        out = capsys.readouterr().out
        assert orjson.loads(out) == payload

    @pytest.mark.asyncio
    async def test_rebuild_uses_300s_timeout(self) -> None:
        """Rebuild can be slow - the helper sets timeout=300.0 explicitly."""
        from aiperf.cli_commands.kube.index import rebuild

        client = _FakeAsyncClient(
            post_payload={
                "runs_indexed": 0,
                "sweep_variations_indexed": 0,
                "duration_seconds": 0.0,
            }
        )
        with _patch_httpx(client):
            await rebuild(api_url=_API_URL)

        assert client.timeout == 300.0

    @pytest.mark.asyncio
    async def test_http_error_propagates(self) -> None:
        from aiperf.cli_commands.kube.index import rebuild

        client = _FakeAsyncClient(post_exc=httpx.ConnectError("refused"))
        with _patch_httpx(client), pytest.raises(httpx.ConnectError):
            await rebuild(api_url=_API_URL)

    @pytest.mark.asyncio
    async def test_rebuild_403_has_actionable_token_error(self) -> None:
        from aiperf.cli_commands.kube.index import rebuild

        client = _FakeAsyncClient(post_status_code=403)
        with _patch_httpx(client), pytest.raises(RuntimeError) as exc_info:
            await rebuild(api_url=_API_URL)

        message = str(exc_info.value)
        assert "mutating-route auth" in message
        assert "AIPERF_OPERATOR_MUTATING_ROUTES_TOKEN" in message
        assert client.post_headers == [{}]

    @pytest.mark.asyncio
    async def test_rebuild_401_does_not_leak_wrong_token(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from aiperf.cli_commands.kube.index import rebuild

        monkeypatch.setenv("AIPERF_OPERATOR_MUTATING_ROUTES_TOKEN", "wrong-secret")
        client = _FakeAsyncClient(post_status_code=401)
        with _patch_httpx(client), pytest.raises(RuntimeError) as exc_info:
            await rebuild(api_url=_API_URL)

        assert client.post_headers == [{"Authorization": "Bearer wrong-secret"}]
        message = str(exc_info.value)
        assert "AIPERF_OPERATOR_MUTATING_ROUTES_TOKEN" in message
        assert "wrong-secret" not in message

    @pytest.mark.asyncio
    async def test_json_mode_restores_logger_level_on_success(self) -> None:
        """Logger toggled to WARNING then restored after completion."""
        from aiperf.cli_commands.kube.index import rebuild

        kube_logger = logging.getLogger("aiperf.kube")
        kube_logger.setLevel(logging.DEBUG)

        client = _FakeAsyncClient(
            post_payload={
                "runs_indexed": 1,
                "sweep_variations_indexed": 0,
                "duration_seconds": 0.0,
            }
        )

        # Capture the logger level WHILE the request is in flight by hooking
        # the post call. After the call returns, finally must restore INFO.
        captured_levels: list[int] = []
        original_post = client.post

        async def _spy_post(
            url: str, *, headers: Mapping[str, str] | None = None
        ) -> Any:
            captured_levels.append(kube_logger.level)
            return await original_post(url, headers=headers)

        client.post = _spy_post  # type: ignore[method-assign]

        with _patch_httpx(client):
            await rebuild(output="json", api_url=_API_URL)

        # During the request the logger was suppressed
        assert captured_levels == [logging.WARNING]
        # After the request finishes the finally block restores the original level.
        assert kube_logger.level == logging.DEBUG
