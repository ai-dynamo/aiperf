# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
from aiohttp import web

from aiperf.auth.base_signer import SignedRequest
from aiperf.common.control_plane_http import ControlPlaneHttpError, control_plane_post


async def _run_server(handler: object) -> tuple[web.AppRunner, str]:
    app = web.Application()
    app.router.add_post("/ctrl", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    assert site._server is not None
    sockets = site._server.sockets
    assert sockets is not None and sockets
    port = sockets[0].getsockname()[1]
    return runner, f"http://127.0.0.1:{port}/ctrl"


@pytest.mark.asyncio
async def test_control_plane_post_succeeds_on_2xx() -> None:
    async def ok(request: web.Request) -> web.Response:
        body = await request.read()
        assert body == b""
        return web.Response(status=204)

    runner, url = await _run_server(ok)
    try:
        await control_plane_post(url=url, headers={}, timeout_s=2.0)
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_control_plane_post_raises_on_non_2xx() -> None:
    async def fail(_request: web.Request) -> web.Response:
        return web.Response(status=500, text="secret-body-should-not-leak")

    runner, url = await _run_server(fail)
    try:
        with pytest.raises(ControlPlaneHttpError, match="status 500") as exc_info:
            await control_plane_post(url=url, headers={}, timeout_s=2.0)
        assert "secret-body" not in str(exc_info.value)
        assert exc_info.value.retryable is False
        assert exc_info.value.status_code == 500
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_control_plane_post_wraps_client_error() -> None:
    with patch("aiperf.common.control_plane_http.aiohttp.ClientSession") as session_cls:
        session_cls.return_value.__aenter__ = AsyncMock(
            side_effect=aiohttp.ClientConnectorError(
                MagicMock(), OSError("connection refused")
            )
        )
        session_cls.return_value.__aexit__ = AsyncMock(return_value=None)
        with pytest.raises(
            ControlPlaneHttpError, match="ClientConnectorError"
        ) as exc_info:
            await control_plane_post(
                url="http://user:secret@127.0.0.1:9/x",
                headers={},
                timeout_s=0.1,
            )
        assert exc_info.value.retryable is True
        assert exc_info.value.status_code is None


@pytest.mark.asyncio
async def test_control_plane_post_wraps_timeout() -> None:
    with patch("aiperf.common.control_plane_http.aiohttp.ClientSession") as session_cls:
        session_cls.return_value.__aenter__ = AsyncMock(side_effect=TimeoutError())
        session_cls.return_value.__aexit__ = AsyncMock(return_value=None)
        with pytest.raises(ControlPlaneHttpError, match="TimeoutError") as exc_info:
            await control_plane_post(
                url="http://127.0.0.1:9/x",
                headers={},
                timeout_s=0.1,
            )
        assert exc_info.value.retryable is True
        assert exc_info.value.status_code is None


@pytest.mark.asyncio
async def test_control_plane_post_error_redacts_url_credentials() -> None:
    async def fail(_request: web.Request) -> web.Response:
        return web.Response(status=503)

    runner, url = await _run_server(fail)
    # Inject userinfo into the URL while still hitting the local server.
    from urllib.parse import urlsplit, urlunsplit

    parts = urlsplit(url)
    cred_url = urlunsplit(
        (parts.scheme, f"user:sekret@{parts.hostname}:{parts.port}", parts.path, "", "")
    )
    try:
        with pytest.raises(ControlPlaneHttpError) as exc_info:
            await control_plane_post(url=cred_url, headers={}, timeout_s=2.0)
        msg = str(exc_info.value)
        assert "sekret" not in msg
        assert "status 503" in msg
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_control_plane_post_uses_trust_env_false() -> None:
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.post.return_value = mock_resp
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    with patch("aiperf.common.control_plane_http.aiohttp.ClientSession") as session_cls:
        session_cls.return_value = mock_session
        await control_plane_post(
            url="http://127.0.0.1:9/x",
            headers={"Authorization": "Bearer t"},
            timeout_s=1.0,
        )
        kwargs = session_cls.call_args.kwargs
        assert kwargs.get("trust_env") is False
        assert isinstance(kwargs.get("timeout"), aiohttp.ClientTimeout)
        mock_session.post.assert_called_once()
        call_kwargs = mock_session.post.call_args
        assert call_kwargs.kwargs.get("data") == b""
        assert call_kwargs.kwargs.get("headers") == {"Authorization": "Bearer t"}


class _TokenSigner:
    """Signs by attaching a session token, like SigV4 with temporary creds."""

    async def sign(
        self, method: str, url: str, headers: dict[str, str], body: bytes | None
    ) -> SignedRequest:
        del method, url, body
        signed = dict(headers)
        signed["Authorization"] = "AWS4-HMAC-SHA256 ..."
        signed["X-Amz-Security-Token"] = "SESSION-SECRET"
        return SignedRequest(headers=signed)


@pytest.mark.asyncio
async def test_signed_control_post_does_not_replay_token_at_a_redirect_target() -> None:
    """aiohttp drops ``Authorization`` across origins but not custom headers,
    so ``X-Amz-Security-Token`` -- a bearer credential -- would be delivered to
    whatever host the control endpoint redirects to."""
    seen: dict[str, str | None] = {}

    async def collect(request: web.Request) -> web.Response:
        seen["authorization"] = request.headers.get("Authorization")
        seen["token"] = request.headers.get("X-Amz-Security-Token")
        return web.Response(status=204)

    target_app = web.Application()
    # A 302 turns the POST into a GET, so accept both or the target answers
    # 405 and the test passes without ever proving anything.
    target_app.router.add_post("/collect", collect)
    target_app.router.add_get("/collect", collect)
    target_runner = web.AppRunner(target_app)
    await target_runner.setup()
    target_site = web.TCPSite(target_runner, "127.0.0.1", 0)
    await target_site.start()
    assert target_site._server is not None and target_site._server.sockets
    target_port = target_site._server.sockets[0].getsockname()[1]
    target_url = f"http://127.0.0.1:{target_port}/collect"

    async def redirect(_request: web.Request) -> web.Response:
        raise web.HTTPFound(location=target_url)

    runner, url = await _run_server(redirect)
    try:
        with pytest.raises(ControlPlaneHttpError):
            await control_plane_post(
                url=url,
                headers={},
                timeout_s=2.0,
                signer=cast(Any, _TokenSigner()),
            )
        assert seen.get("token") is None, (
            "the session token reached the redirect target"
        )
    finally:
        await runner.cleanup()
        await target_runner.cleanup()


@pytest.mark.asyncio
async def test_a_request_phase_failure_is_not_labelled_a_signing_failure() -> None:
    """The catch-all spanned the whole request, so any unexpected error was
    reported as ``signing failed`` and forced retryable -- which also made
    ``_post_with_retry`` burn its full budget on a fatal error. With no signer
    configured, ``sign_request`` returns immediately and the label is simply
    untrue."""
    with patch("aiperf.common.control_plane_http.aiohttp.ClientSession") as session_cls:
        session_cls.return_value.__aenter__ = AsyncMock(
            side_effect=RuntimeError("event loop is closed")
        )
        session_cls.return_value.__aexit__ = AsyncMock(return_value=None)

        with pytest.raises(ControlPlaneHttpError) as exc_info:
            await control_plane_post(
                url="http://example.invalid/ctrl", headers={}, timeout_s=1.0
            )

    assert "signing failed" not in str(exc_info.value)
    assert exc_info.value.retryable is False


@pytest.mark.asyncio
async def test_a_signing_failure_is_still_labelled_and_retryable() -> None:
    """Credential refresh is itself a network operation, so this one stays
    retryable on purpose."""

    class _FailingSigner:
        async def sign(
            self, method: str, url: str, headers: dict[str, str], body: bytes | None
        ) -> SignedRequest:
            raise RuntimeError("credential refresh timed out")

    with pytest.raises(ControlPlaneHttpError) as exc_info:
        await control_plane_post(
            url="http://example.invalid/ctrl",
            headers={},
            timeout_s=1.0,
            signer=cast(Any, _FailingSigner()),
        )

    assert "signing failed" in str(exc_info.value)
    assert exc_info.value.retryable is True
