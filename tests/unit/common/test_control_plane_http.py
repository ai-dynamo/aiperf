# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
from aiohttp import web

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
        with pytest.raises(ControlPlaneHttpError, match="ClientConnectorError"):
            await control_plane_post(
                url="http://user:secret@127.0.0.1:9/x",
                headers={},
                timeout_s=0.1,
            )


@pytest.mark.asyncio
async def test_control_plane_post_wraps_timeout() -> None:
    with patch("aiperf.common.control_plane_http.aiohttp.ClientSession") as session_cls:
        session_cls.return_value.__aenter__ = AsyncMock(side_effect=TimeoutError())
        session_cls.return_value.__aexit__ = AsyncMock(return_value=None)
        with pytest.raises(ControlPlaneHttpError, match="TimeoutError"):
            await control_plane_post(
                url="http://127.0.0.1:9/x",
                headers={},
                timeout_s=0.1,
            )


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
