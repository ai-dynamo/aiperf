# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Video download redirect checks against real HTTP origins."""

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from aiohttp import BasicAuth, web
from pytest import param

from aiperf.common.models import ErrorDetails, TextResponse
from aiperf.plugin.enums import EndpointType
from aiperf.transports.aiohttp_client import AioHttpClient
from aiperf.transports.aiohttp_transport import AioHttpTransport
from aiperf.transports.http_defaults import AioHttpDefaults
from tests.unit.transports.conftest import create_model_endpoint_info
from tests.unit.transports.test_aiohttp_transport import create_request_info

pytestmark = [pytest.mark.component_integration, pytest.mark.asyncio]


@asynccontextmanager
async def serve(
    handler: Callable[[web.Request], Awaitable[web.StreamResponse]],
) -> AsyncIterator[str]:
    """Serve a handler on an ephemeral local port with bounded cleanup."""
    app = web.Application()
    app.router.add_route("*", "/{path:.*}", handler)
    runner = web.AppRunner(app, shutdown_timeout=0.01)
    await runner.setup()
    try:
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = runner.addresses[0][1]
        yield f"http://127.0.0.1:{port}"
    finally:
        await runner.cleanup()


async def test_cross_origin_chain_preserves_signed_url_bytes_and_credentials(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Foreign hops get no inherited/netrc auth, cookie replay or environment proxy."""
    seen: list[tuple[str, str, dict[str, str]]] = []
    encoded_path = "/file%2Fname?key=a%2Fb&key=c%2Bd&sig=%2b%2F"

    async def origin_handler(request: web.Request) -> web.StreamResponse:
        seen.append(("origin", request.raw_path, dict(request.headers)))
        if request.path == "/start":
            response = web.Response(
                status=302,
                headers={"Location": foreign + encoded_path},
                body=b"\xff\xfe",
            )
            response.set_cookie("redirect_cookie", "secret")
            return response
        return web.Response(body=b"final-video", content_type="video/mp4")

    async def foreign_handler(request: web.Request) -> web.StreamResponse:
        seen.append(("foreign", request.raw_path, dict(request.headers)))
        destination = "/bounce" if request.path != "/bounce" else origin + "/final"
        response = web.Response(status=307, headers={"Location": destination})
        response.set_cookie("foreign_cookie", "secret")
        return response

    netrc = tmp_path / "netrc"
    netrc.write_text("machine localhost login leaked-user password leaked-password\n")
    netrc.chmod(0o600)
    monkeypatch.setenv("NETRC", str(netrc))
    for key in ("http_proxy", "https_proxy", "all_proxy", "no_proxy", "ALL_PROXY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setattr(AioHttpDefaults, "TRUST_ENV", True)
    async with serve(origin_handler) as origin, serve(foreign_handler) as foreign:
        foreign = foreign.replace("127.0.0.1", "localhost")
        client = AioHttpClient(timeout=1)
        transport = AioHttpTransport(
            model_endpoint=create_model_endpoint_info(base_url=origin)
        )
        transport.aiohttp_client = client
        try:
            result = await transport._download_video_content(
                "job",
                origin + "/start",
                {
                    "Authorization": "Bearer endpoint-secret",
                    "X-Private": "custom-secret",
                    "User-Agent": "aiperf-test",
                },
                signing_origin_url=origin + "/job",
            )
        finally:
            await client.close()
    assert result == b"final-video"
    assert [entry[0] for entry in seen] == ["origin", "foreign", "foreign", "origin"]
    assert seen[1][1] == encoded_path
    assert seen[1][2].get("User-Agent") == "aiperf-test"
    for entry in seen[1:3]:
        for name in ("Authorization", "X-Private", "Cookie"):
            assert name not in entry[2]
    assert seen[3][2]["Authorization"] == "Bearer endpoint-secret"
    assert seen[3][2]["X-Private"] == "custom-secret"
    assert "Cookie" not in seen[3][2]


@pytest.mark.parametrize("start", [param("direct"), param("origin"), param("foreign")])  # fmt: skip
async def test_configured_basic_auth_is_restored_only_at_origin(
    monkeypatch: pytest.MonkeyPatch, start: str
) -> None:
    """URL auth is decoded once and never follows a hop to another origin."""
    monkeypatch.setattr(AioHttpDefaults, "TRUST_ENV", False)
    seen: list[tuple[str, str, dict[str, str]]] = []
    target = "/file%2Fname?sig=%2b%2F"

    async def origin_handler(request: web.Request) -> web.StreamResponse:
        seen.append(("origin", request.raw_path, dict(request.headers)))
        if request.path == "/start":
            return web.Response(status=302, headers={"Location": "/same"})
        if request.path == "/same":
            return web.Response(status=307, headers={"Location": foreign + target})
        return web.Response(body=b"video", content_type="video/mp4")

    async def foreign_handler(request: web.Request) -> web.StreamResponse:
        seen.append(("foreign", request.raw_path, dict(request.headers)))
        return web.Response(status=302, headers={"Location": origin + "/final"})

    async with serve(origin_handler) as origin, serve(foreign_handler) as foreign:
        configured = origin.replace("://", "://us%65r:p%40ss%3Aword@")
        client = AioHttpClient(timeout=1)
        transport = AioHttpTransport(
            model_endpoint=create_model_endpoint_info(base_url=configured)
        )
        transport.aiohttp_client = client
        content_urls = {
            "direct": origin + "/final",
            "origin": origin + "/start",
            "foreign": foreign + target,
        }
        try:
            result = await transport._download_video_content(
                "job",
                content_urls[start],
                {"X-Private": "endpoint-secret", "User-Agent": "aiperf-test"},
                signing_origin_url=configured + "/job",
            )
        finally:
            await client.close()
    assert result == b"video"
    expected_origins = {
        "direct": ["origin"],
        "origin": ["origin", "origin", "foreign", "origin"],
        "foreign": ["foreign", "origin"],
    }
    assert [entry[0] for entry in seen] == expected_origins[start]
    for kind, path, headers in seen:
        if kind == "origin":
            assert BasicAuth.decode(headers["Authorization"]) == BasicAuth(
                "user", "p@ss:word"
            )
            assert headers["X-Private"] == "endpoint-secret"
        else:
            assert path == target
            assert "Authorization" not in headers
            assert "X-Private" not in headers
            assert headers["User-Agent"] == "aiperf-test"


@pytest.mark.parametrize("source", [param("content"), param("location")])  # fmt: skip
async def test_server_injected_url_credentials_are_rejected(
    monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    """Even matching configured credentials are untrusted in server-selected URLs."""
    monkeypatch.setattr(AioHttpDefaults, "TRUST_ENV", False)
    seen: list[str] = []

    async def handler(request: web.Request) -> web.StreamResponse:
        seen.append(request.path)
        return web.Response(status=302, headers={"Location": injected})

    async with serve(handler) as origin:
        configured = origin.replace("://", "://user:private-password@")
        injected = configured + "/injected?private-query=value"
        client = AioHttpClient(timeout=1)
        transport = AioHttpTransport(
            model_endpoint=create_model_endpoint_info(base_url=configured)
        )
        transport.aiohttp_client = client
        try:
            result = await transport._download_video_content(
                "job",
                injected if source == "content" else origin + "/start",
                {},
                signing_origin_url=configured + "/job",
            )
        finally:
            await client.close()
    assert isinstance(result, ErrorDetails)
    assert result.code == (500 if source == "content" else 302)
    assert seen == ([] if source == "content" else ["/start"])
    assert "Invalid download URL" in result.message
    assert "private-password" not in result.message
    assert "private-query" not in result.message


async def test_url_auth_does_not_override_explicit_authorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Competing authentication sources retain aiohttp's existing failure behavior."""
    monkeypatch.setattr(AioHttpDefaults, "TRUST_ENV", False)
    seen: list[str] = []

    async def handler(request: web.Request) -> web.StreamResponse:
        seen.append(request.path)
        return web.Response(body=b"video", content_type="video/mp4")

    async with serve(handler) as origin:
        configured = origin.replace("://", "://user:private-password@")
        client = AioHttpClient(timeout=1)
        transport = AioHttpTransport(
            model_endpoint=create_model_endpoint_info(base_url=configured)
        )
        transport.aiohttp_client = client
        try:
            result = await transport._download_video_content(
                "job",
                origin + "/content",
                {"Authorization": "Bearer private-token"},
                signing_origin_url=configured + "/job",
            )
        finally:
            await client.close()
    assert isinstance(result, ErrorDetails)
    assert not seen
    assert "private-password" not in result.message
    assert "private-token" not in result.message


async def test_unfinished_redirect_body_does_not_block_final_download() -> None:
    """Headers suffice even if the redirect response never finishes its body."""
    handler_done = asyncio.Event()

    async def handler(request: web.Request) -> web.StreamResponse:
        if request.path == "/start":
            response = web.StreamResponse(status=302, headers={"Location": "/final"})
            await response.prepare(request)
            await response.write(b"\xff" * 65536)
            await handler_done.wait()
            return response
        return web.Response(body=b"video", content_type="video/mp4")

    async with serve(handler) as origin:
        client = AioHttpClient(timeout=0.5)
        transport = AioHttpTransport(
            model_endpoint=create_model_endpoint_info(base_url=origin)
        )
        transport.aiohttp_client = client
        try:
            assert (
                await transport._download_video_content(
                    "job", origin + "/start", {}, signing_origin_url=origin
                )
                == b"video"
            )
        finally:
            handler_done.set()
            await client.close()


async def test_timeout_is_shared_across_redirect_requests() -> None:
    calls: list[str] = []

    async def handler(request: web.Request) -> web.StreamResponse:
        calls.append(request.path)
        await asyncio.sleep(0.07)
        if request.path == "/start":
            return web.Response(status=302, headers={"Location": "/final"})
        return web.Response(body=b"video", content_type="video/mp4")

    async with serve(handler) as origin:
        client = AioHttpClient(timeout=0.11)
        transport = AioHttpTransport(
            model_endpoint=create_model_endpoint_info(base_url=origin)
        )
        transport.aiohttp_client = client
        try:
            result = await transport._download_video_content(
                "job", origin + "/start", {}, signing_origin_url=origin
            )
            assert isinstance(result, ErrorDetails) and "timeout" in result.message
            assert calls == ["/start", "/final"]
        finally:
            await client.close()


@pytest.mark.parametrize("timeout", [param(None), param(0)])  # fmt: skip
async def test_unset_timeout_allows_download(timeout: float | None) -> None:
    async def handler(request: web.Request) -> web.StreamResponse:
        return web.Response(body=b"video", content_type="video/mp4")

    async with serve(handler) as origin:
        client = AioHttpClient(timeout=timeout)
        transport = AioHttpTransport(
            model_endpoint=create_model_endpoint_info(base_url=origin)
        )
        transport.aiohttp_client = client
        try:
            assert (
                await transport._download_video_content(
                    "job", origin, {}, signing_origin_url=origin
                )
                == b"video"
            )
        finally:
            await client.close()


async def test_cancel_closes_download_session() -> None:
    started = asyncio.Event()

    async def handler(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(headers={"Content-Type": "video/mp4"})
        await response.prepare(request)
        started.set()
        await asyncio.Event().wait()
        return response

    async with serve(handler) as origin:
        client = AioHttpClient(timeout=1)
        transport = AioHttpTransport(
            model_endpoint=create_model_endpoint_info(base_url=origin)
        )
        transport.aiohttp_client = client
        try:
            task = asyncio.create_task(
                transport._download_video_content(
                    "job", origin, {}, signing_origin_url=origin
                )
            )
            await asyncio.wait_for(started.wait(), timeout=1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not client.tcp_connector._acquired
        finally:
            await client.close()


@pytest.mark.parametrize("download,url_auth", [param(True, False), param(False, False), param(True, True)])  # fmt: skip
async def test_video_workflow_timing_and_published_responses(
    download: bool, url_auth: bool
) -> None:
    """The full generation workflow measures downloads but publishes only job JSON."""
    seen: list[str] = []
    auth_headers: list[str | None] = []
    submit_ns = 0
    download_end_ns = 0

    async def handler(request: web.Request) -> web.StreamResponse:
        nonlocal submit_ns, download_end_ns
        seen.append(request.path)
        auth_headers.append(request.headers.get("Authorization"))
        if request.method == "POST":
            submit_ns = time.perf_counter_ns()
            return web.json_response({"id": "job", "status": "queued"}, status=202)
        if request.path == "/v1/videos/job":
            return web.json_response({"id": "job", "status": "completed"})
        if request.path.endswith("/content"):
            return web.Response(status=302, headers={"Location": "/final"})
        await asyncio.sleep(0.005)
        download_end_ns = time.perf_counter_ns()
        return web.Response(body=b"binary-video-secret", content_type="video/mp4")

    async with serve(handler) as origin:
        endpoint = create_model_endpoint_info(
            base_url=origin.replace("://", "://user:password@") if url_auth else origin,
            custom_endpoint="/v1/videos",
        )
        endpoint.endpoint.type = EndpointType.VIDEO_GENERATION
        endpoint.endpoint.download_video_content = download
        transport = AioHttpTransport(model_endpoint=endpoint)
        client = AioHttpClient(timeout=1)
        transport.aiohttp_client = client
        try:
            record = await transport.send_request(
                create_request_info(endpoint), {"prompt": "video"}
            )
        finally:
            await client.close()
    assert record.error is None and record.status == 200
    assert auth_headers == [
        BasicAuth("user", "password").encode() if url_auth else None
    ] * len(seen)
    assert len(record.responses) == 2
    assert all(isinstance(response, TextResponse) for response in record.responses)
    assert "binary-video-secret" not in record.model_dump_json()
    assert record.start_perf_ns <= submit_ns <= record.end_perf_ns
    if download:
        assert seen == [
            "/v1/videos",
            "/v1/videos/job",
            "/v1/videos/job/content",
            "/final",
        ]
        assert record.end_perf_ns >= download_end_ns
    else:
        assert seen == ["/v1/videos", "/v1/videos/job"]
