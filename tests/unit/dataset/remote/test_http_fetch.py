# tests/unit/dataset/remote/test_http_fetch.py
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib
from pathlib import Path

import pytest
from aiohttp import web

from aiperf.dataset.remote import http_fetch
from aiperf.dataset.remote.errors import RemoteErrorKind, RemoteSourceError

BODY = b'{"timestamp": 1, "input_length": 4}\n' * 100


def _static_app(tmp_path: Path) -> web.Application:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "t.jsonl").write_bytes(BODY)
    app = web.Application()
    app.router.add_static("/", tmp_path)
    return app


@pytest.mark.asyncio
async def test_metadata_prefix_and_download(http_server, tmp_path: Path) -> None:
    base = await http_server(_static_app(tmp_path / "srv"))
    url = f"{base}/t.jsonl"
    async with http_fetch.open_session() as s:
        meta = await http_fetch.fetch_metadata(s, url, headers={}, ref_display=url)
        assert meta.content_length == len(BODY) and meta.etag
        prefix, complete = await http_fetch.read_prefix(
            s, url, 64, headers={}, ref_display=url
        )
        assert prefix == BODY[:64] and complete is False
        whole, complete = await http_fetch.read_prefix(
            s, url, 10 * len(BODY), headers={}, ref_display=url
        )
        assert whole == BODY and complete is True
        sha, size, _ = await http_fetch.download_to(
            s, url, tmp_path / "out" / "t.jsonl", headers={}, ref_display=url
        )
    assert (sha, size) == (hashlib.sha256(BODY).hexdigest(), len(BODY))


@pytest.mark.asyncio
async def test_prefix_when_server_ignores_range(http_server) -> None:
    async def full(_: web.Request) -> web.Response:
        return web.Response(body=BODY)

    app = web.Application()
    app.router.add_get("/t", full)
    url = f"{await http_server(app)}/t"
    async with http_fetch.open_session() as s:
        prefix, complete = await http_fetch.read_prefix(
            s, url, 64, headers={}, ref_display=url
        )
    assert prefix == BODY[:64] and complete is False


@pytest.mark.asyncio
async def test_cross_origin_redirect_drops_credentials_keeps_range(http_server) -> None:
    seen: dict[str, str | None] = {}

    async def target(request: web.Request) -> web.Response:
        seen["authorization"] = request.headers.get("Authorization")
        seen["range"] = request.headers.get("Range")
        return web.Response(
            body=b"abc", status=206, headers={"Content-Range": "bytes 0-2/3"}
        )

    target_app = web.Application()
    target_app.router.add_get("/t", target)
    target_url = f"{await http_server(target_app)}/t"

    async def redirect(_: web.Request) -> web.Response:
        raise web.HTTPFound(location=target_url)

    source_app = web.Application()
    source_app.router.add_get("/t", redirect)
    source_url = f"{await http_server(source_app)}/t"

    async with http_fetch.open_session() as s:
        data, complete = await http_fetch.read_prefix(
            s,
            source_url,
            8,
            headers={"Authorization": "Bearer hf_secret"},
            ref_display=source_url,
        )
    assert (data, complete) == (b"abc", True)
    assert seen == {"authorization": None, "range": "bytes=0-7"}


@pytest.mark.asyncio
async def test_redirect_loop_is_an_error(http_server) -> None:
    app = web.Application()

    async def loop(_: web.Request) -> web.Response:
        raise web.HTTPFound(location="/t")

    app.router.add_get("/t", loop)
    url = f"{await http_server(app)}/t"
    async with http_fetch.open_session() as s:
        with pytest.raises(RemoteSourceError, match="redirected more than"):
            await http_fetch.fetch_metadata(s, url, headers={}, ref_display=url)


@pytest.mark.parametrize(
    "status,kind",
    [(401, RemoteErrorKind.AUTH_DENIED), (403, RemoteErrorKind.AUTH_DENIED), (404, RemoteErrorKind.NOT_FOUND),
     (429, RemoteErrorKind.TRANSIENT), (503, RemoteErrorKind.TRANSIENT)],
)  # fmt: skip
@pytest.mark.asyncio
async def test_status_mapping(http_server, status: int, kind: RemoteErrorKind) -> None:
    app = web.Application()

    async def fail(_: web.Request) -> web.Response:
        return web.Response(status=status)

    app.router.add_route("*", "/t", fail)
    url = f"{await http_server(app)}/t"
    async with http_fetch.open_session() as s:
        with pytest.raises(RemoteSourceError) as exc:
            await http_fetch.fetch_metadata(s, url, headers={}, ref_display=url)
    assert exc.value.kind == kind


@pytest.mark.asyncio
async def test_head_not_allowed_falls_back_to_get(http_server) -> None:
    app = web.Application()

    async def get_only(_: web.Request) -> web.Response:
        return web.Response(body=b"x", headers={"ETag": '"g"'})

    app.router.add_get("/t", get_only, allow_head=False)
    url = f"{await http_server(app)}/t"
    async with http_fetch.open_session() as s:
        meta = await http_fetch.fetch_metadata(s, url, headers={}, ref_display=url)
    assert meta.etag == '"g"'


@pytest.mark.asyncio
async def test_connection_refused_is_transient() -> None:
    url = "http://127.0.0.1:9/t"
    async with http_fetch.open_session() as s:
        with pytest.raises(RemoteSourceError) as exc:
            await http_fetch.fetch_metadata(s, url, headers={}, ref_display=url)
    assert exc.value.kind == RemoteErrorKind.TRANSIENT


@pytest.mark.asyncio
async def test_redirect_to_presigned_target_never_leaks_signature_in_errors(
    http_server,
) -> None:
    target = (
        "http://127.0.0.1:9/t?X-Amz-Signature=SECRETSIG&X-Amz-Credential=AKIASECRET"
    )

    async def redirect(_: web.Request) -> web.Response:
        raise web.HTTPFound(location=target)

    app = web.Application()
    app.router.add_get("/t", redirect)
    url = f"{await http_server(app)}/t"
    async with http_fetch.open_session() as s:
        with pytest.raises(RemoteSourceError) as exc:
            await http_fetch.fetch_metadata(s, url, headers={}, ref_display=url)
    assert "SECRETSIG" not in str(exc.value) and "AKIASECRET" not in str(exc.value)
    assert exc.value.kind == RemoteErrorKind.TRANSIENT


@pytest.mark.asyncio
async def test_same_origin_redirect_keeps_credentials(http_server) -> None:
    seen: dict[str, str | None] = {}

    async def final(request: web.Request) -> web.Response:
        seen["authorization"] = request.headers.get("Authorization")
        return web.Response(body=b"x")

    async def redirect(_: web.Request) -> web.Response:
        raise web.HTTPFound(location="/final")

    app = web.Application()
    app.router.add_get("/t", redirect)
    app.router.add_get("/final", final)
    url = f"{await http_server(app)}/t"
    async with http_fetch.open_session() as s:
        await http_fetch.fetch_metadata(
            s, url, headers={"Authorization": "Bearer ok"}, ref_display=url
        )
    assert seen["authorization"] == "Bearer ok"


@pytest.mark.parametrize(
    "value,total",
    [("bytes 0-9/100", 100), ("bytes 0-9/*", None), (None, None), ("garbage", None)],
)
def test_content_range_total(value: str | None, total: int | None) -> None:
    assert http_fetch.content_range_total(value) == total
