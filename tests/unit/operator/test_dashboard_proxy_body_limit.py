# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Body bounding and encoding relay for the dashboard reverse proxy.

Request side: the proxy buffers the body to re-frame it for the sidecar, so an
unbounded POST would be resident in the results-server sidecar's memory. Covered
here are the declared-length reject, the chunked (no Content-Length) reject, and
the under-limit pass-through.

Response side: the proxy runs ``auto_decompress=False``, so a compressed
upstream body is relayed still encoded and must keep its ``content-encoding``.
Those tests assert by actually decompressing what the client receives.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterator

import aiohttp
import orjson
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _make_app(monkeypatch: pytest.MonkeyPatch, *, max_body_bytes: int) -> FastAPI:
    monkeypatch.setenv("AIPERF_DASHBOARD_PROXY_ENABLED", "1")
    monkeypatch.setenv("AIPERF_DASHBOARD_PORT", "8082")
    monkeypatch.setenv("AIPERF_DASHBOARD_PROXY_MAX_BODY_BYTES", str(max_body_bytes))

    from aiperf.operator import environment as env_mod

    importlib.reload(env_mod)
    from aiperf.operator.routers import dashboard_proxy

    importlib.reload(dashboard_proxy)

    app = FastAPI()
    app.include_router(dashboard_proxy.create_dashboard_proxy_router())
    return app


def _patch_upstream(
    monkeypatch: pytest.MonkeyPatch, captured: dict[str, object]
) -> None:
    class _FakeContent:
        async def iter_any(self):
            yield b"ok"

    class _FakeResp:
        status = 200
        headers = {"content-type": "text/plain"}
        content = _FakeContent()

    class _FakeStream:
        async def __aenter__(self):
            return _FakeResp()

        async def __aexit__(self, *_a):
            return None

    def _fake_request(self, method, url, **kwargs):
        captured["called"] = True
        captured["data"] = kwargs.get("data")
        return _FakeStream()

    monkeypatch.setattr(aiohttp.ClientSession, "request", _fake_request)


def _chunks(payload: bytes, size: int) -> Iterator[bytes]:
    for offset in range(0, len(payload), size):
        yield payload[offset : offset + size]


def test_read_bounded_body_under_limit_forwards_upstream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _make_app(monkeypatch, max_body_bytes=4096)
    captured: dict[str, object] = {}
    _patch_upstream(monkeypatch, captured)

    payload = b"a" * 1024
    with TestClient(app) as client:
        resp = client.post("/dashboard/refresh", content=payload)

    assert resp.status_code == 200
    assert captured["called"] is True
    assert captured["data"] == payload


def test_read_bounded_body_over_declared_length_returns_413(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _make_app(monkeypatch, max_body_bytes=1024)
    captured: dict[str, object] = {}
    _patch_upstream(monkeypatch, captured)

    with TestClient(app) as client:
        resp = client.post("/dashboard/refresh", content=b"a" * 4096)

    assert resp.status_code == 413
    assert b"exceeds" in resp.content
    assert "called" not in captured


def test_read_bounded_body_chunked_without_content_length_returns_413(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A chunked request declares no length, so the bytes read must be counted."""
    app = _make_app(monkeypatch, max_body_bytes=1024)
    captured: dict[str, object] = {}
    _patch_upstream(monkeypatch, captured)

    with TestClient(app) as client:
        resp = client.post(
            "/dashboard/refresh",
            content=_chunks(b"a" * 8192, 512),
            headers={"transfer-encoding": "chunked"},
        )

    assert resp.status_code == 413
    assert "called" not in captured


def test_read_bounded_body_chunked_under_limit_forwards_upstream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _make_app(monkeypatch, max_body_bytes=4096)
    captured: dict[str, object] = {}
    _patch_upstream(monkeypatch, captured)

    payload = b"b" * 1536
    with TestClient(app) as client:
        resp = client.post(
            "/dashboard/refresh",
            content=_chunks(payload, 512),
            headers={"transfer-encoding": "chunked"},
        )

    assert resp.status_code == 200
    assert captured["data"] == payload


@pytest.mark.asyncio
async def test_read_bounded_body_no_content_length_header_still_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without a declared length only the byte counter can stop the read."""
    _make_app(monkeypatch, max_body_bytes=1024)

    from aiperf.operator.routers import dashboard_proxy

    consumed = 0

    class _Req:
        headers: dict[str, str] = {}

        async def stream(self):
            nonlocal consumed
            for _ in range(100):
                consumed += 512
                yield b"a" * 512

    assert await dashboard_proxy._read_bounded_body(_Req(), 1024) is None  # type: ignore[arg-type]
    # The generator is abandoned as soon as the limit is passed, so only a
    # bounded amount is ever resident.
    assert consumed <= 1536


@pytest.mark.asyncio
async def test_read_bounded_body_malformed_content_length_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _make_app(monkeypatch, max_body_bytes=1024)

    from aiperf.operator.routers import dashboard_proxy

    class _Req:
        headers = {"content-length": "not-a-number"}

        async def stream(self):
            yield b""

    assert await dashboard_proxy._read_bounded_body(_Req(), 1024) is None  # type: ignore[arg-type]


def _patch_compressed_upstream(
    monkeypatch: pytest.MonkeyPatch,
    *,
    body: bytes,
    content_encoding: str,
    captured: dict[str, object],
) -> None:
    class _FakeContent:
        async def iter_any(self):
            # Split so the response is genuinely streamed, as a real body is.
            yield body[: len(body) // 2]
            yield body[len(body) // 2 :]

    class _FakeResp:
        status = 200
        headers = {
            "content-type": "application/json",
            "content-encoding": content_encoding,
            "content-length": str(len(body)),
        }
        content = _FakeContent()

    class _FakeStream:
        async def __aenter__(self):
            return _FakeResp()

        async def __aexit__(self, *_a):
            return None

    def _fake_request(self, method, url, **kwargs):
        captured["headers"] = dict(kwargs.get("headers") or {})
        return _FakeStream()

    monkeypatch.setattr(aiohttp.ClientSession, "request", _fake_request)


def test_proxy_gzip_upstream_body_reaches_client_decodable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A compressed upstream body must keep the header that describes it.

    ``auto_decompress=False`` means the proxy relays still-compressed bytes, so
    dropping ``content-encoding`` made the client decode gzip as identity.
    """
    import zlib

    payload = orjson.dumps({"plots": ["a"] * 200})
    compressed = zlib.compress(payload, wbits=31)
    app = _make_app(monkeypatch, max_body_bytes=4096)
    captured: dict[str, object] = {}
    _patch_compressed_upstream(
        monkeypatch, body=compressed, content_encoding="gzip", captured=captured
    )

    with TestClient(app) as client:
        resp = client.get("/dashboard/plots.json", headers={"accept-encoding": "gzip"})

    assert resp.status_code == 200
    assert resp.headers["content-encoding"] == "gzip"
    # httpx transparently decodes gzip; on the pre-fix code the header was
    # stripped and this returned the raw deflate bytes instead of the payload.
    assert resp.content == payload
    assert orjson.loads(resp.content)["plots"][0] == "a"


def test_proxy_forwards_client_accept_encoding_upstream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The client's negotiation reaches the sidecar so compression stays end to end."""
    app = _make_app(monkeypatch, max_body_bytes=4096)
    captured: dict[str, object] = {}
    _patch_compressed_upstream(
        monkeypatch, body=b"{}", content_encoding="identity", captured=captured
    )

    with TestClient(app) as client:
        client.get("/dashboard/plots.json", headers={"accept-encoding": "gzip, zstd"})

    headers = {k.lower(): v for k, v in captured["headers"].items()}  # type: ignore[union-attr]
    assert headers["accept-encoding"] == "gzip, zstd"


def test_proxy_zstd_upstream_content_encoding_survives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """zstd is relayed labelled, so a zstd-capable client decodes it cleanly."""
    import zstandard

    payload = orjson.dumps({"plots": ["z"] * 200})
    compressed = zstandard.ZstdCompressor().compress(payload)
    app = _make_app(monkeypatch, max_body_bytes=4096)
    captured: dict[str, object] = {}
    _patch_compressed_upstream(
        monkeypatch, body=compressed, content_encoding="zstd", captured=captured
    )

    with TestClient(app) as client:
        resp = client.get("/dashboard/plots.json", headers={"accept-encoding": "zstd"})

    assert resp.headers["content-encoding"] == "zstd"
    # httpx supports zstd, so an intact label yields the original payload;
    # stripping it left the client holding an undecodable zstd frame.
    assert resp.content == payload
