# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Download and cache behavior of ``BasePublicDatasetLoader._load_dataset``.

These tests drive the real ``AioHttpClient`` against a loopback aiohttp server so
that the failure records the loader sees are the ones the transport really builds.
"""

import asyncio
import socket
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from aiohttp import web

from aiperf.common.exceptions import DatasetLoaderError
from aiperf.common.models import RequestRecord
from aiperf.dataset.loader.base_public_dataset import BasePublicDatasetLoader


class _PublicLoader(BasePublicDatasetLoader):
    tag = "TestPublic"
    url = "http://unused.invalid/dataset"
    filename = "test_public.json"


@pytest.fixture
async def http_server() -> AsyncIterator[Callable[[web.Application], Awaitable[str]]]:
    runners: list[web.AppRunner] = []

    async def start(app: web.Application) -> str:
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        runners.append(runner)
        port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
        return f"http://127.0.0.1:{port}"

    yield start
    for runner in runners:
        await runner.cleanup()


@pytest.fixture
def make_loader(tmp_path: Path) -> Callable[[str], _PublicLoader]:
    def make(url: str) -> _PublicLoader:
        loader = _PublicLoader()
        loader.url = url
        loader.cache_filepath = tmp_path / "cache" / "test_public.json"
        return loader

    return make


def _app(status: int, body: str, hits: list[int] | None = None) -> web.Application:
    async def handler(_: web.Request) -> web.Response:
        if hits is not None:
            hits.append(1)
        return web.Response(status=status, text=body)

    app = web.Application()
    app.router.add_get("/dataset", handler)
    return app


def _unused_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class TestFailedDownload:
    async def test_load_dataset_http_error_raises_with_url_status_and_body(
        self, http_server, make_loader
    ):
        base = await http_server(_app(503, "controlled dataset download failure"))
        loader = make_loader(f"{base}/dataset")

        with pytest.raises(DatasetLoaderError) as exc_info:
            await loader._load_dataset(headers={})

        message = str(exc_info.value)
        assert f"{base}/dataset" in message
        assert "503" in message
        assert "controlled dataset download failure" in message

    async def test_load_dataset_http_error_writes_no_cache(
        self, http_server, make_loader
    ):
        base = await http_server(_app(404, "not found"))
        loader = make_loader(f"{base}/dataset")

        with pytest.raises(DatasetLoaderError):
            await loader._load_dataset(headers={})

        assert not loader.cache_filepath.exists()

    async def test_load_dataset_long_error_body_is_truncated(
        self, http_server, make_loader
    ):
        base = await http_server(_app(500, "x" * 10_000))
        loader = make_loader(f"{base}/dataset")

        with pytest.raises(DatasetLoaderError) as exc_info:
            await loader._load_dataset(headers={})

        assert len(str(exc_info.value)) < 1_000

    async def test_load_dataset_connection_error_raises_dataset_loader_error(
        self, make_loader
    ):
        url = f"http://127.0.0.1:{_unused_port()}/dataset"
        loader = make_loader(url)

        with pytest.raises(DatasetLoaderError) as exc_info:
            await loader._load_dataset(headers={})

        message = str(exc_info.value)
        assert url in message
        assert "ClientConnectorError" in message
        assert not loader.cache_filepath.exists()

    async def test_load_dataset_record_without_response_raises_dataset_loader_error(
        self, make_loader
    ):
        loader = make_loader("http://127.0.0.1:1/dataset")

        with (
            patch.object(
                loader.http_client,
                "get_request",
                AsyncMock(return_value=RequestRecord()),
            ),
            pytest.raises(DatasetLoaderError, match="no response"),
        ):
            await loader._load_dataset(headers={})

        assert not loader.cache_filepath.exists()

    async def test_load_dataset_request_raises_closes_http_client(self, make_loader):
        loader = make_loader("http://127.0.0.1:1/dataset")
        client = loader.http_client

        with (
            patch.object(
                client, "get_request", AsyncMock(side_effect=asyncio.CancelledError)
            ),
            patch.object(client, "close", AsyncMock()) as close,
            pytest.raises(asyncio.CancelledError),
        ):
            await loader._load_dataset(headers={})

        close.assert_awaited_once()


class TestSuccessfulDownload:
    async def test_load_dataset_success_caches_and_reuses_body(
        self, http_server, make_loader
    ):
        hits: list[int] = []
        base = await http_server(_app(200, '{"dataset": "ok"}', hits))
        first = make_loader(f"{base}/dataset")

        assert await first._load_dataset(headers={}) == '{"dataset": "ok"}'
        assert first.cache_filepath.read_text() == '{"dataset": "ok"}'

        second = make_loader(f"{base}/dataset")
        assert await second._load_dataset(headers={}) == '{"dataset": "ok"}'
        assert len(hits) == 1
