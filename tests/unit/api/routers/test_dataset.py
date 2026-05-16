# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for DatasetRouter."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest
import zstandard
from fastapi import FastAPI, HTTPException
from httpx import ASGITransport, AsyncClient
from pytest import param
from starlette.testclient import TestClient

from aiperf.api.routers.dataset import DatasetRouter, _stream_dataset_file
from aiperf.common.messages import DatasetConfiguredNotification
from aiperf.common.models import MemoryMapClientMetadata
from aiperf.common.models.dataset_models import DatasetMetadata
from aiperf.config import BenchmarkRun


@pytest.fixture
def dataset_router(mock_zmq, router_config: BenchmarkRun) -> DatasetRouter:
    return DatasetRouter(run=router_config)


def _make_app(router: DatasetRouter) -> FastAPI:
    app = FastAPI()
    app.state.dataset = router
    app.include_router(router.get_router())
    return app


@pytest.fixture
def dataset_client(dataset_router: DatasetRouter) -> TestClient:
    return TestClient(_make_app(dataset_router))


@pytest.fixture
async def dataset_async_client(
    dataset_router: DatasetRouter,
) -> AsyncClient:
    transport = ASGITransport(app=_make_app(dataset_router))
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


class TestDatasetEndpoints:
    """Test the /api/dataset/* endpoints."""

    @pytest.mark.asyncio
    async def test_dataset_data_timeout_returns_503(
        self,
        dataset_async_client: AsyncClient,
        dataset_router: DatasetRouter,
        time_traveler,
    ) -> None:
        dataset_router._dataset_configured = asyncio.Event()
        dataset_router._dataset_client_metadata = None

        response = await dataset_async_client.get("/api/dataset/data")
        assert response.status_code == 503
        assert "not yet configured" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_dataset_index_timeout_returns_503(
        self,
        dataset_async_client: AsyncClient,
        dataset_router: DatasetRouter,
        time_traveler,
    ) -> None:
        dataset_router._dataset_configured = asyncio.Event()
        dataset_router._dataset_client_metadata = None

        response = await dataset_async_client.get("/api/dataset/index")
        assert response.status_code == 503

    @pytest.mark.asyncio
    async def test_dataset_data_no_metadata_returns_503(
        self,
        dataset_async_client: AsyncClient,
        dataset_router: DatasetRouter,
    ) -> None:
        dataset_router._dataset_configured = asyncio.Event()
        dataset_router._dataset_configured.set()
        dataset_router._dataset_client_metadata = None

        response = await dataset_async_client.get("/api/dataset/data")
        assert response.status_code == 503
        assert "metadata not available" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_dataset_data_file_not_found_returns_404(
        self,
        dataset_async_client: AsyncClient,
        dataset_router: DatasetRouter,
        tmp_path,
    ) -> None:
        dataset_router._dataset_configured = asyncio.Event()
        dataset_router._dataset_configured.set()
        dataset_router._dataset_client_metadata = MemoryMapClientMetadata(
            data_file_path=tmp_path / "nonexistent.dat",
            index_file_path=tmp_path / "index.dat",
            conversation_count=100,
        )

        response = await dataset_async_client.get("/api/dataset/data")
        assert response.status_code == 404
        assert "dataset file not found" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_dataset_index_file_not_found_returns_404(
        self,
        dataset_async_client: AsyncClient,
        dataset_router: DatasetRouter,
        tmp_path,
    ) -> None:
        dataset_router._dataset_configured = asyncio.Event()
        dataset_router._dataset_configured.set()
        dataset_router._dataset_client_metadata = MemoryMapClientMetadata(
            data_file_path=tmp_path / "data.dat",
            index_file_path=tmp_path / "nonexistent.dat",
            conversation_count=100,
        )

        response = await dataset_async_client.get("/api/dataset/index")
        assert response.status_code == 404
        assert "index file not found" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "accept_encoding",
        [
            param("gzip", id="no-zstd"),
            param("zstd;q=0, gzip", id="zstd-rejected-q0"),
            param("zstd;q=0", id="zstd-rejected-q0-only"),
            param("", id="empty-header"),
        ],
    )
    async def test_dataset_compress_only_mode_rejects_without_zstd(
        self,
        dataset_async_client: AsyncClient,
        dataset_router: DatasetRouter,
        tmp_path,
        accept_encoding: str,
    ) -> None:
        compressed_file = tmp_path / "data.dat.zst"
        compressed_file.write_bytes(b"compressed data")

        dataset_router._dataset_configured = asyncio.Event()
        dataset_router._dataset_configured.set()
        dataset_router._dataset_client_metadata = MemoryMapClientMetadata(
            data_file_path=compressed_file,
            index_file_path=tmp_path / "index.dat",
            compressed=True,
            conversation_count=100,
        )

        response = await dataset_async_client.get(
            "/api/dataset/data",
            headers={"Accept-Encoding": accept_encoding},
        )
        assert response.status_code == 406
        assert "zstd" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_dataset_compress_only_mode_rejects_no_accept_encoding_header(
        self, tmp_path
    ) -> None:
        compressed_file = tmp_path / "data.dat.zst"
        compressed_file.write_bytes(b"compressed data")

        with pytest.raises(HTTPException) as exc_info:
            await _stream_dataset_file(compressed_file, True, None, "Dataset")
        assert exc_info.value.status_code == 406


class TestDatasetEndpointSuccessfulStreaming:
    """Test successful dataset file streaming scenarios."""

    @pytest.mark.asyncio
    async def test_dataset_data_streams_file(
        self,
        dataset_async_client: AsyncClient,
        dataset_router: DatasetRouter,
        tmp_path,
    ) -> None:
        data_file = tmp_path / "data.dat"
        data_file.write_bytes(b"test dataset content")

        dataset_router._dataset_configured = asyncio.Event()
        dataset_router._dataset_configured.set()
        dataset_router._dataset_client_metadata = MemoryMapClientMetadata(
            data_file_path=data_file,
            index_file_path=tmp_path / "index.dat",
            conversation_count=10,
        )

        response = await dataset_async_client.get(
            "/api/dataset/data",
            headers={"Accept-Encoding": "identity"},
        )
        assert response.status_code == 200
        assert response.content == b"test dataset content"
        assert "data.dat" in response.headers["content-disposition"]

    @pytest.mark.asyncio
    async def test_dataset_index_streams_file(
        self,
        dataset_async_client: AsyncClient,
        dataset_router: DatasetRouter,
        tmp_path,
    ) -> None:
        index_file = tmp_path / "index.dat"
        index_file.write_bytes(b"test index content")

        dataset_router._dataset_configured = asyncio.Event()
        dataset_router._dataset_configured.set()
        dataset_router._dataset_client_metadata = MemoryMapClientMetadata(
            data_file_path=tmp_path / "data.dat",
            index_file_path=index_file,
            conversation_count=10,
        )

        response = await dataset_async_client.get(
            "/api/dataset/index",
            headers={"Accept-Encoding": "identity"},
        )
        assert response.status_code == 200
        assert response.content == b"test index content"

    @pytest.mark.asyncio
    async def test_dataset_compress_only_mode_accepts_zstd(
        self, dataset_router: DatasetRouter, tmp_path
    ) -> None:
        original_data = b"test dataset content for zstd"
        cctx = zstandard.ZstdCompressor()
        compressed_data = cctx.compress(original_data)

        compressed_file = tmp_path / "data.dat.zst"
        compressed_file.write_bytes(compressed_data)

        dataset_router._dataset_configured = asyncio.Event()
        dataset_router._dataset_configured.set()
        dataset_router._dataset_client_metadata = MemoryMapClientMetadata(
            data_file_path=compressed_file,
            index_file_path=tmp_path / "index.dat",
            compressed=True,
            conversation_count=10,
        )

        transport = ASGITransport(app=_make_app(dataset_router))
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as raw_client:
            response = await raw_client.get(
                "/api/dataset/data",
                headers={"Accept-Encoding": "zstd, gzip"},
            )
        assert response.status_code == 200
        assert response.content == original_data

    @pytest.mark.asyncio
    async def test_dataset_compressed_mode_streams_zstd(
        self, dataset_router: DatasetRouter, tmp_path
    ) -> None:
        original_data = b"uncompressed data"
        cctx = zstandard.ZstdCompressor()
        compressed_data = cctx.compress(original_data)

        compressed_file = tmp_path / "data.dat.zst"
        compressed_file.write_bytes(compressed_data)

        dataset_router._dataset_configured = asyncio.Event()
        dataset_router._dataset_configured.set()
        dataset_router._dataset_client_metadata = MemoryMapClientMetadata(
            data_file_path=compressed_file,
            index_file_path=tmp_path / "index.dat",
            compressed=True,
            conversation_count=10,
        )

        transport = ASGITransport(app=_make_app(dataset_router))
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as raw_client:
            response = await raw_client.get(
                "/api/dataset/data",
                headers={"Accept-Encoding": "zstd"},
            )
        assert response.status_code == 200
        assert response.content == original_data

    @pytest.mark.asyncio
    async def test_dataset_gzip_fallback_when_no_zstd(
        self,
        dataset_async_client: AsyncClient,
        dataset_router: DatasetRouter,
        tmp_path,
    ) -> None:
        data_file = tmp_path / "data.dat"
        data_file.write_bytes(b"uncompressed data")

        dataset_router._dataset_configured = asyncio.Event()
        dataset_router._dataset_configured.set()
        dataset_router._dataset_client_metadata = MemoryMapClientMetadata(
            data_file_path=data_file,
            index_file_path=tmp_path / "index.dat",
            conversation_count=10,
        )

        response = await dataset_async_client.get(
            "/api/dataset/data",
            headers={"Accept-Encoding": "gzip"},
        )
        assert response.status_code == 200
        assert response.headers["content-encoding"] == "gzip"


class TestDatasetMixin:
    """Test DatasetMixin message handling and properties."""

    @pytest.mark.asyncio
    async def test_on_dataset_configured_stores_metadata(
        self, dataset_router: DatasetRouter, tmp_path
    ) -> None:
        dataset_router._dataset_configured = asyncio.Event()
        dataset_router._dataset_client_metadata = None

        metadata = MemoryMapClientMetadata(
            data_file_path=tmp_path / "data.dat",
            index_file_path=tmp_path / "index.dat",
            conversation_count=42,
        )
        ds_metadata = DatasetMetadata(sampling_strategy="random")
        message = DatasetConfiguredNotification(
            service_id="dataset_manager",
            metadata=ds_metadata,
            client_metadata=metadata,
            benchmark_generation="test-bench",
            dataset_generation="test-bench:dataset",
        )

        await dataset_router._on_dataset_configured(message)

        assert dataset_router._dataset_client_metadata is metadata
        assert dataset_router._dataset_configured.is_set()

    @pytest.mark.asyncio
    async def test_on_dataset_configured_unsupported_type_warns(
        self, dataset_router: DatasetRouter
    ) -> None:
        dataset_router._dataset_configured = asyncio.Event()
        dataset_router._dataset_client_metadata = None

        message = MagicMock()
        message.client_metadata = "not_a_memory_map_metadata"

        await dataset_router._on_dataset_configured(message)

        assert dataset_router._dataset_client_metadata is None
        assert not dataset_router._dataset_configured.is_set()

    def test_dataset_client_metadata_property(
        self, dataset_router: DatasetRouter
    ) -> None:
        dataset_router._dataset_client_metadata = None
        assert dataset_router.dataset_client_metadata is None

        sentinel = MagicMock()
        dataset_router._dataset_client_metadata = sentinel
        assert dataset_router.dataset_client_metadata is sentinel

    def test_dataset_configured_property(self, dataset_router: DatasetRouter) -> None:
        event = asyncio.Event()
        dataset_router._dataset_configured = event
        assert dataset_router.dataset_configured is event
