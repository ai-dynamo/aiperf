# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dataset router component -- owns dataset metadata and /api/dataset endpoints."""

from __future__ import annotations

import asyncio
import pathlib
from typing import Annotated

import aiofiles.os as aio_os
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from aiperf.api.routers.base_router import BaseRouter, component_dependency
from aiperf.common.compression import (
    CompressionEncoding,
    parse_accept_encoding,
    select_encoding,
    stream_file_compressed,
)
from aiperf.common.enums import MessageType
from aiperf.common.environment import Environment
from aiperf.common.hooks import on_message
from aiperf.common.messages import DatasetConfiguredNotification
from aiperf.common.mixins.message_bus_mixin import MessageBusClientMixin
from aiperf.common.models import MemoryMapClientMetadata

DatasetDep = Annotated["DatasetRouter", component_dependency("dataset")]

dataset_router = APIRouter(tags=["Dataset"], include_in_schema=False)


class DatasetStateResponse(BaseModel):
    """Current dataset snapshot for late-joining Kubernetes workers."""

    ready: bool = Field(..., description="Whether the dataset snapshot is ready.")
    benchmark_generation: str | None = Field(
        default=None,
        description="Current benchmark generation identifier.",
    )
    dataset_generation: str | None = Field(
        default=None,
        description="Current dataset generation identifier.",
    )
    conversation_count: int = Field(
        default=0,
        description="Number of conversations in the current dataset.",
    )
    total_size_bytes: int = Field(
        default=0,
        description="Total dataset size in bytes.",
    )
    data_url: str = Field(..., description="Dataset data download endpoint.")
    index_url: str = Field(..., description="Dataset index download endpoint.")


class DatasetRouter(MessageBusClientMixin, BaseRouter):
    """Owns dataset metadata and exposes /api/dataset endpoints."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._dataset_client_metadata: MemoryMapClientMetadata | None = None
        self._benchmark_generation: str | None = None
        self._dataset_generation: str | None = None
        self._dataset_configured = asyncio.Event()

    def get_router(self) -> APIRouter:
        return dataset_router

    @on_message(MessageType.DATASET_CONFIGURED_NOTIFICATION)
    async def _on_dataset_configured(
        self, message: DatasetConfiguredNotification
    ) -> None:
        """Store dataset file paths from DatasetManager."""
        if isinstance(message.client_metadata, MemoryMapClientMetadata):
            self._dataset_client_metadata = message.client_metadata
            self._benchmark_generation = message.benchmark_generation
            self._dataset_generation = message.dataset_generation
            self._dataset_configured.set()
            self.info(
                f"Dataset configured: {message.client_metadata.conversation_count} conversations, "
                f"compressed={message.client_metadata.compressed}"
            )
        else:
            self.warning(
                f"Received dataset metadata with unsupported type: {type(message.client_metadata)}"
            )

    @property
    def dataset_client_metadata(self) -> MemoryMapClientMetadata | None:
        return self._dataset_client_metadata

    @property
    def benchmark_generation(self) -> str | None:
        return self._benchmark_generation

    @property
    def dataset_generation(self) -> str | None:
        return self._dataset_generation

    @property
    def dataset_configured(self) -> asyncio.Event:
        return self._dataset_configured


async def _stream_dataset_file(
    file_path: pathlib.Path,
    compressed: bool,
    accept_encoding: str | None,
    file_type: str,
) -> StreamingResponse:
    """Stream a dataset file with compression support.

    Args:
        file_path: Path to the file (.dat or .dat.zst depending on mode).
        compressed: Whether the file is pre-compressed with zstd.
        accept_encoding: The Accept-Encoding header value.
        file_type: Type of file for error messages ("Dataset" or "Index").
    """
    if not await aio_os.path.exists(file_path):
        raise HTTPException(
            status_code=404, detail=f"{file_type} file not found: {file_path.name}"
        )

    if compressed:
        accepted = parse_accept_encoding(accept_encoding or "")
        if accepted.get("zstd", 0) <= 0:
            raise HTTPException(
                status_code=406,
                detail=f"{file_type} is pre-compressed with zstd. Client must accept zstd encoding.",
            )
        stream = stream_file_compressed(file_path, CompressionEncoding.IDENTITY)
        encoding = CompressionEncoding.ZSTD
    else:
        encoding = select_encoding(accept_encoding)
        stream = stream_file_compressed(file_path, encoding)

    return StreamingResponse(
        stream,
        media_type="application/octet-stream",
        headers={
            "Content-Encoding": encoding,
            "Content-Disposition": f'attachment; filename="{file_path.name}"',
        },
    )


async def _wait_for_dataset_metadata(component: DatasetRouter) -> None:
    """Wait for dataset configuration with timeout."""
    if not component.dataset_configured.is_set():
        try:
            component.info("Waiting for dataset configuration...")
            await asyncio.wait_for(
                component.dataset_configured.wait(),
                timeout=Environment.DATASET.CONFIGURATION_TIMEOUT,
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=503,
                detail="Dataset not yet configured. DatasetManager has not sent configuration.",
            ) from None

    if component.dataset_client_metadata is None:
        raise HTTPException(status_code=503, detail="Dataset metadata not available")


@dataset_router.get("/api/dataset/data")
async def get_dataset_data(
    component: DatasetDep, request: Request
) -> StreamingResponse:
    """Stream the dataset.dat file for Kubernetes workers (compressed)."""
    await _wait_for_dataset_metadata(component)
    metadata = component.dataset_client_metadata
    return await _stream_dataset_file(
        metadata.data_file_path,
        metadata.compressed,
        request.headers.get("accept-encoding"),
        "Dataset",
    )


@dataset_router.get("/api/dataset/index")
async def get_dataset_index(
    component: DatasetDep, request: Request
) -> StreamingResponse:
    """Stream the index.dat file for Kubernetes workers (compressed)."""
    await _wait_for_dataset_metadata(component)
    metadata = component.dataset_client_metadata
    return await _stream_dataset_file(
        metadata.index_file_path,
        metadata.compressed,
        request.headers.get("accept-encoding"),
        "Index",
    )


@dataset_router.get("/api/dataset/state", response_model=DatasetStateResponse)
async def get_dataset_state(component: DatasetDep) -> DatasetStateResponse:
    """Return the current versioned dataset snapshot for late joiners."""
    await _wait_for_dataset_metadata(component)
    metadata = component.dataset_client_metadata
    return DatasetStateResponse(
        ready=True,
        benchmark_generation=component.benchmark_generation,
        dataset_generation=component.dataset_generation,
        conversation_count=metadata.conversation_count,
        total_size_bytes=metadata.total_size_bytes,
        data_url="/api/dataset/data",
        index_url="/api/dataset/index",
    )
