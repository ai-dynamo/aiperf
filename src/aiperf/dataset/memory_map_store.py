# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Memory-mapped dataset backing store (DatasetManager side).

Streams conversations to disk (optionally zstd-compressed) for later mmap-based
consumption by workers. See :mod:`aiperf.dataset.memory_map_utils` docstring for
the full flow.
"""

import asyncio
import os
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any

import aiofiles

from aiperf.common.constants import BYTES_PER_MIB
from aiperf.common.environment import Environment
from aiperf.common.hooks import on_init, on_stop
from aiperf.common.mixins import AIPerfLifecycleMixin
from aiperf.common.models import Conversation, MemoryMapClientMetadata
from aiperf.dataset.memory_map_models import (
    _CONVERSATION_ENCODER,
    ConversationOffset,
    MemoryMapDatasetIndex,
    _import_zstandard,
)


class MemoryMapDatasetBackingStore(AIPerfLifecycleMixin):
    """Streams conversations to disk as they arrive (DatasetManager side).

    Writes each conversation immediately — constant memory usage regardless of dataset size.
    Preserves insertion order.

    Directory Structure (normal mode)::

        {base_path}/aiperf_mmap_{benchmark_id}/
        ├── dataset.dat   # Serialized conversation data (JSON bytes)
        └── index.dat     # Byte offset index for O(1) lookups

    Directory Structure (compress_only mode for Kubernetes)::

        {base_path}/aiperf_mmap_{benchmark_id}/
        ├── dataset.dat.zst   # zstd-compressed conversation data
        └── index.dat.zst     # zstd-compressed index (offsets are for decompressed data)
    """

    def __init__(
        self,
        benchmark_id: str | None = None,
        compress_only: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize memory-mapped storage.

        Args:
            benchmark_id: Unique identifier for this benchmark run (used for directory isolation)
            compress_only: If True, stream directly to compressed files without creating
                uncompressed versions. Use for Kubernetes where DatasetManager doesn't need
                local mmap access. Workers decompress after download.
            **kwargs: Additional configuration (unused for local mmap)
        """
        super().__init__()
        self._finalized = False
        self._compress_only = compress_only

        # Streaming state (one of _data_file or _stream_writer+_raw_data_file is active)
        self._data_file = None
        self._raw_data_file = None
        self._stream_writer = None
        self._current_offset = 0
        self._offsets: dict[str, ConversationOffset] = {}
        self._session_ids: list[str] = []  # Maintain insertion order

        # File paths point to actual files written:
        # compress_only=True  -> .dat.zst (k8s workers decompress after download)
        # compress_only=False -> .dat    (local mmap access)
        base_path = Environment.DATASET.MMAP_BASE_PATH or Path(tempfile.gettempdir())
        dir_suffix = benchmark_id or f"{os.getpid()}_{id(self)}"
        mmap_dir = base_path / f"aiperf_mmap_{dir_suffix}"
        ext = ".dat.zst" if compress_only else ".dat"
        self._data_path: Path = mmap_dir / f"dataset{ext}"
        self._index_path: Path = mmap_dir / f"index{ext}"
        self._compressed_size: int = 0

    @on_init
    async def _setup(self) -> None:
        """Create output directory and open data file for streaming writes."""
        self._data_path.parent.mkdir(parents=True, exist_ok=True)

        if self._compress_only:
            zstd = _import_zstandard()
            compressor = zstd.ZstdCompressor(level=Environment.COMPRESSION.ZSTD_LEVEL)
            self._raw_data_file = self._data_path.open("wb")
            self._stream_writer = compressor.stream_writer(self._raw_data_file)
            self.info(
                f"Memory-mapped backing store initialized in compress_only mode "
                f"(streaming to {self._data_path})"
            )
        else:
            self._data_file = await aiofiles.open(self._data_path, "wb")
            self.info(
                f"Memory-mapped backing store initialized (streaming to {self._data_path})"
            )

    async def _write_bytes(self, data: bytes) -> None:
        """Write bytes to the active output (compressed stream or async file)."""
        if self._compress_only:
            self._stream_writer.write(data)
        else:
            await self._data_file.write(data)

    async def add_conversation(
        self, conversation_id: str, conversation: Conversation
    ) -> None:
        """Add a single conversation (written immediately to file).

        Args:
            conversation_id: Session ID of the conversation
            conversation: Conversation object to add

        Raises:
            RuntimeError: If already finalized
        """
        if self._finalized:
            raise RuntimeError("Cannot add conversations after finalization")

        conv_bytes = _CONVERSATION_ENCODER.encode(conversation)
        await self._write_bytes(conv_bytes)

        # Track uncompressed offset (workers need this after decompression)
        self._offsets[conversation_id] = ConversationOffset(
            offset=self._current_offset, size=len(conv_bytes)
        )
        self._session_ids.append(conversation_id)
        self._current_offset += len(conv_bytes)

        if len(self._session_ids) % 1000 == 0:
            self.debug(
                f"Streamed {len(self._session_ids)} conversations ({self._current_offset} bytes)"
            )

    async def add_conversations(self, conversations: dict[str, Conversation]) -> None:
        """Add multiple conversations (written immediately to file).

        Args:
            conversations: Dictionary mapping session IDs to Conversation objects

        Raises:
            RuntimeError: If already finalized
        """
        if self._finalized:
            raise RuntimeError("Cannot add conversations after finalization")
        for conversation_id, conversation in conversations.items():
            await self.add_conversation(conversation_id, conversation)

    async def finalize(self) -> None:
        """Finalize by closing data file and writing index.

        Raises:
            RuntimeError: If already finalized
        """
        if self._finalized:
            raise RuntimeError(
                "MemoryMapDatasetBuilder.finalize() called on an already-"
                "finalized builder; each builder instance may be finalized "
                "at most once — construct a new builder for additional datasets."
            )

        index = MemoryMapDatasetIndex(
            conversation_ids=self._session_ids,
            offsets=self._offsets,
            total_size=self._current_offset,
        )
        index_bytes = index.model_dump_json(by_alias=True).encode("utf-8")

        if self._compress_only:
            await self._finalize_compressed(index_bytes)
        else:
            await self._finalize_uncompressed(index_bytes)

        self._finalized = True

    async def _finalize_compressed(self, index_bytes: bytes) -> None:
        """Close zstd stream and write compressed index."""

        def _compress_sync() -> None:
            self._stream_writer.close()
            self._raw_data_file.close()

            zstd = _import_zstandard()
            compressor = zstd.ZstdCompressor(level=Environment.COMPRESSION.ZSTD_LEVEL)
            compressed_index = compressor.compress(index_bytes)
            self._index_path.write_bytes(compressed_index)

        await asyncio.to_thread(_compress_sync)

        compressed_data_size = self._data_path.stat().st_size
        self.info(
            f"Compressed data file finalized: {len(self._session_ids)} conversations, "
            f"{self._current_offset / BYTES_PER_MIB:,.2f} MB uncompressed -> "
            f"{compressed_data_size / BYTES_PER_MIB:,.2f} MB compressed "
            f"({compressed_data_size / self._current_offset * 100 if self._current_offset > 0 else 0:.1f}%)"
        )

        self._compressed_size = compressed_data_size
        self.info(f"Compressed index file created: {self._index_path}")

    async def _finalize_uncompressed(self, index_bytes: bytes) -> None:
        """Close data file and write uncompressed index."""
        await self._data_file.close()
        self.info(
            f"Data file finalized: {len(self._session_ids)} conversations, "
            f"{self._current_offset / BYTES_PER_MIB:,.2f} MB"
        )

        async with aiofiles.open(self._index_path, "wb") as f:
            await f.write(index_bytes)
        self.info(f"Index file created: {self._index_path}")

    def get_client_metadata(self) -> MemoryMapClientMetadata:
        """Return file paths for client initialization.

        Returns:
            MemoryMapClientMetadata with file paths and stats

        Raises:
            RuntimeError: If not finalized
        """
        if not self._finalized:
            raise RuntimeError(
                "Cannot get metadata before finalization. Call finalize() first."
            )

        return MemoryMapClientMetadata(
            data_file_path=self._data_path,
            index_file_path=self._index_path,
            conversation_count=len(self._session_ids),
            total_size_bytes=self._current_offset,
            compressed=self._compress_only,
            compressed_size_bytes=self._compressed_size if self._compress_only else 0,
        )

    @on_stop
    async def _cleanup(self) -> None:
        """Close file handles and delete temp files."""
        if self._stream_writer is not None:
            with suppress(Exception):
                self._stream_writer.close()
        if self._raw_data_file is not None:
            with suppress(Exception):
                self._raw_data_file.close()
        if self._data_file is not None and not self._data_file.closed:
            await self._data_file.close()

        for path in [self._data_path, self._index_path]:
            if path.exists():
                try:
                    path.unlink()
                    self.debug(f"Removed file: {path}")
                except OSError as e:
                    self.warning(f"Error removing file {path}: {e}")

        self.debug("Memory-mapped backing store cleanup complete")
