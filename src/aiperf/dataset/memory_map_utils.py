# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Memory-mapped dataset for zero-copy conversation access.

Eliminates the DatasetManager network bottleneck at high QPS by letting workers
read conversations directly from shared files in O(1) time.

Flow (local):
    1. DatasetManager writes conversations to disk via MemoryMapDatasetBackingStore
    2. Workers read via mmap (zero-copy) through MemoryMapDatasetClientStore

Flow (Kubernetes):
    1. DatasetManager streams conversations to zstd-compressed files (compress_only mode)
    2. WorkerPodManager downloads compressed files once per pod from control-plane via HTTP API
    3. WorkerPodManager decompresses files locally
    4. Workers read via mmap through MemoryMapDatasetClientStore

Storage formats (``MemoryMapDatasetIndex.format``):
    - ``conversation``: Each entry is a JSON-serialized Conversation object.
      Used for normal datasets. Workers deserialize to get a full Conversation.
    - ``payload_bytes``: Each entry is pre-encoded payload bytes (one per turn).
      Used for verbatim API replay. Workers read bytes directly from the mmap
      and send them to the transport without deserialization.
"""

from __future__ import annotations

import asyncio
import mmap
import os
import tempfile
import types
import weakref
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiofiles
import orjson
from pydantic import Field, field_validator

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.constants import BYTES_PER_MIB
from aiperf.common.enums import MemoryMapFormat
from aiperf.common.environment import Environment
from aiperf.common.exceptions import (
    MemoryMapFileOperationError,
    MemoryMapSerializationError,
)
from aiperf.common.hooks import on_init, on_stop
from aiperf.common.mixins import AIPerfLifecycleMixin
from aiperf.common.models import (
    AIPerfBaseModel,
    Conversation,
    MemoryMapClientMetadata,
    Turn,
)

_logger = AIPerfLogger(__name__)


def _import_zstandard() -> types.ModuleType:
    """Lazy-import zstandard or raise a helpful error."""
    try:
        import zstandard

        return zstandard
    except ImportError as e:
        raise ImportError(
            "zstandard library required for compression. Install with: pip install zstandard"
        ) from e


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
        format: MemoryMapFormat = MemoryMapFormat.CONVERSATION,
        **kwargs: Any,
    ) -> None:
        """Initialize memory-mapped storage.

        Args:
            benchmark_id: Unique identifier for this benchmark run (used for directory isolation)
            compress_only: If True, stream directly to compressed files without creating
                uncompressed versions. Use for Kubernetes where DatasetManager doesn't need
                local mmap access. Workers decompress after download.
            format: Storage format for the dataset files. ``CONVERSATION`` serializes
                each Conversation as JSON; ``PAYLOAD_BYTES`` writes pre-encoded per-turn
                payload bytes for verbatim replay.
            **kwargs: Additional configuration (unused for local mmap)
        """
        super().__init__()
        self._finalized = False
        self._compress_only = compress_only
        self._format: MemoryMapFormat = format

        # Streaming state (one of _data_file or _stream_writer+_raw_data_file is active)
        self._data_file = None
        self._raw_data_file = None
        self._stream_writer = None
        self._current_offset = 0
        self._offsets: dict[str, ConversationOffset] = {}
        self._payload_offsets: dict[str, list[PayloadOffset]] = {}
        self._session_ids: list[str] = []  # Maintain insertion order

        # File paths (configurable base path for k8s mounted volumes)
        # Directory structure: {base_path}/aiperf_mmap_{benchmark_id}/
        base_path = Environment.DATASET.MMAP_BASE_PATH or Path(tempfile.gettempdir())
        dir_suffix = benchmark_id or f"{os.getpid()}_{id(self)}"
        mmap_dir = base_path / f"aiperf_mmap_{dir_suffix}"
        self._data_path: Path = mmap_dir / "dataset.dat"
        self._index_path: Path = mmap_dir / "index.dat"
        # Pre-compressed files for Kubernetes HTTP transfer
        self._compressed_data_path: Path = mmap_dir / "dataset.dat.zst"
        self._compressed_index_path: Path = mmap_dir / "index.dat.zst"
        self._compressed_size: int = 0

    @on_init
    async def _setup(self) -> None:
        """Create output directory and open data file for streaming writes."""
        await asyncio.to_thread(
            self._data_path.parent.mkdir, parents=True, exist_ok=True
        )

        if self._compress_only:
            zstd = _import_zstandard()
            compressor = zstd.ZstdCompressor(level=Environment.COMPRESSION.ZSTD_LEVEL)
            # zstd stream_writer expects a sync file-like object; open off the loop.
            self._raw_data_file = await asyncio.to_thread(
                self._compressed_data_path.open, "wb"
            )
            self._stream_writer = compressor.stream_writer(self._raw_data_file)
            self.info(
                f"Memory-mapped backing store initialized in compress_only mode "
                f"(streaming to {self._compressed_data_path})"
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

        if self._format == MemoryMapFormat.PAYLOAD_BYTES:
            # Pre-encode each turn's raw_payload and write the bytes directly;
            # workers replay these verbatim with no deserialization. Persist
            # turn scalars in the index so metric enrichment can restore
            # max_tokens / scheduled_send_ms without the full Conversation.
            turn_offsets: list[PayloadOffset] = []
            for turn in conversation.turns:
                payload_bytes = orjson.dumps(turn.raw_payload)
                turn_offsets.append(
                    PayloadOffset(
                        offset=self._current_offset,
                        size=len(payload_bytes),
                        max_tokens=_resolve_turn_max_tokens(turn),
                        timestamp=turn.timestamp,
                    )
                )
                self._current_offset += len(payload_bytes)
                await self._write_bytes(payload_bytes)
            self._payload_offsets[conversation_id] = turn_offsets
        else:
            conv_bytes = conversation.model_dump_json().encode("utf-8")
            # Track uncompressed offset (workers need this after decompression)
            self._offsets[conversation_id] = ConversationOffset(
                offset=self._current_offset, size=len(conv_bytes)
            )
            self._current_offset += len(conv_bytes)
            await self._write_bytes(conv_bytes)

        self._session_ids.append(conversation_id)

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
                "MemoryMapDatasetBackingStore.finalize called twice; the data file "
                "and index are already written and cannot be re-finalized."
            )

        index = MemoryMapDatasetIndex(
            conversation_ids=self._session_ids,
            format=self._format,
            offsets=self._offsets,
            payload_offsets=self._payload_offsets,
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
        self._stream_writer.close()
        self._raw_data_file.close()
        compressed_data_size = self._compressed_data_path.stat().st_size

        self.info(
            f"Compressed data file finalized: {len(self._session_ids)} conversations, "
            f"{self._current_offset / BYTES_PER_MIB:,.2f} MB uncompressed -> "
            f"{compressed_data_size / BYTES_PER_MIB:,.2f} MB compressed "
            f"({compressed_data_size / self._current_offset * 100 if self._current_offset > 0 else 0:.1f}%)"
        )

        zstd = _import_zstandard()
        compressor = zstd.ZstdCompressor(level=Environment.COMPRESSION.ZSTD_LEVEL)
        compressed_index = await asyncio.to_thread(compressor.compress, index_bytes)
        async with aiofiles.open(self._compressed_index_path, "wb") as f:
            await f.write(compressed_index)

        self._compressed_size = compressed_data_size
        self.info(f"Compressed index file created: {self._compressed_index_path}")

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
            format=self._format,
            conversation_count=len(self._session_ids),
            total_size_bytes=self._current_offset,
            compressed=self._compress_only,
            compressed_data_file_path=self._compressed_data_path if self._compress_only else None,
            compressed_index_file_path=self._compressed_index_path if self._compress_only else None,
            compressed_size_bytes=self._compressed_size if self._compress_only else 0,
        )  # fmt: skip

    def adopt_existing_files(
        self,
        *,
        session_ids: list[str],
        total_size_bytes: int,
        compressed_size_bytes: int = 0,
    ) -> None:
        """Mark this store as finalized over already-on-disk files.

        Used by the dataset cache HIT path: ``dataset.dat`` / ``index.dat`` are
        already on disk in the run mmap dir (copied from the cache), so we never
        call ``initialize()`` (which would open a writer) or ``finalize()``
        (which would re-write the index). The on-stop cleanup hook still runs
        and unlinks the run dir as if the writer had produced the files itself.
        """
        if self._finalized:
            raise RuntimeError(
                "adopt_existing_files called on an already-finalized store."
            )
        # compress_only (Kubernetes) cache HIT restores only .dat.zst files;
        # uncompressed dataset.dat / index.dat are never written.
        if self._compress_only:
            data_ok = self._compressed_data_path.exists()
            index_ok = self._compressed_index_path.exists()
            missing = (self._compressed_data_path, self._compressed_index_path)
        else:
            data_ok = self._data_path.exists()
            index_ok = self._index_path.exists()
            missing = (self._data_path, self._index_path)
        if not data_ok or not index_ok:
            raise FileNotFoundError(
                f"adopt_existing_files requires both files on disk: "
                f"{missing[0]}, {missing[1]}"
            )
        self._session_ids = list(session_ids)
        self._current_offset = total_size_bytes
        self._compressed_size = compressed_size_bytes if self._compress_only else 0
        self._finalized = True

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

        for path in [
            self._data_path,
            self._index_path,
            self._compressed_data_path,
            self._compressed_index_path,
        ]:
            if path.exists():
                try:
                    path.unlink()
                    self.debug(f"Removed file: {path}")
                except OSError as e:
                    self.warning(f"Error removing file {path}: {e}")

        self.debug("Memory-mapped backing store cleanup complete")


class MemoryMapDatasetClientStore(AIPerfLifecycleMixin):
    """Reads conversations from memory-mapped files (Worker side).

    Uses mmap for zero-copy reads — the OS pages data into memory as needed.
    """

    def __init__(self, client_metadata: MemoryMapClientMetadata, **kwargs) -> None:
        """Initialize from metadata provided by backing store.

        Args:
            client_metadata: Typed metadata from MemoryMapDatasetBackingStore.get_client_metadata()
        """
        super().__init__(**kwargs)
        self._data_path: Path = client_metadata.data_file_path
        self._index_path: Path = client_metadata.index_file_path
        self._client: MemoryMapDatasetClient | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    @on_init
    async def _setup(self) -> None:
        """Open memory-mapped files (read-only)."""
        self._loop = asyncio.get_running_loop()
        self.debug(
            lambda: f"Opening memory-mapped files: data={self._data_path}, index={self._index_path}"
        )
        self._client = MemoryMapDatasetClient(self._data_path, self._index_path)
        self.debug(
            lambda: f"Memory-mapped client store initialized with "
            f"{len(self._client.index.conversation_ids)} conversations"
        )

    async def get_conversation(self, conversation_id: str) -> Conversation:
        """Retrieve conversation from memory-mapped file.

        Runs in executor since mmap reads can block on page faults.

        Args:
            conversation_id: Session ID of the conversation

        Returns:
            Conversation object

        Raises:
            KeyError: If conversation_id not found
        """
        if self._client is None or self._loop is None:
            raise RuntimeError("Client store not initialized. Call initialize() first.")
        return await self._loop.run_in_executor(
            None, self._client.get_conversation, conversation_id
        )

    async def get_payload_bytes(
        self, conversation_id: str, turn_index: int
    ) -> bytes | None:
        """Retrieve pre-encoded payload bytes for a specific turn.

        Args:
            conversation_id: The session ID of the conversation
            turn_index: Turn index within the conversation

        Returns:
            Pre-encoded JSON bytes, or None when the dataset is not in
            PAYLOAD_BYTES format or the turn has no payload.
        """
        if self._client is None or self._loop is None:
            raise RuntimeError("Client store not initialized. Call initialize() first.")
        return await self._loop.run_in_executor(
            None, self._client.get_payload_bytes, conversation_id, turn_index
        )

    async def get_payload_turn(
        self, conversation_id: str, turn_index: int
    ) -> PayloadTurnData | None:
        """Retrieve payload bytes plus turn scalars for metric enrichment.

        Args:
            conversation_id: The session ID of the conversation
            turn_index: Turn index within the conversation

        Returns:
            ``PayloadTurnData`` or None when the dataset is not in
            PAYLOAD_BYTES format or the turn has no payload.
        """
        if self._client is None or self._loop is None:
            raise RuntimeError("Client store not initialized. Call initialize() first.")
        return await self._loop.run_in_executor(
            None, self._client.get_payload_turn, conversation_id, turn_index
        )

    @on_stop
    async def _cleanup(self) -> None:
        """Close memory-mapped files."""
        if self._client:
            self.debug("Closing memory-mapped files")
            self._client.close()
            self.debug("Memory-mapped client store cleanup complete")


class ConversationOffset(AIPerfBaseModel):
    """Offset information for a single conversation in the memory-mapped file."""

    offset: int = Field(ge=0, description="Byte offset where conversation data starts")
    size: int = Field(ge=0, description="Size of the conversation data in bytes")


class PayloadOffset(AIPerfBaseModel):
    """Offset information for a single turn's payload in the data file."""

    offset: int = Field(ge=0, description="Byte offset where payload data starts")
    size: int = Field(ge=0, description="Size of the payload data in bytes")
    max_tokens: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Requested output length for this turn (from Turn.max_tokens or "
            "wire JSON). Restored onto reconstructed Turns so OSL-mismatch "
            "metrics stay live on the PAYLOAD_BYTES path."
        ),
    )
    timestamp: int | float | None = Field(
        default=None,
        description=(
            "Schedule timestamp in milliseconds from Turn.timestamp. Restored "
            "onto reconstructed Turns so schedule-lag metrics stay live on "
            "the PAYLOAD_BYTES path."
        ),
    )


# Wire-body keys that encode the same Turn.max_tokens scalar across endpoints.
_WIRE_MAX_TOKEN_KEYS: tuple[str, ...] = (
    "max_tokens",
    "max_completion_tokens",
    "max_output_tokens",
)


@dataclass(slots=True)
class PayloadTurnData:
    """Pre-encoded payload bytes plus turn scalars for metric enrichment."""

    payload_bytes: bytes
    max_tokens: int | None = None
    timestamp: int | float | None = None


def max_tokens_from_wire_payload(payload: dict[str, Any] | None) -> int | None:
    """Extract a positive max-tokens value from a wire JSON body, if present."""
    if not isinstance(payload, dict):
        return None
    for key in _WIRE_MAX_TOKEN_KEYS:
        value = payload.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int) and value >= 1:
            return value
    return None


def _resolve_turn_max_tokens(turn: Turn) -> int | None:
    """Prefer Turn.max_tokens; fall back to wire-body keys in raw_payload."""
    if turn.max_tokens is not None:
        return turn.max_tokens
    raw = turn.raw_payload
    return max_tokens_from_wire_payload(raw if isinstance(raw, dict) else None)


def turn_from_payload_turn(entry: PayloadTurnData) -> Turn:
    """Rebuild a minimal Turn from PAYLOAD_BYTES entry data."""
    raw_payload = orjson.loads(entry.payload_bytes)
    max_tokens = entry.max_tokens
    if max_tokens is None and isinstance(raw_payload, dict):
        max_tokens = max_tokens_from_wire_payload(raw_payload)
    return Turn(
        role="user",
        raw_payload=raw_payload,
        max_tokens=max_tokens,
        timestamp=entry.timestamp,
    )


class MemoryMapDatasetIndex(AIPerfBaseModel):
    """Index structure for the memory-mapped dataset.

    All data is stored as uncompressed JSON bytes serialized with orjson.
    """

    conversation_ids: list[str] = Field(
        default_factory=list, description="List of all conversation IDs in the dataset"
    )
    format: MemoryMapFormat = Field(
        default=MemoryMapFormat.CONVERSATION,
        description="Storage format: 'conversation' for serialized Conversations, "
        "'payload_bytes' for pre-encoded per-turn payload bytes.",
    )
    offsets: dict[str, ConversationOffset] = Field(
        default_factory=dict,
        description="Mapping of conversation IDs to their byte offsets and sizes",
    )
    payload_offsets: dict[str, list[PayloadOffset]] = Field(
        default_factory=dict,
        description="Mapping of conversation IDs to per-turn payload offsets. "
        "Used when format is 'payload_bytes'.",
    )
    total_size: int = Field(
        default=0, ge=0, description="Total size of the serialized dataset in bytes"
    )

    @field_validator("conversation_ids")
    @classmethod
    def validate_conversation_ids(cls, v: list[str]) -> list[str]:
        """Ensure conversation_ids are unique."""
        if len(v) != len(set(v)):
            raise ValueError("conversation_ids must contain unique values")
        return v


class MemoryMapDatasetClient:
    """Low-level mmap client for reading conversations.

    Use as context manager or call close() explicitly.
    """

    def __init__(self, data_file_path: Path | str, index_file_path: Path | str) -> None:
        """Open memory-mapped files and load the index.

        Args:
            data_file_path: Path to the memory-mapped data file
            index_file_path: Path to the memory-mapped index file

        Raises:
            MemoryMapFileOperationError: If files cannot be opened
            MemoryMapSerializationError: If index data is invalid
        """
        self.data_file_path = (
            Path(data_file_path) if isinstance(data_file_path, str) else data_file_path
        )
        self.index_file_path = (
            Path(index_file_path)
            if isinstance(index_file_path, str)
            else index_file_path
        )

        if not self.data_file_path.exists():
            raise MemoryMapFileOperationError(f"Data file not found: {data_file_path}")
        if not self.index_file_path.exists():
            raise MemoryMapFileOperationError(
                f"Index file not found: {index_file_path}"
            )

        try:
            self.data_file = self.data_file_path.open("rb")
            self.data_mmap = mmap.mmap(
                self.data_file.fileno(), 0, access=mmap.ACCESS_READ
            )

            self.index_file = self.index_file_path.open("rb")
            self.index_mmap = mmap.mmap(
                self.index_file.fileno(), 0, access=mmap.ACCESS_READ
            )

            index_data = self.index_mmap.read()
            self.index = MemoryMapDatasetIndex.model_validate_json(index_data)

        except OSError as e:
            self._cleanup_resources()
            raise MemoryMapFileOperationError(
                f"Failed to open memory-mapped files: {e}"
            ) from e
        except ValueError as e:
            self._cleanup_resources()
            raise MemoryMapSerializationError(f"Invalid index data: {e}") from e

        # Safety net: closes resources when object is garbage collected if close() wasn't called.
        # weakref.finalize holds a weak ref to self, and the callback receives the resources
        # as args (not self) so cleanup can run even after self is gone.
        self._finalizer = weakref.finalize(
            self,
            self._cleanup_finalizer,
            self.data_mmap,
            self.index_mmap,
            self.data_file,
            self.index_file,
        )

        _logger.debug(
            lambda: f"MemoryMapDatasetClient initialized successfully: data_file={self.data_file_path}, index_file={self.index_file_path}, conversations={len(self.index.conversation_ids)}, size={self.index.total_size} bytes"
        )

    def __enter__(self) -> MemoryMapDatasetClient:
        """Context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Context manager exit with automatic cleanup."""
        self.close()

    _RESOURCE_ATTRS = ("data_mmap", "index_mmap", "data_file", "index_file")

    @staticmethod
    def _cleanup_finalizer(
        data_mmap: mmap.mmap | None,
        index_mmap: mmap.mmap | None,
        data_file: Any | None,
        index_file: Any | None,
    ) -> None:
        """Called by weakref.finalize during GC to close leaked resources."""
        for resource in (data_mmap, index_mmap, data_file, index_file):
            if resource is not None:
                with suppress(Exception):
                    resource.close()
                    _logger.debug("Finalizer cleaned up resource")

    def _cleanup_resources(self) -> None:
        """Close partially opened resources during __init__ error recovery."""
        for attr in self._RESOURCE_ATTRS:
            if (obj := getattr(self, attr, None)) is not None:
                with suppress(Exception):
                    obj.close()

    def _deserialize_conversation(self, data: bytes) -> Conversation:
        """Deserialize a single conversation from bytes.

        Args:
            data: Serialized conversation data bytes (JSON format)

        Returns:
            Conversation object

        Raises:
            MemoryMapSerializationError: If deserialization fails
        """
        try:
            return Conversation.model_validate_json(data)
        except Exception as e:
            raise MemoryMapSerializationError(
                f"Failed to decode conversation data: {e}"
            ) from e

    def get_conversation(self, conversation_id: str) -> Conversation:
        """Get a conversation by ID. O(1) lookup using byte offset index.

        Args:
            conversation_id: Specific conversation ID to retrieve

        Returns:
            Conversation object

        Raises:
            KeyError: If conversation_id is not found
            MemoryMapSerializationError: If conversation data is corrupted or
                the dataset is in payload_bytes format
        """
        if self.index.format == MemoryMapFormat.PAYLOAD_BYTES:
            raise MemoryMapSerializationError(
                f"Cannot retrieve Conversation '{conversation_id}' in payload_bytes "
                "format. Use get_payload_bytes() instead."
            )

        if conversation_id not in self.index.offsets:
            raise KeyError(f"Conversation '{conversation_id}' not found in dataset")

        offset_info = self.index.offsets[conversation_id]

        try:
            self.data_mmap.seek(offset_info.offset)
            conv_bytes = self.data_mmap.read(offset_info.size)

            _logger.debug(
                lambda: f"Loading conversation '{conversation_id}': offset={offset_info.offset}, size={offset_info.size} bytes"
            )

            return self._deserialize_conversation(conv_bytes)

        except (OSError, MemoryMapSerializationError) as e:
            _logger.error(
                f"Failed to load conversation '{conversation_id}' from {self.data_file_path}: {e}"
            )
            raise

    def get_payload_bytes(self, conversation_id: str, turn_index: int) -> bytes | None:
        """Get pre-encoded payload bytes for a specific turn.

        Returns bytes directly from the mmap -- zero deserialization overhead.

        Args:
            conversation_id: Conversation ID
            turn_index: Turn index within the conversation

        Returns:
            Pre-encoded JSON bytes, or None when the dataset is not in
            PAYLOAD_BYTES format or the turn has no payload.
        """
        if self.index.format != MemoryMapFormat.PAYLOAD_BYTES:
            return None
        turn_offsets = self.index.payload_offsets.get(conversation_id)
        if turn_offsets is None or turn_index >= len(turn_offsets):
            return None
        offset_info = turn_offsets[turn_index]
        return bytes(
            self.data_mmap[offset_info.offset : offset_info.offset + offset_info.size]
        )

    def get_payload_turn(
        self, conversation_id: str, turn_index: int
    ) -> PayloadTurnData | None:
        """Get payload bytes plus turn scalars for a specific turn.

        Scalars (``max_tokens``, ``timestamp``) are restored from the index
        when present. When ``max_tokens`` is missing (legacy indexes or turns
        that never set it on the Turn), it is recovered from wire JSON keys
        ``max_tokens`` / ``max_completion_tokens`` / ``max_output_tokens``.

        Args:
            conversation_id: Conversation ID
            turn_index: Turn index within the conversation

        Returns:
            ``PayloadTurnData`` or None when the dataset is not in
            PAYLOAD_BYTES format or the turn has no payload.
        """
        if self.index.format != MemoryMapFormat.PAYLOAD_BYTES:
            return None
        turn_offsets = self.index.payload_offsets.get(conversation_id)
        if turn_offsets is None or turn_index >= len(turn_offsets):
            return None
        offset_info = turn_offsets[turn_index]
        payload_bytes = bytes(
            self.data_mmap[offset_info.offset : offset_info.offset + offset_info.size]
        )
        max_tokens = offset_info.max_tokens
        if max_tokens is None:
            try:
                raw = orjson.loads(payload_bytes)
            except orjson.JSONDecodeError:
                raw = None
            max_tokens = max_tokens_from_wire_payload(
                raw if isinstance(raw, dict) else None
            )
        return PayloadTurnData(
            payload_bytes=payload_bytes,
            max_tokens=max_tokens,
            timestamp=offset_info.timestamp,
        )

    def close(self) -> None:
        """Close the memory-mapped files and associated resources.

        This method is safe to call multiple times.
        """
        for attr_name in self._RESOURCE_ATTRS:
            resource = getattr(self, attr_name, None)
            if resource is not None:
                try:
                    resource.close()
                except Exception as e:
                    _logger.warning(f"Error closing {attr_name}: {e}")
                finally:
                    setattr(self, attr_name, None)
