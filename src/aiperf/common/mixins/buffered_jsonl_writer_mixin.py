# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mixin for buffered JSONL writing with automatic flushing."""

import asyncio
from pathlib import Path
from typing import Generic

import aiofiles
import orjson

from aiperf.common.environment import Environment
from aiperf.common.hooks import on_init, on_stop
from aiperf.common.mixins.aiperf_lifecycle_mixin import AIPerfLifecycleMixin
from aiperf.common.types import BaseModelT
from aiperf.common.utils import yield_to_event_loop


class BufferedJSONLWriterMixin(AIPerfLifecycleMixin, Generic[BaseModelT]):
    """Mixin for buffered JSONL writing with automatic flushing.

    This mixin provides functionality for efficiently writing Pydantic models to JSONL
    files with automatic buffering and flushing. It handles file lifecycle management
    through the AIPerfLifecycleMixin hooks.

    Type Parameters:
        BaseModelT: A Pydantic BaseModel type that will be serialized to JSON

    Attributes:
        output_file: Path to the JSONL output file
        lines_written: Number of lines written
    """

    def __init__(
        self,
        output_file: Path,
        batch_size: int,
        **kwargs,
    ):
        """Initialize the buffered JSONL writer.

        Args:
            output_file: Path to the JSONL output file
            batch_size: Number of records to buffer before auto-flushing
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(**kwargs)
        self.output_file = output_file
        self.lines_written = 0
        self._file_handle = None
        self._file_lock = asyncio.Lock()
        self._buffer: list[bytes] = []  # Store bytes for binary mode
        self._batch_size = batch_size
        self._buffer_lock = asyncio.Lock()

    @on_init
    async def _open_file(self) -> None:
        """Open the file handle for writing in binary mode (called automatically on initialization)."""
        async with self._file_lock:
            # Binary mode for optimal performance with orjson
            self._file_handle = await aiofiles.open(self.output_file, mode="wb")

    async def buffered_write(self, record: BaseModelT) -> None:
        """Write a Pydantic model to the buffer with automatic flushing.

        This method serializes the provided Pydantic model to JSON bytes using orjson
        and adds it to the internal buffer. If the buffer reaches the configured batch
        size, it automatically flushes the buffer to disk.

        Uses binary mode with orjson for optimal performance:
        - 6x faster for large records (>20KB)
        - No encode/decode overhead
        - Efficient for all record sizes

        Args:
            record: A Pydantic BaseModel instance to write
        """
        try:
            # Serialize to bytes using orjson (faster for large records)
            # Use exclude_none=True to omit None fields (smaller output)
            json_bytes = orjson.dumps(record.model_dump(exclude_none=True, mode="json"))

            buffer_to_flush = None
            async with self._buffer_lock:
                self._buffer.append(json_bytes)
                self.lines_written += 1

                # Check if we need to flush
                if len(self._buffer) >= self._batch_size:
                    buffer_to_flush = self._buffer
                    self._buffer = []

            # Flush outside the lock to avoid blocking other writes
            if buffer_to_flush:
                self.execute_async(self._flush_buffer(buffer_to_flush))

        except Exception as e:
            self.error(f"Failed to write record: {e!r}")

    async def _flush_buffer(self, buffer_to_flush: list[bytes]) -> None:
        """Write buffered records to disk using bulk write.

        Uses bulk write strategy: joins all records with newlines and writes
        in a single I/O operation for much better performance.

        Args:
            buffer_to_flush: List of JSON bytes to write
        """
        if not buffer_to_flush:
            return
        async with self._file_lock:
            if self._file_handle is None:
                self.error(
                    f"Tried to flush buffer, but file handle is not open: {self.output_file}"
                )
                return

            try:
                self.debug(lambda: f"Flushing {len(buffer_to_flush)} records to file")
                # Bulk write: join all records and write in one operation
                # This is 9-10x faster than line-by-line writes
                bulk_data = b"\n".join(buffer_to_flush) + b"\n"
                await self._file_handle.write(bulk_data)
                await self._file_handle.flush()
            except Exception as e:
                self.exception(f"Failed to flush buffer: {e!r}")

    @on_stop
    async def _close_file(self) -> None:
        """Flush remaining buffer and close the file handle (called automatically on shutdown)."""
        # Wait for any pending flush tasks to complete
        if self.tasks:
            try:
                await asyncio.wait_for(
                    self.wait_for_tasks(),
                    timeout=Environment.SERVICE.TASK_CANCEL_TIMEOUT_SHORT,
                )
            except asyncio.TimeoutError:
                self.warning(
                    f"Timeout waiting for {len(self.tasks)} pending flush tasks during shutdown. "
                    "Cancelling tasks and proceeding with cleanup."
                )
                # Cancel any remaining tasks to prevent resource leaks
                await self.cancel_all_tasks()
                yield_to_event_loop()

        async with self._buffer_lock:
            buffer_to_flush = self._buffer
            self._buffer = []

        try:
            await self._flush_buffer(buffer_to_flush)
        except Exception as e:
            self.error(f"Failed to flush remaining buffer during shutdown: {e}")

        async with self._file_lock:
            if self._file_handle is not None:
                try:
                    await self._file_handle.close()
                    self.debug(lambda: f"File handle closed: {self.output_file}")
                except Exception as e:
                    self.exception(f"Failed to close file handle during shutdown: {e}")
                finally:
                    self._file_handle = None

        self.debug(
            f"{self.__class__.__name__}: {self.lines_written} JSONL lines written to {self.output_file}"
        )
