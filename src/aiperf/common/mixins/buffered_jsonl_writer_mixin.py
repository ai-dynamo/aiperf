# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mixin for buffered JSONL writing with automatic flushing."""

import asyncio
from pathlib import Path
from typing import Any, Generic, TypeVar

import aiofiles
import msgspec
import orjson

from aiperf.common.environment import Environment
from aiperf.common.finite import scrub_non_finite
from aiperf.common.hooks import on_init, on_stop
from aiperf.common.mixins.aiperf_lifecycle_mixin import AIPerfLifecycleMixin
from aiperf.common.utils import yield_to_event_loop

# Pydantic ``BaseModel`` and ``msgspec.Struct`` both flow through this mixin.
# The generic is unconstrained so concrete subclasses can pin the precise
# record type for their own type-checker readers.
JsonlRecordT = TypeVar("JsonlRecordT", bound=Any)


class BufferedJSONLWriterMixin(AIPerfLifecycleMixin, Generic[JsonlRecordT]):
    """Mixin for buffered JSONL writing with automatic flushing.

    This mixin provides functionality for efficiently writing Pydantic models to JSONL
    files with automatic buffering and flushing. It handles file lifecycle management
    through the AIPerfLifecycleMixin hooks.

    Type Parameters:
        JsonlRecordT: The record type written to JSONL — a Pydantic
            ``BaseModel`` or ``msgspec.Struct``. The mixin dispatches on the
            concrete type at write time.

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

    @on_init
    async def _open_file(self) -> None:
        """Open the file handle for writing in binary mode (called automatically on initialization)."""

        try:
            # Create the output file directory if it doesn't exist and clear the file
            self.output_file.parent.mkdir(parents=True, exist_ok=True)
            self.output_file.unlink(missing_ok=True)
        except Exception as e:
            self.exception(
                f"Failed to create output file directory or clear file: {self.output_file}: {e!r}"
            )
            raise

        async with self._file_lock:
            # Binary mode for optimal performance with orjson
            self._file_handle = await aiofiles.open(self.output_file, mode="wb")

    async def buffered_write(self, record: JsonlRecordT) -> None:
        """Write a record to the buffer with automatic flushing.

        Serializes the record to JSON bytes and adds it to the internal
        buffer. If the buffer reaches the configured batch size, it
        automatically flushes the buffer to disk. Dispatches by record
        family: Pydantic ``BaseModel`` uses
        ``model_dump(mode="json", exclude_none=True)`` then ``orjson.dumps``;
        ``msgspec.Struct`` encodes via ``msgspec.json.encode`` so
        ``omit_defaults=True`` applies on the Struct.

        Uses binary mode with orjson for optimal performance:
        - 6x faster for large records (>20KB)
        - No encode/decode overhead
        - Efficient for all record sizes

        Args:
            record: A Pydantic ``BaseModel`` or ``msgspec.Struct`` instance.
        """
        try:
            if isinstance(record, msgspec.Struct):
                # msgspec.json.encode honors the Struct's ``omit_defaults``
                # config, so None-defaulted fields drop from the wire shape.
                # scrub_non_finite already enforces "null on disk = absent",
                # but a Struct-level NaN/inf would need to be scrubbed after a
                # decode round-trip; for the current consumers (RawRecordInfo
                # numeric fields are integer-typed) this is a non-issue.
                json_bytes = msgspec.json.encode(record)
            else:
                # ``exclude_none=True`` drops None fields for parity with the
                # msgspec.Struct ``omit_defaults`` shape.
                json_bytes = orjson.dumps(
                    scrub_non_finite(record.model_dump(exclude_none=True, mode="json"))
                )

            buffer_to_flush = None
            self._buffer.append(json_bytes)
            self.lines_written += 1

            # Check if we need to flush
            if len(self._buffer) >= self._batch_size:
                buffer_to_flush = self._buffer
                self._buffer = []

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
                await yield_to_event_loop()

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

        if self.lines_written == 0:
            self.debug(f"No lines written, deleting output file: {self.output_file}")
            self.output_file.unlink(missing_ok=True)
