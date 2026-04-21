# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mixin for buffered JSONL writing with automatic flushing."""

import asyncio
import time
from pathlib import Path
from typing import Any, Generic

import aiofiles
import msgspec
import orjson

from aiperf.common.environment import Environment
from aiperf.common.hooks import background_task, on_init, on_stop
from aiperf.common.mixins.aiperf_lifecycle_mixin import AIPerfLifecycleMixin
from aiperf.common.types import BaseModelT
from aiperf.common.utils import yield_to_event_loop

_MSGSPEC_JSON_ENCODER = msgspec.json.Encoder()


class BufferedJSONLWriterMixin(AIPerfLifecycleMixin, Generic[BaseModelT]):
    """Mixin for buffered JSONL writing with automatic flushing.

    Serializes records to JSONL with automatic buffering and flushing, handling
    file lifecycle through the ``AIPerfLifecycleMixin`` hooks. Records can be
    Pydantic models, ``msgspec.Struct`` instances, or any object with a
    ``to_json_bytes()`` method.

    Attributes:
        output_file: Path to the JSONL output file
        lines_written: Number of lines written
    """

    def __init__(
        self,
        output_file: Path,
        batch_size: int,
        flush_interval: float,
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
        self._flush_interval = flush_interval
        self._last_flush_monotonic = time.monotonic()

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

    async def buffered_write(self, record: BaseModelT) -> None:
        """Write a record to the buffer with automatic flushing."""
        try:
            json_bytes = self._serialize_record(record)

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

    def _serialize_record(self, record: Any) -> bytes:
        # Check to_json_bytes first: some msgspec.Struct records (e.g. the
        # RawRecordInfo / MetricRecordInfo families in record_models.py) carry
        # a specialized encoder hook via to_json_bytes() that the generic
        # msgspec.json encoder doesn't know about.
        if hasattr(record, "to_json_bytes"):
            return record.to_json_bytes()
        if isinstance(record, msgspec.Struct):
            return _MSGSPEC_JSON_ENCODER.encode(record)
        if hasattr(record, "model_dump"):
            return orjson.dumps(record.model_dump(exclude_none=True, mode="json"))
        raise TypeError(f"Unsupported JSONL record type: {type(record)}")

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
                self._last_flush_monotonic = time.monotonic()
            except Exception as e:
                self.exception(f"Failed to flush buffer: {e!r}")

    @background_task(interval=lambda self: self._flush_interval, immediate=False)
    async def _flush_buffer_periodically(self) -> None:
        """Flush buffered records on a time boundary even at low throughput."""
        if not self._buffer:
            return

        buffer_to_flush = self._buffer
        self._buffer = []
        await self._flush_buffer(buffer_to_flush)

    @on_stop
    async def _close_file(self) -> None:
        """Flush remaining buffer and close the file handle (called automatically on shutdown)."""
        # wait_for_tasks() snapshots self.tasks at entry, so any flush task
        # created AFTER entry — by a late buffered_write whose upstream pull
        # hasn't stopped yet — is not awaited. Its records sit in the buffer
        # and the task then hits a closed file handle. Drain in a loop until
        # both self.tasks and self._buffer are stable or we hit the cap.
        for _ in range(3):
            if not self.tasks and not self._buffer:
                break
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
                    break

            buffer_to_flush = self._buffer
            self._buffer = []
            if buffer_to_flush:
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
