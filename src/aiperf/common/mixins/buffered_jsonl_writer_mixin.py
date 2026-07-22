# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mixin for buffered JSONL writing with automatic flushing."""

import asyncio
import contextlib
import time
from pathlib import Path
from typing import ClassVar, Generic

import aiofiles
import orjson

from aiperf.common.environment import Environment
from aiperf.common.finite import scrub_non_finite
from aiperf.common.hooks import on_init, on_start, on_stop
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

    # Field names to exclude from each serialized JSONL line. Subclasses that
    # write a model carrying a wire-only field (e.g. a RecordData ``record_type``
    # discriminator needed for ZMQ reconstruction but not wanted on disk) set this
    # to keep the on-disk output identical to before that field was added.
    _jsonl_exclude_fields: ClassVar[set[str] | None] = None

    def __init__(
        self,
        output_file: Path,
        batch_size: int,
        flush_interval: float = Environment.METRICS.EXPORT_FLUSH_INTERVAL,
        **kwargs,
    ) -> None:
        """Initialize the buffered JSONL writer.

        Args:
            output_file: Path to the JSONL output file
            batch_size: Number of records to buffer before auto-flushing
            flush_interval: Periodic flush interval (seconds) for the background
                task that drains the in-memory buffer at low throughput. Default
                is ``Environment.METRICS.EXPORT_FLUSH_INTERVAL`` so operators can
                bound worst-case freshness without code changes.
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(**kwargs)
        self.output_file = output_file
        self.lines_written = 0
        self._file_handle = None
        self._file_lock = asyncio.Lock()
        self._buffer: list[bytes] = []  # Store bytes for binary mode
        # Per-batch flush tasks only. Tracked separately from ``self.tasks`` so
        # ``_close_file`` drains exactly the pending flushes without waiting for
        # (or cancelling) unrelated tasks the subclass scheduled via execute_async.
        self._flush_tasks: set[asyncio.Task] = set()
        self._batch_size = batch_size
        self._flush_interval = flush_interval
        self._last_flush_monotonic = time.monotonic()
        # Self-managed periodic-flush task. Deliberately NOT registered via
        # @background_task / execute_async (which would add it to ``self.tasks``):
        # callers drain transient writes with ``wait_for_tasks()``, and a
        # perpetual loop in that set would make ``wait_for_tasks()`` block
        # forever. We start it in ``_start_periodic_flush`` and cooperatively
        # stop it in ``_close_file`` (signal, then await) so an in-flight
        # iteration can finish before the final flush+close.
        self._periodic_flush_task: asyncio.Task | None = None
        self._periodic_flush_stop = asyncio.Event()
        # In-flight write from the periodic loop. Tracked separately from
        # ``_flush_tasks`` so ``_close_file`` can drain it after a hard cancel
        # without exposing it to the timeout-cancel branch that would defeat
        # ``asyncio.shield``.
        self._periodic_flush_in_flight: asyncio.Task | None = None

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
            # scrub_non_finite enforces "null on disk = absent" across the
            # JSONL so per-record NaN/inf doesn't masquerade as missing.
            json_bytes = orjson.dumps(
                scrub_non_finite(
                    record.model_dump(
                        exclude_none=True,
                        mode="json",
                        exclude=self._jsonl_exclude_fields,
                    )
                )
            )

            buffer_to_flush = None
            self._buffer.append(json_bytes)
            self.lines_written += 1

            # Check if we need to flush
            if len(self._buffer) >= self._batch_size:
                buffer_to_flush = self._buffer
                self._buffer = []

            if buffer_to_flush:
                task = self.execute_async(self._flush_buffer(buffer_to_flush))
                self._flush_tasks.add(task)
                task.add_done_callback(self._flush_tasks.discard)

        except Exception as e:
            self.error(f"Failed to write record: {e!r}")

    async def flush_buffer(self) -> None:
        """Flush the current internal buffer to disk.

        Public counterpart to ``_flush_buffer``: swaps out the live buffer and
        writes all pending records. Safe to call when the buffer is empty.
        """
        buffer_to_flush = self._buffer
        self._buffer = []
        # Shield so a cancel between detaching the buffer and the write
        # completing can't silently drop the records we already drained.
        await asyncio.shield(self._flush_buffer(buffer_to_flush))

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

    @on_start
    async def _start_periodic_flush(self) -> None:
        """Start the self-managed periodic-flush loop on service start."""
        if self._periodic_flush_task is None or self._periodic_flush_task.done():
            self._periodic_flush_stop.clear()
            self._periodic_flush_task = asyncio.create_task(
                self._flush_buffer_periodically()
            )

    async def _flush_buffer_periodically(self) -> None:
        """Flush buffered records on a time boundary even at low throughput.

        Bounds worst-case freshness of the JSONL file when the in-memory batch
        never reaches ``batch_size`` (e.g. very low arrival rate). The interval
        is the per-instance ``flush_interval`` set in ``__init__``. Runs until
        ``_close_file`` sets ``_periodic_flush_stop`` (cooperative shutdown) or
        the task is hard-cancelled as a last resort.

        Self-managed (not a ``@background_task``) so it never lands in
        ``self.tasks`` and never blocks ``wait_for_tasks()``, which callers use
        to drain only the transient per-batch flush tasks.

        Resilience mirrors ``_background_task_loop``: a non-cancellation error
        in one iteration is logged and the loop continues draining on the next
        interval, so a transient failure never permanently stops periodic
        flushing for the rest of the run.
        """
        while not self._periodic_flush_stop.is_set():
            try:
                # Sleep until the next interval, or exit early when shutdown
                # signals stop. Checking the event only between iterations
                # lets an in-flight flush finish before the loop returns.
                try:
                    await asyncio.wait_for(
                        self._periodic_flush_stop.wait(),
                        timeout=self._flush_interval,
                    )
                    return
                except TimeoutError:
                    pass

                if not self._buffer:
                    continue
                buffer_to_flush = self._buffer
                self._buffer = []
                # Track + shield so a hard cancel (timeout fallback in
                # _close_file, or lifecycle cancellation) cannot drop the
                # already-detached batch: the inner task keeps running and
                # _close_file awaits ``_periodic_flush_in_flight`` before
                # closing the handle.
                flush_task = asyncio.create_task(self._flush_buffer(buffer_to_flush))
                self._periodic_flush_in_flight = flush_task
                try:
                    await asyncio.shield(flush_task)
                finally:
                    if (
                        flush_task.done()
                        and self._periodic_flush_in_flight is flush_task
                    ):
                        self._periodic_flush_in_flight = None
            except asyncio.CancelledError:
                return
            except Exception as e:
                self.exception(f"Error in periodic flush loop: {e!r}")
                # Give some time to recover, just in case.
                await asyncio.sleep(0.001)

    async def _stop_periodic_flush(self) -> None:
        """Cooperatively stop the periodic loop, then drain any in-flight write.

        Prefers signaling ``_periodic_flush_stop`` so the loop finishes its
        current iteration instead of hard-cancelling mid-``shield``. Hard
        cancel is only the timeout fallback; in that case
        ``_periodic_flush_in_flight`` is awaited so an orphaned shielded write
        cannot race ``_final_flush_and_close`` and drop its chunk.
        """
        if self._periodic_flush_task is None:
            return

        self._periodic_flush_stop.set()
        try:
            await asyncio.wait_for(
                self._periodic_flush_task,
                timeout=Environment.SERVICE.TASK_CANCEL_TIMEOUT_SHORT,
            )
        except TimeoutError:
            self.warning(
                "Timeout waiting for periodic flush loop to stop during shutdown. "
                "Cancelling the loop and draining any in-flight flush."
            )
            self._periodic_flush_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._periodic_flush_task
        self._periodic_flush_task = None

        if (in_flight := self._periodic_flush_in_flight) is not None:
            # Never cancel: that would defeat the shield and drop the batch.
            await in_flight
            if self._periodic_flush_in_flight is in_flight:
                self._periodic_flush_in_flight = None

    async def _final_flush_and_close(self, buffer_to_flush: list[bytes]) -> None:
        """Flush detached records then close the file handle.

        Kept as one unit so a cancel cannot close the handle out from under an
        in-flight write (which would drop already-detached records), and so the
        handle is always closed even when the outer wait is cancelled.
        """
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

    async def _await_shielded_cleanup(self, cleanup_task: asyncio.Task) -> None:
        """Await a shielded cleanup task through outer cancellation.

        If we are cancelled while waiting on the shield, the inner task keeps
        running. Drain our cancellation so we can await it to completion before
        leaving — otherwise the caller would return with the handle still open
        (or a naive finally-close would race the still-running flush).
        """
        current = asyncio.current_task()
        try:
            await asyncio.shield(cleanup_task)
        finally:
            if current is not None:
                while current.cancelling():
                    current.uncancel()
            if not cleanup_task.done():
                await cleanup_task

    @on_stop
    async def _close_file(self) -> None:
        """Flush remaining buffer and close the file handle (called automatically on shutdown)."""
        # Stop the self-managed periodic-flush loop first so it can't race the
        # final flush or keep the buffer churning during teardown. Cooperative
        # stop lets a mid-flush iteration finish; any orphaned shielded write
        # from a hard-cancel fallback is drained before we touch the handle.
        await self._stop_periodic_flush()

        # Wait for any pending flush tasks to complete. Drain only the flush
        # tasks (not all of self.tasks) so unrelated subclass tasks are neither
        # waited on nor cancelled here.
        if self._flush_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*list(self._flush_tasks)),
                    timeout=Environment.SERVICE.TASK_CANCEL_TIMEOUT_SHORT,
                )
            except TimeoutError:
                self.warning(
                    f"Timeout waiting for {len(self._flush_tasks)} pending flush tasks during shutdown. "
                    "Cancelling tasks and proceeding with cleanup."
                )
                for task in self._flush_tasks:
                    task.cancel()
                await yield_to_event_loop()

        buffer_to_flush = self._buffer
        self._buffer = []
        # Shield so a shutdown-path cancel can't interrupt the write and drop
        # records already detached from self._buffer.
        await self._await_shielded_cleanup(
            asyncio.create_task(self._final_flush_and_close(buffer_to_flush))
        )

        self.debug(
            f"{self.__class__.__name__}: {self.lines_written} JSONL lines written to {self.output_file}"
        )

        if self.lines_written == 0:
            self.debug(f"No lines written, deleting output file: {self.output_file}")
            self.output_file.unlink(missing_ok=True)
