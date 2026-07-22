# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextlib
import json
import tempfile
from pathlib import Path

import pytest
from pydantic import BaseModel

from aiperf.common.mixins.buffered_jsonl_writer_mixin import BufferedJSONLWriterMixin


class SampleRecord(BaseModel):
    """Sample Pydantic model for testing."""

    id: int
    value: str


class TestBufferedJSONLWriterMixin:
    """Test suite for BufferedJSONLWriterMixin file locking functionality."""

    @pytest.fixture
    def temp_output_file(self):
        """Create a temporary output file for testing."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            temp_path = Path(f.name)
        yield temp_path
        temp_path.unlink(missing_ok=True)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "batch_size,num_tasks,records_per_task",
        [
            (10, 5, 20),  # Standard batching
            (1, 10, 10),  # Frequent flushes
            (100, 3, 50),  # Large batches
        ],
    )
    async def test_concurrent_writes_preserve_data_integrity(
        self, temp_output_file, batch_size, num_tasks, records_per_task
    ):
        """Test that file locking ensures data integrity during concurrent writes."""
        writer = BufferedJSONLWriterMixin[SampleRecord](
            output_file=temp_output_file,
            batch_size=batch_size,
        )
        await writer.initialize()
        await writer.start()

        async def write_records(task_id: int):
            for i in range(records_per_task):
                await writer.buffered_write(
                    SampleRecord(id=task_id * 1000 + i, value=f"task_{task_id}_{i}")
                )

        await asyncio.gather(*[write_records(tid) for tid in range(num_tasks)])
        await writer.stop()

        expected_total = num_tasks * records_per_task
        assert writer.lines_written == expected_total

        with open(temp_output_file) as f:
            lines = [line.strip() for line in f.readlines()]
            assert len(lines) == expected_total
            for line in lines:
                assert "id" in json.loads(line)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "batch_size,num_records",
        [
            (100, 25),  # Buffer not full at stop
            (5, 50),  # Multiple flushes then remainder
        ],
    )
    async def test_buffer_flush_and_cleanup_edge_cases(
        self, temp_output_file, batch_size, num_records
    ):
        """Test that file locking handles buffer flush and cleanup correctly."""
        writer = BufferedJSONLWriterMixin[SampleRecord](
            output_file=temp_output_file,
            batch_size=batch_size,
        )
        await writer.initialize()
        await writer.start()

        for i in range(num_records):
            await writer.buffered_write(SampleRecord(id=i, value=f"record_{i}"))

        await writer.stop()

        assert writer.lines_written == num_records
        assert writer._file_handle is None

        with open(temp_output_file) as f:
            lines = f.readlines()
            assert len(lines) == num_records

    @pytest.mark.asyncio
    async def test_empty_file_deleted_on_stop(self, temp_output_file):
        """Test that output file is deleted when no records are written."""
        writer = BufferedJSONLWriterMixin[SampleRecord](
            output_file=temp_output_file,
            batch_size=10,
        )
        await writer.initialize()
        await writer.start()

        # Don't write anything
        await writer.stop()

        assert writer.lines_written == 0
        assert writer._file_handle is None
        assert not temp_output_file.exists(), "Empty file should be deleted"

    @pytest.mark.asyncio
    async def test_file_preserved_when_records_written(self, temp_output_file):
        """Test that output file is preserved when records are written."""
        writer = BufferedJSONLWriterMixin[SampleRecord](
            output_file=temp_output_file,
            batch_size=10,
        )
        await writer.initialize()
        await writer.start()

        await writer.buffered_write(SampleRecord(id=1, value="test"))
        await writer.stop()

        assert writer.lines_written == 1
        assert temp_output_file.exists(), "File with content should be preserved"

    @pytest.mark.asyncio
    async def test_periodic_flush_loop_survives_unexpected_error(
        self, temp_output_file
    ):
        """A non-cancel error in one periodic-flush iteration must not kill the loop.

        Mirrors ``_background_task_loop`` semantics: the error is logged and the
        loop keeps draining the buffer on subsequent intervals, rather than the
        whole task dying for the rest of the run after one transient failure.

        Drives ``_flush_buffer_periodically`` directly (instead of relying on the
        auto-started background task's scheduling) so the contract is exercised
        deterministically: a flush that raises, followed by one that succeeds.
        """
        writer = BufferedJSONLWriterMixin[SampleRecord](
            output_file=temp_output_file,
            batch_size=1000,  # never auto-flush; only the periodic loop drains
            flush_interval=0.0,  # sleep(0) per iteration: pure event-loop yield
        )
        await writer.initialize()

        # The first periodic flush raises; later ones succeed. The events let us
        # advance the loop one observable step at a time without depending on
        # wall-clock timing (asyncio.sleep is patched to a no-op in unit tests).
        real_flush = writer._flush_buffer
        flush_attempts: list[int] = []
        first_flush_failed = asyncio.Event()
        recovered = asyncio.Event()

        async def flaky_flush(buffer_to_flush: list[bytes]) -> None:
            flush_attempts.append(len(buffer_to_flush))
            if len(flush_attempts) == 1:
                first_flush_failed.set()
                raise RuntimeError("transient flush failure")
            await real_flush(buffer_to_flush)
            recovered.set()

        writer._flush_buffer = flaky_flush

        async def yield_until(event: asyncio.Event) -> None:
            # asyncio.sleep is patched to a no-op in unit tests, so yield the
            # event loop (bounded) until the periodic task sets the event or
            # unexpectedly dies.
            for _ in range(10_000):
                if event.is_set() or loop_task.done():
                    return
                await asyncio.sleep(0)

        loop_task = asyncio.create_task(writer._flush_buffer_periodically())
        try:
            # First record drives the failing iteration; the loop must survive it.
            writer._buffer.append(b'{"id": 1, "value": "boom"}')
            await yield_until(first_flush_failed)
            assert first_flush_failed.is_set(), "failing flush iteration never ran"
            assert not loop_task.done(), "loop should survive the unexpected error"

            # A record written after the failure must still be drained by the
            # still-alive loop on a subsequent iteration.
            writer._buffer.append(b'{"id": 2, "value": "ok"}')
            await yield_until(recovered)
            assert recovered.is_set(), "loop did not resume draining after the error"
            assert len(flush_attempts) >= 2, "loop did not flush again after error"
            assert not writer._buffer, "later record was not flushed"
            assert not loop_task.done()
        finally:
            loop_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await loop_task
            writer._flush_buffer = real_flush
            await writer.stop()

    @pytest.mark.asyncio
    async def test_close_waits_only_for_flush_tasks(self, temp_output_file):
        writer = BufferedJSONLWriterMixin[SampleRecord](
            output_file=temp_output_file,
            batch_size=10,
        )
        await writer.initialize()
        unrelated_task = writer.execute_async(asyncio.Event().wait())

        await asyncio.wait_for(writer._close_file(), timeout=0.1)

        assert not unrelated_task.done()
        await writer.cancel_all_tasks()
        await asyncio.gather(unrelated_task, return_exceptions=True)

    @pytest.mark.asyncio
    async def test_close_file_closes_handle_when_cancelled_mid_flush(
        self, temp_output_file
    ):
        """Cancel during the final flush must still close the file handle.

        ``asyncio.shield`` keeps the flush running, but ``CancelledError`` still
        exits the outer ``_close_file`` await. Cleanup must finish flush+close
        before returning so the handle is not leaked and drained records are
        not dropped by closing under an in-flight write.
        """
        writer = BufferedJSONLWriterMixin[SampleRecord](
            output_file=temp_output_file,
            batch_size=10,
        )
        await writer.initialize()
        await writer.buffered_write(SampleRecord(id=1, value="pending"))

        real_flush = writer._flush_buffer
        flush_started = asyncio.Event()
        flush_continue = asyncio.Event()

        async def blocked_flush(buffer_to_flush: list[bytes]) -> None:
            flush_started.set()
            await flush_continue.wait()
            await real_flush(buffer_to_flush)

        writer._flush_buffer = blocked_flush

        close_task = asyncio.create_task(writer._close_file())
        for _ in range(10_000):
            if flush_started.is_set() or close_task.done():
                break
            await asyncio.sleep(0)
        assert flush_started.is_set(), "final flush never started"

        close_task.cancel()
        # Let the cancel land on the shield await before unblocking the flush.
        await asyncio.sleep(0)
        flush_continue.set()
        with contextlib.suppress(asyncio.CancelledError):
            await close_task

        assert writer._file_handle is None
        with open(temp_output_file) as f:
            lines = [line.strip() for line in f if line.strip()]
        assert len(lines) == 1
        assert json.loads(lines[0])["id"] == 1

    @pytest.mark.asyncio
    async def test_close_file_drains_in_flight_periodic_flush(self, temp_output_file):
        """Shutdown must not drop a periodic flush that outlives the loop task.

        Cooperative stop lets the current iteration finish; if the loop is
        hard-cancelled mid-``shield``, ``_periodic_flush_in_flight`` is still
        drained before the handle closes so the detached batch cannot race
        ``_final_flush_and_close`` and be silently dropped.
        """
        writer = BufferedJSONLWriterMixin[SampleRecord](
            output_file=temp_output_file,
            batch_size=1000,  # never auto-flush; only the periodic loop drains
            flush_interval=0.0,
        )
        await writer.initialize()
        await writer.start()

        real_flush = writer._flush_buffer
        flush_started = asyncio.Event()
        flush_continue = asyncio.Event()

        async def blocked_flush(buffer_to_flush: list[bytes]) -> None:
            # Yield before the real lock+write so an orphaned shielded flush
            # can lose a close race unless _close_file drains it explicitly.
            flush_started.set()
            await flush_continue.wait()
            await real_flush(buffer_to_flush)

        writer._flush_buffer = blocked_flush
        await writer.buffered_write(SampleRecord(id=42, value="periodic"))

        for _ in range(10_000):
            if flush_started.is_set():
                break
            await asyncio.sleep(0)
        assert flush_started.is_set(), "periodic flush never started"
        assert writer._periodic_flush_in_flight is not None

        close_task = asyncio.create_task(writer._close_file())
        # Close must block on the in-flight periodic write, not finish early
        # and close the handle out from under it.
        for _ in range(100):
            if close_task.done():
                break
            await asyncio.sleep(0)
        assert not close_task.done(), "close returned before in-flight periodic flush"

        flush_continue.set()
        await close_task

        assert writer._file_handle is None
        assert writer._periodic_flush_in_flight is None
        with open(temp_output_file) as f:
            lines = [line.strip() for line in f if line.strip()]
        assert len(lines) == 1
        assert json.loads(lines[0])["id"] == 42

    @pytest.mark.asyncio
    async def test_stop_periodic_flush_drains_after_hard_cancel(self, temp_output_file):
        """Hard-cancel fallback must still await the shielded in-flight write.

        Reproduces the mid-shield cancel window: the loop task returns as soon
        as CancelledError lands on the shield await, leaving the inner write
        orphaned. Draining ``_periodic_flush_in_flight`` before close is what
        prevents the drop.
        """
        writer = BufferedJSONLWriterMixin[SampleRecord](
            output_file=temp_output_file,
            batch_size=1000,
            flush_interval=0.0,
        )
        await writer.initialize()
        # Drive the loop directly so we can hard-cancel without the cooperative
        # stop path masking the orphan.
        real_flush = writer._flush_buffer
        flush_started = asyncio.Event()
        flush_continue = asyncio.Event()

        async def blocked_flush(buffer_to_flush: list[bytes]) -> None:
            flush_started.set()
            await flush_continue.wait()
            await real_flush(buffer_to_flush)

        writer._flush_buffer = blocked_flush
        writer._buffer.append(b'{"id": 7, "value": "orphan"}')
        loop_task = asyncio.create_task(writer._flush_buffer_periodically())
        writer._periodic_flush_task = loop_task

        for _ in range(10_000):
            if flush_started.is_set() or loop_task.done():
                break
            await asyncio.sleep(0)
        assert flush_started.is_set(), "periodic flush never started"
        in_flight = writer._periodic_flush_in_flight
        assert in_flight is not None and not in_flight.done()

        # Hard-cancel mid-shield: loop returns, shielded write stays orphaned.
        loop_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await loop_task
        assert not in_flight.done(), "shielded flush should still be running"
        assert writer._periodic_flush_in_flight is in_flight

        # Unblock the orphan while stop drains it (mirrors a real write
        # completing after the loop task has already exited).
        flush_continue.set()
        await writer._stop_periodic_flush()

        assert in_flight.done()
        assert writer._periodic_flush_in_flight is None
        assert writer._file_handle is not None

        # Close with an empty remaining buffer; the orphaned batch must already
        # be on disk from the drained in-flight write.
        await writer._await_shielded_cleanup(
            asyncio.create_task(writer._final_flush_and_close([]))
        )

        assert writer._file_handle is None
        with open(temp_output_file) as f:
            lines = [line.strip() for line in f if line.strip()]
        assert len(lines) == 1
        assert json.loads(lines[0])["id"] == 7
