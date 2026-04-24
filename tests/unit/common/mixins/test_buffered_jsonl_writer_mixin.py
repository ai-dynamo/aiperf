# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import tempfile
from pathlib import Path

import msgspec
import pytest

from aiperf.common.mixins.buffered_jsonl_writer_mixin import BufferedJSONLWriterMixin


class SampleRecord(msgspec.Struct, frozen=True, kw_only=True):
    """Sample msgspec model for testing."""

    id: int
    """Unique record identifier."""

    value: str
    """Record payload value."""

    def to_json_bytes(self) -> bytes:
        return msgspec.json.encode(self)


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
            flush_interval=1.0,
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
            flush_interval=1.0,
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
            flush_interval=1.0,
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
            flush_interval=1.0,
        )
        await writer.initialize()
        await writer.start()

        await writer.buffered_write(SampleRecord(id=1, value="test"))
        await writer.stop()

        assert writer.lines_written == 1
        assert temp_output_file.exists(), "File with content should be preserved"

    @pytest.mark.asyncio
    async def test_late_buffered_write_during_close_is_drained(self, temp_output_file):
        """P1 regression: a buffered_write that arrives while _close_file is
        already awaiting wait_for_tasks schedules a new flush task AFTER
        the wait's self.tasks snapshot. Without the drain loop the new
        task runs on a closed file and its record is lost.
        """
        writer = BufferedJSONLWriterMixin[SampleRecord](
            output_file=temp_output_file,
            batch_size=1,  # every write schedules a flush task
            flush_interval=1000.0,  # disable periodic flush
        )
        await writer.initialize()
        await writer.start()

        original_flush = writer._flush_buffer
        first_flush_started = asyncio.Event()
        release_first_flush = asyncio.Event()

        async def slow_first_flush(buffer_to_flush):
            """Hold the first flush open so stop() parks in wait_for_tasks."""
            first_flush_started.set()
            await release_first_flush.wait()
            await original_flush(buffer_to_flush)

        writer._flush_buffer = slow_first_flush  # type: ignore[assignment]

        # Schedule flush task #1 and wait until it's parked.
        await writer.buffered_write(SampleRecord(id=1, value="first"))
        await first_flush_started.wait()

        # Swap back to real flush so the late write can actually land.
        writer._flush_buffer = original_flush  # type: ignore[assignment]

        # Start stop() in the background. It enters _close_file and calls
        # wait_for_tasks, which snapshots self.tasks = {task_1} and awaits.
        stop_task = asyncio.create_task(writer.stop())
        for _ in range(10):
            await asyncio.sleep(0)

        # Late write: lands AFTER wait_for_tasks took its snapshot. Without
        # the drain loop, task_2 is spawned but never awaited.
        await writer.buffered_write(SampleRecord(id=2, value="late"))

        # Unblock the first flush so wait_for_tasks can return.
        release_first_flush.set()
        await stop_task

        # Both records must be persisted with the drain loop in place.
        with open(temp_output_file) as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        ids = sorted(json.loads(line)["id"] for line in lines)
        assert ids == [1, 2], (
            f"Expected both records persisted, got ids={ids}. "
            "The drain loop in _close_file must catch flush tasks "
            "scheduled after wait_for_tasks' snapshot."
        )
