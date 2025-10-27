# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
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
            (10, 0),  # No writes
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
