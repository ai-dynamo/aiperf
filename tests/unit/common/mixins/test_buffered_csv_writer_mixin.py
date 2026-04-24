# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import csv
import io
import tempfile
from pathlib import Path

import pytest

from aiperf.common.mixins.buffered_csv_writer_mixin import BufferedCSVWriterMixin


class TestBufferedCSVWriterMixin:
    """Test suite for BufferedCSVWriterMixin."""

    @pytest.fixture
    def temp_output_file(self):
        """Create a temporary output file for testing."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            temp_path = Path(f.name)
        yield temp_path
        temp_path.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_basic_write_with_header(self, temp_output_file):
        """Test that header and rows are written correctly."""
        writer = BufferedCSVWriterMixin(
            output_file=temp_output_file,
            batch_size=10,
        )
        writer.set_csv_columns(["id", "name", "value"])
        await writer.initialize()
        await writer.start()

        await writer.buffered_csv_write(["1", "alice", "100"])
        await writer.buffered_csv_write(["2", "bob", "200"])
        await writer.stop()

        assert writer.rows_written == 2
        content = temp_output_file.read_text()
        reader = csv.reader(io.StringIO(content))
        rows = list(reader)
        assert rows[0] == ["id", "name", "value"]
        assert rows[1] == ["1", "alice", "100"]
        assert rows[2] == ["2", "bob", "200"]
        assert len(rows) == 3

    @pytest.mark.asyncio
    async def test_csv_escaping(self, temp_output_file):
        """Test that values with commas, quotes, and newlines are properly escaped."""
        writer = BufferedCSVWriterMixin(
            output_file=temp_output_file,
            batch_size=10,
        )
        writer.set_csv_columns(["id", "message"])
        await writer.initialize()
        await writer.start()

        await writer.buffered_csv_write(["1", 'has "quotes"'])
        await writer.buffered_csv_write(["2", "has,comma"])
        await writer.buffered_csv_write(["3", "has\nnewline"])
        await writer.stop()

        content = temp_output_file.read_text()
        reader = csv.reader(io.StringIO(content))
        rows = list(reader)
        assert rows[1] == ["1", 'has "quotes"']
        assert rows[2] == ["2", "has,comma"]
        assert rows[3] == ["3", "has\nnewline"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "batch_size,num_tasks,records_per_task",
        [
            (10, 5, 20),
            (1, 10, 10),
            (100, 3, 50),
        ],
    )
    async def test_concurrent_writes_preserve_data_integrity(
        self, temp_output_file, batch_size, num_tasks, records_per_task
    ):
        """Test that file locking ensures data integrity during concurrent writes."""
        writer = BufferedCSVWriterMixin(
            output_file=temp_output_file,
            batch_size=batch_size,
        )
        writer.set_csv_columns(["task_id", "record_id"])
        await writer.initialize()
        await writer.start()

        async def write_records(task_id: int):
            for i in range(records_per_task):
                await writer.buffered_csv_write([str(task_id), str(i)])

        await asyncio.gather(*[write_records(tid) for tid in range(num_tasks)])
        await writer.stop()

        expected_total = num_tasks * records_per_task
        assert writer.rows_written == expected_total

        content = temp_output_file.read_text()
        reader = csv.reader(io.StringIO(content))
        rows = list(reader)
        assert rows[0] == ["task_id", "record_id"]
        assert len(rows) == expected_total + 1  # +1 for header

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "batch_size,num_records",
        [
            (100, 25),
            (5, 50),
        ],
    )
    async def test_buffer_flush_and_cleanup_edge_cases(
        self, temp_output_file, batch_size, num_records
    ):
        """Test that file locking handles buffer flush and cleanup correctly."""
        writer = BufferedCSVWriterMixin(
            output_file=temp_output_file,
            batch_size=batch_size,
        )
        writer.set_csv_columns(["id", "value"])
        await writer.initialize()
        await writer.start()

        for i in range(num_records):
            await writer.buffered_csv_write([str(i), f"record_{i}"])

        await writer.stop()

        assert writer.rows_written == num_records
        assert writer._csv_file_handle is None

        content = temp_output_file.read_text()
        reader = csv.reader(io.StringIO(content))
        rows = list(reader)
        assert len(rows) == num_records + 1  # +1 for header

    @pytest.mark.asyncio
    async def test_empty_file_deleted_on_stop(self, temp_output_file):
        """Test that output file is deleted when no records are written."""
        writer = BufferedCSVWriterMixin(
            output_file=temp_output_file,
            batch_size=10,
        )
        writer.set_csv_columns(["id"])
        await writer.initialize()
        await writer.start()
        await writer.stop()

        assert writer.rows_written == 0
        assert writer._csv_file_handle is None
        assert not temp_output_file.exists(), "Empty file should be deleted"

    @pytest.mark.asyncio
    async def test_file_preserved_when_records_written(self, temp_output_file):
        """Test that output file is preserved when records are written."""
        writer = BufferedCSVWriterMixin(
            output_file=temp_output_file,
            batch_size=10,
        )
        writer.set_csv_columns(["id"])
        await writer.initialize()
        await writer.start()

        await writer.buffered_csv_write(["1"])
        await writer.stop()

        assert writer.rows_written == 1
        assert temp_output_file.exists(), "File with content should be preserved"

    @pytest.mark.asyncio
    async def test_header_written_only_once(self, temp_output_file):
        """Test that the header is written exactly once even across multiple flushes."""
        writer = BufferedCSVWriterMixin(
            output_file=temp_output_file,
            batch_size=2,
        )
        writer.set_csv_columns(["a", "b"])
        await writer.initialize()
        await writer.start()

        for i in range(10):
            await writer.buffered_csv_write([str(i), str(i * 10)])

        await writer.stop()

        content = temp_output_file.read_text()
        lines = content.strip().split("\n")
        header_count = sum(1 for line in lines if line == "a,b")
        assert header_count == 1
        assert len(lines) == 11  # 1 header + 10 data rows
