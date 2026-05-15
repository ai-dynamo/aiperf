# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mixin for buffered CSV writing with automatic flushing."""

import asyncio
from pathlib import Path

import aiofiles

from aiperf.common.environment import Environment
from aiperf.common.hooks import on_init, on_stop
from aiperf.common.mixins.aiperf_lifecycle_mixin import AIPerfLifecycleMixin
from aiperf.common.utils import yield_to_event_loop

_CSV_SPECIAL_CHARS = frozenset(',"\n\r')


class BufferedCSVWriterMixin(AIPerfLifecycleMixin):
    """Mixin for buffered CSV writing with automatic flushing.

    Writes flat CSV rows with a header row. The header is written on the first
    flush using the column names provided via ``set_csv_columns``. Each row is
    a list of stringified values in column order.

    Attributes:
        output_file: Path to the CSV output file
        rows_written: Number of data rows written (excludes header)
    """

    def __init__(
        self,
        output_file: Path,
        batch_size: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.output_file = output_file
        self.rows_written = 0
        self._csv_file_handle = None
        self._csv_file_lock = asyncio.Lock()
        self._csv_buffer: list[bytes] = []
        self._csv_batch_size = batch_size
        self._csv_columns: list[str] | None = None
        self._csv_header_written = False

    def set_csv_columns(self, columns: list[str]) -> None:
        """Set the CSV column headers. Must be called before the first write."""
        self._csv_columns = columns

    @on_init
    async def _csv_open_file(self) -> None:
        """Open the file handle for writing in binary mode."""
        try:
            self.output_file.parent.mkdir(parents=True, exist_ok=True)
            self.output_file.unlink(missing_ok=True)
        except Exception as e:
            self.exception(
                f"Failed to create output file directory or clear file: {self.output_file}: {e!r}"
            )
            raise

        async with self._csv_file_lock:
            self._csv_file_handle = await aiofiles.open(self.output_file, mode="wb")

    @staticmethod
    def _csv_escape(value: str) -> str:
        """Escape a CSV field value per RFC 4180."""
        if _CSV_SPECIAL_CHARS.intersection(value):
            return '"' + value.replace('"', '""') + '"'
        return value

    @staticmethod
    def _csv_encode_row(values: list[str]) -> bytes:
        """Encode a list of string values as a CSV row in bytes."""
        escape = BufferedCSVWriterMixin._csv_escape
        return ",".join(escape(v) for v in values).encode("utf-8")

    async def buffered_csv_write(self, row: list[str]) -> None:
        """Write a CSV row to the buffer with automatic flushing.

        Args:
            row: List of string values in column order.
        """
        try:
            row_bytes = self._csv_encode_row(row)

            buffer_to_flush = None
            self._csv_buffer.append(row_bytes)
            self.rows_written += 1

            if len(self._csv_buffer) >= self._csv_batch_size:
                buffer_to_flush = self._csv_buffer
                self._csv_buffer = []

            if buffer_to_flush:
                self.execute_async(self._csv_flush_buffer(buffer_to_flush))

        except (OSError, TypeError, ValueError) as e:
            # OSError from underlying file I/O; Type/Value from encoding a malformed row.
            self.error(f"Failed to write CSV row: {e!r}")

    async def _csv_flush_buffer(self, buffer_to_flush: list[bytes]) -> None:
        """Write buffered CSV rows to disk."""
        if not buffer_to_flush:
            return
        async with self._csv_file_lock:
            if self._csv_file_handle is None:
                self.error(
                    f"Tried to flush CSV buffer, but file handle is not open: {self.output_file}"
                )
                return

            try:
                self.debug(lambda: f"Flushing {len(buffer_to_flush)} CSV rows to file")
                parts: list[bytes] = []

                if not self._csv_header_written and self._csv_columns is not None:
                    header_bytes = self._csv_encode_row(self._csv_columns)
                    parts.append(header_bytes)
                    self._csv_header_written = True

                parts.extend(buffer_to_flush)
                bulk_data = b"\n".join(parts) + b"\n"
                await self._csv_file_handle.write(bulk_data)
                await self._csv_file_handle.flush()
            except OSError as e:
                self.exception(f"Failed to flush CSV buffer: {e!r}")

    @on_stop
    async def _csv_close_file(self) -> None:
        """Flush remaining buffer and close the file handle."""
        # wait_for_tasks() snapshots self.tasks at entry, so a flush task
        # created AFTER entry — by a late buffered_csv_write whose upstream
        # pull hasn't stopped yet — is not awaited. Its rows sit in the
        # buffer and the task then hits a closed file handle. Drain in a
        # loop until both self.tasks and self._csv_buffer are stable.
        for _ in range(3):
            if not self.tasks and not self._csv_buffer:
                break
            if self.tasks:
                try:
                    await asyncio.wait_for(
                        self.wait_for_tasks(),
                        timeout=Environment.SERVICE.TASK_CANCEL_TIMEOUT_SHORT,
                    )
                except asyncio.TimeoutError:
                    self.warning(
                        f"Timeout waiting for {len(self.tasks)} pending CSV flush tasks during shutdown. "
                        "Cancelling tasks and proceeding with cleanup."
                    )
                    await self.cancel_all_tasks()
                    await yield_to_event_loop()
                    break

            buffer_to_flush = self._csv_buffer
            self._csv_buffer = []
            if buffer_to_flush:
                try:
                    await self._csv_flush_buffer(buffer_to_flush)
                except OSError as e:
                    self.error(
                        f"Failed to flush remaining CSV buffer during shutdown: {e}"
                    )

        async with self._csv_file_lock:
            if self._csv_file_handle is not None:
                try:
                    await self._csv_file_handle.close()
                    self.debug(lambda: f"CSV file handle closed: {self.output_file}")
                except OSError as e:
                    self.exception(
                        f"Failed to close CSV file handle during shutdown: {e}"
                    )
                finally:
                    self._csv_file_handle = None

        self.debug(
            lambda: f"{self.__class__.__name__}: {self.rows_written} CSV rows written to {self.output_file}"
        )

        if self.rows_written == 0:
            self.debug(
                lambda: f"No rows written, deleting output file: {self.output_file}"
            )
            self.output_file.unlink(missing_ok=True)
