# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Merge per-processor response fragments into a sorted outputs.json."""

import asyncio
import heapq
import os
import tempfile
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AsyncExitStack, aclosing, asynccontextmanager
from pathlib import Path
from typing import Any, TypeVar

import aiofiles
import orjson

from aiperf.common.enums import CreditPhase
from aiperf.common.exceptions import DataExporterDisabled
from aiperf.common.finite import scrub_non_finite
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.config.artifacts import OutputDefaults
from aiperf.exporters.exporter_config import ExporterConfig, FileExportInfo

JsonObject = dict[str, Any]
SortRow = tuple[int, int, int, int, JsonObject]
T = TypeVar("T")


class OutputsJsonExporter(AIPerfLoggerMixin):
    """Sort output fragments on disk and atomically publish schema 1.1 JSON."""

    SCHEMA_VERSION = "1.1"
    CHUNK_BYTES = 8 * 1024 * 1024
    CHUNK_RECORDS = 20_000
    MERGE_FAN_IN = 32
    READ_BYTES = 64 * 1024
    WRITE_BYTES = 1024 * 1024

    def __init__(self, exporter_config: ExporterConfig, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._cfg = exporter_config.cfg

        if not self._cfg.artifacts.export_outputs_json:
            raise DataExporterDisabled(
                "OutputsJsonExporter is disabled (--export-outputs-json not set)"
            )

        self._file_path = self._cfg.artifacts.outputs_json_file
        self._fragments_dir = (
            self._cfg.artifacts.artifact_directory
            / OutputDefaults.OUTPUT_FRAGMENTS_FOLDER
        )

    def get_export_info(self) -> FileExportInfo:
        """Return export metadata for logging."""
        return FileExportInfo(export_type="Outputs JSON", file_path=self._file_path)

    async def export(self) -> None:
        """Publish a sorted document, retaining source fragments on failure."""
        fragment_files = list(self._fragments_dir.glob("output_fragments_*.jsonl"))
        if not fragment_files:
            self.debug("No output fragment files found, skipping outputs.json export")
            return

        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".outputs-json-", dir=self._file_path.parent
        ) as scratch_name:
            scratch = Path(scratch_name)
            run_count, data_count, warmup_count = await self._make_runs(
                fragment_files, scratch
            )
            final_run = await self._merge_runs(scratch, run_count)
            staged = scratch / "outputs.json"
            await self._write_document(staged, final_run)
            await asyncio.sleep(0)
            # A synchronous rename is the commit point: cancellation cannot land
            # between replacing the old artifact and retiring its fragments.
            os.replace(staged, self._file_path)

        self.info(
            f"Exported {data_count} records ({warmup_count} warmup) "
            f"to {self._file_path}"
        )
        self._cleanup_fragments(fragment_files)

    @staticmethod
    async def _settled(operation: Awaitable[T]) -> T:
        """Let one pending file operation finish before cancellation frees scratch."""
        task = asyncio.ensure_future(operation)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
            if not task.cancelled():
                task.exception()
            raise

    @asynccontextmanager
    async def _open(self, path: Path, mode: str) -> AsyncIterator[Any]:
        opening = asyncio.ensure_future(aiofiles.open(path, mode))
        try:
            handle = await self._settled(opening)
        except asyncio.CancelledError:
            if (
                opening.done()
                and not opening.cancelled()
                and opening.exception() is None
            ):
                await self._settled(opening.result().close())
            raise
        try:
            yield handle
        finally:
            await self._settled(handle.close())

    @staticmethod
    def _entry(fragment: JsonObject) -> JsonObject:
        phase = fragment.get("benchmark_phase")
        return scrub_non_finite(
            {
                "session_num": fragment["session_num"],
                "conversation_id": fragment.get("conversation_id"),
                "turn_index": fragment.get("turn_index"),
                "x_request_id": fragment.get("x_request_id"),
                "benchmark_phase": phase,
                "request_start_ns": fragment.get("request_start_ns"),
                "request_end_ns": fragment.get("request_end_ns"),
                "metrics": fragment.get("metrics") or {},
                "response_text": fragment.get("response_text"),
            }
        )

    async def _lines(self, path: Path) -> AsyncIterator[bytes]:
        """Read bounded batches, apart from an individual oversized line."""
        async with self._open(path, "rb") as source:
            while batch := await self._settled(source.readlines(self.READ_BYTES)):
                for line in batch:
                    yield line

    async def _make_runs(
        self, fragment_files: list[Path], scratch: Path
    ) -> tuple[int, int, int]:
        chunk: list[SortRow] = []
        chunk_bytes = 0
        run_count = data_count = warmup_count = ordinal = 0
        for file in fragment_files:
            async with aclosing(self._lines(file)) as lines:
                async for line in lines:
                    if not line.strip():
                        continue
                    try:
                        fragment = orjson.loads(line)
                    except orjson.JSONDecodeError as exc:
                        self.warning(
                            f"Skipping unparsable fragment line in {file}: {exc}"
                        )
                        continue
                    if not isinstance(fragment, dict):
                        self.warning(
                            f"Skipping non-object fragment line in {file}: {line!r}"
                        )
                        continue
                    entry = self._entry(fragment)
                    phase = int(fragment.get("benchmark_phase") == CreditPhase.WARMUP)
                    warmup_count += phase
                    data_count += 1 - phase
                    chunk.append(
                        (
                            phase,
                            entry["session_num"],
                            entry.get("turn_index") or 0,
                            ordinal,
                            entry,
                        )
                    )
                    ordinal += 1
                    chunk_bytes += len(line)
                    if (
                        len(chunk) >= self.CHUNK_RECORDS
                        or chunk_bytes >= self.CHUNK_BYTES
                    ):
                        await self._write_run(scratch / f"run-0-{run_count}", chunk)
                        run_count += 1
                        chunk = []
                        chunk_bytes = 0
        if chunk:
            await self._write_run(scratch / f"run-0-{run_count}", chunk)
            run_count += 1
        return run_count, data_count, warmup_count

    async def _write_run(self, path: Path, chunk: list[SortRow]) -> None:
        chunk.sort(key=lambda row: row[:4])
        async with self._open(path, "wb") as output:
            buffer = bytearray()
            for row in chunk:
                buffer.extend(orjson.dumps(row) + b"\n")
                if len(buffer) >= self.WRITE_BYTES:
                    await self._settled(output.write(buffer))
                    buffer.clear()
            if buffer:
                await self._settled(output.write(buffer))

    async def _merge_runs(self, scratch: Path, count: int) -> Path | None:
        if count == 0:
            return None
        pass_index = 0
        while count > 1:
            next_count = 0
            for start in range(0, count, self.MERGE_FAN_IN):
                source_paths = [
                    scratch / f"run-{pass_index}-{i}"
                    for i in range(start, min(start + self.MERGE_FAN_IN, count))
                ]
                target = scratch / f"run-{pass_index + 1}-{next_count}"
                if len(source_paths) == 1:
                    await self._settled(
                        asyncio.to_thread(os.replace, source_paths[0], target)
                    )
                else:
                    await self._merge_group(source_paths, target)
                    for path in source_paths:
                        path.unlink()
                next_count += 1
            count = next_count
            pass_index += 1
        return scratch / f"run-{pass_index}-0"

    async def _merge_group(self, sources: list[Path], target: Path) -> None:
        async with AsyncExitStack() as stack:
            readers = [self._lines(path) for path in sources]
            for reader in readers:
                stack.push_async_callback(reader.aclose)
            output = await stack.enter_async_context(self._open(target, "wb"))
            heap: list[tuple[tuple[int, int, int, int], int, bytes]] = []
            buffer = bytearray()
            for index, reader in enumerate(readers):
                if line := await anext(reader, None):
                    row = orjson.loads(line)
                    heapq.heappush(heap, (tuple(row[:4]), index, line))
            while heap:
                _, index, line = heapq.heappop(heap)
                buffer.extend(line)
                if len(buffer) >= self.WRITE_BYTES:
                    await self._settled(output.write(buffer))
                    buffer.clear()
                if next_line := await anext(readers[index], None):
                    row = orjson.loads(next_line)
                    heapq.heappush(heap, (tuple(row[:4]), index, next_line))
            if buffer:
                await self._settled(output.write(buffer))

    async def _write_document(self, staged: Path, run: Path | None) -> None:
        async with self._open(staged, "wb") as output:
            buffer = bytearray()

            async def emit(piece: bytes) -> None:
                buffer.extend(piece)
                if len(buffer) >= self.WRITE_BYTES:
                    await self._settled(output.write(buffer))
                    buffer.clear()

            await emit(b'{\n  "schema_version": "1.1",\n  "data": [')
            previous_phase = await self._write_entries(run, emit)
            if previous_phase == 0:
                await emit(b'\n  ],\n  "warmup": []\n}')
            elif previous_phase == 1:
                await emit(b"\n  ]\n}")
            else:
                await emit(b'],\n  "warmup": []\n}')
            if buffer:
                await self._settled(output.write(buffer))

    async def _write_entries(
        self, run: Path | None, emit: Callable[[bytes], Awaitable[None]]
    ) -> int | None:
        if run is None:
            return None
        previous_phase: int | None = None
        async with aclosing(self._lines(run)) as lines:
            async for line in lines:
                phase, _, _, _, entry = orjson.loads(line)
                if previous_phase is None:
                    if phase == 0:
                        await emit(b"\n")
                    else:
                        await emit(b'],\n  "warmup": [\n')
                elif phase != previous_phase:
                    await emit(b'\n  ],\n  "warmup": [\n')
                else:
                    await emit(b",\n")
                encoded = orjson.dumps(entry, option=orjson.OPT_INDENT_2)
                await emit(b"    " + encoded.replace(b"\n", b"\n    "))
                previous_phase = phase
        return previous_phase

    def _cleanup_fragments(self, fragment_files: list[Path]) -> None:
        """Remove source fragments only after the final document is published."""
        for file in fragment_files:
            file.unlink(missing_ok=True)
        try:
            self._fragments_dir.rmdir()
        except OSError:
            self.debug(
                f"Could not remove fragments directory (may not be empty): {self._fragments_dir}"
            )
