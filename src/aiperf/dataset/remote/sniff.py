# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decide whether a remote trace's first record has timestamps from a leading byte range."""

from __future__ import annotations

import asyncio
import re
import zlib
from collections.abc import Iterator
from pathlib import PurePosixPath

import orjson

from aiperf.dataset.remote import service
from aiperf.dataset.remote.models import ProbeResult
from aiperf.dataset.trace_timestamps import (
    first_record_has_timestamp,
    record_has_timestamp,
)

COLUMNAR_SUFFIXES = frozenset({".parquet", ".arrow", ".feather", ".ipc"})
PREFIX_STEPS: tuple[int, ...] = (64 * 1024, 1024 * 1024, 8 * 1024 * 1024)
# A first line longer than this is left undecided rather than buffered further, so a
# compressible prefix cannot inflate without bound; the full download decides instead.
MAX_FIRST_LINE_BYTES = 2 * PREFIX_STEPS[-1]
_GZIP_MAGIC = b"\x1f\x8b"
_INFLATE_CHUNK = 64 * 1024
# The line endings Python's universal-newlines text mode splits on, as the local read does.
_LINE_END = re.compile(rb"\r\n|\r|\n")


class _CorruptGzip(Exception):
    pass


def _gunzip(data: bytes, *, complete: bool) -> Iterator[bytes]:
    """Inflate every gzip member in ``data`` lazily, one bounded chunk per step.

    The caller stops pulling chunks once it has decided or hit MAX_FIRST_LINE_BYTES,
    so a highly compressible prefix is never inflated beyond that.

    Raises ``_CorruptGzip`` where the local gzip reader would fail: a bad stream, or a
    whole file that ends mid-member.
    """
    try:
        while data:
            decoder = zlib.decompressobj(wbits=31)
            while not decoder.eof:
                out = decoder.decompress(data, _INFLATE_CHUNK)
                data = decoder.unconsumed_tail
                if out:
                    yield out
                elif not data:
                    break
            if not decoder.eof:
                if complete:
                    raise _CorruptGzip
                return
            # Python's gzip reader accepts zero padding between members.
            data = decoder.unused_data.lstrip(b"\x00")
    except zlib.error as e:
        raise _CorruptGzip from e


def _parse_line(line: bytes) -> bool | None:
    """The local answer for one line, or None when the local reader would skip it."""
    try:
        stripped = line.decode("utf-8").strip()
    except UnicodeDecodeError:
        return False
    if not stripped:
        return None
    try:
        data = orjson.loads(stripped)
    except orjson.JSONDecodeError:
        return False
    if not isinstance(data, dict):
        return False
    return record_has_timestamp(data)


def sniff_first_record_timestamp(prefix: bytes, *, complete: bool) -> bool | None:
    """True/False once the first record is fully inside ``prefix``; None if undecided.

    None means more bytes are needed, or the first line (inflated) exceeds
    MAX_FIRST_LINE_BYTES; the caller then reads the full download instead.
    Mirrors first_record_has_timestamp for local files: gzip is decoded, anything else
    is read as text (so zstd and other binaries read as "no timestamp", as they do locally).
    """
    chunks = (
        _gunzip(prefix, complete=complete)
        if prefix.startswith(_GZIP_MAGIC)
        else iter((prefix,))
    )
    pending = bytearray()
    try:
        for chunk in chunks:
            # Bytes already in ``pending`` hold no line end, so only the new chunk is scanned.
            line_start, scan_from = 0, len(pending)
            pending += chunk
            while match := _LINE_END.search(pending, scan_from):
                # A trailing '\r' may be the first half of '\r\n'; either way the line ended.
                result = _parse_line(bytes(pending[line_start : match.start()]))
                if result is not None:
                    return result
                line_start = scan_from = match.end()
            del pending[:line_start]
            if len(pending) > MAX_FIRST_LINE_BYTES:
                return None
    except _CorruptGzip:
        return False
    if not complete:
        return None
    result = _parse_line(bytes(pending))
    return False if result is None else result


async def _downloaded_first_record_has_timestamp(probe: ProbeResult) -> bool:
    source = await service.materialize_probe(probe)
    return await asyncio.to_thread(first_record_has_timestamp, source.local_path)


async def remote_first_record_has_timestamp(probe: ProbeResult) -> bool:
    """first_record_has_timestamp for a probed remote file, read from the fewest bytes.

    Prefix reads escalate through PREFIX_STEPS. Offline mode (which allows no network
    reads), columnar files (whose schema sits in the footer), a Hugging Face parquet
    layout (whose split is downloaded whole), and a first record the sniff leaves
    undecided (longer than the last step, or past MAX_FIRST_LINE_BYTES once inflated)
    read the cached download instead.
    """
    if probe.data_is_dir and probe.data_files is None:
        # A folder with no parquet shards reads as a directory locally, which has no
        # first record; skip the download. The preflight promotes weka_trace by format.
        return False
    if (
        probe.data_files is not None
        or service.remote_offline()
        or PurePosixPath(probe.filename).suffix.lower() in COLUMNAR_SUFFIXES
    ):
        return await _downloaded_first_record_has_timestamp(probe)
    resolver = service.get_resolver(probe.ref.scheme)
    for nbytes in PREFIX_STEPS:
        data, complete = await service.with_transient_retry(
            lambda n=nbytes: resolver.read_prefix(probe, n)
        )
        result = sniff_first_record_timestamp(data, complete=complete)
        if result is not None:
            return result
        if complete:
            break
    return await _downloaded_first_record_has_timestamp(probe)
