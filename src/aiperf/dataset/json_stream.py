# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stream dataset JSON without a document-sized GIL-held encoder call."""

import asyncio
from collections.abc import AsyncIterator, Iterable, Iterator
from itertools import islice
from typing import Any

import orjson

_CHUNK_SIZE = 1024 * 1024
_STRING_CHARS = (_CHUNK_SIZE - 2) // 6  # A control character expands to six bytes.


def _string_parts(value: str) -> Iterator[bytes]:
    """Escape even a single large string through bounded orjson calls."""
    if len(value) <= _STRING_CHARS:
        yield orjson.dumps(value)
        return
    yield b'"'
    for start in range(0, len(value), _STRING_CHARS):
        yield orjson.dumps(value[start : start + _STRING_CHARS])[1:-1]
    yield b'"'


def _array_parts(value: Any, *, pretty: bool, depth: int) -> Iterator[bytes]:
    """Batch numeric arrays and stream other elements recursively."""
    yield b"["
    items = iter(value)
    first = True
    while batch := tuple(islice(items, 1024)):
        # Numeric batches stay below one MiB even at the maximum indentation.
        if depth < 254 and set(map(type, batch)) <= {int, float, bool, type(None)}:
            if not first:
                yield b","
            encoded = orjson.dumps(batch, option=orjson.OPT_INDENT_2 if pretty else 0)
            if pretty and depth:
                encoded = encoded.replace(b"\n", b"\n" + b"  " * depth)
            yield encoded[1 : -(2 * depth + 2) if pretty else -1]
        else:
            for item in batch:
                if not first:
                    yield b","
                if pretty:
                    yield b"\n" + b"  " * (depth + 1)
                yield from _parts(item, pretty=pretty, depth=depth + 1)
                first = False
        first = False
    if pretty and not first:
        yield b"\n" + b"  " * depth
    yield b"]"


def _object_parts(value: dict, *, pretty: bool, depth: int) -> Iterator[bytes]:
    """Stream object members in insertion order, including large keys."""
    yield b"{"
    first = True
    for key, item in value.items():
        if type(key) is not str:
            raise orjson.JSONEncodeError("Dict key must be str")
        if not first:
            yield b","
        if pretty:
            yield b"\n" + b"  " * (depth + 1)
        yield from _string_parts(key)
        yield b": " if pretty else b":"
        yield from _parts(item, pretty=pretty, depth=depth + 1)
        first = False
    if pretty and not first:
        yield b"\n" + b"  " * depth
    yield b"}"


def _parts(value: Any, *, pretty: bool, depth: int = 0) -> Iterator[bytes]:
    """Stream containers while retaining orjson's scalar encoding."""
    if depth > 254:
        raise orjson.JSONEncodeError("Recursion limit reached")
    if isinstance(value, str):
        yield from _string_parts(value)
    elif isinstance(value, dict):
        yield from _object_parts(value, pretty=pretty, depth=depth)
    elif isinstance(value, (list, Iterator)) or type(value) is tuple:
        yield from _array_parts(value, pretty=pretty, depth=depth)
    else:
        data = orjson.dumps(value, option=orjson.OPT_INDENT_2 if pretty else 0)
        yield data.replace(b"\n", b"\n" + b"  " * depth) if pretty and depth else data


def _chunks(parts: Iterator[bytes]) -> Iterator[tuple[bytes, bool]]:
    """Batch small JSON tokens to avoid a thread handoff for every scalar.

    Flags the final chunk so the caller skips a handoff just to see exhaustion.
    """
    buffer = bytearray()
    for part in parts:
        if len(buffer) + len(part) > _CHUNK_SIZE:
            yield bytes(buffer), False
            buffer.clear()
        buffer.extend(part)
    yield bytes(buffer), True


async def _iter_chunks(parts: Iterator[bytes]) -> AsyncIterator[bytes]:
    """Produce each chunk on a worker thread."""
    chunks = _chunks(parts)
    last = False
    while not last:
        chunk, last = await asyncio.to_thread(next, chunks)
        if chunk:
            yield chunk


async def iter_json_bytes(value: Any, *, pretty: bool = False) -> AsyncIterator[bytes]:
    """Yield JSON chunks with orjson's compact or two-space-indented formatting.

    Containers stream recursively; string slices and numeric batches bound
    the work in each native encoder call.
    """
    async for chunk in _iter_chunks(_parts(value, pretty=pretty)):
        yield chunk


async def iter_concatenated_json_bytes(
    values: Iterable[Any], sizes: list[int]
) -> AsyncIterator[bytes]:
    """Yield the compact JSON of each value back to back.

    Small values share a chunk, and so a thread handoff, instead of paying one
    each. Appends each value's encoded byte size to ``sizes`` once encoded.
    """

    def parts() -> Iterator[bytes]:
        for value in values:
            size = 0
            for part in _parts(value, pretty=False):
                size += len(part)
                yield part
            sizes.append(size)

    async for chunk in _iter_chunks(parts()):
        yield chunk
