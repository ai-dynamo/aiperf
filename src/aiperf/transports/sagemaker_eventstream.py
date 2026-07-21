# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decode AWS ``application/vnd.amazon.eventstream`` byte streams (SageMaker's
streaming wire format) directly into :class:`AwsEventStreamMessage` objects.

Used by ``AioHttpTransport`` when the response Content-Type is
``application/vnd.amazon.eventstream`` (SageMaker streaming responses).

SageMaker's PayloadPart format is NOT SSE: no ``event:``/``id:``/``retry:``
field typing, no ``\\n\\n`` message delimiter - just one line of (optionally
``data: ``-prefixed) text per PayloadPart. ``AwsEventStreamMessage`` models
that directly instead of faking an ``SSEMessage`` shape, so this reader never
silently inherits (or misses) ``SSEMessage.parse()``'s SSE-specific behavior
(continuation-line stitching, multi-field parsing) that doesn't apply here.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator

from aiperf.common.models import AwsEventStreamMessage

EVENTSTREAM_CONTENT_TYPE = "application/vnd.amazon.eventstream"


class SageMakerStreamError(RuntimeError):
    """Raised when a SageMaker eventstream carries an error/exception event."""


def _split_complete_lines(buffer: bytearray) -> tuple[list[bytes], bytearray]:
    """Split ``buffer`` on ``\\n``, returning complete non-empty lines and the remainder."""
    lines: list[bytes] = []
    while b"\n" in buffer:
        line, _, rest = buffer.partition(b"\n")
        buffer = bytearray(rest)
        stripped = line.strip()
        if stripped:
            lines.append(bytes(stripped))
    return lines, buffer


def _strip_data_prefix(text: str) -> str:
    """Strip a leading ``data:`` prefix, if present -- some SageMaker
    containers emit the raw SSE-formatted line; others emit bare JSON.

    NOTE: this is a narrow, single-prefix version of the same convention
    ``SSEMessage.parse()`` handles generically as part of its full field-name/
    value split (``record_models.py``, ``SSEMessage.parse``). They are
    intentionally separate implementations -- this format has no other SSE
    field types to parse -- but a change to one's `data:` handling should
    prompt checking the other for consistency.
    """
    if text.startswith("data:"):
        return text[len("data:") :].strip()
    return text


def _stream_message_for_line(line: bytes, perf_ns: int) -> AwsEventStreamMessage:
    """Turn one PayloadPart line into an AwsEventStreamMessage."""
    text = _strip_data_prefix(line.decode("utf-8", errors="replace").strip())
    return AwsEventStreamMessage(perf_ns=perf_ns, line=text, raw_line=bytes(line))


class EventStreamReader:
    """Parse an AWS ``application/vnd.amazon.eventstream`` binary byte stream
    directly into :class:`AwsEventStreamMessage` objects, one per
    ``PayloadPart`` line.

    Mirrors :class:`~aiperf.transports.sse_utils.AsyncSSEStreamReader`'s
    interface (``__aiter__`` yielding a message type,
    ``inspect_message_for_error``) so callers can pick either reader based
    on response content-type and use it identically - no branching needed
    downstream.

    Uses ``botocore.eventstream.EventStreamBuffer`` for the binary frame
    decoding only (pure parsing, no I/O/threading needed).
    """

    def __init__(self, async_iter: AsyncIterator[bytes]):
        self._async_iter = async_iter

    async def __aiter__(self) -> AsyncIterator[AwsEventStreamMessage]:
        from botocore.eventstream import EventStreamBuffer

        decoder = EventStreamBuffer()
        line_buffer = bytearray()
        async for chunk in self._async_iter:
            chunk_perf_ns = time.perf_counter_ns()
            decoder.add_data(chunk)
            while True:
                try:
                    message = decoder.next()
                except StopIteration:
                    break
                message_type = message.headers.get(":message-type")
                if message_type in ("error", "exception"):
                    raise SageMakerStreamError(
                        f"SageMaker eventstream {message_type}: "
                        f"{message.payload.decode('utf-8', errors='replace')}"
                    )
                if message.headers.get(":event-type") != "PayloadPart":
                    continue
                # Accumulate across PayloadPart messages and only split on a
                # real newline -- a JSON/SSE line can legitimately span more
                # than one PayloadPart, so a partial trailing fragment must
                # stay buffered until a later message completes it, rather
                # than being treated as a complete line on its own.
                line_buffer += message.payload
                lines, line_buffer = _split_complete_lines(line_buffer)
                for line in lines:
                    yield _stream_message_for_line(line, chunk_perf_ns)
        stripped = line_buffer.strip()
        if stripped:
            yield _stream_message_for_line(bytes(stripped), time.perf_counter_ns())

    @staticmethod
    def inspect_message_for_error(message: AwsEventStreamMessage) -> None:
        """No-op: :meth:`__aiter__` already raises ``SageMakerStreamError``
        inline when it encounters an error/exception event. Provided only
        for interface parity with ``AsyncSSEStreamReader.inspect_message_for_error``
        so callers can call it unconditionally on either reader's output.
        """
