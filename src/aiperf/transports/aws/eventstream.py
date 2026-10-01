# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decode AWS ``application/vnd.amazon.eventstream`` byte streams directly into
:class:`AwsEventStreamMessage` objects.

Used by ``AioHttpClient`` whenever a response arrives with that content type.
The dispatch is keyed on the content type alone and never on which AWS service
produced the bytes, so Bedrock's streaming responses decode through this same
reader with no change here -- SageMaker is simply the first caller.

The framing is NOT SSE: no ``event:``/``id:``/``retry:`` field typing and no
``\\n\\n`` message delimiter -- just one line of (optionally ``data: ``-prefixed)
text per PayloadPart. :class:`AwsEventStreamMessage` models that directly rather
than faking an ``SSEMessage`` shape, so this reader never silently inherits (or
misses) ``SSEMessage.parse()``'s SSE-specific behavior (continuation-line
stitching, multi-field parsing) that does not apply here.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Iterator
from functools import cache
from typing import TYPE_CHECKING, Any

from aiperf.common.models import AwsEventStreamMessage
from aiperf.common.optional_dependencies import aws_dependency_message

if TYPE_CHECKING:
    from botocore.eventstream import EventStreamBuffer

EVENTSTREAM_CONTENT_TYPE = "application/vnd.amazon.eventstream"


class AwsEventStreamError(RuntimeError):
    """Raised when an AWS eventstream carries an error or exception event.

    ``exception_type`` carries AWS's own label for the failure (the
    ``:exception-type`` or ``:error-code`` frame header, e.g.
    ``ModelStreamError``, ``InternalStreamFailure``). It is surfaced as
    ``ErrorDetails.type`` so the error table groups distinct server-side
    failures separately instead of collapsing them into one generic bucket.
    """

    def __init__(self, message: str, *, exception_type: str | None = None) -> None:
        super().__init__(message)
        self.exception_type: str | None = exception_type


@cache
def _event_stream_buffer_cls() -> type[EventStreamBuffer]:
    """Import botocore's frame decoder once, not once per stream.

    ``__aiter__`` runs for every streamed response, so the import is cached here
    rather than repeated in that hot path. A missing botocore raises an error
    naming the ``aiperf[aws]`` extra rather than a bare ``ModuleNotFoundError``.
    """
    try:
        from botocore.eventstream import EventStreamBuffer
    except ImportError as e:
        raise ImportError(
            aws_dependency_message("An AWS eventstream response was received")
        ) from e
    return EventStreamBuffer


def _split_complete_lines(buffer: bytearray) -> tuple[list[bytes], bytearray]:
    """Split ``buffer`` on ``\\n``, returning complete non-empty lines and the remainder.

    Split once, keeping only the trailing fragment: a PayloadPart can batch many
    lines, and peeling them off one at a time re-copied the remainder per line.
    """
    *complete, fragment = bytes(buffer).split(b"\n")
    # Emptiness is judged on the stripped form, but the line keeps its bytes:
    # ``AwsEventStreamMessage.raw_line`` is the undecoded wire content, and
    # raw-record exports must reproduce what was sent.
    return [line for line in complete if line.strip()], bytearray(fragment)


def _strip_data_prefix(text: str) -> str:
    """Strip a leading ``data:`` prefix, if present -- some containers emit the
    raw SSE-formatted line; others emit bare JSON.

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
    """Turn one PayloadPart line into an AwsEventStreamMessage.

    Decoded strictly: ``raw_line`` keeps the bytes as received, and bytes that
    are not UTF-8 cannot be JSON-serialized with the record, which would drop
    the request from every metric. A non-UTF-8 line is a malformed stream, so it
    fails the request instead. Lines are complete here, so a multi-byte
    character split across PayloadParts has already been reassembled.
    """
    try:
        decoded = line.decode("utf-8")
    except UnicodeDecodeError as e:
        raise AwsEventStreamError(
            f"AWS eventstream PayloadPart line is not valid UTF-8 "
            f"(byte {e.start}): {line[:64]!r}"
        ) from e
    text = _strip_data_prefix(decoded.strip())
    return AwsEventStreamMessage(perf_ns=perf_ns, line=text, raw_line=bytes(line))


def _error_from_frame(
    message_type: str, message: Any, partial_line: bytes = b""
) -> AwsEventStreamError:
    """Build a typed error from an ``error``/``exception`` frame.

    AWS labels the failure in ``:exception-type`` (exception frames) or
    ``:error-code`` (error frames), with prose in ``:error-message``. Both are
    read so the label survives into ``ErrorDetails.type``.

    ``partial_line`` is whatever had been buffered but not yet newline-terminated
    when the failure arrived. The clean-EOF path flushes such a fragment as a
    message; raising would otherwise discard it, and a half-written token is
    often the most diagnostic thing about a mid-stream server failure.
    """
    headers = message.headers
    exception_type = headers.get(":exception-type") or headers.get(":error-code")
    detail = headers.get(":error-message") or message.payload.decode(
        "utf-8", errors="replace"
    )
    label = exception_type or message_type
    text = f"AWS eventstream {message_type} ({label}): {detail}"
    if partial_line.strip():
        decoded = partial_line.decode("utf-8", errors="replace")
        text = f"{text} [partial line buffered at failure: {decoded!r}]"
    return AwsEventStreamError(text, exception_type=exception_type)


def _decode_error(exc: Exception) -> AwsEventStreamError:
    """Wrap a botocore framing/CRC failure as an eventstream error.

    botocore owns frame and checksum validation and raises its own types
    (``ChecksumMismatch``, ``ParserError``, ...). Wrapping gives the HTTP client
    one type to catch for every eventstream failure, with a common message
    prefix, while ``exception_type`` keeps botocore's class name as the error
    table's label.
    """
    return AwsEventStreamError(
        f"Malformed AWS eventstream frame: {type(exc).__name__}: {exc}",
        exception_type=type(exc).__name__,
    )


def _complete_frames(decoder: EventStreamBuffer) -> Iterator[Any]:
    """Yield every complete frame ``decoder`` holds, wrapping botocore failures."""
    while True:
        try:
            yield decoder.next()
        except StopIteration:
            return
        except Exception as e:
            raise _decode_error(e) from e


class AwsEventStreamReader:
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
        decoder = _event_stream_buffer_cls()()
        line_buffer = bytearray()
        # botocore reports an incomplete frame only as "need more data", so a
        # body that ends mid-frame is detected here by counting, rather than by
        # reading the decoder's private buffer.
        bytes_received = 0
        bytes_decoded = 0
        async for chunk in self._async_iter:
            # One timestamp per network read, shared by every message decoded
            # from it. This matches what sse_utils.py already does for SSE
            # (one `chunk_perf_ns` per read). Stamping each frame at decode
            # time instead would only add decode latency to the measurement --
            # the frames genuinely arrived together -- and would make
            # eventstream ITL non-comparable to SSE ITL from the same server.
            chunk_perf_ns = time.perf_counter_ns()
            bytes_received += len(chunk)
            try:
                decoder.add_data(chunk)
            except Exception as e:
                raise _decode_error(e) from e
            for message in _complete_frames(decoder):
                bytes_decoded += message.prelude.total_length
                message_type = message.headers.get(":message-type")
                if message_type in ("error", "exception"):
                    raise _error_from_frame(message_type, message, bytes(line_buffer))
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
        # Same rule as the main loop: emptiness on the stripped form, bytes
        # as received on the message.
        if line_buffer.strip():
            yield _stream_message_for_line(bytes(line_buffer), time.perf_counter_ns())
        if bytes_received > bytes_decoded:
            raise AwsEventStreamError(
                f"AWS eventstream ended inside an incomplete frame: "
                f"{bytes_received - bytes_decoded} trailing byte(s) after the last "
                f"complete frame"
            )

    @staticmethod
    def inspect_message_for_error(message: AwsEventStreamMessage) -> None:
        """No-op: :meth:`__aiter__` already raises :class:`AwsEventStreamError`
        inline when it encounters an error/exception event. Provided only
        for interface parity with ``AsyncSSEStreamReader.inspect_message_for_error``
        so callers can call it unconditionally on either reader's output.
        """
