# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for EventStreamReader, the manual AWS eventstream binary-frame parser
used for SageMaker streaming responses.

Frames are hand-encoded to the real ``application/vnd.amazon.eventstream``
wire format (prelude + headers + payload + CRC) rather than mocked, so these
tests exercise the actual ``botocore.eventstream.EventStreamBuffer`` decode
path plus this module's line-buffering on top of it.
"""

from __future__ import annotations

import struct
import zlib
from collections.abc import AsyncIterator
from unittest.mock import MagicMock

import pytest

from aiperf.transports.sagemaker_eventstream import (
    EventStreamReader,
    SageMakerStreamError,
)


def _encode_header(name: str, value: str) -> bytes:
    name_b = name.encode("utf-8")
    value_b = value.encode("utf-8")
    return (
        struct.pack(">B", len(name_b))
        + name_b
        + struct.pack(">B", 7)  # header value type 7 == string
        + struct.pack(">H", len(value_b))
        + value_b
    )


def encode_frame(
    payload: bytes,
    *,
    message_type: str = "event",
    event_type: str = "PayloadPart",
) -> bytes:
    """Hand-roll one AWS eventstream binary frame matching the wire format
    SageMaker's runtime actually emits: 4-byte total length, 4-byte headers
    length, prelude CRC, headers block, payload, message CRC."""
    headers = (
        _encode_header(":message-type", message_type)
        + _encode_header(":event-type", event_type)
        + _encode_header(":content-type", "application/json")
    )
    headers_len = len(headers)
    total_len = 4 + 4 + 4 + headers_len + len(payload) + 4
    prelude = struct.pack(">II", total_len, headers_len)
    prelude_crc = struct.pack(">I", zlib.crc32(prelude) & 0xFFFFFFFF)
    message_no_crc = prelude + prelude_crc + headers + payload
    message_crc = struct.pack(">I", zlib.crc32(message_no_crc) & 0xFFFFFFFF)
    return message_no_crc + message_crc


async def _chunks(*items: bytes) -> AsyncIterator[bytes]:
    for item in items:
        yield item


class TestEventStreamReaderBasicDecoding:
    @pytest.mark.asyncio
    async def test_single_payload_part_yields_one_message(self) -> None:
        frame = encode_frame(b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n')

        messages = [m async for m in EventStreamReader(_chunks(frame))]

        assert len(messages) == 1
        assert messages[0].line == ('{"choices":[{"delta":{"content":"Hello"}}]}')

    @pytest.mark.asyncio
    async def test_multiple_payload_parts_yield_multiple_messages(self) -> None:
        frame1 = encode_frame(b'data: {"delta":"tok-a"}\n')
        frame2 = encode_frame(b'data: {"delta":"tok-b"}\n')

        messages = [m async for m in EventStreamReader(_chunks(frame1, frame2))]

        assert [m.line for m in messages] == [
            '{"delta":"tok-a"}',
            '{"delta":"tok-b"}',
        ]

    @pytest.mark.asyncio
    async def test_bare_json_line_without_data_prefix(self) -> None:
        """Some SageMaker containers emit bare JSON PayloadPart lines instead
        of SSE-formatted ``data: ...`` lines; both must decode identically."""
        frame = encode_frame(b'{"delta":"no-prefix"}\n')

        messages = [m async for m in EventStreamReader(_chunks(frame))]

        assert messages[0].line == '{"delta":"no-prefix"}'

    @pytest.mark.asyncio
    async def test_frame_split_across_multiple_chunks(self) -> None:
        """A frame arriving fragmented over TCP must still decode -- the
        underlying botocore.eventstream.EventStreamBuffer buffers partial
        frames across add_data() calls."""
        frame = encode_frame(b'data: {"delta":"split"}\n')
        midpoint = len(frame) // 2

        messages = [
            m
            async for m in EventStreamReader(
                _chunks(frame[:midpoint], frame[midpoint:])
            )
        ]

        assert messages[0].line == '{"delta":"split"}'

    @pytest.mark.asyncio
    async def test_payload_part_split_across_two_eventstream_messages_reassembles(
        self,
    ) -> None:
        """A JSON line can legitimately span two separate PayloadPart
        *messages* (as opposed to split across raw TCP bytes within one
        message, which botocore's EventStreamBuffer already reassembles
        before this code ever sees it). The line buffer must accumulate
        across messages and only split on a real newline, so a partial
        trailing fragment from one PayloadPart is completed by the next
        rather than being treated as a complete line on its own."""
        payload = b'data: {"delta":"reassembled"}\n'
        frame1 = encode_frame(payload[:15])
        frame2 = encode_frame(payload[15:])

        messages = [m async for m in EventStreamReader(_chunks(frame1, frame2))]

        assert len(messages) == 1
        assert messages[0].line == '{"delta":"reassembled"}'

    @pytest.mark.asyncio
    async def test_trailing_line_without_newline_is_flushed(self) -> None:
        """A final PayloadPart with no trailing newline must still be
        emitted once the stream ends, not silently dropped."""
        frame = encode_frame(b'data: {"delta":"no-trailing-newline"}')

        messages = [m async for m in EventStreamReader(_chunks(frame))]

        assert messages[0].line == '{"delta":"no-trailing-newline"}'

    @pytest.mark.asyncio
    async def test_non_payload_part_events_are_skipped(self) -> None:
        initial_response = encode_frame(b"", event_type="InitialResponse")
        payload = encode_frame(b'data: {"delta":"only-this"}\n')

        messages = [
            m async for m in EventStreamReader(_chunks(initial_response, payload))
        ]

        assert len(messages) == 1
        assert messages[0].line == '{"delta":"only-this"}'

    @pytest.mark.asyncio
    async def test_empty_stream_yields_no_messages(self) -> None:
        messages = [m async for m in EventStreamReader(_chunks())]
        assert messages == []


class TestEventStreamReaderErrorEvents:
    @pytest.mark.asyncio
    async def test_error_message_type_raises(self) -> None:
        frame = encode_frame(b"internal server error", message_type="error")

        with pytest.raises(SageMakerStreamError, match="internal server error"):
            async for _ in EventStreamReader(_chunks(frame)):
                pass

    @pytest.mark.asyncio
    async def test_exception_message_type_raises(self) -> None:
        frame = encode_frame(b"ModelError: bad input", message_type="exception")

        with pytest.raises(SageMakerStreamError, match="ModelError: bad input"):
            async for _ in EventStreamReader(_chunks(frame)):
                pass

    @pytest.mark.asyncio
    async def test_error_after_valid_messages_still_raises(self) -> None:
        good = encode_frame(b'data: {"delta":"ok"}\n')
        bad = encode_frame(b"stream failed", message_type="error")

        received = []
        with pytest.raises(SageMakerStreamError):
            async for message in EventStreamReader(_chunks(good, bad)):
                received.append(message)

        assert len(received) == 1
        assert received[0].line == '{"delta":"ok"}'


class TestEventStreamReaderInspectMessageForError:
    def test_is_a_noop(self) -> None:
        """inspect_message_for_error exists only for interface parity with
        AsyncSSEStreamReader -- EventStreamReader already raises inline, so
        this must never touch the message it's given."""
        message = MagicMock()
        EventStreamReader.inspect_message_for_error(message)
        message.assert_not_called()
