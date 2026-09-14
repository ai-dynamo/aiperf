# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for AwsEventStreamReader, the manual AWS eventstream binary-frame parser
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

# botocore ships only in the optional aiperf[aws] extra, and CI installs
# "--extra test --no-dev" on Windows-on-ARM. Skip the module rather than
# failing collection there; these tests decode real eventstream frames.
pytest.importorskip("botocore")

from aiperf.transports.aws.eventstream import (
    AwsEventStreamError,
    AwsEventStreamReader,
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
    extra_headers: dict[str, str] | None = None,
) -> bytes:
    """Hand-roll one AWS eventstream binary frame matching the wire format
    AWS runtimes actually emit: 4-byte total length, 4-byte headers
    length, prelude CRC, headers block, payload, message CRC."""
    headers = (
        _encode_header(":message-type", message_type)
        + _encode_header(":event-type", event_type)
        + _encode_header(":content-type", "application/json")
    )
    for name, value in (extra_headers or {}).items():
        headers += _encode_header(name, value)
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


class TestAwsEventStreamReaderBasicDecoding:
    @pytest.mark.asyncio
    async def test_single_payload_part_yields_one_message(self) -> None:
        frame = encode_frame(b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n')

        messages = [m async for m in AwsEventStreamReader(_chunks(frame))]

        assert len(messages) == 1
        assert messages[0].line == ('{"choices":[{"delta":{"content":"Hello"}}]}')

    @pytest.mark.asyncio
    async def test_multiple_payload_parts_yield_multiple_messages(self) -> None:
        frame1 = encode_frame(b'data: {"delta":"tok-a"}\n')
        frame2 = encode_frame(b'data: {"delta":"tok-b"}\n')

        messages = [m async for m in AwsEventStreamReader(_chunks(frame1, frame2))]

        assert [m.line for m in messages] == [
            '{"delta":"tok-a"}',
            '{"delta":"tok-b"}',
        ]

    @pytest.mark.asyncio
    async def test_bare_json_line_without_data_prefix(self) -> None:
        """Some SageMaker containers emit bare JSON PayloadPart lines instead
        of SSE-formatted ``data: ...`` lines; both must decode identically."""
        frame = encode_frame(b'{"delta":"no-prefix"}\n')

        messages = [m async for m in AwsEventStreamReader(_chunks(frame))]

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
            async for m in AwsEventStreamReader(
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

        messages = [m async for m in AwsEventStreamReader(_chunks(frame1, frame2))]

        assert len(messages) == 1
        assert messages[0].line == '{"delta":"reassembled"}'

    @pytest.mark.asyncio
    async def test_trailing_line_without_newline_is_flushed(self) -> None:
        """A final PayloadPart with no trailing newline must still be
        emitted once the stream ends, not silently dropped."""
        frame = encode_frame(b'data: {"delta":"no-trailing-newline"}')

        messages = [m async for m in AwsEventStreamReader(_chunks(frame))]

        assert messages[0].line == '{"delta":"no-trailing-newline"}'

    @pytest.mark.asyncio
    async def test_non_payload_part_events_are_skipped(self) -> None:
        initial_response = encode_frame(b"", event_type="InitialResponse")
        payload = encode_frame(b'data: {"delta":"only-this"}\n')

        messages = [
            m async for m in AwsEventStreamReader(_chunks(initial_response, payload))
        ]

        assert len(messages) == 1
        assert messages[0].line == '{"delta":"only-this"}'

    @pytest.mark.asyncio
    async def test_empty_stream_yields_no_messages(self) -> None:
        messages = [m async for m in AwsEventStreamReader(_chunks())]
        assert messages == []


class TestAwsEventStreamReaderErrorEvents:
    @pytest.mark.asyncio
    async def test_error_message_type_raises(self) -> None:
        frame = encode_frame(b"internal server error", message_type="error")

        with pytest.raises(AwsEventStreamError, match="internal server error"):
            async for _ in AwsEventStreamReader(_chunks(frame)):
                pass

    @pytest.mark.asyncio
    async def test_exception_message_type_raises(self) -> None:
        frame = encode_frame(b"ModelError: bad input", message_type="exception")

        with pytest.raises(AwsEventStreamError, match="ModelError: bad input"):
            async for _ in AwsEventStreamReader(_chunks(frame)):
                pass

    @pytest.mark.asyncio
    async def test_error_after_valid_messages_still_raises(self) -> None:
        good = encode_frame(b'data: {"delta":"ok"}\n')
        bad = encode_frame(b"stream failed", message_type="error")

        received = []
        with pytest.raises(AwsEventStreamError):
            async for message in AwsEventStreamReader(_chunks(good, bad)):
                received.append(message)

        assert len(received) == 1
        assert received[0].line == '{"delta":"ok"}'


class TestAwsEventStreamReaderInspectMessageForError:
    def test_is_a_noop(self) -> None:
        """inspect_message_for_error exists only for interface parity with
        AsyncSSEStreamReader -- AwsEventStreamReader already raises inline, so
        this must never touch the message it's given."""
        message = MagicMock()
        AwsEventStreamReader.inspect_message_for_error(message)
        message.assert_not_called()


class TestAwsEventStreamErrorTyping:
    """AWS labels stream failures in ``:exception-type``/``:error-code``. That
    label has to survive onto the exception, because ``aiohttp_client`` copies
    it into ``ErrorDetails.type`` so the error table can separate a
    ``ModelStreamError`` from an ``InternalStreamFailure`` instead of showing
    one undifferentiated bucket."""

    @pytest.mark.asyncio
    async def test_exception_type_header_is_captured(self) -> None:
        frame = encode_frame(
            b"model blew up",
            message_type="exception",
            extra_headers={":exception-type": "ModelStreamError"},
        )

        with pytest.raises(AwsEventStreamError) as excinfo:
            async for _ in AwsEventStreamReader(_chunks(frame)):
                pass

        assert excinfo.value.exception_type == "ModelStreamError"
        assert "ModelStreamError" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_error_code_header_is_captured(self) -> None:
        """Error frames use ``:error-code`` where exception frames use
        ``:exception-type``; both must land in the same attribute."""
        frame = encode_frame(
            b"",
            message_type="error",
            extra_headers={
                ":error-code": "InternalStreamFailure",
                ":error-message": "backend closed the stream",
            },
        )

        with pytest.raises(AwsEventStreamError) as excinfo:
            async for _ in AwsEventStreamReader(_chunks(frame)):
                pass

        assert excinfo.value.exception_type == "InternalStreamFailure"
        assert "backend closed the stream" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_untyped_error_leaves_exception_type_unset(self) -> None:
        """Without a label there is nothing to group by, so callers fall back
        to the generic class name rather than inventing one."""
        frame = encode_frame(b"something broke", message_type="error")

        with pytest.raises(AwsEventStreamError) as excinfo:
            async for _ in AwsEventStreamReader(_chunks(frame)):
                pass

        assert excinfo.value.exception_type is None


class TestEventStreamContentType:
    def test_mock_server_constant_matches_production(self) -> None:
        """The mock server deliberately redeclares the content type instead of
        importing it, so a change to the production value cannot silently
        propagate into the fixture that is supposed to detect it. This is the
        one place the two copies are compared."""
        from aiperf_mock_server.eventstream import (
            EVENTSTREAM_CONTENT_TYPE as MOCK_CONTENT_TYPE,
        )

        from aiperf.transports.aws.eventstream import EVENTSTREAM_CONTENT_TYPE

        assert MOCK_CONTENT_TYPE == EVENTSTREAM_CONTENT_TYPE


class TestAwsEventStreamMessageProtocolConformance:
    def test_satisfies_inference_server_response(self) -> None:
        """Metrics and every endpoint parser read responses through this
        structural protocol only. That is the whole reason adding SageMaker
        required no parser or metrics changes, so pin it: if the protocol
        gains a member and AwsEventStreamMessage does not, this fails here
        rather than as a mis-parse deep in the pipeline."""
        from aiperf.common.models import AwsEventStreamMessage
        from aiperf.common.models.record_models import InferenceServerResponse

        message = AwsEventStreamMessage(
            perf_ns=1, line='{"a": 1}', raw_line=b'data: {"a": 1}'
        )

        assert isinstance(message, InferenceServerResponse)
        assert message.get_json() == {"a": 1}
        assert message.get_text() == '{"a": 1}'
        assert message.get_raw() == b'data: {"a": 1}'


class TestRawLineFidelity:
    """``raw_line`` is documented as the undecoded bytes exactly as received, and
    ``get_raw()`` feeds raw-record exports. Normalizing it would mean an export
    could not reproduce what the server actually sent."""

    @pytest.mark.asyncio
    async def test_surrounding_whitespace_is_preserved_in_raw_line(self) -> None:
        frame = encode_frame(b'  data: {"a": 1}  \n')

        messages = [m async for m in AwsEventStreamReader(_chunks(frame))]

        assert len(messages) == 1
        # The decoded view is normalized...
        assert messages[0].line == '{"a": 1}'
        # ...while the raw view keeps what arrived.
        assert messages[0].raw_line == b'  data: {"a": 1}  '

    @pytest.mark.asyncio
    async def test_trailing_line_without_newline_keeps_its_bytes(self) -> None:
        """The final flush path builds a message too, and must not normalize
        differently from the main loop."""
        frame = encode_frame(b'  {"b": 2}  ')

        messages = [m async for m in AwsEventStreamReader(_chunks(frame))]

        assert len(messages) == 1
        assert messages[0].line == '{"b": 2}'
        assert messages[0].raw_line == b'  {"b": 2}  '


class TestChunkTimestampSharing:
    """One timestamp per network read, shared by every message decoded from it.

    Deliberate, and matched to what ``sse_utils.py`` does for SSE -- stamping at
    decode time would fold decode latency into ITL and make the two framings
    non-comparable. Pinned here because the existing decode tests happen to put
    each frame in its own chunk, so a refactor to per-frame stamping would not
    have failed anything.
    """

    @pytest.mark.asyncio
    async def test_frames_arriving_in_one_read_share_a_timestamp(self) -> None:
        one_read = encode_frame(b'data: {"i": 1}\n') + encode_frame(b'data: {"i": 2}\n')

        messages = [m async for m in AwsEventStreamReader(_chunks(one_read))]

        assert len(messages) == 2
        assert messages[0].perf_ns == messages[1].perf_ns

    @pytest.mark.asyncio
    async def test_frames_arriving_in_separate_reads_do_not(self) -> None:
        """Guards the test above: equal timestamps must mean 'same read', not
        'the clock is stubbed'."""
        messages = [
            m
            async for m in AwsEventStreamReader(
                _chunks(
                    encode_frame(b'data: {"i": 1}\n'), encode_frame(b'data: {"i": 2}\n')
                )
            )
        ]

        assert len(messages) == 2
        assert messages[0].perf_ns != messages[1].perf_ns


class TestPartialLineOnErrorPath:
    """A line still buffered when an error frame arrives is often the most
    diagnostic thing about a server-side failure, and the clean-EOF path would
    have surfaced it. Raising must not silently discard it."""

    @pytest.mark.asyncio
    async def test_buffered_partial_line_is_reported_with_the_error(self) -> None:
        # Truncated mid-token on purpose. The fragment is deliberately not a
        # near-dictionary word, so spell-checkers do not flag the fixture.
        partial = encode_frame(b'data: {"choices":[{"delta":{"content":"abcd')
        failure = encode_frame(
            b"", message_type="error", extra_headers={":error-code": "ModelStreamError"}
        )

        with pytest.raises(AwsEventStreamError) as excinfo:
            async for _ in AwsEventStreamReader(_chunks(partial, failure)):
                pass

        assert "abcd" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_no_partial_line_leaves_the_message_unchanged(self) -> None:
        """Nothing buffered means nothing appended -- no empty parenthetical."""
        failure = encode_frame(b"boom", message_type="error")

        with pytest.raises(AwsEventStreamError) as excinfo:
            async for _ in AwsEventStreamReader(_chunks(failure)):
                pass

        assert "partial" not in str(excinfo.value).lower()


class TestCorruptFrameHandling:
    """botocore validates framing and CRCs, so a corrupt frame raises one of its
    exceptions. Those should surface as AwsEventStreamError like every other
    stream failure, rather than as a botocore type the error table cannot group.
    """

    @pytest.mark.asyncio
    async def test_a_corrupt_crc_surfaces_as_an_eventstream_error(self) -> None:
        frame = bytearray(encode_frame(b'data: {"a": 1}\n'))
        frame[-1] ^= 0xFF  # break the trailing message CRC

        with pytest.raises(AwsEventStreamError):
            async for _ in AwsEventStreamReader(_chunks(bytes(frame))):
                pass

    @pytest.mark.asyncio
    async def test_a_corrupt_prelude_surfaces_as_an_eventstream_error(self) -> None:
        frame = bytearray(encode_frame(b'data: {"a": 1}\n'))
        frame[0] ^= 0xFF  # break the declared total length

        with pytest.raises(AwsEventStreamError):
            async for _ in AwsEventStreamReader(_chunks(bytes(frame))):
                pass
