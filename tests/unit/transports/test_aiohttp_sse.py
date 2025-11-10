# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Comprehensive unit tests for AsyncSSEStreamReader."""

import time
from collections.abc import AsyncIterator
from unittest.mock import patch

import pytest

from aiperf.common.exceptions import SSEResponseError
from aiperf.common.models import SSEMessage
from aiperf.transports.sse_utils import AsyncSSEStreamReader


class TestAsyncSSEStreamReader:
    """Test suite for AsyncSSEStreamReader class."""

    async def _collect_messages(self, reader: AsyncSSEStreamReader) -> list[SSEMessage]:
        """Helper to collect all messages from async iteration."""
        messages = []
        async for message in reader:
            messages.append(message)
        return messages

    async def _create_byte_iterator(self, chunks: list[bytes]) -> AsyncIterator[bytes]:
        """Helper to create async byte iterator from list of chunks."""
        for chunk in chunks:
            yield chunk

    def test_init_stores_async_iter(self) -> None:
        """Test that initialization properly stores the async iterator."""

        async def mock_iter():
            yield b"test"

        iter_obj = mock_iter()
        reader = AsyncSSEStreamReader(iter_obj)
        assert reader._async_iter is iter_obj

    @pytest.mark.asyncio
    async def test_read_complete_stream_success(self) -> None:
        """Test successful reading of complete SSE stream."""
        # Create SSE stream with multiple messages
        chunks = [
            b"data: Hello\nevent: message\n\n",
            b"data: World\nid: msg-2\n\n",
        ]

        reader = AsyncSSEStreamReader(self._create_byte_iterator(chunks))
        messages = await reader.read_complete_stream()

        assert len(messages) == 2
        assert all(isinstance(msg, SSEMessage) for msg in messages)

        # Check first message
        assert len(messages[0].packets) == 2
        assert messages[0].packets[0].name == "data"
        assert messages[0].packets[0].value == "Hello"
        assert messages[0].packets[1].name == "event"
        assert messages[0].packets[1].value == "message"
        assert messages[0].perf_ns > 0

        # Check second message
        assert len(messages[1].packets) == 2
        assert messages[1].packets[0].name == "data"
        assert messages[1].packets[0].value == "World"
        assert messages[1].packets[1].name == "id"
        assert messages[1].packets[1].value == "msg-2"
        assert messages[1].perf_ns > 0

    @pytest.mark.asyncio
    async def test_read_complete_stream_empty(self) -> None:
        """Test reading empty SSE stream."""
        reader = AsyncSSEStreamReader(self._create_byte_iterator([]))
        messages = await reader.read_complete_stream()
        assert messages == []

    @pytest.mark.asyncio
    async def test_aiter_single_message(self) -> None:
        """Test __aiter__ with single message."""
        chunks = [b"data: Hello\n\n"]

        reader = AsyncSSEStreamReader(self._create_byte_iterator(chunks))
        messages = await self._collect_messages(reader)

        assert len(messages) == 1
        assert isinstance(messages[0], SSEMessage)
        assert messages[0].packets[0].name == "data"
        assert messages[0].packets[0].value == "Hello"
        assert messages[0].perf_ns > 0

    @pytest.mark.asyncio
    async def test_aiter_multiple_messages(self) -> None:
        """Test __aiter__ with multiple messages in separate chunks."""
        chunks = [
            b"data: Hello\nevent: message\n\n",
            b"data: World\nid: msg-2\n\n",
            b"data: [DONE]\n\n",
        ]

        reader = AsyncSSEStreamReader(self._create_byte_iterator(chunks))
        messages = await self._collect_messages(reader)

        assert len(messages) == 3

        expected_data_values = ["Hello", "World", "[DONE]"]
        for i, expected_value in enumerate(expected_data_values):
            assert messages[i].packets[0].name == "data"
            assert messages[i].packets[0].value == expected_value

    @pytest.mark.asyncio
    async def test_aiter_multiple_messages_in_single_chunk(self) -> None:
        """Test __aiter__ with multiple messages in a single chunk."""
        chunks = [b"data: Hello\n\ndata: World\n\n"]

        reader = AsyncSSEStreamReader(self._create_byte_iterator(chunks))
        messages = await self._collect_messages(reader)

        assert len(messages) == 2
        assert messages[0].packets[0].value == "Hello"
        assert messages[1].packets[0].value == "World"

    @pytest.mark.asyncio
    async def test_aiter_message_split_across_chunks(self) -> None:
        """Test __aiter__ with message split across multiple chunks."""
        chunks = [
            b"data: Hel",
            b"lo\n\n",
        ]

        reader = AsyncSSEStreamReader(self._create_byte_iterator(chunks))
        messages = await self._collect_messages(reader)

        assert len(messages) == 1
        assert messages[0].packets[0].value == "Hello"

    @pytest.mark.asyncio
    async def test_aiter_skips_empty_messages(self) -> None:
        """Test __aiter__ skips empty messages."""
        chunks = [
            b"\n\n",
            b"data: Hello\n\n",
            b"\n\n",
        ]

        reader = AsyncSSEStreamReader(self._create_byte_iterator(chunks))
        messages = await self._collect_messages(reader)

        assert len(messages) == 1
        assert messages[0].packets[0].value == "Hello"

    @pytest.mark.asyncio
    async def test_aiter_handles_comments(self) -> None:
        """Test __aiter__ properly parses comment lines."""
        chunks = [b": This is a comment\ndata: Hello\n\n"]

        reader = AsyncSSEStreamReader(self._create_byte_iterator(chunks))
        messages = await self._collect_messages(reader)

        assert len(messages) == 1
        assert any(packet.name == "comment" for packet in messages[0].packets)
        data_packet = next(p for p in messages[0].packets if p.name == "data")
        assert data_packet.value == "Hello"

    @pytest.mark.asyncio
    async def test_aiter_handles_multiline_data(self) -> None:
        """Test __aiter__ with multiline data fields."""
        chunks = [b"data: line1\ndata: line2\ndata: line3\n\n"]

        reader = AsyncSSEStreamReader(self._create_byte_iterator(chunks))
        messages = await self._collect_messages(reader)

        assert len(messages) == 1
        data_packets = [p for p in messages[0].packets if p.name == "data"]
        assert len(data_packets) == 3
        assert data_packets[0].value == "line1"
        assert data_packets[1].value == "line2"
        assert data_packets[2].value == "line3"

    @pytest.mark.asyncio
    async def test_aiter_handles_unicode(self) -> None:
        """Test __aiter__ handles unicode content."""
        chunks = [
            b"data: \xe4\xbd\xa0\xe5\xa5\xbd\xe4\xb8\x96\xe7\x95\x8c \xf0\x9f\x9a\x80\xf0\x9f\x92\xbb\n\n"
        ]

        reader = AsyncSSEStreamReader(self._create_byte_iterator(chunks))
        messages = await self._collect_messages(reader)

        assert len(messages) == 1
        assert messages[0].packets[0].value == "你好世界 🚀💻"

    @pytest.mark.asyncio
    async def test_aiter_handles_invalid_utf8(self) -> None:
        """Test __aiter__ handles invalid UTF-8 with replacement character."""
        chunks = [b"data: \xff\xfe\xfd invalid\n\n"]

        reader = AsyncSSEStreamReader(self._create_byte_iterator(chunks))
        messages = await self._collect_messages(reader)

        assert len(messages) == 1
        assert "�" in messages[0].packets[0].value

    @pytest.mark.asyncio
    async def test_aiter_timing_precision(self) -> None:
        """Test that __aiter__ captures timing with nanosecond precision."""
        chunks = [b"data: test\n\n"]

        with patch("time.perf_counter_ns") as mock_time:
            mock_time.return_value = 123456789

            reader = AsyncSSEStreamReader(self._create_byte_iterator(chunks))
            messages = await self._collect_messages(reader)

            assert len(messages) == 1
            assert messages[0].perf_ns == 123456789

    @pytest.mark.asyncio
    async def test_aiter_timing_multiple_messages(self) -> None:
        """Test that each message gets its own timestamp."""
        chunks = [
            b"data: first\n\n",
            b"data: second\n\n",
        ]

        reader = AsyncSSEStreamReader(self._create_byte_iterator(chunks))
        messages = await self._collect_messages(reader)

        assert len(messages) == 2
        assert messages[1].perf_ns >= messages[0].perf_ns

    @pytest.mark.asyncio
    async def test_aiter_handles_message_without_final_delimiter(self) -> None:
        """Test __aiter__ handles final message without \n\n delimiter."""
        chunks = [b"data: complete\n\ndata: incomplete"]

        reader = AsyncSSEStreamReader(self._create_byte_iterator(chunks))
        messages = await self._collect_messages(reader)

        assert len(messages) == 2
        assert messages[0].packets[0].value == "complete"
        assert messages[1].packets[0].value == "incomplete"

    @pytest.mark.asyncio
    async def test_aiter_all_sse_field_types(self) -> None:
        """Test __aiter__ with all SSE field types."""
        chunks = [b"data: test\nevent: custom\nid: msg-123\nretry: 5000\n: comment\n\n"]

        reader = AsyncSSEStreamReader(self._create_byte_iterator(chunks))
        messages = await self._collect_messages(reader)

        assert len(messages) == 1
        packets = messages[0].packets
        field_names = {p.name for p in packets}
        assert "data" in field_names
        assert "event" in field_names
        assert "id" in field_names
        assert "retry" in field_names
        assert "comment" in field_names

    @pytest.mark.asyncio
    async def test_aiter_complex_json_data(self) -> None:
        """Test __aiter__ with complex JSON in data field."""
        json_data = '{"id":"chatcmpl-123","object":"chat.completion.chunk","choices":[{"delta":{"content":"Hello"}}]}'
        chunks = [f"data: {json_data}\n\n".encode()]

        reader = AsyncSSEStreamReader(self._create_byte_iterator(chunks))
        messages = await self._collect_messages(reader)

        assert len(messages) == 1
        assert messages[0].packets[0].value == json_data

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_sse_stream_performance(self) -> None:
        """Test SSE stream reading performance with large number of messages."""
        num_messages = 10000

        # Create chunks with multiple messages each for efficiency
        chunks = []
        messages_per_chunk = 100
        for i in range(0, num_messages, messages_per_chunk):
            chunk = b""
            for j in range(messages_per_chunk):
                chunk += f"data: Message {i + j}\n\n".encode()
            chunks.append(chunk)

        reader = AsyncSSEStreamReader(self._create_byte_iterator(chunks))

        start_time = time.perf_counter()
        messages = await self._collect_messages(reader)
        end_time = time.perf_counter()
        processing_time = end_time - start_time

        assert len(messages) == num_messages
        assert processing_time < 3.0, (
            f"Processing took {processing_time:.3f}s, expected < 3s"
        )

    @pytest.mark.asyncio
    async def test_aiter_empty_chunks(self) -> None:
        """Test __aiter__ handles empty chunks gracefully."""
        chunks = [
            b"",
            b"data: test\n\n",
            b"",
        ]

        reader = AsyncSSEStreamReader(self._create_byte_iterator(chunks))
        messages = await self._collect_messages(reader)

        assert len(messages) == 1
        assert messages[0].packets[0].value == "test"

    @pytest.mark.asyncio
    async def test_aiter_only_whitespace_chunks(self) -> None:
        """Test __aiter__ handles whitespace-only chunks."""
        chunks = [
            b"   ",
            b"data: test\n\n",
            b"\t\t",
        ]

        reader = AsyncSSEStreamReader(self._create_byte_iterator(chunks))
        messages = await self._collect_messages(reader)

        assert len(messages) == 1
        assert messages[0].packets[0].value == "test"

    @pytest.mark.asyncio
    async def test_aiter_byte_by_byte(self) -> None:
        """Test __aiter__ can handle byte-by-byte streaming."""
        message = b"data: Hello\n\n"
        chunks = [bytes([b]) for b in message]

        reader = AsyncSSEStreamReader(self._create_byte_iterator(chunks))
        messages = await self._collect_messages(reader)

        assert len(messages) == 1
        assert messages[0].packets[0].value == "Hello"

    @pytest.mark.asyncio
    async def test_read_complete_stream_with_exception_in_iterator(self) -> None:
        """Test that exceptions in the async iterator propagate correctly."""

        async def failing_iterator():
            yield b"data: test\n\n"
            raise ValueError("Iterator failed")

        reader = AsyncSSEStreamReader(failing_iterator())

        with pytest.raises(ValueError, match="Iterator failed"):
            await reader.read_complete_stream()

    @pytest.mark.asyncio
    async def test_aiter_crlf_delimiter(self) -> None:
        """Test __aiter__ with CRLF CRLF delimiter (spec-compliant)."""
        chunks = [b"data: Hello\r\nevent: message\r\n\r\n"]

        reader = AsyncSSEStreamReader(self._create_byte_iterator(chunks))
        messages = await self._collect_messages(reader)

        assert len(messages) == 1
        assert messages[0].packets[0].name == "data"
        assert messages[0].packets[0].value == "Hello"
        assert messages[0].packets[1].name == "event"
        assert messages[0].packets[1].value == "message"

    @pytest.mark.asyncio
    async def test_aiter_multiple_messages_crlf(self) -> None:
        """Test __aiter__ with multiple CRLF-delimited messages."""
        chunks = [
            b"data: Hello\r\n\r\n",
            b"data: World\r\n\r\n",
            b"data: [DONE]\r\n\r\n",
        ]

        reader = AsyncSSEStreamReader(self._create_byte_iterator(chunks))
        messages = await self._collect_messages(reader)

        assert len(messages) == 3
        assert messages[0].packets[0].value == "Hello"
        assert messages[1].packets[0].value == "World"
        assert messages[2].packets[0].value == "[DONE]"

    @pytest.mark.asyncio
    async def test_aiter_mixed_delimiters(self) -> None:
        """Test __aiter__ with mixed CRLF and LF delimiters."""
        chunks = [
            b"data: First\r\n\r\n",  # CRLF delimiter
            b"data: Second\n\n",  # LF delimiter
            b"data: Third\r\n\r\n",  # CRLF delimiter
        ]

        reader = AsyncSSEStreamReader(self._create_byte_iterator(chunks))
        messages = await self._collect_messages(reader)

        assert len(messages) == 3
        assert messages[0].packets[0].value == "First"
        assert messages[1].packets[0].value == "Second"
        assert messages[2].packets[0].value == "Third"

    @pytest.mark.asyncio
    async def test_aiter_crlf_split_across_chunks(self) -> None:
        """Test __aiter__ with CRLF delimiter split across chunks."""
        chunks = [
            b"data: Hello\r\n\r",
            b"\ndata: World\r\n\r\n",
        ]

        reader = AsyncSSEStreamReader(self._create_byte_iterator(chunks))
        messages = await self._collect_messages(reader)

        assert len(messages) == 2
        assert messages[0].packets[0].value == "Hello"
        assert messages[1].packets[0].value == "World"

    @pytest.mark.asyncio
    async def test_aiter_crlf_multiple_messages_in_single_chunk(self) -> None:
        """Test __aiter__ with multiple CRLF messages in single chunk."""
        chunks = [b"data: First\r\n\r\ndata: Second\r\n\r\ndata: Third\r\n\r\n"]

        reader = AsyncSSEStreamReader(self._create_byte_iterator(chunks))
        messages = await self._collect_messages(reader)

        assert len(messages) == 3
        assert messages[0].packets[0].value == "First"
        assert messages[1].packets[0].value == "Second"
        assert messages[2].packets[0].value == "Third"

    @pytest.mark.asyncio
    async def test_aiter_crlf_multiline_data(self) -> None:
        """Test __aiter__ with CRLF-separated multiline data fields."""
        chunks = [b"data: line1\r\ndata: line2\r\ndata: line3\r\n\r\n"]

        reader = AsyncSSEStreamReader(self._create_byte_iterator(chunks))
        messages = await self._collect_messages(reader)

        assert len(messages) == 1
        data_packets = [p for p in messages[0].packets if p.name == "data"]
        assert len(data_packets) == 3
        assert data_packets[0].value == "line1"
        assert data_packets[1].value == "line2"
        assert data_packets[2].value == "line3"

    @pytest.mark.asyncio
    async def test_aiter_crlf_all_field_types(self) -> None:
        """Test __aiter__ with CRLF and all SSE field types."""
        chunks = [
            b"data: test\r\nevent: custom\r\nid: msg-123\r\nretry: 5000\r\n: comment\r\n\r\n"
        ]  # fmt: skip

        reader = AsyncSSEStreamReader(self._create_byte_iterator(chunks))
        messages = await self._collect_messages(reader)

        assert len(messages) == 1
        packets = messages[0].packets
        field_names = {p.name for p in packets}
        assert "data" in field_names
        assert "event" in field_names
        assert "id" in field_names
        assert "retry" in field_names
        assert "comment" in field_names

    @pytest.mark.asyncio
    async def test_aiter_crlf_without_final_delimiter(self) -> None:
        """Test __aiter__ handles CRLF message without final delimiter."""
        chunks = [b"data: complete\r\n\r\ndata: incomplete"]

        reader = AsyncSSEStreamReader(self._create_byte_iterator(chunks))
        messages = await self._collect_messages(reader)

        assert len(messages) == 2
        assert messages[0].packets[0].value == "complete"
        assert messages[1].packets[0].value == "incomplete"

    @pytest.mark.asyncio
    async def test_aiter_crlf_real_world_openai_format(self) -> None:
        """Test __aiter__ with real-world OpenAI SSE format."""
        chunks = [
            b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4","choices":[{"index":0,"delta":{"content":"Hello"}}]}\r\n\r\n',
            b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4","choices":[{"index":0,"delta":{"content":" World"}}]}\r\n\r\n',
            b"data: [DONE]\r\n\r\n",
        ]

        reader = AsyncSSEStreamReader(self._create_byte_iterator(chunks))
        messages = await self._collect_messages(reader)

        assert len(messages) == 3
        assert '"content":"Hello"' in messages[0].packets[0].value
        assert '"content":" World"' in messages[1].packets[0].value
        assert messages[2].packets[0].value == "[DONE]"

    @pytest.mark.asyncio
    async def test_aiter_crlf_byte_by_byte(self) -> None:
        """Test __aiter__ can handle CRLF messages byte-by-byte."""
        message = b"data: Hello\r\n\r\n"
        chunks = [bytes([b]) for b in message]

        reader = AsyncSSEStreamReader(self._create_byte_iterator(chunks))
        messages = await self._collect_messages(reader)

        assert len(messages) == 1
        assert messages[0].packets[0].value == "Hello"

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_aiter_crlf_performance(self) -> None:
        """Test CRLF parsing performance with large number of messages."""
        num_messages = 10000

        # Create chunks with multiple CRLF messages each
        chunks = []
        messages_per_chunk = 100
        for i in range(0, num_messages, messages_per_chunk):
            chunk = b""
            for j in range(messages_per_chunk):
                chunk += f"data: Message {i + j}\r\n\r\n".encode()
            chunks.append(chunk)

        reader = AsyncSSEStreamReader(self._create_byte_iterator(chunks))

        start_time = time.perf_counter()
        messages = await self._collect_messages(reader)
        end_time = time.perf_counter()
        processing_time = end_time - start_time

        assert len(messages) == num_messages
        assert processing_time < 3.0, (
            f"CRLF processing took {processing_time:.3f}s, expected < 3s"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "chunks,expected_error",
        [
            ([b"data: Normal message\n\n", b"event: error\n: Rate limit\ndata: {}\n\n"], "Rate limit"),
            ([b"event: error\ndata: Something went wrong\n\n"], "Unknown error in SSE response"),
            ([b"event: error\r\n: Server error\r\ndata: {}\r\n\r\n"], "Server error"),
            ([b"event: error\n: Connection timeout\n\n"], "Connection timeout"),
            ([b"data: Message 1\n\n", b"data: Message 2\n\n", b"event: error\n: Fatal error\n\n"], "Fatal error"),
            ([b'event: error\n: Internal error\ndata: {"error_code": 500}\n\n'], "Internal error"),
        ],
    )  # fmt: skip
    async def test_error_events_raise_in_read_complete_stream(
        self, chunks: list[bytes], expected_error: str
    ) -> None:
        """Test that various error events raise SSEResponseError."""
        reader = AsyncSSEStreamReader(self._create_byte_iterator(chunks))

        with pytest.raises(SSEResponseError) as exc_info:
            await reader.read_complete_stream()

        assert expected_error in str(exc_info.value)
        assert exc_info.value.error_code == 502

    @pytest.mark.asyncio
    async def test_error_in_manual_iteration_with_inspect(self) -> None:
        """Test that manual iteration with inspect raises on error event."""
        chunks = [
            b"data: First message\n\n",
            b"event: error\n: Authentication failed\n\n",
            b"data: Should not reach\n\n",
        ]

        reader = AsyncSSEStreamReader(self._create_byte_iterator(chunks))
        messages = []

        with pytest.raises(SSEResponseError) as exc_info:
            async for message in reader:
                AsyncSSEStreamReader.inspect_message_for_error(message)
                messages.append(message)

        assert len(messages) == 1
        assert "Authentication failed" in str(exc_info.value)
