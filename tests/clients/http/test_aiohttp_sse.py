# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Comprehensive unit tests for aiohttp client components."""

################################################################################
# Test AioHttpSSEStreamReader
################################################################################

import time
from unittest.mock import AsyncMock, Mock, patch

import pytest

from aiperf.common.models import SSEMessage
from aiperf.inference.clients.http.aiohttp_client import AioHttpSSEStreamReader
from tests.clients.http.conftest import (
    create_sse_chunk_list,
    setup_single_sse_chunk,
    setup_sse_content_mock,
)


class TestAioHttpSSEStreamReader:
    """Test suite for AioHttpSSEStreamReader class."""

    async def _collect_chunks(
        self, reader: AioHttpSSEStreamReader
    ) -> list[tuple[str, int]]:
        """Helper to collect all chunks from async iteration."""
        chunks = []
        async for chunk in reader:
            chunks.append(chunk)
        return chunks

    def _mock_timestamps(self, count: int, start: int = 123456789) -> list[int]:
        """Helper to generate mock timestamps."""
        return list(range(start, start + count))

    def test_init_stores_response(self, mock_sse_response: Mock) -> None:
        """Test that initialization properly stores the response object."""
        reader = AioHttpSSEStreamReader(mock_sse_response)
        assert reader.response == mock_sse_response

    @pytest.mark.asyncio
    async def test_read_complete_stream_success(self, mock_sse_response: Mock) -> None:
        """Test successful reading of complete SSE stream."""
        # Setup mock to simulate stream data
        sse_data = [
            ("data: Hello\nevent: message", 123456789),
            ("data: World\nid: msg-2", 123456791),
        ]

        async def mock_aiter():
            for data in sse_data:
                yield data

        with (
            patch.object(
                AioHttpSSEStreamReader, "__aiter__", return_value=mock_aiter()
            ),
            patch("aiperf.clients.http.aiohttp_client.parse_sse_message") as mock_parse,
        ):
            mock_messages = [
                SSEMessage(perf_ns=123456789),
                SSEMessage(perf_ns=123456791),
            ]
            mock_parse.side_effect = mock_messages

            reader = AioHttpSSEStreamReader(mock_sse_response)
            result = await reader.read_complete_stream()

            assert len(result) == 2
            assert all(isinstance(msg, SSEMessage) for msg in result)
            assert mock_parse.call_count == 2

    @pytest.mark.asyncio
    async def test_read_complete_stream_empty(self, mock_sse_response: Mock) -> None:
        """Test reading empty SSE stream."""

        async def mock_aiter():
            return
            yield  # This will never be reached

        with patch.object(
            AioHttpSSEStreamReader, "__aiter__", return_value=mock_aiter()
        ):
            reader = AioHttpSSEStreamReader(mock_sse_response)
            result = await reader.read_complete_stream()

            assert result == []

    @pytest.mark.asyncio
    async def test_aiter_single_chunk(self, mock_sse_response: Mock) -> None:
        """Test __aiter__ with single chunk."""
        setup_single_sse_chunk(mock_sse_response, remaining=b"ata: Hello\n\n")
        with patch("time.perf_counter_ns", side_effect=[123456789]):
            reader = AioHttpSSEStreamReader(mock_sse_response)
            chunks = await self._collect_chunks(reader)
            assert len(chunks) == 1
            raw_message, first_byte_ns = chunks[0]
            assert raw_message == "data: Hello"
            assert first_byte_ns == 123456789

    @pytest.mark.asyncio
    async def test_aiter_multiple_chunks(
        self, mock_sse_response: Mock, sample_sse_chunks: list[tuple[bytes, bytes]]
    ) -> None:
        """Test __aiter__ with multiple chunks."""
        setup_sse_content_mock(mock_sse_response, sample_sse_chunks)

        reader = AioHttpSSEStreamReader(mock_sse_response)
        chunks = await self._collect_chunks(reader)

        assert len(chunks) == 3

        expected_messages = [
            "data: Hello\nevent: message",
            "data: World\nid: msg-2",
            "data: [DONE]",
        ]

        for i, (raw_message, _) in enumerate(chunks):
            assert raw_message == expected_messages[i]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "first_byte,remaining,expected_count,description",
        [
            (b"", b"", 0, "empty chunks"),
            (b"\xff", b"\xfe\xfd invalid utf8\n\n", 1, "unicode decode error"),
            (b"d", b"ata: test\n\n", 1, "timing accuracy"),
        ],
    )
    async def test_aiter_edge_cases(
        self,
        mock_sse_response: Mock,
        first_byte: bytes,
        remaining: bytes,
        expected_count: int,
        description: str,
    ) -> None:
        """Test __aiter__ edge cases with parameterized inputs."""
        setup_single_sse_chunk(
            mock_sse_response, first_byte=first_byte, remaining=remaining
        )

        timestamps = self._mock_timestamps(2)
        with patch("time.perf_counter_ns", side_effect=timestamps):
            reader = AioHttpSSEStreamReader(mock_sse_response)
            chunks = await self._collect_chunks(reader)

            assert len(chunks) == expected_count, f"Failed for {description}"

            if expected_count > 0:
                raw_message, first_byte_ns = chunks[0]
                assert isinstance(raw_message, str)
                assert first_byte_ns == timestamps[0]

                # Special assertions for unicode decode error
                if "unicode" in description:
                    assert "�" in raw_message  # Unicode replacement character

    @pytest.mark.asyncio
    async def test_aiter_timing_precision(self, mock_sse_response: Mock) -> None:
        """Test that __aiter__ captures timing with specific precision."""
        setup_single_sse_chunk(mock_sse_response, remaining=b"ata: test\n\n")

        first_timestamp = 123456789
        second_timestamp = 123456990
        timestamps = [first_timestamp, second_timestamp]

        with patch("time.perf_counter_ns", side_effect=timestamps):
            reader = AioHttpSSEStreamReader(mock_sse_response)
            chunks = await self._collect_chunks(reader)

            assert len(chunks) == 1
            _, first_byte_ns = chunks[0]
            assert first_byte_ns == first_timestamp

    @pytest.mark.asyncio
    async def test_sse_stream_performance(self, mock_sse_response: Mock) -> None:
        """Test SSE stream reading performance with large number of messages."""
        num_messages = 10000

        messages = [f"Message {i}" for i in range(num_messages)]
        sse_chunks = create_sse_chunk_list(messages)

        setup_sse_content_mock(mock_sse_response, sse_chunks)

        reader = AioHttpSSEStreamReader(mock_sse_response)

        start_time = time.perf_counter()
        chunks = await self._collect_chunks(reader)
        end_time = time.perf_counter()
        processing_time = end_time - start_time

        assert len(chunks) == num_messages
        assert processing_time < 3.0, (
            f"Processing took {processing_time:.3f}s, expected < 3s"
        )

    @pytest.mark.asyncio
    async def test_malformed_sse_stream(self, mock_sse_response: Mock) -> None:
        """Test handling of malformed SSE stream."""
        stream_exception = Exception("Stream corruption")
        content_mock = Mock()
        content_mock.at_eof.side_effect = [False, True]
        content_mock.read = AsyncMock(return_value=b"d")
        content_mock.readuntil = AsyncMock(side_effect=stream_exception)
        mock_sse_response.content = content_mock

        reader = AioHttpSSEStreamReader(mock_sse_response)

        with pytest.raises(Exception, match="Stream corruption"):
            async for _ in reader:
                pass
