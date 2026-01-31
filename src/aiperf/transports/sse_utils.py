# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import time
from collections.abc import AsyncIterator

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.enums import SSEEventType, SSEFieldType
from aiperf.common.exceptions import SSEResponseError
from aiperf.common.models import SSEMessage

_logger = AIPerfLogger(__name__)


class AsyncSSEStreamReader:
    """Parse Server-Sent Events (SSE) stream with per-message timestamps.

    Parsing logic based on the official HTML SSE Living Standard:
    https://html.spec.whatwg.org/multipage/server-sent-events.html#parsing-an-event-stream

    This class can be used to read an SSE stream incrementally, parsing individual messages
    as they arrive from the server. Each message will receive its own timestamp for
    accurate Time-To-First-Token (TTFT) and Inter-Chunk-Latency (ICL) measurements.

    SSE Format:
        Server-Sent Events are text-based, with messages delimited by double newlines.
        Supports both \r\n\r\n and \n\n delimiters:

        data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1749678185,"model":"gpt2","choices":[{"index":0,"delta":{"content":"Hello","tool_calls":[]}}]}

        data: {"id":"chatcmpl-2","object":"chat.completion.chunk","created":1749678185,"model":"gpt2","choices":[{"index":0,"delta":{"content":" World","tool_calls":[]},"finish_reason":"length"}]}

        data: [DONE]

    Parsing Strategy:
        1. Read response in chunks
        2. Accumulate chunks in buffer until delimiter found (\r\n\r\n or \n\n)
        3. Parse complete message using SSEMessage.parse()
        4. Timestamp message at arrival time
        5. Repeat until stream ends

    Args:
        async_iter: Async iterator that yields bytes objects of the raw SSE message.

    Returns:
        Async iterator of SSEMessage objects, each containing:
            - perf_ns: Timestamp when message arrived (nanoseconds)
            - packets: List of SSEField objects, each containing:
                - name: Name of the field (e.g. "data", "event", "id", "retry", "comment")
                - value: Value of the field

    Memory Efficiency:
        The buffer is trimmed after each message is parsed, keeping memory usage
        bounded even for very long SSE streams. Peak memory is approximately:
            buffer_size + chunk_size ≈ typical_message_size + async_iter chunk size

    Error Handling:
        - Unicode decode errors use 'replace' strategy (invalid bytes -> �)
        - Malformed messages are parsed as-is (SSEMessage.parse is permissive, so it will not raise an exception)
        - Empty messages are skipped

    Performance:
        - Incremental parsing minimizes latency (messages available as they arrive)
        - Chunk-based reading is memory efficient
        - Per-message timestamps enable accurate token-level timing
    """

    def __init__(self, async_iter: AsyncIterator[bytes]):
        self._async_iter = async_iter

    async def read_complete_stream(self) -> list[SSEMessage]:
        """Read the complete SSE stream and return a list of SSE messages."""
        messages: list[SSEMessage] = []
        async for message in self:
            AsyncSSEStreamReader.inspect_message_for_error(message)
            messages.append(message)
        return messages

    @staticmethod
    def inspect_message_for_error(message: SSEMessage):
        """Check if the message contains an error event packet and raise an SSEResponseError if so.

        If so, look for any comment field and raise an SSEResponseError
        with that comment as the error message, otherwise use the full message.
        """
        has_error_event = any(
            packet.name == SSEFieldType.EVENT and packet.value == SSEEventType.ERROR
            for packet in message.packets
        )

        if has_error_event:
            error_message = None
            for packet in message.packets:
                if packet.name == SSEFieldType.COMMENT:
                    error_message = packet.value
                    break

            if error_message is None:
                error_message = (
                    f"Unknown error in SSE response: {message.model_dump_json()}"
                )

            raise SSEResponseError(
                f"Error occurred in SSE response: {error_message}", error_code=502
            )

    async def __aiter__(self) -> AsyncIterator[SSEMessage]:
        """Iterate over the SSE stream in a performant manner and yield parsed SSE messages as they arrive."""

        # Use bytearray for efficient buffer operations (mutable, no copy overhead)
        buffer = bytearray()

        # Stream response body incrementally from the async iterator
        async for chunk in self._async_iter:
            # Capture timestamp immediately when chunk arrives
            # This will provide us with the most accurate TTFT and ICL measurements
            chunk_perf_ns = time.perf_counter_ns()

            # bytearray is mutable, no copy overhead, so we can append the chunk to the buffer in-place.
            buffer += chunk

            # Parse complete messages from buffer.
            # SSE spec requires "\r\n\r\n" (CRLF CRLF) but we support both "\r\n\r\n" and "\n\n"
            # for compatibility with lenient servers.
            while True:
                # Try to find "\r\n\r\n" first (spec-compliant delimiter)
                delimiter_index = buffer.find(b"\r\n\r\n")
                delimiter_length = 4

                if delimiter_index == -1:
                    # Fall back to "\n\n" for lenient servers
                    delimiter_index = buffer.find(b"\n\n")
                    delimiter_length = 2

                if delimiter_index == -1:
                    # No complete message found, wait for more data
                    break

                # Extract message bytes up to delimiter
                message_bytes = bytes(buffer[:delimiter_index])
                # Remove processed message + delimiter from buffer in-place
                del buffer[: delimiter_index + delimiter_length]

                raw_message = message_bytes.decode("utf-8", errors="replace").strip()
                if not raw_message:
                    if _logger.is_trace_enabled:
                        _logger.trace(
                            f"Skipping empty SSE message at chunk {chunk_perf_ns}"
                        )
                    continue

                yield SSEMessage.parse(raw_message, chunk_perf_ns)

                if _logger.is_trace_enabled:
                    _logger.trace(f"Parsed SSE message: {raw_message}...")

        # Handle any remaining data in buffer after stream ends
        # Some servers don't send final delimiter
        if buffer_remaining := buffer.strip():
            final_perf_ns = time.perf_counter_ns()
            raw_message = buffer_remaining.decode("utf-8", errors="replace")
            yield SSEMessage.parse(raw_message, final_perf_ns)

            if _logger.is_trace_enabled:
                _logger.trace(f"Parsed final SSE message: {raw_message}...")
