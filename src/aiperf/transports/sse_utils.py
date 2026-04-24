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
        Supports both \n\n and \r\n\r\n delimiters:

        data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1749678185,"model":"gpt2","choices":[{"index":0,"delta":{"content":"Hello","tool_calls":[]}}]}

        data: {"id":"chatcmpl-2","object":"chat.completion.chunk","created":1749678185,"model":"gpt2","choices":[{"index":0,"delta":{"content":" World","tool_calls":[]},"finish_reason":"length"}]}

        data: [DONE]

    Parsing Strategy:
        1. Read response in chunks
        2. Accumulate chunks in buffer until delimiter found (\n\n or \r\n\r\n)
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
                error_message = f"Unknown error in SSE response: {message}"

            raise SSEResponseError(
                f"Error occurred in SSE response: {error_message}", error_code=502
            )

    async def __aiter__(self) -> AsyncIterator[SSEMessage]:
        """Iterate over the SSE stream in a performant manner and yield parsed SSE messages as they arrive."""

        # Use bytearray for efficient buffer operations (mutable, no copy overhead).
        # We track a `consumed` offset instead of deleting from the front on every
        # message, which avoids an O(remaining) memmove per message. The buffer is
        # compacted once per chunk instead.
        buffer = bytearray()
        consumed = 0
        # Where to start the next delimiter search (avoids re-scanning checked bytes).
        search_offset = 0

        # Stream response body incrementally from the async iterator
        async for chunk in self._async_iter:
            # Capture timestamp immediately when chunk arrives
            # This will provide us with the most accurate TTFT and ICL measurements
            chunk_perf_ns = time.perf_counter_ns()

            buffer += chunk

            # Parse complete messages from buffer.
            while True:
                # A delimiter can span two chunks: all but its last byte may
                # already be in the buffer from the previous chunk. We must back
                # up by the length of the longest delimiter (\r\n\r\n) minus 1,
                # i.e. 3 bytes, to re-scan that overlap region.
                scan_from = max(consumed, search_offset - 3)

                # Only \n\n and \r\n\r\n are checked. The spec also allows \r\r
                # and mixed combos like \n\r\n, but no real LLM server produces those.
                #
                # We check \n\n first (more common with LLM servers) and only fall
                # back to \r\n\r\n when absent. This is deliberate: a server uses
                # one delimiter style for the entire stream, so searching for both and
                # picking the earliest would double the C-level find() cost on every
                # iteration for no practical correctness gain. Profiling showed this
                # search dominating CPU time against fast mock servers with high OSL.
                delimiter_index = buffer.find(b"\n\n", scan_from)
                delimiter_length = 2

                if delimiter_index == -1:
                    delimiter_index = buffer.find(b"\r\n\r\n", scan_from)
                    delimiter_length = 4

                if delimiter_index == -1:
                    search_offset = len(buffer)
                    break

                raw_message = (
                    buffer[consumed:delimiter_index]
                    .decode("utf-8", errors="replace")
                    .strip()
                )
                consumed = delimiter_index + delimiter_length
                search_offset = consumed

                if not raw_message:
                    if _logger.is_trace_enabled:
                        _logger.trace(
                            f"Skipping empty SSE message at chunk {chunk_perf_ns}"
                        )
                    continue

                yield SSEMessage.parse(raw_message, chunk_perf_ns)

                if _logger.is_trace_enabled:
                    _logger.trace(f"Parsed SSE message: {raw_message}...")

            # Compact: discard consumed bytes once per chunk instead of per message
            if consumed > 0:
                del buffer[:consumed]
                search_offset -= consumed
                consumed = 0

        # Handle any remaining data in buffer after stream ends
        # Some servers don't send final delimiter
        remaining = buffer[consumed:].strip()
        if remaining:
            final_perf_ns = time.perf_counter_ns()
            raw_message = remaining.decode("utf-8", errors="replace")
            yield SSEMessage.parse(raw_message, final_perf_ns)

            if _logger.is_trace_enabled:
                _logger.trace(f"Parsed final SSE message: {raw_message}...")
