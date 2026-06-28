# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the streamed-binary transport helper used by TTS (and any chunked
binary response). Covers per-chunk BinaryResponse emission, chunk-arrival timing,
trace bookkeeping, the collect_chunks toggle, and the empty-stream case."""

from collections.abc import AsyncIterator

import pytest

from aiperf.common.models import AioHttpTraceData, BinaryResponse
from aiperf.transports.sse_utils import collect_streamed_binary


async def _aiter(chunks: list[bytes]) -> AsyncIterator[bytes]:
    for chunk in chunks:
        yield chunk


@pytest.mark.asyncio
class TestCollectStreamedBinary:
    async def test_one_response_per_chunk_with_timing(self):
        trace = AioHttpTraceData()
        chunks = [b"aaa", b"bb", b"c"]

        result = await collect_streamed_binary(
            _aiter(chunks), trace, "audio/wav", collect_chunks=True
        )

        assert [r.raw_bytes for r in result] == chunks
        assert all(isinstance(r, BinaryResponse) for r in result)
        assert all(r.content_type == "audio/wav" for r in result)

        perfs = [r.perf_ns for r in result]
        assert all(p > 0 for p in perfs)
        assert perfs == sorted(perfs)  # non-decreasing arrival times

        assert trace.response_chunks_count == 3
        assert trace.response_bytes_total == 6
        assert len(trace.response_chunks) == 3
        assert trace.response_receive_start_perf_ns == perfs[0]
        assert trace.response_receive_end_perf_ns == perfs[-1]

    async def test_collect_chunks_false_skips_chunk_list_but_keeps_counts(self):
        trace = AioHttpTraceData()

        result = await collect_streamed_binary(
            _aiter([b"x", b"y"]), trace, "audio/mpeg", collect_chunks=False
        )

        assert len(result) == 2
        assert trace.response_chunks == []  # per-chunk list not populated
        assert trace.response_chunks_count == 2
        assert trace.response_bytes_total == 2
        assert trace.response_receive_start_perf_ns is not None
        assert trace.response_receive_end_perf_ns is not None

    async def test_empty_stream_returns_empty_and_leaves_timing_unset(self):
        trace = AioHttpTraceData()

        result = await collect_streamed_binary(
            _aiter([]), trace, "audio/wav", collect_chunks=True
        )

        assert result == []
        assert trace.response_chunks_count == 0
        assert trace.response_bytes_total == 0
        assert trace.response_receive_start_perf_ns is None
        assert trace.response_receive_end_perf_ns is None
