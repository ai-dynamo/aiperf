# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HTTP invocation and body EOF are separate from the last SSE event."""

import asyncio
import time

import pytest
from aiohttp import web

from aiperf.transports.aiohttp_client import AioHttpClient


@pytest.mark.asyncio
async def test_sse_eof_is_captured_after_done_and_exported_separately() -> None:
    done_sent = asyncio.Event()
    release_eof = asyncio.Event()
    server_received_at: list[int] = []

    async def stream(request: web.Request) -> web.StreamResponse:
        server_received_at.append(time.perf_counter_ns())
        response = web.StreamResponse(
            status=200, headers={"Content-Type": "text/event-stream"}
        )
        await response.prepare(request)
        await response.write(b'data: {"choices":[{"delta":{"content":"x"}}]}\n\n')
        await response.write(b"data: [DONE]\n\n")
        done_sent.set()
        await release_eof.wait()
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_post("/v1/chat/completions", stream)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    client = AioHttpClient()
    starts: list[int] = []
    eofs: list[int] = []
    try:
        request = asyncio.create_task(
            client.post_request(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                b"{}",
                {"Content-Type": "application/json"},
                transport_start_callback=starts.append,
                transport_eof_callback=eofs.append,
            )
        )
        await asyncio.wait_for(done_sent.wait(), timeout=2)
        assert not eofs
        asyncio.get_running_loop().call_later(0.06, release_eof.set)
        record = await asyncio.wait_for(request, timeout=2)
        assert record.error is None
        assert len(starts) == len(eofs) == 1
        assert starts[0] <= server_received_at[0]
        assert record.response_body_eof_perf_ns == eofs[0]
        assert record.end_perf_ns >= eofs[0]
        assert (
            eofs[0] - max(message.perf_ns for message in record.responses) >= 30_000_000
        )
    finally:
        release_eof.set()
        await client.close()
        await runner.cleanup()
