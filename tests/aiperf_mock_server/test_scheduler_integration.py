# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end: scheduler-enabled mock server exhibits a real saturation knee."""

import asyncio
import socket
import time

import aiohttp
import pytest
import uvicorn
from aiperf.transports.aiohttp_client import create_tcp_connector
from aiperf.transports.http_defaults import AioHttpDefaults
from aiperf_mock_server.app import asgi_app
from aiperf_mock_server.config import MockServerConfig, set_server_config


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.asyncio
@pytest.mark.slow
async def test_throughput_knees_at_max_batch_size():
    """Drive the server at concurrencies bracketing max_batch_size; throughput
    must saturate, not scale linearly."""
    cfg = MockServerConfig(
        scheduler_enabled=True,
        scheduler_step_ms=2.0,
        scheduler_max_batch_size=16,
        scheduler_max_prefill_chunks_per_step=64,
        scheduler_prefill_chunk_tokens=512,
        ttft=0.0,
        itl=0.0,
        no_tokenizer=True,
    )
    set_server_config(cfg)
    port = _free_port()
    config = uvicorn.Config(asgi_app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())

    try:
        # trust_env comes from AioHttpDefaults (False by default), which keeps
        # localhost requests off any ambient HTTP_PROXY -- a sandbox proxy
        # returns 405 for them.
        base = f"http://127.0.0.1:{port}"
        async with aiohttp.ClientSession(
            base_url=base,
            timeout=aiohttp.ClientTimeout(total=30),
            trust_env=AioHttpDefaults.TRUST_ENV,
            connector=create_tcp_connector(limit=128, limit_per_host=128),
        ) as client:
            for _ in range(50):
                try:
                    async with client.get("/v1/models") as r:
                        if r.status == 200:
                            break
                except (aiohttp.ClientConnectionError, OSError):
                    await asyncio.sleep(0.05)

            async def one_request() -> float:
                t0 = time.perf_counter()
                async with client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "mock-model",
                        "messages": [{"role": "user", "content": "hello world"}],
                        "max_tokens": 32,
                        "stream": False,
                    },
                ) as r:
                    r.raise_for_status()
                    await r.read()
                return time.perf_counter() - t0

            async def measure_throughput(concurrency: int, n: int = 64) -> float:
                sem = asyncio.Semaphore(concurrency)

                async def gated():
                    async with sem:
                        return await one_request()

                t0 = time.perf_counter()
                await asyncio.gather(*[gated() for _ in range(n)])
                return n / (time.perf_counter() - t0)

            tput_low = await measure_throughput(concurrency=8, n=32)
            tput_at = await measure_throughput(concurrency=16, n=64)
            tput_high = await measure_throughput(concurrency=64, n=128)

            assert tput_high < tput_at * 1.5, (
                f"no saturation knee: low={tput_low:.1f} at={tput_at:.1f} "
                f"high={tput_high:.1f} req/s"
            )
            assert tput_at > tput_low * 1.3, (
                f"insufficient ramp: low={tput_low:.1f} at={tput_at:.1f} req/s"
            )
    finally:
        server.should_exit = True
        await server_task
