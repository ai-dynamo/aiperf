# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-session profiling with a separate control origin."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from shlex import quote
from typing import Any

import aiohttp
import orjson
import pytest
from aiohttp import web
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from tests.harness.utils import AIPerfCLI, AIPerfMockServer
from tests.integration.conftest import IntegrationTestDefaults as defaults


@asynccontextmanager
async def _serve(app: web.Application) -> AsyncIterator[str]:
    runner = web.AppRunner(app)
    await runner.setup()
    try:
        await web.TCPSite(runner, "127.0.0.1", 0).start()
        yield f"http://127.0.0.1:{runner.addresses[0][1]}"
    finally:
        await runner.cleanup()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_separate_profiler_starts_after_warmup_and_load_finishes(
    cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer, tmp_path: Path
) -> None:
    tokenizer = Tokenizer(
        models.WordLevel({f"token{i}": i for i in range(256)}, unk_token="token0")
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer_dir = tmp_path / "tokenizer"
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="token0"
    ).save_pretrained(tokenizer_dir)

    events: list[str] = []
    start_bodies: list[dict[str, Any]] = []
    expected_body = {"num_steps": 2, "output_dir": "/tmp/test-capture"}

    async def start(request: web.Request) -> web.Response:
        assert request.content_type == "application/json"
        start_bodies.append(await request.json())
        events.append("profiler.start")
        return web.Response(status=200)

    async def stop(request: web.Request) -> web.Response:
        assert await request.read() == b""
        events.append("profiler.stop")
        return web.Response(status=500, text="mock profiler stop failed")

    admin = web.Application()
    admin.router.add_post("/engine/control/start_profile", start)
    admin.router.add_post("/engine/control/stop_profile", stop)

    async with aiohttp.ClientSession() as session:

        async def forward(request: web.Request) -> web.Response:
            if request.path == "/v1/chat/completions":
                events.append("inference.request")
            async with session.request(
                request.method,
                f"{aiperf_mock_server.url}{request.rel_url}",
                data=await request.read(),
                headers={"Content-Type": request.content_type},
            ) as response:
                body = await response.read()
                if request.path == "/v1/chat/completions":
                    events.append("inference.complete")
                return web.Response(
                    status=response.status,
                    body=body,
                    headers={"Content-Type": response.headers["Content-Type"]},
                )

        inference = web.Application()
        inference.router.add_route("*", "/{path:.*}", forward)
        async with _serve(inference) as inference_url, _serve(admin) as admin_url:
            result = await cli.run(
                f"""
                aiperf profile
                    --model {defaults.model}
                    --tokenizer {quote(str(tokenizer_dir))}
                    --url {inference_url}
                    --endpoint-type chat
                    --warmup-request-count 5
                    --request-count 15
                    --concurrency 1
                    --workers-max 1
                    --ui {defaults.ui}
                    --server-profiler-url {admin_url}
                    --server-profiler-start-path /engine/control/start_profile
                    --server-profiler-stop-path /engine/control/stop_profile
                    --server-profiler-start-body {quote(orjson.dumps(expected_body).decode())}
                """
            )

    assert result.exit_code == 0
    assert result.request_count == 15
    assert start_bodies == [expected_body]
    assert events.count("profiler.start") == events.count("profiler.stop") == 1
    start_index = events.index("profiler.start")
    request_pair = ["inference.request", "inference.complete"]
    assert events[:start_index] == request_pair * 5
    assert events[start_index + 1 : -1] == request_pair * 15
    assert events[-1] == "profiler.stop"
    assert "server_profiler stop failed" in result.log
