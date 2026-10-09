# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Binary bodies must cross the results bus and reach request accounting."""

from collections.abc import AsyncIterator

import orjson
import pytest
from aiohttp import web
from pytest import param

from aiperf.common.models import BinaryResponse
from tests.harness.utils import AIPerfCLI
from tests.integration.conftest import IntegrationTestDefaults as defaults

BINARY_BODY = bytes(range(256))
REQUEST_COUNT = 6


@pytest.fixture
async def binary_response_server() -> AsyncIterator[tuple[str, list[str]]]:
    request_ids: list[str] = []

    async def handle(request: web.Request) -> web.Response:
        await request.read()
        request_ids.append(request.headers["X-Request-ID"])
        # All-error runs skip aggregate export; retain a successful control response.
        if len(request_ids) == 1:
            return web.Response(
                body=orjson.dumps(
                    {
                        "object": "chat.completion",
                        "choices": [
                            {"message": {"role": "assistant", "content": "ok"}}
                        ],
                        "usage": {
                            "prompt_tokens": 8,
                            "completion_tokens": 1,
                            "total_tokens": 9,
                        },
                    }
                ),
                content_type="application/json",
            )
        return web.Response(body=BINARY_BODY, content_type="application/octet-stream")

    app = web.Application()
    app.router.add_post("/v1/chat/completions", handle)
    runner = web.AppRunner(app)
    await runner.setup()
    try:
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        yield f"http://127.0.0.1:{runner.addresses[0][1]}", request_ids
    finally:
        await runner.cleanup()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "export_level", [param("records", id="records"), param("raw", id="raw")]
)  # fmt: skip
async def test_binary_http_response_counted_and_run_completes(
    cli: AIPerfCLI,
    binary_response_server: tuple[str, list[str]],
    export_level: str,
) -> None:
    url, request_ids = binary_response_server
    result = await cli.run(
        f"""
        aiperf profile
            --model {defaults.model}
            --url {url}
            --endpoint-type chat
            --request-count {REQUEST_COUNT}
            --concurrency 2
            --workers-max 1
            --isl 8 --isl-stddev 0 --osl 1
            --export-level {export_level}
            --ui none
        """,
        timeout=60,
    )

    assert result.exit_code == 0
    assert len(request_ids) == REQUEST_COUNT
    assert len(set(request_ids)) == REQUEST_COUNT
    assert result.json is not None
    successes = result.json.request_count.avg if result.json.request_count else 0
    errors = (
        result.json.error_request_count.avg if result.json.error_request_count else 0
    )
    assert successes + errors == REQUEST_COUNT
    assert successes == 1
    # The chat parser cannot extract content from this legitimate binary HTTP body.
    assert errors == REQUEST_COUNT - 1
    assert result.jsonl is not None
    assert len(result.jsonl) == REQUEST_COUNT
    assert {record.metadata.x_request_id for record in result.jsonl} == set(request_ids)
    for record in result.jsonl:
        if record.metadata.x_request_id == request_ids[0]:
            assert record.error is None
        else:
            assert record.error is not None
            assert record.error.type == "InvalidInferenceResultError"

    if export_level == "raw":
        assert result.raw_records is not None
        assert len(result.raw_records) == REQUEST_COUNT
        assert {record.metadata.x_request_id for record in result.raw_records} == set(
            request_ids
        )
        for record in result.raw_records:
            assert record.status == 200
            assert len(record.responses) == 1
            if record.metadata.x_request_id == request_ids[0]:
                continue
            response = record.responses[0]
            assert isinstance(response, BinaryResponse)
            assert response.raw_bytes == BINARY_BODY
            assert response.content_type == "application/octet-stream"
