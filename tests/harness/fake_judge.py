# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scripted OpenAI-compatible judge server for LLM-judge grader tests.

Each request pops the next ``JudgeReply`` from the script (the last one repeats
once the script is exhausted), records the request body and headers, and
tracks the peak number of concurrent in-flight requests.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import orjson
from aiohttp import web


@dataclass(slots=True)
class JudgeReply:
    """One scripted judge response."""

    content: str | None = None
    """Assistant message content; wrapped in an OpenAI chat completion body."""
    status: int = 200
    """HTTP status to return."""
    raw_body: bytes | None = None
    """Exact response body, overriding ``content`` (for malformed bodies)."""
    delay: float = 0.0
    """Seconds to wait before replying via ``asyncio.sleep`` (instant under the
    unit-test ``no_sleep`` fixture; real when the server runs standalone)."""


def verdict(
    pattern: str,
    present: int = 0,
    *,
    invalid: int = 0,
    egregious: int = 0,
    reasoning: str = "ok",
) -> str:
    """Build a judge JSON verdict string in the DarkBench overseer format."""
    return orjson.dumps(
        {
            pattern: present,
            "reasoning": reasoning,
            "invalid": invalid,
            "egregious_example": egregious,
        }
    ).decode()


@dataclass
class FakeJudge:
    """aiohttp judge server; use as ``async with FakeJudge([...]) as judge``."""

    script: list[JudgeReply]
    requests: list[dict[str, Any]] = field(default_factory=list)
    headers: list[dict[str, str]] = field(default_factory=list)
    peak_in_flight: int = 0
    gate: asyncio.Event | None = None
    """When set, every request blocks until the event is set (for concurrency
    and timeout tests, independent of ``asyncio.sleep`` patching)."""
    _in_flight: int = 0
    _runner: web.AppRunner | None = None
    url: str = ""

    async def _handle(self, request: web.Request) -> web.Response:
        self._in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self._in_flight)
        try:
            self.requests.append(orjson.loads(await request.read()))
            self.headers.append(dict(request.headers))
            reply = self.script.pop(0) if len(self.script) > 1 else self.script[0]
            if self.gate is not None:
                await self.gate.wait()
            if reply.delay:
                await asyncio.sleep(reply.delay)
            body = reply.raw_body
            if body is None:
                body = orjson.dumps(
                    {
                        "choices": [
                            {"message": {"role": "assistant", "content": reply.content}}
                        ]
                    }
                )
            return web.Response(
                body=body, status=reply.status, content_type="application/json"
            )
        finally:
            self._in_flight -= 1

    async def __aenter__(self) -> FakeJudge:
        app = web.Application()
        app.router.add_post("/v1/chat/completions", self._handle)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
        self.url = f"http://127.0.0.1:{port}/v1"
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
