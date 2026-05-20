# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Low-level frontend request helpers for Dynamo chaos scenarios."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import aiohttp
import orjson


def chat_completion_url(endpoint_url: str, *, include_v1: bool = False) -> str:
    """Build a chat-completions URL from a Dynamo endpoint root."""
    prefix = "/v1" if include_v1 else ""
    return f"{endpoint_url.rstrip('/')}{prefix}/chat/completions"


def chat_payload(
    prompt: str,
    *,
    model: str = "default",
    stream: bool = False,
    max_tokens: int = 8,
    temperature: float | None = 0.0,
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Return a minimal OpenAI-compatible chat-completions payload."""
    payload: dict[str, object] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": stream,
        "max_tokens": max_tokens,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if extra is not None:
        payload.update(extra)
    return payload


@dataclass(frozen=True, slots=True)
class HTTPJSON:
    """Decoded JSON response with raw text for assertion diagnostics."""

    status: int
    body: dict[str, object]
    text: str
    headers: dict[str, str] = field(default_factory=dict)


async def post_json(
    session: aiohttp.ClientSession,
    url: str,
    payload: Mapping[str, object],
    *,
    headers: Mapping[str, str] | None = None,
) -> HTTPJSON:
    """POST JSON and decode an object response when present."""
    async with session.post(url, json=payload, headers=headers) as resp:
        text = await resp.text()
        response_headers = {k.lower(): v for k, v in resp.headers.items()}
        status = resp.status
    try:
        decoded = orjson.loads(text)
    except orjson.JSONDecodeError:
        decoded = {}
    body = decoded if isinstance(decoded, dict) else {}
    return HTTPJSON(status=status, body=body, text=text, headers=response_headers)


def append_sse_data_lines(buffer: str, payloads: list[str]) -> str:
    """Append complete LF/CRLF-delimited SSE data payloads and return suffix."""
    while "\n" in buffer:
        raw_line, buffer = buffer.split("\n", 1)
        line = raw_line.rstrip("\r")
        if line.startswith("data:"):
            payloads.append(line.removeprefix("data:").strip())
    return buffer


def append_sse_events(buffer: str, payloads: list[str]) -> str:
    """Append complete blank-line-delimited SSE event payloads and return suffix."""
    while "\n\n" in buffer or "\r\n\r\n" in buffer:
        lf_pos = buffer.find("\n\n") if "\n\n" in buffer else len(buffer) + 1
        crlf_pos = buffer.find("\r\n\r\n") if "\r\n\r\n" in buffer else len(buffer) + 1
        delimiter = "\n\n" if lf_pos < crlf_pos else "\r\n\r\n"
        raw_event, buffer = buffer.split(delimiter, 1)
        for raw_line in raw_event.splitlines():
            line = raw_line.rstrip("\r")
            if line.startswith("data:"):
                payloads.append(line.removeprefix("data:").strip())
    return buffer


async def drain_response_chunks(
    resp: aiohttp.ClientResponse,
    *,
    chunk_size: int | None = None,
) -> int:
    """Drain response content and return the number of chunks observed."""
    chunks = 0
    iterator = (
        resp.content.iter_any()
        if chunk_size is None
        else resp.content.iter_chunked(chunk_size)
    )
    async for _chunk in iterator:
        chunks += 1
    return chunks
