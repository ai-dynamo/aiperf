# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D211 -- Dynamo frontend accepts strict CRLF HTTP request delimiters."""

from __future__ import annotations

import asyncio
from urllib.parse import urlparse

import orjson
import pytest

pytestmark = [pytest.mark.k8s_slow, pytest.mark.asyncio]


async def _post_with_raw_crlf(endpoint_url: str, payload: dict[str, object]) -> bytes:
    """Send a raw HTTP/1.1 POST using CRLF line delimiters and return bytes."""
    parsed = urlparse(endpoint_url)
    if parsed.scheme != "http" or parsed.hostname is None or parsed.port is None:
        pytest.skip(
            "D211 requires a host-reachable plain-http Dynamo endpoint with an explicit port; "
            f"got {endpoint_url!r}"
        )
    path = f"{parsed.path.rstrip('/')}/chat/completions"
    body = orjson.dumps(payload)
    request = (
        f"POST {path} HTTP/1.1\r\n"
        f"Host: {parsed.hostname}:{parsed.port}\r\n"
        "Content-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\n"
        "Connection: close\r\n"
        "\r\n"
    ).encode() + body

    reader, writer = await asyncio.open_connection(parsed.hostname, parsed.port)
    try:
        writer.write(request)
        await writer.drain()
        return await asyncio.wait_for(reader.read(-1), timeout=20.0)
    finally:
        writer.close()
        await writer.wait_closed()


def _decode_chunked_body(body: bytes) -> bytes:
    """Decode a minimal HTTP/1.1 chunked body for raw-socket assertions."""
    decoded = bytearray()
    rest = body
    while rest:
        size_text, sep, after_size = rest.partition(b"\r\n")
        if not sep:
            break
        size = int(size_text.split(b";", 1)[0], 16)
        if size == 0:
            break
        decoded.extend(after_size[:size])
        rest = after_size[size + 2 :]
    return bytes(decoded)


def _split_http_response(raw: bytes) -> tuple[int, bytes]:
    """Return ``(status, body)`` from a raw HTTP response."""
    header_bytes, sep, body = raw.partition(b"\r\n\r\n")
    assert sep, (
        f"D211: response did not contain CRLF CRLF header delimiter: {raw[:512]!r}"
    )
    header_lines = header_bytes.split(b"\r\n")
    status_line = header_lines[0].decode(errors="replace")
    parts = status_line.split()
    assert len(parts) >= 2 and parts[1].isdigit(), (
        f"D211: malformed HTTP status line for CRLF request: {status_line!r}"
    )
    headers = b"\r\n".join(header_lines[1:]).lower()
    if b"transfer-encoding: chunked" in headers:
        body = _decode_chunked_body(body)
    return int(parts[1]), body


async def test_d211_crlf_delimited_http_request_returns_openai_json(
    dynamo_endpoint_url: str,
) -> None:
    """Use raw CRLF HTTP framing and assert Dynamo returns a normal JSON shape."""
    payload: dict[str, object] = {
        "model": "default",
        "messages": [{"role": "user", "content": "Reply with the word ok."}],
        "max_tokens": 8,
        "stream": False,
        "temperature": 0.0,
    }
    status, body = _split_http_response(
        await _post_with_raw_crlf(dynamo_endpoint_url, payload)
    )

    assert status == 200, (
        f"D211: CRLF-framed request returned HTTP {status}: {body[:512]!r}"
    )
    decoded = orjson.loads(body)
    assert isinstance(decoded, dict), (
        f"D211: response body is not a JSON object: {decoded!r}"
    )
    assert decoded.get("object") == "chat.completion", (
        f"D211: non-stream response object mismatch: {decoded!r}"
    )
    choices = decoded.get("choices")
    assert isinstance(choices, list) and choices, (
        f"D211: CRLF request response missing non-empty choices: {decoded!r}"
    )
