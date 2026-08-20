# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""RFC 6455 close-reason limits for the operator WebSocket proxy."""

from unittest.mock import AsyncMock

import pytest
from aiperf.operator.routers.jobs_ws import (
    _CLOSE_MAX_REASON_BYTES,
    _close_ws,
    _truncate_close_reason,
)
from pytest import param


@pytest.mark.parametrize(
    "reason, expected",
    [
        param("short", "short", id="unchanged"),
        param("a" * 123, "a" * 123, id="at-limit"),
        param("a" * 200, "a" * 123, id="ascii-truncated"),
        param("é" * 100, "é" * 61, id="utf8-codepoint-safe"),
    ],
)  # fmt: skip
def test_truncate_close_reason(reason: str, expected: str) -> None:
    result = _truncate_close_reason(reason)

    assert result == expected
    assert len(result.encode()) <= _CLOSE_MAX_REASON_BYTES


@pytest.mark.asyncio
async def test_close_ws_always_uses_protocol_safe_reason() -> None:
    websocket = AsyncMock()

    await _close_ws(websocket, code=4502, reason="cannot connect: " + "é" * 100)

    reason = websocket.close.await_args.kwargs["reason"]
    assert len(reason.encode()) <= 123
    assert websocket.close.await_args.kwargs["code"] == 4502
