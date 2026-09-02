# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ETag helper for conditional GET responses on high-frequency poll endpoints."""

from __future__ import annotations

import hashlib
from typing import Any

import orjson
from fastapi import Request
from fastapi.responses import Response


def etag_response(request: Request, data: Any) -> Response:
    """Return 304 if data is unchanged, else 200 with ETag + Cache-Control: no-cache.

    The ETag is computed from the serialized body on every call - no server-side
    cache. Memory footprint is zero; CPU cost is one SHA-1 hash per poll.
    Cache-Control: no-cache tells the browser to revalidate on every request,
    allowing it to send If-None-Match and receive 304 when the resource is stable.
    """
    body = orjson.dumps(data)
    etag = '"' + hashlib.sha1(body, usedforsecurity=False).hexdigest()[:16] + '"'
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers={"ETag": etag})
    return Response(
        content=body,
        media_type="application/json",
        headers={"ETag": etag, "Cache-Control": "no-cache"},
    )
