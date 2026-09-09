# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the conditional-GET ETag helper used by high-frequency poll endpoints.

Covers the three outcomes of :func:`etag_response`: a fresh 200 with headers,
a 304 when the client's ``If-None-Match`` matches, and a 200 when it is stale.
Also includes a route-level round-trip test to verify ETag is correctly wired
into an actual HTTP endpoint (not just the helper in isolation).
"""

import hashlib
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import orjson
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from aiperf.operator.routers._etag import etag_response


def _make_request(etag: str | None = None) -> MagicMock:
    req = MagicMock()
    req.headers = {"if-none-match": etag} if etag else {}
    return req


def _expected_etag(data: object) -> str:
    body = orjson.dumps(data)
    return '"' + hashlib.sha1(body, usedforsecurity=False).hexdigest()[:16] + '"'


def test_etag_response_first_request_returns_200():
    data = {"key": "value", "count": 42}
    resp = etag_response(_make_request(), data)
    assert resp.status_code == 200
    assert resp.headers["etag"] == _expected_etag(data)
    assert resp.headers["cache-control"] == "no-cache"
    assert orjson.loads(resp.body) == data


def test_etag_response_matching_etag_returns_304():
    data = {"key": "value"}
    expected = _expected_etag(data)
    resp = etag_response(_make_request(etag=expected), data)
    assert resp.status_code == 304
    assert resp.headers["etag"] == expected


def test_etag_response_stale_etag_returns_200():
    data = {"key": "new_value"}
    resp = etag_response(_make_request(etag='"stalexxxxxxxx"'), data)
    assert resp.status_code == 200
    assert resp.headers["etag"] == _expected_etag(data)


def test_etag_response_media_type_is_json():
    resp = etag_response(_make_request(), {"a": 1})
    assert resp.headers["content-type"].startswith("application/json")


def test_etag_response_differs_when_data_changes():
    first = etag_response(_make_request(), {"a": 1})
    second = etag_response(_make_request(), {"a": 2})
    assert first.headers["etag"] != second.headers["etag"]


# ---------------------------------------------------------------------------
# Route-level ETag round-trip
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_jobs_etag_200_then_304(tmp_path: Path) -> None:
    """GET /api/v1/jobs returns ETag on first request and 304 on second.

    The helper tests above only exercise the ``etag_response`` function directly
    via MagicMock. This test exercises the full HTTP stack so a future route
    that forgets to call ``etag_response`` cannot hide behind the helper tests.
    """
    from aiperf.operator.routers.jobs import create_jobs_router
    from aiperf.operator.routers.jobs_models import ActiveJobListResponse
    from aiperf.operator.routers.mutating_auth import mutating_route_dependencies

    app = FastAPI()
    app.include_router(
        create_jobs_router([object()], tmp_path, mutating_route_dependencies())
    )

    fixed_response = ActiveJobListResponse(jobs=[])
    list_impl = AsyncMock(return_value=fixed_response)

    with patch("aiperf.operator.routers.jobs._list_jobs_impl", list_impl):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r1 = await client.get("/api/v1/jobs")
            assert r1.status_code == 200
            etag = r1.headers.get("etag")
            assert etag is not None, "ETag header missing from first response"
            assert r1.headers.get("cache-control") == "no-cache"

            r2 = await client.get("/api/v1/jobs", headers={"If-None-Match": etag})
            assert r2.status_code == 304, (
                f"Expected 304 on matching ETag, got {r2.status_code}"
            )
            assert len(r2.content) == 0
