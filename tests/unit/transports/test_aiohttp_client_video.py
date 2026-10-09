# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for AioHttpClient video-specific functionality."""

from unittest.mock import AsyncMock, patch

import aiohttp
import orjson
import pytest
from multidict import CIMultiDict
from pytest import param

from aiperf.transports.aiohttp_client import AioHttpClient
from tests.unit.transports.conftest import create_mock_response, setup_mock_session


class TestAioHttpClientVideo:
    """Tests for AioHttpClient video generation support."""

    @pytest.mark.asyncio
    async def test_get_request_video_content(self, aiohttp_client):
        """Test GET request for video content download."""
        mock_response = create_mock_response(
            status=200,
            content_type="video/mp4",
            text_content="",  # Empty for binary content
        )
        mock_response.headers = {"Content-Type": "video/mp4", "Content-Length": "1024"}
        # Set up the read method that the client actually calls
        mock_response.read = AsyncMock(return_value=b"fake_video_data")

        with patch("aiohttp.ClientSession") as mock_session_class:
            setup_mock_session(mock_session_class, mock_response, ["request"])

            record = await aiohttp_client.get_request(
                "http://localhost:8000/v1/videos/video-123/content",
                {"Authorization": "Bearer token"},
            )

            assert record.status == 200
            assert len(record.responses) == 1
            binary_response = record.responses[0]
            assert hasattr(binary_response, "raw_bytes")
            assert binary_response.raw_bytes == b"fake_video_data"
            assert binary_response.content_type == "video/mp4"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status_code", [200, 202])
    async def test_post_request_video_submit_accepts_success_statuses(
        self, aiohttp_client, status_code: int
    ):
        """Test POST request for video job submission accepts successful statuses."""
        response_text = orjson.dumps(
            {
                "id": "video-123",
                "status": "queued",
                "created_at": "2024-01-01T00:00:00Z",
            }
        ).decode()

        mock_response = create_mock_response(
            status=status_code,
            content_type="application/json",
            text_content=response_text,
        )

        with patch("aiohttp.ClientSession") as mock_session_class:
            setup_mock_session(mock_session_class, mock_response, ["request"])

            payload = orjson.dumps(
                {"prompt": "A cat playing piano", "model": "sora-2"}
            ).decode()

            record = await aiohttp_client.post_request(
                "http://localhost:8000/v1/videos",
                payload,
                {"Content-Type": "application/json"},
            )

            assert record.status == status_code
            assert record.error is None
            assert len(record.responses) == 1
            text_response = record.responses[0]
            assert hasattr(text_response, "text")
            data = orjson.loads(text_response.text)
            assert data["id"] == "video-123"
            assert data["status"] == "queued"

    @pytest.mark.asyncio
    async def test_get_request_video_status(self, aiohttp_client):
        """Test GET request for video job status polling."""
        response_text = orjson.dumps(
            {
                "id": "video-123",
                "status": "processing",
                "progress": 0.75,
                "eta_seconds": 30,
            }
        ).decode()

        mock_response = create_mock_response(
            status=200, content_type="application/json", text_content=response_text
        )

        with patch("aiohttp.ClientSession") as mock_session_class:
            setup_mock_session(mock_session_class, mock_response, ["request"])

            record = await aiohttp_client.get_request(
                "http://localhost:8000/v1/videos/video-123",
                {"Authorization": "Bearer token"},
            )

            assert record.status == 200
            assert len(record.responses) == 1
            text_response = record.responses[0]
            assert hasattr(text_response, "text")
            data = orjson.loads(text_response.text)
            assert data["status"] == "processing"
            assert data["progress"] == 0.75

    @pytest.mark.asyncio
    async def test_request_timeout_handling(self, aiohttp_client):
        """Test timeout handling for long video generation requests."""
        # Create a mock response that will never be reached due to timeout
        mock_response = create_mock_response(status=200)

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = setup_mock_session(
                mock_session_class, mock_response, ["request"]
            )
            # Override the request method to raise timeout
            mock_session.request.side_effect = aiohttp.ServerTimeoutError()

            record = await aiohttp_client.post_request(
                "http://localhost:8000/v1/videos", "{}", {}
            )

            # Should return a record with error details rather than raising
            assert record.error is not None
            assert "timeout" in record.error.message.lower()

    @pytest.mark.asyncio
    async def test_large_video_content_download(self, aiohttp_client):
        """Test handling of large video content downloads."""
        large_video_data = b"fake_video" * 100000  # ~700KB fake data

        mock_response = create_mock_response(
            status=200,
            content_type="video/mp4",
            text_content="",  # Empty for binary content
        )
        mock_response.headers = {
            "Content-Type": "video/mp4",
            "Content-Length": str(len(large_video_data)),
        }
        mock_response.read = AsyncMock(return_value=large_video_data)

        with patch("aiohttp.ClientSession") as mock_session_class:
            setup_mock_session(mock_session_class, mock_response, ["request"])

            record = await aiohttp_client.get_request(
                "http://localhost:8000/v1/videos/video-123/content",
                {"Authorization": "Bearer token"},
            )

            assert record.status == 200
            assert len(record.responses) == 1
            binary_response = record.responses[0]
            assert hasattr(binary_response, "raw_bytes")
            assert len(binary_response.raw_bytes) == len(large_video_data)
            assert binary_response.raw_bytes == large_video_data


@pytest.mark.parametrize("status", [param(301), param(302), param(303), param(307), param(308)])  # fmt: skip
async def test_capture_redirect_without_reading_body(
    aiohttp_client: AioHttpClient, status: int
) -> None:
    """Redirect bodies may be binary or unfinished; only headers are needed."""
    response = create_mock_response(status=status)
    response.headers = CIMultiDict(
        [("Location", "/one?secret=1"), ("Location", "/two")]
    )
    response.read = AsyncMock(side_effect=AssertionError("Must not read"))
    response.text.side_effect = AssertionError("Must not decode")
    with patch("aiohttp.ClientSession") as session_class:
        session = setup_mock_session(session_class, response, ["request"])
        record = await aiohttp_client.get_request(
            "https://origin.test/content",
            {},
            allow_redirects=False,
            capture_redirects=True,
            trust_env=False,
        )
    assert record.status == status and record.error is None
    assert record._redirect_locations == ("/one?secret=1", "/two")
    assert record.end_perf_ns is not None
    assert not record.responses
    assert "secret" not in record.model_dump_json()
    response.read.assert_not_awaited()
    response.text.assert_not_awaited()
    response.close.assert_called_once()
    assert session_class.call_args.kwargs["trust_env"] is False
    assert "capture_redirects" not in session.request.call_args.kwargs
    assert "trust_env" not in session.request.call_args.kwargs


@pytest.mark.parametrize("method,kwargs", [
    param("POST", {"allow_redirects": False}), param("GET", {}),
    param("GET", {"allow_redirects": True}),
])  # fmt: skip
async def test_redirect_capture_requires_manual_get(
    aiohttp_client: AioHttpClient, method: str, kwargs: dict[str, bool]
) -> None:
    with (
        patch("aiohttp.ClientSession") as session_class,
        pytest.raises(ValueError, match="Redirect capture requires GET"),
    ):
        await aiohttp_client._request(
            method,
            "https://origin.test/content",
            {},
            capture_redirects=True,
            **kwargs,
        )
    session_class.assert_not_called()


async def test_generic_redirect_still_returns_http_error(
    aiohttp_client: AioHttpClient,
) -> None:
    response = create_mock_response(status=302, text_content="redirect body")
    with patch("aiohttp.ClientSession") as session_class:
        setup_mock_session(session_class, response, ["request"])
        record = await aiohttp_client.get_request(
            "https://origin.test/content", {}, allow_redirects=False
        )
    assert record.error is not None and record.error.message == "redirect body"
    response.text.assert_awaited_once()
    assert not record._redirect_locations
