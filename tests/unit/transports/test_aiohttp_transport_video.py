# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for AioHttpTransport video generation functionality."""

from unittest.mock import AsyncMock, Mock, patch

import orjson
import pytest

from aiperf.common.models import ErrorDetails, RequestRecord, TextResponse
from aiperf.plugin.enums import EndpointType
from aiperf.transports.aiohttp_transport import AioHttpTransport
from tests.unit.transports.test_aiohttp_transport import create_request_info


def create_request_record(status: int = 200, body: str | bytes = "") -> RequestRecord:
    """Create a test RequestRecord with the given status and body."""
    import time

    from aiperf.common.models import ErrorDetails, TextResponse

    perf_ns = time.perf_counter_ns()

    # Create error for HTTP error status codes
    error = None
    if status >= 400:
        error = ErrorDetails(
            type="HTTPError", message=f"HTTP {status} error", code=status
        )

    # RequestRecord responses field only accepts SSEMessage | TextResponse
    # For binary data, we'll store it as body but create a TextResponse with empty text
    if isinstance(body, bytes):
        response = TextResponse(
            perf_ns=perf_ns,
            text="",  # Empty text for binary data
        )
    else:
        response = TextResponse(perf_ns=perf_ns, text=body)

    return RequestRecord(
        request_headers={},
        start_perf_ns=perf_ns,
        end_perf_ns=perf_ns + 1000000,  # Add 1ms
        status=status,
        body=body,
        response_headers={},
        responses=[response] if status < 400 else [],  # No responses for error status
        error=error,
    )


def create_video_model_endpoint_info():
    """Create a video generation model endpoint."""
    from aiperf.common.enums import ConnectionReuseStrategy, ModelSelectionStrategy
    from aiperf.common.models.model_endpoint_info import (
        EndpointInfo,
        ModelEndpointInfo,
        ModelInfo,
        ModelListInfo,
    )

    return ModelEndpointInfo(
        models=ModelListInfo(
            models=[ModelInfo(name="sora-2")],
            model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
        ),
        endpoint=EndpointInfo(
            type=EndpointType.VIDEO_GENERATION,
            base_urls=["http://localhost:8000"],
            custom_endpoint="/v1/videos",
            streaming=False,
            api_key=None,
            headers=[],
            connection_reuse_strategy=ConnectionReuseStrategy.POOLED,
        ),
    )


@pytest.fixture
def video_model_endpoint():
    """Create a video generation model endpoint."""
    return create_video_model_endpoint_info()


@pytest.fixture
def video_request_info(video_model_endpoint):
    """Create a RequestInfo for video generation."""
    return create_request_info(video_model_endpoint)


@pytest.fixture
def transport(video_model_endpoint):
    """Create an AioHttpTransport instance."""
    transport = AioHttpTransport(model_endpoint=video_model_endpoint)
    transport.aiohttp_client = AsyncMock()
    return transport


class TestVideoTransportParsing:
    """Tests for video response parsing."""

    def test_parse_video_response_success(self, transport):
        """Test successful video response parsing."""
        record = create_request_record(
            status=200,
            body=orjson.dumps(
                {
                    "id": "video-123",
                    "status": "completed",
                    "inference_time_s": 5.5,
                    "peak_memory_mb": 8192.0,
                }
            ).decode(),
        )

        result = transport._parse_video_response(record, "test")
        assert not isinstance(result, ErrorDetails)
        job_data, response = result
        assert job_data["id"] == "video-123"
        assert job_data["status"] == "completed"
        assert isinstance(response, TextResponse)

    def test_parse_video_response_invalid_json(self, transport):
        """Test video response parsing with invalid JSON returns ErrorDetails."""
        record = create_request_record(status=200, body="invalid json")

        result = transport._parse_video_response(record, "test")
        assert isinstance(result, ErrorDetails)
        assert result.type == "VideoGenerationError"
        assert result.code == 500
        assert "Invalid JSON" in result.message
        assert "invalid json" in result.message

    def test_parse_video_response_error_status(self, transport):
        """Test video response parsing with error status."""
        record = create_request_record(
            status=400, body=orjson.dumps({"error": "Bad request"}).decode()
        )

        result = transport._parse_video_response(record, "test")
        assert isinstance(result, ErrorDetails)
        assert result.code == 400


class TestVideoJobSubmission:
    """Tests for video job submission."""

    @pytest.mark.asyncio
    async def test_submit_video_job_success(self, transport):
        """Test successful video job submission."""
        transport.aiohttp_client.post_request.return_value = create_request_record(
            status=201,
            body=orjson.dumps(
                {
                    "id": "video-123",
                    "status": "queued",
                    "created_at": "2024-01-01T00:00:00Z",
                }
            ).decode(),
        )

        result = await transport._submit_video_job(
            "http://localhost/v1/videos",
            {"prompt": "A cat playing piano"},
            {"Content-Type": "application/json"},
        )

        assert not isinstance(result, ErrorDetails)
        job_id, response = result
        assert job_id == "video-123"
        assert isinstance(response, TextResponse)

    @pytest.mark.asyncio
    async def test_submit_video_job_no_id(self, transport):
        """Test video job submission without job ID."""
        transport.aiohttp_client.post_request.return_value = create_request_record(
            status=200, body=orjson.dumps({"status": "queued"}).decode()
        )

        result = await transport._submit_video_job(
            "http://localhost/v1/videos",
            {"prompt": "A cat playing piano"},
            {"Content-Type": "application/json"},
        )

        assert isinstance(result, ErrorDetails)
        assert "No job ID returned" in result.message

    @pytest.mark.asyncio
    async def test_submit_video_job_not_initialized(self, video_model_endpoint):
        """Test video job submission when client not initialized."""
        transport = AioHttpTransport(model_endpoint=video_model_endpoint)
        # Don't initialize aiohttp_client

        with pytest.raises(Exception, match="not initialized"):
            await transport._submit_video_job(
                "http://localhost/v1/videos", {"prompt": "test"}, {}
            )


class TestVideoJobPolling:
    """Tests for video job status polling."""

    @pytest.mark.asyncio
    async def test_poll_video_job_completed(self, transport):
        """Test polling completed video job."""
        transport.aiohttp_client.get_request.return_value = create_request_record(
            status=200,
            body=orjson.dumps(
                {
                    "id": "video-123",
                    "status": "completed",
                    "inference_time_s": 5.5,
                    "peak_memory_mb": 8192.0,
                }
            ).decode(),
        )

        result = await transport._poll_video_job(
            "video-123",
            "http://localhost/v1/videos/video-123",
            {"Authorization": "Bearer token"},
            timeout=60.0,
            poll_interval=1.0,
        )

        assert not isinstance(result, ErrorDetails)
        job_data, elapsed_time = result
        assert job_data["status"] == "completed"
        assert isinstance(elapsed_time, float)

    @pytest.mark.asyncio
    async def test_poll_video_job_failed(self, transport):
        """Test polling failed video job."""
        transport.aiohttp_client.get_request.return_value = create_request_record(
            status=200,
            body=orjson.dumps(
                {"id": "video-123", "status": "failed", "error": "Generation failed"}
            ).decode(),
        )

        result = await transport._poll_video_job(
            "video-123",
            "http://localhost/v1/videos/video-123",
            {"Authorization": "Bearer token"},
            timeout=60.0,
            poll_interval=1.0,
        )

        assert isinstance(result, ErrorDetails)
        assert "Video generation failed" in result.message

    @pytest.mark.asyncio
    async def test_poll_video_job_timeout(self, transport):
        """Test polling job that times out."""
        # Mock multiple responses showing job still in progress
        transport.aiohttp_client.get_request.side_effect = [
            create_request_record(
                status=200,
                body=orjson.dumps({"id": "video-123", "status": "queued"}).decode(),
            ),
            create_request_record(
                status=200,
                body=orjson.dumps({"id": "video-123", "status": "processing"}).decode(),
            ),
        ]

        with patch(
            "aiperf.transports.aiohttp_transport.time.perf_counter_ns"
        ) as mock_time:
            # Mock time progression: poll_start=0, enter loop at 1s, enter loop at 2s, exceed timeout at 61s
            mock_time.side_effect = [0, 1_000_000_000, 2_000_000_000, 61_000_000_000]

            result = await transport._poll_video_job(
                "video-123",
                "http://localhost/v1/videos/video-123",
                {"Authorization": "Bearer token"},
                timeout=60.0,
                poll_interval=1.0,
            )

        assert isinstance(result, ErrorDetails)
        assert "timed out" in result.message.lower()


class TestVideoContentDownload:
    """Tests for video content download."""

    @pytest.mark.asyncio
    async def test_download_video_content_success(self, transport):
        """Test successful video content download."""
        import time

        from aiperf.common.models import BinaryResponse

        perf_ns = time.perf_counter_ns()
        mock_record = RequestRecord(
            request_headers={},
            start_perf_ns=perf_ns,
            end_perf_ns=perf_ns + 1000000,
            status=200,
            responses=[
                BinaryResponse(
                    perf_ns=perf_ns,
                    raw_bytes=b"fake_video_content",
                )
            ],
        )

        transport.aiohttp_client.get_request.return_value = mock_record

        result = await transport._download_video_content(
            "video-123",
            "http://localhost/v1/videos/video-123/content",
            {"Authorization": "Bearer token"},
        )

        assert not isinstance(result, ErrorDetails)
        video_bytes = result
        assert video_bytes == b"fake_video_content"

    @pytest.mark.asyncio
    async def test_download_video_content_not_found(self, transport):
        """Test video content download when content not found."""
        transport.aiohttp_client.get_request.return_value = create_request_record(
            status=404, body="Video content not found"
        )

        result = await transport._download_video_content(
            "video-123",
            "http://localhost/v1/videos/video-123/content",
            {"Authorization": "Bearer token"},
        )

        assert isinstance(result, ErrorDetails)
        # The actual implementation propagates the original error code
        assert result.code == 404
        assert "Failed to download video" in result.message


class TestVideoRequestWorkflow:
    """Tests for end-to-end video request workflow."""

    @pytest.mark.asyncio
    async def test_send_video_request_with_polling_success(
        self, transport, video_request_info
    ):
        """Test successful end-to-end video generation workflow."""
        # Mock the individual methods used in the workflow
        with (
            patch.object(transport, "_submit_video_job") as mock_submit,
            patch.object(transport, "_poll_video_job") as mock_poll,
            patch.object(transport, "_download_video_content") as mock_download,
        ):
            # Configure mocks with proper response objects
            import time

            from aiperf.common.models import BinaryResponse, TextResponse

            submit_response = TextResponse(
                perf_ns=time.perf_counter_ns(),
                text='{"id":"video-123","status":"queued"}',
            )
            download_response = BinaryResponse(
                perf_ns=time.perf_counter_ns(), raw_bytes=b"fake_video_data"
            )

            mock_submit.return_value = ("video-123", submit_response)
            mock_poll.return_value = (
                {"id": "video-123", "status": "completed", "inference_time_s": 5.5},
                5.5,  # elapsed time
            )
            mock_download.return_value = download_response

            # Mock download_video_content to be True for this test
            with patch.object(
                video_request_info.model_endpoint.endpoint,
                "download_video_content",
                True,
            ):
                # Execute workflow
                record = await transport._send_video_request_with_polling(
                    video_request_info, {"prompt": "A cat playing piano"}
                )

                # Verify calls were made
                mock_submit.assert_called_once()
                mock_poll.assert_called_once()
                mock_download.assert_called_once()

                # Verify record structure
                assert isinstance(record, RequestRecord)
                assert record.error is None
                assert len(record.responses) >= 2  # at least submit and poll

    @pytest.mark.asyncio
    async def test_send_video_request_submit_failure(
        self, transport, video_request_info
    ):
        """Test video workflow when job submission fails."""
        with patch.object(transport, "_submit_video_job") as mock_submit:
            error_details = ErrorDetails(
                type="SubmissionError", message="Failed to submit job", code=400
            )
            mock_submit.return_value = error_details

            record = await transport._send_video_request_with_polling(
                video_request_info, {"prompt": "A cat playing piano"}
            )

            assert record.error == error_details
            # The status field may not be set when there's an error in the workflow

    @pytest.mark.asyncio
    async def test_send_video_request_without_download(
        self, transport, video_request_info
    ):
        """Test video workflow without content download."""
        with (
            patch.object(transport, "_submit_video_job") as mock_submit,
            patch.object(transport, "_poll_video_job") as mock_poll,
            patch.object(transport, "_download_video_content") as mock_download,
        ):
            # Configure mocks with proper response objects
            import time

            from aiperf.common.models import TextResponse

            submit_response = TextResponse(
                perf_ns=time.perf_counter_ns(),
                text='{"id":"video-123","status":"queued"}',
            )

            mock_submit.return_value = ("video-123", submit_response)
            mock_poll.return_value = (
                {"id": "video-123", "status": "completed"},
                3.5,  # elapsed time
            )

            # Mock download_video_content as False for this test
            with patch.object(
                video_request_info.model_endpoint.endpoint,
                "download_video_content",
                False,
            ):
                record = await transport._send_video_request_with_polling(
                    video_request_info, {"prompt": "A cat playing piano"}
                )

                # Verify download was not called
                mock_download.assert_not_called()
                assert len(record.responses) >= 2  # at least submit and poll


class TestVideoRequestRouting:
    """Tests for video request routing in main send_request method."""

    @pytest.mark.asyncio
    async def test_video_request_routes_to_polling_method(
        self, transport, video_request_info
    ):
        """Test that video generation requests are routed to polling method."""
        with patch.object(
            transport, "_send_video_request_with_polling"
        ) as mock_polling:
            mock_polling.return_value = Mock(spec=RequestRecord)

            # This should route to the video polling method
            await transport.send_request(
                video_request_info, {"prompt": "A cat playing piano"}
            )

            mock_polling.assert_called_once_with(
                video_request_info, {"prompt": "A cat playing piano"}
            )
