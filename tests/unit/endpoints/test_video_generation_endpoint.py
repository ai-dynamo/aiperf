# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for VideoGenerationEndpoint."""

import pytest

from aiperf.common.models import Text, Turn
from aiperf.endpoints.openai_video_generation import VideoGenerationEndpoint
from aiperf.plugin.enums import EndpointType
from tests.unit.endpoints.conftest import (
    create_endpoint_with_mock_transport,
    create_mock_response,
    create_model_endpoint,
    create_request_info,
)


class TestVideoGenerationEndpoint:
    """Tests for VideoGenerationEndpoint."""

    @pytest.fixture
    def model_endpoint(self):
        """Create a test ModelEndpointInfo for video generation."""
        return create_model_endpoint(EndpointType.VIDEO_GENERATION, model_name="sora-2")

    @pytest.fixture
    def endpoint(self, model_endpoint):
        """Create a VideoGenerationEndpoint instance."""
        return create_endpoint_with_mock_transport(
            VideoGenerationEndpoint, model_endpoint
        )

    # ===== format_payload tests =====

    def test_format_payload_simple_prompt(self, endpoint, model_endpoint):
        """Test simple prompt formatting."""
        turn = Turn(
            texts=[Text(contents=["A cat playing piano"])],
            model="sora-2",
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["prompt"] == "A cat playing piano"
        assert payload["model"] == "sora-2"

    def test_format_payload_with_extra_inputs(self):
        """Test payload formatting with extra inputs."""
        model_endpoint_with_extra = create_model_endpoint(
            EndpointType.VIDEO_GENERATION,
            model_name="sora-2",
            extra=[
                ("size", "1280x720"),
                ("seconds", 8),
                ("seed", 42),
            ],
        )
        endpoint = create_endpoint_with_mock_transport(
            VideoGenerationEndpoint, model_endpoint_with_extra
        )

        turn = Turn(
            texts=[Text(contents=["A dog running"])],
            model="sora-2",
        )
        request_info = create_request_info(
            model_endpoint=model_endpoint_with_extra, turns=[turn]
        )

        payload = endpoint.format_payload(request_info)

        assert payload["size"] == "1280x720"
        assert payload["seconds"] == 8
        assert payload["seed"] == 42

    def test_format_payload_model_from_turn(self, endpoint, model_endpoint):
        """Test that turn model overrides endpoint model."""
        turn = Turn(
            texts=[Text(contents=["A tree"])],
            model="custom-model",
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["model"] == "custom-model"

    def test_format_payload_no_turns_raises(self, endpoint, model_endpoint):
        """Test that missing turns raises ValueError."""
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[])

        with pytest.raises(ValueError, match="requires at least one turn"):
            endpoint.format_payload(request_info)

    def test_format_payload_no_text_raises(self, endpoint, model_endpoint):
        """Test that missing text raises ValueError."""
        turn = Turn(texts=[], model="sora-2")
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        with pytest.raises(ValueError, match="requires text prompt"):
            endpoint.format_payload(request_info)

    def test_format_payload_empty_text_contents_raises(self, endpoint, model_endpoint):
        """Test that empty text contents raises ValueError."""
        turn = Turn(texts=[Text(contents=[])], model="sora-2")
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        with pytest.raises(ValueError, match="requires text prompt"):
            endpoint.format_payload(request_info)

    # ===== parse_response tests =====

    def test_parse_response_queued_status(self, endpoint):
        """Test parsing initial queued job response."""
        json_data = {
            "id": "video_123",
            "object": "video",
            "model": "sora-2",
            "status": "queued",
            "progress": 0,
            "created_at": 1712697600,
            "size": "1280x720",
            "seconds": "8",
        }
        mock_response = create_mock_response(json_data=json_data)

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert parsed.data.video_id == "video_123"
        assert parsed.data.status == "queued"
        assert parsed.data.progress == 0
        assert parsed.data.size == "1280x720"
        assert parsed.data.seconds == "8"
        assert parsed.data.url is None

    def test_parse_response_in_progress_status(self, endpoint):
        """Test parsing in-progress job response."""
        json_data = {
            "id": "video_123",
            "object": "video",
            "model": "sora-2",
            "status": "in_progress",
            "progress": 45,
            "created_at": 1712697600,
        }
        mock_response = create_mock_response(json_data=json_data)

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert parsed.data.status == "in_progress"
        assert parsed.data.progress == 45

    def test_parse_response_completed_status(self, endpoint):
        """Test parsing completed job response."""
        json_data = {
            "id": "video_123",
            "object": "video",
            "model": "sora-2",
            "status": "completed",
            "progress": 100,
            "created_at": 1712697600,
            "completed_at": 1712697660,
            "size": "1280x720",
            "seconds": "8",
            "url": "http://localhost:30010/v1/videos/video_123/content",
            "inference_time_s": 45.2,
        }
        mock_response = create_mock_response(json_data=json_data)

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert parsed.data.video_id == "video_123"
        assert parsed.data.status == "completed"
        assert parsed.data.progress == 100
        assert parsed.data.url == "http://localhost:30010/v1/videos/video_123/content"
        assert parsed.data.completed_at == 1712697660
        assert parsed.data.inference_time_s == 45.2

    def test_parse_response_failed_status(self, endpoint):
        """Test parsing failed job response."""
        json_data = {
            "id": "video_123",
            "object": "video",
            "model": "sora-2",
            "status": "failed",
            "progress": 30,
            "created_at": 1712697600,
            "error": {
                "code": "generation_failed",
                "message": "Failed to generate video due to content policy",
            },
        }
        mock_response = create_mock_response(json_data=json_data)

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert parsed.data.status == "failed"
        assert parsed.data.error is not None
        assert parsed.data.error["code"] == "generation_failed"

    def test_parse_response_with_usage(self, endpoint):
        """Test parsing response with usage information."""
        json_data = {
            "id": "video_123",
            "status": "completed",
            "url": "http://localhost/video.mp4",
            "usage": {
                "prompt_tokens": 50,
                "total_tokens": 50,
            },
        }
        mock_response = create_mock_response(json_data=json_data)

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert parsed.usage is not None
        assert parsed.usage.prompt_tokens == 50

    def test_parse_response_perf_ns(self, endpoint):
        """Test that perf_ns is preserved."""
        json_data = {"id": "video_123", "status": "queued"}
        mock_response = create_mock_response(perf_ns=999888777, json_data=json_data)

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert parsed.perf_ns == 999888777

    def test_parse_response_no_json(self, endpoint):
        """Test parsing when get_json returns None."""
        mock_response = create_mock_response(json_data=None)
        mock_response.get_raw.return_value = ""

        parsed = endpoint.parse_response(mock_response)

        assert parsed is None

    def test_parse_response_extra_fields_ignored(self, endpoint):
        """Test that extra fields in response are ignored."""
        json_data = {
            "id": "video_123",
            "status": "completed",
            "url": "http://localhost/video.mp4",
            "extra_field": "ignored",
            "peak_memory_mb": 24576.0,
        }
        mock_response = create_mock_response(json_data=json_data)

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert parsed.data.video_id == "video_123"
        assert parsed.data.url == "http://localhost/video.mp4"
