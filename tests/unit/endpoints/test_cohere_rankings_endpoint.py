# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import logging

import pytest

from aiperf.common.models import Audio, Image, Text, Turn, Video
from aiperf.endpoints.cohere_rankings import CohereRankingsEndpoint
from aiperf.plugin import plugins
from aiperf.plugin.enums import EndpointType
from aiperf.plugin.schema.schemas import EndpointMetadata
from tests.unit.endpoints.conftest import (
    create_endpoint_with_mock_transport,
    create_model_endpoint,
    create_request_info,
)


class TestCohereRankingsEndpoint:
    """Unit tests for CohereRankingsEndpoint."""

    @pytest.fixture
    def model_endpoint(self):
        """Create a test ModelEndpointInfo for Cohere rankings."""
        return create_model_endpoint(EndpointType.COHERE_RANKINGS)

    @pytest.fixture
    def converter(self, model_endpoint):
        """Create a CohereRankingsEndpoint instance."""
        return create_endpoint_with_mock_transport(
            CohereRankingsEndpoint, model_endpoint
        )

    @pytest.fixture
    def basic_turn(self):
        """Create a basic turn with query and documents."""
        return Turn(
            texts=[
                Text(name="query", contents=["What is deep learning?"]),
                Text(
                    name="passages",  # kept as 'passages' since input dataset uses that
                    contents=[
                        "Deep learning uses neural networks.",
                        "Bananas are yellow.",
                        "Machine learning is related to AI.",
                    ],
                ),
            ],
            model="test-model",
        )

    def test_format_payload_basic(self, converter, model_endpoint, basic_turn):
        """Test basic payload formatting with query and passages."""
        payload = converter.format_payload(
            create_request_info(model_endpoint=model_endpoint, turns=[basic_turn])
        )

        assert payload["model"] == "test-model"
        assert payload["query"] == "What is deep learning?"
        assert len(payload["documents"]) == 3
        assert "Deep learning uses neural networks." in payload["documents"][0]

    def test_build_payload_legacy_signature(self, converter):
        """Test that direct text-only build_payload calls remain compatible."""
        payload = converter.build_payload("What is AI?", ["AI passage"], "test-model")

        assert payload == {
            "model": "test-model",
            "query": "What is AI?",
            "documents": ["AI passage"],
        }

    def test_format_payload_single_passage(self, converter, model_endpoint):
        """Test payload formatting with single passage."""
        turn = Turn(
            texts=[
                Text(name="query", contents=["What is Python?"]),
                Text(name="passages", contents=["Python is a programming language"]),
            ],
            model="test-model",
        )

        payload = converter.format_payload(
            create_request_info(model_endpoint=model_endpoint, turns=[turn])
        )

        assert payload["query"] == "What is Python?"
        assert len(payload["documents"]) == 1
        assert payload["documents"][0] == "Python is a programming language"

    def test_format_payload_multiple_query_contents(
        self, converter, model_endpoint, caplog
    ):
        """Test with multiple contents in query text (uses first one)."""
        turn = Turn(
            texts=[
                Text(name="query", contents=["First query", "Second query"]),
                Text(name="passages", contents=["Some passage"]),
            ],
            model="test-model",
        )

        with caplog.at_level(logging.WARNING):
            payload = converter.format_payload(
                create_request_info(model_endpoint=model_endpoint, turns=[turn])
            )

        assert "Multiple query texts found" in caplog.text
        assert payload["query"] == "First query"

    def test_format_payload_no_passages(self, converter, model_endpoint, caplog):
        """Test with query but no passages (should warn)."""
        turn = Turn(
            texts=[Text(name="query", contents=["What is AI?"])],
            model="test-model",
        )

        with caplog.at_level(logging.WARNING):
            payload = converter.format_payload(
                create_request_info(model_endpoint=model_endpoint, turns=[turn])
            )

        assert "no passages to rank" in caplog.text
        assert payload["query"] == "What is AI?"
        assert payload["documents"] == []

    def test_format_payload_single_passage_with_image(self, converter, model_endpoint):
        """Test multimodal document formatting with one passage and one image."""
        image_url = "data:image/png;base64,img1"
        turn = Turn(
            texts=[
                Text(name="query", contents=["Find the relevant image"]),
                Text(name="passages", contents=["A beach at sunset"]),
            ],
            images=[Image(contents=[image_url])],
            model="test-model",
        )

        payload = converter.format_payload(
            create_request_info(model_endpoint=model_endpoint, turns=[turn])
        )

        assert payload["documents"] == [
            {
                "content": [
                    {"type": "text", "text": "A beach at sunset"},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ]
            }
        ]

    def test_format_payload_multiple_index_paired_modalities(
        self, converter, model_endpoint
    ):
        """Test that passages, images, and videos are paired by index."""
        images = ["data:image/png;base64,img1", "data:image/png;base64,img2"]
        videos = ["data:video/mp4;base64,vid1", "data:video/mp4;base64,vid2"]
        turn = Turn(
            texts=[
                Text(name="query", contents=["Find relevant media"]),
                Text(name="passages", contents=["First document", "Second document"]),
            ],
            images=[Image(contents=images)],
            videos=[Video(contents=videos)],
            model="test-model",
        )

        payload = converter.format_payload(
            create_request_info(model_endpoint=model_endpoint, turns=[turn])
        )

        assert payload["documents"] == [
            {
                "content": [
                    {"type": "text", "text": "First document"},
                    {"type": "image_url", "image_url": {"url": images[0]}},
                    {"type": "video_url", "video_url": {"url": videos[0]}},
                ]
            },
            {
                "content": [
                    {"type": "text", "text": "Second document"},
                    {"type": "image_url", "image_url": {"url": images[1]}},
                    {"type": "video_url", "video_url": {"url": videos[1]}},
                ]
            },
        ]

    def test_format_payload_image_only_documents(self, converter, model_endpoint):
        """Test image-only documents."""
        images = ["data:image/png;base64,img1", "data:image/png;base64,img2"]
        turn = Turn(
            texts=[Text(name="query", contents=["Find relevant images"])],
            images=[Image(contents=images)],
            model="test-model",
        )

        payload = converter.format_payload(
            create_request_info(model_endpoint=model_endpoint, turns=[turn])
        )

        assert payload["documents"] == [
            {"content": [{"type": "image_url", "image_url": {"url": images[0]}}]},
            {"content": [{"type": "image_url", "image_url": {"url": images[1]}}]},
        ]

    def test_format_payload_video_only_documents(self, converter, model_endpoint):
        """Test video-only documents."""
        videos = ["data:video/mp4;base64,vid1", "data:video/mp4;base64,vid2"]
        turn = Turn(
            texts=[Text(name="query", contents=["Find relevant videos"])],
            videos=[Video(contents=videos)],
            model="test-model",
        )

        payload = converter.format_payload(
            create_request_info(model_endpoint=model_endpoint, turns=[turn])
        )

        assert payload["documents"] == [
            {"content": [{"type": "video_url", "video_url": {"url": videos[0]}}]},
            {"content": [{"type": "video_url", "video_url": {"url": videos[1]}}]},
        ]

    def test_format_payload_multimodal_count_mismatch_raises(
        self, converter, model_endpoint
    ):
        """Test that non-zero modality counts must match."""
        turn = Turn(
            texts=[
                Text(name="query", contents=["Find relevant media"]),
                Text(name="passages", contents=["First document", "Second document"]),
            ],
            images=[Image(contents=["data:image/png;base64,img1"])],
            model="test-model",
        )

        with pytest.raises(ValueError, match="matching non-zero counts"):
            converter.format_payload(
                create_request_info(model_endpoint=model_endpoint, turns=[turn])
            )

    def test_format_payload_audio_rejected(self, converter, model_endpoint):
        """Test that audio input is rejected for Cohere rankings."""
        turn = Turn(
            texts=[Text(name="query", contents=["Find relevant audio"])],
            audios=[Audio(contents=["wav,b64audio"])],
            model="test-model",
        )

        with pytest.raises(ValueError, match="does not support audio input"):
            converter.format_payload(
                create_request_info(model_endpoint=model_endpoint, turns=[turn])
            )

    def test_metadata_declares_multimodal_support(self):
        """Test that Cohere rankings metadata declares image and video support."""
        metadata = plugins.get_endpoint_metadata(EndpointType.COHERE_RANKINGS)
        assert isinstance(metadata, EndpointMetadata)
        assert metadata.supports_images is True
        assert metadata.supports_videos is True
        assert metadata.supports_audio is False

    def test_format_payload_no_query(self, converter, model_endpoint):
        """Test with no query text (should raise error)."""
        turn = Turn(
            texts=[Text(name="passages", contents=["Some passage"])],
            model="test-model",
        )

        with pytest.raises(ValueError, match="requires a text with name 'query'"):
            converter.format_payload(
                create_request_info(model_endpoint=model_endpoint, turns=[turn])
            )

    def test_format_payload_empty_query_contents(self, converter, model_endpoint):
        """Test with empty query contents (should raise error)."""
        turn = Turn(
            texts=[
                Text(name="query", contents=[]),
                Text(name="passages", contents=["Some passage"]),
            ],
            model="test-model",
        )

        with pytest.raises(ValueError, match="requires a text with name 'query'"):
            converter.format_payload(
                create_request_info(model_endpoint=model_endpoint, turns=[turn])
            )

    def test_format_payload_model_priority(self, converter, model_endpoint):
        """Test that turn model takes priority over endpoint model."""
        turn = Turn(
            texts=[
                Text(name="query", contents=["Test query"]),
                Text(name="passages", contents=["Test passage"]),
            ],
            model="turn-model",
        )

        payload = converter.format_payload(
            create_request_info(model_endpoint=model_endpoint, turns=[turn])
        )
        assert payload["model"] == "turn-model"

    def test_format_payload_fallback_model(self, converter, model_endpoint):
        """Test fallback to endpoint model when turn model is None."""
        turn = Turn(
            texts=[
                Text(name="query", contents=["Test query"]),
                Text(name="passages", contents=["Test passage"]),
            ],
            model=None,
        )

        payload = converter.format_payload(
            create_request_info(model_endpoint=model_endpoint, turns=[turn])
        )
        assert payload["model"] == model_endpoint.primary_model_name

    def test_extract_rankings(self, converter):
        """Test extraction of ranking results from API response."""
        mock_json = {
            "results": [
                {"index": 0, "relevance_score": 0.95},
                {"index": 2, "relevance_score": 0.10},
            ]
        }

        rankings = converter.extract_rankings(mock_json)
        assert len(rankings) == 2
        assert rankings[0]["index"] == 0
        assert rankings[0]["score"] == 0.95
        assert rankings[1]["index"] == 2
        assert rankings[1]["score"] == 0.10
