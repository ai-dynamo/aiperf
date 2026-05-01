# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for app module."""


class TestAPIEndpoints:
    """Tests for FastAPI endpoints."""

    def test_root_endpoint(self, test_client):
        response = test_client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "AIPerf Mock Server"
        assert data["version"] == "2.0.0"

    def test_health_endpoint(self, test_client):
        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "config" in data

    def test_chat_completions_endpoint(self, test_client):
        response = test_client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "test-model"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert "usage" in data

    def test_completions_endpoint(self, test_client):
        response = test_client.post(
            "/v1/completions",
            json={"model": "test-model", "prompt": "Hello"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "test-model"
        assert len(data["choices"]) == 1
        assert "usage" in data

    def test_embeddings_endpoint(self, test_client):
        response = test_client.post(
            "/v1/embeddings",
            json={"model": "test-model", "input": "test text"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1

    def test_responses_endpoint(self, test_client):
        response = test_client.post(
            "/v1/responses",
            json={"model": "test-model", "input": "Say hello."},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "response"
        assert data["output_text"]
        assert "usage" in data

    def test_video_generation_endpoint(self, test_client):
        response = test_client.post(
            "/v1/videos",
            data={"model": "test-model", "prompt": "A short test video."},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "video"
        assert data["status"] == "completed"
        assert data["url"].endswith("/content")

    def test_video_generation_poll_endpoint(self, test_client):
        response = test_client.get("/v1/videos/video-test-123")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "video-test-123"
        assert data["status"] == "completed"

    def test_video_generation_content_endpoint(self, test_client):
        response = test_client.get("/v1/videos/video-test-123/content")
        assert response.status_code == 200
        assert response.content.startswith(b"mock-video")

    def test_image_retrieval_documented_endpoint(self, test_client):
        response = test_client.post(
            "/v1/infer",
            json={"input": [{"type": "image_url", "url": "data:image/png;base64,abc"}]},
        )
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert data["data"][0]["bounding_boxes"]

    def test_rankings_endpoint(self, test_client):
        response = test_client.post(
            "/v1/ranking",
            json={
                "model": "test-model",
                "query": {"text": "test query"},
                "passages": [{"text": "passage 1"}, {"text": "passage 2"}],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "rankings"
        assert len(data["rankings"]) == 2

    def test_dcgm_metrics_invalid_instance(self, test_client):
        response = test_client.get("/dcgm3/metrics")
        assert response.status_code == 404

    def test_image_generation_endpoint(self, test_client):
        response = test_client.post(
            "/v1/images/generations",
            json={
                "model": "black-forest-labs/FLUX.1-dev",
                "prompt": "A beautiful sunset over mountains",
                "n": 1,
                "response_format": "b64_json",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert len(data["data"]) == 1
        assert "b64_json" in data["data"][0]
        assert "usage" in data

    def test_image_generation_multiple_images(self, test_client):
        response = test_client.post(
            "/v1/images/generations",
            json={
                "model": "black-forest-labs/FLUX.1-dev",
                "prompt": "Test prompt",
                "n": 3,
                "size": "512x512",
                "quality": "standard",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 3
        assert data["size"] == "512x512"
        assert data["quality"] == "standard"

    def test_solido_rag_endpoint(self, test_client):
        response = test_client.post(
            "/rag/api/prompt",
            json={
                "query": ["What is SOLIDO?"],
                "filters": {"family": "Solido", "tool": "SDE"},
                "inference_model": "test-model",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "content" in data
        assert "sources" in data
        assert isinstance(data["sources"], list)
        assert data["inference_model"] == "test-model"
        assert data["filters"] == {"family": "Solido", "tool": "SDE"}

    def test_solido_rag_with_multiple_queries(self, test_client):
        response = test_client.post(
            "/rag/api/prompt",
            json={
                "query": ["Query 1", "Query 2", "Query 3"],
                "filters": {"family": "Test"},
                "inference_model": "rag-model",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "content" in data
        assert "sources" in data
        # Should generate sources based on queries
        assert len(data["sources"]) == 3
