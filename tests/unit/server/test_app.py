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
