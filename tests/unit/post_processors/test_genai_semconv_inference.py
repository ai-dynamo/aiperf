# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Table-driven tests for GenAI provider auto-inference.

Covers every well-known host pattern in the design table, explicit override
precedence, unknown hosts, malformed URLs, and empty URL lists.

# Feature: otel-mlflow-telemetry-takeover, Requirement 14.5
"""

from __future__ import annotations

import pytest
from pytest import param

from aiperf.common.config import EndpointConfig, UserConfig
from aiperf.plugin.enums import EndpointType
from aiperf.post_processors.strategies.genai_semconv import infer_provider_name


def _make_user_config(
    urls: list[str] | None = None,
    gen_ai_provider: str | None = None,
) -> UserConfig:
    """Build a minimal UserConfig with the given URLs and optional provider override."""
    endpoint_kwargs: dict = {
        "model_names": ["test-model"],
        "type": EndpointType.CHAT,
        "streaming": False,
    }
    if urls is not None:
        endpoint_kwargs["urls"] = urls

    user_config_kwargs: dict = {
        "endpoint": EndpointConfig(**endpoint_kwargs),
        "gen_ai_provider": gen_ai_provider,
    }
    # gen_ai_provider is an OTel-only flag: the validator requires --otel-url
    # whenever any OTel-only flag is set, so pair them in test configs.
    if gen_ai_provider is not None:
        user_config_kwargs["otel_url"] = "http://localhost:4318"
    config = UserConfig(**user_config_kwargs)
    return config


class TestProviderHostInference:
    """Verify every well-known host pattern maps to the correct provider."""

    @pytest.mark.parametrize(
        ("url", "expected_provider"),
        [
            param("https://api.openai.com/v1/chat/completions", "openai", id="openai"),
            param("https://api.anthropic.com/v1/messages", "anthropic", id="anthropic"),
            param("https://api.deepseek.com/v1/chat", "deepseek", id="deepseek"),
            param("https://api.mistral.ai/v1/chat/completions", "mistral_ai", id="mistral_ai"),
            param("https://api.cohere.ai/v1/generate", "cohere", id="cohere-ai"),
            param("https://api.cohere.com/v1/generate", "cohere", id="cohere-com"),
            param("https://api.x.ai/v1/chat/completions", "x_ai", id="x_ai"),
            param("https://api.groq.com/openai/v1/chat/completions", "groq", id="groq"),
            param("https://api.perplexity.ai/chat/completions", "perplexity", id="perplexity"),
            param(
                "https://generativelanguage.googleapis.com/v1beta/models",
                "gcp.gemini",
                id="gcp-gemini",
            ),
            param(
                "https://us-central1-aiplatform.googleapis.com/v1/projects/my-proj",
                "gcp.vertex_ai",
                id="gcp-vertex-ai",
            ),
            param(
                "https://bedrock-runtime.us-east-1.amazonaws.com/model/invoke",
                "aws.bedrock",
                id="aws-bedrock",
            ),
            param(
                "https://my-resource.openai.azure.com/openai/deployments/gpt-4",
                "azure.ai.openai",
                id="azure-openai",
            ),
            param(
                "https://my-endpoint.services.ai.azure.com/models/chat/completions",
                "azure.ai.inference",
                id="azure-ai-inference",
            ),
            param(
                "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation",
                "ibm.watsonx.ai",
                id="ibm-watsonx",
            ),
        ],
    )  # fmt: skip
    def test_known_host_patterns(self, url: str, expected_provider: str) -> None:
        config = _make_user_config(urls=[url])
        assert infer_provider_name(config) == expected_provider


class TestExplicitOverrideWins:
    """Explicit --gen-ai-provider override takes precedence over host inference."""

    @pytest.mark.parametrize(
        ("url", "override", "expected"),
        [
            param(
                "https://api.openai.com/v1/chat",
                "custom_provider",
                "custom_provider",
                id="override-beats-openai-host",
            ),
            param(
                "https://api.anthropic.com/v1/messages",
                "my_corp",
                "my_corp",
                id="override-beats-anthropic-host",
            ),
            param(
                "https://unknown-host.example.com/v1/chat",
                "explicit_value",
                "explicit_value",
                id="override-beats-unknown-host",
            ),
        ],
    )  # fmt: skip
    def test_explicit_override_wins(
        self, url: str, override: str, expected: str
    ) -> None:
        config = _make_user_config(urls=[url], gen_ai_provider=override)
        assert infer_provider_name(config) == expected


class TestUnknownHostFallback:
    """Unknown hosts that match no pattern return '_OTHER'."""

    @pytest.mark.parametrize(
        "url",
        [
            param("https://my-custom-server.example.com/v1/chat", id="custom-domain"),
            param("https://localhost:8000/v1/completions", id="localhost"),
            param("https://10.0.0.1:8080/v1/chat", id="ip-address"),
            param("https://internal.corp.net/llm/v1/chat", id="internal-corp"),
            param("https://vllm.my-cluster.svc.local:8000/v1/chat", id="k8s-service"),
        ],
    )  # fmt: skip
    def test_unknown_host_returns_other(self, url: str) -> None:
        config = _make_user_config(urls=[url])
        assert infer_provider_name(config) == "_OTHER"


class TestMalformedURLFallback:
    """Malformed URLs that cannot be parsed return '_OTHER'."""

    @pytest.mark.parametrize(
        "url",
        [
            param("://missing-scheme", id="missing-scheme"),
            param("not a url at all", id="garbage-string"),
            param("", id="empty-string"),
            param("   ", id="whitespace-only"),
        ],
    )  # fmt: skip
    def test_malformed_url_returns_other(self, url: str) -> None:
        config = _make_user_config(urls=[url])
        assert infer_provider_name(config) == "_OTHER"


class TestEmptyURLList:
    """When no URLs are available, provider inference returns '_OTHER'."""

    def test_default_url_returns_other(self) -> None:
        # Default URL is localhost:8000 which matches no known provider
        config = _make_user_config()
        assert infer_provider_name(config) == "_OTHER"
