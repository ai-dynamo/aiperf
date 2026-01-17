# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Literal

from pydantic import BaseModel as PydanticBaseModel
from pydantic import ConfigDict

# ============================================================================
# Base Models (for request parsing only)
# ============================================================================


class BaseModel(PydanticBaseModel):
    """Base model with common configuration for request parsing."""

    model_config = ConfigDict(extra="allow", exclude_none=True)


# ============================================================================
# Request Models
# ============================================================================


class Message(BaseModel):
    """Represents a chat message with role and content."""

    role: str
    content: str | list[dict[str, Any]]


class BaseCompletionRequest(BaseModel):
    """Base request model for completion endpoints with common parameters."""

    model: str
    stream: bool = False
    stream_options: dict[str, Any] | None = None
    max_tokens: int | None = None
    ignore_eos: bool = False
    min_tokens: int | None = None

    @property
    def include_usage(self) -> bool:
        """Check if usage statistics should be included in streaming response."""
        return bool(self.stream_options and self.stream_options.get("include_usage"))


class ChatCompletionRequest(BaseCompletionRequest):
    """Request model for chat completion endpoints."""

    messages: list[Message]
    max_completion_tokens: int | None = None
    reasoning_effort: Literal["low", "medium", "high"] | None = None

    @property
    def max_output_tokens(self) -> int | None:
        """Get max output tokens from either max_completion_tokens or max_tokens field."""
        return self.max_completion_tokens or self.max_tokens


class CompletionRequest(BaseCompletionRequest):
    """Request model for text completion endpoints."""

    prompt: str | list[str]
    reasoning_effort: Literal["low", "medium", "high"] | None = None

    @property
    def prompt_text(self) -> str:
        """Convert prompt to single text string (join array with newlines)."""
        if isinstance(self.prompt, str):
            return self.prompt
        return "\n".join(str(p) for p in self.prompt if p)


class EmbeddingRequest(BaseModel):
    """Request model for embedding endpoints."""

    model: str
    input: str | list[str]

    @property
    def inputs(self) -> list[str]:
        """Get inputs as list (normalizes single string to list)."""
        return (
            [self.input]
            if isinstance(self.input, str)
            else [str(x) for x in self.input]
        )


class RankingRequest(BaseModel):
    """Request model for NIM ranking endpoints."""

    model: str
    query: dict[str, str]
    passages: list[dict[str, str]]

    @property
    def query_text(self) -> str:
        """Extract query text from query dict."""
        return self.query.get("text", "")

    @property
    def passage_texts(self) -> list[str]:
        """Extract all passage texts from passages list."""
        return [p.get("text", "") for p in self.passages]


class HFTEIRerankRequest(BaseModel):
    """Request model for HuggingFace TEI /rerank endpoint."""

    query: str
    texts: list[str] | None = None
    documents: list[str] | None = None
    model: str = "tei-reranker"

    @property
    def query_text(self) -> str:
        return self.query

    @property
    def passage_texts(self) -> list[str]:
        return self.texts or self.documents or []


class CohereRerankRequest(BaseModel):
    """Request model for Cohere /v2/rerank endpoint."""

    query: str
    documents: list[str]
    model: str = "cohere-reranker"

    @property
    def query_text(self) -> str:
        return self.query

    @property
    def passage_texts(self) -> list[str]:
        return self.documents


class TGIParameters(BaseModel):
    """Parameters for HuggingFace TGI generation."""

    max_new_tokens: int = 50


class TGIGenerateRequest(BaseModel):
    """Request model for HuggingFace TGI /generate and /generate_stream endpoints.

    TGI API format:
    - Request: {"inputs": "...", "parameters": {"max_new_tokens": N}}
    - Non-streaming response: {"generated_text": "..."}
    - Streaming response: {"token": {"text": "..."}} per token, then {"generated_text": "..."}
    """

    inputs: str | None = None
    parameters: TGIParameters = TGIParameters()

    # Internal fields for mock server compatibility (not part of TGI API)
    model: str = "tgi"
    ignore_eos: bool = False
    min_tokens: int | None = None

    @property
    def prompt_text(self) -> str:
        return self.inputs or "Hello!"

    @property
    def max_tokens(self) -> int | None:
        return self.parameters.max_new_tokens


class ImageGenerationRequest(BaseModel):
    """Request model for OpenAI /v1/images/generations endpoint."""

    prompt: str
    model: str = "black-forest-labs/FLUX.1-dev"
    n: int = 1
    response_format: Literal["url", "b64_json"] = "b64_json"
    stream: bool = False
    size: str | None = None
    quality: str | None = None
    style: str | None = None


class SolidoRAGRequest(BaseModel):
    """Request model for SOLIDO /rag/api/prompt endpoint."""

    query: list[str]
    filters: dict[str, Any] = {}
    inference_model: str = "default-model"

    # Internal fields for mock server compatibility (not part of SOLIDO API)
    model: str = "solido-rag"
    ignore_eos: bool = False
    min_tokens: int | None = None


# ============================================================================
# Request Type Union
# ============================================================================

RequestT = (
    ChatCompletionRequest
    | CompletionRequest
    | EmbeddingRequest
    | RankingRequest
    | HFTEIRerankRequest
    | CohereRerankRequest
    | TGIGenerateRequest
    | ImageGenerationRequest
    | SolidoRAGRequest
)
