# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Literal, TypeVar

from pydantic import BaseModel as PydanticBaseModel
from pydantic import ConfigDict

# ============================================================================
# Base Models
# ============================================================================


class BaseModel(PydanticBaseModel):
    """Base model with common configuration for all Pydantic models."""

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
    """Request model for ranking/reranking endpoints."""

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

    @property
    def total_tokens(self) -> int:
        """Get total tokens from query and passage texts."""
        return len(self.query_text) + sum(len(p) for p in self.passage_texts)


# ============================================================================
# Response Models
# ============================================================================


class Usage(BaseModel):
    """Token usage statistics for API requests and responses."""

    prompt_tokens: int
    completion_tokens: int | None = None
    total_tokens: int | None = None
    completion_tokens_details: dict[str, int] | None = None


class BaseResponse(BaseModel):
    """Base response model with common fields for all completion responses."""

    id: str
    created: int
    model: str
    usage: Usage | None = None


class BaseChoice(BaseModel):
    """Base choice model with common fields for completion choices."""

    index: int
    finish_reason: str | None = None


class ChatMessage(BaseModel):
    """Message model for chat completions."""

    role: str
    content: str
    reasoning_content: str | None = None


class ChatDelta(BaseModel):
    """Delta model for streaming chat completions."""

    role: str | None = None
    content: str | None = None
    reasoning_content: str | None = None


class ChatChoice(BaseChoice):
    """Choice model for chat completion responses."""

    message: ChatMessage


class TextChoice(BaseChoice):
    """Choice model for text completion responses."""

    text: str


class ChatStreamChoice(BaseChoice):
    """Choice model for streaming chat completion responses."""

    delta: ChatDelta


class TextStreamChoice(BaseChoice):
    """Choice model for streaming text completion responses."""

    text: str


class ChatCompletionResponse(BaseResponse):
    """Response model for chat completion endpoints."""

    object: Literal["chat.completion"] = "chat.completion"
    choices: list[ChatChoice]


class TextCompletionResponse(BaseResponse):
    """Response model for text completion endpoints."""

    object: Literal["text_completion"] = "text_completion"
    choices: list[TextChoice]


class ChatStreamCompletionResponse(BaseResponse):
    """Response model for streaming chat completion endpoints."""

    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    choices: list[ChatStreamChoice]


class TextStreamCompletionResponse(BaseResponse):
    """Response model for streaming text completion endpoints."""

    object: Literal["text_completion.chunk"] = "text_completion.chunk"
    choices: list[TextStreamChoice]


class Embedding(BaseModel):
    """Represents a single embedding vector with its index."""

    object: Literal["embedding"] = "embedding"
    index: int
    embedding: list[float]


class EmbeddingResponse(BaseModel):
    """Response model for embedding endpoints."""

    object: Literal["list"] = "list"
    data: list[Embedding]
    model: str
    usage: Usage


class Ranking(BaseModel):
    """Represents a single ranking result with relevance score."""

    index: int
    relevance_score: float


class RankingResponse(BaseModel):
    """Response model for ranking/reranking endpoints."""

    id: str
    object: Literal["rankings"] = "rankings"
    model: str
    rankings: list[Ranking]
    usage: Usage


# ============================================================================
# Request Type Variables
# ============================================================================

RequestT = ChatCompletionRequest | CompletionRequest | EmbeddingRequest | RankingRequest
RequestTypeVarT = TypeVar("RequestTypeVarT", bound=RequestT)
