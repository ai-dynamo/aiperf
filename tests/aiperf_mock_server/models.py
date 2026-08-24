# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Literal

from pydantic import BaseModel as PydanticBaseModel
from pydantic import ConfigDict, Field

# ============================================================================
# Base Models (for request parsing only)
# ============================================================================

CompletionPrompt = str | list[int] | list[list[int]] | list[str]


def flatten_completion_prompt_token_ids(prompt: CompletionPrompt) -> list[int] | None:
    """Return raw token IDs when a completions prompt is already tokenized."""
    if not isinstance(prompt, list):
        return None
    if all(isinstance(item, int) for item in prompt):
        return [int(item) for item in prompt]
    if all(
        isinstance(item, list) and all(isinstance(token_id, int) for token_id in item)
        for item in prompt
    ):
        return [int(token_id) for item in prompt for token_id in item]
    return None


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
    mock_first_chunk_tokens: int = Field(
        default=1,
        description="Test knob: bundle the first N output tokens into the first "
        "streamed chunk to emulate a server (e.g. TRT-LLM stream-interval) whose "
        "first content chunk carries more than one token. Defaults to 1.",
    )

    @property
    def include_usage(self) -> bool:
        """Check if usage statistics should be included in streaming response."""
        return bool(self.stream_options and self.stream_options.get("include_usage"))

    @property
    def continuous_usage_stats(self) -> bool:
        """Check if per-chunk (cumulative) usage should be reported on every chunk."""
        return bool(
            self.stream_options and self.stream_options.get("continuous_usage_stats")
        )


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

    prompt: CompletionPrompt
    reasoning_effort: Literal["low", "medium", "high"] | None = None

    @property
    def prompt_text(self) -> str:
        """Convert prompt to single text string (join array with newlines)."""
        prompt_token_ids = flatten_completion_prompt_token_ids(self.prompt)
        if prompt_token_ids is not None:
            return " ".join(str(token_id) for token_id in prompt_token_ids)
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


class ImageRetrievalInput(BaseModel):
    """Single image input for NIM image retrieval."""

    type: str
    url: str


class ImageRetrievalRequest(BaseModel):
    """Request model for NIM image retrieval /v1/infer endpoint."""

    input: list[ImageRetrievalInput]


class SolidoRAGRequest(BaseModel):
    """Request model for SOLIDO /rag/api/prompt endpoint."""

    query: list[str]
    filters: dict[str, Any] = {}
    inference_model: str = "default-model"

    # Internal fields for mock server compatibility (not part of SOLIDO API)
    model: str = "solido-rag"
    ignore_eos: bool = False
    min_tokens: int | None = None


class AnthropicMessage(BaseModel):
    """Represents an Anthropic message with role and content."""

    role: str
    content: str | list[dict[str, Any]]


class AnthropicMessagesRequest(BaseCompletionRequest):
    """Request model for Anthropic /v1/messages endpoint."""

    messages: list[AnthropicMessage]
    # Required by the real API and by Dynamo's /v1/messages (u32, request
    # fails deserialization without it) - enforce the same contract so a
    # client regression that stops sending max_tokens fails e2e.
    max_tokens: int
    system: str | list[dict[str, Any]] | None = None
    cache_control: dict[str, Any] | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: dict[str, Any] | None = None
    thinking: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    stop_sequences: list[str] | None = None

    @property
    def max_output_tokens(self) -> int | None:
        """Get max output tokens."""
        return self.max_tokens


class ResponsesRequest(BaseModel):
    """Request model for OpenAI's /v1/responses endpoint.

    The Responses API takes its prompt under `input` (which may be a string,
    a list of strings, or a list of content-block dicts) and caps generation
    via `max_output_tokens` rather than the chat API's `max_completion_tokens`.
    Modeled here so the recorder can capture the real payload instead of the
    synthetic ChatCompletionRequest the latency simulator drives off of.
    """

    model: str = Field(description="Model identifier.")
    input: str | list[Any] = Field(
        default="",
        description="Prompt input — string, list of strings, or list of content-block dicts.",
    )
    max_output_tokens: int | None = Field(
        default=None,
        description="Maximum number of output tokens to generate (the Responses-API name for the OSL cap).",
    )
    stream: bool = Field(default=False, description="Whether to stream the response.")
    reasoning_effort: Literal["low", "medium", "high"] | None = Field(
        default=None,
        description="Reasoning effort level for models that support extended thinking.",
    )

    # Mirrors BaseCompletionRequest so recorder/simulator share field semantics
    # when the client supplies them via extras.
    min_tokens: int | None = Field(
        default=None, description="Minimum number of tokens to generate."
    )
    ignore_eos: bool = Field(
        default=False,
        description="Whether to ignore the EOS token and continue generating up to max_output_tokens.",
    )

    @property
    def prompt_text(self) -> str:
        """Flatten `input` (str | list[str|dict]) into a single string.

        Matches the previous `_extract_responses_prompt` helper that lived in
        the handler; moved here so the recorder and tokenizer dispatch sites
        share one source of truth.
        """
        return _flatten_responses_input(self.input)


def _flatten_responses_input(input_value: Any) -> str:
    """Walk the Responses-API `input` shape and concatenate into text."""
    if isinstance(input_value, str):
        return input_value
    if isinstance(input_value, list):
        parts: list[str] = []
        for item in input_value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                content = item.get("content", "")
                if isinstance(content, str):
                    parts.append(content)
                elif isinstance(content, list):
                    parts.extend(
                        str(part.get("text", ""))
                        for part in content
                        if isinstance(part, dict)
                    )
        return "\n".join(part for part in parts if part)
    return str(input_value)


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
    | ImageRetrievalRequest
    | SolidoRAGRequest
    | AnthropicMessagesRequest
    | ResponsesRequest
)
