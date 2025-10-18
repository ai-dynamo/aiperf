# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from functools import lru_cache

from aiperf_mock_server.models import (
    ChatCompletionRequest,
    CompletionRequest,
    EmbeddingRequest,
    RankingRequest,
    RequestT,
    Usage,
)
from pydantic import BaseModel


class TokenizedText(BaseModel):
    """Tokenized text with metadata."""

    text: str
    tokens: list[str]
    prompt_token_count: int
    reasoning_tokens: int = 0
    reasoning_content_tokens: list[str] = []
    finish_reason: str = "stop"

    @property
    def count(self) -> int:
        """Get output token count."""
        return len(self.tokens)

    @property
    def content(self) -> str:
        """Get content as string."""
        return "".join(self.tokens)

    @property
    def reasoning_content(self) -> str | None:
        """Get reasoning content as string."""
        return (
            "".join(self.reasoning_content_tokens)
            if self.reasoning_content_tokens
            else None
        )

    def create_usage(self) -> Usage:
        """Create Usage object from tokenized text."""
        usage = Usage(
            prompt_tokens=self.prompt_token_count,
            completion_tokens=self.count,
            total_tokens=self.prompt_token_count + self.count,
        )
        if self.reasoning_tokens > 0:
            usage.completion_tokens_details = {
                "reasoning_tokens": self.reasoning_tokens
            }
        return usage


class ReasoningResult(BaseModel):
    """Result of reasoning token generation with budget management."""

    token_count: int
    content_tokens: list[str]
    remaining_budget: int | None


class TokenBudget(BaseModel):
    """Token budget calculation result."""

    total: int
    min_tokens: int
    max_tokens: int


class TokenGenerationParams(BaseModel):
    """Parameters for token generation."""

    prompt_tokens: list[str]
    prompt_token_count: int
    max_tokens: int | None
    min_tokens: int | None
    ignore_eos: bool

    def calculate_budget(self) -> TokenBudget:
        """Calculate min/max token budget for generation."""
        max_budget = self.max_tokens or max(self.prompt_token_count * 2, 16)

        if self.min_tokens is not None:
            min_budget = min(self.min_tokens, max_budget)
        else:
            min_budget = max(1, int(self.prompt_token_count * 0.8))
            min_budget = min(min_budget, max_budget)

        return TokenBudget(
            total=max_budget, min_tokens=min_budget, max_tokens=max_budget
        )


class ReasoningGenerationContext(BaseModel):
    """Context for reasoning token generation."""

    request: ChatCompletionRequest | CompletionRequest
    prompt_tokens: list[str]
    prompt_token_count: int
    max_tokens: int | None

    def calculate_default_budget(self) -> int:
        """Calculate total token budget when not explicitly specified."""
        return (
            self.max_tokens
            if self.max_tokens is not None
            else max(self.prompt_token_count * 2, 16)
        )

    def generate_reasoning_tokens(self) -> ReasoningResult:
        """Generate reasoning tokens if model supports it, managing budget."""
        # Check if model supports reasoning tokens
        model_lower = self.request.model.lower()
        is_reasoning_model = any(m in model_lower for m in ("gpt-oss", "qwen"))

        if not is_reasoning_model:
            return ReasoningResult(
                token_count=0, content_tokens=[], remaining_budget=self.max_tokens
            )

        # Calculate requested reasoning tokens based on effort
        effort_tokens = {"low": 100, "medium": 250, "high": 500}
        effort = self.request.reasoning_effort or "medium"
        requested_reasoning_tokens = effort_tokens.get(effort, 250)

        total_budget = self.calculate_default_budget()
        actual_reasoning_tokens = min(requested_reasoning_tokens, total_budget)

        reasoning_content_tokens = self._create_reasoning_content(
            actual_reasoning_tokens
        )

        return ReasoningResult(
            token_count=actual_reasoning_tokens,
            content_tokens=reasoning_content_tokens,
            remaining_budget=total_budget - actual_reasoning_tokens,
        )

    def _create_reasoning_content(self, num_tokens: int) -> list[str]:
        """Generate reasoning content tokens using reverse cycling pattern."""
        if not self.prompt_tokens or num_tokens == 0:
            return []
        return [
            self.prompt_tokens[
                (len(self.prompt_tokens) - 1 - i) % len(self.prompt_tokens)
            ]
            for i in range(num_tokens)
        ]


@lru_cache(maxsize=1024)
def _tokenize(text: str) -> tuple[str, ...]:
    """Tokenize text using character-based estimation (~4 chars per token).

    Splits text into chunks of approximately 4 characters,
    breaking on whitespace boundaries when possible for more natural tokens.
    """
    if not text:
        return ()

    tokens = []
    i = 0
    while i < len(text):
        end = min(i + 4, len(text))

        # Look ahead for whitespace to break naturally
        if end < len(text) and not text[end].isspace():
            for j in range(end, min(end + 2, len(text))):
                if text[j].isspace():
                    end = j + 1
                    break

        tokens.append(text[i:end])
        i = end

    return tuple(tokens)


class _Tokenizer:
    """Handles text tokenization with caching."""

    def tokenize(self, text: str) -> tuple[str, ...]:
        """Tokenize text with caching."""
        return _tokenize(text)

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenize(text))

    def tokenize_request(self, request: RequestT) -> TokenizedText:
        """Tokenize a request and return TokenizedText with usage."""
        text, max_tokens = self._extract_request_content(request)
        prompt_tokens = list(self.tokenize(text))
        prompt_token_count = len(prompt_tokens)

        # For embeddings and rankings, only count input tokens (no output generation)
        if isinstance(request, EmbeddingRequest | RankingRequest):
            return TokenizedText(
                text=text,
                tokens=[],
                prompt_token_count=prompt_token_count,
                reasoning_tokens=0,
                reasoning_content_tokens=[],
                finish_reason="stop",
            )

        # Handle empty prompts - can't generate tokens without source material
        if not prompt_tokens:
            return TokenizedText(
                text=text,
                tokens=[],
                prompt_token_count=0,
                reasoning_tokens=0,
                reasoning_content_tokens=[],
                finish_reason="stop",
            )

        reasoning_ctx = ReasoningGenerationContext(
            request=request,
            prompt_tokens=prompt_tokens,
            prompt_token_count=prompt_token_count,
            max_tokens=max_tokens,
        )
        reasoning_result = reasoning_ctx.generate_reasoning_tokens()

        params = TokenGenerationParams(
            prompt_tokens=prompt_tokens,
            prompt_token_count=prompt_token_count,
            max_tokens=reasoning_result.remaining_budget,
            min_tokens=request.min_tokens,
            ignore_eos=request.ignore_eos,
        )
        output_tokens, finish_reason = self._generate_output_tokens(params)

        return TokenizedText(
            text=text,
            tokens=output_tokens,
            prompt_token_count=prompt_token_count,
            reasoning_tokens=reasoning_result.token_count,
            reasoning_content_tokens=reasoning_result.content_tokens,
            finish_reason=finish_reason,
        )

    def _extract_request_content(self, request: RequestT) -> tuple[str, int | None]:
        """Extract text and max_tokens from request."""
        if isinstance(request, ChatCompletionRequest):
            text = self._extract_chat_messages(request.messages)
            return text, request.max_output_tokens
        elif isinstance(request, CompletionRequest):
            return request.prompt_text, request.max_tokens
        elif isinstance(request, EmbeddingRequest):
            # For embeddings, join all inputs into a single string for token counting
            text = "\n".join(request.inputs)
            return text, None
        elif isinstance(request, RankingRequest):
            # For rankings, count query + all passages
            text = request.query_text + "\n" + "\n".join(request.passage_texts)
            return text, None
        else:
            raise ValueError(f"Unsupported request type: {type(request)}")

    def _extract_chat_messages(self, messages: list) -> str:
        """Extract text content from chat messages."""
        texts = []
        for msg in messages:
            if msg.role != "user":
                continue
            content = msg.content
            if isinstance(content, str):
                texts.append(content)
            elif isinstance(content, list):
                texts.extend(
                    item.get("text", "")
                    for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                )
        return "\n".join(texts)

    def _generate_output_tokens(
        self, params: TokenGenerationParams
    ) -> tuple[list[str], str]:
        """Generate output tokens based on prompt and constraints."""
        budget = params.calculate_budget()

        if params.ignore_eos:
            return self._generate_max_tokens(params.prompt_tokens, budget.max_tokens)

        num_tokens = self._calculate_variable_token_count(params, budget)
        finish_reason = "length" if num_tokens == budget.max_tokens else "stop"
        output_tokens = self._cycle_tokens(params.prompt_tokens, num_tokens)

        return output_tokens, finish_reason

    def _generate_max_tokens(
        self, prompt_tokens: list[str], num_tokens: int
    ) -> tuple[list[str], str]:
        """Generate maximum tokens with 'length' finish reason."""
        return self._cycle_tokens(prompt_tokens, num_tokens), "length"

    def _calculate_variable_token_count(
        self, params: TokenGenerationParams, budget: TokenBudget
    ) -> int:
        """Calculate target token count using deterministic seed from prompt."""
        seed = self._generate_seed(params.prompt_tokens)

        if params.min_tokens is None:
            target_max = min(
                int(params.prompt_token_count * 1.2),
                budget.max_tokens,
            )
            target_max = max(target_max, budget.min_tokens)
        else:
            target_max = budget.max_tokens

        range_size = target_max - budget.min_tokens + 1
        return budget.min_tokens + (seed % range_size)

    def _generate_seed(self, prompt_tokens: list[str]) -> int:
        """Generate deterministic seed from prompt tokens."""
        if not prompt_tokens:
            return 0
        sample = prompt_tokens[:5]
        return hash(tuple(sample)) % 1000

    def _cycle_tokens(self, prompt_tokens: list[str], num_tokens: int) -> list[str]:
        """Generate tokens by cycling through prompt tokens."""
        if not prompt_tokens or num_tokens == 0:
            return []
        return [prompt_tokens[i % len(prompt_tokens)] for i in range(num_tokens)]


Tokenizer = _Tokenizer()
