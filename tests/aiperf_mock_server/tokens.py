# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from aiperf_mock_server.models import (
    ChatCompletionRequest,
    CohereRerankRequest,
    CompletionRequest,
    EmbeddingRequest,
    HFTEIRerankRequest,
    ImageGenerationRequest,
    RankingRequest,
    RequestT,
    SolidoRAGRequest,
    TGIGenerateRequest,
)

logger = logging.getLogger(__name__)

EFFORT_TOKENS = {"low": 100, "medium": 250, "high": 500}


@dataclass(slots=True)
class TokenizedText:
    """Tokenized text with metadata."""

    text: str
    tokens: list[str]
    prompt_token_count: int
    reasoning_tokens: int = 0
    reasoning_content_tokens: list[str] = field(default_factory=list)
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

    def create_usage(self) -> dict[str, Any]:
        """Create usage dict from tokenized text."""
        # completion_tokens includes both content and reasoning tokens per OpenAI API
        completion_tokens = self.count + self.reasoning_tokens
        usage: dict[str, Any] = {
            "prompt_tokens": self.prompt_token_count,
            "completion_tokens": completion_tokens,
            "total_tokens": self.prompt_token_count + completion_tokens,
        }
        if self.reasoning_tokens > 0:
            usage["completion_tokens_details"] = {
                "reasoning_tokens": self.reasoning_tokens
            }
        return usage


@dataclass(slots=True)
class _ReasoningResult:
    """Result of reasoning token generation with budget management."""

    token_count: int
    content_tokens: list[str]
    remaining_budget: int | None


@dataclass(slots=True)
class _TokenBudget:
    """Token budget calculation result."""

    total: int
    min_tokens: int
    max_tokens: int


# ============================================================================
# Tokenization Functions
# ============================================================================


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


def tokenize(text: str) -> tuple[str, ...]:
    """Tokenize text with caching."""
    return _tokenize(text)


def count_tokens(text: str) -> int:
    """Count tokens in text."""
    return len(_tokenize(text))


def tokenize_request(request: RequestT) -> TokenizedText:
    """Tokenize a request and return TokenizedText with usage."""
    text, max_tokens = _extract_request_content(request)
    prompt_tokens = list(_tokenize(text))
    prompt_token_count = len(prompt_tokens)

    # For embeddings, rankings, and images - simple token counting without generation options
    if isinstance(
        request,
        EmbeddingRequest
        | RankingRequest
        | HFTEIRerankRequest
        | CohereRerankRequest
        | ImageGenerationRequest,
    ):
        return TokenizedText(
            text=text,
            tokens=[],
            prompt_token_count=prompt_token_count,
        )

    # Handle empty prompts - can't generate tokens without source material
    if not prompt_tokens:
        return TokenizedText(text=text, tokens=[], prompt_token_count=0)

    reasoning_result = _generate_reasoning_tokens(
        request, prompt_tokens, prompt_token_count, max_tokens
    )

    output_tokens, finish_reason = _generate_output_tokens(
        prompt_tokens=prompt_tokens,
        prompt_token_count=prompt_token_count,
        max_tokens=reasoning_result.remaining_budget,
        min_tokens=request.min_tokens,
        ignore_eos=request.ignore_eos,
    )

    return TokenizedText(
        text=text,
        tokens=output_tokens,
        prompt_token_count=prompt_token_count,
        reasoning_tokens=reasoning_result.token_count,
        reasoning_content_tokens=reasoning_result.content_tokens,
        finish_reason=finish_reason,
    )


# ============================================================================
# Internal Helpers
# ============================================================================


def _calculate_budget(
    prompt_token_count: int, max_tokens: int | None, min_tokens: int | None
) -> _TokenBudget:
    """Calculate min/max token budget for generation."""
    max_budget = (
        max_tokens if max_tokens is not None else max(prompt_token_count * 2, 16)
    )

    if min_tokens is not None:
        min_budget = min(min_tokens, max_budget)
    else:
        min_budget = max(1, int(prompt_token_count * 0.8))
        min_budget = min(min_budget, max_budget)

    return _TokenBudget(total=max_budget, min_tokens=min_budget, max_tokens=max_budget)


def _generate_reasoning_tokens(
    request: ChatCompletionRequest | CompletionRequest | TGIGenerateRequest,
    prompt_tokens: list[str],
    prompt_token_count: int,
    max_tokens: int | None,
) -> _ReasoningResult:
    """Generate reasoning tokens if model supports it, managing budget."""
    # Only chat completions support reasoning (per OpenAI API spec)
    if not isinstance(request, ChatCompletionRequest):
        return _ReasoningResult(
            token_count=0, content_tokens=[], remaining_budget=max_tokens
        )

    # Check if model supports reasoning tokens
    model_lower = request.model.lower()
    is_reasoning_model = any(m in model_lower for m in ("gpt-oss", "qwen"))

    if not is_reasoning_model:
        return _ReasoningResult(
            token_count=0, content_tokens=[], remaining_budget=max_tokens
        )

    # Calculate requested reasoning tokens based on effort
    effort = getattr(request, "reasoning_effort", None) or "medium"
    requested_reasoning_tokens = EFFORT_TOKENS.get(effort, 250)

    total_budget = (
        max_tokens if max_tokens is not None else max(prompt_token_count * 2, 16)
    )
    actual_reasoning_tokens = min(requested_reasoning_tokens, total_budget)

    # Generate reasoning content tokens (uses corpus if available, else prompt cycling)
    # Use reversed seed offset to differentiate from main content
    reasoning_content_tokens = _cycle_tokens_reversed(
        prompt_tokens, actual_reasoning_tokens
    )

    return _ReasoningResult(
        token_count=actual_reasoning_tokens,
        content_tokens=reasoning_content_tokens,
        remaining_budget=total_budget - actual_reasoning_tokens,
    )


def _extract_request_content(request: RequestT) -> tuple[str, int | None]:
    """Extract text and max_tokens from request."""
    if isinstance(request, ChatCompletionRequest):
        text = _extract_chat_messages(request.messages)
        return text, request.max_output_tokens
    elif isinstance(request, CompletionRequest | TGIGenerateRequest):
        return request.prompt_text, request.max_tokens
    elif isinstance(request, EmbeddingRequest):
        text = "\n".join(request.inputs)
        return text, None
    elif isinstance(request, RankingRequest | HFTEIRerankRequest | CohereRerankRequest):
        text = request.query_text + "\n" + "\n".join(request.passage_texts)
        return text, None
    elif isinstance(request, ImageGenerationRequest):
        return request.prompt, None
    elif isinstance(request, SolidoRAGRequest):
        return " ".join(request.query), None
    else:
        raise ValueError(f"Unsupported request type: {type(request)}")


def _extract_chat_messages(messages: list) -> str:
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
    prompt_tokens: list[str],
    prompt_token_count: int,
    max_tokens: int | None,
    min_tokens: int | None,
    ignore_eos: bool,
) -> tuple[list[str], str]:
    """Generate output tokens based on prompt and constraints."""
    budget = _calculate_budget(prompt_token_count, max_tokens, min_tokens)

    if ignore_eos:
        return _cycle_tokens(prompt_tokens, budget.max_tokens), "length"

    num_tokens = _calculate_variable_token_count(
        prompt_tokens, prompt_token_count, min_tokens, max_tokens, budget
    )
    finish_reason = "length" if num_tokens == budget.max_tokens else "stop"
    return _cycle_tokens(prompt_tokens, num_tokens), finish_reason


def _calculate_variable_token_count(
    prompt_tokens: list[str],
    prompt_token_count: int,
    min_tokens: int | None,
    max_tokens: int | None,
    budget: _TokenBudget,
) -> int:
    """Calculate target token count using deterministic seed from prompt."""
    seed = _generate_seed(prompt_tokens)

    # If max_tokens explicitly set, use full budget range
    # Otherwise, default to 0.8-1.2× prompt length
    if max_tokens is not None or min_tokens is not None:
        target_max = budget.max_tokens
    else:
        target_max = min(int(prompt_token_count * 1.2), budget.max_tokens)
        target_max = max(target_max, budget.min_tokens)

    range_size = target_max - budget.min_tokens + 1
    return budget.min_tokens + (seed % range_size)


def _generate_seed(prompt_tokens: list[str]) -> int:
    """Generate deterministic seed from prompt tokens."""
    if not prompt_tokens:
        return 0
    sample = prompt_tokens[:5]
    return hash(tuple(sample)) % 1000


def _cycle_tokens(
    prompt_tokens: list[str], num_tokens: int, offset: int = 0
) -> list[str]:
    """Generate output tokens.

    If corpus is available, uses it with prompt-hash-based offset for readable output.
    Otherwise falls back to cycling through prompt tokens.
    """
    if num_tokens == 0:
        return []
    if CORPUS_TOKENS is None:
        # Fallback: cycle through prompt tokens (original behavior)
        if not prompt_tokens:
            return []
        return [
            prompt_tokens[(offset + i) % len(prompt_tokens)] for i in range(num_tokens)
        ]
    # Use corpus with deterministic offset
    seed = _generate_seed(prompt_tokens) if prompt_tokens else 0
    start = (seed + offset) % len(CORPUS_TOKENS)
    return [CORPUS_TOKENS[(start + i) % len(CORPUS_TOKENS)] for i in range(num_tokens)]


def _cycle_tokens_reversed(prompt_tokens: list[str], num_tokens: int) -> list[str]:
    """Generate tokens with reversed offset for reasoning content."""
    if num_tokens == 0:
        return []
    # Use a large offset to get different content than main output
    return _cycle_tokens(
        prompt_tokens,
        num_tokens,
        offset=len(CORPUS_TOKENS) // 2 if CORPUS_TOKENS else 0,
    )


def _load_corpus() -> tuple[str, ...] | None:
    """Load and tokenize corpus from aiperf's shakespeare.txt at import time.

    Uses PromptGenerator for multi-threaded tokenization if available,
    falls back to character-based chunking otherwise.
    """
    global CORPUS_TOKENS
    if CORPUS_TOKENS is not None:
        return CORPUS_TOKENS
    from aiperf_mock_server.config import server_config

    try:
        import aiperf.dataset.generator.prompt as prompt_module
        from aiperf.dataset.generator.prompt import DEFAULT_CORPUS_FILE

        corpus_path = Path(prompt_module.__file__).parent / DEFAULT_CORPUS_FILE
    except ImportError:
        logger.warning("aiperf not found, corpus not loaded")
        return None

    if not corpus_path.exists():
        logger.warning(f"Corpus file not found: {corpus_path}")
        return None

    def char_based_fallback() -> tuple[str, ...]:
        """Normalize text and tokenize with character-based chunking."""
        lines = corpus_path.read_text().splitlines()
        text = " ".join(line.strip() for line in lines if line.strip())
        return _tokenize(text)

    start_time = time.perf_counter()

    if server_config.no_tokenizer:
        logger.info("Loading corpus with character-based chunking (--no-tokenizer)...")
        tokens = char_based_fallback()
    else:
        logger.info(
            f"SERVER NOT READY - Loading corpus with tokenizer '{server_config.tokenizer}'... This may take a while..."
        )
        try:
            from aiperf.common.config import PromptConfig
            from aiperf.common.tokenizer import Tokenizer
            from aiperf.dataset.generator.prompt import PromptGenerator

            tokenizer = Tokenizer.from_pretrained(
                server_config.tokenizer,
                trust_remote_code=server_config.tokenizer_trust_remote_code,
                revision=server_config.tokenizer_revision,
            )
            generator = PromptGenerator(config=PromptConfig(), tokenizer=tokenizer)

            # Fast batch conversion, replace BPE space marker (Ġ) with actual space
            raw_tokens = tokenizer._tokenizer.convert_ids_to_tokens(
                generator._tokenized_corpus
            )
            tokens = tuple(tok.replace("Ġ", " ") for tok in raw_tokens)
        except Exception as e:
            logger.warning(
                f"Tokenizer failed ({e}), falling back to character-based chunking"
            )
            tokens = char_based_fallback()

    elapsed = time.perf_counter() - start_time
    logger.info(f"Corpus loaded: {len(tokens)} tokens in {elapsed:.2f}s")
    CORPUS_TOKENS = tokens
    return CORPUS_TOKENS


CORPUS_TOKENS: tuple[str, ...] | None = None
