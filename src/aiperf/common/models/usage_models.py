# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, ClassVar


class Usage(dict):
    """Usage wraps API-reported token consumption data with a unified interface.

    Inference frameworks return token usage in varying shapes — flat dicts
    (OpenAI / vLLM / TGI), camelCase wrappers (Google Gemini's `usageMetadata`,
    AWS Bedrock's `inputTokens` / `cacheReadInputTokens`), nested billing
    envelopes (Cohere's `meta.billed_units`, `meta.tokens`), and provider-
    specific extras (Anthropic's `cache_creation_input_tokens`, DeepSeek's
    `prompt_cache_hit_tokens` / `prompt_cache_miss_tokens`, Mistral's
    `prompt_audio_seconds`).

    Construction normalizes the recognized envelopes — `usageMetadata` (Gemini)
    and `meta.tokens` (Cohere's raw counts) — so all properties read from the
    top level. The underlying dict is preserved verbatim, so framework-specific
    fields the properties don't model still pass through and can be inspected
    by callers (e.g. Cohere's `meta.billed_units` for cost reconciliation).

    Properties consult ordered key-synonym lists; the FIRST present key wins.
    For per-property field-name maps see `*_KEYS` class attributes. Properties
    return None when no synonym is present (so `0` is correctly distinguished
    from "missing").

    Vendor field-name accuracy was verified against SDK source code in early
    2026 for: openai-python, anthropic-sdk-python, google-genai (camelCase
    aliases via Pydantic to_camel), groq-python, together-python,
    cohere-python (v1 ApiMeta + v2 Usage), client-python (Mistral),
    vllm OpenAI-compatible protocol, and AWS Bedrock TokenUsage docs.

    Known unmodelled extras (preserved verbatim on the dict — accessible via
    `usage[key]` for callers that need them):

    - Anthropic: `cache_creation` (TTL breakdown sub-object with
      `ephemeral_5m_input_tokens` / `ephemeral_1h_input_tokens`),
      `server_tool_use` (`web_fetch_requests`, `web_search_requests`),
      `service_tier` ("standard"/"priority"/"batch"), `inference_geo`.
    - AWS Bedrock: `cacheDetails[]` (TTL breakdown array of CacheDetail).
    - Gemini: `*Details[]` arrays (`promptTokensDetails`, `cacheTokensDetails`,
      `candidatesTokensDetails`, `toolUsePromptTokensDetails`) — modality
      breakdown; `trafficType`.
    - Groq: `prompt_time`, `completion_time`, `queue_time`, `total_time`
      (server-side timing in seconds) — useful but not token-shaped.
    - Cohere: `billed_units.search_units`, `billed_units.classifications`
      (non-token billable units); v1 ApiMeta carries `api_version`,
      `warnings[]`.
    - xAI Grok native gRPC: `cached_prompt_text_tokens`, top-level
      `reasoning_tokens`, `prompt_text_tokens`, `prompt_image_tokens`,
      `cost_in_usd_ticks`. Not relevant for the OpenAI-compatible REST
      endpoint AIPerf typically uses; xAI's REST API mirrors OpenAI shape.
    """

    PROMPT_DETAILS_KEYS: ClassVar[list[str]] = [
        "prompt_tokens_details",
        "input_tokens_details",
    ]
    COMPLETION_DETAILS_KEYS: ClassVar[list[str]] = [
        "completion_tokens_details",
        "output_tokens_details",
    ]
    PROMPT_TOKENS_KEYS: ClassVar[list[str]] = [
        "prompt_tokens",  # OpenAI / vLLM / Mistral / DeepSeek / AI21 / Fireworks / Cerebras / Together
        "input_tokens",  # Anthropic / Cohere meta.tokens / Bailian DashScope
        "promptTokenCount",  # Gemini / Vertex AI (camelCase wire)
        "inputTokens",  # AWS Bedrock
        "input_token_count",  # IBM watsonx (response-root field)
    ]
    COMPLETION_TOKENS_KEYS: ClassVar[list[str]] = [
        "completion_tokens",  # OpenAI / vLLM / Mistral / DeepSeek / AI21 / Fireworks / Cerebras / Together / SambaNova / Groq
        "output_tokens",  # Anthropic / Cohere meta.tokens / Bailian DashScope
        "candidatesTokenCount",  # Gemini / Vertex AI (camelCase wire)
        "outputTokens",  # AWS Bedrock
        "generated_token_count",  # IBM watsonx (response-root field)
    ]
    TOTAL_TOKENS_KEYS: ClassVar[list[str]] = [
        "total_tokens",  # OpenAI shape
        "totalTokenCount",  # Gemini
        "totalTokens",  # AWS Bedrock
    ]
    CACHE_READ_TOP_LEVEL_KEYS: ClassVar[list[str]] = [
        "cache_read_input_tokens",  # Anthropic
        "prompt_cache_hit_tokens",  # DeepSeek
        "cachedContentTokenCount",  # Gemini
        "cacheReadInputTokens",  # AWS Bedrock
        "cached_tokens",  # Cohere v2 (top-level under usage; distinct from
        # the OpenAI-nested prompt_tokens_details.cached_tokens which is
        # also handled but via the PROMPT_DETAILS_KEYS path)
    ]
    CACHE_WRITE_TOP_LEVEL_KEYS: ClassVar[list[str]] = [
        "cache_creation_input_tokens",  # Anthropic
        "cacheWriteInputTokens",  # AWS Bedrock
    ]
    CACHE_MISS_TOP_LEVEL_KEYS: ClassVar[list[str]] = [
        "prompt_cache_miss_tokens",  # DeepSeek
    ]
    REASONING_TOP_LEVEL_KEYS: ClassVar[list[str]] = [
        "thoughtsTokenCount",  # Gemini
    ]
    TOOL_USE_PROMPT_KEYS: ClassVar[list[str]] = [
        "toolUsePromptTokenCount",  # Gemini
    ]
    PROMPT_AUDIO_SECONDS_KEYS: ClassVar[list[str]] = [
        "prompt_audio_seconds",  # Mistral
    ]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Wrap an API usage dict, normalizing recognized vendor envelopes.

        - Gemini: `usageMetadata` is unwrapped to the top level so its keys
          (camelCase, e.g. `promptTokenCount`) sit alongside the OpenAI-shape
          synonyms.
        - Cohere v1: `meta.tokens` is unwrapped — its `input_tokens` /
          `output_tokens` raw counts surface alongside the OpenAI-shape
          synonyms. (Cohere v1 puts the meta envelope at the response top
          level; if the parser passes the whole response to Usage() it shows
          up here.)
        - Cohere v2: `tokens` is at the top level of the `usage` dict
          (no `meta` wrapper). We unwrap top-level `tokens` directly so
          `input_tokens` / `output_tokens` surface for v2 too.
        - `billed_units` is intentionally NOT unwrapped under either v1 or
          v2: the billed-vs-raw distinction is a Cohere-specific accounting
          filter, not what the model actually processed. The full
          `billed_units` (and `meta`) are preserved on the underlying dict
          for callers that need billing reconciliation.

        Original keys are preserved if a normalized key would collide with an
        existing top-level key; the original wins.
        """
        super().__init__(*args, **kwargs)
        if "usageMetadata" in self and isinstance(self["usageMetadata"], dict):
            for key, value in self["usageMetadata"].items():
                self.setdefault(key, value)
        # Cohere v1: meta.tokens (response root has `meta` envelope).
        if "meta" in self and isinstance(self["meta"], dict):
            meta = self["meta"]
            tokens = meta.get("tokens")
            if isinstance(tokens, dict):
                for key, value in tokens.items():
                    self.setdefault(key, value)
            # v1 ApiMeta also carries `cached_tokens` (cache-hit count) as
            # a scalar at the meta-level, alongside `tokens` / `billed_units`.
            # Lift it so the standard cache-read synonym lookup finds it.
            if "cached_tokens" in meta:
                self.setdefault("cached_tokens", meta["cached_tokens"])
        # Cohere v2: top-level `tokens` sub-dict inside the `usage` envelope
        # (no `meta` wrapper). Unwrap so `input_tokens` / `output_tokens` are
        # accessible via the standard PROMPT/COMPLETION_TOKENS_KEYS lookup.
        if "tokens" in self and isinstance(self["tokens"], dict):
            for key, value in self["tokens"].items():
                self.setdefault(key, value)

    def _first_present(self, keys: list[str]) -> Any | None:
        """Return the value at the first key in `keys` present in the dict."""
        for key in keys:
            if key in self:
                return self[key]
        return None

    def _first_in_details(
        self, details_keys: list[str], inner_field: str
    ) -> Any | None:
        """Walk PROMPT_DETAILS_KEYS / COMPLETION_DETAILS_KEYS for an inner field."""
        for details_key in details_keys:
            details = self.get(details_key)
            if isinstance(details, dict) and inner_field in details:
                return details[inner_field]
        return None

    @property
    def prompt_tokens(self) -> int | None:
        """Get prompt/input token count from API usage dict.

        Recognized synonyms (in order): prompt_tokens (OpenAI/vLLM/DeepSeek/
        Mistral), input_tokens (Anthropic/Cohere meta.tokens),
        promptTokenCount (Gemini), inputTokens (AWS Bedrock).
        """
        return self._first_present(self.PROMPT_TOKENS_KEYS)

    @property
    def completion_tokens(self) -> int | None:
        """Get completion/output token count from API usage dict.

        Recognized synonyms (in order): completion_tokens (OpenAI/vLLM/
        DeepSeek/Mistral), output_tokens (Anthropic/Cohere meta.tokens),
        candidatesTokenCount (Gemini), outputTokens (AWS Bedrock).
        """
        return self._first_present(self.COMPLETION_TOKENS_KEYS)

    @property
    def total_tokens(self) -> int | None:
        """Get total token count from API usage dict.

        Recognized synonyms (in order): total_tokens (OpenAI shape),
        totalTokenCount (Gemini), totalTokens (AWS Bedrock).
        """
        return self._first_present(self.TOTAL_TOKENS_KEYS)

    @property
    def reasoning_tokens(self) -> int | None:
        """Get reasoning / thinking tokens (reasoning models).

        OpenAI/vLLM/DeepSeek nest these under
        completion_tokens_details.reasoning_tokens (or
        output_tokens_details.reasoning_tokens). Gemini surfaces them at the
        top level as thoughtsTokenCount.
        """
        nested = self._first_in_details(
            self.COMPLETION_DETAILS_KEYS, "reasoning_tokens"
        )
        if nested is not None:
            return nested
        return self._first_present(self.REASONING_TOP_LEVEL_KEYS)

    @property
    def accepted_prediction_tokens(self) -> int | None:
        """Get accepted prediction tokens from nested completion details.

        Read from completion_tokens_details.accepted_prediction_tokens
        or output_tokens_details.accepted_prediction_tokens (whichever the
        framework reports). OpenAI-specific.
        """
        return self._first_in_details(
            self.COMPLETION_DETAILS_KEYS, "accepted_prediction_tokens"
        )

    @property
    def completion_audio_tokens(self) -> int | None:
        """Get audio tokens from nested completion details.

        Read from completion_tokens_details.audio_tokens or
        output_tokens_details.audio_tokens (whichever the framework reports).
        """
        return self._first_in_details(self.COMPLETION_DETAILS_KEYS, "audio_tokens")

    @property
    def rejected_prediction_tokens(self) -> int | None:
        """Get rejected prediction tokens from nested completion details.

        Read from completion_tokens_details.rejected_prediction_tokens
        or output_tokens_details.rejected_prediction_tokens (whichever the
        framework reports). OpenAI-specific.
        """
        return self._first_in_details(
            self.COMPLETION_DETAILS_KEYS, "rejected_prediction_tokens"
        )

    @property
    def prompt_audio_tokens(self) -> int | None:
        """Get audio tokens from nested prompt details.

        Read from prompt_tokens_details.audio_tokens or
        input_tokens_details.audio_tokens (whichever the framework reports).
        """
        return self._first_in_details(self.PROMPT_DETAILS_KEYS, "audio_tokens")

    @property
    def prompt_cache_read_tokens(self) -> int | None:
        """Get cached prompt-token reads (cache hits).

        Vendor synonyms (in precedence order):
        - OpenAI / vLLM: prompt_tokens_details.cached_tokens
          (or input_tokens_details.cached_tokens) — writes are transparent.
        - Anthropic: top-level cache_read_input_tokens.
        - DeepSeek: top-level prompt_cache_hit_tokens.
        - Gemini: top-level cachedContentTokenCount.
        - AWS Bedrock: top-level cacheReadInputTokens.

        See prompt_cache_write_tokens for vendors that surface writes,
        and prompt_cache_miss_tokens for vendors that surface misses.
        """
        nested = self._first_in_details(self.PROMPT_DETAILS_KEYS, "cached_tokens")
        if nested is not None:
            return nested
        return self._first_present(self.CACHE_READ_TOP_LEVEL_KEYS)

    @property
    def prompt_cache_write_tokens(self) -> int | None:
        """Get cached prompt-token writes (cache creations).

        Reported only by APIs that bill cache writes separately:
        - Anthropic: top-level cache_creation_input_tokens.
        - AWS Bedrock: top-level cacheWriteInputTokens.

        OpenAI / DeepSeek / Gemini do not surface writes — writes happen
        transparently or are not separately billed — so this property returns
        None for those shapes.
        """
        return self._first_present(self.CACHE_WRITE_TOP_LEVEL_KEYS)

    @property
    def prompt_cache_miss_tokens(self) -> int | None:
        """Get prompt-token cache misses.

        DeepSeek surfaces this directly as top-level prompt_cache_miss_tokens
        — they bill cache hits and misses at different rates so the split is
        first-class. Other vendors do not surface a separate miss count
        (you can derive it from prompt_tokens - prompt_cache_read_tokens
        on those, but the API doesn't report it as its own field).
        """
        return self._first_present(self.CACHE_MISS_TOP_LEVEL_KEYS)

    @property
    def tool_use_prompt_tokens(self) -> int | None:
        """Get tokens spent on tool/function-call definitions in the prompt.

        Gemini surfaces this as top-level toolUsePromptTokenCount — tokens
        consumed by tool/function declarations sent in the request, separate
        from the user-content prompt tokens. Other vendors currently fold
        this into the regular prompt_tokens count.
        """
        return self._first_present(self.TOOL_USE_PROMPT_KEYS)

    @property
    def prompt_audio_seconds(self) -> float | None:
        """Get input audio duration in seconds (NOT tokens).

        Mistral surfaces this for audio-input requests as top-level
        prompt_audio_seconds. This is a duration, not a token count, so the
        unit differs from prompt_audio_tokens. Both can coexist in the same
        usage dict for some frameworks.

        Defensive note: when no audio is present in the prompt, Mistral has
        been observed to emit `prompt_audio_seconds: {}` (an empty dict
        sentinel) rather than `null` or omitting the key. We treat any
        non-numeric value as "no audio" and return None.
        """
        value = self._first_present(self.PROMPT_AUDIO_SECONDS_KEYS)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return float(value)
