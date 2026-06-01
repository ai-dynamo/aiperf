---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Vendor Usage Field Reference
---

# Vendor Usage Field Reference

This document catalogues the exact JSON shape of the `usage` field that each LLM provider returns in chat / completion responses, cross-referenced against their official SDK source code. It exists so that:

- A maintainer adding a new vendor knows what to look for and where existing vendors agree or differ.
- A debugger investigating "why doesn't my usage metric show a value" can find the canonical field-name list per vendor.
- A reviewer of a future usage-parsing change can verify that no vendor's wire format was missed.

The verification work behind this document was performed by inspecting each provider's Python SDK source (or REST API documentation when no SDK type was available). All conclusions are dated against the SDK / docs commit at the time of verification (early 2026).

## Quick reference: vendor shape map

| Vendor | Wrapper | Token-count fields | Cache fields | Notable extras |
|---|---|---|---|---|
| OpenAI | flat `usage` | `prompt_tokens`, `completion_tokens`, `total_tokens` | `prompt_tokens_details.cached_tokens` (read-only) | nested `*_tokens_details` for audio / reasoning / prediction |
| vLLM | flat `usage` | OpenAI-shape | `prompt_tokens_details.cached_tokens` (opt-in) | OpenAI-shape; `cached_tokens` emitted only with `--enable-prompt-tokens-details` (default off), else `prompt_tokens_details: null` |
| SGLang | flat `usage` | OpenAI-shape | `prompt_tokens_details.cached_tokens` (opt-in) | OpenAI-compatible; `cached_tokens` emitted only with `--enable-cache-report` (default off) |
| TRT-LLM | flat `usage` | OpenAI-shape | `prompt_tokens_details.cached_tokens` | OpenAI-compatible; `cached_tokens` emitted by default (no flag) |
| Anthropic | flat `usage` | `input_tokens`, `output_tokens` | `cache_creation_input_tokens`, `cache_read_input_tokens` | `cache_creation` TTL sub-object; `service_tier`; `server_tool_use` |
| Google Gemini | `usageMetadata` envelope (camelCase) | `promptTokenCount`, `candidatesTokenCount`, `totalTokenCount` | `cachedContentTokenCount` (read-only) | `thoughtsTokenCount`, `toolUsePromptTokenCount`, modality `*Details[]` arrays |
| AWS Bedrock | flat `usage` (camelCase) | `inputTokens`, `outputTokens`, `totalTokens` | `cacheReadInputTokens`, `cacheWriteInputTokens` | `cacheDetails[]` TTL array |
| DeepSeek | flat `usage` | OpenAI-shape | `prompt_cache_hit_tokens`, `prompt_cache_miss_tokens` | OpenAI-style `completion_tokens_details.reasoning_tokens` for thinking mode |
| Cohere v1 | `meta` envelope (response root) | `meta.tokens.{input,output}_tokens` | `meta.cached_tokens` | `meta.billed_units` (raw vs billed split); `api_version`; `warnings[]` |
| Cohere v2 | flat `usage` | top-level `tokens.{input,output}_tokens` | top-level `cached_tokens` | top-level `billed_units` (same split) |
| Mistral | flat `usage` | OpenAI-shape | OpenAI-style nested `cached_tokens` | `prompt_audio_seconds` (audio duration, NOT tokens; emits `{}` sentinel when absent) |
| Groq | flat `usage` | OpenAI-shape | OpenAI-shape | per-stage timings: `prompt_time`, `completion_time`, `queue_time`, `total_time` (seconds) |
| Together / Fireworks / Replicate | flat `usage` | OpenAI-shape | OpenAI-shape | passthrough proxies; whatever underlying model emits |
| Cerebras | flat `usage` | OpenAI-shape | OpenAI-shape (`prompt_tokens_details.cached_tokens`) | OpenAI-compatible Stainless-generated SDK |
| AI21 Labs | flat `usage` | OpenAI-shape | n/a | basic `prompt_tokens`/`completion_tokens`/`total_tokens` only |
| SambaNova | flat `usage` | OpenAI-shape | OpenAI-shape | rich server-side timing/throughput (`time_to_first_token`, `total_latency`, `acceptance_rate`, `*_tokens_per_sec`, etc.) |
| Bailian / DashScope (Alibaba Qwen) | flat `usage` | `input_tokens` / `output_tokens` (Anthropic-style) | n/a | multimodal endpoint adds `characters` (non-token billing); OpenAI-compat endpoint emits OpenAI shape |
| Vertex AI (Gemini) | `usageMetadata` envelope | same camelCase as Gemini direct | same | identical wire format to Gemini |
| **IBM watsonx** | **response root** (no `usage` envelope) | `input_token_count`, `generated_token_count` | n/a | distinct `_count` suffix; sibling fields `stop_reason`, `response_time` at response root too |
| xAI Grok (REST) | flat `usage` | OpenAI-shape | OpenAI-shape | xAI's REST endpoint is OpenAI-compatible |
| xAI Grok (gRPC) | proto message | `prompt_tokens`, `completion_tokens`, `total_tokens` | `cached_prompt_text_tokens` (top-level) | top-level `reasoning_tokens`, `prompt_text_tokens`, `prompt_image_tokens`, `cost_in_usd_ticks` — NOT exposed via REST so AIPerf doesn't model them |

## How AIPerf normalizes these shapes

AIPerf wraps every API-reported usage dict in a `Usage` class ([`src/aiperf/common/models/usage_models.py`](https://github.com/ai-dynamo/aiperf/blob/main/src/aiperf/common/models/usage_models.py)). On construction, two recognized vendor envelopes are unwrapped to the top level so all properties read from a single flat dict:

- **Gemini** `usageMetadata` → top-level (lifts `promptTokenCount`, `candidatesTokenCount`, etc.).
- **Cohere v1** `meta` → top-level (lifts `meta.tokens.{input,output}_tokens`, `meta.cached_tokens`).
- **Cohere v2** top-level `tokens` sub-dict → top-level (lifts `tokens.{input,output}_tokens`).

The original keys are preserved if a normalized key would collide; the original wins.

After normalization, each property reads through an ordered synonym list (the `*_KEYS` class attributes). The first present key wins. Properties return `None` when no synonym is present, so `0` is correctly distinguished from "missing".

## Per-vendor verification details

### OpenAI

**Verified against:** [`openai-python` / `src/openai/types/completion_usage.py`](https://github.com/openai/openai-python/blob/main/src/openai/types/completion_usage.py).

```python
class CompletionUsage(BaseModel):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int
    completion_tokens_details: Optional[CompletionTokensDetails] = None
    prompt_tokens_details: Optional[PromptTokensDetails] = None

class CompletionTokensDetails(BaseModel):
    accepted_prediction_tokens: Optional[int] = None
    audio_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    rejected_prediction_tokens: Optional[int] = None

class PromptTokensDetails(BaseModel):
    audio_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None
```

All field names match AIPerf's modelled synonyms. `cached_tokens` is read-only on OpenAI (writes are transparent and free), so we do not raise NoMetricValue for OpenAI when the cache-write metric is queried — we just return None. OpenAI does NOT surface a separate cache-miss count; you can derive it from `prompt_tokens - prompt_tokens_details.cached_tokens` if needed.

### vLLM

**Verified against:** [`vllm` / `vllm/entrypoints/openai/engine/protocol.py`](https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/engine/protocol.py).

```python
class UsageInfo(OpenAIBaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: int | None = 0
    prompt_tokens_details: PromptTokenUsageInfo | None = None

class PromptTokenUsageInfo(OpenAIBaseModel):
    cached_tokens: int | None = None
```

vLLM is OpenAI-compatible. Its `prompt_tokens_details` is narrower than OpenAI's (only `cached_tokens`, no `audio_tokens`). vLLM may emit `prompt_tokens_details: null` and `completion_tokens_details: null` explicitly; AIPerf's nested-field walk handles that case (the `isinstance(details, dict)` guard returns False, and the property returns None).

**Enablement (opt-in):** vLLM populates `cached_tokens` only when the server is launched with `--enable-prompt-tokens-details` (default off, per [vllm-project/vllm#10174](https://github.com/vllm-project/vllm/pull/10174) — "guarded by a flag ... OFF by default"). Without the flag, `prompt_tokens_details` is `null` and AIPerf's cache-read metrics are all-None even when prefix caching is active. The flag is independent of `--enable-prefix-caching` (the cache must still be on for `cached_tokens` to be non-zero).

### SGLang

**Verified against:** `sglang` / `python/sglang/srt/entrypoints/openai/serving_chat.py` → `usage_processor.py` (`v0.5.12.post1`).

SGLang is OpenAI-compatible and surfaces cache reads as `prompt_tokens_details.cached_tokens` — but **only when the server is launched with `--enable-cache-report`** (`server_args.enable_cache_report`, default off, per [sgl-project/sglang#1599](https://github.com/sgl-project/sglang/pull/1599) — kept opt-in "to limit the impact to current users"). Without the flag the field is absent and AIPerf's cache-read metrics are all-None. SGLang also exposes a prefix-cache hit rate as a scraped Prometheus gauge (`sglang:cache_hit_rate`), but that gauge is *instantaneous* — its windowed average badly understates steady-state reuse, so prefer the usage-derived `cached_tokens` for cache reporting.

### TRT-LLM

**Verified against:** `TensorRT-LLM` / `tensorrt_llm/serve/postprocess_handlers.py` (`v1.3.0rc15.post1`).

TensorRT-LLM (`trtllm-serve`) is OpenAI-compatible and populates `prompt_tokens_details.cached_tokens` **by default** — `postprocess_handlers.py` sets it unconditionally from the engine's KV-cache reuse, so no flag is needed. This is separate from exposing Prometheus `/metrics` (which requires `return_perf_metrics: true`); the `usage` cache field works regardless.

### Anthropic

**Verified against:** [`anthropic-sdk-python` / `src/anthropic/types/usage.py`](https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/types/usage.py), [`message_delta_usage.py`](https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/types/message_delta_usage.py), [`cache_creation.py`](https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/types/cache_creation.py), and [`server_tool_usage.py`](https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/types/server_tool_usage.py).

```python
class Usage(BaseModel):
    cache_creation: Optional[CacheCreation] = None
    cache_creation_input_tokens: Optional[int] = None
    cache_read_input_tokens: Optional[int] = None
    inference_geo: Optional[str] = None
    input_tokens: int
    output_tokens: int
    server_tool_use: Optional[ServerToolUsage] = None
    service_tier: Optional[Literal["standard", "priority", "batch"]] = None

class CacheCreation(BaseModel):
    ephemeral_1h_input_tokens: int
    ephemeral_5m_input_tokens: int

class ServerToolUsage(BaseModel):
    web_fetch_requests: int
    web_search_requests: int
```

Streaming chunks use `MessageDeltaUsage`, which carries the same fields as `Usage` for cache and tokens (a non-streaming chunk + `MessageDeltaUsage` contain the same shape for our purposes).

**Modelled:** `input_tokens`, `output_tokens`, `cache_creation_input_tokens`, `cache_read_input_tokens`.

**Not modelled (preserved on dict):**
- `cache_creation` TTL breakdown (sum of `ephemeral_1h_input_tokens + ephemeral_5m_input_tokens` equals the parent `cache_creation_input_tokens`). Could be added if TTL-aware analysis is needed.
- `server_tool_use` (`web_fetch_requests`, `web_search_requests`). Non-token metadata.
- `service_tier` ("standard"/"priority"/"batch"). String label, not a count.
- `inference_geo`. String label.

### Google Gemini

**Verified against:** [`google-genai` / `google/genai/types.py`](https://github.com/googleapis/python-genai/blob/main/google/genai/types.py) (`GenerateContentResponseUsageMetadata`) and [`_common.py`](https://github.com/googleapis/python-genai/blob/main/google/genai/_common.py) (`alias_generator=to_camel`).

The Python SDK declares fields in `snake_case` for Python ergonomics, but the Pydantic `alias_generator=to_camel` config means the wire (JSON) format is camelCase. AIPerf operates at the JSON level, so **the camelCase names are what we synonym-match**.

```python
class GenerateContentResponseUsageMetadata(BaseModel):
    cached_content_token_count: Optional[int]
    candidates_token_count: Optional[int]
    prompt_token_count: Optional[int]
    thoughts_token_count: Optional[int]
    tool_use_prompt_token_count: Optional[int]
    total_token_count: Optional[int]

    # Modality-detail breakdown arrays (not modelled)
    cache_tokens_details: Optional[list[ModalityTokenCount]]
    candidates_tokens_details: Optional[list[ModalityTokenCount]]
    prompt_tokens_details: Optional[list[ModalityTokenCount]]
    tool_use_prompt_tokens_details: Optional[list[ModalityTokenCount]]
    traffic_type: Optional[TrafficType]
```

**Wire-format field names (after `to_camel`):** `cachedContentTokenCount`, `candidatesTokenCount`, `promptTokenCount`, `thoughtsTokenCount`, `toolUsePromptTokenCount`, `totalTokenCount`.

The whole object is wrapped in `usageMetadata` at the response top level; AIPerf's `Usage.__init__` unwraps it.

**Not modelled (preserved on dict):** the four `*Details[]` arrays of `ModalityTokenCount` objects (per-modality breakdowns: TEXT / IMAGE / AUDIO / VIDEO). Useful for multimodal benchmarks where you want to know what fraction of input tokens were images, but currently surfaced verbatim as a list rather than as a metric.

**Note on `prompt_token_count`:** Gemini's docs say "When `cached_content` is set, `prompt_token_count` includes the number of tokens in the cached content." So for Gemini, `prompt_tokens` is total-including-cached, and `cached_content_token_count` is the subset that was cached. This matches OpenAI's semantic where `prompt_tokens` is the total and `cached_tokens` is the subset of those that hit cache.

### AWS Bedrock

**Verified against:** [AWS Bedrock TokenUsage API reference](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_TokenUsage.html). No Python SDK clone needed — boto3 follows the documented API verbatim.

```
TokenUsage:
  inputTokens: int (required)
  outputTokens: int (required)
  totalTokens: int (required)
  cacheReadInputTokens: int (optional)
  cacheWriteInputTokens: int (optional)
  cacheDetails: list[CacheDetail] (optional, sorted by TTL: 1h before 5m)
```

**Modelled:** `inputTokens`, `outputTokens`, `totalTokens`, `cacheReadInputTokens`, `cacheWriteInputTokens`. All synonyms in the `*_KEYS` lists.

**Not modelled (preserved on dict):** `cacheDetails[]` TTL breakdown array.

Note that Bedrock's field names exactly match Anthropic's *concept names* but use camelCase. This is because Bedrock primarily proxies Anthropic models and converted the snake_case names to camelCase for AWS API conventions. The semantic mapping is one-to-one:

| Anthropic | Bedrock |
|---|---|
| `input_tokens` | `inputTokens` |
| `output_tokens` | `outputTokens` |
| `cache_read_input_tokens` | `cacheReadInputTokens` |
| `cache_creation_input_tokens` | `cacheWriteInputTokens` |

### DeepSeek

**Verified against:** [DeepSeek API documentation](https://api-docs.deepseek.com/api/create-chat-completion).

```
usage:
  prompt_tokens: int
  completion_tokens: int
  total_tokens: int
  prompt_cache_hit_tokens: int       # DeepSeek-specific
  prompt_cache_miss_tokens: int      # DeepSeek-specific (genuinely novel)
  completion_tokens_details:         # OpenAI-shape (thinking mode)
    reasoning_tokens: int
```

**Modelled:** all of the above. `prompt_cache_hit_tokens` is mapped to `prompt_cache_read_tokens` via the synonym list. `prompt_cache_miss_tokens` is its own first-class metric (`UsagePromptCacheMissTokensMetric`) since DeepSeek bills hits and misses at different rates and no other vendor surfaces the miss count as its own field.

**Invariant:** `prompt_tokens == prompt_cache_hit_tokens + prompt_cache_miss_tokens` for DeepSeek responses. AIPerf has a test asserting this end-to-end.

### Cohere

Cohere has TWO API versions with different envelopes. AIPerf handles both.

**v1 — verified against:** [`cohere-python` / `src/cohere/types/api_meta.py`](https://github.com/cohere-ai/cohere-python/blob/main/src/cohere/types/api_meta.py) and [`api_meta_tokens.py`](https://github.com/cohere-ai/cohere-python/blob/main/src/cohere/types/api_meta_tokens.py).

```python
class ApiMeta(BaseModel):
    api_version: Optional[ApiMetaApiVersion]
    billed_units: Optional[ApiMetaBilledUnits]
    tokens: Optional[ApiMetaTokens]
    cached_tokens: Optional[float]
    warnings: Optional[List[str]]
```

The `meta` envelope is at the **response root** (not under a `usage` key). If the parser hands the full response to `Usage()`, `meta` is what's there. AIPerf unwraps:
- `meta.tokens.input_tokens` → top-level (resolved via `PROMPT_TOKENS_KEYS`)
- `meta.tokens.output_tokens` → top-level (resolved via `COMPLETION_TOKENS_KEYS`)
- `meta.cached_tokens` → top-level (resolved via `CACHE_READ_TOP_LEVEL_KEYS`)

**v2 — verified against:** [`cohere-python` / `src/cohere/types/usage.py`](https://github.com/cohere-ai/cohere-python/blob/main/src/cohere/types/usage.py), [`usage_tokens.py`](https://github.com/cohere-ai/cohere-python/blob/main/src/cohere/types/usage_tokens.py), and [`usage_billed_units.py`](https://github.com/cohere-ai/cohere-python/blob/main/src/cohere/types/usage_billed_units.py).

```python
class Usage(BaseModel):
    billed_units: Optional[UsageBilledUnits]
    tokens: Optional[UsageTokens]
    cached_tokens: Optional[float]
```

The `usage` field at the response root contains `billed_units`, `tokens`, and `cached_tokens` directly — no `meta` wrapper. AIPerf treats top-level `tokens` (a sub-dict) the same way as `meta.tokens` and unwraps it. Top-level `cached_tokens` is in `CACHE_READ_TOP_LEVEL_KEYS`.

**`billed_units` is intentionally NOT surfaced as a metric.** Cohere's billed-vs-raw distinction is a Cohere-specific accounting filter (the framework injects special tokens that count toward the raw `tokens` total but aren't billed). For perf benchmarks, the raw count is what the model actually processed — which is what every other vendor reports — so we keep `prompt_tokens` consistent across vendors. Callers that need billing reconciliation can read `usage["meta"]["billed_units"]` (v1) or `usage["billed_units"]` (v2) directly off the underlying dict.

`billed_units` for chat:
- `input_tokens`, `output_tokens` — billed token counts
- `search_units`, `classifications` — non-token billable units (RAG / classification endpoints)

### Mistral

**Verified against:** [`mistralai/client-python` / `src/mistralai/client/models/usageinfo.py`](https://github.com/mistralai/client-python/blob/main/src/mistralai/client/models/usageinfo.py).

```python
class UsageInfo(BaseModel):
    prompt_tokens: Optional[int] = 0
    completion_tokens: Optional[int] = 0
    total_tokens: Optional[int] = 0
    prompt_audio_seconds: OptionalNullable[int] = UNSET
```

The SDK type declares `prompt_audio_seconds` as `Optional[int]`, but observed wire responses on Mistral's **agents endpoint** have shown the field emit as `{}` (an empty dict) when no audio is present in the prompt — visible in Mistral's documented response examples. AIPerf's `prompt_audio_seconds` property is defensive — it only coerces numeric values (`int` / `float`, excluding `bool`); any other type returns `None` rather than raising `TypeError` from `float({})`. The defensiveness is cheap and protects against either SDK / wire-format drift.

**Note:** `prompt_audio_seconds` is in `MetricTimeUnit.SECONDS`, distinct from `UsagePromptAudioTokensMetric` which is in `GenericMetricUnit.TOKENS`. The two metrics can coexist for the same response when Mistral reports both.

### Groq

**Verified against:** [`groq-python` / `src/groq/types/completion_usage.py`](https://github.com/groq/groq-python/blob/main/src/groq/types/completion_usage.py).

```python
class CompletionUsage(BaseModel):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int
    completion_time: Optional[float]    # seconds
    prompt_time: Optional[float]        # seconds
    queue_time: Optional[float]         # seconds
    total_time: Optional[float]         # seconds
    completion_tokens_details: Optional[CompletionTokensDetails]
    prompt_tokens_details: Optional[PromptTokensDetails]

class CompletionTokensDetails(BaseModel):
    reasoning_tokens: int

class PromptTokensDetails(BaseModel):
    cached_tokens: int
```

Token fields are pure OpenAI shape. The four `*_time` fields are **server-side timing** in seconds — useful for performance benchmarks (queue time + prompt time + completion time = end-to-end latency components). Currently preserved on the dict but not surfaced as metrics. Adding them as optional `BaseUsageRecordMetric[float]` subclasses with `MetricTimeUnit.SECONDS` would be a small follow-up if Groq benchmarking becomes a priority.

### Together AI / Fireworks / Replicate / Azure OpenAI

These are **passthrough proxies** that emit OpenAI-compatible usage shapes. Verified Together via [`together-python` / `src/together/types/common.py`](https://github.com/togethercomputer/together-python/blob/main/src/together/types/common.py):

```python
class UsageData(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
```

Verified Fireworks via [`fw-ai-external/python-sdk` / `src/fireworks/types/shared/usage_info.py`](https://github.com/fw-ai-external/python-sdk/blob/main/src/fireworks/types/shared/usage_info.py):

```python
class UsageInfo(BaseModel):
    prompt_tokens: int
    total_tokens: int
    completion_tokens: Optional[int] = None
    prompt_tokens_details: Optional[PromptTokensDetails] = None  # {cached_tokens}
```

Replicate's SDK does not declare a fixed Usage type because it passes through whatever the underlying hosted model emits. Azure OpenAI uses the openai-python SDK directly, so it inherits OpenAI's exact shape.

No vendor-specific changes needed for any of these; they're covered by the OpenAI synonyms.

### Cerebras

**Verified against:** [`Cerebras/cerebras-cloud-sdk-python` / `src/cerebras/cloud/sdk/types/chat/chat_completion.py`](https://github.com/Cerebras/cerebras-cloud-sdk-python/blob/main/src/cerebras/cloud/sdk/types/chat/chat_completion.py).

```python
class ChatCompletionResponseUsage(BaseModel):
    completion_tokens: Optional[int]
    completion_tokens_details: Optional[ChatCompletionResponseUsageCompletionTokensDetails]
    prompt_tokens: Optional[int]
    prompt_tokens_details: Optional[ChatCompletionResponseUsagePromptTokensDetails]
    total_tokens: Optional[int]

class ChatCompletionResponseUsageCompletionTokensDetails(BaseModel):
    accepted_prediction_tokens: Optional[int]
    rejected_prediction_tokens: Optional[int]
    # NOTE: NO audio_tokens, NO reasoning_tokens (narrower than OpenAI)

class ChatCompletionResponseUsagePromptTokensDetails(BaseModel):
    cached_tokens: Optional[int]
    # NOTE: NO audio_tokens (narrower than OpenAI)
```

OpenAI-shape token-count fields (Stainless-generated SDK), but the `*_tokens_details` sub-objects are **a strict subset** of OpenAI's: no `audio_tokens` in either, no `reasoning_tokens` in completion details. AIPerf's broader OpenAI-shape coverage is forward-compatible — Cerebras responses simply don't populate the missing inner keys, and the corresponding metrics raise `NoMetricValue` rather than crashing.

### AI21 Labs

**Verified against:** [`AI21Labs/ai21-python` / `ai21/models/usage_info.py`](https://github.com/AI21Labs/ai21-python/blob/main/ai21/models/usage_info.py).

```python
class UsageInfo(AI21BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
```

Minimal OpenAI-shape — only the three baseline fields. No nested details, no cache info, no extras. Already covered.

### SambaNova

**Verified against:** [`sambanova/sambanova-python` / `src/sambanova/types/chat/chat_completion_response.py`](https://github.com/sambanova/sambanova-python/blob/main/src/sambanova/types/chat/chat_completion_response.py).

The `Usage` class is unusually rich because SambaNova bakes server-side timing/throughput data directly into the usage envelope:

```python
class Usage(BaseModel):
    # Standard OpenAI token-count fields (already covered):
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]
    prompt_tokens_details: Optional[UsagePromptTokensDetails]
    completion_tokens_details: Optional[UsageCompletionTokensDetails]

    # SambaNova-specific server-side timing (preserved on dict, not modelled):
    acceptance_rate: Optional[float]                                  # speculative-decoding accept rate
    completion_tokens_after_first_per_sec: Optional[float]            # post-TTFT throughput
    completion_tokens_after_first_per_sec_first_ten: Optional[float]  # first-10 post-TTFT throughput
    completion_tokens_after_first_per_sec_graph: Optional[float]      # adjusted for graph rendering
    completion_tokens_per_sec: Optional[float]                        # full-run completion throughput
    end_time: Optional[float]                                         # Unix timestamp seconds
    start_time: Optional[float]                                       # Unix timestamp seconds
    time_to_first_token: Optional[float]                              # TTFT seconds
    time_to_first_token_graph: Optional[float]                        # adjusted TTFT
    total_latency: Optional[float]                                    # full-run latency seconds
    total_tokens_per_sec: Optional[float]                             # full-run throughput
    is_last_response: Optional[Literal[True]]
    stop_reason: Optional[str]
```

**Modelled:** all token-count fields via OpenAI synonyms.

**Not modelled (preserved on dict):** the rich timing/throughput data. AIPerf computes equivalents client-side (`TTFTMetric`, `RequestLatencyMetric`, `OutputTokenThroughputPerUserMetric`, `InterTokenLatencyMetric`); SambaNova's server-side measurements are parallel/redundant signals. They could be surfaced as their own metrics if a workflow needed server-vs-client divergence checking.

### Bailian / DashScope (Alibaba Qwen)

**Verified against:** [`dashscope/dashscope-sdk-python` / `dashscope/api_entities/dashscope_response.py`](https://github.com/dashscope/dashscope-sdk-python/blob/main/dashscope/api_entities/dashscope_response.py).

```python
@dataclass
class GenerationUsage:                # text endpoints
    input_tokens: int
    output_tokens: int

@dataclass
class MultiModalConversationUsage:    # multimodal endpoints
    input_tokens: int
    output_tokens: int
    characters: int                   # non-token billing for non-tokenizable inputs
```

**Modelled:** `input_tokens` and `output_tokens` are already in `PROMPT_TOKENS_KEYS` / `COMPLETION_TOKENS_KEYS` (Anthropic-shape synonyms).

**Notable absences:** no `total_tokens` field (in either Bailian variant). The `total_tokens` property returns None for native DashScope responses; callers that need it can compute `input_tokens + output_tokens` themselves.

**Not modelled:** `characters` (multimodal-only). It represents image/audio inputs measured in characters rather than tokens — useful for billing reconciliation but not a standard cross-vendor metric.

**Note:** Bailian also offers an OpenAI-compatible REST endpoint (`compatible-mode`) that emits standard OpenAI shape. AIPerf benchmarking either endpoint is supported.

### Vertex AI (Gemini)

**Verified against:** [`googleapis/python-aiplatform` / `google/cloud/aiplatform_v1/types/usage_metadata.py`](https://github.com/googleapis/python-aiplatform/blob/main/google/cloud/aiplatform_v1/types/usage_metadata.py) (the protobuf message definition).

```python
class UsageMetadata(proto.Message):
    prompt_token_count: int
    candidates_token_count: int
    total_token_count: int
    tool_use_prompt_token_count: int
    thoughts_token_count: int
    cached_content_token_count: int
    prompt_tokens_details: MutableSequence[ModalityTokenCount]
    cache_tokens_details: MutableSequence[ModalityTokenCount]
    candidates_tokens_details: MutableSequence[ModalityTokenCount]
    tool_use_prompt_tokens_details: MutableSequence[ModalityTokenCount]
    traffic_type: TrafficType  # ON_DEMAND or PROVISIONED_THROUGHPUT
```

The Python proto attributes are snake_case but Google's proto JSON serialization emits **camelCase on the wire** (per the protobuf JSON style: `prompt_token_count` → `promptTokenCount`). This matches Gemini Direct's wire format exactly. Already covered by the existing Gemini synonyms.

The `traffic_type` enum (ON_DEMAND vs PROVISIONED_THROUGHPUT) is Vertex-specific — useful for cost attribution but not modelled as a metric. Preserved on the dict.

### IBM watsonx

**Verified against:** IBM watsonx text generation API documentation. The `IBM/ibm-watsonx-ai` GitHub repo I cloned was a stub (README only) and has since been removed (returns 404 as of the verification re-check); the real Python SDK ships only via PyPI / IBM Cloud Pak Foundation Models endpoints, and I did not download it. **This vendor is therefore documented from API reference rather than SDK type definitions** — flagged here so future maintainers know it's the lowest-confidence entry in this catalog.

watsonx is the only verified vendor that does **not** wrap usage in a `usage` (or equivalent) envelope. Token counts are emitted as **response-root fields**:

```json
{
  "generated_text": "...",
  "input_token_count": 100,
  "generated_token_count": 50,
  "stop_reason": "eos_token",
  "response_time": 1234,
  "scoring_id": "..."
}
```

**Modelled** (added to synonym lists at lowest precedence): `input_token_count` (in `PROMPT_TOKENS_KEYS`), `generated_token_count` (in `COMPLETION_TOKENS_KEYS`). No `total_tokens` analog — callers needing it should compute the sum themselves.

**Caveat:** because watsonx has no `usage` envelope, an AIPerf parser for watsonx would need to either pass the response-root dict to `Usage()` directly or pluck out the relevant fields. The synonym lookup handles either approach.

### xAI Grok

**Verified against:** [`xai-org/xai-sdk-python` / `src/xai_sdk/chat.py`](https://github.com/xai-org/xai-sdk-python/blob/main/src/xai_sdk/chat.py).

xAI offers two APIs: a native gRPC API and an OpenAI-compatible REST endpoint at `https://api.x.ai/v1/chat/completions`.

The gRPC path exposes additional fields not present in the REST shape:
- `cached_prompt_text_tokens` — cache hits (top-level, not nested)
- `reasoning_tokens` — top-level (not under `completion_tokens_details`)
- `prompt_text_tokens`, `prompt_image_tokens` — multimodal input split
- `cost_in_usd_ticks` — pricing in micro-cents

**AIPerf does not model these** because we benchmark via REST endpoints, not gRPC. The REST endpoint is OpenAI-compatible, so xAI usage flows through the existing OpenAI synonyms.

If gRPC-native xAI benchmarking is ever needed, adding the four gRPC field names to the appropriate `*_KEYS` lists would be a one-line change per field.

## Adding a new vendor: checklist

When you encounter a vendor not yet supported:

1. **Find the SDK source** for the vendor. Look for the type that wraps the response's `usage` field (often called `Usage`, `UsageInfo`, `CompletionUsage`, or similar). If no SDK exists, find the API documentation's response schema.
2. **Identify the wrapper.** Is the usage field at the response root, nested inside `usage`, nested inside `usageMetadata`, or in some other envelope? Snake-case or camelCase? If a Python SDK uses Pydantic with `alias_generator=to_camel`, the wire format is camelCase even though Python sees snake_case.
3. **Map each token-count field to AIPerf's properties.** Look for synonyms of `prompt_tokens`, `completion_tokens`, `total_tokens`, `reasoning_tokens`, cache reads, cache writes, etc. Add any new field names to the appropriate `*_KEYS` list in `Usage`.
4. **Identify any genuinely novel concepts** (i.e. fields with no AIPerf-side analog). If they're token-shaped and useful, add a new `BaseUsageRecordMetric` subclass in `usage_extras_metrics.py` (or `usage_cache_metrics.py` for cache-related) plus a matching `DerivedSumMetric` total in `usage_total_metrics.py`. Subclass declarations are 5–10 lines: just `tag`, `header`, `unit`, `flags`, `usage_field`, `missing_message`.
5. **If the vendor uses an envelope** (like Gemini's `usageMetadata` or Cohere's `meta`), extend `Usage.__init__` to unwrap it. Use `setdefault` so original keys win on collision.
6. **Add a fixture** to `tests/unit/common/models/test_usage_models_adversarial.py::VENDOR_FIXTURES` with a verbatim payload from the vendor's docs. Add it to the parametrized basic-token-count test.
7. **Add specific tests** for any novel fields the vendor introduces (e.g. cache misses, audio durations, modality breakdowns).
8. **Update this document.** Add a row to the quick-reference table and a per-vendor section with the SDK source citation.

## Change history

- **2026-05** — Initial cross-vendor verification. Added support for Gemini `usageMetadata`, AWS Bedrock camelCase, DeepSeek `prompt_cache_hit_tokens`/`prompt_cache_miss_tokens`, Mistral `prompt_audio_seconds`, Cohere v1 `meta` and v2 `usage` envelopes. Three real bugs found and fixed during SDK-source verification: Cohere v1 `meta.cached_tokens` lift, Cohere v2 envelope (no `meta` wrapper), Mistral `{}` sentinel defense.
- **2026-05** — Second-wave SDK-source verification covering AI21, Cerebras, SambaNova, Bailian/DashScope, Vertex AI, Fireworks, IBM watsonx. Added `input_token_count` (watsonx) to `PROMPT_TOKENS_KEYS` and `generated_token_count` (watsonx) to `COMPLETION_TOKENS_KEYS`. SambaNova's rich server-side timing fields catalogued as preserved-on-dict (parallel to client-computed metrics). Bailian's multimodal `characters` field catalogued as non-token billing unit. Vertex AI confirmed identical to Gemini direct.
