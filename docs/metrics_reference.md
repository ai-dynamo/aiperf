<!--
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# AIPerf Metrics Reference

This document provides a comprehensive reference of all metrics available in AIPerf for benchmarking LLM inference performance. Metrics are organized by computation type to help you understand when and how each metric is calculated.

## Understanding Metric Types

AIPerf computes metrics in three distinct phases during benchmark execution:

### Record Metrics

Record Metrics are computed **individually** for **each request** and its **response(s)** during the benchmark run. A single request may have one response (non-streaming) or multiple responses (streaming). These metrics capture **per-request characteristics** such as latency, token counts, and streaming behavior. Record metrics produce **statistical distributions** (min, max, mean, median, p90, p99) that reveal performance variability across requests.

**Examples:** `request_latency`, `ttft`, `inter_token_latency`, `output_token_count`, `input_sequence_length`

**Dependencies:** Record Metrics can depend on raw request/response data and other Record Metrics from the same request.

#### Example:

`request_latency` measures the time for each individual request from start to final response. If you send 100 requests, you get 100 latency values that form a distribution showing how latency varies across requests.

### Aggregate Metrics

Aggregate Metrics are computed by **tracking** or **accumulating** values across **all requests** in **real-time** during the benchmark. These include counters, min/max timestamps, and other global statistics. Aggregate metrics produce a **single value** representing the entire benchmark run.

**Examples:** `request_count`, `error_request_count`, `min_request_timestamp`, `max_response_timestamp`

**Dependencies:** Aggregate Metrics can depend on Record Metrics and other Aggregate Metrics.

#### Example:

`request_count` increments by 1 for each successful request. At the end of a benchmark with 100 successful requests, this metric equals 100 (a single value, not a distribution).

### Derived Metrics

Derived Metrics are computed by applying **mathematical formulas** to other metric results, but are **not** computed per-record like Record Metrics. Instead, these metrics depend on one or more **prerequisite metrics** being available first and are calculated either **after the benchmark completes** for final results or in **real-time** across **all current data** for live metrics display. Derived metrics can produce either single values or distributions depending on their dependencies.

**Examples:** `request_throughput`, `output_token_throughput`, `benchmark_duration`

**Dependencies:** Derived Metrics can depend on Record Metrics, Aggregate Metrics, and other Derived Metrics.

#### Example:

`request_throughput` is computed from `request_count / benchmark_duration_seconds`. This requires both `request_count` and `benchmark_duration` to be available first, then applies a formula to produce a single throughput value (e.g., 10.5 requests/sec).

---

## Quick Reference

For a quick reference of all metrics with their tags, formulas, and units, see the **[Metrics Reference section in the README](../README.md#metrics-reference)**.

The sections below provide detailed descriptions, requirements, and notes for each metric.

---

## Detailed Metric Descriptions

### Latency & Timing Metrics

These metrics measure time and latency characteristics of requests and responses.

#### Request Latency

| Tag | Type | Flags |
|-----|------|-------|
| `request_latency` | Record Metric | [`NONE`](#flag-none) |

**Description:** Measures the total end-to-end time from sending a request until receiving the final response. For streaming requests with multiple responses, this measures until the last response is received. This is the complete time experienced by the client for a single request.

**Notes:**
- Available for all request types (streaming and non-streaming); no special requirements.
- Includes all components: network time, queuing, prompt processing, token generation, and response transmission.
- For streaming requests, measures from request start to the final chunk received.

---

#### Time to First Token (TTFT)

| Tag | Type | Flags |
|-----|------|-------|
| `ttft` | Record Metric | [`STREAMING_TOKENS_ONLY`](#flag-streaming-tokens-only) |

**Description:** Measures how long it takes to receive the first token (or chunk of tokens) after sending a request. This is critical for user-perceived responsiveness in streaming scenarios, as it represents how quickly the model begins generating output.

**Notes:**
- Requires `--streaming` flag, with a token-producing endpoint, and at least 1 response chunk.
- Includes network latency, queuing time, prompt processing, and generation of the first token (or chunk of tokens).

---

#### Time to Second Token (TTST)

| Tag | Type | Flags |
|-----|------|-------|
| `ttst` | Record Metric | [`STREAMING_TOKENS_ONLY`](#flag-streaming-tokens-only) |

**Description:** Measures the time gap between the first and second chunk of tokens (SSE messages). This metric helps identify generation startup overhead separate from steady-state streaming throughput.

**Notes:**
- Requires `--streaming` flag, with a token-producing endpoint, and at least 2 response chunks (tokens).

---

#### Inter Token Latency (ITL)

| Tag | Type | Flags |
|-----|------|-------|
| `inter_token_latency` | Record Metric | [`STREAMING_TOKENS_ONLY`](#flag-streaming-tokens-only), [`LARGER_IS_BETTER`](#flag-larger-is-better) |

**Description:** Measures the average time between consecutive tokens during generation, excluding the initial TTFT overhead. This represents the steady-state token generation rate.

**Formula Details:**
```
ITL = (request_latency - ttft) / (output_sequence_length - 1)
```

**Notes:**
- Requires `--streaming` flag, with a token-producing endpoint, at least 2 output tokens, and valid `ttft`, `request_latency`, and `output_sequence_length` metrics.

---

#### Inter Chunk Latency (ICL)

| Tag | Type | Flags |
|-----|------|-------|
| `inter_chunk_latency` | Record Metric | [`STREAMING_TOKENS_ONLY`](#flag-streaming-tokens-only), [`EXPERIMENTAL`](#flag-experimental) |

**Description:** Captures the time gaps between all consecutive response chunks (SSE messages) in a streaming response, providing a distribution of chunk arrival times rather than a single average. Note that this is different from the ITL metric, which measures the time between consecutive tokens regardless of chunk size.

**Formula Details:**
```
ICL = [responses[i].perf_ns - responses[i-1].perf_ns for i in range(1, len(responses))]
```

**Notes:**
- Requires `--streaming` flag, with a token-producing endpoint, and at least 2 response chunks.
- Unlike ITL (which produces a single average), ICL provides the full distribution of inter-chunk times.
- Useful for detecting variability, jitter, or issues in streaming delivery.
- Analyzing ICL distributions can reveal batching behavior, scheduling issues, or network variability.

---

### Token Count Metrics

These metrics track token counts for individual requests and aggregated across all requests.

#### Output Token Count

| Tag | Type | Flags |
|-----|------|-------|
| `output_token_count` | Record Metric | [`PRODUCES_TOKENS_ONLY`](#flag-produces-tokens-only), [`LARGER_IS_BETTER`](#flag-larger-is-better) |

**Description:** The number of output tokens generated for a single request, _excluding reasoning tokens_. This represents the visible output tokens returned to the user across all responses for the request.

**Notes:**
- Requires token-producing endpoints that return actual token content (text); excludes embeddings and other non-generative endpoints.
- AIPerf counts tokens from the returned content using a tokenizer.
- For streaming requests with multiple responses, the responses are joined together and then tokens are counted.
- For models and endpoints that support reasoning tokens, this metric counts only the non-reasoning output tokens.
- This **will** count tokens inside of the `<think>` tags, if they are present in the `content` field of the response.

---

#### Reasoning Token Count

| Tag | Type | Flags |
|-----|------|-------|
| `reasoning_token_count` | Record Metric | [`SUPPORTS_REASONING`](#flag-supports-reasoning), [`LARGER_IS_BETTER`](#flag-larger-is-better) |

**Description:** The number of reasoning tokens generated for a single request. These are tokens used for "thinking" or chain-of-thought reasoning before generating the final output.

**Notes:**
- Requires models/backends that support reasoning output with reasoning content separated into a `reasoning_content` field, distinct from the regular `content` field in the response(s).
- AIPerf counts tokens from the `reasoning_content` field using a tokenizer, just like other token metrics.
- Does NOT differentiate `<think>` tags or extract reasoning from within the regular `content` field.
- The backend must provide reasoning as a separate field in the response.
- Standard models/backends that don't expose reasoning content separately will skip this metric.

---

#### Output Sequence Length (OSL)

| Tag | Type | Flags |
|-----|------|-------|
| `output_sequence_length` | Record Metric | [`PRODUCES_TOKENS_ONLY`](#flag-produces-tokens-only), [`LARGER_IS_BETTER`](#flag-larger-is-better) |

**Description:** The total number of completion tokens (output + reasoning) generated for a single request across all its responses. This represents the complete token generation workload for the request.

**Formula Details:**
```
OSL = (output_token_count or 0) + (reasoning_token_count or 0)
```

**Notes:**
- Requires token-producing endpoints that return text content; excludes embeddings and other non-generative endpoints.
- AIPerf counts tokens from the generated text content across all responses.
- For models and endpoints that do not support/separate reasoning tokens, OSL equals the output token count.

---

#### Input Sequence Length (ISL)

| Tag | Type | Flags |
|-----|------|-------|
| `input_sequence_length` | Record Metric | [`PRODUCES_TOKENS_ONLY`](#flag-produces-tokens-only), [`LARGER_IS_BETTER`](#flag-larger-is-better) |

**Description:** The number of input/prompt tokens for a single request. This represents the size of the input sent to the model.

**Notes:**
- Requires token-producing endpoints (chat, completion, etc.).
- AIPerf tokenizes the input prompt to compute the count using the appropriate tokenizer for the model.
- Useful for understanding the relationship between input size and latency/throughput.

---

#### Total Output Tokens

| Tag | Type | Flags |
|-----|------|-------|
| `total_output_tokens` | Derived Metric | [`PRODUCES_TOKENS_ONLY`](#flag-produces-tokens-only), [`LARGER_IS_BETTER`](#flag-larger-is-better) |

**Description:** The sum of all output tokens (excluding reasoning tokens) generated across all requests. This represents the total visible output token workload.

**Formula Details:**
```
Total Output Tokens = sum(output_token_count for record in records)
```

**Notes:**
- Requires token-producing endpoints that return text content, with valid `output_token_count` for processed records.
- AIPerf counts tokens from the returned content using a tokenizer.
- Aggregates output tokens across all successful requests.
- Useful for capacity planning and cost estimation.

---

#### Total Reasoning Tokens

| Tag | Type | Flags |
|-----|------|-------|
| `total_reasoning_tokens` | Derived Metric | [`SUPPORTS_REASONING`](#flag-supports-reasoning), [`LARGER_IS_BETTER`](#flag-larger-is-better) |

**Description:** The sum of all reasoning tokens generated across all requests. This represents the total reasoning/thinking workload.

**Formula Details:**
```
Total Reasoning Tokens = sum(reasoning_token_count for record in records)
```

**Notes:**
- Requires models/backends that support reasoning output with `reasoning_content` as a separate field, and valid `reasoning_token_count` for processed records.
- AIPerf counts tokens from the `reasoning_content` field.
- Only available for models like OpenAI o1 that expose reasoning tokens separately.
- Useful for understanding the reasoning overhead and cost for reasoning-enabled models.

---

#### Total Output Sequence Length

| Tag | Type | Flags |
|-----|------|-------|
| `total_osl` | Derived Metric | [`PRODUCES_TOKENS_ONLY`](#flag-produces-tokens-only), [`LARGER_IS_BETTER`](#flag-larger-is-better) |

**Description:** The sum of all completion tokens (output + reasoning) generated across all requests. This represents the complete token generation workload.

**Formula Details:**
```
Total Output Sequence Length = sum(output_sequence_length for record in records)
```

**Notes:**
- Requires token-producing endpoints that return text content, with valid `output_sequence_length` for processed records.
- Aggregates the complete token generation workload including both output and reasoning tokens.
- For models without reasoning tokens, this equals Total Output Tokens.
- Numerator for Output Token Throughput calculations.

---

#### Total Input Sequence Length

| Tag | Type | Flags |
|-----|------|-------|
| `total_isl` | Derived Metric | [`PRODUCES_TOKENS_ONLY`](#flag-produces-tokens-only), [`LARGER_IS_BETTER`](#flag-larger-is-better) |

**Description:** The sum of all input/prompt tokens processed across all requests. This represents the total input workload sent to the model.

**Formula Details:**
```
Total Input Sequence Length = sum(input_sequence_length for record in records)
```

**Notes:**
- Requires token-producing endpoints, with valid `input_sequence_length` for processed records.
- AIPerf tokenizes input prompts to compute token counts.
- Useful for understanding the input workload, capacity planning, and analyzing the relationship between input size and system performance.

---

### Throughput Metrics

These metrics measure the rate of requests and token generation.

#### Request Throughput

| Tag | Type | Flags |
|-----|------|-------|
| `request_throughput` | Derived Metric | [`LARGER_IS_BETTER`](#flag-larger-is-better) |

**Description:** The overall rate of completed requests per second across the entire benchmark. This represents the system's ability to process requests under the given concurrency and load.

**Formula Details:**
```
Request Throughput = request_count / benchmark_duration_seconds
```

**Notes:**
- Requires valid `request_count` and `benchmark_duration` metrics.
- Captures the aggregate request processing rate; higher values indicate better system throughput.
- Affected by concurrency level, request complexity, and system capacity.

---

#### Output Token Throughput

| Tag | Type | Flags |
|-----|------|-------|
| `output_token_throughput` | Derived Metric | [`PRODUCES_TOKENS_ONLY`](#flag-produces-tokens-only), [`LARGER_IS_BETTER`](#flag-larger-is-better) |

**Description:** The aggregate token generation rate across all concurrent requests, measured as total tokens per second. This represents the system's overall token generation capacity.

**Formula Details:**
```
Output Token Throughput = total_osl / benchmark_duration_seconds
```

**Important:** This metric specifically includes the TTFT in the equation, so it is **not** directly comparable to the [Output Token Throughput Per User](#output-token-throughput-per-user) metric.

**Notes:**
- Requires token-producing endpoints that generate text content, with valid `total_osl` and `benchmark_duration` metrics.
- Measures aggregate throughput across all concurrent requests; represents the overall system token generation rate.
- Not applicable to embeddings or other non-generative endpoints.
- Higher values indicate better system utilization and capacity.

---

#### Output Token Throughput Per User

| Tag | Type | Flags |
|-----|------|-------|
| `output_token_throughput_per_user` | Record Metric | [`STREAMING_TOKENS_ONLY`](#flag-streaming-tokens-only), [`LARGER_IS_BETTER`](#flag-larger-is-better) |

**Description:** The token generation rate experienced by an individual user/request, measured as the inverse of inter-token latency. This represents single-request streaming performance.

**Formula Details:**
```
Output Token Throughput Per User = 1.0 / inter_token_latency_seconds
```

**Important:** This metric specifically excludes the TTFT from the equation, so it is **not** directly comparable to the [Output Token Throughput](#output-token-throughput) metric.

**Notes:**
- Requires `--streaming` flag, with a token-producing endpoint, and valid `inter_token_latency` metric.
- Computes the inverse of ITL to show tokens per second from an individual user's perspective.
- Differs from Output Token Throughput (aggregate across all concurrent requests) by focusing on single-request experience.
- Useful for understanding the user experience independent of concurrency effects.

---

### System & Benchmark Metrics

These metrics track overall benchmark execution and system-level counters.

#### Request Count

| Tag | Type | Flags |
|-----|------|-------|
| `request_count` | Aggregate Metric | [`LARGER_IS_BETTER`](#flag-larger-is-better) |

**Description:** The total number of **successfully completed** requests in the benchmark. This includes all requests that received valid responses, regardless of streaming mode.

---

#### Error Request Count

| Tag | Type | Flags |
|-----|------|-------|
| `error_request_count` | Aggregate Metric | [`ERROR_ONLY`](#flag-error-only) |

**Description:** The total number of failed/error requests encountered during the benchmark. This includes network errors, HTTP errors, timeout errors, and other failures.

**Notes:**
- Error rate can be computed as `error_request_count / (request_count + error_request_count)`.

---

#### Minimum Request Timestamp

| Tag | Type | Flags |
|-----|------|-------|
| `min_request_timestamp` | Aggregate Metric | [`NO_CONSOLE`](#flag-no-console) |

**Description:** The wall-clock timestamp of the first request sent in the benchmark. This is used to calculate the benchmark duration and represents the start of the benchmark run.

---

#### Maximum Response Timestamp

| Tag | Type | Flags |
|-----|------|-------|
| `max_response_timestamp` | Aggregate Metric | [`NO_CONSOLE`](#flag-no-console) |

**Description:** The wall-clock timestamp of the last response received in the benchmark. This is used to calculate the benchmark duration and represents the end of the benchmark run.

---

#### Benchmark Duration

| Tag | Type | Flags |
|-----|------|-------|
| `benchmark_duration` | Derived Metric | [`NO_CONSOLE`](#flag-no-console) |

**Description:** The total elapsed time from the first request sent to the last response received. This represents the complete wall-clock duration of the benchmark run.

**Formula Details:**
```
Benchmark Duration = max_response_timestamp - min_request_timestamp
```

**Notes:**
- Requires valid `min_request_timestamp` and `max_response_timestamp` metrics.
- Uses wall-clock timestamps representing real calendar time.
- Denominator for throughput calculations; represents the effective measurement window.

---

## Metric Flags Reference

Metric flags are used to control when and how metrics are computed, displayed, and grouped. Flags can be combined using bitwise operations to create composite behaviors.

| Flag | Description | Impact |
|------|-------------|--------|
| <a id="flag-none"></a>`NONE` | No flags set | Metric has default behavior with no special restrictions |
| <a id="flag-streaming-only"></a>`STREAMING_ONLY` | Only computed for streaming responses | Requires Server-Sent Events (SSE) with multiple response chunks; skipped for non-streaming requests |
| <a id="flag-error-only"></a>`ERROR_ONLY` | Only computed for error requests | Tracks error-specific information; skipped for successful requests |
| <a id="flag-produces-tokens-only"></a>`PRODUCES_TOKENS_ONLY` | Only computed for token-producing endpoints | Requires endpoints that return text/token content; skipped for embeddings and non-generative endpoints |
| <a id="flag-no-console"></a>`NO_CONSOLE` | Not displayed in console output | Metric computed but excluded from terminal display; available in JSON/CSV exports |
| <a id="flag-larger-is-better"></a>`LARGER_IS_BETTER` | Higher values indicate better performance | Used for throughput and count metrics to indicate optimization direction |
| <a id="flag-internal"></a>`INTERNAL` | Internal system metric (also `NO_CONSOLE`) | Not user-facing; used for internal processing and debugging |
| <a id="flag-supports-audio-only"></a>`SUPPORTS_AUDIO_ONLY` | Only applicable to audio endpoints | Requires audio-based input/output; skipped for text-only endpoints |
| <a id="flag-supports-image-only"></a>`SUPPORTS_IMAGE_ONLY` | Only applicable to image endpoints | Requires image-based input/output; skipped for text-only endpoints |
| <a id="flag-supports-reasoning"></a>`SUPPORTS_REASONING` | Requires reasoning token support | Only available for models that expose reasoning content in separate fields (e.g., OpenAI o1) |
| <a id="flag-experimental"></a>`EXPERIMENTAL` | Experimental feature (also `NO_CONSOLE`) | Not production-ready; subject to change; excluded from default display |
| <a id="flag-streaming-tokens-only"></a>`STREAMING_TOKENS_ONLY` | Combination: `STREAMING_ONLY` + `PRODUCES_TOKENS_ONLY` | Requires both streaming support and token-producing endpoints |

---
