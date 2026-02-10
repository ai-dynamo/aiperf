<!--
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# AIPerf Metrics Reference

This document provides a comprehensive reference of all metrics available in AIPerf for benchmarking LLM inference performance. Metrics are organized by computation type to help you understand when and how each metric is calculated.

## Table of Contents

- [Quick Reference](#quick-reference)
- [Understanding Metric Types](#understanding-metric-types)
  - [Record Metrics](#record-metrics)
  - [Aggregate Metrics](#aggregate-metrics)
  - [Derived Metrics](#derived-metrics)
- [Detailed Metric Descriptions](#detailed-metric-descriptions)
  - [Streaming Metrics](#streaming-metrics)
    - [Time to First Token (TTFT)](#time-to-first-token-ttft)
    - [Time to Second Token (TTST)](#time-to-second-token-ttst)
    - [Time to First Output Token (TTFO)](#time-to-first-output-token-ttfo)
    - [Inter Token Latency (ITL)](#inter-token-latency-itl)
    - [Inter Chunk Latency (ICL)](#inter-chunk-latency-icl)
    - [Output Token Throughput Per User](#output-token-throughput-per-user)
    - [Prefill Throughput Per User](#prefill-throughput-per-user)
  - [Token Based Metrics](#token-based-metrics)
    - [Output Token Count](#output-token-count)
    - [Output Sequence Length (OSL)](#output-sequence-length-osl)
    - [Input Sequence Length (ISL)](#input-sequence-length-isl)
    - [Total Output Tokens](#total-output-tokens)
    - [Total Output Sequence Length](#total-output-sequence-length)
    - [Total Input Sequence Length](#total-input-sequence-length)
    - [Output Token Throughput](#output-token-throughput)
    - [Total Token Throughput](#total-token-throughput)
  - [Image Metrics](#image-metrics)
    - [Image Throughput](#image-throughput)
    - [Image Latency](#image-latency)
  - [Reasoning Metrics](#reasoning-metrics)
    - [Reasoning Token Count](#reasoning-token-count)
    - [Total Reasoning Tokens](#total-reasoning-tokens)
  - [Usage Field Metrics](#usage-field-metrics)
    - [Usage Prompt Tokens](#usage-prompt-tokens)
    - [Usage Completion Tokens](#usage-completion-tokens)
    - [Usage Total Tokens](#usage-total-tokens)
    - [Usage Reasoning Tokens](#usage-reasoning-tokens)
    - [Total Usage Prompt Tokens](#total-usage-prompt-tokens)
    - [Total Usage Completion Tokens](#total-usage-completion-tokens)
    - [Total Usage Total Tokens](#total-usage-total-tokens)
  - [Usage Discrepancy Metrics](#usage-discrepancy-metrics)
    - [Usage Prompt Tokens Diff %](#usage-prompt-tokens-diff-)
    - [Usage Completion Tokens Diff %](#usage-completion-tokens-diff-)
    - [Usage Reasoning Tokens Diff %](#usage-reasoning-tokens-diff-)
    - [Usage Discrepancy Count](#usage-discrepancy-count)
  - [OSL Mismatch Metrics](#osl-mismatch-metrics)
    - [OSL Mismatch Diff %](#osl-mismatch-diff-)
    - [OSL Mismatch Count](#osl-mismatch-count)
  - [Goodput Metrics](#goodput-metrics)
    - [Good Request Count](#good-request-count)
    - [Goodput](#goodput)
  - [Error Metrics](#error-metrics)
    - [Error Input Sequence Length](#error-input-sequence-length)
    - [Total Error Input Sequence Length](#total-error-input-sequence-length)
  - [General Metrics](#general-metrics)
    - [Request Latency](#request-latency)
    - [Request Throughput](#request-throughput)
    - [Request Count](#request-count)
    - [Error Request Count](#error-request-count)
    - [Minimum Request Timestamp](#minimum-request-timestamp)
    - [Maximum Response Timestamp](#maximum-response-timestamp)
    - [Benchmark Duration](#benchmark-duration)
  - [HTTP Trace Metrics](#http-trace-metrics)
    - [HTTP Blocked](#http-blocked)
    - [HTTP DNS Lookup](#http-dns-lookup)
    - [HTTP Connecting](#http-connecting)
    - [HTTP Sending](#http-sending)
    - [HTTP Waiting (TTFB)](#http-waiting-ttfb)
    - [HTTP Receiving](#http-receiving)
    - [HTTP Duration](#http-duration)
    - [HTTP Connection Overhead](#http-connection-overhead)
    - [HTTP Total Time](#http-total-time)
    - [HTTP Data Sent](#http-data-sent)
    - [HTTP Data Received](#http-data-received)
    - [HTTP Connection Reused](#http-connection-reused)
- [Metric Flags Reference](#metric-flags-reference)

---

## Quick Reference

For a quick reference of all metrics with their tags, formulas, and units, see the **[Metrics Reference section in the README](../README.md#metrics-reference)**.

The sections below provide detailed descriptions, requirements, and notes for each metric.

---

## Understanding Metric Types

AIPerf computes metrics in three distinct phases during benchmark execution: **Record Metrics**, **Aggregate Metrics**, and **Derived Metrics**.

## Record Metrics

Record Metrics are computed **individually** for **each request** and its **response(s)** during the benchmark run. A single request may have one response (non-streaming) or multiple responses (streaming). These metrics capture **per-request characteristics** such as latency, token counts, and streaming behavior. Record metrics produce **statistical distributions** (min, max, mean, median, p90, p99, etc.) that reveal performance variability across requests.

### Example Metrics
`request_latency`, `time_to_first_token`, `inter_token_latency`, `output_token_count`, `input_sequence_length`

### Dependencies
Record Metrics can depend on raw request/response data and other Record Metrics from the same request.

### Example Scenario
`request_latency` measures the time for each individual request from start to final response. If you send 100 requests, you get 100 latency values that form a distribution showing how latency varies across requests.

## Aggregate Metrics

Aggregate Metrics are computed by **tracking** or **accumulating** values across **all requests** in **real-time** during the benchmark. These include counters, min/max timestamps, and other global statistics. Aggregate metrics produce a **single value** representing the entire benchmark run.

### Example Metrics
`request_count`, `error_request_count`, `min_request_timestamp`, `max_response_timestamp`

### Dependencies
Aggregate Metrics can depend on raw request/response data, Record Metrics and other Aggregate Metrics.

### Example Scenario
`request_count` increments by 1 for each successful request. At the end of a benchmark with 100 successful requests, this metric equals 100 (a single value, not a distribution).

## Derived Metrics

Derived Metrics are computed by applying **mathematical formulas** to other metric results, but are **not** computed per-record like Record Metrics. Instead, these metrics depend on one or more **prerequisite metrics** being available first and are calculated either **after the benchmark completes** for final results or in **real-time** across **all current data** for live metrics display. Derived metrics can produce either single values or distributions depending on their dependencies.

### Example Metrics
`request_throughput`, `output_token_throughput`, `benchmark_duration`

### Dependencies
Derived Metrics can depend on Record Metrics, Aggregate Metrics, and other Derived Metrics, but do not have
any knowledge of the individual request/response data.

### Example Scenario
`request_throughput` is computed from `request_count / benchmark_duration_seconds`. This requires both `request_count` and `benchmark_duration` to be available first, then applies a formula to produce a single throughput value (e.g., 10.5 requests/sec).

---

# Detailed Metric Descriptions

## Streaming Metrics

> [!NOTE]
> All metrics in this section require the `--streaming` flag with a token-producing endpoint and at least one non-empty response chunk.

### Time to First Token (TTFT)

**Type:** [Record Metric](#record-metrics)

Measures how long it takes to receive the first token (or chunk of tokens) after sending a request. This is critical for user-perceived responsiveness in streaming scenarios, as it represents how quickly the model begins generating output.

**Formula:**
```python
# nanoseconds
ttft_ns = request.content_responses[0].perf_ns - request.start_perf_ns

# Convert to milliseconds for display
ttft_ms = ttft_ns / 1e6

# Convert to seconds for throughput calculations
ttft_seconds = ttft_ns / 1e9
```

**Notes:**
- Includes network latency, queuing time, prompt processing, and generation of the first token (or chunk of tokens).
- Raw timestamps are in nanoseconds; converted to milliseconds for display and seconds for rate calculations.
- Response chunks refer to individual messages with non-empty content received during streaming.

---

### Time to Second Token (TTST)

**Type:** [Record Metric](#record-metrics)

Measures the time gap between the first and second chunk of tokens. This metric helps identify generation startup overhead separate from steady-state streaming throughput.

**Formula:**
```python
# nanoseconds
ttst_ns = request.content_responses[1].perf_ns - request.content_responses[0].perf_ns

# Convert to milliseconds for display
ttst_ms = ttst_ns / 1e6
```

**Notes:**
- Requires at least 2 non-empty response chunks to compute the time between first and second tokens.
- Raw timestamps are in nanoseconds; converted to milliseconds for display.

---

### Time to First Output Token (TTFO)

**Type:** [Record Metric](#record-metrics)

Calculates the time elapsed from request start to the first non-reasoning output token. This metric measures the latency from when a request is initiated to when the first actual output token (non-reasoning content) is received. It is particularly relevant for models that perform extended reasoning before generating output.

**Formula:**
```python
# nanoseconds
# First non-reasoning token: TextResponseData with non-empty text, or
# ReasoningResponseData with non-empty content field
ttfo_ns = first_non_reasoning_token_perf_ns - request.start_perf_ns

# Convert to milliseconds for display
ttfo_ms = ttfo_ns / 1e6
```

**Notes:**
- TTFO vs TTFT: Time to First Output (TTFO) measures time to the first non-reasoning token, while Time to First Token (TTFT) measures time to any first token including reasoning tokens. For models without reasoning, TTFO and TTFT are equivalent.
- Non-reasoning tokens include TextResponseData with non-empty text, or ReasoningResponseData with non-empty content field (regardless of reasoning field).
- Requires at least one non-empty non-reasoning response chunk.

---

### Inter Token Latency (ITL)

**Type:** [Record Metric](#record-metrics)

Measures the average time between consecutive tokens during generation, excluding the initial TTFT overhead. This represents the steady-state token generation rate.

**Formula:**
```python
# Calculate in nanoseconds, then convert to seconds
inter_token_latency_ns = (request_latency_ns - time_to_first_token_ns) / (output_sequence_length - 1)

# Convert to seconds for throughput calculations
inter_token_latency_seconds = inter_token_latency_ns / 1e9

# Convert to milliseconds for display
inter_token_latency_ms = inter_token_latency_ns / 1e6
```

**Notes:**
- Requires at least 2 non-empty response chunks and valid `time_to_first_token`, `request_latency`, and `output_sequence_length` metrics.
- Result is in seconds when used for throughput calculations (Output Token Throughput Per User).

---

### Inter Chunk Latency (ICL)

**Type:** [Record Metric](#record-metrics)

Captures the time gaps between all consecutive response chunks in a streaming response, providing a distribution of chunk arrival times rather than a single average. Note that this is different from the ITL metric, which measures the time between consecutive tokens regardless of chunk size.

**Formula:**
```python
inter_chunk_latency = [request.content_responses[i].perf_ns - request.content_responses[i-1].perf_ns for i in range(1, len(request.content_responses))]
```

**Notes:**
- Requires at least 2 response chunks.
- Unlike ITL (which produces a single average), ICL provides the full distribution of inter-chunk times.
- Useful for detecting variability, jitter, or issues in streaming delivery.
- Analyzing ICL distributions can reveal batching behavior, scheduling issues, or network variability.

---

### Output Token Throughput Per User

**Type:** [Record Metric](#record-metrics)

> [!IMPORTANT]
> This metric is computed per-request, and it excludes the TTFT from the equation, so it is **not** directly comparable to the [Output Token Throughput](#output-token-throughput) metric.

The token generation rate experienced by an individual user/request, measured as the inverse of inter-token latency. This represents single-request streaming performance.

**Formula:**
```python
output_token_throughput_per_user = 1.0 / inter_token_latency_seconds
```

**Notes:**
- Computes the inverse of ITL to show tokens per second from an individual user's perspective.
- Differs from Output Token Throughput (aggregate across all concurrent requests) by focusing on single-request experience.
- Useful for understanding the user experience independent of concurrency effects.

---

### Prefill Throughput Per User

**Type:** [Record Metric](#record-metrics)

Measures the rate at which input tokens are processed during the prefill phase, calculated as input tokens per second based on TTFT. This is only applicable to streaming responses.

**Formula:**
```python
prefill_throughput_per_user = input_sequence_length / time_to_first_token_seconds
```

**Notes:**
- Higher values indicate faster prompt processing.
- Useful for understanding input processing capacity and bottlenecks.
- Depends on Input Sequence Length and TTFT metrics.

---

## Token Based Metrics

> [!NOTE]
> All metrics in this section require token-producing endpoints that return text content (chat, completion, etc.). These metrics are not available for embeddings or other non-generative endpoints.

### Output Token Count

**Type:** [Record Metric](#record-metrics)

The number of output tokens generated for a single request, _excluding reasoning tokens_. This represents the output tokens returned to the user across all responses for the request.

**Formula:**
```python
output_token_count = len(tokenizer.encode(content, add_special_tokens=False))
```

**Notes:**
- Tokenization uses `add_special_tokens=False` to count only content tokens, excluding special tokens added by the tokenizer.
- For streaming requests with multiple responses, the responses are joined together and then tokens are counted.
- For models that expose reasoning in a separate `reasoning_content` field, this metric counts only non-reasoning output tokens.
- If reasoning appears inside the regular `content` (e.g., `<think>` blocks), those tokens will be counted unless explicitly filtered.

---

### Output Sequence Length (OSL)

**Type:** [Record Metric](#record-metrics)

The total number of completion tokens (output + reasoning) generated for a single request across all its responses. This represents the complete token generation workload for the request.

**Formula:**
```python
output_sequence_length = (output_token_count or 0) + (reasoning_token_count or 0)
```

**Notes:**
- For models that do not support/separate reasoning tokens, OSL equals the output token count.

---

### Input Sequence Length (ISL)

**Type:** [Record Metric](#record-metrics)

The number of input/prompt tokens for a single request. This represents the size of the input sent to the model.

**Formula:**
```python
input_sequence_length = len(tokenizer.encode(prompt, add_special_tokens=False))
```

**Notes:**
- Tokenization uses `add_special_tokens=False` to count only content tokens, excluding special tokens added by the tokenizer.
- Useful for understanding the relationship between input size and latency/throughput.

---

### Total Output Tokens

**Type:** [Aggregate Metric](#aggregate-metrics)

The sum of all output tokens (excluding reasoning tokens) generated across all requests. This represents the total output token workload.

**Formula:**
```python
total_output_tokens = sum(r.output_token_count for r in records if r.valid)
```

**Notes:**
- Aggregates output tokens across all successful requests.
- Useful for capacity planning and cost estimation.

---

### Total Output Sequence Length

**Type:** [Aggregate Metric](#aggregate-metrics)

The sum of all completion tokens (output + reasoning) generated across all requests. This represents the complete token generation workload.

**Formula:**
```python
total_osl = sum(r.output_sequence_length for r in records if r.valid)
```

**Notes:**
- Aggregates the complete token generation workload including both output and reasoning tokens.
- For models without reasoning tokens, this equals Total Output Tokens.

---

### Total Input Sequence Length

**Type:** [Aggregate Metric](#aggregate-metrics)

The sum of all input/prompt tokens processed across all requests. This represents the total input workload sent to the model.

**Formula:**
```python
total_isl = sum(r.input_sequence_length for r in records if r.valid)
```

**Notes:**
- Useful for understanding the input workload, capacity planning, and analyzing the relationship between input size and system performance.

---

### Output Token Throughput

**Type:** [Derived Metric](#derived-metrics)

> [!IMPORTANT]
> This metric is computed as a single value across all requests and includes TTFT in the equation, so it is **not** directly comparable to the [Output Token Throughput Per User](#output-token-throughput-per-user) metric.

The aggregate token generation rate across all concurrent requests, measured as total tokens per second. This represents the system's overall token generation capacity.

**Formula:**
```python
output_token_throughput = total_osl / benchmark_duration_seconds
```

**Notes:**
- Measures aggregate throughput across all concurrent requests; represents the overall system token generation rate.
- Higher values indicate better system utilization and capacity.

---

### Total Token Throughput

**Type:** [Derived Metric](#derived-metrics)

Calculates the total token throughput metric, combining both input and output token processing across all concurrent requests.

**Formula:**
```python
total_token_throughput = (total_isl + total_osl) / benchmark_duration_seconds
```

**Notes:**
- Measures the combined input and output token processing rate.
- Includes reasoning tokens in the output count (via total_osl).
- Useful for understanding total system token processing capacity.

---

## Image Metrics

> [!NOTE]
> All metrics in this section require image-capable endpoints (e.g., image generation APIs). These metrics are not available for text-only or other non-image endpoints.

### Image Throughput

**Type:** [Record Metric](#record-metrics)

Calculates the image throughput from the record by dividing the number of images by the request latency.

**Formula:**
```python
image_throughput = num_images / request_latency_seconds
```

**Notes:**
- Higher values indicate faster image generation.

---

### Image Latency

**Type:** [Record Metric](#record-metrics)

Calculates the image latency from the record by dividing the request latency by the number of images.

**Formula:**
```python
image_latency = request_latency_ms / num_images
```

**Notes:**
- Lower values indicate faster per-image generation.

---

## Reasoning Metrics

> [!NOTE]
> All metrics in this section require models and backends that expose reasoning content in a separate `reasoning_content` field, distinct from the regular `content` field.

### Reasoning Token Count

**Type:** [Record Metric](#record-metrics)

The number of reasoning tokens generated for a single request. These are tokens used for "thinking" or chain-of-thought reasoning before generating the final output.

**Formula:**
```python
reasoning_token_count = len(tokenizer.encode(reasoning_content, add_special_tokens=False))
```

**Notes:**
- Tokenization uses `add_special_tokens=False` to count only content tokens, excluding special tokens added by the tokenizer.
- Does **not** differentiate `<think>` tags or extract reasoning from within the regular `content` field.

---

### Total Reasoning Tokens

**Type:** [Aggregate Metric](#aggregate-metrics)

The sum of all reasoning tokens generated across all requests. This represents the total reasoning/thinking workload.

**Formula:**
```python
total_reasoning_tokens = sum(r.reasoning_token_count for r in records if r.valid)
```

**Notes:**
- Useful for understanding the reasoning overhead and cost for reasoning-enabled models.

---

## Usage Field Metrics

> [!NOTE]
> All metrics in this section track API-reported token counts from the `usage` field in API responses. These are **not displayed in console output** but are available in exports. These metrics are useful for comparing client-side token counts with server-reported counts to detect discrepancies.

### Usage Prompt Tokens

**Type:** [Record Metric](#record-metrics)

The number of input/prompt tokens as reported by the API's `usage.prompt_tokens` field for a single request.

**Formula:**
```python
usage_prompt_tokens = response.usage.prompt_tokens  # from last non-None response
```

**Notes:**
- Taken from the API response `usage` object, not computed by AIPerf.
- May differ from client-side Input Sequence Length due to different tokenizers or special tokens.
- For streaming responses, uses the last non-None value reported.

---

### Usage Completion Tokens

**Type:** [Record Metric](#record-metrics)

The number of completion tokens as reported by the API's `usage.completion_tokens` field for a single request.

**Formula:**
```python
usage_completion_tokens = response.usage.completion_tokens  # from last non-None response
```

**Notes:**
- Taken from the API response `usage` object, not computed by AIPerf.
- May differ from client-side Output Sequence Length due to different tokenizers or counting methods.
- For streaming responses, uses the last non-None value reported.

---

### Usage Total Tokens

**Type:** [Record Metric](#record-metrics)

The total number of tokens (prompt + completion) as reported by the API's `usage.total_tokens` field for a single request.

**Formula:**
```python
usage_total_tokens = response.usage.total_tokens  # from last non-None response
```

**Notes:**
- Taken from the API response `usage` object, not computed by AIPerf.
- Should generally equal `usage_prompt_tokens + usage_completion_tokens`.
- For streaming responses, uses the last non-None value reported.

---

### Usage Reasoning Tokens

**Type:** [Record Metric](#record-metrics)

The number of reasoning tokens as reported by the API's `usage.completion_tokens_details.reasoning_tokens` field for a single request. Only available for reasoning-enabled models.

**Formula:**
```python
usage_reasoning_tokens = response.usage.completion_tokens_details.reasoning_tokens
```

**Notes:**
- Taken from the API response for reasoning-enabled models.
- May differ from client-side Reasoning Token Count due to different tokenizers.
- For streaming responses, uses the last non-None value reported.

---

### Total Usage Prompt Tokens

**Type:** [Aggregate Metric](#aggregate-metrics)

The sum of all API-reported prompt tokens across all requests.

**Formula:**
```python
total_usage_prompt_tokens = sum(r.usage_prompt_tokens for r in records if r.valid)
```

**Notes:**
- Aggregates server-reported input tokens across all requests.

---

### Total Usage Completion Tokens

**Type:** [Aggregate Metric](#aggregate-metrics)

The sum of all API-reported completion tokens across all requests.

**Formula:**
```python
total_usage_completion_tokens = sum(r.usage_completion_tokens for r in records if r.valid)
```

**Notes:**
- Aggregates server-reported completion tokens across all requests.

---

### Total Usage Total Tokens

**Type:** [Aggregate Metric](#aggregate-metrics)

The sum of all API-reported total tokens across all requests.

**Formula:**
```python
total_usage_total_tokens = sum(r.usage_total_tokens for r in records if r.valid)
```

**Notes:**
- Aggregates server-reported total tokens across all requests.

---

## Usage Discrepancy Metrics

> [!NOTE]
> These metrics measure the percentage difference between API-reported token counts (`usage` fields) and client-computed token counts. They are **not displayed in console output** but help identify tokenizer mismatches or counting discrepancies.

### Usage Prompt Tokens Diff %

**Type:** [Record Metric](#record-metrics)

The percentage difference between API-reported prompt tokens and client-computed Input Sequence Length.

**Formula:**
```python
usage_prompt_tokens_diff_pct = abs((usage_prompt_tokens - input_sequence_length) / input_sequence_length) * 100
```

**Notes:**
- Values close to 0% indicate good agreement between client and server token counts.
- Large differences may indicate tokenizer mismatches or special token handling differences.

---

### Usage Completion Tokens Diff %

**Type:** [Record Metric](#record-metrics)

The percentage difference between API-reported completion tokens and client-computed Output Sequence Length.

**Formula:**
```python
usage_completion_tokens_diff_pct = abs((usage_completion_tokens - output_sequence_length) / output_sequence_length) * 100
```

**Notes:**
- Values close to 0% indicate good agreement between client and server token counts.
- Large differences may indicate tokenizer mismatches or different counting methods.

---

### Usage Reasoning Tokens Diff %

**Type:** [Record Metric](#record-metrics)

The percentage difference between API-reported reasoning tokens and client-computed Reasoning Token Count.

**Formula:**
```python
usage_reasoning_tokens_diff_pct = abs((usage_reasoning_tokens - reasoning_token_count) / reasoning_token_count) * 100
```

**Notes:**
- Only available for reasoning-enabled models.
- Values close to 0% indicate good agreement between client and server reasoning token counts.

---

### Usage Discrepancy Count

**Type:** [Aggregate Metric](#aggregate-metrics)

The number of requests where token count differences exceed a threshold (default 10%).

**Formula:**
```python
usage_discrepancy_count = sum(1 for r in records if r.any_diff > threshold)
```

**Notes:**
- Default threshold is 10% difference.
- Counts requests where prompt, completion, or reasoning token differences are significant.
- Useful for monitoring overall token count agreement quality.

---

## OSL Mismatch Metrics

> [!NOTE]
> These metrics measure the difference between requested output sequence length (`--osl`/`max_tokens`) and actual output tokens generated. They help identify when the server is not honoring the requested output length, typically because EOS tokens stop generation early. These metrics are **not displayed in console output** but are available in exports and used by the end-of-benchmark warning.

### OSL Mismatch Diff %

**Type:** [Record Metric](#record-metrics)

The signed percentage difference between actual output sequence length and requested OSL. Negative values mean the server stopped early (actual < requested), positive values mean it generated more than requested.

**Formula:**
```python
osl_mismatch_diff_pct = ((output_sequence_length - requested_osl) / requested_osl) * 100
```

**Notes:**
- Negative = stopped early (hit EOS before max_tokens)
- Positive = generated more than requested
- 0% = exact match between requested and actual
- Example: Requested 100 tokens, got 50 → Diff = -50%
- Example: Requested 100 tokens, got 120 → Diff = 20%

---

### OSL Mismatch Count

**Type:** [Aggregate Metric](#aggregate-metrics)

The count of requests where the absolute token difference exceeds the effective threshold. Used to trigger the end-of-benchmark warning panel.

**Formula:**
```python
# Effective threshold is capped to be tighter for large OSL values
threshold_tokens = min(requested_osl * (pct_threshold / 100), max_token_threshold)
diff_tokens = abs(actual_osl - requested_osl)
osl_mismatch_count = sum(1 for r in records if diff_tokens > threshold_tokens)
```

**Notes:**
- Default percentage threshold is 5% (`AIPERF_METRICS_OSL_MISMATCH_PCT_THRESHOLD`).
- Default max token threshold is 50 (`AIPERF_METRICS_OSL_MISMATCH_MAX_TOKEN_THRESHOLD`).
- The `min()` makes threshold tighter for large OSL: requesting 2000 tokens caps at 50 token diff instead of 100 (5%).
- Counts both early stops (negative diff) and over-generation (positive diff).
- When this count is non-zero, a warning panel is displayed at the end of the benchmark.
- To ensure servers honor `--osl`, use `--extra-inputs ignore_eos:true` or `--extra-inputs min_tokens:<value>`.
- If discrepancy is due to tokenizer mismatch between client and server, use `--use-server-token-count`.

**Server support for `min_tokens`:**

| Server | Parameter | Notes |
|--------|-----------|-------|
| [vLLM](https://docs.vllm.ai/en/latest/api/vllm/sampling_params/) | `min_tokens` | Default: 0 |
| [TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/llm-api/reference.html) | `min_tokens` | Default: 1 |
| [SGLang](https://github.com/sgl-project/sglang) | `min_new_tokens` | Default: 0 |
| [TGI](https://github.com/huggingface/text-generation-inference) | `min_new_tokens` | ⚠️ Unclear API support; TGI in maintenance mode |

---

## Goodput Metrics

> [!NOTE]
> Goodput metrics measure the throughput of requests that meet user-defined Service Level Objectives (SLOs). See the [Goodput tutorial](../docs/tutorials/goodput.md) for configuration details.

### Good Request Count

**Type:** [Aggregate Metric](#aggregate-metrics)

The number of requests that meet all user-defined SLO thresholds during the benchmark.

**Formula:**
```python
good_request_count = sum(1 for r in records if r.all_slos_met)
```

**Notes:**
- Requires SLO thresholds to be configured (e.g., `--goodput`).
- Only counts requests where ALL SLO constraints are satisfied.
- Used to calculate Goodput metric.

---

### Goodput

**Type:** [Derived Metric](#derived-metrics)

The rate of SLO-compliant requests per second. This represents the effective throughput of requests meeting quality requirements.

**Formula:**
```python
goodput = good_request_count / benchmark_duration_seconds
```

**Notes:**
- Requires SLO thresholds to be configured.
- Always less than or equal to Request Throughput.
- Useful for capacity planning and comparing systems based on quality-adjusted throughput.

---

## Error Metrics

> [!NOTE]
> These metrics are computed only for failed/error requests and are **not displayed in console output**.

### Error Input Sequence Length

**Type:** [Record Metric](#record-metrics)

The number of input tokens for requests that resulted in errors. This helps analyze whether input size correlates with errors.

**Formula:**
```python
error_isl = input_sequence_length  # for error requests only
```

**Notes:**
- Only computed for requests that failed.
- Useful for identifying if certain input sizes trigger errors.

---

### Total Error Input Sequence Length

**Type:** [Aggregate Metric](#aggregate-metrics)

The sum of all input tokens from requests that resulted in errors.

**Formula:**
```python
total_error_isl = sum(r.input_sequence_length for r in records if not r.valid)
```

**Notes:**
- Aggregates input tokens across all failed requests.

---

## General Metrics

> [!NOTE]
> Metrics in this section are available for all benchmark runs with no special requirements.

### Request Latency

**Type:** [Record Metric](#record-metrics)

Measures the total end-to-end time from sending a request until receiving the final response. For streaming requests with multiple responses, this measures until the last response is received. This is the complete time experienced by the client for a single request.

**Formula:**
```python
request_latency_ns = request.content_responses[-1].perf_ns - request.start_perf_ns
```

**Notes:**
- Includes all components: network time, queuing, prompt processing, token generation, and response transmission.
- For streaming requests, measures from request start to the final chunk received.

---

### Request Throughput

**Type:** [Derived Metric](#derived-metrics)

The overall rate of completed requests per second across the entire benchmark. This represents the system's ability to process requests under the given concurrency and load.

**Formula:**
```python
request_throughput = request_count / benchmark_duration_seconds
```

**Notes:**
- Captures the aggregate request processing rate; higher values indicate better system throughput.
- Affected by concurrency level, request complexity, output sequence length, and system capacity.

---

### Request Count

**Type:** [Aggregate Metric](#aggregate-metrics)

The total number of **successfully completed** requests in the benchmark. This includes all requests that received valid responses, regardless of streaming mode.

**Formula:**
```python
request_count = sum(1 for r in records if r.valid)
```

---

### Error Request Count

**Type:** [Aggregate Metric](#aggregate-metrics)

The total number of failed/error requests encountered during the benchmark. This includes network errors, HTTP errors, timeout errors, and other failures.

**Formula:**
```python
error_request_count = sum(1 for r in records if not r.valid)
```

**Notes:**
- Error rate can be computed as `error_request_count / (request_count + error_request_count)`.

---

### Minimum Request Timestamp

**Type:** [Aggregate Metric](#aggregate-metrics)

The wall-clock timestamp of the first request sent in the benchmark. This is used to calculate the benchmark duration and represents the start of the benchmark run.

**Formula:**
```python
min_request_timestamp = min(r.timestamp_ns for r in records)
```

---

### Maximum Response Timestamp

**Type:** [Aggregate Metric](#aggregate-metrics)

The wall-clock timestamp of the last response received in the benchmark. This is used to calculate the benchmark duration and represents the end of the benchmark run.

**Formula:**
```python
max_response_timestamp = max(r.timestamp_ns + r.request_latency for r in records)
```

---

### Benchmark Duration

**Type:** [Derived Metric](#derived-metrics)

The total elapsed time from the first request sent to the last response received. This represents the complete wall-clock duration of the benchmark run.

**Formula:**
```python
benchmark_duration = max_response_timestamp - min_request_timestamp
```

**Notes:**
- Uses wall-clock timestamps representing real calendar time.
- Used as the denominator for throughput calculations; represents the effective measurement window.

---

## HTTP Trace Metrics

> [!NOTE]
> All metrics in this section require HTTP trace data to be collected during requests. These metrics provide detailed HTTP request lifecycle timing following k6 naming conventions. See the [HTTP Trace Metrics tutorial](../docs/tutorials/http-trace-metrics.md) for configuration details.

### HTTP Blocked

**Type:** [Record Metric](#record-metrics)

Time spent blocked waiting for a free TCP connection slot from the pool. This metric measures the time a request spent waiting in the connection pool queue before a connection became available. High values indicate connection pool saturation.

**Formula:**
```python
http_req_blocked = connection_pool_wait_end_perf_ns - connection_pool_wait_start_perf_ns
```

**Notes:**
- k6 equivalent: `http_req_blocked`
- HAR equivalent: `blocked`
- Returns 0 if no pool wait occurred (connection immediately available).
- Only available for AioHttpTraceData.

---

### HTTP DNS Lookup

**Type:** [Record Metric](#record-metrics)

Time spent on DNS resolution. This metric measures the time spent resolving the hostname to an IP address.

**Formula:**
```python
http_req_dns_lookup = dns_lookup_end_perf_ns - dns_lookup_start_perf_ns
```

**Notes:**
- k6 equivalent: `http_req_looking_up`
- HAR equivalent: `dns`
- Returns 0 if DNS cache hit or connection was reused.
- Only available for AioHttpTraceData.

---

### HTTP Connecting

**Type:** [Record Metric](#record-metrics)

Time spent establishing TCP connection to the remote host. For HTTPS requests, this includes both TCP connection establishment and TLS handshake time (combined measurement from aiohttp).

**Formula:**
```python
http_req_connecting = tcp_connect_end_perf_ns - tcp_connect_start_perf_ns
```

**Notes:**
- k6 equivalent: `http_req_connecting`
- HAR equivalent: `connect`
- Returns 0 if connection was reused.
- Only available for AioHttpTraceData.

---

### HTTP Sending

**Type:** [Record Metric](#record-metrics)

Time spent sending data to the remote host. This metric measures the time from when the request started being sent to when the full request (headers + body) was transmitted.

**Formula:**
```python
http_req_sending = request_send_end_perf_ns - request_send_start_perf_ns
```

**Notes:**
- k6 equivalent: `http_req_sending`
- HAR equivalent: `send`

---

### HTTP Waiting (TTFB)

**Type:** [Record Metric](#record-metrics)

Time to First Byte (TTFB) - time waiting for the server to respond. This metric measures the time from when the request was fully sent to when the first byte of the response body was received. This represents server processing time plus network latency.

**Formula:**
```python
http_req_waiting = response_chunks[0][0] - request_send_end_perf_ns
```

**Notes:**
- k6 equivalent: `http_req_waiting` (also known as TTFB)
- HAR equivalent: `wait`
- Note that this is not the same as the time to first token (TTFT), which is the time from request start to the first valid token received. The server may send non-token data first.

---

### HTTP Receiving

**Type:** [Record Metric](#record-metrics)

Time spent receiving response data from the remote host. This metric measures the time from when the first byte of the response was received to when the last byte was received.

**Formula:**
```python
http_req_receiving = response_chunks[-1][0] - response_chunks[0][0]
```

**Notes:**
- k6 equivalent: `http_req_receiving`
- HAR equivalent: `receive`
- Returns 0 if response was a single chunk.

---

### HTTP Duration

**Type:** [Record Metric](#record-metrics)

Time for HTTP request/response exchange, excluding connection overhead. This measures only the request/response exchange time: `sending + waiting + receiving`.

**Formula:**
```python
http_req_duration = response_receive_end_perf_ns - request_send_start_perf_ns
```

**Notes:**
- k6 equivalent: `http_req_duration`
- HAR equivalent: `time`
- EXCLUDES connection overhead (blocked, dns_lookup, connecting).
- For full end-to-end time including connection setup, use `http_req_total`.
- Note: This uses trace-level timestamps for more accurate measurement than application-level request latency.

---

### HTTP Connection Overhead

**Type:** [Record Metric](#record-metrics)

Total connection overhead time (blocked + dns_lookup + connecting). This metric combines all pre-request overhead.

**Formula:**
```python
http_req_connection_overhead = http_req_blocked + http_req_dns_lookup + http_req_connecting
```

**Notes:**
- Useful for identifying total connection establishment costs.
- Returns 0 if connection was reused with no pool wait.
- Only available for AioHttpTraceData.

---

### HTTP Total Time

**Type:** [Record Metric](#record-metrics)

Sum of all HTTP timing phases from connection pool to last chunk received. This is the sum of all 6 timing components: `blocked + dns_lookup + connecting + sending + waiting + receiving`.

**Formula:**
```python
http_req_total = http_req_blocked + http_req_dns_lookup + http_req_connecting + http_req_sending + http_req_waiting + http_req_receiving
```

**Notes:**
- This ensures the math adds up: individual timing metrics sum exactly to this total.
- Only available for AioHttpTraceData (requires connection overhead metrics).

---

### HTTP Data Sent

**Type:** [Record Metric](#record-metrics)

Total bytes sent in the HTTP request (headers + body).

**Formula:**
```python
http_req_data_sent = sum(size for _, size in request_chunks)
```

**Notes:**
- k6 equivalent: `data_sent` (per request)
- Measures total bytes written to the transport layer.

---

### HTTP Data Received

**Type:** [Record Metric](#record-metrics)

Total bytes received in the HTTP response (headers + body).

**Formula:**
```python
http_req_data_received = sum(size for _, size in response_chunks)
```

**Notes:**
- k6 equivalent: `data_received` (per request)
- Measures total bytes read from the transport layer.

---

### HTTP Connection Reused

**Type:** [Record Metric](#record-metrics)

Whether the HTTP connection was reused from the connection pool. Returns 1 if reused, 0 if new connection was established.

**Formula:**
```python
http_req_connection_reused = 1 if connection_reused_perf_ns is not None else 0
```

**Notes:**
- Helps identify connection reuse patterns and keep-alive effectiveness.
- Only available for AioHttpTraceData.

---

# Metric Flags Reference

Metric flags are used to control when and how metrics are computed, displayed, and grouped. Flags can be combined using bitwise operations to create composite behaviors.

## Individual Flags

| Flag | Description | Impact |
|------|-------------|--------|
| <a id="flag-none"></a>`NONE` | No flags set | Metric has default behavior with no special restrictions |
| <a id="flag-streaming-only"></a>`STREAMING_ONLY` | Only computed for streaming responses | Requires Server-Sent Events (SSE) with multiple response chunks; skipped for non-streaming requests |
| <a id="flag-error-only"></a>`ERROR_ONLY` | Only computed for error requests | Tracks error-specific information; computed only for invalid/failed requests |
| <a id="flag-produces-tokens-only"></a>`PRODUCES_TOKENS_ONLY` | Only computed for token-producing endpoints | Requires endpoints that return text/token content; skipped for embeddings and non-generative endpoints |
| <a id="flag-no-console"></a>`NO_CONSOLE` | Not displayed in console output | Metric computed but excluded from terminal display; available in JSON/CSV/JSONL exports and used by other metrics |
| <a id="flag-larger-is-better"></a>`LARGER_IS_BETTER` | Higher values indicate better performance | Used for throughput and count metrics to indicate optimization direction |
| <a id="flag-internal"></a>`INTERNAL` | Internal AIPerf metric | Used for AIPerf system diagnostics; not displayed in console or exported without developer mode |
| <a id="flag-supports-audio-only"></a>`SUPPORTS_AUDIO_ONLY` | Only computed for audio endpoints | Requires audio-capable endpoints; skipped for other endpoint types |
| <a id="flag-supports-image-only"></a>`SUPPORTS_IMAGE_ONLY` | Only computed for image endpoints | Requires image-capable endpoints; skipped for other endpoint types |
| <a id="flag-supports-reasoning"></a>`SUPPORTS_REASONING` | Requires reasoning token support | Only available for models and endpoints that expose reasoning content in separate fields |
| <a id="flag-experimental"></a>`EXPERIMENTAL` | Experimental/unstable metric | May change or be removed in future releases; not displayed in console or exported without developer mode |
| <a id="flag-goodput"></a>`GOODPUT` | Only computed when goodput is enabled | Requires SLO thresholds to be configured (e.g., `--goodput-constraints`); skipped otherwise |
| <a id="flag-no-individual-records"></a>`NO_INDIVIDUAL_RECORDS` | Not exported for individual records | Aggregate metrics not relevant to individual records (e.g., request count, min/max timestamps); excluded from per-record exports |
| <a id="flag-tokenizes-input-only"></a>`TOKENIZES_INPUT_ONLY` | Only computed when endpoint tokenizes input | Requires endpoints that process and tokenize input text; skipped for non-text endpoints |
| <a id="flag-http-trace-only"></a>`HTTP_TRACE_ONLY` | Only computed when HTTP trace data is available | Requires HTTP request tracing to be enabled; provides detailed HTTP lifecycle timing metrics |
| <a id="flag-usage-diff-only"></a>`USAGE_DIFF_ONLY` | Only computed when usage field data is available | Requires API responses to include usage field with token counts for comparison with client-computed values |

## Composite Flags

These flags are combinations of multiple individual flags for convenience:

| Flag | Composition | Description |
|------|-------------|-------------|
| <a id="flag-streaming-tokens-only"></a>`STREAMING_TOKENS_ONLY` | `STREAMING_ONLY` + `PRODUCES_TOKENS_ONLY` | Requires both streaming support and token-producing endpoints |

---
