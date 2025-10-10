<!--
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# AIPerf Metrics Reference

This document provides a comprehensive reference of all metrics available in AIPerf for benchmarking LLM inference performance. Metrics are organized by computation type to help you understand when and how each metric is calculated.

## Table of Contents

- [Understanding Metric Types](#understanding-metric-types)
  - [Record Metrics](#record-metrics)
  - [Aggregate Metrics](#aggregate-metrics)
  - [Derived Metrics](#derived-metrics)
- [Quick Reference](#quick-reference)
- [Detailed Metric Descriptions](#detailed-metric-descriptions)
  - [Streaming Metrics](#streaming-metrics)
    - [Time to First Token (TTFT)](#time-to-first-token-ttft)
    - [Time to Second Token (TTST)](#time-to-second-token-ttst)
    - [Inter Token Latency (ITL)](#inter-token-latency-itl)
    - [Inter Chunk Latency (ICL)](#inter-chunk-latency-icl)
    - [Output Token Throughput Per User](#output-token-throughput-per-user)
  - [Token Based Metrics](#token-based-metrics)
    - [Output Token Count](#output-token-count)
    - [Output Sequence Length (OSL)](#output-sequence-length-osl)
    - [Input Sequence Length (ISL)](#input-sequence-length-isl)
    - [Total Output Tokens](#total-output-tokens)
    - [Total Output Sequence Length](#total-output-sequence-length)
    - [Total Input Sequence Length](#total-input-sequence-length)
    - [Output Token Throughput](#output-token-throughput)
  - [Reasoning Metrics](#reasoning-metrics)
    - [Reasoning Token Count](#reasoning-token-count)
    - [Total Reasoning Tokens](#total-reasoning-tokens)
  - [General Metrics](#general-metrics)
    - [Request Latency](#request-latency)
    - [Request Throughput](#request-throughput)
    - [Request Count](#request-count)
    - [Error Request Count](#error-request-count)
    - [Minimum Request Timestamp](#minimum-request-timestamp)
    - [Maximum Response Timestamp](#maximum-response-timestamp)
    - [Benchmark Duration](#benchmark-duration)
- [Metric Flags Reference](#metric-flags-reference)

---

## Understanding Metric Types

AIPerf computes metrics in three distinct phases during benchmark execution: **Record Metrics**, **Aggregate Metrics**, and **Derived Metrics**.

## Record Metrics

Record Metrics are computed **individually** for **each request** and its **response(s)** during the benchmark run. A single request may have one response (non-streaming) or multiple responses (streaming). These metrics capture **per-request characteristics** such as latency, token counts, and streaming behavior. Record metrics produce **statistical distributions** (min, max, mean, median, p90, p99, etc.) that reveal performance variability across requests.

#### Example Metrics
`request_latency`, `ttft`, `inter_token_latency`, `output_token_count`, `input_sequence_length`

#### Dependencies
Record Metrics can depend on raw request/response data and other Record Metrics from the same request.

#### Example Scenario
`request_latency` measures the time for each individual request from start to final response. If you send 100 requests, you get 100 latency values that form a distribution showing how latency varies across requests.

## Aggregate Metrics

Aggregate Metrics are computed by **tracking** or **accumulating** values across **all requests** in **real-time** during the benchmark. These include counters, min/max timestamps, and other global statistics. Aggregate metrics produce a **single value** representing the entire benchmark run.

#### Example Metrics
`request_count`, `error_request_count`, `min_request_timestamp`, `max_response_timestamp`

#### Dependencies
Aggregate Metrics can depend on raw request/response data, Record Metrics and other Aggregate Metrics.

#### Example Scenario
`request_count` increments by 1 for each successful request. At the end of a benchmark with 100 successful requests, this metric equals 100 (a single value, not a distribution).

## Derived Metrics

Derived Metrics are computed by applying **mathematical formulas** to other metric results, but are **not** computed per-record like Record Metrics. Instead, these metrics depend on one or more **prerequisite metrics** being available first and are calculated either **after the benchmark completes** for final results or in **real-time** across **all current data** for live metrics display. Derived metrics can produce either single values or distributions depending on their dependencies.

#### Example Metrics
`request_throughput`, `output_token_throughput`, `benchmark_duration`

#### Dependencies
Derived Metrics can depend on Record Metrics, Aggregate Metrics, and other Derived Metrics, but do not have
any knowledge of the individual request/response data.

#### Example Scenario
`request_throughput` is computed from `request_count / benchmark_duration_seconds`. This requires both `request_count` and `benchmark_duration` to be available first, then applies a formula to produce a single throughput value (e.g., 10.5 requests/sec).

---

## Quick Reference

For a quick reference of all metrics with their tags, formulas, and units, see the **[Metrics Reference section in the README](../README.md#metrics-reference)**.

The sections below provide detailed descriptions, requirements, and notes for each metric.

---

# Detailed Metric Descriptions

## Streaming Metrics

> [!NOTE]
> All metrics in this section require the `--streaming` flag with a token-producing endpoint and at least one non-empty response chunk.

### Time to First Token (TTFT)

Measures how long it takes to receive the first token (or chunk of tokens) after sending a request. This is critical for user-perceived responsiveness in streaming scenarios, as it represents how quickly the model begins generating output.

**Formula:**
```python
ttft = responses[0].perf_ns - request.start_perf_ns
```

**Notes:**
- Includes network latency, queuing time, prompt processing, and generation of the first token (or chunk of tokens).

---

### Time to Second Token (TTST)

Measures the time gap between the first and second chunk of tokens (SSE messages). This metric helps identify generation startup overhead separate from steady-state streaming throughput.

**Formula:**
```python
ttst = responses[1].perf_ns - responses[0].perf_ns
```

**Notes:**
- Requires at least 2 non-empty response chunks to compute the time between first and second tokens.

---

### Inter Token Latency (ITL)

Measures the average time between consecutive tokens during generation, excluding the initial TTFT overhead. This represents the steady-state token generation rate.

**Formula:**
```python
inter_token_latency = (request_latency - ttft) / (output_sequence_length - 1)
```

**Notes:**
- Requires at least 2 non-empty response chunks and valid `ttft`, `request_latency`, and `output_sequence_length` metrics.

---

### Inter Chunk Latency (ICL)

Captures the time gaps between all consecutive response chunks (SSE messages) in a streaming response, providing a distribution of chunk arrival times rather than a single average. Note that this is different from the ITL metric, which measures the time between consecutive tokens regardless of chunk size.

**Formula:**
```python
inter_chunk_latency = [responses[i].perf_ns - responses[i-1].perf_ns for i in range(1, len(responses))]
```

**Notes:**
- Requires at least 2 response chunks.
- Unlike ITL (which produces a single average), ICL provides the full distribution of inter-chunk times.
- Useful for detecting variability, jitter, or issues in streaming delivery.
- Analyzing ICL distributions can reveal batching behavior, scheduling issues, or network variability.

---

### Output Token Throughput Per User

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

## Token Based Metrics

> [!NOTE]
> All metrics in this section require token-producing endpoints that return text content (chat, completion, etc.). These metrics are not available for embeddings or other non-generative endpoints.

### Output Token Count

The number of output tokens generated for a single request, _excluding reasoning tokens_. This represents the visible output tokens returned to the user across all responses for the request.

**Formula:**
```python
output_token_count = len(tokenizer.encode(content))
```

**Notes:**
- For streaming requests with multiple responses, the responses are joined together and then tokens are counted.
- For models that support reasoning tokens, this metric counts only the non-reasoning output tokens.
- This **will** count tokens inside of the `<think>` tags, if they are present in the `content` field of the response.

---

### Output Sequence Length (OSL)

The total number of completion tokens (output + reasoning) generated for a single request across all its responses. This represents the complete token generation workload for the request.

**Formula:**
```python
output_sequence_length = (output_token_count or 0) + (reasoning_token_count or 0)
```

**Notes:**
- For models that do not support/separate reasoning tokens, OSL equals the output token count.

---

### Input Sequence Length (ISL)

The number of input/prompt tokens for a single request. This represents the size of the input sent to the model.

**Formula:**
```python
input_sequence_length = len(tokenizer.encode(prompt))
```

**Notes:**
- Useful for understanding the relationship between input size and latency/throughput.

---

### Total Output Tokens

The sum of all output tokens (excluding reasoning tokens) generated across all requests. This represents the total visible output token workload.

**Formula:**
```python
total_output_tokens = sum(output_token_count for record in records)
```

**Notes:**
- Aggregates output tokens across all successful requests.
- Useful for capacity planning and cost estimation.

---

### Total Output Sequence Length

The sum of all completion tokens (output + reasoning) generated across all requests. This represents the complete token generation workload.

**Formula:**
```python
total_osl = sum(output_sequence_length for record in records)
```

**Notes:**
- Aggregates the complete token generation workload including both output and reasoning tokens.
- For models without reasoning tokens, this equals Total Output Tokens.

---

### Total Input Sequence Length

The sum of all input/prompt tokens processed across all requests. This represents the total input workload sent to the model.

**Formula:**
```python
total_isl = sum(input_sequence_length for record in records)
```

**Notes:**
- Useful for understanding the input workload, capacity planning, and analyzing the relationship between input size and system performance.

---

### Output Token Throughput

> [!IMPORTANT]
> This metric is computed as a single values across all requests, and it includes the TTFT in the equation, so it is **not** directly comparable to the [Output Token Throughput Per User](#output-token-throughput-per-user) metric.

The aggregate token generation rate across all concurrent requests, measured as total tokens per second. This represents the system's overall token generation capacity.

**Formula:**
```python
output_token_throughput = total_osl / benchmark_duration_seconds
```

**Notes:**
- Measures aggregate throughput across all concurrent requests; represents the overall system token generation rate.
- Higher values indicate better system utilization and capacity.

---

## Reasoning Metrics

> [!NOTE]
> All metrics in this section require models and backends that expose reasoning content in a separate `reasoning_content` field, distinct from the regular `content` field.

### Reasoning Token Count

The number of reasoning tokens generated for a single request. These are tokens used for "thinking" or chain-of-thought reasoning before generating the final output.

**Formula:**
```python
reasoning_token_count = len(tokenizer.encode(reasoning_content))
```

**Notes:**
- Does **not** differentiate `<think>` tags or extract reasoning from within the regular `content` field.

---

### Total Reasoning Tokens

The sum of all reasoning tokens generated across all requests. This represents the total reasoning/thinking workload.

**Formula:**
```python
total_reasoning_tokens = sum(reasoning_token_count for record in records)
```

**Notes:**
- Useful for understanding the reasoning overhead and cost for reasoning-enabled models.

---

## General Metrics

> [!NOTE]
> Metrics in this section are available for all benchmark runs with no special requirements.

### Request Latency

Measures the total end-to-end time from sending a request until receiving the final response. For streaming requests with multiple responses, this measures until the last response is received. This is the complete time experienced by the client for a single request.

**Formula:**
```python
request_latency = responses[-1].perf_ns - start_perf_ns
```

**Notes:**
- Includes all components: network time, queuing, prompt processing, token generation, and response transmission.
- For streaming requests, measures from request start to the final chunk received.

---

### Request Throughput

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

The total number of **successfully completed** requests in the benchmark. This includes all requests that received valid responses, regardless of streaming mode.

**Formula:**
```python
request_count = sum(1 for record if record.valid)
```

---

### Error Request Count

The total number of failed/error requests encountered during the benchmark. This includes network errors, HTTP errors, timeout errors, and other failures.

**Formula:**
```python
error_request_count = sum(1 for record if not record.valid)
```

**Notes:**
- Error rate can be computed as `error_request_count / (request_count + error_request_count)`.

---

### Minimum Request Timestamp

The wall-clock timestamp of the first request sent in the benchmark. This is used to calculate the benchmark duration and represents the start of the benchmark run.

**Formula:**
```python
min_request_timestamp = min(timestamp_ns for record in records)
```

---

### Maximum Response Timestamp

The wall-clock timestamp of the last response received in the benchmark. This is used to calculate the benchmark duration and represents the end of the benchmark run.

**Formula:**
```python
max_response_timestamp = max(timestamp_ns + request_latency for record in records)
```

---

### Benchmark Duration

The total elapsed time from the first request sent to the last response received. This represents the complete wall-clock duration of the benchmark run.

**Formula:**
```python
benchmark_duration = max_response_timestamp - min_request_timestamp
```

**Notes:**
- Uses wall-clock timestamps representing real calendar time.
- Used as the denominator for throughput calculations; represents the effective measurement window.

---

# Metric Flags Reference

Metric flags are used to control when and how metrics are computed, displayed, and grouped. Flags can be combined using bitwise operations to create composite behaviors.

| Flag | Description | Impact |
|------|-------------|--------|
| <a id="flag-none"></a>`NONE` | No flags set | Metric has default behavior with no special restrictions |
| <a id="flag-streaming-only"></a>`STREAMING_ONLY` | Only computed for streaming responses | Requires Server-Sent Events (SSE) with multiple response chunks; skipped for non-streaming requests |
| <a id="flag-error-only"></a>`ERROR_ONLY` | Only computed for error requests | Tracks error-specific information; skipped for successful requests |
| <a id="flag-produces-tokens-only"></a>`PRODUCES_TOKENS_ONLY` | Only computed for token-producing endpoints | Requires endpoints that return text/token content; skipped for embeddings and non-generative endpoints |
| <a id="flag-no-console"></a>`NO_CONSOLE` | Not displayed in console output | Metric computed but excluded from terminal display; available in JSON/CSV exports |
| <a id="flag-larger-is-better"></a>`LARGER_IS_BETTER` | Higher values indicate better performance | Used for throughput and count metrics to indicate optimization direction |
| <a id="flag-supports-reasoning"></a>`SUPPORTS_REASONING` | Requires reasoning token support | Only available for models and endpoints that expose reasoning content in separate fields |
| <a id="flag-streaming-tokens-only"></a>`STREAMING_TOKENS_ONLY` | Combination: `STREAMING_ONLY` + `PRODUCES_TOKENS_ONLY` | Requires both streaming support and token-producing endpoints |

---
