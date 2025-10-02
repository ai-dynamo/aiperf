<!--
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# AIPerf Metrics Reference

This document provides a comprehensive reference of all metrics available in AIPerf for benchmarking LLM inference performance. Metrics are organized by computation type to help you understand when and how each metric is calculated.

## Understanding Metric Types

AIPerf computes metrics in three distinct phases during benchmark execution:

**Record Metrics** are computed individually for each request and its response(s) during the benchmark run. A single request may have one response (non-streaming) or multiple responses (streaming). These metrics capture per-request characteristics such as latency, token counts, and streaming behavior. Record metrics produce statistical distributions (min, max, mean, median, p90, p99) that reveal performance variability across requests.

**Aggregate Metrics** are computed by tracking or accumulating values across all requests in real-time during the benchmark. These include counters, min/max timestamps, and other global statistics. Aggregate metrics produce single scalar values representing the entire benchmark run.

**Derived Metrics** are computed by applying mathematical formulas to other metric results, but are **not** computed per-record like Record Metrics. Instead, these metrics depend on one or more prerequisite metrics being available first and are calculated either after the benchmark completes for final results or in real-time across all current data for live metrics display. Derived metrics can produce either single values or distributions depending on their dependencies.

---

## Record Metrics

| Metric | Tag | Formula | Unit |
|--------|-----|---------|------|
| [**Request Latency**](#request-latency) | `request_latency` | `responses[-1].perf_ns - start_perf_ns` | `ms` |
| [**Time to First Token (TTFT)**](#time-to-first-token-ttft) | `ttft` | `responses[0].perf_ns - request.start_perf_ns` | `ms` |
| [**Time to Second Token (TTST)**](#time-to-second-token-ttst) | `ttst` | `responses[1].perf_ns - responses[0].perf_ns` | `ms` |
| [**Inter Token Latency (ITL)**](#inter-token-latency-itl) | `inter_token_latency` | `(request_latency - ttft) / (output_sequence_length - 1)` | `ms` |
| [**Inter Chunk Latency (ICL)**](#inter-chunk-latency-icl) | `inter_chunk_latency` | `[responses[i].perf_ns - responses[i-1].perf_ns for i in range(1, len(responses))]` | `ms` |
| [**Output Token Count**](#output-token-count) | `output_token_count` | `output_token_count` | `tokens` |
| [**Reasoning Token Count**](#reasoning-token-count) | `reasoning_token_count` | `reasoning_token_count` | `tokens` |
| [**Output Sequence Length (OSL)**](#output-sequence-length-osl) | `output_sequence_length` | `(output_token_count or 0) + (reasoning_token_count or 0)` | `tokens` |
| [**Input Sequence Length (ISL)**](#input-sequence-length-isl) | `input_sequence_length` | `input_token_count` | `tokens` |
| [**Output Token Throughput Per User**](#output-token-throughput-per-user) | `output_token_throughput_per_user` | `1.0 / inter_token_latency_seconds` | `tokens/sec/user` |

---

## Aggregate Metrics

| Metric | Tag | Formula | Unit |
|--------|-----|---------|------|
| [**Request Count**](#request-count) | `request_count` | `sum(1 for request in valid_requests)` | `requests` |
| [**Error Request Count**](#error-request-count) | `error_request_count` | `sum(1 for request in error_requests)` | `requests` |
| [**Minimum Request Timestamp**](#minimum-request-timestamp) | `min_request_timestamp` | `min(timestamp_ns for record in records)` | `datetime` |
| [**Maximum Response Timestamp**](#maximum-response-timestamp) | `max_response_timestamp` | `max(timestamp_ns + request_latency for record in records)` | `datetime` |

---

## Derived Metrics

| Metric | Tag | Formula | Unit |
|--------|-----|---------|------|
| [**Request Throughput**](#request-throughput) | `request_throughput` | `request_count / benchmark_duration_seconds` | `requests/sec` |
| [**Output Token Throughput**](#output-token-throughput) | `output_token_throughput` | `total_osl / benchmark_duration_seconds` | `tokens/sec` |
| [**Benchmark Duration**](#benchmark-duration) | `benchmark_duration` | `max_response_timestamp - min_request_timestamp` | `sec` |
| [**Total Output Tokens**](#total-output-tokens) | `total_output_tokens` | `sum(output_token_count for record in records)` | `tokens` |
| [**Total Reasoning Tokens**](#total-reasoning-tokens) | `total_reasoning_tokens` | `sum(reasoning_token_count for record in records)` | `tokens` |
| [**Total Output Sequence Length**](#total-output-sequence-length) | `total_osl` | `sum(output_sequence_length for record in records)` | `tokens` |
| [**Total Input Sequence Length**](#total-input-sequence-length) | `total_isl` | `sum(input_sequence_length for record in records)` | `tokens` |

---

## Detailed Metric Descriptions

### Request Latency

**Type:** Record Metric

**Description:** Measures the total end-to-end time from sending a request until receiving the final response. For streaming requests with multiple responses, this measures until the last response is received. This is the complete wall-clock time experienced by the client for a single request.

**Requirements:**
- Available for all request types (streaming and non-streaming)
- No special requirements

**Notes:** Request latency includes all components: network time, queuing, prompt processing, token generation, and response transmission. For streaming requests, it measures from request start to the final chunk received.

---

### Time to First Token (TTFT)

**Type:** Record Metric

**Description:** Measures how long it takes to receive the first token after sending a request. This is critical for user-perceived responsiveness in streaming scenarios, as it represents how quickly the model begins generating output.

**Requirements:**
- Streaming responses with Server-Sent Events (SSE)
- At least 1 response chunk

**Notes:** TTFT includes network latency, queuing time, prompt processing, and generation of the first token. The metric is skipped for non-streaming endpoints. TTFT is a key indicator of interactive performance and perceived latency for end users.

---

### Time to Second Token (TTST)

**Type:** Record Metric

**Description:** Measures the time gap between the first and second tokens. This metric helps identify generation startup overhead separate from steady-state streaming throughput.

**Requirements:**
- Streaming responses with Server-Sent Events (SSE)
- At least 2 response chunks (tokens)

**Notes:** Records with fewer than 2 tokens will skip this metric. TTST is useful for diagnosing issues in the token generation pipeline that may not be apparent from TTFT alone. A high TTST relative to subsequent inter-token latencies may indicate startup inefficiencies.

---

### Inter Token Latency (ITL)

**Type:** Record Metric

**Description:** Measures the average time between consecutive tokens during generation, excluding the initial TTFT overhead. This represents the steady-state token generation rate.

**Requirements:**
- Streaming responses with Server-Sent Events (SSE)
- At least 2 tokens in the output sequence
- Valid `ttft`, `request_latency`, and `output_sequence_length` metrics

**Formula Details:** ITL computes the average time between tokens by dividing the remaining latency (after TTFT) by the number of token intervals:
```
ITL = (request_latency - ttft) / (output_sequence_length - 1)
```

**Notes:** Records with fewer than 2 tokens will skip this metric. ITL is a critical metric for understanding streaming performance and predicting generation times for different output lengths.

---

### Inter Chunk Latency (ICL)

**Type:** Record Metric

**Description:** Captures the time gaps between all consecutive response chunks in a streaming response, providing a distribution of chunk arrival times rather than a single average.

**Requirements:**
- Streaming responses with Server-Sent Events (SSE)
- At least 2 response chunks

**Formula Details:** ICL produces an array of latencies:
```
ICL = [responses[i].perf_ns - responses[i-1].perf_ns for i in range(1, len(responses))]
```

**Notes:** Unlike ITL (which produces a single average), ICL provides the full distribution of inter-chunk times. This is useful for detecting variability, jitter, or issues in streaming delivery. Analyzing ICL distributions can reveal batching behavior, scheduling issues, or network variability.

---

### Output Token Count

**Type:** Record Metric

**Description:** The number of output tokens generated for a single request, excluding reasoning tokens. This represents the visible output tokens returned to the user across all responses for the request.

**Requirements:**
- Token-producing endpoints that return actual token content (text)
- Excludes embeddings and other non-generative endpoints

**Notes:** AIPerf counts tokens from the returned content using a tokenizer. For streaming requests with multiple responses, tokens are counted across all response chunks. For models that support reasoning tokens, this metric counts only the non-reasoning output tokens.

---

### Reasoning Token Count

**Type:** Record Metric

**Description:** The number of reasoning tokens generated for a single request. These are tokens used for "thinking" or chain-of-thought reasoning before generating the final output.

**Requirements:**
- Models/backends that support reasoning output (e.g., OpenAI o1, o1-mini, o1-preview)
- The backend must separate reasoning content into a `reasoning_content` field, distinct from the regular `content` field in the response(s)

**Notes:** AIPerf counts tokens from the `reasoning_content` field using a tokenizer, just like other token metrics. The metric does NOT differentiate `<think>` tags or extract reasoning from within the regular `content` field. The backend must provide reasoning as a separate field in the response. Standard models/backends that don't expose reasoning content separately will skip this metric.

---

### Output Sequence Length (OSL)

**Type:** Record Metric

**Description:** The total number of completion tokens (output + reasoning) generated for a single request across all its responses. This represents the complete token generation workload for the request.

**Requirements:**
- Token-producing endpoints that return text content
- Excludes embeddings and other non-generative endpoints

**Formula Details:**
```
OSL = (output_token_count or 0) + (reasoning_token_count or 0)
```

**Notes:** AIPerf counts tokens from the generated text content across all responses. If no token content is available (e.g., embeddings endpoints), this metric is skipped. OSL represents the total completion tokens generated, sometimes called "completion token count" in other tools. For models without reasoning tokens, OSL equals the output token count.

---

### Input Sequence Length (ISL)

**Type:** Record Metric

**Description:** The number of input/prompt tokens for a single request. This represents the size of the input sent to the model.

**Requirements:**
- Token-producing endpoints (chat, completion, etc.)
- AIPerf tokenizes the input prompt to compute the count

**Notes:** ISL represents the number of tokens in the input prompt sent to the model. AIPerf computes this by tokenizing the input using the appropriate tokenizer for the model. This metric is useful for understanding the relationship between input size and latency/throughput.

---

### Output Token Throughput Per User

**Type:** Record Metric

**Description:** The token generation rate experienced by an individual user/request, measured as the inverse of inter-token latency. This represents single-request streaming performance.

**Requirements:**
- Streaming responses with Server-Sent Events (SSE)
- Valid `inter_token_latency` metric

**Formula Details:**
```
Output Token Throughput Per User = 1.0 / inter_token_latency_seconds
```

**Notes:** This metric computes the inverse of ITL to show tokens per second from an individual user's perspective. It differs from Output Token Throughput (aggregate across all concurrent requests) by focusing on single-request experience. This is useful for understanding the user experience independent of concurrency effects.

---

### Request Count

**Type:** Aggregate Metric

**Description:** The total number of successfully completed requests in the benchmark. This includes all requests that received valid responses, regardless of streaming mode.

**Requirements:**
- No special requirements

**Notes:** This is a fundamental metric for calculating throughput and success rates. Requests that encounter errors are tracked separately in Error Request Count.

---

### Error Request Count

**Type:** Aggregate Metric

**Description:** The total number of failed/error requests encountered during the benchmark. This includes network errors, HTTP errors, timeout errors, and other failures.

**Requirements:**
- No special requirements

**Notes:** Error requests are tracked separately from successful requests. The error rate can be computed as `error_request_count / (request_count + error_request_count)`.

---

### Minimum Request Timestamp

**Type:** Aggregate Metric

**Description:** The wall-clock timestamp of the first request sent in the benchmark. This is used to calculate the benchmark duration and represents the start of the benchmark run.

**Requirements:**
- No special requirements

**Notes:** This uses wall-clock timestamps (not performance counters), representing real calendar time. Useful for correlating benchmark results with external system monitoring and logs.

---

### Maximum Response Timestamp

**Type:** Aggregate Metric

**Description:** The wall-clock timestamp of the last response received in the benchmark. This is used to calculate the benchmark duration and represents the end of the benchmark run.

**Requirements:**
- Valid `request_latency` for at least one request

**Formula Details:**
```
Maximum Response Timestamp = max(timestamp_ns + request_latency for record in records)
```

**Notes:** This uses wall-clock timestamps (not performance counters), representing real calendar time. Combined with Minimum Request Timestamp, this defines the total benchmark duration.

---

### Request Throughput

**Type:** Derived Metric

**Description:** The overall rate of completed requests per second across the entire benchmark. This represents the system's ability to process requests under the given concurrency and load.

**Requirements:**
- Valid `request_count` metric
- Valid `benchmark_duration` metric

**Formula Details:**
```
Request Throughput = request_count / benchmark_duration_seconds
```

**Notes:** This metric captures the aggregate request processing rate. Higher values indicate better system throughput. Request throughput is affected by concurrency level, request complexity, and system capacity.

---

### Output Token Throughput

**Type:** Derived Metric

**Description:** The aggregate token generation rate across all concurrent requests, measured as total tokens per second. This represents the system's overall token generation capacity.

**Requirements:**
- Token-producing endpoints that generate text content
- Valid `total_osl` and `benchmark_duration` metrics

**Formula Details:**
```
Output Token Throughput = total_osl / benchmark_duration_seconds
```

**Notes:** This metric measures aggregate throughput across all concurrent requests and represents the overall system token generation rate. Not applicable to embeddings or other non-generative endpoints. Higher values indicate better system utilization and capacity.

---

### Benchmark Duration

**Type:** Derived Metric

**Description:** The total elapsed time from the first request sent to the last response received. This represents the complete wall-clock duration of the benchmark run.

**Requirements:**
- Valid `min_request_timestamp` metric
- Valid `max_response_timestamp` metric

**Formula Details:**
```
Benchmark Duration = max_response_timestamp - min_request_timestamp
```

**Notes:** Uses wall-clock timestamps representing real calendar time. This is the denominator for throughput calculations and represents the effective measurement window.

---

### Total Output Tokens

**Type:** Derived Metric

**Description:** The sum of all output tokens (excluding reasoning tokens) generated across all requests. This represents the total visible output token workload.

**Requirements:**
- Token-producing endpoints that return text content
- Valid `output_token_count` for processed records

**Formula Details:**
```
Total Output Tokens = sum(output_token_count for record in records)
```

**Notes:** AIPerf counts tokens from the returned content using a tokenizer. This metric aggregates output tokens across all successful requests and is useful for capacity planning and cost estimation.

---

### Total Reasoning Tokens

**Type:** Derived Metric

**Description:** The sum of all reasoning tokens generated across all requests. This represents the total reasoning/thinking workload.

**Requirements:**
- Models/backends that support reasoning output
- Backend must provide `reasoning_content` as a separate field
- Valid `reasoning_token_count` for processed records

**Formula Details:**
```
Total Reasoning Tokens = sum(reasoning_token_count for record in records)
```

**Notes:** AIPerf counts tokens from the `reasoning_content` field. This metric is only available for models like OpenAI o1 that expose reasoning tokens separately. Useful for understanding the reasoning overhead and cost for reasoning-enabled models.

---

### Total Output Sequence Length

**Type:** Derived Metric

**Description:** The sum of all completion tokens (output + reasoning) generated across all requests. This represents the complete token generation workload.

**Requirements:**
- Token-producing endpoints that return text content
- Valid `output_sequence_length` for processed records

**Formula Details:**
```
Total Output Sequence Length = sum(output_sequence_length for record in records)
```

**Notes:** This aggregates the complete token generation workload including both output and reasoning tokens. For models without reasoning tokens, this equals Total Output Tokens. This is the numerator for Output Token Throughput calculations.

---

### Total Input Sequence Length

**Type:** Derived Metric

**Description:** The sum of all input/prompt tokens processed across all requests. This represents the total input workload sent to the model.

**Requirements:**
- Token-producing endpoints
- Valid `input_sequence_length` for processed records

**Formula Details:**
```
Total Input Sequence Length = sum(input_sequence_length for record in records)
```

**Notes:** AIPerf tokenizes input prompts to compute token counts. This metric is useful for understanding the input workload, capacity planning, and analyzing the relationship between input size and system performance.

---

## Reference Tables

### Metric Summary

| Type | Computation | Output |
|------|-------------|--------|
| **Record** | Per-request during benchmark | Statistical distributions (min, max, mean, p50, p90, p99) |
| **Aggregate** | Real-time accumulation across all requests | Single scalar values |
| **Derived** | Post-benchmark from other metrics | Single values or distributions |

### Time Units

| Aspect | Details |
|--------|---------|
| **Internal Storage** | Nanoseconds (`perf_ns`) for maximum precision |
| **Display Format** | Milliseconds (ms) or Seconds (s) for readability |
| **Conversion** | Automatic based on metric `display_unit` setting |

### Model Requirements

| Requirement | Description | Example Metrics |
|-------------|-------------|-----------------|
| **Token-producing endpoints** | Endpoints that return token content (text/tokens) in responses that can be counted; excludes embeddings and other non-generative endpoints | `output_token_count`, `input_sequence_length`, `output_token_throughput` |
| **Streaming responses** | Endpoints that support Server-Sent Events (SSE) returning multiple response chunks | `ttft`, `inter_token_latency`, `inter_chunk_latency` |
| **Reasoning token support** | Models/backends that expose reasoning content in a separate `reasoning_content` field in responses (not embedded in `<think>` tags) | `reasoning_token_count`, `total_reasoning_tokens` |

### Metric Flags Reference

| Flag | Description | Impact |
|------|-------------|--------|
| `LARGER_IS_BETTER` | Higher values indicate better performance | Used for throughput and count metrics to indicate optimization direction |
| `PRODUCES_TOKENS_ONLY` | Only computed for token-producing endpoints | Metric skipped for non-generative endpoints like embeddings |
| `STREAMING_TOKENS_ONLY` | Only computed for streaming responses with tokens | Requires both streaming support and token information |
| `STREAMING_ONLY` | Only computed for streaming responses | Requires Server-Sent Events (SSE) support |
| `SUPPORTS_REASONING` | Requires reasoning token support | Only available for models like OpenAI o1 that expose reasoning tokens |
| `NO_CONSOLE` | Not displayed in console output | Metric available in JSON/CSV exports but hidden from terminal display |
| `ERROR_ONLY` | Only computed for error requests | Tracks error-specific information |

