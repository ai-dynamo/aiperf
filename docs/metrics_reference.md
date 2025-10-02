<!--
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# AIPerf Metrics Reference

This document provides a comprehensive reference of all metrics available in AIPerf for benchmarking LLM inference performance. Metrics are organized by computation type to help you understand when and how each metric is calculated.

## Understanding Metric Types

AIPerf computes metrics in three distinct phases during benchmark execution:

### Record Metrics
Computed **individually for each request/response(s) pair** during the benchmark run. These metrics capture per-request characteristics such as latency, token counts, and streaming behavior. Record metrics produce **statistical distributions** (min, max, mean, median, p90, p99) that reveal performance variability across requests.

**Examples**: Request Latency, TTFT, Token Counts, Inter-Token Latency

### Aggregate Metrics
Computed by **tracking or accumulating values** across all requests in **real-time** during the benchmark. These include counters, min/max timestamps, and other global statistics. Aggregate metrics produce **single scalar values** representing the entire benchmark run.

**Examples**: Request Count, Error Count, Min/Max Timestamps

### Derived Metrics
Computed **after the benchmark completes** by applying mathematical formulas to other metric results. These metrics depend on one or more prerequisite metrics being available first. Derived metrics can produce either single values or distributions depending on their dependencies.

**Examples**: Request Throughput, Benchmark Duration, Total Token Counts

---

## Record Metrics

| Metric | Tag | Explanation | Formula |
|--------|-----|-------------|---------|
| **Request Latency** | `request_latency` | Measures the total end-to-end time from sending a request until receiving the final response. | `responses[-1].perf_ns - start_perf_ns` |
| **Time to First Token (TTFT)** | `ttft` | Measures how long it takes to receive the first token after sending a request. Critical for user-perceived responsiveness in streaming scenarios. | `responses[0].perf_ns - request.start_perf_ns` |
| **Time to Second Token (TTST)** | `ttst` | Measures the time gap between the first and second tokens. Helps identify generation startup overhead separate from streaming throughput. | `responses[1].perf_ns - responses[0].perf_ns` |
| **Inter Token Latency (ITL)** | `inter_token_latency` | Measures the average time between consecutive tokens during generation, excluding the initial TTFT overhead. | `(request_latency - ttft) / (output_sequence_length - 1)` |
| **Inter Chunk Latency (ICL)** | `inter_chunk_latency` | Captures the time gaps between all consecutive response chunks in a streaming response, providing a distribution of chunk arrival times. | `[responses[i].perf_ns - responses[i-1].perf_ns for i in range(1, len(responses))]` |
| **Output Token Count** | `output_token_count` | The number of output tokens generated for a single request, excluding reasoning tokens. | `output_token_count` |
| **Reasoning Token Count** | `reasoning_token_count` | The number of reasoning tokens generated for a single request (e.g., chain-of-thought tokens in reasoning models). | `reasoning_token_count` |
| **Output Sequence Length (OSL)** | `output_sequence_length` | The total number of completion tokens (output + reasoning) generated for a single request. | `(output_token_count or 0) + (reasoning_token_count or 0)` |
| **Input Sequence Length (ISL)** | `input_sequence_length` | The number of input/prompt tokens for a single request. | `input_token_count` |
| **Output Token Throughput Per User** | `output_token_throughput_per_user` | The token generation rate experienced by an individual user/request, measured as the inverse of inter-token latency. | `1.0 / inter_token_latency_seconds` |

---

## Aggregate Metrics

| Metric | Tag | Explanation | Formula |
|--------|-----|-------------|---------|
| **Request Count** | `request_count` | The total number of successfully completed requests in the benchmark. | `sum(1 for request in valid_requests)` |
| **Error Request Count** | `error_request_count` | The total number of failed/error requests encountered during the benchmark. | `sum(1 for request in error_requests)` |
| **Minimum Request Timestamp** | `min_request_timestamp` | The wall-clock timestamp of the first request sent in the benchmark, used to calculate benchmark duration. | `min(timestamp_ns for record in records)` |
| **Maximum Response Timestamp** | `max_response_timestamp` | The wall-clock timestamp of the last response received in the benchmark, used to calculate benchmark duration. | `max(timestamp_ns + request_latency for record in records)` |

---

## Derived Metrics

| Metric | Tag | Explanation | Formula |
|--------|-----|-------------|---------|
| **Request Throughput** | `request_throughput` | The overall rate of completed requests per second across the entire benchmark. | `request_count / benchmark_duration_seconds` |
| **Output Token Throughput** | `output_token_throughput` | The aggregate token generation rate across all concurrent requests, measured as total tokens per second. | `total_osl / benchmark_duration_seconds` |
| **Benchmark Duration** | `benchmark_duration` | The total elapsed time from the first request sent to the last response received. | `max_response_timestamp - min_request_timestamp` |
| **Total Output Tokens** | `total_output_tokens` | The sum of all output tokens (excluding reasoning tokens) generated across all requests. | `sum(output_token_count for record in records)` |
| **Total Reasoning Tokens** | `total_reasoning_tokens` | The sum of all reasoning tokens generated across all requests. | `sum(reasoning_token_count for record in records)` |
| **Total Output Sequence Length** | `total_osl` | The sum of all completion tokens (output + reasoning) generated across all requests. | `sum(output_sequence_length for record in records)` |
| **Total Input Sequence Length** | `total_isl` | The sum of all input/prompt tokens processed across all requests. | `sum(input_sequence_length for record in records)` |

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
| **Token-producing models** | Models that return `usage` information with input/output token counts | `output_token_count`, `input_sequence_length`, `output_token_throughput` |
| **Streaming responses** | Endpoints that support Server-Sent Events (SSE) | `ttft`, `inter_token_latency`, `inter_chunk_latency` |
| **Reasoning token support** | Models that expose reasoning/thinking token counts separately | `reasoning_token_count`, `total_reasoning_tokens` |

### Metric Flags Reference

| Flag | Description | Impact |
|------|-------------|--------|
| `LARGER_IS_BETTER` | Higher values indicate better performance | Used for throughput and count metrics to indicate optimization direction |
| `PRODUCES_TOKENS_ONLY` | Only computed for token-producing models | Metric skipped if model doesn't provide token count information |
| `STREAMING_TOKENS_ONLY` | Only computed for streaming responses with tokens | Requires both streaming support and token information |
| `STREAMING_ONLY` | Only computed for streaming responses | Requires Server-Sent Events (SSE) support |
| `SUPPORTS_REASONING` | Requires reasoning token support | Only available for models like OpenAI o1 that expose reasoning tokens |
| `NO_CONSOLE` | Not displayed in console output | Metric available in JSON/CSV exports but hidden from terminal display |
| `ERROR_ONLY` | Only computed for error requests | Tracks error-specific information |

