<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AIPerf

[![PyPI version](https://img.shields.io/pypi/v/AIPerf)](https://pypi.org/project/aiperf/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Codecov](https://codecov.io/gh/ai-dynamo/aiperf/graph/badge.svg)](https://codecov.io/gh/ai-dynamo/aiperf)
[![Discord](https://dcbadge.limes.pink/api/server/D92uqZRjCZ?style=flat)](https://discord.gg/D92uqZRjCZ)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/ai-dynamo/aiperf)


**[Architecture](docs/architecture.md)** | **[Design Proposals](https://github.com/ai-dynamo/enhancements)** | **[Migrating from Genai-Perf](docs/migrating.md)** | **[CLI Options](docs/cli_options.md)** | **[Metrics Reference](docs/metrics_reference.md)**


AIPerf is a comprehensive benchmarking tool that measures the performance of generative AI models served by your preferred inference solution.
It provides detailed metrics using a command line display as well as extensive benchmark performance reports.

AIPerf provides multiprocess support out of the box for a single scalable solution.


<!--
======================
Features
======================
-->

<img width="1724" height="670" alt="AIPerf UI Dashboard" src="https://github.com/user-attachments/assets/7eb40867-b1c1-4ebe-bd57-7619f2154bba" />

## Features

- Scalable via multiprocess support
- Modular design for easy user modification
- **3 UI types**: `--ui-type dashboard` (default in TTY, real-time TUI), `--ui-type simple` (progress bars), `--ui-type none` (default in non-TTY). Automatically detects interactive terminals. See [CLI Options](docs/cli_options.md) or [User Interface](docs/tutorials/ui-types.md) for details.
- Several benchmarking modes:
  - concurrency
  - request-rate
  - [request-rate with a maximum concurrency](docs/tutorials/request-rate-concurrency.md)
  - [trace replay](docs/benchmark_modes/trace_replay.md)
- [Public dataset support](docs/benchmark_datasets.md)

</br>

## Tutorials & Advanced Features

### Getting Started
- **[Basic Tutorial](docs/tutorial.md)** - Learn the fundamentals with Dynamo and vLLM examples
- **[User Interface](docs/tutorials/ui-types.md)** - Choose between dashboard, simple, or none for displaying progress

### Load Control & Timing

| Feature | Description | Use Cases |
|---------|-------------|-----------|
| **[Request Rate with Max Concurrency](docs/tutorials/request-rate-concurrency.md)** | Dual control of request timing and concurrent connection ceiling (Poisson or constant modes) | Testing API rate/concurrency limits, avoiding thundering herd, realistic client simulation |
| **[Arrival Patterns](docs/tutorials/arrival-patterns.md)** | Configure traffic patterns (constant, Poisson, gamma) with tunable burstiness | Realistic traffic simulation, stress testing, vLLM-compatible benchmarks |
| **[Prefill Concurrency](docs/tutorials/prefill-concurrency.md)** | Limit concurrent prefill operations to prevent memory exhaustion with long-context workloads | Long-context benchmarking, OOM prevention, memory-safe stress testing |
| **[Gradual Ramping](docs/tutorials/ramping.md)** | Smooth ramp-up of concurrency and request rate over time | Capacity discovery, avoiding cold-start spikes, server warm-up |
| **[Warmup Phase](docs/tutorials/warmup.md)** | Configure pre-benchmark warmup to eliminate cold-start effects | Accurate measurements, JIT warm-up, cache priming |
| **[User-Centric Timing](docs/tutorials/user-centric-timing.md)** | Per-user rate limiting with precise timing for KV cache benchmarking | KV cache effectiveness, multi-user simulation, cache TTL testing |
| **[Request Cancellation](docs/tutorials/request-cancellation.md)** | Test timeout behavior and service resilience | SLA validation, cancellation modeling |
| **[Multi-URL Load Balancing](docs/tutorials/multi-url-load-balancing.md)** | Distribute requests across multiple inference server instances | Multi-GPU scaling, distributed inference, high-throughput benchmarking |

### Workloads & Data

| Feature | Description | Use Cases |
|---------|-------------|-----------|
| **[Trace Benchmarking](docs/tutorials/trace-benchmarking.md)** | Deterministic workload replay with custom datasets | Regression testing, A/B testing |
| **[Custom Prompt Benchmarking](docs/tutorials/custom-prompt-benchmarking.md)** | Send each prompt from your file exactly as-is, without sampling or generation | Regression testing, A/B testing, debugging specific prompts |
| **[Fixed Schedule](docs/tutorials/fixed-schedule.md)** | Precise timestamp-based request execution | Traffic replay, temporal analysis, burst testing |
| **[Time-based Benchmarking](docs/tutorials/time-based-benchmarking.md)** | Duration-based testing with grace period control | Stability testing, sustained performance |
| **[Sequence Distributions](docs/tutorials/sequence-distributions.md)** | Mixed ISL/OSL pairings | Benchmarking mixed use cases |
| **[Random Number Generation & Reproducibility](docs/reproducibility.md)** | Deterministic dataset generation with `--random-seed` | Debugging, regression testing, controlled experiments |
| **[Template Endpoint](docs/tutorials/template-endpoint.md)** | Benchmark custom APIs with flexible Jinja2 request templates | Custom API formats, rapid prototyping, non-standard endpoints |
| **[SGLang Image Generation](docs/tutorials/sglang-image-generation.md)** | Benchmark image generation APIs using SGLang with FLUX.1-dev model | Image generation testing, text-to-image benchmarking, extracting generated images |

### Analysis & Monitoring

| Feature | Description | Use Cases |
|---------|-------------|-----------|
| **[Timeslice Metrics](docs/tutorials/timeslices.md)** | Split up benchmark into timeslices and calculate metrics for each timeslice | Load pattern impact, detecting warm-up effects, performance degradation analysis |
| **[Goodput](docs/tutorials/goodput.md)** | Throughput of requests meeting user-defined SLOs | SLO validation, capacity planning, runtime/model comparisons |
| **[HTTP Trace Metrics](docs/tutorials/http-trace-metrics.md)** | Detailed HTTP request lifecycle timing (DNS, TCP/TLS, TTFB) following k6 and HAR conventions | Connection debugging, latency breakdown, transport-layer analysis |
| **[Profile Exports](docs/tutorials/working-with-profile-exports.md)** | Parse and analyze `profile_export.jsonl` with Pydantic models, custom metrics, and async processing | Custom analysis, data pipelines, post-processing |
| **[Visualization & Plotting](docs/tutorials/plot.md)** | Generate PNG visualizations with automatic mode detection (single-run analysis or multi-run comparison) | Parameter sweep analysis, performance debugging, model comparison |
| **[GPU Telemetry](docs/tutorials/gpu-telemetry.md)** | Real-time GPU metrics collection via DCGM (power, utilization, memory, temperature, etc) | Performance optimization, resource monitoring, multi-node telemetry |
| **[Server Metrics](docs/server_metrics/server-metrics.md)** | Collect Prometheus-compatible server metrics during benchmarking | Performance optimization, resource monitoring, multi-node telemetry |

### Quick Navigation
```bash
# Basic profiling
aiperf profile --model Qwen/Qwen3-0.6B --url localhost:8000 --endpoint-type chat

# Request timeout testing
aiperf profile --request-timeout-seconds 30.0 [other options...]

# Trace-based benchmarking
aiperf profile --input-file trace.jsonl --custom-dataset-type single_turn [other options...]

# Fixed schedule execution
aiperf profile --input-file schedule.jsonl --fixed-schedule --fixed-schedule-auto-offset [other options...]

# Time-based benchmarking
aiperf profile --benchmark-duration 300.0 --benchmark-grace-period 30.0 [other options...]
```

</br>

## Supported APIs

- OpenAI chat completions
- OpenAI completions
- OpenAI embeddings
- OpenAI audio: request throughput and latency
- OpenAI images: request throughput and latency
- NIM embeddings
- NIM rankings

</br>

<!--
======================
INSTALLATION
======================
-->

## Installation
```
pip install aiperf
```

</br>

<!--
======================
QUICK START
======================
-->

## Quick Start

### Basic Usage

Run a simple benchmark against a model:

```bash
aiperf profile \
  --model your_model_name \
  --url http://localhost:8000 \
  --endpoint-type chat \
  --streaming
```

### Example with Custom Configuration

```bash
aiperf profile \
  --model Qwen/Qwen3-0.6B \
  --url http://localhost:8000 \
  --endpoint-type chat \
  --concurrency 10 \
  --request-count 100 \
  --streaming
```

Example output:
<div align="center">

```
NVIDIA AIPerf | LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━┓
┃                               Metric ┃       avg ┃    min ┃    max ┃    p99 ┃    p90 ┃    p75 ┃   std ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━┩
│             Time to First Token (ms) │     18.26 │  11.22 │ 106.32 │  68.82 │  27.76 │  16.62 │ 12.07 │
│            Time to Second Token (ms) │     11.40 │   0.02 │  85.91 │  34.54 │  12.59 │  11.65 │  7.01 │
│                 Request Latency (ms) │    487.30 │ 267.07 │ 769.57 │ 715.99 │ 580.83 │ 536.17 │ 79.60 │
│             Inter Token Latency (ms) │     11.23 │   8.80 │  13.17 │  12.48 │  11.73 │  11.37 │  0.45 │
│     Output Token Throughput Per User │     89.23 │  75.93 │ 113.60 │ 102.28 │  90.91 │  90.29 │  3.70 │
│                    (tokens/sec/user) │           │        │        │        │        │        │       │
│      Output Sequence Length (tokens) │     42.83 │  24.00 │  65.00 │  64.00 │  52.00 │  47.00 │  7.21 │
│       Input Sequence Length (tokens) │     10.00 │  10.00 │  10.00 │  10.00 │  10.00 │  10.00 │  0.00 │
│ Output Token Throughput (tokens/sec) │ 10,944.03 │    N/A │    N/A │    N/A │    N/A │    N/A │   N/A │
│    Request Throughput (requests/sec) │    255.54 │    N/A │    N/A │    N/A │    N/A │    N/A │   N/A │
│             Request Count (requests) │    711.00 │    N/A │    N/A │    N/A │    N/A │    N/A │   N/A │
└──────────────────────────────────────┴───────────┴────────┴────────┴────────┴────────┴────────┴───────┘
```
</div>



<!--
======================
METRICS REFERENCE
======================
-->

## Metrics Reference

AIPerf provides comprehensive metrics organized into multiple functional categories. For detailed descriptions, requirements, and nuances of each metric, see the **[Complete Metrics Reference](docs/metrics_reference.md)**.

### Streaming Metrics

Metrics specific to streaming requests that measure real-time token generation characteristics. Requires `--streaming` flag.

| Metric | Tag | Formula | Unit |
|--------|-----|---------|------|
| [**Time to First Token (TTFT)**](docs/metrics_reference.md#time-to-first-token-ttft) | `time_to_first_token` | `content_responses[0].perf_ns - request.start_perf_ns` | `ms` |
| [**Time to Second Token (TTST)**](docs/metrics_reference.md#time-to-second-token-ttst) | `time_to_second_token` | `content_responses[1].perf_ns - content_responses[0].perf_ns` | `ms` |
| [**Inter Token Latency (ITL)**](docs/metrics_reference.md#inter-token-latency-itl) | `inter_token_latency` | `(request_latency - time_to_first_token) / (output_sequence_length - 1)` | `ms` |
| [**Inter Chunk Latency (ICL)**](docs/metrics_reference.md#inter-chunk-latency-icl) | `inter_chunk_latency` | `[content_responses[i].perf_ns - content_responses[i-1].perf_ns for i in range(1, len(content_responses))]` | `ms` |
| [**Output Token Throughput Per User**](docs/metrics_reference.md#output-token-throughput-per-user) | `output_token_throughput_per_user` | `1.0 / inter_token_latency_seconds` | `tokens/sec/user` |
| [**Time to First Output Token (TTFO)**](docs/metrics_reference.md#time-to-first-output-token-ttfo) | `time_to_first_output_token` | `first_non_reasoning_token_perf_ns - request.start_perf_ns` | `ms` |
| [**Prefill Throughput Per User**](docs/metrics_reference.md#prefill-throughput-per-user) | `prefill_throughput_per_user` | `input_sequence_length / time_to_first_token_seconds` | `tokens/sec/user` |

### Token Based Metrics

Metrics for token-producing endpoints that track token counts and throughput. Requires text-generating endpoints (chat, completion, etc.).

| Metric | Tag | Formula | Unit |
|--------|-----|---------|------|
| [**Output Token Count**](docs/metrics_reference.md#output-token-count) | `output_token_count` | `len(tokenizer.encode(content, add_special_tokens=False))` | `tokens` |
| [**Output Sequence Length (OSL)**](docs/metrics_reference.md#output-sequence-length-osl) | `output_sequence_length` | `(output_token_count or 0) + (reasoning_token_count or 0)` | `tokens` |
| [**Input Sequence Length (ISL)**](docs/metrics_reference.md#input-sequence-length-isl) | `input_sequence_length` | `len(tokenizer.encode(prompt, add_special_tokens=False))` | `tokens` |
| [**Total Output Tokens**](docs/metrics_reference.md#total-output-tokens) | `total_output_tokens` | `sum(r.output_token_count for r in records if r.valid)` | `tokens` |
| [**Total Output Sequence Length**](docs/metrics_reference.md#total-output-sequence-length) | `total_osl` | `sum(r.output_sequence_length for r in records if r.valid)` | `tokens` |
| [**Total Input Sequence Length**](docs/metrics_reference.md#total-input-sequence-length) | `total_isl` | `sum(r.input_sequence_length for r in records if r.valid)` | `tokens` |
| [**Output Token Throughput**](docs/metrics_reference.md#output-token-throughput) | `output_token_throughput` | `total_osl / benchmark_duration_seconds` | `tokens/sec` |
| [**Total Token Throughput**](docs/metrics_reference.md#total-token-throughput) | `total_token_throughput` | `(total_isl + total_osl) / benchmark_duration_seconds` | `tokens/sec` |

### Image Metrics

Metrics for image processing endpoints. Requires image-capable endpoints.

| Metric | Tag | Formula | Unit |
|--------|-----|---------|------|
| [**Image Throughput**](docs/metrics_reference.md#image-throughput) | `image_throughput` | `num_images / request_latency_seconds` | `images/sec` |
| [**Image Latency**](docs/metrics_reference.md#image-latency) | `image_latency` | `request_latency_ms / num_images` | `ms/image` |

### Reasoning Metrics

Metrics specific to models that support reasoning/thinking tokens. Requires models with separate `reasoning_content` field.

| Metric | Tag | Formula | Unit |
|--------|-----|---------|------|
| [**Reasoning Token Count**](docs/metrics_reference.md#reasoning-token-count) | `reasoning_token_count` | `len(tokenizer.encode(reasoning_content, add_special_tokens=False))` | `tokens` |
| [**Total Reasoning Tokens**](docs/metrics_reference.md#total-reasoning-tokens) | `total_reasoning_tokens` | `sum(r.reasoning_token_count for r in records if r.valid)` | `tokens` |

### Usage Field Metrics

Metrics tracking API-reported token counts from the `usage` field in responses. Useful for comparing client-side vs server-side token counts.

| Metric | Tag | Formula | Unit |
|--------|-----|---------|------|
| [**Usage Prompt Tokens**](docs/metrics_reference.md#usage-prompt-tokens) | `usage_prompt_tokens` | `response.usage.prompt_tokens` | `tokens` |
| [**Usage Completion Tokens**](docs/metrics_reference.md#usage-completion-tokens) | `usage_completion_tokens` | `response.usage.completion_tokens` | `tokens` |
| [**Usage Total Tokens**](docs/metrics_reference.md#usage-total-tokens) | `usage_total_tokens` | `response.usage.total_tokens` | `tokens` |
| [**Usage Reasoning Tokens**](docs/metrics_reference.md#usage-reasoning-tokens) | `usage_reasoning_tokens` | `response.usage.completion_tokens_details.reasoning_tokens` | `tokens` |
| [**Total Usage Prompt Tokens**](docs/metrics_reference.md#total-usage-prompt-tokens) | `total_usage_prompt_tokens` | `sum(r.usage_prompt_tokens for r in records if r.valid)` | `tokens` |
| [**Total Usage Completion Tokens**](docs/metrics_reference.md#total-usage-completion-tokens) | `total_usage_completion_tokens` | `sum(r.usage_completion_tokens for r in records if r.valid)` | `tokens` |
| [**Total Usage Total Tokens**](docs/metrics_reference.md#total-usage-total-tokens) | `total_usage_total_tokens` | `sum(r.usage_total_tokens for r in records if r.valid)` | `tokens` |

### Usage Discrepancy Metrics

Metrics measuring differences between API-reported and client-computed token counts.

| Metric | Tag | Formula | Unit |
|--------|-----|---------|------|
| [**Usage Prompt Tokens Diff %**](docs/metrics_reference.md#usage-prompt-tokens-diff-) | `usage_prompt_tokens_diff_pct` | `abs((usage_prompt_tokens - input_sequence_length) / input_sequence_length) * 100` | `%` |
| [**Usage Completion Tokens Diff %**](docs/metrics_reference.md#usage-completion-tokens-diff-) | `usage_completion_tokens_diff_pct` | `abs((usage_completion_tokens - output_sequence_length) / output_sequence_length) * 100` | `%` |
| [**Usage Reasoning Tokens Diff %**](docs/metrics_reference.md#usage-reasoning-tokens-diff-) | `usage_reasoning_tokens_diff_pct` | `abs((usage_reasoning_tokens - reasoning_token_count) / reasoning_token_count) * 100` | `%` |
| [**Usage Discrepancy Count**](docs/metrics_reference.md#usage-discrepancy-count) | `usage_discrepancy_count` | `sum(1 for r in records if r.any_diff > threshold)` | `requests` |

### OSL Mismatch Metrics

Metrics measuring differences between requested `max_tokens` and actual output length. Negative diff = stopped early (EOS), positive diff = over-generated.

| Metric | Tag | Formula | Unit |
|--------|-----|---------|------|
| [**OSL Mismatch Diff %**](docs/metrics_reference.md#osl-mismatch-diff-) | `osl_mismatch_diff_pct` | `((actual_osl - requested_osl) / requested_osl) * 100` | `%` |
| [**OSL Mismatch Count**](docs/metrics_reference.md#osl-mismatch-count) | `osl_mismatch_count` | `sum(1 for r if abs(actual - requested) > min(requested * pct / 100, token_cap))` | `requests` |

### Goodput Metrics

Metrics measuring throughput of requests meeting user-defined Service Level Objectives (SLOs).

| Metric | Tag | Formula | Unit |
|--------|-----|---------|------|
| [**Good Request Count**](docs/metrics_reference.md#good-request-count) | `good_request_count` | `sum(1 for r in records if r.all_slos_met)` | `requests` |
| [**Goodput**](docs/metrics_reference.md#goodput) | `goodput` | `good_request_count / benchmark_duration_seconds` | `requests/sec` |

### Error Metrics

Metrics computed for failed/error requests.

| Metric | Tag | Formula | Unit |
|--------|-----|---------|------|
| [**Error Input Sequence Length**](docs/metrics_reference.md#error-input-sequence-length) | `error_isl` | `input_sequence_length` (for error requests) | `tokens` |
| [**Total Error Input Sequence Length**](docs/metrics_reference.md#total-error-input-sequence-length) | `total_error_isl` | `sum(r.input_sequence_length for r in records if not r.valid)` | `tokens` |
| [**Error Request Count**](docs/metrics_reference.md#error-request-count) | `error_request_count` | `sum(1 for r in records if not r.valid)` | `requests` |

### General Metrics

Metrics available for all benchmark runs with no special requirements.

| Metric | Tag | Formula | Unit |
|--------|-----|---------|------|
| [**Request Latency**](docs/metrics_reference.md#request-latency) | `request_latency` | `content_responses[-1].perf_ns - request.start_perf_ns` | `ms` |
| [**Request Throughput**](docs/metrics_reference.md#request-throughput) | `request_throughput` | `request_count / benchmark_duration_seconds` | `requests/sec` |
| [**Request Count**](docs/metrics_reference.md#request-count) | `request_count` | `sum(1 for r in records if r.valid)` | `requests` |
| [**Minimum Request Timestamp**](docs/metrics_reference.md#minimum-request-timestamp) | `min_request_timestamp` | `min(r.timestamp_ns for r in records)` | `datetime` |
| [**Maximum Response Timestamp**](docs/metrics_reference.md#maximum-response-timestamp) | `max_response_timestamp` | `max(r.timestamp_ns + r.request_latency for r in records)` | `datetime` |
| [**Benchmark Duration**](docs/metrics_reference.md#benchmark-duration) | `benchmark_duration` | `max_response_timestamp - min_request_timestamp` | `sec` |

### HTTP Trace Metrics

Low-level HTTP timing metrics following k6 and HAR conventions. Requires HTTP trace data collection enabled.

| Metric | Tag | Formula | Unit |
|--------|-----|---------|------|
| [**HTTP Request Blocked**](docs/metrics_reference.md#http-request-blocked) | `http_req_blocked` | `connection_pool_wait_end_perf_ns - connection_pool_wait_start_perf_ns` | `ms` |
| [**HTTP Request DNS Lookup**](docs/metrics_reference.md#http-request-dns-lookup) | `http_req_dns_lookup` | `dns_lookup_end_perf_ns - dns_lookup_start_perf_ns` | `ms` |
| [**HTTP Request Connecting**](docs/metrics_reference.md#http-request-connecting) | `http_req_connecting` | `tcp_connect_end_perf_ns - tcp_connect_start_perf_ns` | `ms` |
| [**HTTP Request Sending**](docs/metrics_reference.md#http-request-sending) | `http_req_sending` | `request_send_end_perf_ns - request_send_start_perf_ns` | `ms` |
| [**HTTP Request Waiting**](docs/metrics_reference.md#http-request-waiting) | `http_req_waiting` | `response_chunks[0][0] - request_send_end_perf_ns` | `ms` |
| [**HTTP Request Receiving**](docs/metrics_reference.md#http-request-receiving) | `http_req_receiving` | `response_chunks[-1][0] - response_chunks[0][0]` | `ms` |
| [**HTTP Request Duration**](docs/metrics_reference.md#http-request-duration) | `http_req_duration` | `response_receive_end_perf_ns - request_send_start_perf_ns` | `ms` |
| [**HTTP Request Connection Overhead**](docs/metrics_reference.md#http-request-connection-overhead) | `http_req_connection_overhead` | `http_req_blocked + http_req_dns_lookup + http_req_connecting` | `ms` |
| [**HTTP Request Total**](docs/metrics_reference.md#http-request-total) | `http_req_total` | `http_req_blocked + http_req_dns_lookup + http_req_connecting + http_req_sending + http_req_waiting + http_req_receiving` | `ms` |
| [**HTTP Request Data Sent**](docs/metrics_reference.md#http-request-data-sent) | `http_req_data_sent` | `sum(size for _, size in request_chunks)` | `bytes` |
| [**HTTP Request Data Received**](docs/metrics_reference.md#http-request-data-received) | `http_req_data_received` | `sum(size for _, size in response_chunks)` | `bytes` |
| [**HTTP Request Connection Reused**](docs/metrics_reference.md#http-request-connection-reused) | `http_req_connection_reused` | `1 if connection_reused_perf_ns is not None else 0` | `boolean` |

</br>


## Known Issues

- Output sequence length constraints (`--output-tokens-mean`) cannot be guaranteed unless you pass `ignore_eos` and/or `min_tokens` via `--extra-inputs` to an inference server that supports them.
- Very high concurrency settings (typically >15,000 concurrency) may lead to port exhaustion on some systems, causing connection failures during benchmarking. If encountered, consider adjusting system limits or reducing concurrency.
- Startup errors caused by invalid configuration settings can cause AIPerf to hang indefinitely. If AIPerf appears to freeze during initialization, terminate the process and check configuration settings.

