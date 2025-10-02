<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AIPerf

[![PyPI version](https://img.shields.io/pypi/v/AIPerf)](https://pypi.org/project/aiperf/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Codecov](https://codecov.io/gh/ai-dynamo/aiperf/graph/badge.svg)](https://codecov.io/gh/ai-dynamo/aiperf)
[![Discord](https://dcbadge.limes.pink/api/server/D92uqZRjCZ?style=flat)](https://discord.gg/D92uqZRjCZ)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/ai-dynamo/aiperf)


**[Architecture](docs/architecture.md)** |  **[Design Proposals](https://github.com/ai-dynamo/enhancements)** | **[Migrating from Genai-Perf](docs/migrating.md)** | **[CLI Options](docs/cli_options.md)** | **[Metrics Reference](docs/metrics_reference.md)** |


AIPerf is a comprehensive benchmarking tool that measures the performance of generative AI models served by your preferred inference solution.
It provides detailed metrics using a command line display as well as extensive benchmark performance reports.

AIPerf provides multiprocess support out of the box for a single scalable solution.


<!--
======================
Features
======================
-->

## Features

- Scalable via multiprocess support
- Modular design for easy user modification
- Several benchmarking modes:
  - concurrency
  - request-rate
  - request-rate with a maximum concurrency
  - [trace replay](docs/benchmark_modes/trace_replay.md)
- [Public dataset support](docs/benchmark_datasets.md)

</br>

## Tutorials & Advanced Features

### Getting Started
- **[Basic Tutorial](docs/tutorial.md)** - Learn the fundamentals with Dynamo and vLLM examples

### Advanced Benchmarking Features
| Feature | Description | Use Cases |
|---------|-------------|-----------|
| **[Request Cancellation](docs/tutorials/request-cancellation.md)** | Test timeout behavior and service resilience | SLA validation, cancellation modeling |
| **[Trace Benchmarking](docs/tutorials/trace-benchmarking.md)** | Deterministic workload replay with custom datasets | Regression testing, A/B testing |
| **[Fixed Schedule](docs/tutorials/fixed-schedule.md)** | Precise timestamp-based request execution | Traffic replay, temporal analysis, burst testing |
| **[Time-based Benchmarking](docs/tutorials/time-based-benchmarking.md)** | Duration-based testing with grace period control | Stability testing, sustained performance |

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

AIPerf provides comprehensive metrics organized into three categories. For detailed descriptions, requirements, and nuances of each metric, see the **[Complete Metrics Reference](docs/metrics_reference.md)**.

### Record Metrics

Computed individually for each request and its response(s). Record metrics produce statistical distributions (min, max, mean, p50, p90, p99, etc.).

| Metric | Tag | Formula | Unit |
|--------|-----|---------|------|
| [**Request Latency**](docs/metrics_reference.md#request-latency) | `request_latency` | `responses[-1].perf_ns - start_perf_ns` | `ms` |
| [**Time to First Token (TTFT)**](docs/metrics_reference.md#time-to-first-token-ttft) | `ttft` | `responses[0].perf_ns - request.start_perf_ns` | `ms` |
| [**Time to Second Token (TTST)**](docs/metrics_reference.md#time-to-second-token-ttst) | `ttst` | `responses[1].perf_ns - responses[0].perf_ns` | `ms` |
| [**Inter Token Latency (ITL)**](docs/metrics_reference.md#inter-token-latency-itl) | `inter_token_latency` | `(request_latency - ttft) / (output_sequence_length - 1)` | `ms` |
| [**Inter Chunk Latency (ICL)**](docs/metrics_reference.md#inter-chunk-latency-icl) | `inter_chunk_latency` | `[responses[i].perf_ns - responses[i-1].perf_ns for i in range(1, len(responses))]` | `ms` |
| [**Output Token Count**](docs/metrics_reference.md#output-token-count) | `output_token_count` | `output_token_count` | `tokens` |
| [**Reasoning Token Count**](docs/metrics_reference.md#reasoning-token-count) | `reasoning_token_count` | `reasoning_token_count` | `tokens` |
| [**Output Sequence Length (OSL)**](docs/metrics_reference.md#output-sequence-length-osl) | `output_sequence_length` | `(output_token_count or 0) + (reasoning_token_count or 0)` | `tokens` |
| [**Input Sequence Length (ISL)**](docs/metrics_reference.md#input-sequence-length-isl) | `input_sequence_length` | `input_token_count` | `tokens` |
| [**Output Token Throughput Per User**](docs/metrics_reference.md#output-token-throughput-per-user) | `output_token_throughput_per_user` | `1.0 / inter_token_latency_seconds` | `tokens/sec/user` |

### Aggregate Metrics

Computed by tracking values across all requests in real-time. Aggregate metrics produce single scalar values.

| Metric | Tag | Formula | Unit |
|--------|-----|---------|------|
| [**Request Count**](docs/metrics_reference.md#request-count) | `request_count` | `sum(1 for request in valid_requests)` | `requests` |
| [**Error Request Count**](docs/metrics_reference.md#error-request-count) | `error_request_count` | `sum(1 for request in error_requests)` | `requests` |
| [**Minimum Request Timestamp**](docs/metrics_reference.md#minimum-request-timestamp) | `min_request_timestamp` | `min(timestamp_ns for record in records)` | `datetime` |
| [**Maximum Response Timestamp**](docs/metrics_reference.md#maximum-response-timestamp) | `max_response_timestamp` | `max(timestamp_ns + request_latency for record in records)` | `datetime` |

### Derived Metrics

Computed using formulas based on other metrics, but **not** computed per-record. These are calculated either after the benchmark completes for final results or in real-time across all current data for live metrics display.

| Metric | Tag | Formula | Unit |
|--------|-----|---------|------|
| [**Request Throughput**](docs/metrics_reference.md#request-throughput) | `request_throughput` | `request_count / benchmark_duration_seconds` | `requests/sec` |
| [**Output Token Throughput**](docs/metrics_reference.md#output-token-throughput) | `output_token_throughput` | `total_osl / benchmark_duration_seconds` | `tokens/sec` |
| [**Benchmark Duration**](docs/metrics_reference.md#benchmark-duration) | `benchmark_duration` | `max_response_timestamp - min_request_timestamp` | `sec` |
| [**Total Output Tokens**](docs/metrics_reference.md#total-output-tokens) | `total_output_tokens` | `sum(output_token_count for record in records)` | `tokens` |
| [**Total Reasoning Tokens**](docs/metrics_reference.md#total-reasoning-tokens) | `total_reasoning_tokens` | `sum(reasoning_token_count for record in records)` | `tokens` |
| [**Total Output Sequence Length**](docs/metrics_reference.md#total-output-sequence-length) | `total_osl` | `sum(output_sequence_length for record in records)` | `tokens` |
| [**Total Input Sequence Length**](docs/metrics_reference.md#total-input-sequence-length) | `total_isl` | `sum(input_sequence_length for record in records)` | `tokens` |

</br>


## Known Issues

- Output sequence length constraints (`--output-tokens-mean`) cannot be guaranteed unless you pass `ignore_eos` and/or `min_tokens` via `--extra-inputs` to an inference server that supports them.
- Very high concurrency settings (typically >15,000 concurrency) may lead to port exhaustion on some systems, causing connection failures during benchmarking. If encountered, consider adjusting system limits or reducing concurrency.
- Startup errors caused by invalid configuration settings can cause AIPerf to hang indefinitely. If AIPerf appears to freeze during initialization, terminate the process and check configuration settings.
- Dashboard UI may cause corrupted ANSI sequences on macOS or certain terminal environments, making the terminal unusable. Run `reset` command to restore normal terminal functionality, or switch to `--ui simple` for a lightweight progress bar interface.
