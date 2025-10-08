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


**[Architecture](docs/architecture.md)**| **[Design Proposals](https://github.com/ai-dynamo/enhancements)** | **[Migrating from Genai-Perf](docs/migrating.md)** | **[CLI Options](docs/cli_options.md)**


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
| **[Sequence Distributions](docs/tutorials/sequence-distributions.md)** | Mixed ISL/OSL pairings | Benchmarking mixed use cases |
| **[Goodput](docs/tutorials/goodput.md)** | Goodput Metric | Measure requests per second that satisfy user-provided SLOs | SLO validation, capacity planning, runtime/model comparisons |

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


## Known Issues

- Output sequence length constraints (`--output-tokens-mean`) cannot be guaranteed unless you pass `ignore_eos` and/or `min_tokens` via `--extra-inputs` to an inference server that supports them.
- Very high concurrency settings (typically >15,000 concurrency) may lead to port exhaustion on some systems, causing connection failures during benchmarking. If encountered, consider adjusting system limits or reducing concurrency.
- Startup errors caused by invalid configuration settings can cause AIPerf to hang indefinitely. If AIPerf appears to freeze during initialization, terminate the process and check configuration settings.
- Dashboard UI may cause corrupted ANSI sequences on macOS or certain terminal environments, making the terminal unusable. Run `reset` command to restore normal terminal functionality, or switch to `--ui simple` for a lightweight progress bar interface.
