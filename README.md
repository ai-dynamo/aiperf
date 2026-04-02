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

AIPerf is a comprehensive benchmarking tool that measures the performance of generative AI models served by your preferred inference solution. It provides detailed metrics using a command line display as well as extensive benchmark performance reports.

<img width="1724" height="670" alt="AIPerf UI Dashboard" src="https://github.com/user-attachments/assets/7eb40867-b1c1-4ebe-bd57-7619f2154bba" />

## Quick Start

This quick start guide leverages [Ollama](https://ollama.com/) via
 [Docker Desktop](https://www.docker.com/products/docker-desktop/).

### Setting up a Local Server

In order to set up an Ollama server, run `granite4:350m` using the following commands:

```bash
docker run -d \
  --name ollama \
  -p 11434:11434 \
  -v ollama-data:/root/.ollama \
  ollama/ollama:latest
docker exec -it ollama ollama pull granite4:350m
```

### Basic Usage

Create a virtual environment and install AIPerf:

```bash
python3 -m venv venv
source venv/bin/activate
pip install aiperf
```

To run a simple benchmark against your Ollama server:

```bash
aiperf profile \
  --model "granite4:350m" \
  --streaming \
  --endpoint-type chat \
  --tokenizer ibm-granite/granite-4.0-micro \
  --url http://localhost:11434
```


### Example with Custom Configuration

```bash
aiperf profile \
  --model "granite4:350m" \
  --streaming \
  --endpoint-type chat \
  --tokenizer ibm-granite/granite-4.0-micro \
  --url http://localhost:11434
  --concurrency 5 \
  --request-count 10
```

Example output:


**NOTE:** The example performance is reflective of a CPU-only run and does not represent an official benchmark.

```bash
                                               NVIDIA AIPerf | LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━┓
┃                               Metric ┃       avg ┃      min ┃       max ┃       p99 ┃       p90 ┃       p50 ┃      std ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━┩
│             Time to First Token (ms) │  7,463.28 │ 7,125.81 │  9,484.24 │  9,295.48 │  7,596.62 │  7,240.23 │   677.23 │
│            Time to Second Token (ms) │     68.73 │    32.01 │    102.86 │    102.55 │     99.80 │     67.37 │    24.95 │
│      Time to First Output Token (ms) │  7,463.28 │ 7,125.81 │  9,484.24 │  9,295.48 │  7,596.62 │  7,240.23 │   677.23 │
│                 Request Latency (ms) │ 13,829.40 │ 9,029.36 │ 27,905.46 │ 27,237.77 │ 21,228.48 │ 11,338.31 │ 5,614.32 │
│             Inter Token Latency (ms) │     65.31 │    53.06 │     81.31 │     81.24 │     80.64 │     63.79 │     9.09 │
│     Output Token Throughput Per User │     15.60 │    12.30 │     18.85 │     18.77 │     18.08 │     15.68 │     2.05 │
│                    (tokens/sec/user) │           │          │           │           │           │           │          │
│      Output Sequence Length (tokens) │     95.20 │    29.00 │    295.00 │    283.12 │    176.20 │     63.00 │    77.08 │
│       Input Sequence Length (tokens) │    550.00 │   550.00 │    550.00 │    550.00 │    550.00 │    550.00 │     0.00 │
│ Output Token Throughput (tokens/sec) │      6.85 │      N/A │       N/A │       N/A │       N/A │       N/A │      N/A │
│    Request Throughput (requests/sec) │      0.07 │      N/A │       N/A │       N/A │       N/A │       N/A │      N/A │
│             Request Count (requests) │     10.00 │      N/A │       N/A │       N/A │       N/A │       N/A │      N/A │
└──────────────────────────────────────┴───────────┴──────────┴───────────┴───────────┴───────────┴───────────┴──────────┘

CLI Command: aiperf profile --model 'granite4:350m' --streaming --endpoint-type 'chat' --tokenizer 'ibm-granite/granite-4.0-micro' --url 'http://localhost:11434'
Benchmark Duration: 138.89 sec
CSV Export: /home/user/aiperf/artifacts/granite4:350m-openai-chat-concurrency1/profile_export_aiperf.csv
JSON Export: /home/user/Code/aiperf/artifacts/granite4:350m-openai-chat-concurrency1/profile_export_aiperf.json
Log File: /home/user/Code/aiperf/artifacts/granite4:350m-openai-chat-concurrency1/logs/aiperf.log
```

## Features

- Scalable multiprocess architecture with 9 services communicating via ZMQ
- 3 UI modes: `dashboard` (real-time TUI), `simple` (progress bars), `none` (headless)
- Multiple benchmarking modes: concurrency, request-rate, [request-rate with max concurrency](docs/tutorials/request-rate-concurrency.md), [trace replay](docs/benchmark-modes/trace-replay.md)
- Extensible plugin system for endpoints, datasets, transports, and metrics
- [Public dataset support](docs/benchmark-datasets.md) including ShareGPT and custom formats

## Supported APIs

- OpenAI chat completions, completions, embeddings, audio, images
- NIM embeddings, rankings

## Tutorials and Feature Guides

### Getting Started
- [Basic Tutorial](docs/tutorial.md) - Profile Qwen3-0.6B with vLLM
- [Comprehensive Benchmarking Guide](docs/comprehensive-llm-benchmarking.md) - 5 real-world use cases
- [User Interface](docs/tutorials/ui-types.md) - Dashboard, simple, or headless
- [Hugging Face TGI](docs/tutorials/huggingface-tgi.md) - Profile Hugging Face TGI models
- [OpenAI Text Endpoints](docs/tutorials/openai-text-endpoints.md) - Profile OpenAI-compatible text APIs

### Load Control and Timing
- [Request Rate with Max Concurrency](docs/tutorials/request-rate-concurrency.md) - Dual request control
- [Arrival Patterns](docs/tutorials/arrival-patterns.md) - Constant, Poisson, gamma traffic
- [Prefill Concurrency](docs/tutorials/prefill-concurrency.md) - Memory-safe long-context benchmarking
- [Gradual Ramping](docs/tutorials/ramping.md) - Smooth ramp-up of concurrency and request rate
- [Warmup Phase](docs/tutorials/warmup.md) - Eliminate cold-start effects
- [User-Centric Timing](docs/tutorials/user-centric-timing.md) - Per-user rate limiting for KV cache benchmarking
- [Request Cancellation](docs/tutorials/request-cancellation.md) - Timeout and resilience testing
- [Multi-URL Load Balancing](docs/tutorials/multi-url-load-balancing.md) - Distribute across servers

### Workloads and Data
- [Trace Benchmarking](docs/benchmark-modes/trace-replay.md) - Deterministic workload replay
- [Bailian Traces](docs/tutorials/bailian-trace.md) - Bailian production trace replay
- [Custom Prompt Benchmarking](docs/tutorials/custom-prompt-benchmarking.md) - Send exact prompts as-is
- [Custom Dataset](docs/tutorials/custom-dataset.md) - Custom dataset formats
- [ShareGPT Dataset](docs/tutorials/sharegpt.md) - Profile with ShareGPT dataset
- [AIMO Dataset](docs/tutorials/aimo.md) - Profile with AIMO math reasoning dataset
- [MMStar Dataset](docs/tutorials/mmstar.md) - Profile vision language models with MMStar visual QA benchmark
- [VisionArena Dataset](docs/tutorials/vision-arena.md) - Profile with real-world vision conversations from Chatbot Arena
- [LLaVA-OneVision Dataset](docs/tutorials/llava-onevision.md) - Profile with diverse multimodal instruction-following data
- [Synthetic Dataset Generation](docs/tutorials/synthetic-dataset.md) - Generate synthetic datasets
- [Fixed Schedule](docs/tutorials/fixed-schedule.md) - Precise timestamp-based execution
- [Time-based Benchmarking](docs/tutorials/time-based-benchmarking.md) - Duration-based testing
- [Sequence Distributions](docs/tutorials/sequence-distributions.md) - Mixed ISL/OSL pairings
- [Prefix Synthesis](docs/tutorials/prefix-synthesis.md) - Prefix data synthesis for KV cache testing
- [Reproducibility](docs/reproducibility.md) - Deterministic datasets with `--random-seed`
- [Template Endpoint](docs/tutorials/template-endpoint.md) - Custom Jinja2 request templates
- [Multi-Turn Conversations](docs/tutorials/multi-turn.md) - Multi-turn conversation benchmarking
- [Local Tokenizer](docs/tutorials/local-tokenizer.md) - Use local tokenizers without HuggingFace

### Endpoint Types
- [Embeddings](docs/tutorials/embeddings.md) - Profile embedding models
- [Rankings](docs/tutorials/rankings.md) - Profile ranking models
- [Audio](docs/tutorials/audio.md) - Profile audio language models
- [Vision](docs/tutorials/vision.md) - Profile vision language models
- [Image Generation](docs/tutorials/image-generation.md) - Benchmark any OpenAI-compatible image generation API
- [SGLang Video Generation](docs/tutorials/sglang-video-generation.md) - Video generation benchmarking
- [Synthetic Video](docs/tutorials/synthetic-video.md) - Synthetic video generation

### Analysis and Monitoring
- [Timeslice Metrics](docs/tutorials/timeslices.md) - Per-timeslice performance analysis
- [Goodput](docs/tutorials/goodput.md) - SLO-based throughput measurement
- [HTTP Trace Metrics](docs/tutorials/http-trace-metrics.md) - DNS, TCP/TLS, TTFB timing
- [Multi-Run Confidence](docs/tutorials/multi-run-confidence.md) - Confidence intervals across repeated runs
- [Profile Exports](docs/tutorials/working-with-profile-exports.md) - Post-processing with Pydantic models
- [Visualization and Plotting](docs/tutorials/plot.md) - PNG charts and multi-run comparison
- [GPU Telemetry](docs/tutorials/gpu-telemetry.md) - DCGM metrics collection
- [Server Metrics](docs/server-metrics/server-metrics.md) - Prometheus-compatible metrics

## Documentation

| Document | Purpose |
|----------|---------|
| [Architecture](docs/architecture.md) | Three-plane architecture, core components, credit system, data flow |
| [CLI Options](docs/cli-options.md) | Complete command and option reference |
| [Metrics Reference](docs/metrics-reference.md) | All metric definitions, formulas, and requirements |
| [Environment Variables](docs/environment-variables.md) | All `AIPERF_*` configuration variables |
| [Plugin System](docs/plugins/plugin-system.md) | Plugin architecture, 25+ categories, creation guide |
| [Creating Plugins](docs/plugins/creating-your-first-plugin.md) | Step-by-step plugin tutorial |
| [Accuracy Benchmarks](docs/accuracy/accuracy_stubs.md) | Accuracy evaluation stubs and datasets |
| [Benchmark Modes](docs/benchmark-modes/trace-replay.md) | Trace replay and timing modes |
| [Server Metrics](docs/server-metrics/server-metrics.md) | Prometheus-compatible server metrics collection |
| [Tokenizer Auto-Detection](docs/reference/tokenizer-auto-detection.md) | Pre-flight tokenizer detection |
| [Conversation Context Mode](docs/reference/conversation-context-mode.md) | How conversation history accumulates in multi-turn |
| [Dataset Synthesis API](docs/api/synthesis.md) | Synthesis module API reference |
| [Code Patterns](docs/dev/patterns.md) | Code examples for services, models, messages, plugins |
| [Migrating from Genai-Perf](docs/migrating.md) | Migration guide and feature comparison |
| [Design Proposals](https://github.com/ai-dynamo/enhancements) | Enhancement proposals and discussions |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, coding conventions, and contribution guidelines.

## Known Issues

- Output sequence length constraints (`--output-tokens-mean`) cannot be guaranteed unless you pass `ignore_eos` and/or `min_tokens` via `--extra-inputs` to an inference server that supports them.
- Very high concurrency settings (typically >15,000) may lead to port exhaustion on some systems. Adjust system limits or reduce concurrency if connection failures occur.
- Startup errors caused by invalid configuration settings can cause AIPerf to hang indefinitely. Terminate the process and check configuration settings.
- Copying selected text may not work reliably in the dashboard UI. Use the `c` key to copy all logs.
