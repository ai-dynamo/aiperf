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

```bash
pip install aiperf

aiperf profile \
  --model Qwen/Qwen3-0.6B \
  --url http://localhost:8000 \
  --endpoint-type chat \
  --concurrency 10 \
  --request-count 100 \
  --streaming
```

## Features

- Scalable multiprocess architecture with 9 services communicating via ZMQ
- 3 UI modes: `dashboard` (real-time TUI), `simple` (progress bars), `none` (headless)
- Multiple benchmarking modes: concurrency, request-rate, [request-rate with max concurrency](docs/tutorials/request-rate-concurrency.md), [trace replay](docs/benchmark_modes/trace_replay.md)
- Extensible plugin system for endpoints, datasets, transports, and metrics
- [Public dataset support](docs/benchmark_datasets.md) including ShareGPT and custom formats

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
- [Trace Benchmarking](docs/benchmark_modes/trace_replay.md) - Deterministic workload replay
- [Custom Prompt Benchmarking](docs/tutorials/custom-prompt-benchmarking.md) - Send exact prompts as-is
- [Custom Dataset](docs/tutorials/custom-dataset.md) - Custom dataset formats
- [ShareGPT Dataset](docs/tutorials/sharegpt.md) - Profile with ShareGPT dataset
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
- [SGLang Image Generation](docs/tutorials/sglang-image-generation.md) - Image generation benchmarking
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
- [Server Metrics](docs/server_metrics/server-metrics.md) - Prometheus-compatible metrics

## Documentation

| Document | Purpose |
|----------|---------|
| [Architecture](docs/architecture.md) | Three-plane architecture, core components, credit system, data flow |
| [CLI Options](docs/cli_options.md) | Complete command and option reference |
| [Metrics Reference](docs/metrics_reference.md) | All metric definitions, formulas, and requirements |
| [Environment Variables](docs/environment_variables.md) | All `AIPERF_*` configuration variables |
| [Plugin System](docs/plugins/plugin-system.md) | Plugin architecture, 25+ categories, creation guide |
| [Creating Plugins](docs/plugins/creating-your-first-plugin.md) | Step-by-step plugin tutorial |
| [Accuracy Benchmarks](docs/accuracy/accuracy_stubs.md) | Accuracy evaluation stubs and datasets |
| [Benchmark Modes](docs/benchmark_modes/trace_replay.md) | Trace replay and timing modes |
| [Server Metrics](docs/server_metrics/server-metrics.md) | Prometheus-compatible server metrics collection |
| [Tokenizer Auto-Detection](docs/reference/tokenizer-auto-detection.md) | Pre-flight tokenizer detection |
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
