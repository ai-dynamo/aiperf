<!--
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# AIPerf Configuration Templates

Bundled YAML templates for common benchmarking scenarios. Each template embeds
`# @template` metadata that powers `aiperf config init`.

## Quick Start

```bash
# List all templates grouped by category
aiperf config init --list

# Search by keyword
aiperf config init --search sweep

# Generate a template (prints to stdout)
aiperf config init --template goodput_slo

# Generate with endpoint pre-filled and save to a file
aiperf config init --template latency_test \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --url http://localhost:8000/v1/chat/completions \
    --output benchmark.yaml

# Run the generated config
aiperf profile --config benchmark.yaml
```

## Templates

### Getting Started

| Template | Title | Description |
|----------|-------|-------------|
| [minimal.yaml](minimal.yaml) | Minimal Configuration | Bare minimum config using shorthand forms -- the fastest way to get started |
| [warmup_profiling.yaml](warmup_profiling.yaml) | Warmup + Profiling (Two-Phase) | Proper benchmark setup: warmup phase for JIT/cache, then clean profiling |

### Load Testing

| Template | Title | Description |
|----------|-------|-------------|
| [latency_test.yaml](latency_test.yaml) | Latency Test (Controlled QPS) | Measure TTFT, ITL, and E2E latency at a controlled request rate |
| [goodput_slo.yaml](goodput_slo.yaml) | Goodput / SLO Benchmark | Measure good requests/sec that meet latency SLO thresholds at multiple load levels |
| [multi_url_load_balancing.yaml](multi_url_load_balancing.yaml) | Multi-URL Load Balancing | Distribute requests across multiple server replicas for load balancer testing |
| [request_cancellation.yaml](request_cancellation.yaml) | Request Cancellation Test | Test server behavior when clients cancel in-flight requests |
| [ramping.yaml](ramping.yaml) | Gradual Load Ramping | Smoothly ramp concurrency and request rate to avoid cold-start transients and connection storms |
| [time_based_soak.yaml](time_based_soak.yaml) | Time-Based Soak Test | Run a fixed-duration benchmark (e.g., 1h) to validate stability and detect leaks under sustained load |
| [fixed_schedule.yaml](fixed_schedule.yaml) | Fixed Schedule (Hand-Authored Timestamps) | Send requests at exact millisecond timestamps from a JSONL file for deterministic temporal testing |

### Datasets

| Template | Title | Description |
|----------|-------|-------------|
| [public_dataset.yaml](public_dataset.yaml) | Public Dataset (ShareGPT) | Use real multi-turn conversations from the ShareGPT public dataset |
| [multi_turn_conversation.yaml](multi_turn_conversation.yaml) | Multi-Turn Conversation | Simulate realistic chatbot workloads with multi-turn context accumulation |
| [inline_dataset.yaml](inline_dataset.yaml) | Inline Dataset | Embed dataset records directly in the YAML config (no separate JSONL file required) |
| [trace_replay.yaml](trace_replay.yaml) | Production Trace Replay | Replay production traffic from a trace file with exact request timestamps |

### Sweep & Multi-Run

| Template | Title | Description |
|----------|-------|-------------|
| [scenario_workload_profiles.yaml](scenario_workload_profiles.yaml) | Scenario Sweep: Workload Profiles | Hand-curated named scenarios testing distinct workload shapes |
| [speed_bench_sweep.yaml](speed_bench_sweep.yaml) | SPEED-Bench Per-Category Sweep | Sweep all 11 qualitative SPEED-Bench categories to populate the speed-bench-report matrix |
| [sweep_distributions.yaml](sweep_distributions.yaml) | Grid Sweep + Multi-Run | Cartesian product sweep over ISL x rate with statistical multi-run aggregation |
| [sweep_with_plot.yaml](sweep_with_plot.yaml) | Concurrency Sweep with Inline Plot Envelope | Sweep concurrency to draw a Pareto frontier, with the visualization config inlined in the same YAML |

### Advanced

| Template | Title | Description |
|----------|-------|-------------|
| [env_var_production.yaml](env_var_production.yaml) | Environment Variable Production Config | CI/CD-friendly template where all deployment-specific values come from env vars |
| [jinja2_variables.yaml](jinja2_variables.yaml) | Jinja2 Computed Config | Define variables once and compute derived values with Jinja2 expressions |
| [kv_cache_test.yaml](kv_cache_test.yaml) | KV Cache / Prefix Caching | Test KV cache efficiency with shared system prompts using user-centric mode |
| [long_context.yaml](long_context.yaml) | Long Context Benchmark (32K+) | Test performance with long input contexts and prefill concurrency limits |
| [user_files.yaml](user_files.yaml) | User Files (Templated Artifacts) | Capture run parameters as JSON/YAML/text files materialized into the artifact directory |
| [gpu_telemetry.yaml](gpu_telemetry.yaml) | GPU Telemetry (DCGM / pynvml) | Collect GPU power, utilization, memory, and temperature during a benchmark via DCGM or pynvml |
| [http_trace_metrics.yaml](http_trace_metrics.yaml) | HTTP Trace Metrics (Latency Breakdown) | Capture per-phase HTTP timing (DNS, connect, sending, waiting/TTFB, receiving) for transport-layer debugging |

### Multimodal

| Template | Title | Description |
|----------|-------|-------------|
| [multimodal_vision.yaml](multimodal_vision.yaml) | Vision-Language Model Benchmark | Benchmark VLMs with synthetic images of varying resolutions |
| [audio_multimodal.yaml](audio_multimodal.yaml) | Audio/Speech Model Benchmark | Benchmark speech-to-text or audio understanding models |

### Specialized Endpoints

| Template | Title | Description |
|----------|-------|-------------|
| [embeddings.yaml](embeddings.yaml) | Embeddings Endpoint Benchmark | Benchmark embedding models with single-text and batched requests |

## Configuration Structure

Templates use the schema-2.0 envelope shape: cross-variation envelope keys
(`sweep`, `multi_run` / `multiRun`, `variables`, `random_seed` / `randomSeed`)
stay at the top level; the swept workload body nests under `benchmark:`.

Singular shorthand (`model`, `dataset`, `warmup`, `profiling`) and plural list
forms (`models`, `datasets`, `phases` as a list of named entries) are both
accepted under `benchmark:` — use whichever fits the config complexity.

```yaml
# Shorthand (minimal configs)
benchmark:
  model: meta-llama/Llama-3.1-8B-Instruct
  endpoint:
    url: http://localhost:8000
  dataset:
    type: synthetic
    entries: 100
    prompts: {isl: 512, osl: 128}
  phases:
    type: concurrency
    concurrency: 8
    requests: 100

# Named / multi-phase form (complex configs)
benchmark:
  models: [meta-llama/Llama-3.1-8B-Instruct]
  endpoint:
    urls: [http://localhost:8000/v1/chat/completions]
  datasets:
    - {name: main, type: synthetic, ...}
  phases:
    - {name: warmup, type: concurrency, ...}
    - {name: profiling, type: poisson, rate: 30.0, ...}

  # Optional benchmark-body sections
  artifacts: {...}        # Export paths and console settings
  slos: {...}             # SLO thresholds for goodput
  tokenizer: {...}        # Token counting settings
  gpuTelemetry: {...}     # GPU metrics collection
  serverMetrics: {...}    # Server metrics scraping
  runtime: {...}          # Worker settings
  logging: {...}          # Logging / debug

# Envelope-level keys (stay at top level)
randomSeed: 42            # Reproducibility
multiRun: {...}           # Statistical multi-run aggregation
variables: {...}          # Jinja2 user-defined values
sweep: {...}              # Parameter sweep (grid / scenarios / adaptive_search)
```

## Environment Variables

Use `${VAR}` or `${VAR:default}` syntax for deployment-specific values:

```yaml
benchmark:
  endpoint:
    api_key: ${OPENAI_API_KEY}
    urls:
      - ${SERVER_URL:http://localhost:8000/v1/chat/completions}
```

## Adding a Template

1. Create `<name>.yaml` in this directory
2. Add the `# @template` metadata block near the top:

```yaml
# @template
# title: Human-Readable Title
# description: One-line summary of what this template demonstrates.
# category: Getting Started
# tags: tag1, tag2
# difficulty: beginner
# features: feature1, feature2
```

Valid categories: `Getting Started`, `Load Testing`, `Datasets`,
`Sweep & Multi-Run`, `Advanced`, `Multimodal`, `Specialized Endpoints`.

Valid difficulties: `beginner`, `intermediate`, `advanced`.

The template is automatically picked up by `aiperf config init` — no
registration required.
