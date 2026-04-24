---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: "YAML Configuration Reference"
---

# YAML Configuration Reference

This guide covers the complete AIPerf YAML configuration format. Every section, field, and distribution type is documented with working examples you can copy into your own config files.

## Config File Basics

AIPerf accepts a single YAML file that declaratively describes your entire benchmark: which models and endpoints to hit, what data to send, how to pace requests, and where to write results.

```bash
aiperf --config benchmark.yaml
```

A minimal config needs four sections: a model name, an endpoint URL, a dataset, and at least one phase.

```yaml
model: meta-llama/Llama-3.1-8B-Instruct

endpoint:
  url: http://localhost:8000

dataset:
  type: synthetic
  entries: 100
  prompts:
    isl: 512
    osl: 128

phases:
  type: concurrency
  concurrency: 8
  requests: 100
```

Shorthand forms (`model` instead of `models`, `url` instead of `urls`, inline `phases` without a name) are normalized automatically. The rest of this guide uses the explicit forms.

## Models and Endpoints

### Models

Accepts a single string, a list, or advanced format with selection strategy.

```yaml
# List of models (round-robin by default)
models:
  - meta-llama/Llama-3.1-8B-Instruct
  - mistralai/Mistral-7B-Instruct-v0.3

# Weighted routing
models:
  strategy: weighted
  items:
    - name: meta-llama/Llama-3.1-8B-Instruct
      weight: 0.7
    - name: mistralai/Mistral-7B-Instruct-v0.3
      weight: 0.3
```

### Endpoints

Configures server connection, API type, streaming, and authentication.

```yaml
endpoint:
  urls:
    - ${INFERENCE_URL:http://localhost:8000/v1/chat/completions}
  type: chat           # chat | completions | embeddings | rankings | template
  streaming: true      # Required for TTFT measurement
  timeout: 600.0       # Request timeout in seconds
  api_key: ${OPENAI_API_KEY}
  headers:
    X-Request-ID: ${TRACE_ID:none}
  extra:
    temperature: 0.7
    top_p: 0.95
```

## Datasets

Datasets are named data sources referenced by phases. Four types are supported.

### Synthetic

Generated prompts with configurable token length distributions.

```yaml
datasets:
  main:
    type: synthetic
    entries: 1000
    prompts:
      isl: {type: normal, mean: 512, stddev: 100}
      osl: {type: normal, mean: 256, stddev: 50}
    sampling: shuffle    # sequential | random | shuffle
    random_seed: 42
```

Multi-turn conversations:

```yaml
datasets:
  conversation:
    type: synthetic
    entries: 500
    prompts:
      isl: {type: normal, mean: 200, stddev: 50}
      osl: {type: normal, mean: 300, stddev: 75}
    turns: {type: normal, mean: 5, stddev: 2}
    turn_delay: {type: exponential, mean: 2000}  # ms between turns
```

### File

Load prompts from a local JSONL file.

```yaml
datasets:
  custom:
    type: file
    path: ./data/prompts.jsonl
    format: single_turn   # single_turn | multi_turn | mooncake_trace
    sampling: sequential
    entries: 500           # Limit to first 500 records
```

### Public

Auto-downloaded public benchmarking datasets.

```yaml
datasets:
  sharegpt:
    type: public
    name: sharegpt
    entries: 1000
```

### Composed

Combines a file source with synthetic augmentation (e.g., applying OSL control to file-based prompts).

```yaml
datasets:
  with_osl:
    type: composed
    source:
      type: file
      path: ./data/prompts.jsonl
      format: single_turn
      sampling: shuffle
    augment:
      osl: {type: normal, mean: 256, stddev: 50}
      osl_mode: fill   # fill (only if missing) | override (always replace)
    entries: 1000
```

### Using Multiple Datasets

Name each dataset and reference it from phases. Phases that omit `dataset` use the first one.

```yaml
datasets:
  warmup_data:
    type: synthetic
    entries: 100
    prompts: {isl: 256, osl: 64}

  profiling_data:
    type: synthetic
    entries: 5000
    prompts:
      isl: {type: lognormal, mean: 512, sigma: 0.8}
      osl: {type: normal, mean: 256, stddev: 50}

phases:
  warmup:
    type: concurrency
    dataset: warmup_data
    requests: 100
    concurrency: 8
    exclude_from_results: true

  profiling:
    type: poisson
    dataset: profiling_data
    rate: 30.0
    duration: 300
    concurrency: 64
    seamless: true
    grace_period: 60
```

## Phases

Phases define how requests are paced and when to stop. They run in declaration order.

Every phase requires at least one stop condition: `requests`, `duration`, or `sessions` (except `fixed_schedule`, which infers stop from the dataset).

### concurrency

Dispatch a new request immediately when a concurrency slot opens. No rate limiting.

```yaml
phases:
  burst_test:
    type: concurrency
    concurrency: 32
    requests: 1000
```

### poisson

Rate-controlled with Poisson-distributed inter-arrival times (realistic traffic).

```yaml
phases:
  realistic:
    type: poisson
    rate: 50.0        # Requests per second
    concurrency: 128  # Cap on in-flight requests
    duration: 300     # Supports: 300, "5m", "2h"
    grace_period: 60
```

### gamma

Rate-controlled with tunable burstiness via `smoothness`.

```yaml
phases:
  bursty:
    type: gamma
    rate: 30.0
    smoothness: 0.5   # <1 bursty, 1 = Poisson, >1 smooth
    duration: 120
    concurrency: 64
    grace_period: 60
```

### constant

Fixed inter-arrival time (deterministic, reproducible).

```yaml
phases:
  deterministic:
    type: constant
    rate: 10.0
    duration: 60
    concurrency: 32
```

### user_centric

N simulated users sharing a global request rate. Requires multi-turn dataset.

```yaml
phases:
  users:
    type: user_centric
    rate: 20.0
    users: 10
    sessions: 50
    duration: 300
```

### fixed_schedule

Replay requests at timestamps from a trace dataset. No stop condition required.

```yaml
phases:
  replay:
    type: fixed_schedule
    dataset: trace_data
    auto_offset: true    # Normalize timestamps to start at 0
```

### Common Phase Fields

**All phases:**

- `dataset` -- Named dataset to use (default: first dataset)
- `requests` / `duration` / `sessions` -- Stop conditions (at least one required)
- `concurrency` -- Max concurrent in-flight requests
- `concurrency_ramp` -- Ramp gradually: `30` (seconds) or `{duration: 30, strategy: linear}`
- `prefill_concurrency` -- Max concurrent requests in prefill stage
- `grace_period` -- Seconds to wait for in-flight requests after duration expires
- `seamless` -- Start immediately after previous phase (cannot be first)
- `exclude_from_results` -- Exclude from final metrics (for warmup)
- `cancellation` -- `{rate: 10.0, delay: 0.5}` to cancel 10% of requests after 0.5s

**Rate-controlled phases** add: `rate` (req/s) and `rate_ramp` (e.g., `"30s"` or `{duration: 30, strategy: exponential}`)

### Warmup + Profiling Pattern

```yaml
phases:
  warmup:
    type: concurrency
    requests: 100
    concurrency: 8
    exclude_from_results: true

  profiling:
    type: gamma
    rate: 50.0
    smoothness: 1.5
    duration: 300
    concurrency: 64
    seamless: true
    rate_ramp: 30s
    grace_period: 60
```

## Distribution Types

Any field that accepts a `SamplingDistribution` (ISL, OSL, turns, turn_delay, image width/height, audio length, passages, etc.) supports all 9 distribution types.

### Fixed

A constant value. Bare integers coerce to this automatically.

```yaml
isl: 512
# Equivalent to:
isl: {type: fixed, value: 512}
```

### Normal

Gaussian distribution (truncated at 0).

```yaml
isl: {type: normal, mean: 512, stddev: 100}
```

### LogNormal

Right-skewed distribution. Most values near the mean, with a long right tail. Parameterized by desired output mean and shape (sigma).

```yaml
isl: {type: lognormal, mean: 512, sigma: 0.8}
```

**Use cases:**
- Production token-length distributions where most requests are short but some are very long
- Latency modeling

### Uniform

Flat distribution between bounds.

```yaml
isl: {type: uniform, min: 128, max: 2048}
```

**Use cases:**
- Stress testing across a range of input sizes
- Even coverage of a parameter space

### Exponential

Memoryless distribution. Always positive. Parameterized by mean.

```yaml
turn_delay: {type: exponential, mean: 2000}
```

**Use cases:**
- Think times between conversation turns
- Inter-event delays

### Zipf

Power-law distribution. A few ranks dominate; long tail of rare values. Output is `offset + rank * scale`.

```yaml
isl: {type: zipf, alpha: 2.0, scale: 128}
# Produces values: 128, 256, 384, ... (with rank 1 most frequent)
```

**Use cases:**
- Request popularity distributions
- Modeling real-world frequency patterns

### Mixture

Weighted combination of sub-distributions. Requires at least 2 components.

```yaml
isl:
  type: mixture
  components:
    - distribution: {type: normal, mean: 128, stddev: 30}
      weight: 70
    - distribution: {type: lognormal, mean: 2048, sigma: 0.4}
      weight: 30
```

**Use cases:**
- Bimodal workloads (short Q&A + long summarization)
- Mixed traffic from different application types

### Clamped

Wraps any distribution with hard min/max bounds. Requires at least one of min or max.

```yaml
isl:
  type: clamped
  distribution: {type: normal, mean: 1024, stddev: 500}
  min: 64
  max: 4096
```

**Use cases:**
- Bounding normal distributions to safe hardware limits
- Preventing runaway generation lengths

### Empirical

Sample from exact observed values with weights.

```yaml
isl:
  type: empirical
  points:
    - {value: 128, weight: 40}
    - {value: 256, weight: 25}
    - {value: 512, weight: 20}
    - {value: 1024, weight: 10}
    - {value: 4096, weight: 5}
```

**Use cases:**
- Replaying exact production token-count histograms
- Testing specific input sizes with known frequency

### Where Distributions Can Be Used

Distributions work anywhere the config accepts `SamplingDistribution`: `prompts.isl`, `prompts.osl`, `turns`, `turn_delay`, `images.width`, `images.height`, `audio.length`, `rankings.passages`, `rankings.passage_tokens`, `rankings.query_tokens`, and `augment.osl`.

## Sweeps

Sweeps run multiple benchmark variations from a single config file.

### Grid Sweep

Cartesian product of all variable combinations. Dot-notation paths reference any config field.

```yaml
sweep:
  type: grid
  variables:
    datasets.profiling.prompts.isl: [128, 512, 2048]
    phases.profiling.rate: [10.0, 30.0, 50.0]
# Produces 3 x 3 = 9 benchmark runs
```

### Scenario Sweep

Hand-picked configurations deep-merged with the base config. Each scenario can override any nested field.

```yaml
sweep:
  type: scenarios
  runs:
    - name: chatbot
      datasets:
        workload:
          prompts:
            isl: {type: normal, mean: 128, stddev: 20}
            osl: {type: normal, mean: 64, stddev: 10}
      phases:
        test:
          rate: 50.0

    - name: summarization
      datasets:
        workload:
          prompts:
            isl: {type: lognormal, mean: 4096, sigma: 0.6}
            osl: {type: normal, mean: 128, stddev: 30}
      phases:
        test:
          rate: 5.0
```

## Multi-Run Statistics

Run the same benchmark multiple times and compute aggregate statistics (mean, std, confidence intervals).

```yaml
multi_run:
  num_runs: 5                     # 1-10 runs
  cooldown_seconds: 15.0          # Seconds between runs
  confidence_level: 0.95          # 0.90, 0.95, or 0.99
  set_consistent_seed: true       # Auto-set seed for workload consistency
  disable_warmup_after_first: true  # Warmup only on first run
```

Multi-run works with sweeps: each sweep variation gets `num_runs` repetitions with aggregate statistics.

## SLOs (Goodput)

Service Level Objectives define thresholds for "good" requests. A request counts toward goodput only if it meets ALL specified thresholds. Latency metrics are in milliseconds; `tokens_per_second` is a minimum.

```yaml
slos:
  request_latency: 5000        # Max 5s end-to-end
  time_to_first_token: 500     # Max 500ms TTFT
  inter_token_latency: 50      # Max 50ms between tokens
  tokens_per_second: 20.0      # Min 20 tokens/sec output rate
```

Streaming must be enabled on the endpoint for TTFT and ITL measurement.

## Environment Variables

Use `${VAR}` syntax in any string value. Variables are resolved at config load time.

```yaml
# Required (error if not set)
endpoint:
  api_key: ${OPENAI_API_KEY}

# Optional with default
models:
  - ${MODEL_NAME:meta-llama/Llama-3.1-8B-Instruct}

# Works in nested fields, numeric fields, and lists
random_seed: ${BENCHMARK_SEED:42}
phases:
  profiling:
    type: gamma
    rate: ${TARGET_RATE:30.0}
    duration: ${DURATION:300}
    concurrency: ${MAX_CONCURRENCY:64}
```

## Artifacts

Control output directory and export formats.

```yaml
artifacts:
  dir: ./artifacts/my_benchmark   # Output directory
  summary: [json]                 # [json, yaml] or false
  records: [jsonl, csv]           # [jsonl, csv] or false
  slice_duration: 60              # Time-slice window in seconds
```

When `slice_duration` is set, AIPerf computes per-window statistics for trend analysis over time.

## Jinja2 Config Templates

Define variables once and compute values throughout your config using `{{ }}` expressions.

```yaml
variables:
  base_concurrency: 16
  target_isl: 512

datasets:
  main:
    type: synthetic
    entries: "{{ base_concurrency * 10 }}"    # 160
    prompts:
      isl: "{{ target_isl }}"                 # 512
      osl: "{{ target_isl // 4 }}"            # 128

phases:
  test:
    type: concurrency
    concurrency: "{{ base_concurrency }}"      # 16
    requests: "{{ base_concurrency * 100 }}"   # 1600
```

Jinja2 templates also support self-referencing config values via dot notation:

```yaml
phases:
  test:
    type: poisson
    rate: 50.0
    duration: 120
    # Reference another config value:
    concurrency: "{{ phases.test.rate * 2 }}"  # 100
```

Templates render twice: once at load time (for validation) and again after sweep expansion (so sweep-injected values are available). This means Jinja2 and sweeps compose naturally.

### Endpoint Payload Templates

The `template` endpoint type uses Jinja2 for custom request payloads. These are rendered at request time (not config load time), with different variables (`text`, `model`, `max_tokens`, etc.).

```yaml
endpoint:
  type: template
  template:
    body: |
      {
        "prompt": {{ text|tojson }},
        "model": {{ model|tojson }},
        "max_tokens": {{ max_tokens|tojson }}
      }
```

See the [Template Endpoint tutorial](template-endpoint.md) for complete examples.

## Complete End-to-End Example

This example combines distributions, scenario sweep, multi-run statistics, SLOs, and environment variables into a single production benchmark config.

```yaml
random_seed: ${BENCHMARK_SEED:42}

models:
  - ${MODEL_NAME:meta-llama/Llama-3.1-8B-Instruct}

endpoint:
  urls:
    - ${INFERENCE_URL:http://localhost:8000/v1/chat/completions}
  type: chat
  streaming: true
  timeout: 600.0
  api_key: ${API_KEY:}

datasets:
  warmup:
    type: synthetic
    entries: 100
    prompts:
      isl: 256
      osl: 64

  workload:
    type: synthetic
    entries: 2000
    prompts:
      isl:
        type: mixture
        components:
          - distribution: {type: lognormal, mean: 256, sigma: 0.6}
            weight: 70
          - distribution: {type: lognormal, mean: 2048, sigma: 0.4}
            weight: 30
      osl:
        type: clamped
        distribution: {type: lognormal, mean: 256, sigma: 0.5}
        min: 16
        max: 2048

phases:
  warmup:
    type: concurrency
    dataset: warmup
    requests: 100
    concurrency: 8
    exclude_from_results: true

  profiling:
    type: gamma
    dataset: workload
    rate: 30.0
    smoothness: 1.5
    duration: ${DURATION:300}
    concurrency: ${MAX_CONCURRENCY:64}
    rate_ramp: 30s
    seamless: true
    grace_period: 60

sweep:
  type: scenarios
  runs:
    - name: low_load
      phases:
        profiling:
          rate: 10.0
          concurrency: 32
    - name: medium_load
      phases:
        profiling:
          rate: 30.0
          concurrency: 64
    - name: high_load
      phases:
        profiling:
          rate: 50.0
          concurrency: 128

slos:
  request_latency: 10000
  time_to_first_token: 1000
  inter_token_latency: 50
  tokens_per_second: 20.0

multi_run:
  num_runs: ${NUM_RUNS:3}
  cooldown_seconds: ${COOLDOWN:30.0}
  confidence_level: 0.95
  set_consistent_seed: true
  disable_warmup_after_first: true

artifacts:
  dir: ${ARTIFACTS_DIR:./artifacts/production}
  summary: [json]
  records: [jsonl, csv]
  slice_duration: 60
```

Run it:

```bash
INFERENCE_URL=http://gpu-server:8000/v1/chat/completions \
MODEL_NAME=meta-llama/Llama-3.1-70B-Instruct \
NUM_RUNS=5 \
aiperf --config production_benchmark.yaml
```

This produces 3 scenarios x 5 runs = 15 total benchmark executions, with per-scenario aggregate statistics including confidence intervals, goodput metrics, and time-slice trends.

## Related Documentation

- [Arrival Patterns](./arrival-patterns.md) -- Poisson, Gamma, and Constant inter-arrival distributions
- [Sequence Length Distributions](./sequence-distributions.md) -- ISL/OSL pair distributions
- [Multi-Run Confidence Reporting](./multi-run-confidence.md) -- Aggregate statistics deep dive
- [Goodput](./goodput.md) -- SLO-based quality metrics
- [Request Cancellation](./request-cancellation.md) -- Mid-flight cancellation testing
- [Prefill Concurrency](./prefill-concurrency.md) -- Long context memory management
- [Ramping](./ramping.md) -- Gradual rate and concurrency ramp-up
