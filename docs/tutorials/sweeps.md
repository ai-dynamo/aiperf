---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Parameter Sweeps and Multi-Run Statistics
---

# Parameter Sweeps and Multi-Run Statistics

Finding the optimal operating point for an inference server requires exploring a multi-dimensional space of concurrency, request rate, input lengths, and batch sizes. Rather than hand-tuning one variable at a time, parameter sweeps let you define the search space declaratively and let AIPerf run every combination, collecting statistically rigorous results for each.

## Sweep Strategies

AIPerf supports two sweep strategies:

| Strategy | How it works | Best for | Variations generated |
|---|---|---|---|
| **Grid** | Cartesian product of variable lists | Systematic exploration of 2-3 variables | `len(v1) * len(v2) * ...` |
| **Scenarios** | Named configs deep-merged onto base | Comparing hand-picked workload profiles | One per scenario |

## Grid Sweep

A grid sweep takes one or more variables, each with a list of values, and runs every combination (Cartesian product). Variables use dot-notation paths that map to fields in the YAML config tree.

### Example: Sweep Concurrency x Rate to Find Saturation

```yaml
models:
  - meta-llama/Llama-3.1-8B-Instruct

endpoint:
  urls:
    - http://localhost:8000/v1/chat/completions
  type: chat
  streaming: true

datasets:
  main:
    type: synthetic
    entries: 2000
    prompts:
      isl: {type: normal, mean: 512, stddev: 50}
      osl: {type: normal, mean: 128, stddev: 25}

phases:
  profiling:
    type: poisson
    dataset: main
    duration: 120
    rate: 10       # overridden by sweep
    concurrency: 8 # overridden by sweep
    grace_period: 30

sweep:
  type: grid
  variables:
    phases.profiling.concurrency: [8, 32, 64, 128]
    phases.profiling.rate: [10, 50, 100]

artifacts:
  dir: ./artifacts/saturation_sweep
  console: table
  summary: [json]
```

This produces `4 * 3 = 12` benchmark runs. Each variation overrides the dot-path fields on a deep copy of the base config, so `phases.profiling.concurrency: 32` sets the `concurrency` field inside the `profiling` phase to 32.

The results directory will contain one subdirectory per variation, making it straightforward to compare throughput and latency across the concurrency-rate surface.

## Scenario Sweep

A scenario sweep defines named configurations that are deep-merged onto the base config. Each scenario overrides only the fields it specifies; everything else inherits from the base. This is ideal when comparing qualitatively different workload profiles that touch multiple config sections.

### Example: Compare Workload Profiles

```yaml
models:
  - meta-llama/Llama-3.1-8B-Instruct

endpoint:
  urls:
    - http://localhost:8000/v1/chat/completions
  type: chat
  streaming: true

datasets:
  main:
    type: synthetic
    entries: 2000
    prompts:
      isl: {type: normal, mean: 512, stddev: 50}
      osl: {type: normal, mean: 128, stddev: 25}

phases:
  profiling:
    type: poisson
    dataset: main
    duration: 120
    rate: 20
    concurrency: 32
    grace_period: 30

sweep:
  type: scenarios
  runs:
    - name: short_chatbot
      datasets:
        main:
          prompts:
            isl: {type: normal, mean: 64, stddev: 10}
            osl: {type: normal, mean: 32, stddev: 8}
      phases:
        profiling:
          rate: 100

    - name: summarization
      datasets:
        main:
          prompts:
            isl: {type: normal, mean: 2048, stddev: 200}
            osl: {type: normal, mean: 256, stddev: 50}
      phases:
        profiling:
          concurrency: 16
          rate: 10

    - name: long_context_qa
      datasets:
        main:
          prompts:
            isl: {type: normal, mean: 8192, stddev: 500}
            osl: {type: normal, mean: 512, stddev: 100}
      phases:
        profiling:
          concurrency: 8
          rate: 5

artifacts:
  dir: ./artifacts/workload_comparison
  console: table
  summary: [json]
```

Deep-merge means nested dicts are merged recursively. In the `short_chatbot` scenario, `datasets.main.prompts` is replaced entirely because it is the leaf being overridden, while `datasets.main.type` and `datasets.main.entries` remain inherited from the base. Each scenario's `name` field becomes its label in the output directory.

## Sweep + Distributions

Distribution parameters are just nested fields in the config tree, so they can be sweep variables like any other field. This lets you study how sequence length affects latency and throughput.

### Example: Sweep ISL Across Fixed Values

Use a grid sweep to test three different input sequence lengths:

```yaml
models:
  - meta-llama/Llama-3.1-8B-Instruct

endpoint:
  urls:
    - http://localhost:8000/v1/chat/completions
  type: chat
  streaming: true

datasets:
  main:
    type: synthetic
    entries: 2000
    prompts:
      isl: 128  # overridden by sweep
      osl: {type: normal, mean: 128, stddev: 25}

phases:
  profiling:
    type: poisson
    dataset: main
    duration: 120
    rate: 30
    concurrency: 32
    grace_period: 30

sweep:
  type: grid
  variables:
    datasets.main.prompts.isl: [128, 512, 2048]

artifacts:
  dir: ./artifacts/isl_sweep
  console: table
  summary: [json]
```

This produces 3 runs, one per ISL value. Since ISL accepts both fixed integers and distribution objects, each value is set as a fixed distribution (no variance).

### Example: Sweep Distribution Type via Scenarios

To compare different distribution shapes, use a scenario sweep that replaces the entire distribution object:

```yaml
sweep:
  type: scenarios
  runs:
    - name: fixed_512
      datasets:
        main:
          prompts:
            isl: 512

    - name: normal_512_wide
      datasets:
        main:
          prompts:
            isl: {type: normal, mean: 512, stddev: 100}

    - name: normal_512_narrow
      datasets:
        main:
          prompts:
            isl: {type: normal, mean: 512, stddev: 20}
```

## Multi-Run Statistics

When a single benchmark run is insufficient to account for system jitter, multi-run mode repeats each benchmark multiple times and computes aggregate statistics with confidence intervals.

### Configuration

```yaml
multi_run:
  num_runs: 5
  cooldown_seconds: 10.0
  confidence_level: 0.95
  set_consistent_seed: true
  disable_warmup_after_first: true
```

### Field Reference

| Field | Type | Default | Description |
|---|---|---|---|
| `num_runs` | int (1-10) | 1 | Number of benchmark executions. Set >1 to enable statistical reporting. |
| `cooldown_seconds` | float (>=0) | 0.0 | Seconds to wait between runs. Allows GPU thermals and server state to stabilize. |
| `confidence_level` | float (0-1) | 0.95 | Confidence level for interval computation. Common values: 0.90, 0.95, 0.99. |
| `set_consistent_seed` | bool | true | Auto-set `random_seed: 42` if no seed is specified. Ensures identical workloads across runs so variance reflects system noise, not workload differences. |
| `disable_warmup_after_first` | bool | true | Skip warmup phases on runs 2-N. The server is already warm after the first run, so re-running warmup wastes time and can introduce variance. |

### Sample Output with Confidence Intervals

With `num_runs: 5` and `confidence_level: 0.95`, the aggregate report includes:

```json
{
  "metadata": {
    "aggregation_type": "confidence",
    "num_profile_runs": 5,
    "num_successful_runs": 5,
    "confidence_level": 0.95
  },
  "metrics": {
    "request_throughput_avg": {
      "mean": 47.2,
      "std": 1.8,
      "min": 44.9,
      "max": 49.6,
      "cv": 0.038,
      "se": 0.80,
      "ci_low": 44.9,
      "ci_high": 49.4,
      "t_critical": 2.776,
      "unit": "requests/sec"
    },
    "ttft_p99_ms": {
      "mean": 85.3,
      "std": 4.1,
      "min": 79.8,
      "max": 91.2,
      "cv": 0.048,
      "se": 1.83,
      "ci_low": 80.2,
      "ci_high": 90.4,
      "t_critical": 2.776,
      "unit": "ms"
    }
  }
}
```

A CV below 0.05 (5%) indicates excellent repeatability. The confidence interval tells you the range likely containing the true mean -- if two configurations have non-overlapping intervals, the performance difference is statistically meaningful.

## Sweep + Multi-Run

Sweeps and multi-run combine naturally: each sweep variation is executed `num_runs` times. The total number of benchmark executions is:

```
total_runs = sweep_variations * num_runs
```

### Example: 3 Concurrency Levels x 3 Runs = 9 Total

```yaml
models:
  - meta-llama/Llama-3.1-8B-Instruct

endpoint:
  urls:
    - http://localhost:8000/v1/chat/completions
  type: chat
  streaming: true

datasets:
  main:
    type: synthetic
    entries: 2000
    prompts:
      isl: {type: normal, mean: 512, stddev: 50}
      osl: {type: normal, mean: 128, stddev: 25}

phases:
  warmup:
    type: concurrency
    exclude_from_results: true
    requests: 100
    concurrency: 8

  profiling:
    type: poisson
    dataset: main
    duration: 120
    rate: 30
    concurrency: 16  # overridden by sweep
    seamless: true
    grace_period: 30

sweep:
  type: grid
  variables:
    phases.profiling.concurrency: [16, 64, 128]

multi_run:
  num_runs: 3
  cooldown_seconds: 5.0
  confidence_level: 0.95
  disable_warmup_after_first: true

random_seed: 42

artifacts:
  dir: ./artifacts/concurrency_confidence
  console: table
  summary: [json]
```

This produces `3 * 3 = 9` total benchmark executions. For each of the 3 concurrency levels, AIPerf runs the benchmark 3 times and computes aggregate statistics. The `disable_warmup_after_first` setting means warmup runs once per variation, not once per repetition.

The output directory structure looks like:

```
artifacts/concurrency_confidence/
  variation_0000_concurrency_16/
    profile_runs/
      run_0001/
      run_0002/
      run_0003/
    aggregate/
      profile_export_aiperf_aggregate.json
  variation_0001_concurrency_64/
    profile_runs/
      run_0001/
      run_0002/
      run_0003/
    aggregate/
      profile_export_aiperf_aggregate.json
  variation_0002_concurrency_128/
    ...
```

## Environment Variables in Sweeps

YAML configs support `${VAR}` and `${VAR:default}` syntax for environment variable substitution. This is useful for CI pipelines that override sweep base values without editing the YAML file.

```yaml
endpoint:
  urls:
    - ${INFERENCE_URL:http://localhost:8000/v1/chat/completions}
  type: chat
  streaming: true
  api_key: ${API_KEY:}

models:
  - ${MODEL_NAME:meta-llama/Llama-3.1-8B-Instruct}

datasets:
  main:
    type: synthetic
    entries: ${NUM_PROMPTS:2000}
    prompts:
      isl: {type: normal, mean: 512, stddev: 50}
      osl: {type: normal, mean: 128, stddev: 25}

phases:
  profiling:
    type: poisson
    dataset: main
    duration: ${DURATION:120}
    rate: 30
    concurrency: 32
    grace_period: 30

sweep:
  type: grid
  variables:
    phases.profiling.concurrency: [16, 32, 64, 128]

multi_run:
  num_runs: ${NUM_RUNS:3}
  cooldown_seconds: 5.0
```

A CI job can then override any default:

```bash
INFERENCE_URL=http://gpu-server:8000/v1/chat/completions \
MODEL_NAME=nvidia/Llama-3.1-Nemotron-70B-Instruct \
NUM_RUNS=5 \
DURATION=300 \
aiperf --config sweep_ci.yaml
```

`${VAR}` (without a default) is a required variable -- AIPerf will error if it is not set. `${VAR:default}` falls back to the default value when the variable is unset.

## Best Practices

**Start coarse, then refine.** Begin with a grid sweep over 2-3 values per variable to identify the interesting region. Then define a scenario sweep with hand-picked configurations around that region for detailed comparison.

**Use warmup exclusion and `disable_warmup_after_first`.** Define a warmup phase with `exclude_from_results: true` and enable `disable_warmup_after_first` in multi-run mode. This ensures the server is warm without wasting time re-warming on every repetition.

**Set `random_seed` for reproducibility.** A fixed seed ensures identical prompt selection and request ordering across runs. When `set_consistent_seed` is enabled (default), multi-run mode auto-sets seed 42 if none is specified.

**Use cooldown between runs.** Even a few seconds of cooldown (`cooldown_seconds: 5.0`) lets GPU thermals settle and server-side caches reach steady state, reducing correlation between consecutive runs.

**Keep sweep dimensions small.** Two to three variables with three to five values each keeps total runtime manageable. A `3 * 4 * 5 = 60` variation grid with `num_runs: 3` produces 180 benchmark executions -- plan your time budget accordingly.

**Choose the right strategy.** Use grid sweep when variables are independent (concurrency and ISL). Use scenarios when variables are coupled and you want to control exact combinations (e.g., rate and concurrency increasing together along a load curve).

## Related Documentation

- [Multi-Run Confidence Reporting](./multi-run-confidence.md) -- Statistical methodology and aggregate output format
- [Warmup Phase Configuration](./warmup.md) -- Warmup phase setup and best practices
- [Sequence Length Distributions](./sequence-distributions.md) -- ISL/OSL distribution configuration
- [Arrival Patterns](./arrival-patterns.md) -- Rate-controlled arrival distributions
