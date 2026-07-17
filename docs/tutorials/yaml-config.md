---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: YAML Configuration Files
---

# YAML Configuration Files

## Overview

AIPerf can be driven entirely from a single YAML file instead of a long string of CLI flags. The YAML format is more readable, easier to version-control, and unlocks features that have no CLI equivalent — sweeps, multi-run aggregation, environment variable substitution, and computed values.

This tutorial walks through what a config file looks like, how to grow it from a tiny example to a full sweep, and how it compares to running everything through `aiperf profile` flags.

You don't need to choose between the two: CLI flags still work, and they layer on top of a YAML file when you pass both.

## Why use a YAML config?

A typical concurrency sweep on the command line looks like this:

```bash
aiperf profile \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --url http://localhost:8000/v1/chat/completions \
  --endpoint-type chat --streaming \
  --synthetic-input-tokens-mean 512 --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean 128 --output-tokens-stddev 0 \
  --concurrency 8,16,32,64 \
  --request-count 500 \
  --warmup-request-count 50 \
  --artifact-dir ./artifacts/my-test
```

The same run as a YAML file:

```yaml
# benchmark.yaml
schemaVersion: "2.0"

benchmark:
  model: meta-llama/Llama-3.1-8B-Instruct
  endpoint:
    url: http://localhost:8000/v1/chat/completions
    type: chat
    streaming: true
  dataset:
    type: synthetic
    entries: 500
    prompts: {isl: 512, osl: 128}
  phases:
    - {name: warmup, type: concurrency, concurrency: 8, requests: 50, exclude_from_results: true}
    - {name: profiling, type: concurrency, requests: 500}
  artifacts:
    dir: ./artifacts/my-test

sweep:
  type: grid
  parameters:
    concurrency: [8, 16, 32, 64]
```

Run it with:

```bash
aiperf profile --config benchmark.yaml
```

Note the two `500`s map to different things. The CLI's `--request-count 500` is the *stop condition* — keep firing until 500 requests complete — and corresponds to `phases.profiling.requests: 500`. The `dataset.entries: 500` is the *dataset size* — how many unique synthetic prompts to generate up front — and has no CLI shorthand; it's recycled across requests if the phase runs longer than the dataset. They happen to share a value here but tune independently.

What you gain over the flag form:

- **It's all in one place** and you can comment it. No more lost shell history.
- **Sweeps are first-class.** Grid, lockstep zip, named scenarios, and quasi-random search all work out of the box.
- **You can substitute values** from environment variables (`${VAR:default}`) and compute values with simple expressions (`{{ var * 2 }}`).
- **Editors validate as you type** if you wire up the bundled JSON Schema.
- **Errors are kinder.** Misspelled keys produce a "did you mean...?" hint instead of being silently ignored.

## Your first config — five lines that actually work

The smallest legal config is short:

```yaml
# minimal.yaml
schemaVersion: "2.0"

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
```

Then:

```bash
aiperf profile --config minimal.yaml
```

That's a complete benchmark — model, endpoint, dataset, and one profiling phase. The endpoint path (`/v1/chat/completions`) is auto-detected from `endpoint.type` (defaulting to `chat`).

You can scaffold this exact file from the bundle without typing it:

```bash
aiperf config init --template minimal --output minimal.yaml
```

`aiperf config init --list` prints every bundled template, grouped by category.

## Anatomy of a config

A YAML config has two layers:

```yaml
# --- envelope (cross-run knobs) ---
schemaVersion: "2.0"
random_seed: 42
variables: {...}
sweep: {...}
multi_run: {...}

# --- benchmark body (the workload itself) ---
benchmark:
  model: ...
  endpoint: {...}
  dataset: {...}
  phases: [...]
```

The **envelope** holds settings that apply across runs — sweep definitions, multi-run aggregation, the random seed, and reusable variables.

The **`benchmark:` body** holds everything that defines a single benchmark workload. When a sweep is active, this body is what gets varied across runs.

### Shorthand vs named forms

Short configs use singular keys. Bigger configs use plural lists with names:

```yaml
# Shorthand — fastest to read for simple cases
benchmark:
  model: meta-llama/Llama-3.1-8B-Instruct
  dataset: {type: synthetic, prompts: {isl: 512, osl: 128}}
  phases: {type: concurrency, concurrency: 8, requests: 100}
```

```yaml
# Named — clearer for phases or models; datasets are currently limited to one entry
benchmark:
  models: [meta-llama/Llama-3.1-8B-Instruct]
  datasets:
    - {name: main, type: synthetic, prompts: {isl: 512, osl: 128}}
  phases:
    - {name: warmup, type: concurrency, concurrency: 4, requests: 50, exclude_from_results: true}
    - {name: profiling, type: poisson, rate: 30.0, duration: 120}
```

You can mix and match — the loader auto-expands `model:` into a one-element `models:` list, `dataset:` into a one-entry `datasets:` list named `default`, and a flat `phases:` block into a one-element list named `profiling`. The normalized `datasets:` form is future-facing but currently accepts exactly one dataset; multiple datasets are a roadmap item.

### Inline datasets

Instead of pointing at a `prompts.jsonl` file with `dataset.path:`, you can embed records directly in the YAML:

```yaml
benchmark:
  dataset:
    type: file
    format: single_turn
    records:
      - {text: "What is machine learning?"}
      - {text: "Explain GANs.", output_length: 200}
```

Useful for shareable repros, k8s ConfigMaps, and small regression fixtures. See [Inline Datasets](inline-datasets.md) for full coverage including multi-turn, random_pool (with multi-pool dict-of-lists), and mooncake_trace examples.

### Both naming styles work

AIPerf accepts either `snake_case` or `camelCase` for any field. These two are equivalent:

```yaml
multi_run: {num_runs: 3, cooldown_seconds: 15.0}
```

```yaml
multiRun: {numRuns: 3, cooldownSeconds: 15.0}
```

Pick one and stick with it within a file.

## Editor autocomplete and validation

A bundled JSON Schema gives you autocomplete, type-checking, and inline docs in any editor that speaks YAML language server (VS Code, JetBrains, Vim/Neovim with `coc-yaml`, Helix, etc.). The schema lives at `src/aiperf/config/schema/aiperf-config.schema.json` in the AIPerf repo. Copy or symlink it next to your config and point your editor at it with a relative path:

```yaml
# yaml-language-server: $schema=./aiperf-config.schema.json
```

Now the editor will:

- Suggest valid keys as you type.
- Underline misspelled fields in red.
- Show field descriptions on hover.
- Catch type errors (e.g. setting `concurrency: "eight"` instead of `8`).

If your editor already has a workspace mapping for `**/aiperf-config.yaml` or `**/benchmark.yaml`, you can skip the header. See `src/aiperf/config/schema/README.md` for VS Code workspace and IntelliJ configuration examples.

## Helpful errors when you typo

Top-level envelope keys reject unknown names with a "did you mean" hint. Writing `sweeps:` instead of `sweep:` produces:

```text
Unknown top-level envelope key(s): 'sweeps' (did you mean 'sweep'?). Known keys: ['benchmark', 'multiRun', 'multi_run', 'noSweepTable', 'no_sweep_table', 'plot', 'randomSeed', 'random_seed', 'schemaVersion', 'schema_version', 'sweep', 'variables']
```

Inside the `benchmark:` body and inside sweep parameter paths, every section is set to reject unknown fields outright. A typo'd sweep parameter like `phases.profiling.concurency` (one `r`) is caught at validate time — `aiperf config validate` runs the same sweep-expansion pipeline `profile` does and surfaces the error before any compute is spent:

```text
ValidationError: 1 validation error for BenchmarkConfig
phases.0.concurrency.profiling
  Extra inputs are not permitted [type=extra_forbidden, input_value={'concurency': 8}, ...]
```

Use `aiperf config validate <file>` for routine linting. Use `aiperf config expand <file>` when you want to preview the actual variations a sweep will produce (see below). Both catch sweep-path typos; `expand` additionally renders the variation list.

## Substituting environment variables

Use `${VAR}` for required values and `${VAR:default}` for optional ones:

```yaml
benchmark:
  model: ${MODEL_NAME:meta-llama/Llama-3.1-8B-Instruct}
  endpoint:
    url: ${INFERENCE_URL:http://localhost:8000/v1/chat/completions}
    api_key: ${OPENAI_API_KEY}             # required, errors if unset
    timeout: ${TIMEOUT:600.0}
```

Run it across deployments without editing the file:

```bash
MODEL_NAME=meta-llama/Llama-3.1-70B-Instruct \
INFERENCE_URL=http://prod.example.com:8000/v1/chat/completions \
OPENAI_API_KEY=sk-... \
aiperf profile --config benchmark.yaml
```

Strings are auto-coerced to the right type — `TIMEOUT=600.0` becomes a float, `STREAMING=true` becomes a bool.

If a required `${VAR}` is unset, you get a clean error naming the variable, not a silent fallback.

## Reusable values and computed expressions

Define values once at the top, reference them anywhere with `{{ }}` Jinja expressions:

```yaml
variables:
  base_concurrency: 16
  isl_target: 512
  test_duration: 120

benchmark:
  dataset:
    type: synthetic
    entries: "{{ base_concurrency * 10 }}"      # 160
    prompts:
      isl:
        mean: "{{ isl_target }}"
        stddev: "{{ isl_target // 10 }}"        # 51 (integer division)
  phases:
    - name: warmup
      type: concurrency
      concurrency: "{{ base_concurrency // 2 }}"  # 8
      requests: "{{ base_concurrency * 5 }}"      # 80
      exclude_from_results: true
    - name: profiling
      type: gamma
      rate: 30
      duration: "{{ test_duration }}"
      concurrency: "{{ base_concurrency * 4 }}"   # 64
      rate_ramp: "{{ test_duration // 4 }}"       # 30
```

A few things worth knowing:

- Variables can reference other variables, in any order. AIPerf resolves the dependency graph for you.
- Typos like `{{ base_concurrancy }}` raise an error immediately — they don't silently render as an empty string.
- Numeric strings like `"42"` and `"3.14"` are coerced to `int`/`float` automatically, so you don't have to remember which fields expect numbers.
- Env vars run *before* Jinja, so you can do `entries: "{{ base * '${MULT:10}' | int }}"`.

## Multiple phases in one file

A typical benchmark is a quick warmup followed by the real measurement. CLI warmup flags are limited to scalar values per phase shape (`--warmup-request-count`, `--warmup-duration`, `--warmup-concurrency`, `--warmup-request-rate`, `--warmup-arrival-pattern`, and a handful of ramp/grace-period siblings). YAML lets you describe warmup as a full phase with all the same fields available to profiling:

```yaml
benchmark:
  phases:
    - name: warmup
      type: concurrency
      concurrency: 8
      requests: 50
      exclude_from_results: true   # don't pollute the report

    - name: profiling
      type: poisson
      rate: 30.0
      duration: 120
      concurrency: 64
      grace_period: 60             # finish in-flight requests after duration
```

Each phase is a complete arrival pattern in its own right, with its own concurrency, duration, and arrival shape (`concurrency`, `constant`, `poisson`, `gamma`, `fixed_schedule`, `user_centric`, ...).

## Adaptive scale in YAML

Adaptive scale is for single-run boundary discovery. Instead of launching a sweep or separate search trials, AIPerf runs one profiling phase, starts at a low control value, evaluates SLA windows, ramps up while every SLA filter passes, and then sustains near the discovered boundary.

The canonical adaptive shape uses a nested `control` block. Supported v1 control variables are `concurrency`, `prefill_concurrency`, `request_rate`, and `users`. Existing flat concurrency fields such as `control_variable: concurrency` and `min_concurrency` are still accepted as compatibility aliases, but new configs should use `control.variable`, `control.min`, and `control.max`.

```yaml
schemaVersion: "2.0"

benchmark:
  model: meta-llama/Llama-3.1-8B-Instruct
  endpoint:
    url: http://localhost:8000/v1/chat/completions
    type: chat
    streaming: true
  dataset:
    type: synthetic
    entries: 1000
    prompts: {isl: 512, osl: 128}
  phases:
    - name: profiling
      type: concurrency
      concurrency: 200
      prefill_concurrency: 64
      duration: 3600
      adaptive_scale:
        enabled: true
        control:
          variable: prefill_concurrency
          min: 1
          max: 64
        assessment_period: 60
        min_completed_requests: 20
        sustain_duration: 1800
        strategy:
          type: ramp_until_fail
          step_policy: sla_margin
          base_step: 10
          max_step_multiplier: 4
      sla:
        request_latency:
          p95:
            le: 30000
        error_rate:
          avg:
            le: 0.01
```

Adaptive scale rejects fixed ramps on the same variable it controls. For example, do not combine `control.variable: prefill_concurrency` with `prefill_ramp`. Fixed ramps for other variables are allowed.

The CLI exposes a compact sweep-like control flag for the common single-phase case: `--adaptive-scale-control variable:min,max:type`, plus repeated `--adaptive-scale-sla metric:stat:op:threshold` flags. For example: `--adaptive-scale-control "concurrency:1,1000:int" --adaptive-scale-sla "request_latency:p95:le:30000"`. Expanded `--adaptive-control-variable`, `--adaptive-control-min`, and `--adaptive-control-max` flags remain supported for advanced scripting; if expanded `--adaptive-control-max` is omitted, AIPerf infers it from the matching phase target such as `--concurrency`, `--prefill-concurrency`, `--request-rate`, or `--num-users`. Do not mix compact and expanded control forms.

Adaptive scale combines SLA filters with simple AND semantics. A window passes only when every configured filter passes. Step sizing uses the smallest normalized passing margin, so the closest SLA boundary controls the next increase. There are no weights, formulas, or multi-objective scoring in single-run adaptive scale.

For lower-is-better SLA filters such as latency, TTFT, error rate, or cancellation rate, use `lt` or `le`:

```yaml
sla:
  time_to_first_token:
    p95:
      le: 3000
  cancellation_rate:
    avg:
      le: 0.05
```

For higher-is-better SLA filters such as throughput or adaptive-window success ratio, use `gt` or `ge`:

```yaml
sla:
  request_throughput:
    avg:
      ge: 80
  success_rate:
    avg:
      ge: 0.95
```

Quality-qualified adaptive goodput is also higher-is-better, but must be paired with at least one per-request quality filter:

```yaml
sla:
  request_latency:
    p95:
      le: 30000
  goodput:
    avg:
      ge: 20
```

For streaming token quality, pair goodput with TTFT and ITL filters:

```yaml
sla:
  ttft:
    p95:
      le: 2000
  itl:
    p95:
      le: 100
  goodput:
    avg:
      ge: 20
```

### Adaptive SLA metric support

Timing thresholds use milliseconds. This table is the adaptive SLA metric/stat support matrix:

| Metric family | Metric tags and aliases | Supported stats | Window semantics |
| --- | --- | --- | --- |
| E2E latency | `request_latency` | `avg`, `min`, `max`, `p1`, `p5`, `p10`, `p25`, `p50`, `p75`, `p90`, `p95`, `p99` | Per-request latency samples from successful requests. |
| Time to first token | `time_to_first_token`, `ttft` | `avg`, `min`, `max`, `p1`, `p5`, `p10`, `p25`, `p50`, `p75`, `p90`, `p95`, `p99` | Per-request TTFT samples from successful streaming requests. Missing TTFT samples fail TTFT SLA windows. |
| Inter-token latency | `inter_token_latency`, `itl`, `tpot` | `avg`, `min`, `max`, `p1`, `p5`, `p10`, `p25`, `p50`, `p75`, `p90`, `p95`, `p99` | Per-request ITL/TPOT samples from successful streaming requests. Missing ITL samples fail ITL SLA windows. |
| Request throughput | `throughput`, `request_throughput`, `completed_request_throughput` | `avg`, `min`, `max` | Successful completed requests per second in the adaptive window. |
| Output token throughput | `output_token_throughput` | `avg`, `min`, `max` | Output tokens from successful completed requests per second in the adaptive window. |
| Quality goodput | `goodput` | `avg`, `min`, `max` | Quality-passing successful requests per second. Requires at least one request-latency, TTFT, or ITL quality filter. |
| Goodput ratio | `goodput_ratio` | `avg`, `min`, `max` | Quality-passing successful requests divided by total attempts. Requires at least one request-latency, TTFT, or ITL quality filter. |
| Success rate | `success_rate`, `request_success_rate` | `avg`, `min`, `max` | Successful completed requests divided by total attempts. |
| Error rate | `error_rate`, `request_error_rate` | `avg`, `min`, `max` | Failed requests divided by total attempts. |
| Cancellation rate | `cancellation_rate`, `request_cancellation_rate` | `avg`, `min`, `max` | Cancelled requests divided by total attempts. |

For window-level rate and ratio metrics, `avg`, `min`, and `max` currently evaluate the same scalar value for each adaptive window.

Adaptive `users` is valid only on `user_centric` phases. It controls target live simulated user timelines while keeping total QPS fixed; changing users changes per-user turn gap and population pressure rather than acting as another spelling of request rate.

```yaml
benchmark:
  phases:
    - name: profiling
      type: user_centric
      users: 5000
      rate: 100
      duration: 8h
      adaptive_scale:
        enabled: true
        control:
          variable: users
          min: 500
          max: 5000
        assessment_period: 300
        sustain_duration: 6h
      sla:
        time_to_first_token:
          p95:
            le: 3000
        cancellation_rate:
          avg:
            le: 0.10
```

Adaptive scale writes two timing-owned artifacts into the run directory:

```text
adaptive_scale_events.jsonl
adaptive_scale_summary.json
```

These artifacts use schema version 2 and generic control fields such as `control_variable`, `control_value_before`, `control_value_after`, `boundary_value`, `last_passing_value`, and `first_failing_value`. Every `adaptive_window` event includes all evaluated SLA values and the binding constraint. Dynamo-style pollers should gate fault injection on explicit events such as `sustain_started` rather than fixed sleeps.

Use adaptive scale when you want continuous pressure inside one benchmark invocation. Use `sweep` or `adaptive_search` when you want offline multi-run exploration across many independent trials.

## Sweeps in YAML

Sweeps are the killer feature of YAML configs. The CLI only ever supported list-style flags like `--concurrency 8,16,32`. YAML lets you sweep any field, combine multiple parameters, or pull from a quasi-random distribution.

Here's a 3 × 3 = 9-run grid sweep over input length and request rate:

```yaml
schemaVersion: "2.0"

sweep:
  type: grid
  parameters:
    datasets.default.prompts.isl: [128, 512, 2048]
    rate: [10.0, 30.0, 50.0]

benchmark:
  model: meta-llama/Llama-3.1-8B-Instruct
  endpoint:
    url: http://localhost:8000/v1/chat/completions
  dataset:
    type: synthetic
    entries: 500
    prompts: {isl: 512, osl: 128}      # isl is overridden by the sweep
  phases:
    - name: profiling
      type: poisson
      rate: 20.0                        # overridden by the sweep
      duration: 120
```

The `parameters:` keys are dot-paths into the `benchmark:` body. For lists, the second segment is the entry's `name`:

- `phases.profiling.rate` → the phase named `profiling`, field `rate`
- `datasets.default.prompts.isl` → the dataset named `default` (the singular `dataset:` shorthand auto-names it `default`)

The 12 most-swept phase fields also have bare-name sugar: `concurrency`, `prefill_concurrency`, `rate`, `requests`, `duration`, `sessions`, `users`, `smoothness`, `grace_period`, `concurrency_ramp`, `prefill_ramp`, `rate_ramp`. Each expands to `phases.profiling.<name>` (resolves to the unique non-warmup phase). The two forms are equivalent — see [Bare-Name Aliases](sweeps.md#bare-name-aliases-for-common-phase-fields).

Other sweep modes available in YAML:

- **`zip`** — pair parameters lockstep instead of cross-product (useful for paired ISL/OSL).
- **`scenarios`** — hand-curated named workload profiles, each a deep-merge over the base body.
- **`sobol`** / **`latin_hypercube`** — quasi-random space-filling samples.
- **`adaptive_search`** — Bayesian optimization over multiple objectives.

For a guided picker, see [Parameter Sweeps — Choosing a sweep mode](sweeps.md#choosing-a-sweep-mode).

You can preview what a sweep will run *before* spending any compute:

```bash
aiperf config expand sweep.yaml             # lists the variations
aiperf config expand sweep.yaml --full      # dumps each variation's full body
aiperf config expand sweep.yaml --index 2 --full   # inspect one variation
```

## Repeating a benchmark for confidence intervals

Running the same benchmark several times and taking the mean ± confidence interval is a separate envelope-level setting:

```yaml
multi_run:
  num_runs: 3
  cooldown_seconds: 15.0
  confidence_level: 0.95
  set_consistent_seed: true
  disable_warmup_after_first: true   # warmup once, reuse the warm cache
```

`multi_run` and `sweep` compose: a 9-variation grid × 3 runs = 27 benchmarks, with confidence intervals computed *per variation*. See [Multi-Run Confidence Reporting](multi-run-confidence.md) for what the report looks like.

## CLI helpers for working with configs

Three commands cover the common authoring tasks:

```bash
# Scaffold from a template (27+ bundled, covering most workloads)
aiperf config init --list                          # browse
aiperf config init --search sweep                  # search by keyword
aiperf config init --template goodput_slo \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --url http://localhost:8000/v1/chat/completions \
  --output benchmark.yaml

# Lint a config without running it
aiperf config validate benchmark.yaml

# Preview sweep variations
aiperf config expand sweep.yaml --full
```

`validate` runs the same load pipeline `profile` does, so anything wrong shows up here — typos, missing required fields, sweep paths that don't resolve, env vars that aren't set.

## Mixing YAML with CLI flags

YAML configs and CLI flags are not either/or. Flags overlay whatever's in the file:

```bash
aiperf profile --config benchmark.yaml \
  --concurrency 32 \
  --artifact-dir ./run-2026-05-09
```

This loads `benchmark.yaml` as the base, then overrides the *profiling* phase's `concurrency` with `32` and the artifact directory with the new path. (CLI loadgen flags overlay onto the phase named `profiling` — they don't broadcast to every named phase, so multi-phase configs need YAML edits to tweak warmup or other phases.) Useful when most of your config is stable but you want to tweak one knob from a script or CI job.

The precedence order, lowest to highest:

1. Defaults baked into AIPerf
2. Values in the YAML file
3. Explicit CLI flags

## Where to go next

- **[Bundled templates](https://github.com/ai-dynamo/aiperf/tree/main/src/aiperf/config/templates)** — 27+ ready-to-run examples grouped by category (`Getting Started`, `Load Testing`, `Datasets`, `Sweep & Multi-Run`, `Advanced`, `Multimodal`, `Specialized Endpoints`).
- **[Parameter Sweeps — Choosing a sweep mode](sweeps.md#choosing-a-sweep-mode)** — picker for grid vs. zip vs. scenarios vs. Sobol vs. adaptive search.
- **[Parameter Sweeps](sweeps.md)** — deeper dive on sweep mechanics, output structure, and Pareto analysis.
- **[Multi-Run Confidence Reporting](multi-run-confidence.md)** — how `multi_run` propagates through reports.
- **[CLI Options](../cli-options.md)** — every CLI flag, in case you want to overlay one onto a YAML config.
