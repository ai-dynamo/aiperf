---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: "Distributions: Statistical Workload Modeling"
---

# Distributions: Statistical Workload Modeling

Real inference traffic is not uniform. Token lengths, think times, and image dimensions all follow statistical patterns that vary by workload. A chatbot receives mostly short queries with occasional long ones. A RAG pipeline generates medium-length prompts with high variance. A summarization service processes long documents and produces short outputs.

AIPerf's distribution system lets you describe these patterns declaratively in YAML. Instead of hardcoding a single ISL value, you specify the statistical shape of your workload, and AIPerf samples from that distribution for each request. This produces benchmarks that stress your server the way real traffic does, revealing queuing effects, memory pressure, and scheduling behavior that fixed-length tests miss.

## Quick Reference

AIPerf supports exactly **five** distribution types. The type is auto-detected from the shape of the fields you provide -- you do not need a `type:` key (it is accepted but optional).

| Type | YAML Syntax | Detected By | Use Case |
|------|-------------|-------------|----------|
| Fixed | `isl: 512` or `{value: 512}` | scalar, or `value` | Baselines, controlled experiments |
| Normal | `{mean: 512, stddev: 100}` | `stddev` (or `mean` alone) | General-purpose symmetric variance |
| LogNormal | `{mean: 512, median: 400}` | `median` | Right-skewed production token distributions |
| Multimodal | `{peaks: [{...}, {...}]}` | `peaks` | Bimodal/multi-modal workloads |
| Empirical | `{points: [{value, weight}, ...]}` | `points` | Replaying production histograms |
| Bounds | `min: N, max: M` | (optional on any type) | Clamps samples post-draw into a safe range |

There is no `uniform`, `exponential`, `zipf`, or `mixture` type. Multi-population workloads are expressed with `multimodal` (`peaks:`), and discrete replay with `empirical` (`points:`).

## Auto-Detection and the Optional `type:` Key

The distribution type is inferred structurally from the keys you provide, checked in this order:

1. A bare scalar (`isl: 512`) -> **Fixed**
2. `peaks:` present -> **Multimodal**
3. `points:` present -> **Empirical**
4. `median:` present -> **LogNormal**
5. `stddev:` present -> **Normal**
6. `value:` present -> **Fixed**
7. `mean:` alone -> **Normal** (stddev defaults to 0, i.e. deterministic)

You may add an explicit `type:` key (`{type: normal, mean: 512, stddev: 100}`) for clarity, but it is never required. An unknown `type:` (e.g. `type: uniform`) is rejected at config-load time.

## Scalar Shorthand

Any field that accepts a distribution also accepts a bare number, which is automatically converted to a Fixed distribution:

```yaml
# These are equivalent:
prompts:
  isl: 512
  osl: 128

prompts:
  isl: {value: 512}
  osl: {value: 128}
```

## Base Distributions

### Fixed

```yaml
prompts:
  isl: {value: 1024}
  osl: {value: 256}
```

Returns the same value on every sample. Use this for controlled experiments where you need exact token counts, or as a baseline to compare against variable distributions. In practice, you will almost always use the scalar shorthand (`isl: 1024`) instead of the explicit form.

**Parameters:**
- `value` -- The constant value returned on every sample.

### Normal

```yaml
prompts:
  isl: {mean: 512, stddev: 100}
  osl: {mean: 256, stddev: 50}
```

Gaussian distribution centered on `mean` with spread controlled by `stddev`. Values are truncated at zero (negative samples are drawn as positive). This is the most common choice for adding realistic variance to token counts without making strong assumptions about the shape.

**Parameters:**
- `mean` -- Center of the distribution. Must be `>= 0`. For token counts, your target number of tokens.
- `stddev` -- Standard deviation. Controls the spread. Defaults to `0` (deterministic). A `stddev` of ~20% of the mean produces moderate variance.

**When to use:** General-purpose benchmarking where you want variance but expect a symmetric spread around the target.

### LogNormal

```yaml
prompts:
  isl: {mean: 512, median: 400}
  osl: {mean: 256, median: 220}
```

Produces right-skewed positive values. Most samples cluster near the median, but a long tail extends to much larger values, pulling the mean above the median. This closely matches real-world token length distributions observed in production, where most requests are short or medium but some are very long.

The shape is controlled by the **ratio of `mean` to `median`**: the larger the gap, the heavier the right tail. When `mean == median` the distribution is deterministic. `median` must be `<= mean`. Internally AIPerf derives the log-space parameters (`sigma = sqrt(2 * ln(mean / median))`, `mu = ln(median)`) so the output distribution has the mean you specify.

**Parameters:**
- `mean` -- Desired mean of the output distribution (must be `> 0`).
- `median` -- Desired median (must be `> 0` and `<= mean`). Lower median relative to the mean produces more right skew. `median = 0.9 * mean` is mild skew, `0.8 * mean` is moderate, `0.5 * mean` is heavy.

**When to use:** Modeling production LLM traffic where most prompts are moderate length but a fraction are much longer.

## Multi-Population and Discrete Distributions

### Multimodal

```yaml
prompts:
  isl:
    peaks:
      - {mean: 128, stddev: 30, weight: 70}
      - {mean: 2048, median: 1800, weight: 30}
```

A weighted mixture of two or more peaks. Each sample first selects a peak based on relative weights, then draws from that peak's distribution. This is how you model bimodal or multi-modal workloads where traffic comes from distinct populations.

Each peak is written **inline**: the distribution fields (`mean`/`stddev`, `mean`/`median`, `value`, etc.) live directly in the peak dict alongside an optional `weight`. Weights are relative and normalized internally -- `weight: 70` / `weight: 30` produces a 70/30 split, and so does `weight: 7` / `weight: 3`. Omit `weight` for an equal split (it defaults to `1.0`).

Requires at least 2 peaks. Peaks can be any base distribution, and each peak may carry its own `min`/`max` bounds:

```yaml
# Three-tier workload: chatbot, RAG, and batch summarization
prompts:
  isl:
    peaks:
      - {mean: 64, stddev: 15, weight: 50}
      - {mean: 1024, median: 820, min: 256, max: 4096, weight: 35}
      - {mean: 8192, stddev: 500, weight: 15}
```

### Empirical

```yaml
prompts:
  isl:
    points:
      - {value: 128, weight: 40}
      - {value: 512, weight: 35}
      - {value: 2048, weight: 20}
      - {value: 8192, weight: 5}
```

Discrete distribution that samples from a fixed set of weighted values. Each sample returns exactly one of the listed values, chosen by relative weight. This is the right choice when you have production histogram data and want to replay the exact distribution of token lengths observed in real traffic.

Weights default to `1.0` if omitted, producing uniform selection across values:

```yaml
# Equal probability for each bucket
prompts:
  isl:
    points:
      - {value: 128}
      - {value: 256}
      - {value: 512}
      - {value: 1024}
```

### Bounds (min/max)

```yaml
prompts:
  isl: {mean: 1024, stddev: 500, min: 64, max: 4096}
  osl: {mean: 4096, median: 2048, max: 8192}
```

`min:` and `max:` are optional fields on **every** distribution type. Samples that fall outside the bounds are clamped (not redrawn), preventing wide-spread distributions from producing values that exceed your model's context window or go below meaningful minimums. Either or both may be specified; bounds compose with all other distribution fields without nesting.

## Real-World Workload Recipes

### Chatbot Traffic

Short prompts with low variance. Most user queries are brief questions, with a small fraction of longer follow-ups. Think time between turns is right-skewed, so it is modeled with a lognormal.

```yaml
benchmark:
  datasets:
    - name: chatbot
      type: synthetic
      entries: 1000
      prompts:
        isl: {mean: 64, stddev: 15}
        osl: {mean: 128, stddev: 30}
      turns: {mean: 2, stddev: 1}
      turn_delay: {mean: 3000, median: 2000} # ~3s avg, right-skewed think time
```

### RAG Pipeline

Medium ISL with high variance from variable-length retrieved context. LogNormal models the skew from context injection: most retrievals add moderate context, but some include many long passages.

```yaml
benchmark:
  datasets:
    - name: rag
      type: synthetic
      entries: 500
      prompts:
        isl: {mean: 1024, median: 820}
        osl: {mean: 256, stddev: 50}
```

### Summarization Service

Long input documents, short output summaries. Clamped to stay within the model's context window and guarantee a minimum output length.

```yaml
benchmark:
  datasets:
    - name: summarization
      type: synthetic
      entries: 500
      prompts:
        isl: {mean: 4096, median: 3600, min: 1024, max: 16384}
        osl: {mean: 256, stddev: 80, min: 64, max: 512}
```

### Production Traffic Replay (Bimodal)

Two distinct user populations hit the same endpoint: interactive chat (high volume, short) and batch analysis (lower volume, long). The multimodal peaks capture both modes.

```yaml
benchmark:
  datasets:
    - name: production_bimodal
      type: synthetic
      entries: 2000
      prompts:
        isl:
          peaks:
            - {mean: 128, stddev: 30, weight: 65}
            - {mean: 2048, median: 1600, weight: 35}
        osl:
          peaks:
            - {mean: 96, stddev: 20, weight: 65}
            - {mean: 512, stddev: 100, weight: 35}
```

### Multi-Tier Service (Empirical from Production Data)

When you have histogram data from production logs, use empirical distributions to replay the exact observed distribution. This example models a service where token lengths cluster at specific tiers corresponding to different API consumers.

```yaml
benchmark:
  datasets:
    - name: production_replay
      type: synthetic
      entries: 2000
      prompts:
        isl:
          points:
            - {value: 64, weight: 15}  # health checks and pings
            - {value: 256, weight: 30}  # mobile app short queries
            - {value: 512, weight: 25}  # web app standard queries
            - {value: 1024, weight: 15} # web app with context
            - {value: 2048, weight: 10} # internal batch jobs
            - {value: 8192, weight: 5}  # document processing
        osl:
          points:
            - {value: 32, weight: 15}
            - {value: 128, weight: 35}
            - {value: 256, weight: 30}
            - {value: 512, weight: 15}
            - {value: 1024, weight: 5}
```

## Where Distributions Are Used

Every config field listed below accepts a distribution -- meaning any of the five distribution types, or a bare scalar.

### Token Lengths

| Field | Path | Description |
|-------|------|-------------|
| `isl` | `datasets.<name>.prompts.isl` | Input sequence length in tokens |
| `osl` | `datasets.<name>.prompts.osl` | Output sequence length (max_completion_tokens) |

### Multi-Turn Conversations

| Field | Path | Description |
|-------|------|-------------|
| `turns` | `datasets.<name>.turns` | Number of request-response turns per conversation |
| `turn_delay` | `datasets.<name>.turn_delay` | Delay in milliseconds between consecutive turns |

### Images (Multimodal)

| Field | Path | Description |
|-------|------|-------------|
| `width` | `datasets.<name>.images.width` | Image width in pixels |
| `height` | `datasets.<name>.images.height` | Image height in pixels |

### Audio (Multimodal)

| Field | Path | Description |
|-------|------|-------------|
| `length` | `datasets.<name>.audio.length` | Audio duration in seconds |

### Rankings/Reranking

| Field | Path | Description |
|-------|------|-------------|
| `passages` | `datasets.<name>.rankings.passages` | Number of passages per ranking request |
| `passage_tokens` | `datasets.<name>.rankings.passage_tokens` | Token length per passage |
| `query_tokens` | `datasets.<name>.rankings.query_tokens` | Token length for the query |

## Distributions and Sweeps

Sweep variables use dot-notation paths to override any config field, including distribution parameters. This lets you systematically explore how workload shape affects server performance.

### Grid Sweep over ISL Distribution Mean

Sweep the mean of a normal ISL distribution while holding the spread (`stddev`) constant:

```yaml
benchmark:
  datasets:
    - name: profiling
      type: synthetic
      entries: 500
      prompts:
        isl: {mean: 512, stddev: 100}
        osl: {mean: 256, stddev: 50}

  phases:
    - name: profiling
      type: poisson
      rate: 20.0
      duration: 120
      concurrency: 64

sweep:
  type: grid
  variables:
    benchmark.datasets.profiling.prompts.isl.mean: [128, 512, 2048, 8192]
```

This produces 4 benchmark runs, each with a different ISL mean. The `stddev: 100` spread is preserved across all runs.

> **Sweeping a lognormal:** because a lognormal requires `median <= mean`, sweeping only `.mean` below the fixed `median` fails validation. To sweep a lognormal's location while holding its shape, sweep `.mean` and `.median` together in lockstep with a `zip` sweep.

### Scenario Sweep with Different Distribution Shapes

Compare how the server handles different workload shapes at the same average ISL:

```yaml
benchmark:
  datasets:
    - name: profiling
      type: synthetic
      entries: 500
      prompts:
        isl: 512
        osl: 128

  phases:
    - name: profiling
      type: poisson
      rate: 20.0
      duration: 120
      concurrency: 64

sweep:
  type: scenarios
  runs:
    - name: fixed_baseline
      benchmark:
        datasets:
          - name: profiling
            prompts:
              isl: 512

    - name: normal_moderate
      benchmark:
        datasets:
          - name: profiling
            prompts:
              isl: {mean: 512, stddev: 100}

    - name: lognormal_skewed
      benchmark:
        datasets:
          - name: profiling
            prompts:
              isl: {mean: 512, median: 300}

    - name: bimodal_production
      benchmark:
        datasets:
          - name: profiling
            prompts:
              isl:
                peaks:
                  - {mean: 128, stddev: 30, weight: 70}
                  - {mean: 2048, stddev: 200, weight: 30}
```

All four scenarios target a similar average ISL, but the variance and shape differ. Comparing results reveals how your server handles workload variance.

## Sampling and Reproducibility

### How `random_seed` Works

When `random_seed` is set at the top level, AIPerf initializes its random number generator deterministically. Every distribution in the config draws from this same seeded generator, producing identical sequences of samples across runs:

```yaml
random_seed: 42

benchmark:
  datasets:
    - name: profiling
      type: synthetic
      entries: 500
      prompts:
        isl: {mean: 512, median: 400}
        osl: {mean: 256, stddev: 50}
```

Running this config twice produces the same set of 500 (ISL, OSL) pairs both times, assuming no other configuration changes.

### Per-Dataset Seed Override

Individual datasets can override the global seed. This is useful when you want one dataset to be deterministic for A/B comparison while another varies:

```yaml
random_seed: 42

benchmark:
  datasets:
    - name: stable_baseline
      type: synthetic
      entries: 500
      random_seed: 100 # Always the same
      prompts:
        isl: {mean: 512, stddev: 100}
        osl: 128

    - name: variable_workload
      type: synthetic
      entries: 500
    # Uses global seed (42) -- same across runs with same config
      prompts:
        isl: {mean: 1024, median: 800}
        osl: {mean: 256, stddev: 50}
```

### Deterministic Benchmarks

For reproducible A/B testing (comparing two server configurations, two model versions, or before/after an optimization), set `random_seed` so both runs receive identical request sequences:

```yaml
random_seed: 42

benchmark:
  datasets:
    - name: profiling
      type: synthetic
      entries: 1000
      prompts:
        isl: {mean: 512, median: 400}
        osl: {mean: 256, stddev: 50}

multi_run:
  num_runs: 5
  set_consistent_seed: true  # Each run uses the same seed
```

With `set_consistent_seed: true`, every run within a multi-run benchmark uses the same seed, ensuring the same dataset is generated each time. Combined with the same arrival pattern, this produces directly comparable results across runs.

Without `random_seed`, AIPerf uses system entropy and each run produces a different sample sequence. This is appropriate for measuring aggregate behavior but not for controlled comparisons.

## Related Documentation

- [Sequence Length Distributions](./sequence-distributions.md) -- CLI-based sequence distribution for mixed ISL/OSL pairs
- [Arrival Patterns](./arrival-patterns.md) -- Statistical control over request inter-arrival times
- [Multi-Run Confidence](./multi-run-confidence.md) -- Statistical confidence across repeated runs
- [Warmup Phase](./warmup.md) -- Configuring warmup before profiling
