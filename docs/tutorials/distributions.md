---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: "Distributions: Statistical Workload Modeling"
---

# Distributions: Statistical Workload Modeling

Real inference traffic is not uniform. Token lengths, think times, and image dimensions all follow statistical patterns that vary by workload. A chatbot receives mostly short queries with occasional long ones. A RAG pipeline generates medium-length prompts with high variance. A summarization service processes long documents and produces short outputs.

AIPerf's distribution system lets you describe these patterns declaratively in YAML. Instead of hardcoding a single ISL value, you specify the statistical shape of your workload, and AIPerf samples from that distribution for each request. This produces benchmarks that stress your server the way real traffic does, revealing queuing effects, memory pressure, and scheduling behavior that fixed-length tests miss.

## Quick Reference

| Type | YAML Syntax | Use Case | Description |
|------|-------------|----------|-------------|
| Fixed | `isl: 512` | Baselines, controlled experiments | Constant value on every sample |
| Normal | `{type: normal, mean: 512, stddev: 100}` | General-purpose variance | Gaussian spread around a center |
| LogNormal | `{type: lognormal, mean: 512, sigma: 0.5}` | Production token distributions | Right-skewed, most values near mean with long tail |
| Uniform | `{type: uniform, min: 128, max: 2048}` | Stress testing across a range | Flat probability between bounds |
| Exponential | `{type: exponential, mean: 2000}` | Think times, inter-event delays | Memoryless, always positive |
| Zipf | `{type: zipf, alpha: 1.5, scale: 128}` | Power-law popularity patterns | Heavy-tailed: few values dominate |
| Mixture | `{type: mixture, components: [...]}` | Bimodal/multi-modal workloads | Weighted combination of sub-distributions |
| Clamped | `{type: clamped, distribution: {...}, max: 4096}` | Hardware-safe bounds | Wraps any distribution with min/max limits |
| Empirical | `{type: empirical, points: [...]}` | Replaying production histograms | Discrete weighted values |

## Scalar Shorthand

Any field that accepts a distribution also accepts a bare number. A bare number is automatically converted to a `FixedDistribution`:

```yaml
# These are equivalent:
prompts:
  isl: 512
  osl: 128

prompts:
  isl: {type: fixed, value: 512}
  osl: {type: fixed, value: 128}
```

This is the only shorthand. All other distribution types require an explicit `type` field in the YAML dict.

## Base Distributions

### Fixed

```yaml
prompts:
  isl: {type: fixed, value: 1024}
  osl: {type: fixed, value: 256}
```

Returns the same value on every sample. Use this for controlled experiments where you need exact token counts, or as a baseline to compare against variable distributions. In practice, you will almost always use the scalar shorthand (`isl: 1024`) instead of the explicit form.

**Parameters:**
- `value` -- The constant value returned on every sample.

### Normal

```yaml
prompts:
  isl: {type: normal, mean: 512, stddev: 100}
  osl: {type: normal, mean: 256, stddev: 50}
```

Gaussian distribution centered on `mean` with spread controlled by `stddev`. Values are truncated at zero (negative samples are redrawn as positive). This is the most common choice for adding realistic variance to token counts without making strong assumptions about the shape.

**Parameters:**
- `mean` -- Center of the distribution. For token counts, your target number of tokens.
- `stddev` -- Standard deviation. Controls the spread. Defaults to `0` (deterministic). A `stddev` of ~20% of the mean produces moderate variance.

**When to use:** General-purpose benchmarking where you want variance but expect a symmetric spread around the target.

### LogNormal

```yaml
prompts:
  isl: {type: lognormal, mean: 512, sigma: 0.8}
  osl: {type: lognormal, mean: 256, sigma: 0.5}
```

Produces right-skewed positive values. Most samples cluster near the mean, but a long tail extends to much larger values. This closely matches real-world token length distributions observed in production, where most requests are short or medium but some are very long.

The `mean` parameter is the desired output mean (not the log-space mean). AIPerf internally computes the log-space parameters so the distribution has the expected value you specify.

**Parameters:**
- `mean` -- Desired mean of the output distribution (must be > 0).
- `sigma` -- Shape parameter (log-space standard deviation). `0` means deterministic. Higher values produce more right skew: `0.3` is mild skew, `0.8` is moderate, `1.5` is extreme.

**When to use:** Modeling production LLM traffic where most prompts are moderate length but a fraction are much longer.

### Uniform

```yaml
prompts:
  isl: {type: uniform, min: 128, max: 2048}
  osl: {type: uniform, min: 64, max: 512}
```

Equal probability for any value between `min` and `max`. Every length in the range is equally likely. This is useful for stress testing across a full range of input sizes, ensuring your server handles both short and long sequences.

**Parameters:**
- `min` -- Lower bound (inclusive).
- `max` -- Upper bound (inclusive). Must be >= `min`.

**When to use:** Sweeping across a range of sequence lengths to find where performance degrades, or when you have no prior information about workload shape.

### Exponential

```yaml
turn_delay: {type: exponential, mean: 2000}
```

Memoryless distribution that always produces positive values. The most likely values are near zero, with an exponentially decreasing tail. This is the natural model for waiting times and inter-event delays.

**Parameters:**
- `mean` -- Mean (expected value) of the distribution (must be > 0). For example, `mean: 2000` produces a distribution where the average delay is 2000ms, but individual delays range from near-zero to several times the mean.

**When to use:** Modeling user think time between multi-turn conversation rounds, or any delay where events are independent and memoryless.

### Zipf

```yaml
prompts:
  isl: {type: zipf, alpha: 2.0, scale: 128}
```

Power-law distribution over positive integers. Rank `k` has probability proportional to `1/k^alpha`. A few values dominate while a long tail of rare values exists. The raw rank is mapped to output values via `offset + rank * scale`.

**Parameters:**
- `alpha` -- Exponent parameter (must be > 1). Higher values concentrate more probability on the first few ranks. `1.5` has a heavy tail, `3.0` is sharply peaked.
- `scale` -- Multiplied by rank to produce the output value. Default `1.0`. With `scale: 128`, rank 1 produces 128, rank 2 produces 256, etc.
- `offset` -- Added to the scaled rank. Default `0.0`.

**When to use:** Modeling popularity patterns where a few common prompt lengths account for most traffic (e.g., a small set of template-driven queries dominate).

## Composable Distributions

Composable distributions wrap or combine other distributions. They accept any distribution type as input, including other composable distributions, allowing arbitrary nesting.

### Mixture

```yaml
prompts:
  isl:
    type: mixture
    components:
      - distribution: {type: normal, mean: 128, stddev: 30}
        weight: 70
      - distribution: {type: lognormal, mean: 2048, sigma: 0.3}
        weight: 30
```

Weighted combination of sub-distributions. Each sample first selects a component based on relative weights, then draws from that component's distribution. This is how you model bimodal or multi-modal workloads where traffic comes from distinct populations.

Weights are relative and normalized internally. `weight: 70` and `weight: 30` produce 70/30 split. `weight: 7` and `weight: 3` produce the same result.

Requires at least 2 components. Components can themselves be any distribution type, including other mixtures, clamped distributions, or empirical distributions:

```yaml
# Three-tier workload: chatbot, RAG, and batch summarization
prompts:
  isl:
    type: mixture
    components:
      - distribution: {type: normal, mean: 64, stddev: 15}
        weight: 50
      - distribution:
          type: clamped
          distribution: {type: lognormal, mean: 1024, sigma: 0.4}
          min: 256
          max: 4096
        weight: 35
      - distribution: {type: normal, mean: 8192, stddev: 500}
        weight: 15
```

### Clamped

```yaml
prompts:
  isl:
    type: clamped
    distribution: {type: normal, mean: 1024, stddev: 500}
    min: 64
    max: 4096
```

Wraps any distribution with hard minimum and/or maximum bounds. Values that fall outside the bounds are clamped (not redrawn). This prevents distributions with wide spread from producing values that exceed your model's context window or go below meaningful minimums.

At least one of `min` or `max` must be specified. You can use just `max` to cap without a floor, or just `min` to enforce a minimum:

```yaml
# Cap ISL at the model's context window, no lower bound
prompts:
  isl:
    type: clamped
    distribution: {type: lognormal, mean: 2048, sigma: 0.8}
    max: 8192

# Enforce minimum OSL without an upper bound
prompts:
  osl:
    type: clamped
    distribution: {type: exponential, mean: 200}
    min: 32
```

### Empirical

```yaml
prompts:
  isl:
    type: empirical
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
    type: empirical
    points:
      - {value: 128}
      - {value: 256}
      - {value: 512}
      - {value: 1024}
```

## Real-World Workload Recipes

### Chatbot Traffic

Short prompts with low variance. Most user queries are brief questions, with a small fraction of longer follow-ups.

```yaml
datasets:
  chatbot:
    type: synthetic
    entries: 1000
    prompts:
      isl: {type: normal, mean: 64, stddev: 15}
      osl: {type: normal, mean: 128, stddev: 30}
    turns: {type: normal, mean: 2, stddev: 1}
    turn_delay: {type: exponential, mean: 3000}  # 3s avg think time
```

### RAG Pipeline

Medium ISL with high variance from variable-length retrieved context. LogNormal models the skew from context injection: most retrievals add moderate context, but some include many long passages.

```yaml
datasets:
  rag:
    type: synthetic
    entries: 500
    prompts:
      isl: {type: lognormal, mean: 1024, sigma: 0.6}
      osl: {type: normal, mean: 256, stddev: 50}
```

### Summarization Service

Long input documents, short output summaries. Clamped to stay within the model's context window and guarantee a minimum output length.

```yaml
datasets:
  summarization:
    type: synthetic
    entries: 500
    prompts:
      isl:
        type: clamped
        distribution: {type: lognormal, mean: 4096, sigma: 0.5}
        min: 1024
        max: 16384
      osl:
        type: clamped
        distribution: {type: normal, mean: 256, stddev: 80}
        min: 64
        max: 512
```

### Production Traffic Replay (Bimodal)

Two distinct user populations hit the same endpoint: interactive chat (high volume, short) and batch analysis (lower volume, long). The mixture captures both modes.

```yaml
datasets:
  production_bimodal:
    type: synthetic
    entries: 2000
    prompts:
      isl:
        type: mixture
        components:
          - distribution: {type: normal, mean: 128, stddev: 30}
            weight: 65
          - distribution: {type: lognormal, mean: 2048, sigma: 0.4}
            weight: 35
      osl:
        type: mixture
        components:
          - distribution: {type: normal, mean: 96, stddev: 20}
            weight: 65
          - distribution: {type: normal, mean: 512, stddev: 100}
            weight: 35
```

### Multi-Tier Service (Empirical from Production Data)

When you have histogram data from production logs, use empirical distributions to replay the exact observed distribution. This example models a service where token lengths cluster at specific tiers corresponding to different API consumers.

```yaml
datasets:
  production_replay:
    type: synthetic
    entries: 2000
    prompts:
      isl:
        type: empirical
        points:
          - {value: 64, weight: 15}    # health checks and pings
          - {value: 256, weight: 30}    # mobile app short queries
          - {value: 512, weight: 25}    # web app standard queries
          - {value: 1024, weight: 15}   # web app with context
          - {value: 2048, weight: 10}   # internal batch jobs
          - {value: 8192, weight: 5}    # document processing
      osl:
        type: empirical
        points:
          - {value: 32, weight: 15}
          - {value: 128, weight: 35}
          - {value: 256, weight: 30}
          - {value: 512, weight: 15}
          - {value: 1024, weight: 5}
```

## Where Distributions Are Used

Every config field listed below accepts a `SamplingDistribution` -- meaning any of the 9 distribution types, or a bare scalar.

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

### Augmentation (Composed Datasets)

| Field | Path | Description |
|-------|------|-------------|
| `osl` | `datasets.<name>.augment.osl` | Output sequence length for augmented records |
| `prefix length` | `datasets.<name>.augment.prefix.length` | Token length of synthetic prefix |
| `suffix length` | `datasets.<name>.augment.suffix.length` | Token length of synthetic suffix |

## Distributions and Sweeps

Sweep variables use dot-notation paths to override any config field, including distribution parameters. This lets you systematically explore how workload shape affects server performance.

### Grid Sweep over ISL Distribution Mean

Sweep the mean of a lognormal ISL distribution while holding all other parameters constant:

```yaml
datasets:
  profiling:
    type: synthetic
    entries: 500
    prompts:
      isl: {type: lognormal, mean: 512, sigma: 0.5}
      osl: {type: normal, mean: 256, stddev: 50}

phases:
  profiling:
    type: poisson
    rate: 20.0
    duration: 120
    concurrency: 64

sweep:
  type: grid
  variables:
    datasets.profiling.prompts.isl.mean: [128, 512, 2048, 8192]
```

This produces 4 benchmark runs, each with a different ISL mean. The lognormal shape (`sigma: 0.5`) is preserved across all runs.

### Scenario Sweep with Different Distribution Shapes

Compare how the server handles different workload shapes at the same average ISL:

```yaml
datasets:
  profiling:
    type: synthetic
    entries: 500
    prompts:
      isl: 512
      osl: 128

phases:
  profiling:
    type: poisson
    rate: 20.0
    duration: 120
    concurrency: 64

sweep:
  type: scenarios
  runs:
    - name: fixed_baseline
      datasets:
        profiling:
          prompts:
            isl: 512

    - name: normal_moderate
      datasets:
        profiling:
          prompts:
            isl: {type: normal, mean: 512, stddev: 100}

    - name: lognormal_skewed
      datasets:
        profiling:
          prompts:
            isl: {type: lognormal, mean: 512, sigma: 0.8}

    - name: bimodal_production
      datasets:
        profiling:
          prompts:
            isl:
              type: mixture
              components:
                - distribution: {type: normal, mean: 128, stddev: 30}
                  weight: 70
                - distribution: {type: normal, mean: 2048, stddev: 200}
                  weight: 30
```

All four scenarios target similar average ISL, but the variance and shape differ. Comparing results reveals how your server handles workload variance.

## Sampling and Reproducibility

### How `random_seed` Works

When `random_seed` is set at the top level, AIPerf initializes its random number generator deterministically. Every distribution in the config draws from this same seeded generator, producing identical sequences of samples across runs:

```yaml
random_seed: 42

datasets:
  profiling:
    type: synthetic
    entries: 500
    prompts:
      isl: {type: lognormal, mean: 512, sigma: 0.5}
      osl: {type: normal, mean: 256, stddev: 50}
```

Running this config twice produces the same set of 500 (ISL, OSL) pairs both times, assuming no other configuration changes.

### Per-Dataset Seed Override

Individual datasets can override the global seed. This is useful when you want one dataset to be deterministic for A/B comparison while another varies:

```yaml
random_seed: 42

datasets:
  stable_baseline:
    type: synthetic
    entries: 500
    random_seed: 100  # Always the same
    prompts:
      isl: {type: normal, mean: 512, stddev: 100}
      osl: 128

  variable_workload:
    type: synthetic
    entries: 500
    # Uses global seed (42) -- same across runs with same config
    prompts:
      isl: {type: lognormal, mean: 1024, sigma: 0.6}
      osl: {type: normal, mean: 256, stddev: 50}
```

### Deterministic Benchmarks

For reproducible A/B testing (comparing two server configurations, two model versions, or before/after an optimization), set `random_seed` so both runs receive identical request sequences:

```yaml
random_seed: 42

datasets:
  profiling:
    type: synthetic
    entries: 1000
    prompts:
      isl: {type: lognormal, mean: 512, sigma: 0.5}
      osl: {type: normal, mean: 256, stddev: 50}

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
