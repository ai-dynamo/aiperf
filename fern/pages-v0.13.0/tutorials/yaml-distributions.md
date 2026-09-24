---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Sampling Distributions in YAML Configs
---

# Sampling Distributions in YAML Configs

Several fields in an AIPerf YAML config — input/output token lengths, conversation turn counts, turn delays, image dimensions, audio length, and ranking passage counts — accept a *sampling distribution* instead of a single number. This tutorial covers all five distribution shapes AIPerf supports, the auto-detection rules that pick between them, and the optional `min:`/`max:` clamps that compose with any of them.

If you only ever write `isl: 512`, you've already used a distribution — that scalar is the shorthand for a `FixedDistribution`. Everything below extends from there.

## Where distributions show up

Any field in a YAML config typed as a sampling distribution accepts the full set of shapes described in this tutorial. The current list:

| Field | Section | What it controls |
|---|---|---|
| `isl` | `dataset.prompts` (and shorthand at `dataset.isl`) | Input sequence length, in tokens |
| `osl` | `dataset.prompts` and `dataset.osl` shorthand; also on file datasets | Output sequence length, in tokens |
| `turns` | `dataset` | Number of request/response turns per conversation |
| `turn_delay` | `dataset` | Delay between turns, in milliseconds |
| `width`, `height` | `dataset.images` | Synthetic image dimensions, in pixels |
| `length` | `dataset.audio` | Synthetic audio duration, in seconds |
| `passages`, `passage_tokens`, `query_tokens` | `dataset.rankings` | Rankings/reranking endpoint shapes |

Wherever you see `{mean: ..., stddev: ...}` in a template, you can swap in any other shape from this page.

## The five distribution types

AIPerf supports five distribution shapes, and **figures out which one you mean from the keys you wrote** — you don't have to add a `type:` key. The discriminator is purely structural:

| What you wrote | Type | Why |
|---|---|---|
| `isl: 512` | Fixed | Bare scalar |
| `isl: {mean: 512, stddev: 50}` | Normal | `stddev` present |
| `isl: {mean: 512, median: 400}` | Log-normal | `median` present |
| `isl: {peaks: [...]}` | Multimodal | `peaks` present |
| `isl: {points: [...]}` | Empirical | `points` present |

You can override the inference with an explicit `type:` if you'd rather be loud:

```yaml
isl: {type: normal, mean: 512, stddev: 50}
```

`type:` accepts one of `fixed`, `normal`, `lognormal`, `multimodal`, `empirical`. AIPerf strips it after dispatch, so the rest of the dict is parsed normally.

### Fixed — a constant

The simplest case. Every sample returns the same value.

```yaml
prompts:
  isl: 512                        # scalar shorthand
  osl: {value: 128}               # explicit object form (rarely needed)
```

Use a fixed distribution when you want a deterministic input or output size — e.g. reproducing a sizing study or feeding a controlled stress test.

### Normal — Gaussian around a mean

A truncated Gaussian implemented via rejection sampling (samples below 0 are redrawn; falls back to clamped-mean if 10k iterations fail to land in range). Parameterised by `mean` and `stddev`.

```yaml
prompts:
  isl: {mean: 512, stddev: 50}
  osl: {mean: 128, stddev: 25}
```

This is the workhorse for "vary around a target." If `stddev: 0` is set or omitted, the distribution collapses to deterministic — equivalent to `fixed`.

A few details worth knowing:

- `mean` must be `>= 0`. Zero is allowed (e.g. `osl: {mean: 0}` disables output, `turn_delay: {mean: 0}` disables inter-turn delay).
- `stddev` must be `>= 0`. Default is `0`.
- A bare `{mean: 512}` (no `stddev`, no `median`) is still treated as Normal — a Normal with zero stddev. This is intentional: it keeps the rule "set `mean` and you get a Normal" simple. If you want a *log-normal* with no skew, write `{mean: 512, median: 512}`.

### Log-normal — right-skewed, always positive

A log-normal distribution parameterised by `mean` and `median`. Skew is controlled by the `mean / median` ratio: the larger the ratio, the heavier the right tail. When `mean == median` it collapses to deterministic.

```yaml
prompts:
  isl: {mean: 1024, median: 512}      # heavy right tail
  osl: {mean: 200, median: 180}       # mild skew
```

Constraints:

- Both `mean` and `median` must be `> 0`.
- `median` must be `<= mean`. (A log-normal with median > mean is mathematically impossible.)

Use log-normal when modelling sizes that are bounded below by zero and have a long right tail — chat prompt lengths, retrieval-augmented context windows, "most requests are small but some are huge" workloads.

### Multimodal — a mixture of N peaks

A weighted mixture of two or more sub-distributions. Each `peak` is itself a distribution, written inline, with an optional `weight`.

```yaml
prompts:
  isl:
    peaks:
      - {mean: 128, stddev: 20, weight: 60}     # 60% — short queries
      - {mean: 2048, median: 1800, weight: 30}  # 30% — long contexts (log-normal)
      - {value: 8192, weight: 10}               # 10% — exact 8K stress

  # Equal-weight peaks: omit `weight` and they're split evenly.
  osl:
    peaks:
      - {mean: 64, stddev: 10}
      - {mean: 256, stddev: 40}
      - {mean: 1024, stddev: 100}
```

Notes:

- Requires **at least 2 peaks**.
- Each peak follows the same auto-detection rules — write `{stddev: ...}` for Normal peaks, `{median: ...}` for log-normal peaks, `{value: N}` for fixed peaks.
- Weights are *relative* — they're normalised internally, so `[60, 30, 10]` and `[6, 3, 1]` produce the same mixture.
- `weight` is optional and defaults to `1.0`. Omit it on every peak to get an equal split.

Use multimodal when your real workload is a *mix* of distinct request shapes — e.g. a chat product where 70% of traffic is one-shot Q&A and 30% is long document summarisation. A single Normal can't capture that.

### Empirical — discrete weighted values

A discrete distribution sampled from a set of weighted values. No interpolation, no Gaussian — each draw returns one of the values you listed.

```yaml
prompts:
  isl:
    points:
      - {value: 128,  weight: 40}
      - {value: 512,  weight: 35}
      - {value: 2048, weight: 20}
      - {value: 8192, weight: 5}
```

Notes:

- Requires at least one point. Weights must be `> 0` and are normalised internally.
- `weight` defaults to `1.0` — omit it for an equal-probability sampler over the listed values.

Use empirical when you have measured frequencies from production traces and want to reproduce them exactly without smoothing into a parametric shape.

## Clamping with `min:` / `max:`

Every distribution shape — including the scalar shorthand — accepts optional `min:` and `max:` bounds. Samples outside the range are clamped (not resampled), so the bounds are *hard limits*, not statistical guarantees.

```yaml
prompts:
  isl:
    mean: 512
    stddev: 200
    min: 32          # never below 32 tokens
    max: 4096        # never above 4096 tokens

  osl:
    peaks:
      - {mean: 64,  stddev: 30}
      - {mean: 1024, stddev: 200}
    min: 16
    max: 2048
```

A few rules:

- Bounds are *inclusive*: `min: 32` means values down to and including 32 are kept; below 32 is clamped up to 32.
- `min:` and `max:` must be finite. NaN/inf are rejected at config-validation time so they can't silently disable clamping.
- If both are set, `min <= max` is enforced.
- Bounds compose with every shape — Fixed, Normal, Log-normal, Multimodal, and Empirical.

For multimodal distributions, a top-level `min`/`max` applies to the *output* of the mixture. If you want different bounds per peak, set `min`/`max` on each peak's sub-distribution instead.

## Disambiguation cheat-sheet

If AIPerf can't figure out what shape you meant, it errors at config-load time with a message that names the keys it saw. The most common causes:

| Mistake | What AIPerf does |
|---|---|
| `isl: {mean: 512}` (no `stddev`, no `median`) | Treated as Normal with `stddev=0` (deterministic). |
| `isl: {stddev: 50}` (no `mean`) | Error — Normal requires `mean`. |
| `isl: {peaks: [...one entry...]}` | Error — Multimodal requires at least 2 peaks. |
| `isl: {value: 512, mean: 600}` | Error — `value` selects Fixed, but `mean` is unknown to Fixed. |
| Passing a string like `"128,64:50;512,128:50"` | Error — that's the legacy `sequence_distribution` string format (semicolon-separated `ISL,OSL:prob` pairs summing to 100), not a sampling distribution. See [Sequence Length Distributions](sequence-distributions.md). |

When in doubt, run:

```bash
aiperf config validate my-config.yaml
```

The validator runs the same load pipeline `aiperf profile` does, so any distribution-shape problem surfaces here before you spend compute.

## Combining with sweeps

Sweep parameters (`sweep.parameters`) can replace a distribution wholesale. The right-hand side of a sweep entry is the *value* that gets substituted into the body, so you can sweep across distribution shapes the same way you sweep across scalars:

```yaml
sweep:
  type: grid
  parameters:
    # Sweep across three different ISL distributions.
    datasets.default.prompts.isl:
      - 512
      - {mean: 512, stddev: 100}
      - {peaks: [{mean: 128, stddev: 20}, {mean: 2048, stddev: 200}]}
```

That gives you three benchmark variations, each with a different ISL shape, while the rest of the body stays constant. Pair with `multi_run` for confidence intervals per shape — see [Multi-Run Confidence Reporting](multi-run-confidence.md).

## Worked example — a realistic chat workload

Putting it all together: a synthetic dataset that mixes short and long queries, with a log-normal output shape and clamped bounds.

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
    entries: 500
    prompts:
      # Bimodal ISL — most traffic is short, but 20% is a long context.
      isl:
        peaks:
          - {mean: 200,  stddev: 50,  weight: 80}
          - {mean: 4096, median: 3500, weight: 20}
        min: 32
        max: 8192

      # OSL has a long right tail — a few responses are unusually long.
      osl:
        mean: 256
        median: 200
        max: 1024

    # Multi-turn chat: most conversations are 2-3 turns, some run longer.
    turns:
      mean: 3
      stddev: 1
      min: 1
      max: 8

    # User think-time between turns, in milliseconds.
    turn_delay:
      mean: 1500
      stddev: 800
      min: 100

  phases:
    - name: warmup
      type: concurrency
      concurrency: 4
      requests: 50
      exclude_from_results: true
    - name: profiling
      type: poisson
      rate: 30.0
      duration: 120
      concurrency: 64
```

Run it with:

```bash
aiperf profile --config chat-mixed.yaml
```

## Where to go next

- **[YAML Configuration Files](yaml-config.md)** — the broader walkthrough of YAML configs, sweeps, and multi-run.
- **[Sequence Length Distributions](sequence-distributions.md)** — the legacy `--sequence-distribution` *string* format used on the CLI for paired ISL/OSL mixtures (separate feature).
- **[Multi-Run Confidence Reporting](multi-run-confidence.md)** — repeating a benchmark for confidence intervals on top of any of these shapes.
- **[Parameter Sweeps](sweeps.md)** — how to sweep across distribution shapes themselves.
