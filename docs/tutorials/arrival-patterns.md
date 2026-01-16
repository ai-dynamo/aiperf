<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Arrival Patterns: Simulating Realistic Traffic

When benchmarking with `--request-rate`, AIPerf can vary how requests arrive over time. The `--arrival-pattern` option controls the distribution of inter-arrival times, letting you simulate everything from perfectly regular traffic to bursty real-world patterns.

## Why Arrival Patterns Matter

Real traffic doesn't arrive at perfectly regular intervals. Traffic comes in bursts—quiet periods followed by sudden spikes. How your server handles this variance affects real-world performance.

```
Constant Pattern:         Poisson Pattern:         Gamma (bursty):
|  |  |  |  |  |  |       |   | || |    | |         |||    |    |||  |
└──────────────────▶     └──────────────────▶    └──────────────────▶
  Perfect spacing          Natural variance        Clustered bursts
  (unrealistic)            (typical traffic)       (stress testing)
```

## Quick Start

```bash
# Default: Poisson (realistic)
aiperf profile --request-rate 50 ...

# Explicit: Constant (deterministic)
aiperf profile --request-rate 50 --arrival-pattern constant ...

# Bursty: Gamma with low smoothness
aiperf profile --request-rate 50 --arrival-pattern gamma --arrival-smoothness 0.5 ...
```

## Available Patterns

### Constant

```bash
--arrival-pattern constant
```

Requests arrive at perfectly regular intervals: exactly `1/rate` seconds apart.

```
Inter-arrival times:
10 QPS → every 100ms:  |····|····|····|····|····|····|
                       0   100  200  300  400  500  600 ms
```

**Use cases:**
- Baseline measurements with no variance
- Debugging timing issues
- Comparing against variable patterns
- Deterministic, reproducible tests

### Poisson (Default)

```bash
--arrival-pattern poisson
```

Requests arrive according to a Poisson process—the mathematical model for random events at a constant average rate. Inter-arrival times follow an exponential distribution.

```
Inter-arrival times (exponential):
10 QPS average:  |··|······|·|···|····|··|·······|···|
                 Varied gaps, same average rate over time
```

**Characteristics:**
- **Mean** inter-arrival = `1/rate` (same as constant)
- **Variance** = `(1/rate)²` (natural randomness)
- Sometimes requests cluster, sometimes gaps appear
- Models real user behavior where arrivals are independent

**Use cases:**
- Default realistic traffic simulation
- Standard load testing
- Comparing to theoretical queueing models

### Gamma (Tunable Burstiness)

```bash
--arrival-pattern gamma --arrival-smoothness <value>
```

Gamma distribution generalizes Poisson with a **smoothness** parameter that controls how bursty or regular arrivals are:

| Smoothness | Behavior | Variance | Use Case |
|------------|----------|----------|----------|
| `< 1.0` | **Bursty** — clustered arrivals with gaps | Higher | Stress testing, worst-case scenarios |
| `= 1.0` | **Poisson** — natural randomness | Medium | Same as `--arrival-pattern poisson` |
| `> 1.0` | **Smooth** — more regular arrivals | Lower | Controlled testing, less noise |

```
Smoothness = 0.5 (bursty):
||||      |||        |||||    ||
 Clusters of requests with quiet gaps

Smoothness = 1.0 (Poisson):
|  || |   | |  ||  |   | ||  |
 Natural variance

Smoothness = 2.0 (smooth):
| | | |  | | | | | |  | | | |
 More regular, approaches constant
```

**Mathematical note:** The smoothness parameter is the Gamma distribution's shape parameter (k). Scale is automatically computed to maintain the correct mean rate.

### Concurrency Burst

```bash
# No --request-rate, just --concurrency
aiperf profile --concurrency 50 ...
```

When you omit `--request-rate` and only specify `--concurrency`, AIPerf uses burst mode: zero delay between request dispatches, limited only by the concurrency semaphore.

```
Burst mode (concurrency=3):
[Req1]────────────────────────────▶
[Req2]────────────────────────────▶
[Req3]────────────────────────────▶
      [Req4]──────────────────────▶  ← Starts when any slot frees
```

**Use cases:**
- Maximum throughput discovery
- Saturation testing
- Finding server capacity limits

## vLLM Compatibility

AIPerf's `--arrival-smoothness` is compatible with vLLM's `--burstiness` parameter:

```bash
# Same distribution as vLLM with --burstiness 0.5
aiperf profile \
    --request-rate 50 \
    --arrival-pattern gamma \
    --arrival-smoothness 0.5 \
    ...
```

This allows direct comparison between AIPerf and vLLM benchmark results when using the same smoothness/burstiness value.

## Examples

### Baseline vs Realistic Comparison

Compare how your server handles ideal vs realistic traffic:

```bash
# Run 1: Constant (baseline)
aiperf profile \
    --model your-model \
    --url localhost:8000 \
    --endpoint-type chat \
    --streaming \
    --request-rate 100 \
    --arrival-pattern constant \
    --benchmark-duration 60 \
    --output-dir results/constant

# Run 2: Poisson (realistic)
aiperf profile \
    --model your-model \
    --url localhost:8000 \
    --endpoint-type chat \
    --streaming \
    --request-rate 100 \
    --arrival-pattern poisson \
    --benchmark-duration 60 \
    --output-dir results/poisson
```

Compare TTFT and throughput between runs. Higher variance under Poisson indicates sensitivity to traffic patterns.

### Stress Testing with Bursty Traffic

Test how your server handles request bursts:

```bash
aiperf profile \
    --model your-model \
    --url localhost:8000 \
    --endpoint-type chat \
    --streaming \
    --request-rate 100 \
    --arrival-pattern gamma \
    --arrival-smoothness 0.3 \
    --benchmark-duration 120
```

Smoothness of 0.3 creates highly bursty traffic—several requests arrive nearly simultaneously, then quiet periods.

### Smooth Traffic for Noise Reduction

Reduce variance in measurements for controlled experiments:

```bash
aiperf profile \
    --model your-model \
    --url localhost:8000 \
    --endpoint-type chat \
    --streaming \
    --request-rate 50 \
    --arrival-pattern gamma \
    --arrival-smoothness 5.0 \
    --benchmark-duration 60
```

Smoothness of 5.0 produces very regular arrivals, reducing measurement noise while still having some natural variance.

### Progressive Burstiness Test

Run multiple benchmarks with increasing burstiness to find where performance degrades:

```bash
for smoothness in 2.0 1.0 0.7 0.5 0.3; do
    aiperf profile \
        --model your-model \
        --url localhost:8000 \
        --endpoint-type chat \
        --streaming \
        --request-rate 100 \
        --arrival-pattern gamma \
        --arrival-smoothness $smoothness \
        --benchmark-duration 60 \
        --output-dir results/smoothness_$smoothness
done
```

### Warmup with Stable Pattern, Profile with Realistic

Use constant arrivals during warmup, then realistic patterns for profiling:

```bash
aiperf profile \
    --model your-model \
    --url localhost:8000 \
    --endpoint-type chat \
    --streaming \
    --request-rate 100 \
    --arrival-pattern gamma \
    --arrival-smoothness 0.8 \
    --warmup-arrival-pattern constant \
    --warmup-duration 30 \
    --benchmark-duration 120
```

## CLI Reference

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--arrival-pattern` | str | `poisson` | Pattern for request arrivals: `constant`, `poisson`, `gamma` |
| `--arrival-smoothness` | float | None | Gamma smoothness: `<1` = bursty, `1` = Poisson, `>1` = smooth. Defaults to `1.0` when using `gamma` pattern. |
| `--warmup-arrival-pattern` | str | Inherits | Override pattern for warmup phase |

**Constraints:**
- `--arrival-pattern` requires `--request-rate` to be set
- `--arrival-smoothness` only applies when `--arrival-pattern gamma`
- Cannot use with `--user-centric-rate` (deterministic per-user scheduling)
- Cannot use with `--fixed-schedule` (timestamp-based scheduling)

## Pattern Selection Guide

| Goal | Pattern | Smoothness |
|------|---------|------------|
| Reproducible baseline | `constant` | N/A |
| Realistic traffic simulation | `poisson` | N/A |
| Match vLLM benchmark | `gamma` | Same as vLLM `--burstiness` |
| Stress test burst handling | `gamma` | `0.3 - 0.7` |
| Reduce measurement noise | `gamma` | `2.0 - 5.0` |
| Maximum throughput | N/A (burst mode) | N/A |

## Understanding the Math

For those who want to understand the statistical properties:

| Pattern | Distribution | Mean | Variance | CV (Coeff. of Variation) |
|---------|--------------|------|----------|--------------------------|
| Constant | Degenerate | `1/λ` | `0` | `0` |
| Poisson | Exponential | `1/λ` | `1/λ²` | `1` |
| Gamma(k) | Gamma | `1/λ` | `1/(k·λ²)` | `1/√k` |

Where `λ` = request rate and `k` = smoothness.

- **CV (Coefficient of Variation)** = standard deviation / mean
- Lower CV = more regular arrivals
- Gamma with k=1 equals Poisson (CV=1)
- As k→∞, Gamma approaches Constant (CV→0)

## Related Documentation

- [Request Rate with Concurrency](./request-rate-concurrency.md) — Combining rate and concurrency
- [Warmup Phase](./warmup.md) — Configuring warmup with different patterns
- [Timing Modes Reference](../benchmark_modes/timing-modes-reference.md) — Complete CLI compatibility matrix
