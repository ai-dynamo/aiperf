<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Gradual Ramping

Gradual ramping lets you increase concurrency and request rate smoothly over time, rather than jumping to full load immediately. This prevents overwhelming your server at benchmark start.

## Why Use Ramping?

When a benchmark starts, immediately hitting your target load can cause problems:

```
Without ramping:                     With ramping:

Concurrency                          Concurrency
 100 ┤●━━━━━━━━━━━━━━━━━━━            100 ┤           ●━━━━━━━━━
     │                                    │         ╱
     │  SPIKE! Server overwhelmed         │       ╱
     │  - Connection storms               │     ╱  Gradual increase
     │  - Memory allocation spikes     50 ┤   ╱    Server stabilizes
     │  - Cold-start pollution            │ ╱      at each level
   0 ┼──────────────────────▶          0 ┼●─────────────────────▶
     0                      Time           0                    30s Time
```

**Problems without ramping:**
- **Connection storms** — Hundreds of simultaneous connections overwhelming the server
- **Memory spikes** — Sudden KV-cache allocation causing OOM or degraded performance
- **Misleading metrics** — Cold-start effects polluting your steady-state measurements

**Benefits of ramping:**
- Server warms up gradually (caches, JIT compilation, connection pools)
- Early detection of capacity limits before hitting full load
- Cleaner measurements once you reach steady state

## What Can Be Ramped?

AIPerf supports ramping three dimensions:

| Dimension | CLI Option | What It Controls |
|-----------|-----------|------------------|
| **Session Concurrency** | `--concurrency-ramp-duration` | Max simultaneous requests |
| **Prefill Concurrency** | `--prefill-concurrency-ramp-duration` | Max requests in prefill phase |
| **Request Rate** | `--request-rate-ramp-duration` | Requests per second |

Each ramps from a low starting value up to your target over the specified duration.

## Basic Usage

### Ramping Concurrency

Gradually increase from 1 concurrent request to 100 over 30 seconds:

```bash
aiperf profile \
    --model your-model \
    --url localhost:8000 \
    --concurrency 100 \
    --concurrency-ramp-duration 30 \
    --request-count 1000
```

**What happens:**
```
Concurrency
 100 ┤                    ●━━━━━━━━━━━
  75 ┤               ●────┘
  50 ┤          ●────┘
  25 ┤     ●────┘
   1 ┤●────┘
     └─────┬─────┬─────┬─────┬─────────────────▶
          7.5s  15s  22.5s  30s              Time
```

### Ramping Request Rate

Gradually increase from a low starting rate to 100 QPS over 60 seconds:

```bash
aiperf profile \
    --model your-model \
    --url localhost:8000 \
    --request-rate 100 \
    --request-rate-ramp-duration 60 \
    --benchmark-duration 120
```

**What happens:**
```
Request Rate (QPS)
 100 ┤                    ●━━━━━━━━━━━
  75 ┤               ●────┘
  50 ┤          ●────┘
  25 ┤     ●────┘
 ~0 ┤●────┘
     └─────┬─────┬─────┬─────┬─────────────────▶
          15s   30s   45s   60s              Time
```

> **Note**: The starting rate is calculated proportionally: `start = target * (update_interval / duration)`. With default settings (0.1s updates), ramping to 100 QPS over 60 seconds starts at ~0.17 QPS (not zero).

### Combining Rate and Concurrency Ramping

Ramp both dimensions together for maximum control:

```bash
aiperf profile \
    --model your-model \
    --url localhost:8000 \
    --request-rate 200 \
    --request-rate-ramp-duration 30 \
    --concurrency 100 \
    --concurrency-ramp-duration 30 \
    --benchmark-duration 120
```

Both ramp in parallel, reaching their targets at 30 seconds.

## Prefill Concurrency Ramping

For long-context workloads, you may want to limit how many requests are in the "prefill" phase (processing input tokens) simultaneously. This prevents memory spikes from multiple large prompts being processed at once.

```bash
aiperf profile \
    --model your-model \
    --url localhost:8000 \
    --concurrency 100 \
    --prefill-concurrency 20 \
    --prefill-concurrency-ramp-duration 20 \
    --synthetic-input-tokens-mean 8000
```

This limits prefill to 20 concurrent requests (ramped over 20 seconds), while allowing up to 100 total concurrent requests.

## Warmup Phase Ramping

Each phase can have its own ramp settings. Warmup uses `--warmup-*` prefixed options:

```bash
aiperf profile \
    --model your-model \
    --url localhost:8000 \
    # Warmup phase: ramp to 50 concurrency over 10 seconds
    --warmup-concurrency 50 \
    --warmup-concurrency-ramp-duration 10 \
    --warmup-request-count 500 \
    # Profiling phase: ramp to 200 concurrency over 30 seconds
    --concurrency 200 \
    --concurrency-ramp-duration 30 \
    --request-count 2000
```

## Common Scenarios

### High-Concurrency Stress Test

Ramp slowly to avoid overwhelming the server, then sustain full load:

```bash
aiperf profile \
    --model your-model \
    --url localhost:8000 \
    --concurrency 500 \
    --concurrency-ramp-duration 60 \
    --benchmark-duration 300
```

The 60-second ramp gives the server time to allocate resources (~8 new connections per second).

### Long-Context Memory Protection

Limit memory spikes from large prompts with prefill concurrency:

```bash
aiperf profile \
    --model your-model \
    --url localhost:8000 \
    --concurrency 50 \
    --prefill-concurrency 5 \
    --prefill-concurrency-ramp-duration 30 \
    --synthetic-input-tokens-mean 32000
```

Only 5 requests process their 32K input tokens simultaneously, preventing KV-cache OOM.

### Capacity Discovery

Find your server's limits by ramping slowly and watching for degradation:

```bash
aiperf profile \
    --model your-model \
    --url localhost:8000 \
    --request-rate 500 \
    --request-rate-ramp-duration 120 \
    --concurrency 200 \
    --concurrency-ramp-duration 120 \
    --benchmark-duration 180
```

Watch latency and throughput metrics as load increases. When latency spikes or errors appear, you've found the limit.

## How It Works

### Concurrency Ramping (Discrete Steps)

Concurrency increases by +1 at evenly spaced intervals:

- **100 concurrency over 30 seconds** = +1 every ~0.3 seconds
- **500 concurrency over 60 seconds** = +1 every ~0.12 seconds

Each step allows one more concurrent request immediately.

### Request Rate Ramping (Smooth Interpolation)

Rate updates continuously (every 0.1 seconds by default):

- **100 QPS over 60 seconds** = updates ~600 times, smoothly increasing
- Linear interpolation from start rate to target rate

This creates smooth traffic curves without sudden jumps.

## Quick Reference

| Option | Description | Default |
|--------|-------------|---------|
| `--concurrency-ramp-duration <sec>` | Ramp session concurrency over N seconds | No ramping |
| `--prefill-concurrency-ramp-duration <sec>` | Ramp prefill concurrency over N seconds | No ramping |
| `--request-rate-ramp-duration <sec>` | Ramp request rate over N seconds | No ramping |
| `--warmup-concurrency-ramp-duration <sec>` | Warmup phase concurrency ramp | Uses main value |
| `--warmup-prefill-concurrency-ramp-duration <sec>` | Warmup phase prefill ramp | Uses main value |
| `--warmup-request-rate-ramp-duration <sec>` | Warmup phase rate ramp | Uses main value |

**Key behaviors:**
- Concurrency starts at 1 and increases by +1 at even intervals
- Request rate starts proportionally low and interpolates smoothly
- Ramps complete exactly at the specified duration
- After ramping, values stay at the target for the rest of the phase

## Related Documentation

- [Prefill Concurrency](./prefill-concurrency.md) — Memory-safe long-context benchmarking with prefill limiting
- [Request Rate with Concurrency](./request-rate-concurrency.md) — Combining rate and concurrency controls
- [Timing Modes Reference](../benchmark_modes/timing-modes-reference.md) — Complete CLI compatibility matrix
