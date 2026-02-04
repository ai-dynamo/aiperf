<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Time-Based Benchmarking

Time-based benchmarking runs for a specific duration rather than a fixed number of requests. Use it for SLA validation, stability testing, capacity planning, and A/B comparisons where consistent time windows matter.

## Quick Start

```bash
aiperf profile \
    --model your-model \
    --url localhost:8000 \
    --endpoint-type chat \
    --streaming \
    --concurrency 10 \
    --benchmark-duration 60
```

Requests are sent continuously until the duration expires. AIPerf then waits for in-flight requests to complete (up to the grace period).

## How It Works

```
│          BENCHMARK DURATION           │   GRACE PERIOD    │
│        (sending requests)             │   (drain only)    │
├───────────────────────────────────────┼───────────────────┤
│ New requests dispatched               │ No new requests   │
│ Responses collected                   │ Wait for in-flight│
└───────────────────────────────────────┴───────────────────┘
                    ▲                             ▲
           Duration expires              Grace period ends
```

- **Grace period default**: 30 seconds (use `inf` to wait forever, `0` for immediate completion)
- Responses received within grace period are included in metrics; responses still pending when grace expires are not

> [!IMPORTANT]
> `--benchmark-grace-period` requires `--benchmark-duration` to be set.

## Combining with Request Count

Duration can be combined with count-based stopping—**first condition reached wins**:

```bash
# Stop when EITHER 1000 requests sent OR 120 seconds pass
aiperf profile \
    --model your-model \
    --url localhost:8000 \
    --endpoint-type chat \
    --streaming \
    --request-rate 20 \
    --benchmark-duration 120 \
    --request-count 1000
```

## Examples

### Stability Test (5 minutes)

```bash
aiperf profile \
    --model Qwen/Qwen2.5-7B-Instruct \
    --url localhost:8000 \
    --endpoint-type chat \
    --streaming \
    --concurrency 50 \
    --benchmark-duration 300 \
    --benchmark-grace-period 60 \
    --warmup-duration 30
```

**Sample Output (Successful Run):**
```
INFO     Starting AIPerf System
INFO     AIPerf System is WARMING UP

Warming Up: [00:30] - Running for 30 seconds...

INFO     Warmup completed, starting profiling phase
INFO     AIPerf System is PROFILING

Profiling: [05:00] - Running for 300 seconds...

INFO     Benchmark duration reached, draining in-flight requests
INFO     Grace period: waiting up to 60 seconds for responses

INFO     Benchmark completed successfully
INFO     Results saved to: artifacts/Qwen_Qwen2.5-7B-Instruct-chat-concurrency50/

            NVIDIA AIPerf | LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
┃                     Metric ┃    avg ┃    min ┃    max ┃    p99 ┃    p50 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
│       Request Latency (ms) │ 245.67 │ 178.90 │ 398.12 │ 367.89 │ 239.45 │
│   Time to First Token (ms) │  56.78 │  42.34 │  89.12 │  82.45 │  55.23 │
│   Inter Token Latency (ms) │  13.45 │  10.23 │  19.67 │  18.45 │  13.12 │
│ Request Throughput (req/s) │  89.23 │      - │      - │      - │      - │
└────────────────────────────┴────────┴────────┴────────┴────────┴────────┘

JSON Export: artifacts/Qwen_Qwen2.5-7B-Instruct-chat-concurrency50/profile_export_aiperf.json
```

### Soak Test (1 hour)

```bash
aiperf profile \
    --model Qwen/Qwen2.5-7B-Instruct \
    --url localhost:8000 \
    --endpoint-type chat \
    --streaming \
    --concurrency 20 \
    --benchmark-duration 3600 \
    --benchmark-grace-period 120 \
    --warmup-duration 60
```

**Sample Output (Successful Run):**
```
INFO     Starting AIPerf System
INFO     AIPerf System is WARMING UP

Warming Up: [01:00] - Running for 60 seconds...

INFO     Warmup completed, starting profiling phase
INFO     AIPerf System is PROFILING

Profiling: [60:00] - Running for 3600 seconds...

INFO     Benchmark duration reached, draining in-flight requests
INFO     Grace period: waiting up to 120 seconds for responses

INFO     Benchmark completed successfully
INFO     Results saved to: artifacts/Qwen_Qwen2.5-7B-Instruct-chat-concurrency20/

            NVIDIA AIPerf | LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
┃                     Metric ┃    avg ┃    min ┃    max ┃    p99 ┃    p50 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
│       Request Latency (ms) │ 198.34 │ 156.78 │ 312.45 │ 289.67 │ 194.23 │
│   Time to First Token (ms) │  48.90 │  38.45 │  76.34 │  71.23 │  47.89 │
│   Inter Token Latency (ms) │  12.01 │   9.56 │  17.89 │  16.78 │  11.78 │
│ Request Throughput (req/s) │  45.67 │      - │      - │      - │      - │
└────────────────────────────┴────────┴────────┴────────┴────────┴────────┘

JSON Export: artifacts/Qwen_Qwen2.5-7B-Instruct-chat-concurrency20/profile_export_aiperf.json
```

## CLI Reference

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--benchmark-duration` | float | None | Stop sending requests after this many seconds |
| `--benchmark-grace-period` | float | 30.0 | Seconds to wait for in-flight requests after duration. Use `inf` for unlimited. Requires `--benchmark-duration`. |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Requests cut off mid-response | Increase `--benchmark-grace-period` or use `inf` |
| Grace period error | Add `--benchmark-duration` (grace period requires it) |

## Related Documentation

- [Warmup Phase](./warmup.md) — Configure pre-benchmark warmup
- [User-Centric Timing](./user-centric-timing.md) — Multi-turn benchmarking (auto-sets infinite grace)
- [Timing Modes Reference](../benchmark_modes/timing-modes-reference.md) — Complete CLI compatibility matrix
