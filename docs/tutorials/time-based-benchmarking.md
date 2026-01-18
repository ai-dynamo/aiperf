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
