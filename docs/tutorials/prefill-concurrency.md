<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Prefill Concurrency: Fine-Grained Benchmarking Control

Prefill concurrency (`--prefill-concurrency`) limits how many requests can be in the **prefill phase** simultaneously—the compute-intensive phase where the LLM processes input tokens before generating output. Instead of tuning request rate broadly, this gives you fine-grained control over how much queueing occurs at the prefill stage—especially valuable for disaggregated serving architectures where you want to directly control TTFT behavior.

## Why Prefill Concurrency Matters

Every LLM request has two phases:

```
Request Lifecycle
─────────────────────────────────────────────────────────────────────────────
│           PREFILL                    │           DECODE                   │
│      (reading your prompt)           │      (generating the response)     │
├──────────────────────────────────────┼────────────────────────────────────┤
│ • Processes all input tokens         │ • Generates tokens one at a time   │
│ • Uses lots of memory upfront        │ • Steady, lower memory usage       │
└──────────────────────────────────────┴────────────────────────────────────┘
                 ▲                                      ▲
         First token appears                   Response streams back
```

Limiting simultaneous prefills also prevents memory exhaustion when benchmarking long prompts.

## How It Works

AIPerf limits how many requests can be in the prefill phase at once:

```
With --prefill-concurrency 3:

  ┌─────────────────────────────────────────────────────────────────────┐
  │  PREFILL GATE (max 3 at a time)                                     │
  │  ┌─────────┐  ┌─────────┐  ┌─────────┐                              │
  │  │ Slot 1  │  │ Slot 2  │  │ Slot 3  │  ← Slots free up when        │
  │  │  busy   │  │  busy   │  │  free   │    first token arrives       │
  │  └─────────┘  └─────────┘  └─────────┘                              │
  └─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  DECODE (limited by session --concurrency)                          │
  │  Request A ████████████████░░░░░░░░░░░░░░░░░░░░                     │
  │  Request B ██████████░░░░░░░░░░░░░░░░░░░░░░░░░░                     │
  │  Request C ██████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░                     │
  │  ... many more can decode simultaneously                            │
  └─────────────────────────────────────────────────────────────────────┘
```

Once a request receives its first token, it releases its prefill slot and moves to decode—allowing the next request to start prefilling.

> [!IMPORTANT]
> Requires `--streaming` to be enabled. Without streaming, AIPerf can't detect when the first token arrives.

> [!WARNING]
> **Coordinated omission trade-off:** When requests wait for prefill slots, the benchmark operates as a closed loop, throttling itself to match server capacity. This is [coordinated omission](https://www.scylladb.com/2021/04/22/on-coordinated-omission/)—your measured latencies will be **lower** than what users would experience if traffic kept arriving at the original rate. For accurate latency measurement, use open-loop benchmarking (request rate without prefill limits).

## Two Concurrency Limits

AIPerf has two separate limits that work together:

- **`--concurrency`** — Session concurrency: total active requests at once (per-request in single-turn mode, per-conversation in multi-turn mode)
- **`--prefill-concurrency`** — Prefill concurrency: how many can be in prefill phase at once

**Example:**

```bash
--concurrency 50 --prefill-concurrency 5
```

This means:
- Up to 50 requests can be active at once
- But only 5 can be reading their prompts (prefilling) at the same time
- The other 45 are either waiting to prefill OR already generating responses

## Examples

### Controlling Prefill Queue Depth

Benchmark with 16K token prompts, limiting how many can prefill simultaneously:

```bash
aiperf profile \
    --model Qwen/Qwen2.5-7B-Instruct \
    --endpoint-type chat \
    --streaming \
    --url localhost:8000 \
    --concurrency 30 \
    --prefill-concurrency 3 \
    --synthetic-input-tokens-mean 16000 \
    --output-tokens-mean 500 \
    --request-count 100
```

**What happens:**

- 30 total concurrent sessions allowed
- Only 3 can prefill their 16K tokens simultaneously

### Gradual Prefill Ramp-Up

Ramp prefill concurrency gradually to observe how TTFT changes as queue depth increases:

```bash
aiperf profile \
    --model Qwen/Qwen2.5-7B-Instruct \
    --endpoint-type chat \
    --streaming \
    --url localhost:8000 \
    --concurrency 50 \
    --prefill-concurrency 5 \
    --prefill-concurrency-ramp-duration 30 \
    --synthetic-input-tokens-mean 32000 \
    --output-tokens-mean 200 \
    --benchmark-duration 120
```

**Ramp behavior:**

```
Prefill Concurrency
  5 ┤                    ●━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  4 ┤               ●────┘
  3 ┤          ●────┘
  2 ┤     ●────┘
  1 ┤●────┘
    └─────┬─────┬─────┬─────┬─────────────────────────▶
         7.5s  15s  22.5s  30s                      Time
```

### Combined with Request Rate

Prefill concurrency works with all scheduling modes:

```bash
aiperf profile \
    --model Qwen/Qwen2.5-7B-Instruct \
    --endpoint-type chat \
    --streaming \
    --url localhost:8000 \
    --request-rate 10 \
    --concurrency 100 \
    --prefill-concurrency 10 \
    --synthetic-input-tokens-mean 8000 \
    --output-tokens-mean 300 \
    --benchmark-duration 60
```

Requests arrive at 10 QPS, up to 100 can be active, but only 10 can prefill at once.

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| OOM despite prefill limit | Limit too high, or decode memory not considered | Lower `--prefill-concurrency`, also limit `--concurrency` |
| Requests stuck waiting | Expected when prefill > inter-arrival time | Increase limit or lower `--request-rate` |
| Slots not releasing | `--streaming` not enabled or server not streaming | Ensure `--streaming` is set, verify server supports it |

## CLI Reference

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--prefill-concurrency` | int | None | Max requests in prefill phase (requires `--streaming`) |
| `--prefill-concurrency-ramp-duration` | float | None | Seconds to ramp from 1 to target |
| `--warmup-prefill-concurrency` | int | None | Prefill limit during warmup (falls back to main) |
| `--warmup-prefill-concurrency-ramp-duration` | float | None | Warmup prefill ramp duration |

**Constraints:**
- `--prefill-concurrency` must be ≤ `--concurrency` (if both set)
- Requires `--streaming` to be enabled
- Works with all scheduling modes (`--request-rate`, `--user-centric-rate`, `--fixed-schedule`, burst mode)

## Related Documentation

- [Gradual Ramping](./ramping.md) — Smooth ramp-up for all concurrency dimensions
- [Request Rate with Concurrency](./request-rate-concurrency.md) — Combining rate and concurrency controls
- [User-Centric Timing](./user-centric-timing.md) — Multi-turn benchmarking for KV cache
- [Timing Modes Reference](../benchmark_modes/timing-modes-reference.md) — Complete CLI compatibility matrix
