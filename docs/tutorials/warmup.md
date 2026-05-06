---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Warmup Phase Configuration
---

# Warmup Phase Configuration

The warmup phase runs before your actual benchmark to prepare the system for steady-state measurement. This guide explains when and how to configure warmup for accurate benchmarking results.

> **Heads-up: agentic-replay mode has its own warmup.** When the run uses the `agentic_replay` timing mode (set today by `--scenario inferencex-agentx-mvp`), the warmup phase is **trajectory-based** rather than rate-based: it dispatches exactly one credit per trajectory at that trajectory's sampled starting turn `k_i`, and most of the warmup CLI flags below are ignored. `--warmup-grace-period` is honored on top of the inherited `--concurrency` / `--prefill-concurrency` (which set the trajectory pool size) — and unlike under rate-based scheduling, it works on its own without `--warmup-duration` (since `_build_warmup_config` in `src/aiperf/timing/config.py` ignores duration under `agentic_replay`). `--arrival-smoothness` is also propagated through but has no effect because the warmup arrival pattern is hard-coded to `concurrency_burst`. See [InferenceX AgentX MVP](agentx-mvp.md) for the trajectory-warmup mechanics.

## Why Use Warmup?

When benchmarking starts, several "cold-start" effects can pollute your measurements:

```
Without warmup:                          With warmup:

Latency                                  Latency
   ▲                                        ▲
   │ ▓▓                                     │              Profiling starts
   │ ▓▓▓    ← Cold-start spikes             │              after system is warm
   │ ▓▓▓▓▓    pollute results               │                    │
   │ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓                  │  Warmup    ▼  ▓▓▓▓▓▓▓▓▓▓▓▓▓
   │ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓               │  ▓▓▓▓▓▓▓▓▓▓▓  ▓▓▓▓▓▓▓▓▓▓▓▓▓
   └─────────────────────────────▶          └──────────────────────────────▶
                              Time                                       Time
```

**Cold-start effects include:**

| Effect | Cause | Impact |
|--------|-------|--------|
| **JIT compilation** | Python/PyTorch compiling code paths | Higher initial latency |
| **KV cache allocation** | Server allocating GPU memory | Memory pressure, timeouts |
| **Connection establishment** | New TCP/TLS handshakes for HTTP connections | Network latency spikes |
| **CUDA kernel compilation** | First-run kernel JIT | GPU stalls |
| **Model loading** | Lazy weight loading on first inference | Extreme latency outliers |

## Quick Start

Add warmup with a simple request count:

```bash
aiperf profile \
    --model your-model \
    --url localhost:8000 \
    --endpoint-type chat \
    --streaming \
    --request-rate 10 \
    --warmup-request-count 50 \
    --request-count 500
```

**Sample Output (Successful Run):**

> Output below is illustrative — the exact format of `INFO`/`NOTICE` lines and the progress display depends on the UI mode you select (`--ui simple` vs the default Textual dashboard). Real `Phase ... started/complete` lines are emitted by `src/aiperf/timing/phase/runner.py` at NOTICE level.

```
NOTICE   Phase warmup started | target: 50 requests
Warming Up: 50/50 |████████████████████████| 100% [00:05<00:00]
NOTICE   Phase warmup complete | completed=50, cancelled=0, errors=0 | elapsed=5.23s
NOTICE   Phase profiling started | target: 500 requests
Profiling: 500/500 |████████████████████████| 100% [00:50<00:00]
NOTICE   Phase profiling complete | completed=500, cancelled=0, errors=0 | elapsed=50.12s
INFO     Results saved to: artifacts/your-model-openai-chat-request_rate10.0/
JSON Export: artifacts/your-model-openai-chat-request_rate10.0/profile_export_aiperf.json
```

This sends 50 warmup requests before the 500 profiling requests begin. Warmup metrics are discarded.

## Warmup Trigger Options

You can trigger warmup with **count-based** or **duration-based** stopping:

### Count-Based Warmup

```bash
# Stop after 100 warmup requests
--warmup-request-count 100

# OR stop after 20 sessions complete (for multi-turn)
--num-warmup-sessions 20
```

### Duration-Based Warmup

```bash
# Run warmup for 30 seconds
--warmup-duration 30
```

### Combined (First One Wins)

```bash
# Warmup stops when EITHER condition is met
--warmup-duration 60 \
--warmup-request-count 200
```

## Warmup-Specific Load Settings

By default, warmup inherits your profiling settings. Override them for different warmup behavior:

### Different Concurrency

```bash
aiperf profile \
    --model your-model \
    --url localhost:8000 \
    --endpoint-type chat \
    --streaming \
    --concurrency 100 \
    --warmup-concurrency 20 \
    --warmup-request-count 50 \
    --request-count 500
```

**Sample Output (Successful Run):**
```
NOTICE   Phase warmup started | target: 50 requests
Warming Up: 50/50 |████████████████████████| 100% [00:12<00:00]
NOTICE   Phase warmup complete | completed=50, cancelled=0, errors=0 | elapsed=12.04s
NOTICE   Phase profiling started | target: 500 requests
Profiling: 500/500 |████████████████████████| 100% [01:15<00:00]
NOTICE   Phase profiling complete | completed=500, cancelled=0, errors=0 | elapsed=75.31s
INFO     Results saved to: artifacts/your-model-openai-chat-concurrency100/
JSON Export: artifacts/your-model-openai-chat-concurrency100/profile_export_aiperf.json
```

Warmup runs at 20 concurrent requests, then profiling runs at 100.

### Different Request Rate

```bash
aiperf profile \
    --model your-model \
    --url localhost:8000 \
    --endpoint-type chat \
    --streaming \
    --request-rate 50 \
    --warmup-request-rate 10 \
    --warmup-duration 30 \
    --benchmark-duration 120
```

**Sample Output (Successful Run):**
```
NOTICE   Phase warmup started | target: 30.0s duration
Warming Up: [00:30] - Running for 30 seconds...
NOTICE   Phase warmup complete | completed=298, cancelled=0, errors=0 | elapsed=30.04s
NOTICE   Phase profiling started | target: 120.0s duration
Profiling: [02:00] - Running for 120 seconds...
NOTICE   Phase profiling complete | completed=5980, cancelled=0, errors=0 | elapsed=120.07s
INFO     Results saved to: artifacts/your-model-openai-chat-request_rate50.0/
JSON Export: artifacts/your-model-openai-chat-request_rate50.0/profile_export_aiperf.json
```

Warmup sends at 10 QPS, then profiling runs at 50 QPS.

### Different Arrival Pattern

```bash
aiperf profile \
    --model your-model \
    --url localhost:8000 \
    --endpoint-type chat \
    --streaming \
    --request-rate 20 \
    --arrival-pattern gamma \
    --arrival-smoothness 2.0 \
    --warmup-arrival-pattern constant \
    --warmup-duration 30 \
    --benchmark-duration 120
```

**Sample Output (Successful Run):**
```
NOTICE   Phase warmup started | target: 30.0s duration
Warming Up: [00:30] - Running for 30 seconds...
NOTICE   Phase warmup complete | completed=596, cancelled=0, errors=0 | elapsed=30.05s
NOTICE   Phase profiling started | target: 120.0s duration
Profiling: [02:00] - Running for 120 seconds...
NOTICE   Phase profiling complete | completed=2387, cancelled=0, errors=0 | elapsed=120.09s
INFO     Results saved to: artifacts/your-model-openai-chat-request_rate20.0/
JSON Export: artifacts/your-model-openai-chat-request_rate20.0/profile_export_aiperf.json
```

Warmup uses predictable constant arrivals; profiling uses gamma arrivals with reduced variance (smoothness > 1 = smoother than Poisson).

## Warmup with Ramping

Warmup can include its own gradual ramp-up:

```bash
aiperf profile \
    --model your-model \
    --url localhost:8000 \
    --endpoint-type chat \
    --streaming \
    --concurrency 100 \
    --concurrency-ramp-duration 30 \
    --warmup-concurrency 50 \
    --warmup-concurrency-ramp-duration 10 \
    --warmup-request-count 200 \
    --benchmark-duration 120
```

**Sample Output (Successful Run):**
```
NOTICE   Phase warmup started | target: 200 requests
Warming Up: 200/200 |████████████████████████| 100% [00:15<00:00]
NOTICE   Phase warmup complete | completed=200, cancelled=0, errors=0 | elapsed=15.18s
NOTICE   Phase profiling started | target: 120.0s duration
Profiling: [02:00] - Running for 120 seconds...
NOTICE   Phase profiling complete | completed=11423, cancelled=0, errors=0 | elapsed=120.04s
INFO     Results saved to: artifacts/your-model-openai-chat-concurrency100/
JSON Export: artifacts/your-model-openai-chat-concurrency100/profile_export_aiperf.json
```

**Timeline:**

```
    Warmup Phase (ramps 1→50 over 10s)     Profiling Phase (ramps 1→100 over 30s)
    ─────────────────────────────────────  ──────────────────────────────────────────
                   ●━━━━━━ 50                                              ●━━━━━━ 100
              ●────┘                                                  ●────┘
         ●────┘                                                  ●────┘
    ●────┘                                                  ●────┘
    └──────────┬────────────────────────┬──────────────────────────────────────────▶
              10s                    Warmup ends              30s                Time
                                   (wait for responses)
```

## Grace Period

By default, AIPerf waits indefinitely for all warmup responses before starting profiling. When using duration-based warmup (`--warmup-duration`), you can limit this wait:

```bash
# Wait max 10 seconds for stragglers after warmup requests sent
--warmup-grace-period 10
```

This prevents slow warmup responses from delaying the profiling phase indefinitely.

## Multi-Turn Warmup

For multi-turn benchmarks, warmup by session count ensures complete conversations:

```bash
aiperf profile \
    --model your-model \
    --url localhost:8000 \
    --endpoint-type chat \
    --streaming \
    --session-turns-mean 5 \
    --num-warmup-sessions 10 \
    --request-count 500
```

This completes 10 full conversations (each ~5 turns) before profiling begins.

## Prefill Concurrency Warmup

When using [prefill concurrency](./prefill-concurrency.md) to limit simultaneous prefill operations, you can configure warmup separately:

```bash
aiperf profile \
    --model your-model \
    --url localhost:8000 \
    --endpoint-type chat \
    --streaming \
    --concurrency 50 \
    --prefill-concurrency 5 \
    --warmup-concurrency 20 \
    --warmup-prefill-concurrency 2 \
    --warmup-request-count 50 \
    --benchmark-duration 120
```

Warmup runs with lower limits (20 concurrent, 2 prefill), then profiling uses full limits.

## Examples

### Minimal Warmup

Just warm up connections and caches:

```bash
aiperf profile \
    --model Qwen/Qwen2.5-7B-Instruct \
    --url localhost:8000 \
    --endpoint-type chat \
    --streaming \
    --concurrency 50 \
    --warmup-request-count 20 \
    --request-count 500
```

### Production-Like Warmup

Simulate gradual traffic increase:

```bash
aiperf profile \
    --model Qwen/Qwen2.5-7B-Instruct \
    --url localhost:8000 \
    --endpoint-type chat \
    --streaming \
    --request-rate 100 \
    --concurrency 200 \
    --concurrency-ramp-duration 60 \
    --warmup-request-rate 20 \
    --warmup-concurrency 50 \
    --warmup-concurrency-ramp-duration 15 \
    --warmup-duration 30 \
    --benchmark-duration 300
```

### Long-Context Warmup

For long prompts, use lower warmup concurrency to avoid OOM:

```bash
aiperf profile \
    --model Qwen/Qwen2.5-7B-Instruct \
    --url localhost:8000 \
    --endpoint-type chat \
    --streaming \
    --synthetic-input-tokens-mean 32000 \
    --output-tokens-mean 500 \
    --concurrency 20 \
    --prefill-concurrency 3 \
    --warmup-concurrency 5 \
    --warmup-prefill-concurrency 1 \
    --warmup-request-count 10 \
    --benchmark-duration 120
```

## CLI Reference

### Stop Conditions (at least one required for warmup)

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--warmup-request-count` | int | None | Stop warmup after this many requests (alias: `--num-warmup-requests`, GenAI-Perf compat) |
| `--num-warmup-sessions` | int | None | Stop **starting new** warmup sessions after this many; in-flight sessions complete their remaining turns |
| `--warmup-duration` | float | None | Stop warmup after this many seconds |

### Load Settings (inherit from profiling if not set)

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--warmup-concurrency` | int | `--concurrency` | Concurrency during warmup |
| `--warmup-prefill-concurrency` | int | `--prefill-concurrency` | Prefill concurrency during warmup |
| `--warmup-request-rate` | float | `--request-rate` | Request rate during warmup |
| `--warmup-arrival-pattern` | str | `--arrival-pattern` | Arrival pattern during warmup |

### Ramping (inherit from profiling if not set)

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--warmup-concurrency-ramp-duration` | float | `--concurrency-ramp-duration` | Ramp duration for warmup concurrency |
| `--warmup-prefill-concurrency-ramp-duration` | float | `--prefill-concurrency-ramp-duration` | Ramp duration for warmup prefill |
| `--warmup-request-rate-ramp-duration` | float | `--request-rate-ramp-duration` | Ramp duration for warmup rate |

### Other

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--warmup-grace-period` | float | ∞ | Max seconds to wait for warmup responses after stop condition. Requires `--warmup-duration`. |
| `--profile-run-disable-warmup-after-first` | bool | True | Multi-run only (`--num-profile-runs > 1`): when True (default), only the first run includes warmup; subsequent runs measure pure steady-state. Pass `--no-profile-run-disable-warmup-after-first` to include warmup on every run. |

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Warmup takes too long | Grace period waiting for slow responses | Set `--warmup-grace-period` |
| Cold-start still visible | Insufficient warmup | Increase `--warmup-request-count` or `--warmup-duration` |
| OOM during warmup | Warmup concurrency too high | Lower `--warmup-concurrency` and `--warmup-prefill-concurrency` |
| Warmup not running | No warmup trigger set | Add `--warmup-request-count`, `--num-warmup-sessions`, or `--warmup-duration` |

## Related Documentation

- [Gradual Ramping](./ramping.md) — Smooth ramp-up for concurrency and rate
- [Prefill Concurrency](./prefill-concurrency.md) — Memory-safe long-context benchmarking
- [Timing Modes Reference](../benchmark-modes/timing-modes-reference.md) — Complete CLI compatibility matrix
