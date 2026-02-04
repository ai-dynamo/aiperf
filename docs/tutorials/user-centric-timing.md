<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# User-Centric Timing for KV Cache Benchmarking

![User-centric rate mode](../media/images/user-centric-rate.png)

## When to Use This Mode

Use user-centric timing when you need to:

- **Control per-user turn gaps precisely** — Each user waits at least `num_users / QPS` seconds between their turns, enabling controlled cache TTL testing
- **Simulate steady-state from the start** — Virtual history creates an immediate mix of new and continuing users (no cold-start transient)
- **Per-user timing independence** — Each user maintains their own schedule, not affected by other users' response times
- **Measure prefix caching benefits** — Quantify TTFT improvements when a shared system prompt is cached across all users

### The Real-World Scenario

Imagine a customer support chatbot serving 15 concurrent users. Each user:
1. Sends a question
2. Reads the response (takes ~15 seconds)
3. Sends a follow-up question
4. Repeats for ~20 turns until their issue is resolved

User-centric timing recreates this pattern with **controlled, consistent timing**. You can test whether your KV cache retains entries for exactly 15 seconds, 30 seconds, or any specific gap—something request-rate mode doesn't guarantee because continuation turns are issued at the next available rate interval rather than after a fixed per-user delay.

### Contrast with Other Modes

| Mode | Turn Timing | Startup Behavior | Best For |
|------|-------------|------------------|----------|
| **User-centric rate** | Fixed per-user gap (`num_users/QPS`) | Steady-state via virtual history | KV cache TTL testing, controlled multi-turn timing |
| **Request rate** | Next turn at next rate interval (variable per-user gap) | Cold start (all sessions start fresh) | Throughput testing, arrival pattern simulation |
| **Concurrency** | Immediate (maintain N in-flight) | Cold start | Max throughput discovery, stress testing |

## Quick Start

```bash
aiperf profile \
    --model your-model \
    --url localhost:8000 \
    --endpoint-type chat \
    --streaming \
    --user-centric-rate 1.0 \
    --num-users 15 \
    --session-turns-mean 20 \
    --shared-system-prompt-length 1000 \
    --user-context-prompt-length 20000 \
    --synthetic-input-tokens-mean 26 \
    --osl 100 \
    --num-dataset-entries 1000 \
    --benchmark-duration 100
```

This configures 15 simulated users with sessions averaging 20 turns:

- **Turn gap**: 15 users / 1.0 req/s = 15 seconds between each user's turns
- **System throughput**: ~1.0 requests/second across all users
- **Shared system prompt**: 1000 tokens shared across ALL users (KV cache prefix)
- **User context**: 20000 tokens unique per user (synthetic padding to simulate context length)
- **Per-turn input**: 26 tokens (the new question each turn)

## Key Parameters

| Parameter | Description |
|-----------|-------------|
| `--user-centric-rate` | Target requests per second (QPS) across all users (enables user-centric mode) |
| `--num-users` | Number of concurrent simulated users |
| `--session-turns-mean` | Mean number of conversation turns per user (must be >= 2) |

## Why This Mode Enables Accurate KV Cache Measurement

In request-rate mode, after a turn completes, the next turn is queued and issued at the next rate interval. This means per-user turn gaps vary depending on when the previous turn finished relative to the rate clock—making it hard to test specific cache TTL thresholds.

User-centric timing solves this with **fixed per-user turn gaps**:

| Feature | Why It Matters for Cache Measurement |
|---------|--------------------------------------|
| **Fixed turn gap per user** | Each user's turns are spaced at least `num_users / QPS` seconds apart (exactly this interval when responses complete before the scheduled time). A 15-second gap tests whether your cache retains entries for 15+ seconds. |
| **Per-user independent scheduling** | User A's timing isn't affected by User B's slow response. Each user maintains their own schedule. |
| **Deterministic scheduling** | Same benchmark configuration = same request timing = reproducible results across runs. |
| **Steady-state from t=0** | Virtual history simulates an already-running system, so metrics aren't skewed by cold-start transients from all users starting at Turn 0 simultaneously. |

## How It Works

### Turn Gap Calculation

The gap between each user's requests is:

```
turn_gap = num_users / user_centric_rate
```

| Users | Request Rate | Turn Gap |
|-------|--------------|----------|
| 15    | 1.0 req/s    | 15.0s    |
| 15    | 0.5 req/s    | 30.0s    |
| 15    | 4.0 req/s    | 3.75s    |
| 15    | 8.0 req/s    | 1.875s   |

### Steady-State from the Start

User-centric mode uses "virtual history" to simulate steady-state behavior immediately. Instead of all users starting at turn 0 simultaneously, users are assigned virtual "ages" at startup—creating an immediate mix of new users and continuations that simulates joining an already-running system.

```
Evaluate: Benchmark Execution Timeline (t=0 to t=30s)
---------------------------------------------------------------------
TIME (s) >>>   0   1   2   3   4   5   6   7   8   9  10  11  12 ...

EVENT:
t=0: User 1 (virtually done) LEAVES instantly.
t=0: User 16 ENTERS instantly to replace User 1.

ACTUAL TURNS REMAINING (Visualized):
User 16 (New): ████████████████████████████████████████ (20 turns)
User  5      :  ████████████ (6 turns)
User  9      :   ██████████████████████ (11 turns)
User 13      :    ████████████████████████████████ (16 turns)
User  2      :     ████ (2 turns - finishes quickly)
User  6      :      ██████████████ (7 turns)
User 10      :       ████████████████████████ (12 turns)
User 14      :        ██████████████████████████████████ (17 turns)
... (remaining users follow staggered pattern) ...

RESULT:
Immediate mix of fresh sessions (User 16) and deep sessions (User 14),
with users finishing and churning naturally from t=6s onwards.
```

### Handling Slow Responses

When a response takes longer than the turn gap, the scheduler:

- Sends the next turn immediately when the response arrives
- Resets the timing baseline to "now" for subsequent turns
- Maintains the turn gap minimum going forward

This avoids burst load from catching up to the original schedule.

## Prompt Configuration

For effective KV cache benchmarking, configure prompts to create realistic prefix sharing patterns:

```
┌─────────────────────────────────────────────────────────────┐
│ Shared System Prompt (1000 tokens)                          │ ← Same across ALL users
│ "You are a helpful assistant..."                            │   (KV cache shared prefix)
├─────────────────────────────────────────────────────────────┤
│ User Context Prompt (20000 tokens)                          │ ← Unique per user
│ [synthetic text representing prior conversation context]    │   (unique prefix per user)
├─────────────────────────────────────────────────────────────┤
│ Per-Turn Input (26 tokens)                                  │ ← New content each turn
│ "What is the weather today?"                                │   (the actual question)
└─────────────────────────────────────────────────────────────┘
```

**Note**: In multi-turn conversations, previous turns (inputs + responses) also accumulate in the request, growing the total prompt size with each turn. The user context prompt is synthetic padding separate from this accumulated history—both contribute to the total context length.

| Option | Description | Typical Value |
|--------|-------------|---------------|
| `--shared-system-prompt-length` | System prompt shared across ALL users (enables prefix sharing) | 1000 |
| `--user-context-prompt-length` | Per-user unique prefix (synthetic text representing conversation history) | 20000 |
| `--synthetic-input-tokens-mean` | Per-turn input tokens (the question) | 26 |
| `--osl` | Output sequence length (answer tokens) | 100 |
| `--num-dataset-entries` | Required when using `--user-context-prompt-length` | ≥1000 (recommended) |

## Concurrency

**Important**: User-centric mode does NOT automatically limit concurrency. While the timing model spaces out requests, slow server responses can cause request buildup.

To prevent overwhelming the server, you can cap concurrency with `--concurrency`. If you set this, use a value at least equal to `--num-users` to avoid constraining user sessions.

```bash
# Cap concurrency to num_users
aiperf profile \
    --user-centric-rate 1.0 \
    --num-users 15 \
    --concurrency 15 \
    --model your-model \
    --url localhost:8000
```

## Examples

### Complete KV Cache Benchmark

```bash
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --url localhost:8000 \
    --endpoint-type chat \
    --streaming \
    --user-centric-rate 1.0 \
    --num-users 15 \
    --session-turns-mean 20 \
    --shared-system-prompt-length 1000 \
    --user-context-prompt-length 20000 \
    --synthetic-input-tokens-mean 26 \
    --osl 100 \
    --num-dataset-entries 1000 \
    --benchmark-duration 100 \
    --random-seed 42
```

**Sample Output (Successful Run):**
```
INFO     Starting AIPerf System
INFO     User-centric mode: 15 users, 1.0 req/s (15.0s turn gap per user)
INFO     Shared system prompt: 1000 tokens
INFO     User context: 20000 tokens per user
INFO     AIPerf System is PROFILING

Profiling: [01:40] - Running for 100 seconds...

INFO     Benchmark completed successfully
INFO     Results saved to: artifacts/Qwen_Qwen3-0.6B-chat-rate1.0/

            NVIDIA AIPerf | LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┓
┃                      Metric ┃     avg ┃     min ┃     max ┃     p99 ┃     p50 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━┩
│        Request Latency (ms) │ 3456.78 │ 2890.34 │ 4123.45 │ 3998.67 │ 3423.12 │
│    Time to First Token (ms) │ 1234.56 │  987.89 │ 1567.90 │ 1498.23 │ 1212.34 │
│    Inter Token Latency (ms) │   21.45 │   17.89 │   28.34 │   27.12 │   21.01 │
│ Output Token Count (tokens) │  100.00 │   90.00 │  110.00 │  109.00 │   99.00 │
│  Request Throughput (req/s) │    0.98 │       - │       - │       - │       - │
└─────────────────────────────┴─────────┴─────────┴─────────┴─────────┴─────────┘

JSON Export: artifacts/Qwen_Qwen3-0.6B-chat-rate1.0/profile_export_aiperf.json
```

- **15-second gaps** between each user's turns (15 / 1.0 = 15s)
- **1000-token shared system prompt** (prefix shared across ALL users)
- **20000-token user context** (unique per user)

### High Throughput Cache Test

Test with higher QPS (shorter per-user gaps):

```bash
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --url localhost:8000 \
    --endpoint-type chat \
    --streaming \
    --user-centric-rate 4.0 \
    --num-users 15 \
    --session-turns-mean 20 \
    --shared-system-prompt-length 1000 \
    --user-context-prompt-length 20000 \
    --synthetic-input-tokens-mean 26 \
    --osl 100 \
    --num-dataset-entries 1000 \
    --benchmark-duration 100
```

**Sample Output (Successful Run):**
```
INFO     Starting AIPerf System
INFO     User-centric mode: 15 users, 4.0 req/s (3.75s turn gap per user)
INFO     Shared system prompt: 1000 tokens
INFO     User context: 20000 tokens per user
INFO     AIPerf System is PROFILING

Profiling: [01:40] - Running for 100 seconds...

INFO     Benchmark completed successfully
INFO     Results saved to: artifacts/Qwen_Qwen3-0.6B-chat-rate4.0/

            NVIDIA AIPerf | LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┓
┃                      Metric ┃     avg ┃     min ┃     max ┃     p99 ┃     p50 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━┩
│        Request Latency (ms) │ 3234.56 │ 2678.90 │ 3890.12 │ 3798.45 │ 3198.67 │
│    Time to First Token (ms) │ 1145.67 │  912.34 │ 1456.89 │ 1389.23 │ 1123.45 │
│    Inter Token Latency (ms) │   20.34 │   16.78 │   26.90 │   25.67 │   20.01 │
│ Output Token Count (tokens) │  100.00 │   90.00 │  110.00 │  109.00 │   99.00 │
│  Request Throughput (req/s) │    3.89 │       - │       - │       - │       - │
└─────────────────────────────┴─────────┴─────────┴─────────┴─────────┴─────────┘

JSON Export: artifacts/Qwen_Qwen3-0.6B-chat-rate4.0/profile_export_aiperf.json
```

Gap = 15 / 4.0 = **3.75 seconds** between each user's requests.

### Low QPS Cache TTL Test

Test cache TTL limits with 30-second per-user gaps:

```bash
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --url localhost:8000 \
    --endpoint-type chat \
    --streaming \
    --user-centric-rate 0.5 \
    --num-users 15 \
    --session-turns-mean 20 \
    --shared-system-prompt-length 1000 \
    --user-context-prompt-length 20000 \
    --synthetic-input-tokens-mean 26 \
    --osl 100 \
    --num-dataset-entries 1000 \
    --benchmark-duration 300
```

**Sample Output (Successful Run):**
```
INFO     Starting AIPerf System
INFO     User-centric mode: 15 users, 0.5 req/s (30.0s turn gap per user)
INFO     Shared system prompt: 1000 tokens
INFO     User context: 20000 tokens per user
INFO     AIPerf System is PROFILING

Profiling: [05:00] - Running for 300 seconds...

INFO     Benchmark completed successfully
INFO     Results saved to: artifacts/Qwen_Qwen3-0.6B-chat-rate0.5/

            NVIDIA AIPerf | LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┓
┃                      Metric ┃     avg ┃     min ┃     max ┃     p99 ┃     p50 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━┩
│        Request Latency (ms) │ 3567.89 │ 2956.78 │ 4234.56 │ 4098.23 │ 3512.34 │
│    Time to First Token (ms) │ 1289.45 │ 1023.67 │ 1598.90 │ 1534.12 │ 1267.89 │
│    Inter Token Latency (ms) │   21.89 │   18.23 │   29.12 │   28.01 │   21.56 │
│ Output Token Count (tokens) │  100.00 │   90.00 │  110.00 │  109.00 │   99.00 │
│  Request Throughput (req/s) │    0.49 │       - │       - │       - │       - │
└─────────────────────────────┴─────────┴─────────┴─────────┴─────────┴─────────┘

JSON Export: artifacts/Qwen_Qwen3-0.6B-chat-rate0.5/profile_export_aiperf.json
```

Gap = 15 / 0.5 = **30 seconds** between each user's requests.

## Interpreting Results

### Key Metrics for Cache Benchmarking

| Metric | What It Tells You |
|--------|-------------------|
| **TTFT (Time to First Token)** | Lower TTFT on subsequent turns indicates cache hits |
| **TTFT by Turn Index** | Compare Turn 0 vs Turn 1+ to measure cache benefit |
| **Throughput** | Higher throughput with caching enabled indicates cache effectiveness |

### Expected Patterns

**With effective caching:**
- Turn 0 (first turn): Higher TTFT (cache miss, full prefill)
- Turn 1+: Lower TTFT (cache hit, reduced prefill)

**Without caching or cache misses:**
- Similar TTFT across all turns
- Higher variance in TTFT

## Troubleshooting

### Requests Not Following Expected Timing

1. Verify `--user-centric-rate` is set (not `--request-rate`)
2. Confirm `--num-users` is specified
3. Check if response latencies exceed the turn gap (triggers schedule reset)

### Cache Not Being Hit

**Possible causes:**
1. Cache TTL shorter than your gap interval
2. Cache not enabled on the server
3. No shared system prompt configured

**Solutions:**
1. Reduce gap by increasing `--user-centric-rate` or decreasing `--num-users`
2. Verify server cache configuration
3. Use `--shared-system-prompt-length` to enable prefix sharing

### High Variance in Results

1. Use `--random-seed` for reproducible dataset sampling
2. Increase `--benchmark-duration` for more samples
3. Ensure server is warmed up before benchmarking

## Incompatible Options

| Option | Reason |
|--------|--------|
| `--request-rate` | Use `--user-centric-rate` instead |
| `--arrival-pattern` | User-centric mode uses deterministic scheduling |

## References

- [Multi-Turn Tutorial](multi-turn.md) — General multi-turn conversation benchmarking
