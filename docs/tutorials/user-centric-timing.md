<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# User-Centric Timing for KV Cache Benchmarking

User-centric timing mode provides per-user rate limiting designed specifically for KV cache benchmarking. This mode models realistic multi-user behavior where each user maintains their own request cadence, making it ideal for measuring cache effectiveness.

## Overview

In user-centric mode, each user (session) operates independently with a fixed time gap between their requests:

```
gap = num_users / qps
```

For example, with 15 users at 1.0 QPS:
- **System-wide target**: 1.0 requests/second total
- **Per-user gap**: 15 / 1.0 = 15 seconds between each user's requests

This models realistic behavior where users wait for responses before continuing, rather than artificial load patterns where requests arrive independently of user state.

## Why User-Centric Timing for KV Cache?

KV cache systems store computed key-value pairs from previous requests to accelerate subsequent requests from the same user/conversation. Cache effectiveness depends critically on:

1. **Consistent per-user timing** — Cache entries have TTLs; predictable request intervals help measure cache hit rates at specific time gaps
2. **No request interleaving** — Each user blocks on their previous request, matching how real users interact with LLMs
3. **Reproducible patterns** — Consistent timing enables meaningful comparisons across benchmark runs

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

This creates 15 simulated users, each sending 20 turns:

- **Gap**: 15 users / 1.0 req/s = 15 seconds between each user's turns
- **Stagger**: 1 / 1.0 req/s = 1 second between each user's first turn
- **Startup sequence**: User 0 at t=0s, User 1 at t=1s, User 2 at t=2s, ... User 14 at t=14s
- **System throughput**: ~1.0 requests/second (one request every second)
- **Shared system prompt**: 1000 tokens shared across ALL users (KV cache prefix)
- **User context**: 20000 tokens unique per user (simulated chat history)
- **Per-turn input**: 26 tokens (the new question each turn)

The stagger ensures requests are evenly distributed from the start, avoiding a thundering herd of all users sending simultaneously.

> **Note**: User-centric mode uses `--num-users` to specify the number of concurrent users. Each user has exactly one request in flight at a time, so maximum concurrency equals `--num-users`. You can optionally use `--concurrency` as an additional limiter, but it's typically unnecessary.

## Timing Model

### Virtual History & Start Order

Simulates steady-state from t=0 by distributing users across the "session lifetime"
(the time from a user's first turn to their last, measured in gaps = session_turns - 1).

Each user is assigned a virtual "age" representing how far through their session they are:
- User 1 (oldest): virtually done - all turns completed before t=0, replaced immediately
- User N (youngest): just started - most turns remaining

The user who just finished (User 1) is replaced by a fresh user who fires first at t=0.
Other users fire in staggered order based on their position in the session lifetime.
This creates immediate user churn rather than waiting for the first natural completions.

#### Example: 15 users, 20 turns, 1.0 QPS

```
-------------------------------------------
 User | Turns | Time | Turn Visualization
-------------------------------------------
    1 |     - |    - | (All turns completed before t=0) ← User 1 is "virtually done"
   16 |    20 |   0s | ████████████████████ ← New user at t=0 with all turns remaining
    5 |     6 |   1s | ██████
    9 |    11 |   2s | ███████████
   13 |    16 |   3s | ████████████████
    2 |     2 |   4s | ██
    6 |     7 |   5s | ███████
   10 |    12 |   6s | ████████████
   14 |    17 |   7s | █████████████████
    3 |     3 |   8s | ███
    7 |     8 |   9s | ████████
   11 |    13 |  10s | █████████████
   15 |    18 |  11s | ██████████████████
    4 |     4 |  12s | ████
    8 |     9 |  13s | █████████
   12 |    14 |  14s | ██████████████
```

### Gap Calculation

The gap between each user's requests is calculated as:

```
gap = num_users / request_rate
```

| Users | Request Rate | Gap (seconds) |
|-------|--------------|---------------|
| 15    | 1.0 req/s    | 15.0          |
| 15    | 0.5 req/s    | 30.0          |
| 15    | 4.0 req/s    | 3.75          |
| 15    | 8.0 req/s    | 1.875         |

### Initial Stagger

First turns are staggered to distribute load evenly:

```
stagger = 1 / request_rate
```

With 15 users at 1.0 QPS (gap=15s, stagger=1.0s):

```
User 0:   t = 0.0s
User 1:   t = 1.0s
User 2:   t = 2.0s
...
User 14:  t = 14.0s
User 0:   t = 15.0s  (second turn)
User 1:   t = 16.0s  (second turn)
...
```

This ensures uniform request distribution from the start, avoiding thundering herd effects.

### Subsequent Turn Scheduling

After a request completes, the next turn is scheduled based on when the previous turn was **sent** (not when it completed):

```
next_eligible_time = last_send_time + gap
```

**Two cases:**

1. **On-time** (request completed before gap elapsed): Wait until `next_eligible_time`, then send.

2. **Catch-up** (request took longer than gap): Send immediately. The baseline resets to current time for subsequent turns.

## Catch-Up Behavior

When a request exceeds the gap duration, we reset to actual time rather than attempting to recover the original schedule:

```
Example: gap = 10s, request takes 25s

Original schedule: t=0, t=10, t=20, t=30...

Actual behavior:
    t=0:   Send Turn 1
    t=25:  Turn 1 completes (missed t=10, t=20 slots)
           Send Turn 2 immediately
           Reset baseline: last_send = 25
    t=35:  Next eligible (25 + 10)
```

### Why Reset Instead of Catching Up?

**No burst load**: Catching up to the original schedule would require sending multiple requests rapidly after a slow request, creating unpredictable server load spikes.

**KV cache reality**: Cache state is determined by actual request timing. If a user's last request was 25 seconds ago, the cache has already aged 25 seconds—rapid catch-up requests don't change this.

**Realistic user behavior**: Real users don't "catch up"—they wait for responses and continue from where they are. This models actual user interaction patterns.

**Measurement accuracy**: The benchmark measures what actually happened. Slow requests already affected results; masking this by catching up would distort measurements.

## Prompt Configuration for KV Cache

For effective KV cache benchmarking, configure prompts to create realistic prefix sharing patterns:

```
┌─────────────────────────────────────────────────────────────┐
│ Shared System Prompt (1000 tokens)                          │ ← Same across ALL users
│ "You are a helpful assistant..."                            │   (KV cache shared prefix)
├─────────────────────────────────────────────────────────────┤
│ User Context Prompt (20000 tokens)                          │ ← Unique per user
│ "Previous conversation history..."                          │   (simulates chat history)
├─────────────────────────────────────────────────────────────┤
│ Per-Turn Input (26 tokens)                                  │ ← New content each turn
│ "What is the weather today?"                                │   (the actual question)
└─────────────────────────────────────────────────────────────┘
```

### Key Prompt Options

| Option | Description |
|--------|-------------|
| `--shared-system-prompt-length <N>` | Tokens for system prompt shared across ALL users. Creates prefix sharing for KV cache. |
| `--user-context-prompt-length <N>` | Tokens for per-user context (unique per session). Simulates accumulated chat history. |
| `--synthetic-input-tokens-mean <N>` | Tokens per turn (the new question). Also: `--isl` |
| `--osl <N>` | Output sequence length (answer tokens per turn) |
| `--num-dataset-entries <N>` | Required when using `--user-context-prompt-length`. Each entry gets unique context. |

### Why This Structure Matters

1. **Shared system prompt**: Creates prefix sharing across ALL users. The KV cache can reuse these computed key-values for every request, regardless of user.

2. **User context prompt**: Simulates realistic chat history that accumulates over a conversation. This is unique per user but consistent across their turns.

3. **Per-turn input**: The new question added each turn. Keep this small relative to context to measure cache effectiveness.

### Recommended Values

| Parameter | Typical Value | Purpose |
|-----------|---------------|---------|
| `--shared-system-prompt-length` | 1000 | System instructions shared by all users |
| `--user-context-prompt-length` | 20000 | Simulated chat history per user |
| `--synthetic-input-tokens-mean` | 26 | Small per-turn question |
| `--osl` | 100 | Answer length per turn |

## Configuration Options

### Required Options

| Option | Description |
|--------|-------------|
| `--user-centric-rate <N>` | Enable user-centric timing with target QPS (system-wide) |
| `--num-users <N>` | Number of concurrent simulated users |
| `--session-turns-mean <N>` | Mean number of turns per user session (must be >= 2) |

### Recommended Options

| Option | Description |
|--------|-------------|
| `--shared-system-prompt-length <N>` | System prompt shared across all users (enables KV cache prefix sharing) |
| `--user-context-prompt-length <N>` | Per-user context prompt (requires `--num-dataset-entries`) |
| `--random-seed <N>` | Seed for reproducible dataset sampling |

### Incompatible Options

| Option | Reason |
|--------|--------|
| `--request-rate` | Use `--user-centric-rate` instead |
| `--arrival-pattern` | User-centric mode uses deterministic stagger-based scheduling |

### Concurrency Options

| Option | Behavior |
|--------|----------|
| `--concurrency` | **Important**: Controls max concurrent sessions. Without this, session concurrency is **unlimited**. Set to `--num-users` or higher to allow all users to have in-flight requests simultaneously. |

> **Note**: User-centric mode does NOT automatically set concurrency. While the timing model spaces out requests, slow server responses can still cause request buildup. Setting `--concurrency` equal to or greater than `--num-users` ensures all users can have their requests in flight without blocking on concurrency limits.

## Examples

### Complete KV Cache Benchmark

Full configuration with shared system prompt and user context for realistic KV cache testing:

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

This creates:
- **15 users** with **20 turns** each
- **15-second gaps** between each user's turns (15 / 1.0 = 15s)
- **1000-token shared system prompt** (prefix shared across ALL users)
- **20000-token user context** (unique per user, simulates chat history)
- **26-token per-turn input** (the new question each turn)
- **100-token output** per turn

### High Throughput Cache Test

Test with higher QPS (shorter per-user gaps for aggressive caching):

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

Here, gap = 15 / 4.0 = 3.75 seconds between each user's requests.

### Low QPS Cache TTL Test

Test cache with 30-second per-user gaps (tests cache TTL limits):

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

Here, gap = 15 / 0.5 = 30 seconds between each user's requests. Useful for testing cache TTL expiration.

### Maximum Throughput Test

Stress test with 8.0 QPS (very short gaps):

```bash
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --url localhost:8000 \
    --endpoint-type chat \
    --streaming \
    --user-centric-rate 8.0 \
    --num-users 15 \
    --session-turns-mean 20 \
    --shared-system-prompt-length 1000 \
    --user-context-prompt-length 20000 \
    --synthetic-input-tokens-mean 26 \
    --osl 100 \
    --num-dataset-entries 1000 \
    --benchmark-duration 60
```

Here, gap = 15 / 8.0 = 1.875 seconds between each user's requests.

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
- Consistent TTFT on subsequent turns within same user session

**Without caching or cache misses:**
- Similar TTFT across all turns
- Higher variance in TTFT for subsequent turns

## Troubleshooting

### Requests Not Following Expected Timing

**Symptom**: Actual request timing doesn't match expected gaps

**Check**:
1. Verify `--user-centric-rate` is set
2. Confirm `--num-users` is specified
3. Check logs for catch-up warnings (requests taking longer than gap)

### Cache Not Being Hit

**Symptom**: TTFT similar across all turns

**Possible causes**:
1. Cache TTL shorter than your gap interval
2. Cache not enabled on the server
3. No shared system prompt configured (no prefix sharing)

**Solutions**:
1. Reduce gap by increasing `--user-centric-rate` or decreasing `--num-users`
2. Verify server cache configuration
3. Use `--shared-system-prompt-length` to enable prefix sharing (see [Prompt Configuration](#prompt-configuration-for-kv-cache))

### High Variance in Results

**Symptom**: Results vary significantly between runs

**Solutions**:
1. Use `--random-seed` for reproducible dataset sampling
2. Increase `--request-count` for more samples
3. Ensure server is warmed up before benchmarking

## Technical Details

### Event-Driven Architecture

AIPerf uses an event-driven model for precise timing:

1. **First turns**: Sent with staggered timing (`stagger = 1 / qps`)
2. **Credit return**: When a request completes, a "credit" is returned
3. **Next turn scheduling**: On credit return, schedule the next turn at `last_send_time + gap`
4. **Scheduler-based timing**: Uses internal scheduler for precise timestamp-based scheduling

This schedules requests at calculated times with ~1ms precision.

### Timing Baseline Management

The next turn time is computed using `max()` to handle both cases in a single expression:

```python
next_send_time = max(current_time, last_send_time + turn_gap)
```

**On-time** (response arrived before gap elapsed):
- `last_send_time + turn_gap` is in the future
- `max()` returns `last_send_time + turn_gap`
- Request is scheduled for that future time

**Catch-up** (response took longer than gap):
- `last_send_time + turn_gap` is already in the past
- `max()` returns `current_time`
- Request is sent immediately, baseline resets to now

This maintains slot-based timing when on schedule, but properly resets when falling behind.

## References

- [User-Centric Rate Mode Guide](../benchmark_modes/user-centric-rate.md) — Detailed timing mode documentation
- [Multi-Turn Tutorial](multi-turn.md) — General multi-turn conversation benchmarking
