<!--
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# User-Centric Rate Mode Guide

This guide explains user-centric rate mode for benchmarking KV cache performance with realistic multi-turn chat patterns.

## Overview

User-centric rate mode simulates realistic multi-turn chat patterns where multiple users interact with an LLM service concurrently. Each user maintains their own conversation with consistent timing between their turns.

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| `--user-centric-rate` | Target requests per second (QPS) across all users (enables user-centric mode) |
| `--num-users` | Number of concurrent simulated users |
| `--session-turns-mean` | Mean number of conversation turns per user (must be >= 2) |

## Core Timing Concepts

User-centric rate mode uses precise, pre-calculated scheduling to ensure collision-free request timing.

### Turn Gap (Time Between a User's Turns)

The **turn gap** is the minimum time between a single user's consecutive requests:

```
turn_gap = num_users / request_rate
```

This represents the "think time" between messages in a chat session. With 15 users at 1.0 QPS, the turn gap is 15 seconds. Each user waits at least 15 seconds between their turns, allowing all 15 users to share the 1.0 QPS rate.

### Stagger Slots

All request times are **pre-calculated at setup** using stagger slot math:

```
stagger = 1 / request_rate
user_offset = slot_index * stagger
```

With 15 users at 1.0 QPS:
- Stagger = 1 / 1.0 = 1 second
- User 0 fires at offset 0s, User 1 at 1s, User 2 at 2s, etc.

### Spawn Timing (When New Users Join)

New users join to maintain steady-state as existing users complete their conversations. Spawn times are derived from the stagger math:

```
next_spawn_time = current_spawn_time + (max_turns * turn_gap)
```

Each stagger slot's next spawn time is calculated from the previous user's spawn time plus their lifetime (`max_turns * turn_gap`). This ensures exactly `turn_gap` spacing between a user's last turn and the next user's first turn in that slot.

### Collision-Free Scheduling

Since spawn times are derived from the same stagger slot math as turn times, they never overlap - collisions are mathematically impossible (unless response delays shift schedules).

The result:
- No two requests are ever scheduled at the same moment
- New user spawns and continuation turns never coincide
- The collision-free property holds from the first request and continues indefinitely

## Timing Behavior

### Schedule Characteristics

| Aspect | Behavior |
|--------|----------|
| Request timing precision | ~1ms (uvloop timer resolution) |
| New user / continuation collisions | None when responses arrive on-time |
| Turn gap enforcement | Always maintained |
| Scheduling method | Pre-calculated timestamps |

### Schedule Adjustment When Responses Are Slow

The scheduler ensures the turn_gap minimum is maintained even when responses take longer than expected:

- Uses `max(now, last_send_time + turn_gap)` for the next turn time
- If response arrives early: waits until last_send_time + turn_gap
- If response arrives late: re-aligns schedule to now (immediate send)
- Maintains tight timing precision when responses finish on-time

**Example:**
```
turn_gap = 7.5s
Turn 0 latency = 65s (much longer than turn_gap)

Without schedule adjustment:
  Turn 1 scheduled at Turn 0 start + 7.5s = 7.5s
  Turn 1 actually fires at 65s (when Turn 0 finishes)
  Turn 2 scheduled at 7.5s + 7.5s = 15s (already passed!)
  Turn 2 would fire immediately -> gap between Turn 1 and Turn 2 < turn_gap

With schedule adjustment:
  Turn 1 fires at 65s, schedule realigns
  Turn 2 scheduled at 65s + 7.5s = 72.5s
  Gap between Turn 1 and Turn 2 = 7.5s = turn_gap
```

## Virtual History

User-centric rate mode uses "virtual history" to simulate steady-state behavior from the start of the benchmark.

Without virtual history:
- All users start at turn 0
- No user completions until `(turns - 1) * turn_gap` seconds pass
- Initial period has only new users, no continuations

With virtual history:
- Users are assigned virtual "ages" at startup
- Some users appear to have already completed turns before t=0
- Creates immediate mix of new users and continuations
- Simulates joining an already-running system

This ensures stagger integrity from t=0, providing realistic steady-state simulation from the first request.

## Prompt Configuration

For KV cache benchmarking, configure prompts to create realistic prefix sharing:

| Option | Description | Typical Value |
|--------|-------------|---------------|
| `--shared-system-prompt-length` | System prompt shared across ALL users (enables prefix sharing) | 1000 |
| `--user-context-prompt-length` | Per-user context (simulates chat history) | 20000 |
| `--synthetic-input-tokens-mean` | Per-turn input tokens (the question) | 26 |
| `--osl` | Output sequence length (answer tokens) | 100 |
| `--num-dataset-entries` | Required when using user-context-prompt-length | 1000+ |

The shared system prompt creates prefix sharing across all users, maximizing KV cache effectiveness.

## Concurrency Configuration

**Important**: User-centric mode does NOT automatically limit concurrency. Without `--concurrency`, session concurrency is **unlimited**.

| Option | Recommendation |
|--------|----------------|
| `--concurrency` | Set to `--num-users` or higher to allow all users to have in-flight requests simultaneously |

While the timing model naturally spaces requests by `turn_gap`, slow server responses can still cause request buildup. If you want to cap concurrent requests (e.g., to avoid overwhelming the server), set `--concurrency` explicitly.

```bash
# Explicitly cap concurrency to num_users
aiperf profile \
    --user-centric-rate 1.0 \
    --num-users 15 \
    --concurrency 15 \    # Ensures at most 15 concurrent requests
    ...
```

## Configuration Examples

```bash
# Complete KV cache benchmark (15s gaps)
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
    --benchmark-duration 100

# Higher throughput benchmark (3.75s gaps)
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

# Low QPS for cache TTL testing (30s gaps)
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

## When to Use User-Centric Rate Mode

User-centric rate mode is ideal for:

- **KV cache benchmarking**: The consistent turn gaps and collision-free scheduling provide controlled conditions for measuring cache hit rates and prefix caching performance
- **Multi-turn conversation simulation**: Realistic simulation of chat applications where users have think time between messages
- **Steady-state analysis**: Virtual history initialization ensures metrics represent steady-state behavior from the start

## Summary

| Feature | Description |
|---------|-------------|
| Collision-free scheduling | Pre-calculated stagger slots ensure no request overlaps |
| Precise timing | ~1ms request timing precision |
| Turn gap enforcement | Minimum time between user turns always maintained |
| Virtual history | Steady-state simulation from t=0 |
| Schedule adjustment | Handles slow responses while maintaining turn gaps |
