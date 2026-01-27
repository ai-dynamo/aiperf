<!--
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# Load Generator Options Reference

This guide provides a comprehensive reference for all load generator CLI options in AIPerf, including a compatibility matrix showing which options work together.

## Request Scheduling Options

AIPerf determines how to schedule requests based on which CLI options you specify:

| CLI Option | Use Case | Description |
|------------|----------|-------------|
| `--request-rate` | Rate-based load testing | Schedule requests at a target QPS with configurable arrival patterns |
| `--concurrency` (alone) | Saturation/throughput testing | Send requests as fast as possible within concurrency limits |
| `--fixed-schedule` | Trace replay | Replay requests at exact timestamps from dataset |
| `--user-centric-rate` | KV cache benchmarking | Per-user rate limiting with consistent turn gaps |

### Option Priority

When multiple options are specified, AIPerf uses this priority:

1. `--fixed-schedule` or mooncake_trace dataset → Timestamp-based scheduling
2. `--user-centric-rate` → Per-user turn gap scheduling
3. `--request-rate` → Rate-based scheduling with arrival patterns
4. `--concurrency` only → Burst mode (as fast as possible within limits)

---

## Compatibility Matrix

### Legend
- ✅ **Compatible** - Option works with this configuration
- ⚠️ **Conditional** - Works with restrictions (see notes)
- ❌ **Incompatible** - Option conflicts or is ignored
- 🔧 **Required** - Option is required for this configuration

### Scheduling Options

| Option | `--request-rate` | `--fixed-schedule` | `--user-centric-rate` | Notes |
|--------|:----------------:|:------------------:|:---------------------:|-------|
| `--request-rate` | ✅ | ❌ | ❌ | Conflicts with `--user-centric-rate` |
| `--user-centric-rate` | ❌ | ❌ | 🔧 | Requires `--num-users` |
| `--fixed-schedule` | ❌ | 🔧 | ❌ | Requires trace dataset with timestamps |
| `--num-users` | ❌ | ❌ | 🔧 | Required with `--user-centric-rate`; **raises error** otherwise |
| `--request-rate-ramp-duration` | ✅ | ❌ | ❌ | **Raises error** with `--fixed-schedule` or `--user-centric-rate` |

### Stop Conditions (at least one required)

| Option | `--request-rate` | `--fixed-schedule` | `--user-centric-rate` | Notes |
|--------|:----------------:|:------------------:|:---------------------:|-------|
| `--request-count` | ✅ | ✅ | ✅ | Mutually exclusive with `--num-sessions` |
| `--num-sessions` | ✅ | ✅ | ✅ | Mutually exclusive with `--request-count` |
| `--benchmark-duration` | ✅ | ✅ | ✅ | Enables `--benchmark-grace-period` |

### Arrival Pattern Options

| Option | `--request-rate` | `--fixed-schedule` | `--user-centric-rate` | Notes |
|--------|:----------------:|:------------------:|:---------------------:|-------|
| `--arrival-pattern` | ✅ | ❌ | ❌ | Conflicts with `--user-centric-rate`; values: `constant`, `poisson`, `gamma` |
| `--arrival-smoothness` | ⚠️ | ❌ | ❌ | Only with `--arrival-pattern gamma` |

**Arrival Pattern Values:**
- `constant` - Fixed inter-arrival times (1/rate)
- `poisson` - Exponential inter-arrivals (default with `--request-rate`)
- `gamma` - Tunable smoothness via `--arrival-smoothness`
- `concurrency_burst` - As fast as possible within concurrency limits (auto-set when no rate specified)

### Concurrency Options

| Option | `--request-rate` | `--fixed-schedule` | `--user-centric-rate` | Notes |
|--------|:----------------:|:------------------:|:---------------------:|-------|
| `--concurrency` | ✅ | ✅ | ✅ | Limits concurrent sessions with any scheduling option |
| `--prefill-concurrency` | ⚠️ | ⚠️ | ⚠️ | Requires `--streaming`; must be ≤ `--concurrency` |
| `--concurrency-ramp-duration` | ✅ | ✅ | ✅ | Works with any scheduling option |
| `--prefill-concurrency-ramp-duration` | ⚠️ | ⚠️ | ⚠️ | Requires `--streaming`; works with any scheduling option |

**Concurrency behavior by configuration:**
- **With `--request-rate`**: Concurrency acts as a ceiling; requests scheduled by rate are blocked if at limit
- **With `--concurrency` only** (no rate options): Concurrency is the primary driver; sends as fast as possible within limit
- **With `--fixed-schedule`**: Concurrency acts as a ceiling; requests fire at scheduled times but blocked if at limit
- **With `--user-centric-rate`**: Concurrency acts as a ceiling; user turns fire based on turn_gap but blocked if at limit

> **Important**: If `--concurrency` is not set, session concurrency limiting is **disabled** (unlimited). For `--user-centric-rate` mode, consider setting `--concurrency` to at least `--num-users` to ensure all users can have in-flight requests.

> **See also**: [Prefill Concurrency Tutorial](../tutorials/prefill-concurrency.md) for detailed guidance on memory-safe long-context benchmarking.

### Grace Period Options

| Option | `--request-rate` | `--fixed-schedule` | `--user-centric-rate` | Notes |
|--------|:----------------:|:------------------:|:---------------------:|-------|
| `--benchmark-grace-period` | ⚠️ | ⚠️ | ⚠️ | Requires `--benchmark-duration`; default: 30s (`--user-centric-rate` defaults to ∞ when duration-based) |

### Fixed Schedule Options

| Option | `--request-rate` | `--fixed-schedule` | `--user-centric-rate` | Notes |
|--------|:----------------:|:------------------:|:---------------------:|-------|
| `--fixed-schedule-auto-offset` | ❌ | ✅ | ❌ | **Raises error** without `--fixed-schedule`; conflicts with `--fixed-schedule-start-offset` |
| `--fixed-schedule-start-offset` | ❌ | ✅ | ❌ | **Raises error** without `--fixed-schedule`; conflicts with `--fixed-schedule-auto-offset` |
| `--fixed-schedule-end-offset` | ❌ | ✅ | ❌ | **Raises error** without `--fixed-schedule`; must be ≥ start offset |

### Request Cancellation Options

| Option | `--request-rate` | `--fixed-schedule` | `--user-centric-rate` | Notes |
|--------|:----------------:|:------------------:|:---------------------:|-------|
| `--request-cancellation-rate` | ✅ | ✅ | ✅ | Percentage (0-100) |
| `--request-cancellation-delay` | ⚠️ | ⚠️ | ⚠️ | Requires `--request-cancellation-rate`; **raises error** otherwise |

### Dataset Options

| Option | `--request-rate` | `--fixed-schedule` | `--user-centric-rate` | Notes |
|--------|:----------------:|:------------------:|:---------------------:|-------|
| `--dataset-sampling-strategy` | ✅ | ❌ | ✅ | Not compatible with `--fixed-schedule` |

### Session Configuration

| Option | `--request-rate` | `--fixed-schedule` | `--user-centric-rate` | Notes |
|--------|:----------------:|:------------------:|:---------------------:|-------|
| `--session-turns-mean` | ✅ | ✅ | ⚠️ | `--user-centric-rate` requires ≥ 2 |
| `--session-turns-stddev` | ✅ | ✅ | ✅ | |

---

## Warmup Options

Warmup options work **independently of the main benchmark configuration**. The warmup phase always uses rate-based scheduling internally.

| Option | All Configurations | Notes |
|--------|:------------------:|-------|
| `--warmup-request-count` | ✅ | Stop condition for warmup; mutually exclusive with `--num-warmup-sessions` |
| `--warmup-duration` | ✅ | Stop condition for warmup |
| `--num-warmup-sessions` | ✅ | Stop condition for warmup; mutually exclusive with `--warmup-request-count` |
| `--warmup-concurrency` | ✅ | Falls back to `--concurrency` |
| `--warmup-prefill-concurrency` | ⚠️ | Requires `--streaming` |
| `--warmup-request-rate` | ✅ | Falls back to `--request-rate` |
| `--warmup-arrival-pattern` | ✅ | Falls back to `--arrival-pattern` |
| `--warmup-grace-period` | ⚠️ | Requires warmup to be enabled; default: ∞ |
| `--warmup-concurrency-ramp-duration` | ✅ | Falls back to `--concurrency-ramp-duration` |
| `--warmup-prefill-concurrency-ramp-duration` | ⚠️ | Requires `--streaming` |
| `--warmup-request-rate-ramp-duration` | ✅ | Falls back to `--request-rate-ramp-duration` |

---

## Configuration Examples

### Using `--request-rate` (Rate-Based Scheduling)

Sends requests at a target average rate with configurable arrival patterns.

```bash
# Poisson arrivals at 10 QPS
aiperf profile --url localhost:8000 --model llama \
    --request-rate 10 \
    --arrival-pattern poisson \
    --request-count 100

# Constant arrivals with concurrency limit
aiperf profile --url localhost:8000 --model llama \
    --request-rate 20 \
    --arrival-pattern constant \
    --concurrency 5 \
    --benchmark-duration 60
```

### Using `--concurrency` Only (Burst Mode)

Sends requests as fast as possible within concurrency limits. Triggered when no rate option is specified.

```bash
# Maximum throughput within concurrency=10
aiperf profile --url localhost:8000 --model llama \
    --concurrency 10 \
    --request-count 100

# Prefill-limited throughput
aiperf profile --url localhost:8000 --model llama \
    --concurrency 20 \
    --prefill-concurrency 5 \
    --streaming \
    --benchmark-duration 60
```

### Using `--fixed-schedule` (Trace Replay)

Replays requests at exact timestamps from dataset metadata. Used for trace replay benchmarking.

```bash
# Replay mooncake trace
aiperf profile --url localhost:8000 --model llama \
    --input-file trace.jsonl \
    --custom-dataset-type mooncake_trace \
    --fixed-schedule

# With time window filtering
aiperf profile --url localhost:8000 --model llama \
    --input-file trace.jsonl \
    --custom-dataset-type mooncake_trace \
    --fixed-schedule \
    --fixed-schedule-start-offset 60000 \
    --fixed-schedule-end-offset 120000
```

### Using `--user-centric-rate` (KV Cache Benchmarking)

Per-user rate limiting for KV cache benchmarking. Each user has a consistent gap between their turns.

```bash
# 15 users at 1 QPS total (basic example)
aiperf profile --url localhost:8000 --model llama \
    --user-centric-rate 1.0 \
    --num-users 15 \
    --session-turns-mean 20 \
    --streaming \
    --benchmark-duration 300
```

**Key formula:** `turn_gap = num_users / user_centric_rate`

With `--num-users 15` and `--user-centric-rate 1.0`, each user has 15 seconds between their turns.

> **For complete KV cache benchmarking**, also configure shared system prompts and user context prompts. See the [User-Centric Timing Tutorial](../tutorials/user-centric-timing.md) for full configuration including `--shared-system-prompt-length`, `--user-context-prompt-length`, and other prompt options.

---

## Common Validation Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `--user-centric-rate cannot be used together with --request-rate or --arrival-pattern` | Conflicting options | Use only one scheduling option |
| `--user-centric-rate requires --num-users to be set` | Missing required option | Add `--num-users` |
| `--user-centric-rate requires multi-turn conversations (--session-turns-mean >= 2)` | Single-turn with `--user-centric-rate` | Use `--request-rate` for single-turn or increase `--session-turns-mean` |
| `--benchmark-grace-period can only be used with duration-based benchmarking` | Grace period without duration | Add `--benchmark-duration` |
| `--warmup-grace-period can only be used when warmup is enabled` | Warmup grace without warmup | Add `--warmup-request-count`, `--warmup-duration`, or `--num-warmup-sessions` |
| `--prefill-concurrency requires --streaming to be enabled` | Prefill without streaming | Add `--streaming` |
| `--arrival-smoothness can only be used with --arrival-pattern gamma` | Wrong arrival pattern | Change to `--arrival-pattern gamma` |
| `Dataset sampling strategy is not compatible with fixed schedule mode` | Sampling with `--fixed-schedule` | Remove `--dataset-sampling-strategy` |
| `Both a request-count and number of conversations are set` | Conflicting stop conditions | Use only one of `--request-count` or `--num-sessions` |
| `Both --warmup-request-count and --num-warmup-sessions are set` | Conflicting warmup stop conditions | Use only one of `--warmup-request-count` or `--num-warmup-sessions` |
| `--num-users can only be used with --user-centric-rate` | `--num-users` without `--user-centric-rate` | Add `--user-centric-rate` or remove `--num-users` |
| `--request-cancellation-delay can only be used with --request-cancellation-rate` | Delay without cancellation rate | Add `--request-cancellation-rate` or remove `--request-cancellation-delay` |
| `--fixed-schedule-* can only be used with --fixed-schedule` | Fixed schedule options without `--fixed-schedule` | Add `--fixed-schedule` or remove the offset options |
| `--request-rate-ramp-duration cannot be used with --user-centric-rate` | Rate ramping with `--user-centric-rate` | Remove `--request-rate-ramp-duration` |
| `--request-rate-ramp-duration cannot be used with --fixed-schedule` | Rate ramping with `--fixed-schedule` | Remove `--request-rate-ramp-duration` |

---

## Quick Reference: Which Options to Use

```
┌─────────────────────────────────────────────────────────────────┐
│                    Which options should I use?                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Replaying a trace with timestamps?                             │
│  └─► --fixed-schedule (with mooncake_trace dataset)             │
│                                                                  │
│  Multi-turn KV cache benchmarking?                              │
│  └─► --user-centric-rate + --num-users                          │
│                                                                  │
│  Controlled request rate testing?                               │
│  └─► --request-rate (+ optional --arrival-pattern)              │
│                                                                  │
│  Maximum throughput / saturation testing?                       │
│  └─► --concurrency only (no rate options)                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Full Options Reference

### Scheduling Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--request-rate` | float | None | Target QPS; enables rate-based scheduling |
| `--user-centric-rate` | float | None | Per-user QPS; enables turn-gap scheduling (requires `--num-users`) |
| `--fixed-schedule` | bool | false | Enable timestamp-based scheduling from dataset |
| `--num-users` | int | None | Concurrent users (required with `--user-centric-rate`) |
| `--arrival-pattern` | enum | poisson | Request arrival distribution: `constant`, `poisson`, `gamma` (only with `--request-rate`) |
| `--arrival-smoothness` | float | 1.0 | Gamma distribution shape (only with `--arrival-pattern gamma`) |
| `--request-rate-ramp-duration` | float | None | Seconds to ramp request rate from proportional minimum to target (only with `--request-rate`) |

### Concurrency Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--concurrency` | int | None | Max concurrent sessions; drives throughput when no rate option specified |
| `--prefill-concurrency` | int | None | Max requests in prefill stage (requires `--streaming`) |
| `--concurrency-ramp-duration` | float | None | Seconds to ramp concurrency from 1 to target |
| `--prefill-concurrency-ramp-duration` | float | None | Seconds to ramp prefill concurrency |

### Stop Conditions

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--benchmark-duration` | float | None | Max duration in seconds for benchmarking |
| `--benchmark-grace-period` | float | 30.0 | Grace period after duration ends (requires `--benchmark-duration`) |
| `--request-count` | int | Auto | Max requests to send |
| `--num-sessions` | int | None | Number of conversations to run |

### Request Cancellation

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--request-cancellation-rate` | float | None | Percentage of requests to cancel (0-100) |
| `--request-cancellation-delay` | float | 0.0 | Seconds to wait before cancelling (requires `--request-cancellation-rate`) |

### Warmup Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--warmup-request-count` | int | None | Max warmup requests; mutually exclusive with `--num-warmup-sessions` |
| `--warmup-duration` | float | None | Max warmup duration in seconds |
| `--num-warmup-sessions` | int | None | Number of warmup sessions; mutually exclusive with `--warmup-request-count` |
| `--warmup-concurrency` | int | `--concurrency` | Warmup max concurrent requests |
| `--warmup-prefill-concurrency` | int | `--prefill-concurrency` | Warmup prefill concurrency |
| `--warmup-request-rate` | float | `--request-rate` | Warmup request rate |
| `--warmup-arrival-pattern` | enum | `--arrival-pattern` | Warmup arrival pattern |
| `--warmup-grace-period` | float | ∞ | Seconds to wait for warmup responses |
| `--warmup-concurrency-ramp-duration` | float | `--concurrency-ramp-duration` | Warmup concurrency ramp |
| `--warmup-prefill-concurrency-ramp-duration` | float | `--prefill-concurrency-ramp-duration` | Warmup prefill ramp |
| `--warmup-request-rate-ramp-duration` | float | `--request-rate-ramp-duration` | Warmup rate ramp |

### Fixed Schedule Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--fixed-schedule-auto-offset` | bool | false | Auto-offset timestamps to start at 0 (requires `--fixed-schedule`) |
| `--fixed-schedule-start-offset` | int | None | Start offset in milliseconds (requires `--fixed-schedule`) |
| `--fixed-schedule-end-offset` | int | None | End offset in milliseconds (requires `--fixed-schedule`) |

### Session Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--session-turns-mean` | float | 1.0 | Mean turns per session (`--user-centric-rate` requires ≥ 2) |
| `--session-turns-stddev` | float | 0.0 | Standard deviation of turns |
| `--dataset-sampling-strategy` | enum | shuffle | Dataset sampling: `sequential`, `shuffle` (not with `--fixed-schedule`) |

### Multi-URL Load Balancing

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--url` | list | localhost:8000 | One or more endpoint URLs; multiple URLs enable load balancing |
| `--url-strategy` | enum | round_robin | Strategy for distributing requests across multiple URLs |

> **See also**: [Multi-URL Load Balancing Tutorial](../tutorials/multi-url-load-balancing.md) for detailed configuration and examples.
