---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Load Generator Options Reference
---
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
| *(none — auto-selected)* | Agent graph replay | Replay a recorded conversation DAG; the strategy owns pacing and trace admission |
| `--scenario inferencex-agentx-mvp` | Agentic trajectory replay | Multi-turn trace replay resumed from a randomized per-trajectory start instant |

Two additional modes are **not** selected by a scheduling flag:

- **Agent Graph** (`agent_graph`) is selected automatically whenever the input resolves to an
  agent graph workload — an `--input-file` sniffed against the graph-adapter registry, or an
  explicit `--graph-format`. Detection flips the profiling phase's timing mode to `AGENT_GRAPH`
  regardless of the phase type.
- **Agentic replay** (`agentic_replay`) is selected by a benchmark scenario. `--scenario
  inferencex-agentx-mvp` stamps `timing_mode = AGENTIC_REPLAY` onto every profiling phase.

Both replay the recorded timeline of a trace instead of generating arrivals, so neither consults
`--request-rate` or `--arrival-pattern` — and both **reject** those flags rather than ignoring them
(see [Common Validation Errors](#common-validation-errors)).

> AgentX is a different thing from Agent Graph: it is the legacy branch orchestrator that
> `agentic_replay` is benchmarked against, and the name of the SemiAnalysis
> `inferencex-agentx-mvp` scenario. It is not the `agent_graph` timing mode.

### Option Priority

When multiple options are specified, AIPerf uses this priority:

1. Agent graph workload detected → `agent_graph` replay (wins over the phase type entirely)
2. `--scenario` timing-mode lock (e.g. `agentic_replay`) → the scenario's mode
3. `--fixed-schedule` → Timestamp-based scheduling
4. `--user-centric-rate` → Per-user turn gap scheduling
5. `--request-rate` → Rate-based scheduling with arrival patterns
6. `--concurrency` only → Burst mode (as fast as possible within limits)

Levels 1 and 2 do not silently override an explicit scheduling flag: a rate-controlled,
user-centric, or fixed-schedule phase raises instead.

**Trace auto-promotion.** Any `--custom-dataset-type` whose loader is registered as a trace dataset
(not just `mooncake_trace`) is auto-promoted to timestamp-based scheduling when its first record
carries a `timestamp` (or, for Parquet, a `timestamp_start_unix_ms` column). Two things suppress the
promotion:

- `--disable-auto-fixed-schedule` (`--no-fixed-schedule`) — keeps your rate/concurrency mode and
  ignores the trace timestamps.
- `--scenario <name>` — a scenario locks its own timing mode, so the auto-promotion is skipped
  silently (only an *explicit* `--fixed-schedule` conflicts with a scenario).

If a rate-shaped flag (`--request-rate`, `--user-centric-rate`, `--arrival-smoothness`,
`--num-users`) was set alongside a timestamped trace, the promotion is refused with an error rather
than silently dropping the flag.

---

## High-Resolution Rate Pacing

At high request rates (≥ 500 req/s), event-loop timers quantize sub-millisecond
sleeps to ~1ms intervals, causing AIPerf to silently under-deliver the target QPS.
High-resolution pacing bypasses the timer wheel with a platform-specific mechanism:

| Platform | Pacer | Precision |
|----------|-------|-----------|
| Linux | `TimerFdPacer` — `timerfd_create(CLOCK_MONOTONIC)` kernel hrtimer | ~50µs |
| macOS / Windows / other | `ThreadPacer` — dedicated sleep thread with `clock_nanosleep` / waitable timer | ~100µs (POSIX), ~500µs (Windows) |

High-res pacing is **enabled by default** when `--request-rate` is used. The
selection is automatic: timerfd on Linux, thread-based elsewhere.

### Pacing env vars

| Variable | Default | Description |
|----------|---------|-------------|
| `AIPERF_TIMING_HIGH_RES_TIMER` | `true` | Set to `false` to force event-loop timer pacing (useful for isolating scheduling behaviour). |
| `AIPERF_TIMING_MAX_CATCHUP_SECONDS` | `0.01` | Maximum schedule backlog (seconds) the rate loop may catch up on before re-anchoring to the current time. Without a catch-up window, every oversleep permanently forfeits one schedule slot — at 5,000 req/s the loop falls behind on each tick. Increase if you see a schedule-backlog warning; decrease toward `0` for strict no-burst behaviour (at the cost of some under-delivery). |

### Diagnosing under-delivery at high QPS

If measured QPS is consistently lower than `--request-rate`:

1. **Confirm the pacer is active.** Check the startup log for `"Rate loop pacing via timerfd"` or `"Rate loop pacing via dedicated sleep thread"`. If absent, `AIPERF_TIMING_HIGH_RES_TIMER` may be `false`.
2. **Check for schedule-backlog warnings.** If present, the pacer is waking on time but the event loop is stalling between wake and dispatch. Try `uvloop` (`pip install uvloop`) or reduce `--concurrency`.
3. **Widen the catch-up window.** At very high rates (≥ 5,000 req/s), increase `AIPERF_TIMING_MAX_CATCHUP_SECONDS` (e.g. `0.05`) to let the loop absorb brief stalls without re-anchoring too eagerly.

---

## Compatibility Matrix

### Legend
- ✅ **Compatible** - Option works with this configuration
- ⚠️ **Conditional** - Works with restrictions (see notes)
- ❌ **Incompatible** - Option conflicts or is ignored
- 🔧 **Required** - Option is required for this configuration

**Mode columns.** `agent_graph` is the auto-selected agent graph replay mode; `agentic_replay` is
the scenario-locked trajectory replay mode. Neither is chosen by a flag — see
[Request Scheduling Options](#request-scheduling-options).

### Scheduling Options

| Option | `--request-rate` | `--fixed-schedule` | `--user-centric-rate` | `agent_graph` | `agentic_replay` | Notes |
|--------|:----------------:|:------------------:|:---------------------:|:-------------:|:----------------:|-------|
| `--request-rate` | ✅ | ❌ | ⚠️ | ❌ | ❌ | Silently ignored when `--user-centric-rate` is also set (last-wins routing). **Raises error** under both replay modes |
| `--user-centric-rate` | ⚠️ | ❌ | 🔧 | ❌ | ❌ | Wins over `--request-rate` when both are passed; **raises error** under both replay modes |
| `--fixed-schedule` | ❌ | 🔧 | ❌ | ❌ | ❌ | Requires trace dataset with timestamps; **raises error** under both replay modes (they carry their own recorded timeline) |
| `--num-users` | ❌ | ❌ | 🔧 | ❌ | ❌ | Required with `--user-centric-rate`; **raises error** otherwise |
| `--request-rate-ramp-duration` | ✅ | ❌ | ✅ | ❌ | ❌ | Allowed with any rate-controlled phase (poisson/gamma/constant/user-centric); **raises error** with `--fixed-schedule`, plain `--concurrency`, or either replay mode |

### Stop Conditions (at least one required)

| Option | `--request-rate` | `--fixed-schedule` | `--user-centric-rate` | `agent_graph` | `agentic_replay` | Notes |
|--------|:----------------:|:------------------:|:---------------------:|:-------------:|:----------------:|-------|
| `--request-count` | ✅ | ✅ | ✅ | ✅ | ✅ | Combinable with `--num-sessions`; each set bound becomes its own cap. Under `agent_graph` it caps total node dispatches, not conversations |
| `--num-sessions` | ✅ | ✅ | ✅ | ⚠️ | ✅ | Combinable with `--request-count`; the run ends when the first cap is reached. Under `agent_graph` it caps the number of trace instances *started* and never truncates a fan-out trace mid-flight |
| `--benchmark-duration` | ✅ | ✅ | ✅ | ✅ | 🔧 | Enables `--benchmark-grace-period`. Under `agent_graph` it also cancels executors still parked on recorded idle gaps. The `inferencex-agentx-mvp` scenario requires ≥ 900s and auto-fills 1800s |

**No stop condition set** is legal only for `agent_graph`: a bare graph run makes a single pass over
the corpus (each trace once, no recycle) and terminates. Every other mode gets a `--request-count 10`
fallback injected.

### Arrival Pattern Options

| Option | `--request-rate` | `--fixed-schedule` | `--user-centric-rate` | `agent_graph` | `agentic_replay` | Notes |
|--------|:----------------:|:------------------:|:---------------------:|:-------------:|:----------------:|-------|
| `--arrival-pattern` | ✅ | ❌ | ⚠️ | ❌ | ❌ | Only consulted for `--request-rate` phases; silently ignored under `--user-centric-rate`. Values: `constant`, `poisson`, `gamma`, `concurrency_burst` |
| `--arrival-smoothness` | ⚠️ | ❌ | ❌ | ❌ | ❌ | Only with `--arrival-pattern gamma` |

**Arrival Pattern Values:**
- `constant` - Fixed inter-arrival times (1/rate)
- `poisson` - Exponential inter-arrivals (default with `--request-rate`)
- `gamma` - Tunable smoothness via `--arrival-smoothness`
- `concurrency_burst` - As fast as possible within concurrency limits (auto-set when no rate specified)

### Concurrency Options

| Option | `--request-rate` | `--fixed-schedule` | `--user-centric-rate` | `agent_graph` | `agentic_replay` | Notes |
|--------|:----------------:|:------------------:|:---------------------:|:-------------:|:----------------:|-------|
| `--concurrency` | ✅ | ✅ | ✅ | 🔧 | 🔧 | Limits concurrent sessions with any scheduling option. Both replay modes instead read it as a **lane count** (see below); under `agent_graph` open-loop replay it bounds concurrently running traces instead — see [Open-loop replay under `agent_graph`](#open-loop-replay-under-agent_graph) |
| `--prefill-concurrency` | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | Requires `--streaming`; must be ≤ `--concurrency`. Still enforced per request under `agent_graph` even though graph credits bypass session slots |
| `--concurrency-ramp-duration` | ✅ | ✅ | ✅ | ✅ | ✅ | Works with any scheduling option; under `agent_graph` it gates lane admission (1 → lane count) |
| `--prefill-concurrency-ramp-duration` | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | Requires `--streaming`; works with any scheduling option |

**Concurrency behavior by configuration:**
- **With `--request-rate`**: Concurrency acts as a ceiling; requests scheduled by rate are blocked if at limit
- **With `--concurrency` only** (no rate options): Concurrency is the primary driver; sends as fast as possible within limit
- **With `--fixed-schedule`**: Concurrency acts as a ceiling; requests fire at scheduled times but blocked if at limit
- **With `--user-centric-rate`**: Concurrency acts as a ceiling; user turns fire based on turn_gap but blocked if at limit
- **With `agent_graph`**: Concurrency is the number of **replay lanes**, not a session-slot ceiling. Each lane runs one trace instance at a time and recycles onto the next template while a stop condition still admits new sessions, so concurrency is *sustained* even when it exceeds the corpus size. Graph credits bypass the linear session-slot lifecycle entirely; the lane pool is the only cross-trace bound. Defaults to 1 when unset
- **With `agentic_replay`**: Concurrency sizes the **trajectory list** — one trajectory lane per unit of concurrency, sampled once at startup and held for the whole run

> **Over-subscription is a hard error in both replay modes.** If concurrency exceeds the number of
> distinct loaded traces and dataset wrapping is not enabled, the run fails at configure time.
> Reduce `--concurrency`, or pass `--allow-dataset-wrap` (and a non-`none` `--cache-bust`, so the
> cloned lanes do not collide on identical prefixes in the server KV cache).

> **Important**: If `--concurrency` is not set, session concurrency limiting is **disabled** (unlimited). For `--user-centric-rate` mode, consider setting `--concurrency` to at least `--num-users` to ensure all users can have in-flight requests.

> **See also**: [Prefill Concurrency Tutorial](../tutorials/prefill-concurrency.md) for detailed guidance on memory-safe long-context benchmarking.

### Grace Period Options

| Option | `--request-rate` | `--fixed-schedule` | `--user-centric-rate` | `agent_graph` | `agentic_replay` | Notes |
|--------|:----------------:|:------------------:|:---------------------:|:-------------:|:----------------:|-------|
| `--benchmark-grace-period` | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | Requires `--benchmark-duration`; default: 30s for every scheduling mode. Both replay modes synthesize their own warmup phase with a separate barrier grace (see [Warmup Options](#warmup-options)) |

### Fixed Schedule Options

| Option | `--request-rate` | `--fixed-schedule` | `--user-centric-rate` | `agent_graph` | `agentic_replay` | Notes |
|--------|:----------------:|:------------------:|:---------------------:|:-------------:|:----------------:|-------|
| `--fixed-schedule-auto-offset` | ❌ | ✅ | ❌ | ❌ | ❌ | **Raises error** without `--fixed-schedule`; conflicts with `--fixed-schedule-start-offset` |
| `--fixed-schedule-start-offset` | ❌ | ✅ | ❌ | ❌ | ❌ | **Raises error** without `--fixed-schedule`; conflicts with `--fixed-schedule-auto-offset` |
| `--fixed-schedule-end-offset` | ❌ | ✅ | ❌ | ❌ | ❌ | **Raises error** without `--fixed-schedule`; must be ≥ start offset |

### Request Cancellation Options

| Option | `--request-rate` | `--fixed-schedule` | `--user-centric-rate` | `agent_graph` | `agentic_replay` | Notes |
|--------|:----------------:|:------------------:|:---------------------:|:-------------:|:----------------:|-------|
| `--request-cancellation-rate` | ✅ | ✅ | ✅ | ✅ | ✅ | Percentage (0-100); carried on the phase config for every timing mode |
| `--request-cancellation-delay` | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | Requires `--request-cancellation-rate`; **raises error** otherwise |

### Dataset Options

| Option | `--request-rate` | `--fixed-schedule` | `--user-centric-rate` | `agent_graph` | `agentic_replay` | Notes |
|--------|:----------------:|:------------------:|:---------------------:|:-------------:|:----------------:|-------|
| `--dataset-sampling-strategy` | ✅ | ❌ | ✅ | ✅ | ✅ | Not compatible with `--fixed-schedule`. Under `agent_graph` it remaps the per-lane trace draw (`shuffle`/`random` permute the draw; `sequential` keeps the byte-identical cursor); under `agentic_replay` it drives both the initial trajectory sample and each recycle draw |
| `--allow-dataset-wrap` | ❌ | ❌ | ❌ | ✅ | ✅ | Opt-in to reusing the same trace across lanes when concurrency exceeds the distinct loaded corpus; without it, over-subscription is a configuration error |

### Session Configuration

| Option | `--request-rate` | `--fixed-schedule` | `--user-centric-rate` | `agent_graph` | `agentic_replay` | Notes |
|--------|:----------------:|:------------------:|:---------------------:|:-------------:|:----------------:|-------|
| `--session-turns-mean` | ✅ | ✅ | ⚠️ | ❌ | ❌ | `--user-centric-rate` requires ≥ 2. Both replay modes take their turn structure from the recorded trace, so this is REJECTED (not ignored): `_reject_file_dataset_incompatible` raises for any file or public dataset |
| `--session-turns-stddev` | ✅ | ✅ | ✅ | ❌ | ❌ | Rejected for the same reason |

### Replay Mode Options

These knobs are consumed only by the two replay modes.

| Option | `agent_graph` | `agentic_replay` | Notes |
|--------|:-------------:|:----------------:|-------|
| `--trajectory-start-min-ratio` / `--trajectory-start-max-ratio` | ✅ | ✅ | Bounds of the random per-trace start instant t*. `agentic_replay` defaults to 0.0/1.0 (the full trace), which `inferencex-agentx-mvp` also pins. `agent_graph` defaults to 0.0/0.0 — full replay, no snapshot rewrite — and only engages the t* chop (and its auto-warmup) when the window is named explicitly |
| `--burst-phase-starts` | ✅ | ✅ | Collapse the WARMUP-start and PROFILING-start dispatches into synchronized bursts instead of spreading them by each request's recorded offset from t*. Governs only the phase-start pattern; the rest of replay timing is faithful either way |
| `--agentic-cache-warmup-duration` | ❌ | ✅ | **Raises error** on any run that does not resolve to `agentic_replay` |
| `--agentic-warmup-grace-period` | ❌ | ✅ | Barrier grace for the synthesized agentic warmup; default is infinite (or a bounded drain under an accelerated cache-pressure warmup) |
| `--graph-format` | ✅ | ❌ | Forces graph mode, skipping input-file sniffing |
| `--replay-speedup`, `--open-loop-replay` / `--no-open-loop-replay`, `--open-loop-strict` | ✅ | ❌ | General trace-replay pacing knobs, also honored by graph trace replay |
| `--random-seed` | ✅ | ✅ | Makes per-trace t* sampling deterministic in both modes |
| `--cache-bust` | ✅ | ✅ | Required when lanes recycle or wrap, so cloned instances do not collide on identical prefixes. `inferencex-agentx-mvp` locks `first_turn_prefix` |

---

## Warmup Options

Warmup options work **independently of the main benchmark configuration**: any warmup flag left unset falls back to its profiling counterpart. The warmup phase is rate-controlled when a warmup rate resolves (from `--warmup-request-rate` or `--request-rate`) and concurrency-driven otherwise.

A warmup phase is emitted **only** when one of the three triggers (`--warmup-request-count`, `--num-warmup-sessions`, `--warmup-duration`) is explicitly set; other `--warmup-*` flags without a trigger are ignored.

**The two replay modes synthesize their own warmup instead.** `agentic_replay` replaces any declared
warmup phase with a trajectory warmup: one priming credit per lane, dispatched as a single
concurrency burst, with an infinite barrier grace so the phase holds until every primed trajectory
returns. Its grace comes from `--agentic-warmup-grace-period`, not `--warmup-grace-period`. Terminal
failures on a warmup credit abort the run rather than letting a degraded trajectory pool bias
steady-state metrics. `agent_graph` prepends an equivalent warmup — one priming credit per chain
live at t*, same burst-plus-infinite-grace shape — but only when the t* window is engaged
(`--trajectory-start-max-ratio > 0`); at the default `[0, 0]` full-replay window there is no
pre-t* prefix to prime and no warmup is added.

| Option | All Configurations | Notes |
|--------|:------------------:|-------|
| `--warmup-request-count` | ✅ | Warmup trigger + stop condition; combinable with the other two |
| `--warmup-duration` | ✅ | Warmup trigger + stop condition; combinable with the other two |
| `--num-warmup-sessions` | ✅ | Warmup trigger + stop condition; combinable with the other two |
| `--warmup-concurrency` | ✅ | Falls back to `--concurrency` |
| `--warmup-prefill-concurrency` | ⚠️ | Requires `--streaming` |
| `--warmup-request-rate` | ✅ | Falls back to `--request-rate` |
| `--warmup-arrival-pattern` | ✅ | Falls back to `--arrival-pattern` |
| `--warmup-grace-period` | ⚠️ | Requires `--warmup-duration` specifically (grace is a duration-phase tail); `--warmup-request-count` / `--num-warmup-sessions` do **not** satisfy it. Default when unset: ∞ |
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

### Using `agent_graph` (Agent Graph Replay)

Selected automatically when the input is an agent graph workload. There is no mode flag: point
`--input-file` at a recorded conversation-DAG corpus (or force it with `--graph-format`).

```bash
# Bare run: one pass over the corpus, each trace replayed once, then terminate
aiperf profile --url localhost:8000 --model llama \
    --input-file traces.jsonl

# 16 replay lanes recycling over the corpus for 10 minutes
aiperf profile --url localhost:8000 --model llama \
    --input-file traces.jsonl \
    --concurrency 16 \
    --allow-dataset-wrap \
    --cache-bust first_turn_prefix \
    --benchmark-duration 600
```

The strategy runs one dataflow executor per trace and owns both completion and trace admission, so
`--concurrency` is a lane count rather than a session-slot ceiling and `--num-sessions` bounds how
many trace instances start rather than truncating a fan-out trace mid-flight. A faithful replay
honors every recorded inter-turn idle gap, so a count-bounded run on a human-pace corpus spans the
slowest admitted trace's full recorded wall time with no console output during the parked gaps —
`--benchmark-duration` bounds that (it cancels the still-parked nodes and keeps the records
dispatched so far).

### Using `agentic_replay` (Trajectory Replay)

Selected by a scenario lock, not a flag. Each lane resumes a trace at a randomized start instant t*:
turns before t* are history, turns at or after t* are profiled.

```bash
# SemiAnalysis InferenceX AgentX-MVP scenario
aiperf profile --url localhost:8000 --model llama \
    --scenario inferencex-agentx-mvp \
    --input-file weka-traces.jsonl \
    --concurrency 64 \
    --streaming \
    --benchmark-duration 1800
```

The scenario pins the timing mode, requires `--streaming` and `ignore_eos`, requires a duration of at
least 900s (auto-filling 1800s when unset), widens the t* window to the full 0-100% of each trace,
and locks `--cache-bust first_turn_prefix`. When a lane's whole session tree drains, it recycles into
a fresh root drawn from the shared dataset sampler, so concurrency stays at the configured width for
the whole run.

> **For complete KV cache benchmarking**, also configure shared system prompts and user context prompts. See the [User-Centric Timing Tutorial](../tutorials/user-centric-timing.md) for full configuration including `--shared-system-prompt-length`, `--user-context-prompt-length`, and other prompt options.

---

## Common Validation Errors

Error text below is quoted from the CLI-to-phase converters for the profiling and warmup phases and
from the resolved-config validators on `AIPerfConfig`.

| Error (abridged) | Cause | Solution |
|-------|-------|----------|
| `--num-users requires --user-centric-rate.` | `--num-users` without `--user-centric-rate` | Add `--user-centric-rate`, or drop `--num-users` |
| `User-centric rate mode requires --session-turns-mean >= 2.` | Single-turn workload under `--user-centric-rate` | Raise `--session-turns-mean`, or use `--request-rate` |
| `--request-rate and --request-rate-series are mutually exclusive.` | Both rate sources set | Pass only one |
| `--request-rate-series is not supported with --user-centric-rate.` | Series with user-centric mode | Drop one of the two |
| `--request-rate-series can only be used with rate-controlled scheduling.` | Series on a concurrency/fixed-schedule phase | Add `--request-rate`, or drop the series |
| `--request-rate-ramp-duration can only be used with rate-controlled scheduling (--request-rate or --user-centric-rate).` | Rate ramp on a concurrency-only or `--fixed-schedule` phase | Add `--request-rate` or `--user-centric-rate`, or drop the ramp |
| `--arrival-smoothness is only supported with --arrival-pattern gamma.` | Smoothness on a non-gamma phase | Add `--arrival-pattern gamma`, or drop `--arrival-smoothness` |
| `--fixed-schedule-{auto,start,end}-offset requires --fixed-schedule.` | Offsets without fixed-schedule mode | Add `--fixed-schedule`, or drop the offsets |
| `Trace dataset has per-record timestamps and would be auto-promoted to fixed_schedule, but the following flags are incompatible ...` | Rate-shaped flag against a timestamped trace | Drop the conflicting flags, or pass `--no-fixed-schedule` |
| `Parameter sweeps (e.g., --concurrency 8,16) cannot be used with --fixed-schedule mode ...` | Magic-list sweep against fixed-schedule (including trace auto-promotion) | Use a single value, or pass `--no-fixed-schedule` |
| `--benchmark-grace-period requires --benchmark-duration to be set.` | Grace period without a duration bound | Add `--benchmark-duration`, or drop the grace period |
| `--request-cancellation-delay requires --request-cancellation-rate to be set` | Delay without a cancellation rate | Add `--request-cancellation-rate > 0`, or drop the delay |
| `--warmup-grace-period was supplied without any warmup trigger` | Grace period but no warmup phase at all | Add `--warmup-duration` (see next row), or drop the grace period |
| `--warmup-grace-period requires --warmup-duration; grace_period applies only to duration-bounded warmup phases.` | Warmup bounded only by `--warmup-request-count` / `--num-warmup-sessions` | Add `--warmup-duration`, or drop `--warmup-grace-period` |
| `--warmup-request-rate-ramp-duration requires warmup rate-controlled scheduling.` | Warmup rate ramp with no warmup rate | Add `--warmup-request-rate`, or drop the ramp |
| `Phase '<name>': prefill_concurrency requires endpoint.streaming=true` | `--prefill-concurrency` without streaming | Add `--streaming` |
| `phase '<name>' (type=...) is not supported for graph workloads: the recorded graph replay ... owns pacing` | `--request-rate` / `--user-centric-rate` / `--fixed-schedule` against an agent graph input | Drop the rate/schedule options (`--concurrency` bounds the replay lanes), or pin a non-graph loader with `--custom-dataset-type` |
| `adaptive_scale is not supported for graph workloads` | `--adaptive-scale` against an agent graph input | Remove `--adaptive-scale`, or pin a non-graph loader |
| `scenario '<name>' requires timing_mode=agentic_replay; do not pass --request-rate / --arrival-pattern / --user-centric-rate / --fixed-schedule ... alongside --scenario` | A rate-shaped or fixed-schedule phase under a scenario lock | Drop the scheduling flag, or drop `--scenario` |
| `concurrency N exceeds M distinct loaded traces ... Dataset wrapping is disabled` | Replay concurrency over-subscribes the corpus | Reduce `--concurrency` to ≤ the distinct trace count, or pass `--allow-dataset-wrap` plus a non-`none` `--cache-bust` |
| `--agentic-cache-warmup-duration requires the agentic_replay timing mode` | Flag set on a run that does not resolve to `agentic_replay` | Add `--scenario inferencex-agentx-mvp`, or drop the flag |
| `scenario '<name>' requires duration >= 900s to reach steady state and trigger KV offloading` | `--benchmark-duration` below the scenario floor | Raise `--benchmark-duration` to ≥ 900 |

> **Not an error:** `--request-rate` together with `--user-centric-rate`, or `--arrival-pattern`
> together with `--user-centric-rate`. No validator rejects these — `--user-centric-rate` is routed
> last and wins, and the ignored flag is silently dropped. Likewise `--request-count` +
> `--num-sessions`, and `--warmup-request-count` + `--num-warmup-sessions` + `--warmup-duration`,
> are all valid combinations: every set bound becomes an independent cap.

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
│  Replaying a recorded conversation DAG (agent graph)?           │
│  └─► --input-file <graph corpus>  (agent_graph auto-selected)   │
│                                                                  │
│  SemiAnalysis InferenceX agentic trajectory replay?             │
│  └─► --scenario inferencex-agentx-mvp  (agentic_replay)         │
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
| `--arrival-pattern` | enum | poisson | Request arrival distribution: `constant`, `poisson`, `gamma`, `concurrency_burst` (only consulted with `--request-rate`) |
| `--arrival-smoothness` | float | None | Gamma distribution shape (only with `--arrival-pattern gamma`). Setting it without an explicit `--arrival-pattern` auto-selects gamma |
| `--request-rate-ramp-duration` | float | None | Seconds to ramp request rate from proportional minimum to target (only with `--request-rate`) |

### Concurrency Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--concurrency` | int | None | Max concurrent sessions; drives throughput when no rate option specified. Under `agent_graph` open-loop replay it bounds concurrently running traces — see [Open-loop replay under `agent_graph`](#open-loop-replay-under-agent_graph) |
| `--prefill-concurrency` | int | None | Max requests in prefill stage (requires `--streaming`) |
| `--concurrency-ramp-duration` | float | None | Seconds to ramp concurrency from 1 to target |
| `--prefill-concurrency-ramp-duration` | float | None | Seconds to ramp prefill concurrency |

### Stop Conditions

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--benchmark-duration` | float | None | Max duration in seconds for benchmarking |
| `--benchmark-grace-period` | float | 30.0 | Grace period after duration ends (requires `--benchmark-duration`) |
| `--request-count` | int | None | Max requests to send. When no bound at all (`--request-count` / `--num-sessions` / `--benchmark-duration`) is given, a flat `10` is injected so the run terminates. Three inputs opt out: fixed-schedule runs default to the dataset record count, `dag_jsonl` runs default `--num-sessions` to the DAG root count, and an agent graph run stays unbounded so its lanes make one full pass over the corpus. Under `agent_graph` the cap counts node dispatches, not conversations |
| `--num-sessions` | int | None | Number of conversations to run. Under `agent_graph` it caps the trace instances started (and clamps the lane count) without truncating a fan-out trace mid-flight; on the `agent_graph` open-loop path it bounds the replayed corpus to the first N traces in corpus order (an explicit `--dataset-sampling-strategy shuffle` opts into a seeded shuffled draw) — see [Open-loop replay under `agent_graph`](#open-loop-replay-under-agent_graph). Under `agentic_replay` it gates lane recycling |

### Request Cancellation

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--request-cancellation-rate` | float | None | Percentage of requests to cancel (0-100) |
| `--request-cancellation-delay` | float | 0.0 | Seconds to wait before cancelling (requires `--request-cancellation-rate`) |

### Warmup Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--warmup-request-count` | int | None | Max warmup requests; combinable with the other two triggers |
| `--warmup-duration` | float | None | Max warmup duration in seconds; combinable with the other two triggers |
| `--num-warmup-sessions` | int | None | Number of warmup sessions; combinable with the other two triggers |
| `--warmup-concurrency` | int | `--concurrency` | Warmup max concurrent requests |
| `--warmup-prefill-concurrency` | int | `--prefill-concurrency` | Warmup prefill concurrency |
| `--warmup-request-rate` | float | `--request-rate` | Warmup request rate |
| `--warmup-arrival-pattern` | enum | `--arrival-pattern` | Warmup arrival pattern |
| `--warmup-grace-period` | float | ∞ | Seconds to wait for warmup responses; requires `--warmup-duration` |
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
| `--session-turns-mean` | int | 1 | Mean turns per session (`--user-centric-rate` requires ≥ 2). Accepts a comma-separated list to sweep |
| `--session-turns-stddev` | int | 0 | Standard deviation of turns |
| `--dataset-sampling-strategy` | enum | dataset-type-dependent | Dataset sampling: `sequential`, `random`, `shuffle`. Default depends on dataset type (`sequential` for traces, `shuffle` for synthetic). Not compatible with `--fixed-schedule` |

### Replay Mode Options

Consumed by `agent_graph` and `agentic_replay` only.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--graph-format` | str | None | Force agent graph mode with an explicit adapter, skipping input-file sniffing |
| `--trajectory-start-min-ratio` | float | 0.0 | Lower bound of the random per-trace start instant t*. Unset means the full trace on `agentic_replay`; `agent_graph` treats an unnamed window as `0.0` (full replay) |
| `--trajectory-start-max-ratio` | float | 1.0 | Upper bound of t*. Unset means the full trace on `agentic_replay` (the pair `inferencex-agentx-mvp` also pins); `agent_graph` engages the t* chop (and its auto-warmup) only when the window is named explicitly |
| `--burst-phase-starts` | bool | false | Collapse the warmup-start and profiling-start dispatches into synchronized bursts instead of spreading them by each request's recorded offset from t* |
| `--allow-dataset-wrap` | bool | false | Allow lanes to reuse the same trace when concurrency exceeds the distinct loaded corpus. Without it, over-subscription raises |
| `--agentic-cache-warmup-duration` | float | None | `agentic_replay` only: seconds of accelerated cache-pressure warmup after the baseline priming pass. **Raises error** on any other mode |
| `--agentic-warmup-grace-period` | float | ∞ | `agentic_replay` only: barrier grace for the synthesized warmup phase (bounded automatically under an accelerated cache-pressure warmup) |
| `--replay-speedup` | float | None | Trace replay wall-clock compression (10 = 10x faster than recorded); `None` = real time. A general trace-replay knob also honored by `agent_graph` |
| `--open-loop-replay` / `--no-open-loop-replay` | bool | true | Open-loop replay on the recorded timeline, or closed-loop think-time back-pressure. A general trace-replay knob also honored by `agent_graph`. Under `agent_graph`, open-loop replay also governs trace admission, corpus bounding and timestamp validation — see [Open-loop replay under `agent_graph`](#open-loop-replay-under-agent_graph) below |
| `--open-loop-strict` | bool | false | Fire every trace row at its absolute recorded timestamp as an independent single-turn session, trading away multi-turn grouping. Also honored by `agent_graph` |

#### Open-loop replay under `agent_graph`

Open-loop replay (`--open-loop-replay`, the default) starts each trace at its
recorded time rather than gating it behind the previous trace's think time.
Three options change meaning on that path.

**`--concurrency N` — trace admission.** An explicit `--concurrency` bounds how
many traces run concurrently: a trace whose recorded start has arrived waits for
a free slot, slipping only itself. The recorded schedule is never re-anchored,
so relative spacing between traces is preserved and only execution slips.
`--concurrency-ramp-duration` paces admission through the same gate. Without an
explicit `--concurrency` admission is not gated — the phase's inherited default
of 1 is not read as a ceiling.

**`--num-conversations N` (`--num-sessions`) — corpus bound.** Bounds the replay
to N traces. The N are the **first N in corpus order** by default: bounding a
recorded corpus is a *temporal* subsample — which slice of the captured traffic
to replay — and a shuffled draw would destroy the arrival process, turning
3 traces out of a 500-trace hour-long capture into three sparse arrivals
separated by large idle gaps. Corpus order keeps the selection contiguous and
deterministic (`--random-seed` does not affect it). Passing
`--dataset-sampling-strategy shuffle` (or `random`) opts into a seeded shuffled
draw instead, for a *content* subsample; explicit `sequential` matches the
default. N at or above the corpus size replays every trace exactly once, in
unchanged corpus order.

The bound is applied when traces are selected, so a fan-out DAG is never
truncated mid-trace, and the replay timeline is anchored on the earliest
SELECTED trace — a bound that excludes the earliest traces starts immediately
rather than idling until its first recorded timestamp, with all inter-trace
spacing preserved.

**Timestamp validation.** A PARTIALLY timestamped corpus is rejected: if some
replayed traces carry a recorded start and others do not, the untimestamped ones
cannot be paced and would all fire at t=0 on top of the faithful replay of the
rest, so `agent_graph` refuses to start and names the offending trace ids
(matching the fixed-schedule behavior of linear trace replay). A corpus with no
recorded starts anywhere — which no shipped producer emits — is
unaffected and still replays paced by its edge delays. Only the traces
actually selected are checked, so bounding with `--num-conversations` onto a
fully timestamped subset is accepted. Warmup is exempt (it never paces), and
`--no-open-loop-replay` needs no timestamps at all.

### Multi-URL Load Balancing

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--url` | list | `['http://localhost:8000']` | One or more endpoint URLs; multiple URLs enable load balancing |
| `--url-strategy` | enum | round_robin | Strategy for distributing requests across multiple URLs |

> **See also**: [Multi-URL Load Balancing Tutorial](../tutorials/multi-url-load-balancing.md) for detailed configuration and examples.
