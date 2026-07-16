<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AIPerf-Rust: user-centric + fixed-schedule timing strategies

**Date:** 2026-07-11
**Author:** Anthony Casagrande (Tech Lead) + Claude
**Status:** built (online + offline). Both scheduled strategies run over shared
Clock-backed `ScheduledRuntime` traits in `rust/runtime`; Python Config v2 authors both,
and the registered scheduled pair injects either the online or the feature-gated offline
(`dynosim`) backend without changing workload policy.
**Grounding:** end-to-end line-by-line read of
`src/aiperf/timing/strategies/core.py`,
`src/aiperf/timing/strategies/user_centric_rate.py`,
`src/aiperf/timing/strategies/fixed_schedule.py`,
`src/aiperf/timing/conversation_source.py`,
`src/aiperf/timing/intervals.py`, and the Rust realization in
`rust/runtime/src/{scheduled,scheduler,multiturn,fixed_schedule,user_centric}.rs`.
**Companion (read first, not re-derived here):**
`specs/2026-07-11-aiperf-rust-request-rate-multiturn-design.md` — establishes the
single-loop credit-issuer model, the session/prefill `SlotPool` contract,
`StopChecker` semantics, the two-plane (control/data) throughput framing, and house
style. This spec references those and covers only what is **different** for
user-centric and fixed-schedule.

---

## 0. What this is

`request_rate`, `user_centric_rate`, and `fixed_schedule` are three sibling
implementations of the **same** strategy contract (`core.py:22-82`): a fresh
instance per phase, dependencies injected via `__init__`
(`config, conversation_source, scheduler, stop_checker, credit_issuer, lifecycle` —
`core.py:36-45`), then `setup_phase()` (async init) → `execute_phase()` (send first
turns) → `handle_credit_return(credit)` (dispatch turn *k+1* on worker completion).
They share the credit issuer, the `LoopScheduler`, the `ConversationSource`, and the
two `SlotPool`s; they differ **only** in *when* first turns fire and *how* the next
turn is paced.

- **request-rate** (companion spec): paces **turns** — one credit per rate interval,
  continuation-priority, a single-loop pacer.
- **user-centric-rate** (this spec, §1–3): paces **users**. Each of `num_users`
  simulated users holds a session and fires its turns at a fixed per-user `turn_gap`.
  The novelty is a **virtual-history seeding** so a steady-state of mid-session users
  exists at `t=0`, plus **open-loop user churn** (new users spawn on an absolute
  schedule and replace finished ones). Distinct from request-rate: rate sets a
  *per-user cadence*, not a global turn cadence.
- **fixed-schedule** (this spec, §4): **absolute-timestamp trace replay**. No rate,
  no users, no slots — first turns are scheduled at the trace's own `timestamp_ms`,
  subsequent turns at `timestamp_ms`/`delay_ms` from the trace. It is the one sibling
  that is *not* `RateSettableProtocol` (`core.py:108-126`, explicitly noted
  `core.py:116-117`).

Both realize the unified-graph-runtime `Workload` seam; both run unchanged on
`{Clock, RequestSink}` (online-real / online-mock / offline-sim).

---

## 1. User-centric — what it IS and how it differs from request-rate

The strategy simulates a realistic multi-turn chat: at `t=0` there is *already* a
steady state of users at varying stages of their sessions; over time new users join
and old users leave (`user_centric_rate.py:4-11`). Each user keeps a fixed inter-turn
gap (`turn_gap`) — the KV-cache-pressure knob: too short keeps caches artificially
warm, too long evicts before reuse (`user_centric_rate.py:8-11`).

Two derived timing constants (**these formulas are the contract**):

- **`stagger = 1 / request_rate`** (`user_centric_rate.py:156-158`) — smallest gap
  between any two users' *first* turns. Rate is req/s; stagger is s/req.
- **`turn_gap = num_users / request_rate`** (`user_centric_rate.py:261-263`) — the
  per-user inter-turn gap. `num_users` users each firing once per `turn_gap` gives
  `qps = num_users / turn_gap = request_rate` (`user_centric_rate.py:262`).

Config gate: `num_users` and `request_rate` must both be set and positive, else
`ValueError` (`user_centric_rate.py:147-154`).

**Difference from request-rate in one line:** request-rate pops one turn from a global
continuation queue per rate interval; user-centric gives every user its *own* clock
(`User.next_send_time`, `user_centric_rate.py:108-109`) and paces that user's turns
independently — "their next turn is scheduled based on THEIR last send time, not a
global clock" (`user_centric_rate.py:100-102`).

### 1.1 Virtual history & partial conversations

Steady state is faked by giving each user a virtual "age" — how far through its
session it already is at `t=0` (`user_centric_rate.py:16-23`). User 1 (oldest) is
virtually *done* (all turns completed before `t=0`) and is replaced immediately; user
N (youngest) has the most turns remaining (`user_centric_rate.py:18-21`). A user's
remaining turns become its `max_turns`, and `build_first_turn(max_turns=...)`
(`conversation_source.py:68-85`) caps the session to a **partial** conversation
(`num_turns = max_turns or len(turns)`, `conversation_source.py:80`). This is the
mechanism by which some users finish soon after `t=0` and others just started.

### 1.2 User replacement / churn

Because user 1 is virtually done, a fresh user with **all** turns is always spawned at
`t=0` to replace it (`user_centric_rate.py:257-259`). Thereafter new users spawn on an
**absolute** open-loop schedule (`next_spawn = prev_spawn + max_turns * turn_gap`,
`user_centric_rate.py:51-55`, `:310`), unaffected by response times
(`user_centric_rate.py:55`, `:344-347`) — "open-loop… the replacement spawn user will
spawn at the specified time regardless of whether the previous spawn user completed on
time. The only exception is if `--concurrency` is set" (`user_centric_rate.py:344-347`).

---

## 2. The exact scheduling math

### 2.1 `setup_phase` — seed the steady state (`user_centric_rate.py:195-259`)

```
session_turns   = round(dataset_metadata.average_turn_count)         # :206-208
turn_gap        = num_users / request_rate                           # :261-263
session_lifetime = max(1, session_turns - 1)                         # :211-213 (gaps, floored)
use_alt_spacing = gcd(num_users, session_lifetime) > 1               # :218
spacing_step    = smallest step in 2..n coprime with n, else 1       # :79-88, :219-224

for i in 0..num_users:
    virtual_age  = (num_users - i) * session_lifetime                # :230
    session_age  = virtual_age // num_users                          # :232
    turns_to_send = session_lifetime - session_age                   # :233
    if turns_to_send <= 0:                                           # :235-239
        next_user_id += 1        # burn id (user virtually done), emit nothing
        continue
    slot_index    = (i*spacing_step)%num_users  if use_alt_spacing   # :244-247
                    else virtual_age % num_users
    starting_order = num_users - slot_index                          # :248
    user = generate(max_turns=turns_to_send, order=starting_order)   # :252-254
    initial_users.append(user)

initial_users.append(generate(order=0))   # fresh replacement, full turns   # :257-259
```

Why alternate spacing: when `num_users` and `session_lifetime` share a factor,
`virtual_age % num_users` produces **duplicate** slot positions
(`user_centric_rate.py:214-218`); a coprime step makes `(i*step)%n` visit every slot
exactly once (`user_centric_rate.py:79-88`).

Why burn the id (`next_user_id += 1` on `continue`, `:238`): user ids are assigned in
order even for virtually-done users, so downstream id ordering stays monotone
(`user_centric_rate.py:236-237`, `:250-251`).

### 2.2 `execute_phase` — schedule initial users + open-loop spawn loop (`user_centric_rate.py:318-387`)

Initial fire times are absolute from phase start (`user_centric_rate.py:353-364`):

```
for user in initial_users:
    user.next_send_time = started_at_perf_sec + user.order * stagger        # :355-357
    scheduler.schedule_at_perf_sec(user.next_send_time,
                                   issue_credit(user.build_first_turn()))    # :358-361
    heapq.push(spawn_queue, user.next_send_time + user.max_turns*turn_gap)   # :363-364
```

Then a perpetual spawn pump (`user_centric_rate.py:367-387`):

```
loop:
    if spawn_queue empty: await sleep(0.1); continue                        # :368-370
    spawn_sec = heappop(spawn_queue)                                        # :372
    await sleep(max(0, spawn_sec - now))                                    # :373  (absolute pace)
    if not should_spawn_user(): defer(now + stagger); continue             # :375-376 (adaptive)
    user = generate(spawn_sec)                                              # :379
    if not await issue_credit(user.build_first_turn()): return             # :381-383 (stop signal)
    schedule_replacement(spawn_queue, spawn_sec, user)                      # :387
        # -> push spawn_sec + user.max_turns*turn_gap, if should_spawn_replacement  :305-311
```

Note `issue_credit` here is **awaited** and its falsy return terminates the phase
(`user_centric_rate.py:381-383`) — this is the stop-signal path (session/request/
duration bound reached), distinct from request-rate's `try_issue`/`NoSlot` triad.

### 2.3 `handle_credit_return` — per-user turn_gap pacing with catch-up (`user_centric_rate.py:389-418`)

```
if credit.is_final_turn:
    session_to_user.pop(credit.x_correlation_id)   # user done; churn continues in execute_phase  :399-402
    return
user = session_to_user[credit.x_correlation_id]                            # :405-409
turn = TurnToSend.from_previous_credit(credit)                            # :410
user.next_send_time = max(now, user.next_send_time + turn_gap)            # :414  ← catch-up
scheduler.schedule_at_perf_sec(user.next_send_time, issue_credit(turn))  # :415-418
```

The `max(now, last + turn_gap)` (`user_centric_rate.py:414`) is the earned-in-blood
catch-up: ideal pacing when replies arrive on time; if a reply is late the schedule
**re-anchors to now** (fires immediately) rather than compounding drift — the same
"behind → re-anchor, no burst" rule request-rate applies to its absolute pacer, here
applied per-user.

### 2.4 Adaptive target (ramp down/up users) (`user_centric_rate.py:265-316`)

`set_target_users(v)` enables adaptive mode, recomputes `turn_gap` for the new user
count, and if scaling **up** pushes `(v-old)` extra spawns spaced by `stagger`
(`user_centric_rate.py:269-281`). Scale-down is *passive*: `_should_spawn_user` gates
new spawns on `active < target` (`user_centric_rate.py:297-300`) and
`_should_spawn_replacement` on `active <= target` (`user_centric_rate.py:313-316`), so
excess users drain by attrition (their replacements are suppressed). `_defer_next_spawn`
re-queues a blocked spawn at `now + stagger` (`user_centric_rate.py:302-303`).
`user_control_snapshot` reports `target/actual/active/retiring/cancelled`
(`user_centric_rate.py:283-292`).

---

## 3. The `plan_user_centric` seed and its binding

`rust/runtime/src/user_centric.rs` carries `plan_user_centric`, which realizes
**exactly the §2.1 setup math** — the pure, RNG-free, clock-free seeding
(`user_centric.rs:4-13`). It is the deterministic core; `UserPool` /
`UserCentricWorkload` / `UserTargetController` (same file) bind it to sampled sessions
and drive the run loop.

**Matches (verified line-for-line):**

| Behavior | Python | Rust |
|---|---|---|
| `stagger = 1/rate`, `turn_gap = num_users/rate` | `:158`, `:263` | `user_centric.rs:113-116` |
| `session_lifetime = max(1, turns-1)` | `:213` | `user_centric.rs:120` (`saturating_sub` guards avg 0) |
| gcd → alternate coprime step | `:218-224`, `:79-88` | `user_centric.rs:124-129`, `:45-66` |
| `virtual_age / session_age / turns_to_send` | `:230-233` | `user_centric.rs:137-140` |
| burn id on `turns_to_send<=0` | `:235-239` | `user_centric.rs:142-147` |
| `slot_index`, `starting_order = n - slot_index` | `:244-248` | `user_centric.rs:149-154` |
| fresh `order=0` replacement appended last | `:257-259` | `user_centric.rs:167-173` |
| open-loop replacement `prev + max_turns*turn_gap` | `:310` | `user_centric.rs:186-188` (`next_replacement_spawn_ns`) |
| positivity asserts | `:147-154` | `user_centric.rs:107-111` |

Rust improves on Python by using **integer nanoseconds** with half-to-even rounding
(`user_centric.rs:36-42`) instead of `float` perf-seconds — deterministic under
`SimClock`.

**Fresh replacement user's `max_turns`.** The pure plan is **sampler-free** — it emits
schedule math, not bound sessions — so `plan_user_centric` alone cannot know a sampled
session's actual length. `UserCentricWorkload` therefore re-derives `max_turns` from the
**concrete sampled session** at bind time, matching Python's
`generate(order=0)` → `len(sampled.metadata.turns)` (`user_centric_rate.py:189-190`)
rather than the dataset average, and clamps every planned length to the available turns
so turn-`k+1` materialization never walks past the end of a shorter template.

**The full run loop is built on top of the seed** (`user_centric.rs`):

- `execute_phase` equivalent: initial users are scheduled at `started + order*stagger`,
  the spawn heap is seeded, and a perpetual spawn pump drives open-loop replacement
  churn (`user_centric_rate.py:353-387`).
- `handle_credit_return` equivalent: per-user continuation pacing uses
  `max(now, previous_issue + turn_gap)` with final-turn user eviction
  (`user_centric_rate.py:389-418`).
- Adaptive target: `UserTargetController` mirrors `set_target_users` /
  `_should_spawn_user` / `_should_spawn_replacement` / `_defer_next_spawn` and the churn
  snapshot (`user_centric_rate.py:265-316`); a target change interrupts a pending spawn
  sleep and applies the new turn gap only to subsequent calculations.
- Binding users to sampled sessions goes through the `ConversationSource` seam in
  `multiturn.rs` (`user_centric_rate.py:183-193`).

---

## 4. Fixed-schedule — absolute-timestamp trace replay

The simplest sibling and the only non-rate one. It replays conversations at the exact
timestamps recorded in the dataset (`fixed_schedule.py:3-8`). No `num_users`, no
`turn_gap`, no `SlotPool`, no `IntervalGenerator`. `stop_checker` is injected
(`fixed_schedule.py:53`) but **never referenced** in the loop — replay is fully open.

### 4.1 `setup_phase` — build the absolute schedule (`fixed_schedule.py:76-123`)

```
for conv in dataset_metadata.conversations:                              # :85
    if not conv.turns: continue                                         # :86-87
    if conv.turns[0].timestamp_ms is None:                              # :90-93
        raise ValueError("First turn ... missing timestamp_ms")          #   ← required
    schedule.append(ScheduleEntry(conv.turns[0].timestamp_ms,
                    TurnToSend(conv_id, uuid4(), turn_index=0,
                               num_turns=len(conv.turns))))               # :95-105
if not schedule: raise ValueError("No conversations with valid ...")     # :107-108
schedule.sort(key=timestamp_ms)                                          # :110
schedule_zero_ms =                                                       # :112-117
    schedule[0].timestamp_ms          if auto_offset_timestamps
    else fixed_schedule_start_offset  if set
    else 0.0
```

The dataset is **already filtered by the loader** (e.g.
`mooncake_trace._timestamp_within_offsets`) — setup only validates and sorts
(`fixed_schedule.py:78-81`). Each first turn gets a fresh `uuid4()` correlation id
(`fixed_schedule.py:100`).

### 4.2 timestamp → clock conversion (`fixed_schedule.py:68-74`)

```
target_offset_sec = (timestamp_ms - schedule_zero_ms) / 1000
perf_sec          = started_at_perf_sec + target_offset_sec
```

`auto_offset_timestamps` slides the whole trace so its first event lands at phase
start; `fixed_schedule_start_offset` pins an explicit zero; else absolute-since-epoch
offsets (`fixed_schedule.py:112-117`).

### 4.3 `execute_phase` — schedule *all* first turns up front (`fixed_schedule.py:125-140`)

No pacing loop, no catch-up: every first turn is handed to
`scheduler.schedule_at_perf_sec(perf_sec, issue_credit(turn))` in one pass
(`fixed_schedule.py:136-140`). The scheduler owns the timeline. `FixedScheduleWorkload`
keeps this all-up-front scheduling (the loader pre-filters the trace, so the materialized
set is bounded in practice).

### 4.4 `handle_credit_return` — trace-timed subsequent turns (`fixed_schedule.py:142-171`)

```
if credit.is_final_turn: return                                         # :151-152
next_meta = conversation_source.get_next_turn_metadata(credit)          # :155  (turn k+1 meta)
turn      = TurnToSend.from_previous_credit(credit)                     # :156
if next_meta.timestamp_ms is not None:
    scheduler.schedule_at_perf_sec(ts_to_perf(next_meta.timestamp_ms), issue)  # :158-162 absolute
elif next_meta.delay_ms is not None:
    scheduler.schedule_later(next_meta.delay_ms/1000, issue)           # :163-167 relative think-time
else:
    scheduler.execute_async(issue)                                     # :168-171 immediate
```

`get_next_turn_metadata` returns `metadata.turns[turn_index+1]`, raising if the credit
was already the final turn (`conversation_source.py:184-198`). Three cases:
absolute-timestamp (re-anchored to schedule zero), relative `delay_ms` (think-time,
identical to request-rate's `schedule_later`), or immediate.

**Difference from user-centric:** fixed-schedule has *no* per-user state, no `turn_gap`
re-anchoring, no `max()` catch-up — a late reply simply means the next absolute
timestamp may already be in the past, and `ts_to_perf` can return a past `perf_sec`
(the scheduler fires it immediately). It is pure open-loop replay.

---

## 5. Mapping onto the modules

Sixteen former `aiperf-*` library crates are now modules of `aiperf`; these workloads
live in `aiperf_runtime::{scheduled, scheduler, multiturn, fixed_schedule, user_centric}`.

| Concern | Primitive / seam | Module | Status |
|---|---|---|---|
| User-centric steady-state seeding math | `plan_user_centric` / `UserCentricPlan` / `InitialUser` | `aiperf_runtime::user_centric` (`user_centric.rs`) | **built** |
| Open-loop replacement spawn time | `next_replacement_spawn_ns` | `aiperf_runtime::user_centric` (`user_centric.rs:186`) | **built** |
| Inter-arrival (Poisson/Gamma/Const/Burst) + `set_rate` | `IntervalGenerator` | `aiperf_runtime::timing` | **built** (unused by these two: user-centric derives its own stagger/turn_gap; fixed-schedule has no rate) |
| Session + prefill caps (debt-drain) | `SlotPool` / `ConcurrencyManager` | `aiperf_runtime::timing` | **built** (user-centric uses only under `--concurrency`; fixed-schedule uses none) |
| Stop bounds | `StopChecker` / `RunState` | `aiperf_runtime::timing` | **built** (user-centric: the issuance-return stop path; fixed-schedule: injected-but-unused) |
| Absolute-at pacing / think-time sleeps | `Clock::sleep`, `now_ns` | `aiperf_runtime::clock` | **built** |
| Turn prompt = prior replies spliced | `SegmentStore` + `materialize` | `aiperf_runtime::dataset` / `aiperf_runtime::graph` | **built** |
| Dispatch turn + record TTFT/ITL | `RequestSink` + observer | `loadgen-core` / `aiperf_runtime::transport_http` | **built** |
| `schedule_at` / `schedule_later` / `execute_async` scheduler | `LocalTaskScheduler` / `ClockTaskScheduler` over `Clock` | `aiperf_runtime::scheduler` | **built** |
| Shared scheduled runtime (issuance/dispatch, metrics, drain) | `ScheduledRuntime` / `Workload` / `TurnDispatcher` | `aiperf_runtime::scheduled` | **built** |
| User-centric run loop (spawn heap + per-user pacer + churn) | `UserPool` + `UserCentricWorkload` | `aiperf_runtime::user_centric` | **built** |
| Adaptive user target (ramp) | `UserTargetController` | `aiperf_runtime::user_centric` | **built** |
| Fixed-schedule source (sorted absolute schedule + zero-offset) | `FixedScheduleSource` + `FixedScheduleWorkload` | `aiperf_runtime::fixed_schedule` | **built** |
| `ConversationSource` (sample template, mint corr-id, next-turn meta) | `ConversationSource` trait | `aiperf_runtime::multiturn` | **built** |

### 5.1 The seams (every extension point a trait)

- **`Clock`-backed scheduler** (`scheduler.rs`) — the object-safe `LocalTaskScheduler`
  and Clock-injected `ClockTaskScheduler` realize the Python `LoopScheduler` verbs
  `schedule_at` / `schedule_later` / `execute_async` (used at
  `fixed_schedule.py:137,159,164,169`; `user_centric_rate.py:358,415`). In the
  single-loop `!Send` model each is a `spawn_local` of
  `async { clock.sleep(target - clock.now_ns()).await; fut.await }`; absolute, relative,
  and immediate work share one `LocalSet`, and pending timers can be cancelled without
  cancelling dispatched work. The priority heap is the `Clock`'s own event queue under
  `SimClock`; the only explicit heap is the user-centric spawn `BinaryHeap` with an
  `(at_ns, seq_no)` tie-break (matching `user_centric_rate.py:348`).

- **`ScheduledRuntime`** (`scheduled.rs`) — the shared runtime, `Workload`, and
  `TurnDispatcher` seams for request-rate, user-centric, and fixed-schedule. It owns
  issuance/dispatch observation, native metrics, per-turn
  scheduled/issued/dispatch/first-token/TTFT/terminal timestamps, aggregate early-issue
  and lateness analysis, stop notification, cancellation, and final drain.

- **`UserPool`** (`user_centric.rs`) — owns
  `Vec<User { corr_id, session, next_send_ns, max_turns, order }>`
  (mirror of `user_centric_rate.py:91-119`) keyed by correlation id
  (`session_to_user`, `user_centric_rate.py:164`). Seeds from `UserCentricPlan`, paces
  continuations at `max(now, previous_issue + turn_gap)` (`user_centric_rate.py:414`),
  retires a user on its final turn (`user_centric_rate.py:401`), and — via
  `UserTargetController` — applies the adaptive `set_target` / `should_spawn` /
  `should_spawn_replacement` gates (`user_centric_rate.py:269-316`).
  `Rc<RefCell<..>>`, no `Arc`.

- **`UserCentricWorkload`** (the `Workload` impl) — drives §2.2 (seed the spawn heap
  from the plan, pump spawns, bind each spawn to a `ConversationSource` sample) and
  delegates turn-`k+1` pacing to `UserPool` on credit return. Optional concurrency gates
  whole sessions; request/session/duration stops prevent new sessions while already
  started sessions drain.

- **`FixedScheduleSource`** (`fixed_schedule.rs`) — builds the sorted
  `Vec<ScheduleEntry { at_ns, TurnToSend }>` and resolves `schedule_zero_ms` from
  `auto_offset_timestamps` / `fixed_schedule_start_offset` (`fixed_schedule.py:82-117`),
  validating finite non-negative offsets; exposes
  `ts_to_ns(timestamp_ms) = started_ns + (ts - zero)*1e6` (`fixed_schedule.py:68-74`).
  `FixedScheduleWorkload` schedules all first turns (§4.3) and resolves next-turn timing
  with the required precedence — absolute `timestamp_ms`, then `delay_ms` relative to the
  preceding terminal return, then immediate (§4.4) — from `ConversationSource`.

- **`ConversationSource`** (`multiturn.rs`, shared with the companion spec) — the
  object-safe seam with synthetic and JSON/JSONL dataset sources: `next(corr_id) ->
  SampledSession` (`conversation_source.py:112-121`), `next_turn_meta(credit) ->
  TurnMeta` (`conversation_source.py:184-198`), prefix-dependent segment-backed prompt
  materialization, response splicing, and correlation identity.
  `SampledSession::build_first_turn(max_turns)` yields the partial-conversation
  `TurnToSend` (`conversation_source.py:68-85`).

Neither strategy needs the graph executor for its *linear* turns — the per-user pacer
(user-centric) and the trace schedule (fixed) are the sequencers; `aiperf_runtime::graph` is
only for FORK/SPAWN DAG branching, exactly as in the companion spec.

---

## 6. Online / mock / offline parity

Both `Workload`s are pure schedule generators over `{Clock, RequestSink}`:

- **ONLINE-real / ONLINE-mock** — `RealClock`; issuer + pacers on one `LocalSet`,
  dispatch `spawn_local`'d; the two differ only by target URL. High-rate user-centric
  runs that exceed one core's HTTP CPU fan the data plane to worker threads (control
  stays single-loop), per the companion's two-plane framing.
- **OFFLINE** — `SimClock`; every scheduled/later dispatch, the spawn-heap `sleep`, and
  the think-time `delay_ms` becomes a virtual-ns advance under `drive_sim`.
  Single-owner-of-time is mandatory and is exactly the single-loop shape. Full
  OFFLINE-mock inference is built behind the `dynosim` feature: the same
  `ScheduledRuntime` dispatches user-centric and fixed turns through the in-process
  Dynamo `TurnDispatcher` with no HTTP server. Authored ramps, request cancellation, and
  adaptive user-target control remain explicit unsupported combinations in that optional
  composition.

Fixed-schedule is the *cleanest* parity case: with `auto_offset_timestamps` the trace
is anchored to `started_ns` and, under `SimClock`, replays byte-identically regardless
of host speed — a real determinism win over Python's `perf_counter` schedule. Parity is
**code-path + report-schema, not byte-identical metric values** (per the port-exact
ledger addendum): simulated vs real timings differ by construction.

### 6.1 `SimClock` determinism specifics

- User-centric integer-ns seeding is already deterministic (`user_centric.rs:36-42`).
  The remaining nondeterminism source is the per-user catch-up `max(now, ...)`
  (`user_centric_rate.py:414`) — under `SimClock` `now` is exact virtual ns, so it is
  reproducible; under `RealClock` it re-anchors to wall time (non-reproducible by
  design, since it reacts to real reply latency).
- Fixed-schedule has no RNG and no rate draw — fully deterministic under `SimClock`
  once the (already-filtered, already-sorted) schedule is fixed.
- The spawn heap tie-break must be deterministic (`(at_ns, seq_no)`, matching the
  `SimClock` event-queue tie-break) so equal-timestamp spawns fire in a stable order.

---

## 7. How it is wired

Python Config v2 authors both strategies once; the runner's registered scheduled pair
(`aiperf-cli`) direct-loads the prepared operation and injects either the online
Clock-injected hyper transport or, under `dynosim`, the offline in-process Dynamo
`TurnDispatcher` — **the workload policy is identical either way**. The build layered
cleanly on the shared seams:

1. **`ConversationSource`** (`multiturn.rs`) — synthetic and JSON/JSONL dataset sources
   yielding sessions + `next_turn_meta`, prefix-dependent segment-backed prompt
   materialization, and response splicing.
2. **`Clock`-backed scheduler** (`scheduler.rs`) — `LocalTaskScheduler` /
   `ClockTaskScheduler` as `spawn_local` over `Clock::sleep`, shared by all three
   siblings, with cancellable pending timers.
3. **`ScheduledRuntime`** (`scheduled.rs`) — the shared runtime/`Workload`/`TurnDispatcher`
   seams that own issuance, dispatch observation, native metrics, and final drain.
4. **`FixedScheduleWorkload`** (`fixed_schedule.rs`) — sorted schedule + zero-offset
   (§4.1), all first turns up front (§4.3), timestamp/delay/immediate continuation
   precedence (§4.4).
5. **`UserPool` + `UserCentricWorkload`** (`user_centric.rs`) — seed from
   `plan_user_centric`, schedule initial users at `order*stagger`, run the spawn-heap
   pump (§2.2), per-user `max()` pacing (§2.3), with fresh-replacement `max_turns` bound
   to the sampled session.
6. **`UserTargetController`** — adaptive user target (spawn/replacement gates,
   defer-next-spawn, churn snapshot, §2.4) for ramping.
7. **`--concurrency` interaction** — the one place user-centric becomes closed-loop
   (`user_centric_rate.py:346-347`): session gating stays opt-in, off by default.

---

## 8. Design decisions locked in

These were the subtle risks; the built code resolves each:

- **Fresh-user `max_turns`.** `UserCentricWorkload` re-derives `max_turns` from the
  concrete sampled session (Python parity, `user_centric_rate.py:189-190`), not the
  dataset average — the pure `plan_user_centric` seed stays sampler-free and the binding
  supplies the real length.
- **`max_turns` vs actual session length.** All planned lengths are clamped to the
  available turns (`min(turns_to_send, len(session.turns))`), so turn-`k+1`
  materialization never walks past the end of a shorter template
  (`conversation_source.py:80` + `:184-198`, where `get_next_turn_metadata` raises past
  the end).
- **Fixed-schedule schedules everything up front** (`fixed_schedule.py:136-140`). The
  built workload keeps Python's all-up-front scheduling; the loader pre-filters the
  trace, so the materialized set is bounded in practice (`fixed_schedule.py:78-81`).
- **Fixed-schedule ignores `stop_checker`** (injected `fixed_schedule.py:53`, never
  used). Pure replay intentionally omits request/session/duration stop bounds; the trace
  defines the run length.
- **Past-timestamp fires.** Both `ts_to_ns` (`fixed_schedule.py:68-74`) and the
  user-centric `max()` (`user_centric_rate.py:414`) can produce a target ≤ now; the
  `Clock` scheduler treats a non-positive sleep as "fire immediately," never panicking on
  a negative duration.
- **Adaptive `turn_gap` recompute mid-run.** A target change interrupts a pending spawn
  sleep and applies the new gap only to subsequent calculations — an intentional
  step-change, not a smooth ramp; outstanding users are not retro-fit
  (`user_centric_rate.py:276`).
- **Open-loop vs `--concurrency`.** Only under `--concurrency` does user-centric become
  closed-loop (`user_centric_rate.py:346-347`); the default is strictly open-loop, and a
  session-count stop blocks replacement sessions but never drops continuations from
  sessions already admitted. Session gating is opt-in, not always-on, so the
  steady-state math is preserved.

### 8.1 Validation

`tests/scheduled_sim.rs` uses `SimClock` with a Clock-injected fake dispatcher to assert
exact nanosecond schedules, stable ordering, timestamp/delay/immediate continuation,
response splicing, steady-state seeding, churn, concurrency caps, stop-and-drain,
duration cancellation, and adaptive wake-up. `tests/scheduled_real_mock.rs` launches the
real `aiperf-mock-server` process and asserts zero early issues, bounded wall-clock lateness,
configured TTFT, per-user non-overlap and gaps, terminal-relative delays, counts, and
detailed JSON. The same `ScheduledRuntime` runs deterministically under `SimClock` and,
behind `dynosim`, over the in-process Dynamo `TurnDispatcher`.

---

## 9. One-line summary

User-centric-rate is a **per-user open-loop pacer** — `stagger=1/rate`,
`turn_gap=num_users/rate`, a **virtual-history steady-state seed** (`plan_user_centric`)
plus an absolute **spawn-heap churn** loop and a `max(now, last+turn_gap)` catch-up per
user — while fixed-schedule is **absolute-timestamp trace replay** (sort by
`timestamp_ms`, anchor to a schedule-zero, fire all first turns up front, then
absolute/`delay_ms`/immediate subsequent turns). Both are `Workload` schedule generators
over the shared `ScheduledRuntime` `{Clock, RequestSink}` seams, built and running
identically online, mock, and (deterministically) offline; Python Config v2 authors both,
and the registered scheduled pair injects either the online hyper transport or the
feature-gated offline (`dynosim`) Dynamo `TurnDispatcher` without changing the workload
policy.
