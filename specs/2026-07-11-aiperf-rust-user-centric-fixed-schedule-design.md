<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AIPerf-Rust: user-centric + fixed-schedule timing strategies

**Date:** 2026-07-11
**Author:** Anthony Casagrande (Tech Lead) + Claude
**Status:** design (not built) — user-centric is *partly built* (the pure seeding math exists as `aiperf-timing::plan_user_centric`); fixed-schedule is unbuilt
**Grounding:** end-to-end line-by-line read of
`src/aiperf/timing/strategies/core.py`,
`src/aiperf/timing/strategies/user_centric_rate.py`,
`src/aiperf/timing/strategies/fixed_schedule.py`,
`src/aiperf/timing/conversation_source.py`,
`src/aiperf/timing/intervals.py`, and the already-ported
`crates/aiperf-timing/src/user_centric.rs`.
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

## 3. Reconcile with the already-built `aiperf-timing::plan_user_centric`

`crates/aiperf-timing/src/user_centric.rs` ports **exactly the §2.1 setup math** — the
pure, RNG-free, clock-free seeding — and nothing else (`user_centric.rs:4-13`).

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

**One deliberate divergence to flag (§7):** the fresh replacement user's `max_turns`.
Python sets it to the length of the *actually sampled* session (`generate(order=0)` →
`max_turns=None` → `len(sampled.metadata.turns)`, `user_centric_rate.py:189-190`),
whereas Rust hard-codes `avg_session_turns` (`user_centric.rs:171`). This is a
consequence of the plan being **sampler-free**: it emits schedule math, not bound
sessions. It matches only when the sampled session equals the dataset average. The
online runner must decide whether to (a) keep the avg (pure, deterministic seeding) or
(b) re-derive `max_turns` from the concrete sampled session at bind time to match
Python. Recommend (b) for behavioral parity; document if (a) is chosen.

**Missing (everything past setup — the async run loop):**

- `execute_phase`: scheduling initial users at `started + order*stagger`, seeding the
  spawn heap, and the perpetual `spawn_queue` pump (`user_centric_rate.py:353-387`).
- `handle_credit_return`: per-user `next_send_time = max(now, last+turn_gap)` pacing
  and final-turn user eviction (`user_centric_rate.py:389-418`).
- Adaptive target: `set_target_users` / `_should_spawn_user` /
  `_should_spawn_replacement` / `_defer_next_spawn` and the churn snapshot
  (`user_centric_rate.py:265-316`).
- Binding users to sampled `SampledSession`s via a `ConversationSource`
  (`user_centric_rate.py:183-193`).

So `plan_user_centric` is the deterministic *seed*; the run loop, per-user pacer, and
churn are the unbuilt design surface below.

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
(`fixed_schedule.py:136-140`). The `LoopScheduler` owns the timeline. (Design note:
this materializes the *entire* schedule at once — fine for filtered traces, but the
Rust design should bound it, §6.)

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

## 5. Mapping onto the crates — built vs designed

| Concern | Primitive / seam | Crate | Status |
|---|---|---|---|
| User-centric steady-state seeding math | `plan_user_centric` / `UserCentricPlan` / `InitialUser` | `aiperf-timing` (`user_centric.rs`) | **built** |
| Open-loop replacement spawn time | `next_replacement_spawn_ns` | `aiperf-timing` (`user_centric.rs:186`) | **built** |
| Inter-arrival (Poisson/Gamma/Const/Burst) + `set_rate` | `IntervalGenerator` | `aiperf-timing` | **built** (unused by these two: user-centric derives its own stagger/turn_gap; fixed-schedule has no rate) |
| Session + prefill caps (debt-drain) | `SlotPool` / `ConcurrencyManager` | `aiperf-timing` | **built** (user-centric uses only under `--concurrency`; fixed-schedule uses none) |
| Stop bounds | `StopChecker` / `RunState` | `aiperf-timing` | **built** (user-centric: the `issue_credit`→falsy stop path; fixed-schedule: injected-but-unused) |
| Absolute-at pacing / think-time sleeps | `Clock::sleep`, `now_ns` | `aiperf-clock` | **built** |
| Turn prompt = prior replies spliced | `SegmentStore` + `materialize` | `aiperf-graph` | **built** |
| Dispatch turn + record TTFT/ITL | `RequestSink` + observer | `loadgen-core`/`aiperf-transport-http` | **built** |
| **`schedule_at_perf_sec` / `schedule_later` / `execute_async` scheduler** | a `LoopScheduler` seam over `Clock` | new | **designed** |
| **User-centric run loop** (spawn heap + per-user pacer + churn) | `UserPool` + `UserCentricWorkload` | new / `aiperf-timing` | **designed** |
| **Adaptive user target** (ramp) | `set_target_users` on `UserPool` | new | **designed** |
| **Fixed-schedule source** (sorted absolute schedule + zero-offset) | `FixedScheduleSource` + `FixedScheduleWorkload` | new | **designed** |
| **`ConversationSource`** (sample template, mint corr-id, next-turn meta) | `ConversationSource` trait | new (shared w/ companion) | **designed** |

### 5.1 The new seams (every extension point a trait)

- **`Clock`-backed scheduler** — the Python `LoopScheduler` verbs
  `schedule_at_perf_sec(abs_ns, fut)` / `schedule_later(dur_ns, fut)` /
  `execute_async(fut)` (used at `fixed_schedule.py:137,159,164,169`;
  `user_centric_rate.py:358,415`). In the single-loop `!Send` model this is
  `spawn_local` of `async { clock.sleep(target - clock.now_ns()).await; fut.await }`
  — the priority heap is the `Clock`'s own event queue under `SimClock`; no separate
  `heapq` is needed except the user-centric spawn heap (which stays an explicit
  `BinaryHeap<Reverse<i64>>` on the loop, matching `user_centric_rate.py:348`).

- **`UserPool`** — owns `Vec<User { corr_id, session, next_send_ns, max_turns, order }>`
  (mirror of `user_centric_rate.py:91-119`) keyed by correlation id
  (`session_to_user`, `user_centric_rate.py:164`). Methods: seed-from-`UserCentricPlan`,
  `pace_next(corr) -> next_send_ns = max(now, last + turn_gap)`
  (`user_centric_rate.py:414`), `retire(corr)` on final turn
  (`user_centric_rate.py:401`), and the adaptive `set_target(v)` /
  `should_spawn` / `should_spawn_replacement` gates
  (`user_centric_rate.py:269-316`). `Rc<RefCell<..>>`, no `Arc`.

- **`UserCentricWorkload`** (the `Workload` impl) — drives §2.2 (seed the spawn heap
  from the plan, pump spawns, bind each spawn to a `ConversationSource::next` sample)
  and delegates turn-`k+1` pacing to `UserPool` on credit return.

- **`FixedScheduleSource`** — builds the sorted `Vec<ScheduleEntry { at_ns, TurnToSend }>`
  and resolves `schedule_zero_ms` from `auto_offset_timestamps` /
  `fixed_schedule_start_offset` (`fixed_schedule.py:82-117`); exposes
  `ts_to_ns(timestamp_ms) = started_ns + (ts - zero)*1e6` (`fixed_schedule.py:68-74`).
  `FixedScheduleWorkload` schedules all first turns (§4.3) and resolves next-turn
  timing (absolute / delay / immediate, §4.4) from `ConversationSource::next_turn_meta`.

- **`ConversationSource`** (shared with the companion spec) — `next(corr_id) ->
  SampledSession` (`conversation_source.py:112-121`) and `next_turn_meta(credit) ->
  TurnMeta` (`conversation_source.py:184-198`). `SampledSession::build_first_turn(
  max_turns)` yields the partial-conversation `TurnToSend`
  (`conversation_source.py:68-85`).

Neither strategy needs the graph executor for its *linear* turns — the per-user pacer
(user-centric) and the trace schedule (fixed) are the sequencers; `aiperf-graph` is
only for FORK/SPAWN DAG branching, exactly as in the companion spec.

---

## 6. Online / mock / offline parity

Both `Workload`s are pure schedule generators over `{Clock, RequestSink}`:

- **ONLINE-real / ONLINE-mock** — `RealClock`; issuer + pacers on one `LocalSet`,
  dispatch `spawn_local`'d; the two differ only by target URL. High-rate user-centric
  runs that exceed one core's HTTP CPU fan the data plane to worker threads (control
  stays single-loop), per the companion's two-plane framing.
- **OFFLINE** — `SimClock`; every `schedule_at_perf_sec` / `schedule_later` / the
  spawn-heap `sleep` / think-time `delay_ms` becomes a virtual-ns advance under
  `drive_sim`. Single-owner-of-time is mandatory and is exactly the single-loop shape.

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

## 7. Build order (increments)

1. **`ConversationSource` (synthetic)** — shared with the companion; yields fixed
   K-turn sessions + `next_turn_meta`. Unblocks both strategies without a dataset.
2. **`Clock`-backed scheduler seam** — `schedule_at_ns` / `schedule_later` /
   `execute_async` as `spawn_local` over `Clock::sleep`. Shared by all three siblings.
3. **`FixedScheduleWorkload`** — thinnest strategy: build sorted schedule + zero-offset
   (§4.1), schedule all first turns (§4.3), resolve next-turn timing (§4.4). No slots,
   no rate — validates the scheduler seam end-to-end online + offline first.
4. **`UserPool` + `UserCentricWorkload` (non-adaptive)** — seed from
   `plan_user_centric` (already built), schedule initial users at `order*stagger`,
   run the spawn-heap pump (§2.2), per-user `max()` pacing (§2.3). Bind the fresh
   replacement user's `max_turns` to the sampled session (the §3 divergence fix).
5. **Adaptive user target** — `set_target_users` + spawn/replacement gates +
   `_defer_next_spawn` + churn snapshot (§2.4). Enables ramping.
6. **`--concurrency` interaction** — the one place user-centric becomes closed-loop
   (`user_centric_rate.py:346-347`): gate spawns/turns on the session `SlotPool`.
7. **Dataset-backed `ConversationSource`** — real traces (fixed-schedule needs
   per-turn `timestamp_ms`/`delay_ms`; the loader pre-filters as today).

Increments 1–4 deliver both strategies online + offline for linear multi-turn; 5–7 add
ramping, closed-loop concurrency, and real datasets.

---

## 8. Risks / open questions

- **Fresh-user `max_turns` divergence (§3).** Rust `plan_user_centric` uses
  `avg_session_turns` (`user_centric.rs:171`); Python uses the sampled session length
  (`user_centric_rate.py:189-190`). Decide at bind time; recommend re-deriving from the
  concrete sample for parity. **The plan itself is correct — the binding is the open
  question.**
- **`max_turns` vs actual session length.** Seeded users pass `turns_to_send` as
  `num_turns` (`user_centric_rate.py:80`, `:253`) regardless of whether the sampled
  conversation actually has that many turns. If a sampled template is *shorter* than
  `turns_to_send`, downstream turn-`k+1` materialization can walk past the end. Python
  relies on the dataset average being representative; the Rust `ConversationSource`
  should clamp `max_turns` to `min(turns_to_send, len(session.turns))` or document the
  assumption. Verified subtle at `conversation_source.py:80` + `:184-198`
  (`get_next_turn_metadata` raises past the end).
- **Fixed-schedule schedules everything up front** (`fixed_schedule.py:136-140`) — for
  a large unfiltered trace this is O(N) tasks pinned at once. Bound with a windowed
  scheduler (only materialize the next W entries) if traces get large; Python leans on
  loader-side filtering (`fixed_schedule.py:78-81`).
- **Fixed-schedule ignores `stop_checker`** (injected `fixed_schedule.py:53`, never
  used). Confirm the Rust design intentionally omits stop bounds for pure replay, or
  wire duration/cancel lifecycle stops (the trace defines the run length).
- **Past-timestamp fires.** Both `ts_to_perf` (`fixed_schedule.py:68-74`) and the
  user-centric `max()` (`user_centric_rate.py:414`) can produce a target ≤ now; the
  `Clock`-scheduler must treat a non-positive sleep as "fire immediately," never panic
  on a negative duration.
- **Adaptive `turn_gap` recompute mid-run.** `set_target_users` mutates `turn_gap`
  (`user_centric_rate.py:276`) while users already hold `next_send_time` computed under
  the *old* gap; the next `handle_credit_return` uses the new gap — an intentional
  step-change, not a smooth ramp. Preserve that (do not retro-fit outstanding users).
- **Open-loop vs `--concurrency`.** Only under `--concurrency` does user-centric become
  closed-loop (`user_centric_rate.py:346-347`); the default is strictly open-loop. The
  `SlotPool` wiring must be opt-in, not always-on, or the strategy's steady-state math
  is distorted.

---

## 9. One-line summary

User-centric-rate is a **per-user open-loop pacer** — `stagger=1/rate`,
`turn_gap=num_users/rate`, a **virtual-history steady-state seed** (already ported as
`aiperf-timing::plan_user_centric`) plus an absolute **spawn-heap churn** loop and a
`max(now, last+turn_gap)` catch-up per user — while fixed-schedule is **absolute-
timestamp trace replay** (sort by `timestamp_ms`, anchor to a schedule-zero, fire all
first turns up front, then absolute/`delay_ms`/immediate subsequent turns); both are
`Workload` schedule generators over `{Clock, RequestSink}` that run identically online,
mock, and (deterministically) offline, with only the seeding math built today and the
run loops, `UserPool`, `FixedScheduleSource`, and `ConversationSource` seams still to
build.

---

## Addendum — 2026-07-11: implemented end to end

This addendum supersedes the original designed/partly-built status, the build-order
future tense, and the implementation questions in §8. The scheduled workload plane
is now built in `crates/aiperf` over the existing `{Clock, RequestSink}` seams:

- `multiturn.rs` provides the object-safe `ConversationSource` seam, synthetic and
  JSON/JSONL dataset sources, prefix-dependent segment-backed prompt materialization,
  response splicing, correlation identity, and per-turn timing metadata.
- `scheduler.rs` provides the object-safe `LocalTaskScheduler` and the
  Clock-injected `ClockTaskScheduler`. Absolute, relative, and immediate work shares
  one `LocalSet`; pending timers can be cancelled without cancelling dispatched work.
- `scheduled.rs` provides the shared `ScheduledRuntime`, `Workload`, and
  `TurnDispatcher` seams. It owns issuance/dispatch observation, native metrics,
  per-turn scheduled/issued/dispatch/first-token/TTFT/terminal timestamps, aggregate
  early-issue and lateness analysis, stop notification, cancellation, and final drain.
- `fixed_schedule.rs` provides `FixedScheduleSource` and `FixedScheduleWorkload`.
  It validates finite non-negative offsets, filters and stable-sorts traces, supports
  auto or explicit schedule zero, schedules all first turns up front, and applies the
  required continuation precedence: absolute `timestamp_ms`, then `delay_ms` relative
  to the preceding terminal return, then immediate dispatch. Targets at or before
  `now` fire immediately. Pure replay intentionally ignores request/session/duration
  stop bounds, matching the cited Python behavior.
- `user_centric.rs` provides `UserPool`, `UserCentricWorkload`, and the live
  `UserTargetController`. Initial users bind the exact `plan_user_centric` virtual
  history to sampled sessions; a deterministic `(at_ns, seq_no)` spawn heap drives
  open-loop replacement churn; continuation targets use
  `max(now, previous_issue + turn_gap)`; optional concurrency gates whole sessions;
  request/session/duration stops prevent new sessions while already-started sessions
  drain. Adaptive target changes interrupt a pending spawn sleep and apply the new
  turn gap only to subsequent calculations.
- `http.rs`, `run.rs`, `main.rs`, and `report.rs` wire both workloads through the
  Clock-injected hyper transport. CLI entry points are `--user-centric-rate` with
  `--num-users` (synthetic or `--input-file`) and `--fixed-schedule --input-file`;
  `--timing-json` serializes the schedule evidence alongside the unified native-v2
  `--json` report.

The §8 choices are therefore resolved as follows: fresh users derive `max_turns`
from the concrete sampled session; all planned lengths are clamped to the available
turns; fixed replay keeps Python's all-up-front scheduling and no-stop semantics;
past targets are immediate; adaptive gap changes are step changes; and the session
cap remains opt-in. A session-count stop blocks replacement sessions but never drops
continuations from sessions already admitted.

Validation covers both time domains available today. `tests/scheduled_sim.rs` uses
`SimClock` with a Clock-injected fake dispatcher to assert exact nanosecond schedules,
stable ordering, timestamp/delay/immediate continuation behavior, response splicing,
steady-state seeding, churn, concurrency caps, stop-and-drain, duration cancellation,
and adaptive wake-up. `tests/scheduled_real_mock.rs` launches the real Rust
`aiperf-mock-rs` process and exercises both the library APIs and the compiled CLI,
asserting zero early issues, bounded wall-clock lateness, configured TTFT, per-user
non-overlap and gaps, terminal-relative delays, counts, and detailed JSON. CLI conflict
and required-input validation lives in `tests/scheduled_cli_validation.rs`.

The strategy code is Clock- and dispatcher-neutral and is exercised deterministically
under `SimClock`. Full OFFLINE-mock inference is now built behind
`aiperf/dynosim`: the same `ScheduledRuntime` dispatches user-centric and fixed
turns through the in-process Dynamo `TurnDispatcher`, and
`tests/dynosim_cli.rs` covers both policies without an HTTP server. Authored
ramps, request cancellation, and adaptive user-target control remain explicit
unsupported combinations in that optional composition.
