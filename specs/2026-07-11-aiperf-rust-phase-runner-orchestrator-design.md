<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AIPerf-Rust: timing phase runner, orchestrator, and manager

**Date:** 2026-07-11
**Author:** Anthony Casagrande (Tech Lead) + Claude
**Status:** built (`aiperf_runtime::timing::phase`, adapted for graph via `rust/runtime/src/engine/graph_phase_runtime.rs`) — see the implementation-status sections below for the per-item build state.
**Grounding:** end-to-end line-by-line read of the Python phase-driver stack —
`src/aiperf/timing/phase/runner.py` (786 lines), `src/aiperf/timing/phase_orchestrator.py`,
`src/aiperf/timing/phase/publisher.py`, `src/aiperf/timing/manager.py`,
`src/aiperf/timing/config.py`; plus the machinery they drive (already grounded in the
companion spec but re-read here): `timing/phase/lifecycle.py`, `timing/phase/stop_conditions.py`,
`timing/phase/progress_tracker.py`. **Companion (read, referenced, NOT duplicated):**
`specs/2026-07-11-aiperf-rust-request-rate-multiturn-design.md` — the single-loop credit
issuer + `ConversationSource` + continuation queue + two `SlotPool`s that this spec's
runner *drives*. This spec covers the layer ABOVE that: the per-phase execution driver, the
warmup→profiling sequencer, and the top-level manager. Crown-jewel seams already built:
`aiperf_runtime::clock::Clock`, `loadgen-core::{RequestSink<R>, RequestObserver, Dispatchable}`,
`aiperf_runtime::timing::{IntervalGenerator, SlotPool, StopChecker/StopConfig/RunState, ConcurrencyManager}`,
`aiperf_runtime::graph`.

---

## 0. What this is (and what collapses)

The Python phase stack is four nested layers:

```
TimingManager (BaseComponentService, ZMQ-command-driven)     manager.py
  └── PhaseOrchestrator (warmup → profiling sequencer)       phase_orchestrator.py
        └── PhaseRunner (per-phase execution driver)         phase/runner.py
              └── TimingStrategy (the credit issue loop)      strategies/*  ← companion spec
```

Two orthogonal truths govern the port:

1. **The phase *policy* is a benchmarking feature and MUST be kept faithfully** — the
   lifecycle state machine (CREATED→STARTED→SENDING_COMPLETE→COMPLETE), grace periods,
   duration timeouts, the cancel→drain→force-complete escalation, the warmup→profiling
   handoff, and seamless (gapless) transitions. These are the earned-in-blood parts.

2. **Most of the *coordination* is ZMQ/multiprocess scaffolding and MUST be deleted.**
   `PhasePublisher` is pure message-bus fan-out (`publisher.py` is 105 lines and does
   nothing but `pub_client.publish(...)`). `TimingManager` is a `BaseComponentService`
   whose whole job is to translate `PROFILE_CONFIGURE`/`PROFILE_START`/`PROFILE_CANCEL`
   ZMQ commands into method calls (`manager.py:117-186`). `credit_router.wait_for_workers`
   (`runner.py:370`) exists because workers are separate processes that register over ZMQ.
   In the single-process Rust rewrite these become **direct calls** (§4).

The residue after deletion is small and sharp: a `PhaseRunner` future, a
`PhaseOrchestrator` loop, a `PhaseObserver` trait for progress, and a `PhaseConfig`
struct — all `!Send`, all on one `current_thread` runtime + `LocalSet`, all time via `Clock`.

---

## 1. The phase model

### 1.1 What a phase IS

A **phase** = one bounded credit-issuance episode with its own stop bounds, concurrency
caps, arrival pattern, and lifecycle. Two kinds, both `CreditPhase` enum values
(`config.py:11`, used at `_build_warmup_config`/`_build_profiling_config`,
`config.py:347-447`):

- **WARMUP** — always `TimingMode.REQUEST_RATE` (`config.py:366-367`); triggers JIT /
  allocation / connection-pool warm-up so profiling isn't polluted (`config.py:351-353`).
  Key divergence: **grace defaults to `+inf`** when unset (`config.py:360-362`) — warmup
  *always* drains all in-flight requests before handing off, vs the field default of
  `None`=disabled.
- **PROFILING** — the measured phase; `timing_mode` chosen from the phase type
  (`_phase_timing_mode`, `config.py:47-55`: adaptive / fixed-schedule / user-centric /
  request-rate). Grace = the user's `phase.grace_period` (may be `None`=disabled)
  (`config.py:408`).

`TimingConfig.from_run` builds the **ordered list**: all warmup configs first, then all
profiling configs (`config.py:100-104`). Multiple of each are allowed (sweeps).

### 1.2 The lifecycle state machine

`PhaseLifecycle` (`lifecycle.py:35`) is an explicit, transition-validated state machine:

```
CREATED ──start()──► STARTED ──mark_sending_complete()──► SENDING_COMPLETE ──mark_complete()──► COMPLETE
```

Transitions raise on out-of-order calls (`lifecycle.py:84-121`). **Cancellation is a flag,
not a state** — `was_cancelled` is orthogonal and settable at any point
(`lifecycle.py:123-125`); a cancelled phase still walks its lifecycle to COMPLETE for
cleanup (`lifecycle.py:44-46`). Two transition-metadata flags ride along:
`timeout_triggered` (sending timed out) and `grace_period_triggered` (returns timed out)
(`lifecycle.py:63-64`, set at `:103` / `:120`).

**Timestamp duality (a scar to fold in Rust):** the lifecycle stores *wall-clock*
timestamps (`time.time_ns()`) for `started_at_ns` / `sending_complete_at_ns` /
`complete_at_ns` because they are *published across services and must be comparable*
(`lifecycle.py:47-48, 87-89, 102, 119`), but uses *process-local* `perf_counter_ns()` for
`time_left_in_seconds()` because that is the monotonic authority for duration math
(`lifecycle.py:59-60, 131-156`). In single-process Rust **both collapse onto one `Clock`**
(§7): `RealClock::now_ns` is the single monotonic authority; there is no cross-service
comparability requirement, so the wall/perf split disappears.

### 1.3 Grace, timeout, drain, force-completion

Four distinct time bounds, all funneled through `PhaseLifecycle.time_left_in_seconds`:

- **Duration timeout** (sending) — `time_left_in_seconds()` (no grace), used as the
  sending-wait timeout (`runner.py:597`). Returns `None` if no duration configured (wait
  forever), `0.0` if already elapsed (`lifecycle.py:145-156`).
- **Grace period** (returns) — `time_left_in_seconds(include_grace_period=True)` adds
  `grace_period_sec` (`lifecycle.py:153`), used as the returns-wait timeout
  (`runner.py:640`). This is the window in-flight requests get to finish after the
  duration deadline.
- **Cancel-drain timeout** — a *fixed* `Environment.TIMING.CANCEL_DRAIN_TIMEOUT`
  (`runner.py:669`), the bounded wait for *cancelled* credits to drain after the router
  is told to cancel everything.
- **Force-completion** — if the drain also times out, `_release_stuck_slots()` frees
  concurrency slots for credits that will never return, then `mark_complete` is forced
  (`runner.py:679-689, 701-710`). This is the "stuck slot" backstop that prevents an
  eternal hang.

---

## 2. The per-phase run loop (`runner.py`)

`PhaseRunner` is the **single owner** of every per-phase component (`runner.py:42-70,
126-136`): the `LoopScheduler` (think-time deferrals), `PhaseLifecycle`,
`PhaseProgressTracker` (wraps `CreditCounter` + the two events), `StopConditionChecker`,
and the `CreditIssuer`. The orchestrator injects the *shared, long-lived*
dependencies (conversation source, concurrency manager, cancellation policy, callback
handler, credit router, publisher) via the constructor (`runner.py:72-85`).

### 2.1 The happy-path ordering (`_run_strategy`, `runner.py:352-421`)

The sequence is deliberate; each step's ordering is load-bearing:

1. **Configure concurrency for the phase** — `configure_for_phase(phase, concurrency,
   prefill_concurrency)` layers this phase's caps onto the shared manager
   (`runner.py:359-363`).
2. **`strategy.setup_phase()`** (`runner.py:365`).
3. **`credit_router.wait_for_workers(timeout=SERVICE.START_TIMEOUT)`** (`runner.py:370`).
   **Earned-in-blood race** (`runner.py:368-369`): "on fast startup the first credit can
   otherwise be issued before any worker registers, which deadlocks the phase." This is a
   *multiprocess* artifact — see §4.
4. **`_create_rampers(strategy)`** (`runner.py:374`) — build (don't start) concurrency /
   prefill / rate rampers from the `*_ramp_duration_sec` config (`runner.py:457-534`).
5. **`lifecycle.start()`** → STARTED, stamps start time (`runner.py:376`).
6. **`publish_phase_start`** (`runner.py:379`).
7. **Spawn the progress-report loop** background task (`runner.py:381`).
8. **Start rampers BEFORE execution** (`runner.py:386-387`). **Earned-in-blood ordering**
   (`runner.py:383-385`): "Otherwise, credits could be issued at full concurrency before
   the ramper sets the initial (lower) limit."
9. **Pre-dispatch DAG `pre` branches** (`runner.py:392-393`) — no-op for non-DAG.
10. **Spawn `strategy.execute_phase()`** as the execution task (`runner.py:395`) — this is
    the companion spec's single-loop credit issuer.
11. **`_wait_for_sending_complete()`** (`runner.py:397`).
12. **Cancellation short-circuit** (`runner.py:399-404`): if cancelled during sending,
    force `mark_complete(grace_period_triggered=True)`, freeze completed counts, set the
    returned event, and return immediately — do NOT wait for returns.
13. **Returns handling** (`runner.py:408-415`): if **seamless AND not final**, spawn
    `_wait_for_returning_complete()` as a *background* task with an
    `_on_return_wait_complete` done-callback (§3.2); otherwise `await` it synchronously
    and cancel the progress loop.
14. **Cleanup**: stop rampers, `scheduler.cancel_all()` (`runner.py:417-419`).
15. Return the `CreditPhaseStats` snapshot.

### 2.2 `_wait_for_sending_complete` (`runner.py:588-619`)

Waits on `all_credits_sent_event` with `timeout = time_left_in_seconds()`
(`runner.py:597`), cancelling the execution task on timeout (`set_event_on_timeout=True`,
`runner.py:598-604`). The `finally` block is the earned-in-blood ordering
(`runner.py:609-619`), and the **order matters**:

```
mark_sending_complete(timeout_triggered)   # lifecycle → SENDING_COMPLETE
freeze_sent_counts()                        # final_requests_sent becomes authoritative
scheduler.cancel_all_pending()              # drop not-yet-issued scheduled turns
all_credits_sent_event.set()                # unblock anyone waiting
publish sending-complete stats
```

`freeze_sent_counts` MUST precede any return-completion check, because
`check_all_returned_or_cancelled` compares against the *frozen* `final_requests_sent`
(`progress_tracker.py:144-152, 165-171`).

### 2.3 `_wait_for_returning_complete` (`runner.py:621-699`) — the multi-stage drain

This is the most race-sensitive method in the file. Stages:

1. **Fast path** (`runner.py:632-638`): if `check_all_returned_or_cancelled()` **AND** no
   pending branch work, set the event and return. The branch-work conjunct is essential —
   without it a DAG phase closes while children are still in flight (§9).
2. **Grace wait** (`runner.py:640-647`): wait on `all_credits_returned_event` with
   `timeout = time_left_in_seconds(include_grace_period=True)`, `set_event_on_timeout=False`
   (we must NOT lie that returns completed).
3. **On grace timeout** (`runner.py:648-689`):
   a. `credit_router.cancel_all_credits()` (`runner.py:657`).
   b. Compute `need = final_requests_sent − completed − cancelled` (`runner.py:659-663`).
   c. **Drain wait**: `asyncio.wait_for(all_credits_returned_event, CANCEL_DRAIN_TIMEOUT)`
      (`runner.py:669-677`).
   d. **On drain timeout**: `_release_stuck_slots()`, force
      `mark_complete(grace_period_triggered=True)`, freeze, set the event
      (`runner.py:679-689`).
4. **`finally`** (`runner.py:690-699`): if not already complete, `mark_complete(
   grace_period_triggered=timed_out)`, freeze completed counts, publish progress +
   phase-complete (with a branch-stats snapshot).

### 2.4 The two events + freeze protocol

The whole runner synchronizes on exactly two `asyncio.Event`s owned by
`PhaseProgressTracker` (`progress_tracker.py:54-56`): `all_credits_sent_event` (set by the
issuer when `is_final_credit`, or by the runner on sending timeout) and
`all_credits_returned_event` (set by the callback handler on the final return, or by the
runner on force-complete). The **freeze** calls (`freeze_sent_counts`,
`freeze_completed_counts`) snapshot mutable counters into `final_*` fields so late arrivals
after COMPLETE don't corrupt the reported totals (`progress_tracker.py:144-159`;
`increment_returned` late-arrival note at `:126-127`). In Rust these two events map to two
`Rc<RefCell<...>>`-backed one-shot notifications (or `tokio::sync::Notify` on the local
set); the freeze is a plain struct copy under the single-loop `!Send` invariant (no lock).

### 2.5 Cancellation & failure paths

- **External cancel** (`cancel()`, `runner.py:254-266`): set `was_cancelled`, flag the
  lifecycle, cancel the execution/progress/return-wait tasks, stop rampers,
  `scheduler.cancel_all()`. The `_run_strategy` short-circuit (§2.1 step 12) then finalizes.
- **Failure** (`run()` `except`, `runner.py:298-303` → `_publish_phase_failure_lifecycle`,
  `runner.py:423-455`): flush start/sending/complete lifecycle transitions + publishes so
  *other services see the phase end and the benchmark doesn't hang forever*. In
  single-process Rust this "flush so peers don't hang" motivation largely evaporates
  (there are no peers), but the local finalization (freeze + emit a terminal
  phase-complete to the observer) stays so the report and console UI resolve.
- **`finally`** (`runner.py:304-305`): `_detach_orchestrator_and_cleanup` detaches the
  branch orchestrator from the shared callback handler so a later phase doesn't dispatch
  into a torn-down orchestrator (`runner.py:340-350`).

---

## 3. Multi-phase orchestration (`phase_orchestrator.py`)

### 3.1 What the orchestrator owns vs creates

**Long-lived, shared across phases** (`phase_orchestrator.py:127-149`): `ConversationSource`
(dataset + sampler), `ConcurrencyManager`, `RequestCancellationSimulator`,
`CreditCallbackHandler` (wired directly to the router's return + first-token callbacks),
and the optional multi-URL `url_sampler`. **Created per-phase**: a fresh `PhaseRunner`
(`phase_orchestrator.py:202-211`).

The sequencer `_execute_phases` (`phase_orchestrator.py:184-235`) walks the ordered phase
configs, computing `is_final_phase` and `is_seamless_non_final` per iteration, then
`await runner.run(is_final_phase=...)`.

### 3.2 The warmup→profiling handoff and seamless mode

Two handoff modes:

- **Non-seamless (default)**: `runner.run()` returns only after *all returns complete*
  (§2.1 step 13 synchronous branch). The runner is removed from `_active_runners`
  immediately (`phase_orchestrator.py:234-235`). Warmup fully drains (grace=`+inf`) before
  profiling starts.
- **Seamless (`seamless=True`, non-final)**: `runner.run()` returns *after sending
  complete* — the phase's return-wait runs as a background task (`runner.py:408-412`), and
  the runner stays in `_active_runners` until its `_on_return_wait_complete` callback fires
  the orchestrator's cleanup callback (`runner.py:268-278`, `phase_orchestrator.py:213-218,
  237-245`). **Multiple runners are active simultaneously** — the old phase waits for
  returns while the new phase already sends (`phase_orchestrator.py:192-197`). This is what
  keeps concurrency from collapsing to zero at a phase boundary.

### 3.3 Cross-phase concurrency debt-drain

The `ConcurrencyManager` is **shared and long-lived**, so slot state carries across phases.
`configure_for_phase` (`runner.py:359`) layers each phase's caps onto the persistent global
layer; the companion spec §1.1 documents the underlying `DynamicConcurrencyLimit` =
semaphore + **debt tracking** for graceful ramp-down. When profiling's caps differ from
warmup's, the debt mechanism drains the delta rather than hard-cancelling in-flight work.
This is the already-built `SlotPool` debt-drain in the `aiperf_runtime::timing` module; the orchestrator's job
is only to hold the single shared `ConcurrencyManager` across the runner instances.

### 3.4 Orchestrator lifecycle & teardown

`@on_start` runs all phases then, in `finally`, marks credits complete + publishes
credits-complete (`phase_orchestrator.py:170-182`). `cancel()` cancels all in-flight
credits first, then every active runner (`phase_orchestrator.py:247-258`). `@on_stop`
cancels leftover active runners — a leak fix, since only `cancel()` (Ctrl+C) cleaned them
up before (`phase_orchestrator.py:260-285`). Router callback registrations are *not*
explicitly unregistered because the router is a child lifecycle torn down alongside the
orchestrator (`phase_orchestrator.py:266-273`) — in Rust this is just `Drop` order.

---

## 4. IPC scaffolding to DELETE vs essential policy to KEEP

| Python thing | What it does | Verdict | Rust replacement |
|---|---|---|---|
| `PhasePublisher` (`publisher.py`, all 105 lines) | `pub_client.publish(CreditPhase*Message)` — start/progress/sending-complete/complete/credits-complete over ZMQ | **DELETE the transport; KEEP the event *content*** | A `PhaseObserver` trait with `on_phase_start/on_progress/on_sending_complete/on_phase_complete` taking a `PhaseStats` value; console/report impls call directly. No message bus, no `service_id`, no `pub_client`. |
| `CreditPhase*Message` types (`credit/messages`) | Wire envelopes for the above | **DELETE** | The typed `PhaseStats` / `BranchStats` structs passed by value to the observer. |
| `TimingManager` command handlers (`manager.py:117-271`): `PROFILE_CONFIGURE`, `PROFILE_START`, `PROFILE_CANCEL`, `DATASET_CONFIGURED_NOTIFICATION` | ZMQ command/notification routing into orchestrator method calls | **DELETE the command layer; KEEP the sequence** | Direct calls: `configure()` (build orchestrator once dataset is ready) → `start()` → `cancel()`. The dataset-ready wait (`manager.py:144-169`) becomes an `await` on the in-process dataset build, not an `asyncio.Event` fed by a notification. |
| `credit_router.wait_for_workers` (`runner.py:370`) | Blocks until N worker *processes* register over ZMQ (avoids first-credit-before-worker deadlock) | **DELETE** | Workers are `spawn_local` tasks on the same `LocalSet`; they exist before the issuer runs. No registration handshake, no `START_TIMEOUT`. Call this out — it is the single clearest multiprocess artifact. |
| `CreditCallbackHandler` registered via `set_return_callback` / `set_first_token_callback` on the router (`phase_orchestrator.py:146-149`) | Worker→router→handler return/TTFT callbacks over ZMQ | **KEEP the logic, DELETE the transport** | `RequestObserver` terminal + first-token hooks call directly into the progress tracker + `SlotPool` release. The companion spec's "prefill-release-on-TTFT" wiring (§3, §7 risks) is exactly this edge. |
| `credit_router.cancel_all_credits` (`runner.py:657`, `phase_orchestrator.py:256`) | Broadcast cancel to worker processes | **KEEP semantics, DELETE transport** | A cancellation token / direct `.abort()` of in-flight `spawn_local` dispatch tasks. |
| `mark_credits_complete` / `publish_credits_complete` (`phase_orchestrator.py:181-182`) | Signal the records pipeline over ZMQ | **DELETE / trivialize** | Direct "all done" flag the report finalizer reads. |
| `EventLoopMonitor`, `_publish_service_error_*`, `_on_phase_orchestrator_done` (`manager.py:81, 189-258`) | Multiprocess health monitoring + surfacing fire-and-forget task errors to a SystemController over ZMQ | **DELETE** | In-process the orchestrator future is `await`ed directly; errors propagate as `Result`/`anyhow` up the call stack. No `BaseServiceErrorMessage`. |
| `LoopScheduler` (single owner, `runner.py:127`) | Think-time deferred credit enqueue | **KEEP** | Already the companion spec's `Clock::sleep`-backed deferred-enqueue. Single owner = single loop. |
| `_progress_report_loop` at `CREDIT_PROGRESS_REPORT_INTERVAL` (`runner.py:765-786`) | Periodic progress publish | **KEEP, retarget** | A local `Clock::sleep` tick that pushes `PhaseStats` to the `PhaseObserver` (console UI), not the bus. |

**The essential policy that survives:** the lifecycle state machine, the two events + freeze
protocol, grace/timeout/drain/force-complete escalation, warmup→profiling sequencing,
seamless multi-runner overlap, cross-phase debt-drain, and stop-condition gating. Everything
else in these five files is IPC glue.

---

## 5. `CreditPhaseConfig` → the Rust config shape

`CreditPhaseConfig` (`config.py:123-242`) is the per-phase knob bag. It fans out onto the
already-built `aiperf_runtime::timing` primitives:

| Python field (`config.py`) | Rust target | Notes |
|---|---|---|
| `phase` (`:134`) | `PhaseConfig.kind: PhaseKind::{Warmup, Profiling}` | drives grace default (§1.1). |
| `timing_mode` (`:135`) | selects the `Workload` impl | request-rate / fixed-schedule / user-centric / adaptive (companion + unified-runtime specs). |
| `total_expected_requests` (`:140`) | `StopConfig.request_cap` | `RequestCountStopCondition` (`stop_conditions.py:139-149`). FIXED_SCHEDULE overrides from actual dataset size (`runner.py:108-118`). |
| `expected_num_sessions` (`:143`) | `StopConfig.session_cap` | `SessionCountStopCondition` (`stop_conditions.py:152-193`); root-only counting. |
| `expected_duration_sec` (`:146`) | `StopConfig.duration_sec` + lifecycle time-left | `DurationStopCondition` (`stop_conditions.py:196-207`). |
| `grace_period_sec` (`:187`) | `PhaseConfig.grace_period: Option<Duration>` | warmup default `+inf` (`config.py:360-362`); `None`=disabled. |
| `seamless` (`:151`) | `PhaseConfig.seamless: bool` | orchestration overlap (§3.2). |
| `concurrency` / `prefill_concurrency` (`:158,165`) | `SlotPool` limits via `ConcurrencyManager::configure_for_phase` | session + prefill caps (companion §1.1). |
| `request_rate` (`:172`) | `IntervalGenerator` rate | ramp target for the rate ramper. |
| `arrival_pattern` / `arrival_smoothness` (`:175,179`) | `IntervalGenerator` config | Poisson/Gamma/Const/Burst. |
| `num_users` (`:196`) | user-centric workload param | only for `USER_CENTRIC_RATE`. |
| `concurrency_ramp_duration_sec` / `prefill_..._ramp_..._sec` / `request_rate_ramp_..._sec` (`:202,208,214`) | `Ramper` configs | stepped (concurrency) vs continuous (rate) (`runner.py:457-534`). |
| `auto_offset_timestamps` / `fixed_schedule_{start,end}_offset` (`:220,224,229`) | fixed-schedule workload params | |
| `adaptive` (`:239`, folded fields `:244-256`) | adaptive-scale config | deferred to the adaptive spec. |
| `artifact_dir` (`:235`) | phase-owned artifact path | |

`TimingConfig` (`config.py:58-120`) → a Rust `RunConfig { phases: Vec<PhaseConfig>,
request_cancellation, urls, url_selection_strategy }`. `from_run` (`config.py:86-120`)
becomes a plain builder: warmup phases first, then profiling phases, cancellation sourced
from the first profiling phase that declares one (`config.py:106-113`).

---

## 6. Mapping onto crates — built vs designed

| Concern | Primitive | Module / crate | Status |
|---|---|---|---|
| Clock (grace/timeout/duration/drain all via `now_ns`/`sleep`) | `Clock` (Real/Sim) | `aiperf_runtime::clock` | **built** |
| Stop bounds (count/session/duration/lifecycle) | `StopChecker` / `StopConfig` / `RunState` | `aiperf_runtime::timing` | **built** |
| Session + prefill caps, cross-phase debt-drain | `SlotPool` / `ConcurrencyManager` | `aiperf_runtime::timing` | **built** |
| Inter-arrival + `set_rate` ramp target | `IntervalGenerator` | `aiperf_runtime::timing` | **built** |
| Terminal / first-token → progress + slot release | `RequestObserver` | `loadgen-core` | **built** (terminal turn-final + TTFT-release hooks still needed — companion §7) |
| Single-loop credit issuer (the strategy) | `CreditIssuer` + continuation queue | `aiperf_runtime::timing` | **designed** (companion spec) |
| **Phase lifecycle state machine** | `PhaseLifecycle` (CREATED→…→COMPLETE + cancel flag) | `aiperf_runtime::timing` | **designed** (this spec) |
| **Per-phase execution driver** | `PhaseRunner` trait + impl | `aiperf_runtime::timing` | **designed** (this spec) |
| **Warmup→profiling sequencer, seamless overlap** | `PhaseOrchestrator` | `aiperf_runtime::timing` | **designed** (this spec) |
| **Progress / lifecycle emission (replaces publisher+bus)** | `PhaseObserver` trait | `aiperf_runtime::timing` | **designed** (this spec) |
| **Think-time deferred enqueue** | `LoopScheduler` (single owner) | `aiperf_runtime::timing` | **designed** (companion) |
| Ramping (stepped concurrency / continuous rate) | `Ramper` | `aiperf_runtime::timing` | **designed** |
| Top-level driver (replaces `TimingManager` service) | direct `run()` entry in the CLI | `aiperf` | **designed** — collapses the ZMQ service into a function |

### 6.1 New trait seams (every extension point a trait)

- **`PhaseRunner`** — `async fn run(&mut self, is_final_phase: bool) -> PhaseStats`. Owns the
  per-phase components; drives setup → issue → wait-sending → wait-returns → drain →
  finalize. One impl today; the trait keeps the door open for alternate drivers (e.g. a
  fixed-schedule replay driver with different timeout semantics).
- **`PhaseOrchestrator`** — `async fn run_all(&mut self) -> Vec<PhaseStats>`. Sequences the
  phase list, owns the shared `ConversationSource` / `ConcurrencyManager` / cancellation
  policy, manages seamless overlap + `active_runners`.
- **`PhaseObserver`** — the deletion target for `PhasePublisher`: `fn on_phase_start`,
  `on_progress`, `on_sending_complete`, `on_phase_complete(stats, branch_stats)`. Impls:
  console/live UI, report accumulator, silent. This is the seam that makes "progress
  publishing" a local trait call instead of a ZMQ broadcast.
- **`PhaseLifecycle`** — kept as a concrete state-machine struct (not a trait; there is one
  correct transition set) with `start/mark_sending_complete/mark_complete/cancel` +
  `time_left(include_grace)`; all timestamps via `Clock`.

---

## 7. Offline / online parity

Every phase time bound routes through `PhaseLifecycle.time_left_in_seconds` (duration +
grace) and the two fixed `Environment.TIMING` constants (drain, progress-interval). In
Python these read `perf_counter_ns()` and wall clock separately (§1.2). **In Rust all of it
goes through the injected `Clock`:**

- **Duration / grace / drain / progress-tick** become `Clock::now_ns` deltas and
  `Clock::sleep` waits.
- **ONLINE** (`RealClock`): wall-time behavior identical to Python (grace period is real
  seconds; drain timeout is real seconds).
- **OFFLINE** (`SimClock`): grace, duration, drain, and the progress-tick interval are all
  *virtual* — the DES advances to the next event, so a 30 s grace period costs zero
  wall-time and the whole phase escalation (timeout → cancel → drain → force-complete) is
  **deterministic and reproducible**. The `set_event_on_timeout` / force-complete paths
  fire at exact virtual instants.
- **Parity is code-path + `PhaseStats` schema, not byte-identical values** — the lifecycle
  emits the same transitions and the same stats struct in all three modes; only the
  underlying `Clock` differs (consistent with the companion spec §4 and the port-exact
  ledger). The wall-vs-perf timestamp duality (`lifecycle.py:47-60`) is *eliminated*, not
  ported — single-process needs no cross-service-comparable wall clock, so one monotonic
  `Clock` authority replaces both.

---

## 8. Build order (increments)

1. **`PhaseLifecycle` state machine** over `Clock` — CREATED→STARTED→SENDING_COMPLETE→
   COMPLETE + `was_cancelled` flag + `time_left(include_grace)`. Unit-test the invalid-
   transition guards and grace math against `lifecycle.py`.
2. **`PhaseObserver` trait + console/report impls** — replaces `PhasePublisher`; pass
   `PhaseStats` by value. Deletes the message-bus dependency wholesale.
3. **`PhaseRunner` single-phase happy path** — wire lifecycle + progress events + the
   companion's `CreditIssuer`; implement `_wait_for_sending_complete` (timeout, freeze,
   set event) and the non-seamless synchronous `_wait_for_returning_complete` fast-path +
   grace wait. No drain escalation yet.
4. **Drain escalation + force-completion** — cancel-all → `CANCEL_DRAIN_TIMEOUT` →
   `release_stuck_slots` → forced `mark_complete`. Port the `runner.py:648-689` ordering
   exactly; test with a mock sink that never returns some requests.
5. **`PhaseOrchestrator` sequential warmup→profiling** — shared `ConcurrencyManager` /
   `ConversationSource`; per-phase runner; non-seamless handoff + cross-phase debt-drain.
6. **Seamless overlap** — background return-wait task + `active_runners` bookkeeping +
   completion callback; verify concurrency doesn't collapse at the boundary.
7. **Ramping** — concurrency (stepped) + rate (continuous) rampers started before
   execution.
8. **Cancellation & failure finalization** — external cancel short-circuit + the failure
   flush (local finalization only; drop the peer-notification motivation).

Increments 1–5 deliver a full non-seamless warmup→profiling run online + offline; 6–8 add
seamless transitions, ramps, and robust cancel/fail handling.

---

## 9. Risks / open questions (the subtle races)

- **Freeze-before-check ordering.** `check_all_returned_or_cancelled` reads the *frozen*
  `final_requests_sent`; if the fast-path return check runs before `freeze_sent_counts`,
  it compares against a moving target (`runner.py:610-614, 632`;
  `progress_tracker.py:144-171`). The single-loop `!Send` model makes the freeze a plain
  struct copy, but the *sequence* (mark sending-complete → freeze → set event → then any
  return check) must be preserved exactly.
- **The branch-work conjunct in the returns fast-path.** `_wait_for_returning_complete`
  must AND the "all returned" check with "no pending branch work" (`runner.py:632-635`) and
  `_is_phase_complete` must consult `has_pending_branch_work` (`runner.py:237-240`).
  Dropping this closes a DAG phase mid-tree and freezes sent counts with children still in
  flight. Non-DAG runs skip it (orchestrator is `None`).
- **`grace_period_triggered` overloading.** The flag is set both on a real returns-timeout
  (`runner.py:692`) *and* on the cancellation short-circuit (`runner.py:401`) and on
  force-complete (`runner.py:687`). Consumers reading `grace_period_timeout_triggered`
  (`progress_tracker.py:218`) can't distinguish "graceful grace expiry" from "cancelled" —
  decide in Rust whether to split these into distinct reason enums.
- **Worker-readiness deadlock is deleted, but verify.** `wait_for_workers` guards
  "first credit before any worker registers" (`runner.py:368-370`). In single-process the
  dispatch tasks live on the same loop, so there is no registration window — but confirm
  the issuer never `spawn_local`s a dispatch before the sink is constructed.
- **Seamless multi-runner + shared `ConcurrencyManager`.** With two runners active
  (`phase_orchestrator.py:192-197`), both touch the *same* `ConcurrencyManager` while one
  is draining and one is issuing. The debt-drain must be correct across the overlap; test a
  warmup(seamless)→profiling handoff where warmup's returns land *after* profiling starts
  issuing.
- **Late arrivals after COMPLETE.** `increment_returned` may be called after
  `mark_complete` (`progress_tracker.py:126-127`); the caller must gate on
  `lifecycle.is_complete`. The frozen `final_*` counts protect the report, but the
  `SlotPool` release on a post-complete return must not underflow.
- **Force-completion slot release.** `_release_stuck_slots` frees session + prefill slots
  for credits that will never return (`runner.py:701-710`); if the counts are wrong the
  *next* phase (sharing the manager) starts with corrupted slot availability.
- **Progress-loop cancellation timing.** The progress task is cancelled at different points
  in seamless (`_on_return_wait_complete`, `runner.py:275`) vs non-seamless
  (`runner.py:415`) vs external-cancel (`runner.py:261`) paths; ensure the final
  phase-complete emission isn't lost to a too-early cancel.

---

## 10. One-line summary

The phase driver is a **`PhaseLifecycle` state machine** (CREATED→STARTED→SENDING_COMPLETE→
COMPLETE + orthogonal cancel flag) wrapped by a **`PhaseRunner`** that sequences
setup → issue → wait-sending(duration timeout) → wait-returns(grace) → cancel-drain →
force-complete → finalize, and a **`PhaseOrchestrator`** that runs warmup→profiling over a
shared debt-draining `ConcurrencyManager` with optional seamless overlap — with the entire
ZMQ layer (`PhasePublisher`, the `TimingManager` service, `wait_for_workers`, the credit
router callbacks) collapsing into direct `PhaseObserver` / `RequestObserver` trait calls on
one `!Send` loop, all time via `Clock` so grace/timeout/drain run in virtual time offline
for a deterministic, reproducible phase escalation.

---

## Implementation addendum (2026-07-11)

**Status: built.** This addendum supersedes the original designed-status tables. The
process-independent phase policy is implemented in the `aiperf_runtime::timing` module; the normal scheduled
application pipeline is connected in the rest of the `aiperf-runtime` crate. No ZMQ, service lifecycle, worker-registration
handshake, or credit-router wire type was reintroduced.

### Built symbols and ownership

- `rust/runtime/src/timing/phase/config.rs` owns validated `PhaseConfig`, `PhaseKind`, and an
  explicit `GracePeriod::{Disabled, Finite, Infinite}`. The enum removes the ambiguous
  `None`/zero/infinity encoding while preserving warmup's infinite-drain default.
- `lifecycle.rs` implements the validated CREATED → STARTED → SENDING_COMPLETE → COMPLETE
  state machine over one injected `Clock`. `PhaseCompletionReason` splits Python's overloaded
  grace bit into completed, grace-timeout, cancelled, force-completed, and failed outcomes;
  compatibility booleans remain in `PhaseStats`.
- `progress.rs` implements local-loop counters, sent/completed freeze snapshots, the two
  one-shot notifications, first-token prefill accounting, late-return protection, and the
  pending-branch conjunct. No hot-path lock or cross-thread atomic is used.
- `runner.rs` provides the object-safe `PhaseRunner`, `PhaseExecution`, and
  `PhaseExecutionFactory` seams plus `ClockPhaseRunner`. Its production ordering is configure
  → setup → STARTED/progress → ramps → execute → duration timeout/freeze/cancel-pending →
  return grace → cancel-all → bounded drain → stuck-slot release → COMPLETE/finalize. Setup
  and execution failures flush the local lifecycle and terminal observer event.
- `orchestrator.rs` provides `PhaseOrchestrator`, `PhaseRunnerFactory`, and
  `ClockPhaseOrchestrator`. It validates unique ids, warmup-before-profiling order, and the
  presence of a profiling phase; retains overlapping seamless runners until their background
  return waits finish; and shares one execution factory so slot debt survives handoff. An
  orchestration-level cancellation latch prevents a cancelled warmup from advancing into
  profiling.
- `observer.rs` replaces `PhasePublisher` with direct `PhaseObserver` calls and supplies
  no-op, recording/report, and console implementations. `on_phases_complete` replaces the
  former credits-complete publication for in-process consumers.
- `aiperf/src/phase_runtime.rs` adapts ordinary `Workload` + `ScheduledRuntime` +
  `TurnDispatcher` instances into phase executions. `TurnLifecycleObserver` records accepted
  sends synchronously before dispatch-task polling, then TTFT and terminal callbacks on the
  same `LocalSet`. Each phase owns independent metrics/report state while the factory retains
  shared admission resources and the backend dispatcher.
- `ScheduledPhaseResources` configures shared session/prefill `SlotPool`s before issuance;
  `SlotPoolPhaseResources` preserves debt across seamless phases. Workload-specific resource
  implementations can additionally release guards held outside scheduler tasks on the force
  path.
- `RampScheduledPhaseController` owns prepared `RampDriver`s, applies their initial values
  synchronously before issuance, and stops/joins their tasks at sending handoff. Report
  finalization waits for returns; detached terminal record processors are joined only after
  the phase window and report have closed, so grading/consumer latency cannot stretch phase
  timing.
- The pre-existing scheduled application entry points now lower even a one-phase run through
  `run_scheduled_phases`; this is the direct in-process top-level driver replacing
  `TimingManager`. Their historical drain behavior is retained with infinite grace. Explicit
  multi-phase callers choose disabled, finite, or infinite grace per plan and receive ordered
  `PhaseStats` plus phase-tagged scheduled reports.

### Executable proof

- `aiperf_runtime::timing` unit tests pin lifecycle transition errors, one-clock deadline arithmetic,
  source-compatible defaults, freeze/late-return behavior, session/root counting, and branch
  completion gating.
- `aiperf/tests/timing_phase_runner.rs` uses `SimClock` to prove the happy path, exact duration+grace
  deadline, cancellation drain, exact force-completion instant and stuck-slot release,
  external-cancel short circuit, progress ticks, and failure lifecycle flush.
- `aiperf/tests/timing_phase_orchestrator.rs` proves both non-seamless drain-before-start and seamless
  overlap. Its shared `SlotPool` case lowers warmup capacity 4 → profiling capacity 3 while
  four warmup guards remain live: debt is one, profiling blocks until a warmup return repays
  it, and the pool finishes with limit three/debt zero. It also proves cancellation cannot
  start the next phase.
- `aiperf/tests/phase_runtime_sim.rs` repeats seamless overlap and debt drain through the real
  `ScheduledRuntime` adapter, proves processors are joined outside the phase window, and pins
  ramp-before-issuance/stop-at-handoff behavior.
- `aiperf/tests/phase_runtime_online.rs` dispatches a seamless warmup and profiling phase
  through the real Clock-injected hyper `TransportSink` against an in-process SSE server and
  proves profiling starts before warmup's delayed HTTP return. Thus virtual and wall-clock
  modes use the same phase driver and stats schema.

## Graph-IR convergence addendum (2026-07-12)

**Status: built.** `rust/runtime/src/engine/graph_phase_runtime.rs` is the one
backend-neutral Graph-IR phase adapter. It owns root source/arrival/admission
composition, `ClockPhaseOrchestrator`, exact node and first-token progress,
duration/grace/drain/force escalation, seamless handoff, ramps, adaptive
concurrency/prefill/request-rate control, and phase-tagged record capture.

`RunnerGraphPhaseBackendFactory` is the only mode seam beneath that policy.
The online implementation constructs the existing thread-per-core HTTP
placement; the Dynamo implementation returns phase-local execution backends
over one run-scoped `SimClock`, engine, compatibility observer, native-metrics
observer, segment store, and UUID stream. Both therefore execute the same
already-lowered `GraphInputBundle`; neither reparses the authored DAG nor
converts it through `Dataset`, `Conversation`, protocol v1, or another Graph-IR
representation.

The process tests in `rust/cli/tests/{online_v2_stdio,offline_stdio}.rs`
prove direct pair loading. The offline proof authors warmup plus profiling in one protocol-v2
request and observes both phases on the same Dynamo engine/clock, including
warmup metrics and six node terminals from two executions of one three-node
trace.
