<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Unified phase orchestration — one lifecycle seam for scheduled and graph runs

One `Clock`-native phase lifecycle (`ClockPhaseOrchestrator` → `ClockPhaseRunner` →
`PhaseLifecycle`) drives BOTH the scheduled (`ScheduledPhaseExecutionFactory`) and
graph (`GraphPhaseExecutionFactory`) execution paths through the same
`PhaseExecutionFactory` / `PhaseExecution` seam, the same `drive_phases` signal-cancel
helper, and the same `start_phase_sidecars` / `finish_phase_sidecars` barriers — so a
change to the duration → grace → cancel → drain → force escalation, the warmup → profiling
ordering rule, or the Ctrl-C latch happens once and both workloads inherit it.

- **Date:** 2026-07-16
- **Status:** Built (grounded in `rust/runtime/src/timing/phase/**` +
  `rust/runtime/src/phase_runtime.rs` + `rust/runtime/src/engine/graph_phase_runtime.rs`)
- **Scope:** the phase-orchestration seam only — not the transports, workloads,
  metrics, or the graph t*/warmup-handoff machinery those files also host.

---

## 1. Why this seam exists

AIPerf runs a benchmark as an ordered list of *phases* — zero or more warmup phases
followed by at least one profiling phase. Each phase issues load, closes issuance at a
duration/count bound, then waits for outstanding requests to return within a grace
window before escalating to cancellation. This policy is identical whether the load is a
scheduled arrival stream (Poisson/Gamma/constant/burst/concurrency) or a Graph-IR trace
replay. Duplicating it would mean two copies of the lifecycle state machine, two copies
of the escalation ladder, and — as §5 documents — two copies of the signal-cancellation
wiring that drifted apart and left graph runs un-cancellable.

The seam is therefore a single object-safe trait pair:

- `PhaseExecutionFactory::create(config, context) -> Rc<dyn PhaseExecution>`
  (`rust/runtime/src/timing/phase/runner.rs:287`) — the workload's only extension point.
- `PhaseExecution` (`rust/runtime/src/timing/phase/runner.rs:231`) — the
  configure/setup/start_ramps/execute/stop/cancel/finalize adapter, all methods `!Send`
  and object-safe, all default hooks no-ops.

Everything above that trait — lifecycle transitions, deadlines, the escalation ladder,
progress emission, ordered multi-phase sequencing, signal cancellation — is owned by the
shared driver and is written exactly once.

---

## 2. Component map

```
                         ┌───────────────────────────────────────────────┐
                         │  timing::phase  (the shared, workload-neutral   │
                         │                  orchestration crate-module)    │
                         └───────────────────────────────────────────────┘

   PhaseOrchestrator (trait)                 orchestrator.rs:75
     └─ ClockPhaseOrchestrator               orchestrator.rs:136   run_all / cancel
          owns: Vec<PhaseConfig>, active runners, seamless-failure signal,
                run_started / cancelled latches
          uses: validate_phase_order()       orchestrator.rs:339   (warmup→profiling)

   PhaseRunnerFactory (trait)                orchestrator.rs:26
     └─ ClockPhaseRunnerFactory              orchestrator.rs:37     one shared
          {clock, observer, execution_factory} → fresh runner per phase

   PhaseRunner (trait)                       runner.rs:320
     └─ ClockPhaseRunner                     runner.rs:356   run_inner escalation ladder
          owns per phase: PhaseLifecycle, PhaseProgress, StopChecker,
                          CancellationSignal, progress loop, return task

   PhaseLifecycle                            lifecycle.rs:80    Created→Started→
          the transition-validated state machine                SendingComplete→Complete

   PhaseExecutionFactory (trait)             runner.rs:287   ← THE WORKLOAD SEAM
   PhaseExecution (trait)                    runner.rs:231
        ├─ ScheduledPhaseExecution           phase_runtime.rs:1135
        │     built by ScheduledPhaseExecutionFactory   phase_runtime.rs:907
        └─ GraphPhaseExecution               graph_phase_runtime.rs:1505
              built by GraphPhaseExecutionFactory        graph_phase_runtime.rs:1713

   drive_phases(orchestrator, clock_is_virtual)   orchestrator.rs:434   ← THE ENTRY HELPER
     └─ spawn_cancel_on_signal / SignalCancelGuard  orchestrator.rs:451 / 529

   start_phase_sidecars / finish_phase_sidecars   phase_runtime.rs:210 / 230
     └─ shared barrier helpers used by both PhaseExecution impls
```

---

## 3. The phase lifecycle state machine

`PhaseLifecycle` (`rust/runtime/src/timing/phase/lifecycle.rs:80`) is a strict
transition-validated state machine over one injected `Clock` — Rust deliberately removes
Python's wall-clock/perf-clock duality; every timestamp comes from the same clock
(`lifecycle.rs:4-8`). The four states are `Created → Started → SendingComplete → Complete`
(`PhaseState`, `lifecycle.rs:22`). Cancellation is an *orthogonal* boolean latch
(`was_cancelled`), not a state — set by `cancel()` (`lifecycle.rs:162`) at any point,
recorded in the snapshot, and disambiguated from a grace timeout by a distinct
`PhaseCompletionReason` (`lifecycle.rs:39`): `Completed`, `GraceTimeout`, `Cancelled`,
`ForceCompleted`, `Failed`.

The runner (`ClockPhaseRunner::run_inner`, `runner.rs:439`) walks this machine and drives
the escalation ladder around it. The full lifecycle including the
duration → grace → cancel → drain → force escalation:

```
  ┌──────────┐
  │ Created  │   PhaseLifecycle::new (lifecycle.rs:89)
  └────┬─────┘   was_cancelled = false (orthogonal latch, never a state)
       │
       │  runner: configure() → setup() (execution seam)      runner.rs:440-448
       │  lifecycle.start()  (stamps started_at_ns)           lifecycle.rs:110
       ▼
  ┌──────────┐   progress loop spawned (runner.rs:735)
  │ Started  │   observer.on_phase_start                      runner.rs:452
  └────┬─────┘   execution.start_ramps() then
       │         execute_until_sending_complete()             runner.rs:464 / 511
       │
       │  ┌──────────────────── SENDING WINDOW ───────────────────────┐
       │  │ wait_for_sent(timeout = time_left_ns(false))  runner.rs:524│
       │  │   timeout = expected_duration_ns − elapsed    lifecycle.rs:175│
       │  │                                                            │
       │  │   Event     → issuer exhausted plan   ─┐                   │
       │  │   TimedOut  → duration deadline hit    ├─→ stop_issuing()  │
       │  │   Cancelled → latch / signal fired     ─┘   runner.rs:526  │
       │  └────────────────────────────────────────────────────────────┘
       ▼
  ┌──────────────────┐  finish_sending(timed_out)             runner.rs:540
  │ SendingComplete  │  freeze_sent_counts, on_sending_complete
  └────┬─────────────┘
       │
       │  Cancelled during sending?  ── yes ──► cancel_inflight → stop_ramps
       │      run_inner runner.rs:467              → complete_phase(Cancelled)  runner.rs:479
       │  no │
       ▼
  ┌──────────────────── RETURN GRACE / ESCALATION LADDER ─────────────────────┐
  │ finish_returning()                                          runner.rs:556  │
  │                                                                            │
  │  (1) all already returned?  → reason = Completed                           │
  │        check_all_returned_or_cancelled                      runner.rs:561  │
  │                                                                            │
  │  (2) else wait_for_returned(timeout = time_left_ns(true))   runner.rs:564  │
  │        grace window = duration + grace_period               lifecycle.rs:178│
  │        ├─ Event    → every request returned → Completed                    │
  │        ├─ TimedOut → grace deadline elapsed  ─┐                            │
  │        └─ Cancelled→ external cancel observed ─┤                           │
  │                                                ▼                           │
  │  (3) CANCEL:  execution.cancel_inflight()                   runner.rs:572  │
  │                                                                            │
  │  (4) DRAIN:   wait_for_returned(cancel_drain_timeout_ns,    runner.rs:575  │
  │               observe_cancellation = false)                                │
  │        ├─ drained → GraceTimeout (if TimedOut) or Cancelled runner.rs:580  │
  │        └─ NOT drained ▼                                                     │
  │                                                                            │
  │  (5) FORCE:   mark_cancel_drain_timeout()                   runner.rs:588  │
  │               release_stuck_slots() + force_all_returned()  runner.rs:590  │
  │               reason = ForceCompleted                       runner.rs:595  │
  └────────────────────────────────┬───────────────────────────────────────────┘
                                    │  complete_phase(reason)   runner.rs:600 / 607
                                    ▼
                          ┌──────────────┐  lifecycle.mark_complete(reason)  lifecycle.rs:140
                          │  Complete    │  freeze_completed_counts, finalize()
                          └──────────────┘  on_phase_complete, stop_progress_loop
```

Notes grounded in code:

- **Deadlines are `Clock`-relative, never `tokio::time`.** `time_left_ns(false)` is the
  bare duration; `time_left_ns(true)` adds the grace period, saturating at zero, and
  returns `None` for infinite grace or missing duration (`lifecycle.rs:175-192`). Every
  wait selects on a `clock.sleep(...)` future (`runner.rs:665`, `runner.rs:700`).
- **`GraceTimeout` vs `Cancelled` vs `ForceCompleted` are kept distinct** precisely
  because Python overloaded one `grace_period_triggered` bit for three paths
  (`lifecycle.rs:34-38`); `mark_complete` folds only `GraceTimeout`/`ForceCompleted` into
  `grace_period_timeout_triggered` and only `ForceCompleted` into `forced_completion`
  (`lifecycle.rs:152-156`).
- **Failure path.** Any execution error routes through `finalize_failure`
  (`runner.rs:629`): it stops issuing, cancels in-flight, force-returns, and completes
  with `PhaseCompletionReason::Failed` so a partial native report is still written.

---

## 4. Multi-phase sequencing and the warmup → profiling rule

`ClockPhaseOrchestrator::new` validates the whole phase list up front via
`validate_phase_order` (`orchestrator.rs:339`) before any runner is constructed:

- non-empty (`NoPhases`, `orchestrator.rs:340`),
- unique stable ids (`DuplicatePhaseId`, `orchestrator.rs:352`),
- **no warmup may follow a profiling phase** (`WarmupAfterProfiling`,
  `orchestrator.rs:356`), and
- **at least one profiling phase must exist** (`ProfilingPhaseRequired`,
  `orchestrator.rs:365`).

`run_all_entry` (`orchestrator.rs:161`) then constructs one fresh `ClockPhaseRunner` per
config in order, awaits each phase's `run(is_final)` and — for non-seamless phases —
`wait_complete()`, pruning completed runners between phases. A seamless non-final phase
hands off in the background (`spawn_active_cleanup`, `orchestrator.rs:222`) and its
failure is surfaced through the `SeamlessFailureSignal` (`orchestrator.rs:102`).
`cancel_active` (`orchestrator.rs:296`) flips each active runner's lifecycle latch to
cancelled *before* cancelling the shared backend, guaranteeing a `was_cancelled = true`
snapshot rather than a racing normal completion.

```
   run_all_entry (orchestrator.rs:161)
     for config in configs:                     ── ordered, config order
        create fresh ClockPhaseRunner            orchestrator.rs:184
        runner.run(is_final)                     orchestrator.rs:194  ── sending
        if seamless && !final: spawn cleanup     orchestrator.rs:198
        else: runner.wait_complete()             orchestrator.rs:200  ── returns
     collect final PhaseStats in config order    orchestrator.rs:209-217
     observer.on_phases_complete                 orchestrator.rs:218
```

---

## 5. `drive_phases` — the shared signal-cancellation entry (and the bug it fixed)

`drive_phases(orchestrator, clock_is_virtual)`
(`rust/runtime/src/timing/phase/orchestrator.rs:434`) is the one helper both entry points
call to run the orchestrator under the standard SIGINT/SIGTERM discipline:

```rust
pub(crate) async fn drive_phases(
    orchestrator: ClockPhaseOrchestrator,
    clock_is_virtual: bool,
) -> Result<Vec<PhaseStats>, PhaseOrchestratorError> {
    let signal_guard = spawn_cancel_on_signal(orchestrator.clone(), !clock_is_virtual); // :438
    let result = orchestrator.run_all().await;                                           // :439
    drop(signal_guard);                                                                   // :440
    result
}
```

`spawn_cancel_on_signal` (`orchestrator.rs:451`, unix; `:491`, windows) spawns a
`LocalSet` task that awaits the first SIGINT/SIGTERM (`tokio::signal`, async, runtime
driven — no raw OS handler, so the thread-per-core `!Send` model holds) and calls
`orchestrator.cancel()` once. The returned `SignalCancelGuard` (`orchestrator.rs:529`)
aborts the listener on drop (`orchestrator.rs:535`). The listener is **armed only under a
wall clock**: `enabled = !clock_is_virtual`, because the virtual (offline) clock builds a
bare `current_thread` runtime with no I/O/signal driver, where `tokio::signal` would panic
— so a deterministic offline run has no wall-clock owner to Ctrl-C and skips the listener
(`orchestrator.rs:456-460`, guard `handle: None` at `:483`).

### The bug this fixed

`spawn_cancel_on_signal` and `SignalCancelGuard` now live in
`timing::phase::orchestrator`, and both the scheduled and graph entries funnel through
`drive_phases`. Previously the signal listener was wired into the **scheduled** entry only
and was NOT wired into the **graph** entry, so graph runs (`dag_jsonl`/`weka_trace`/
`dynamo_trace`) were **not Ctrl-C-cancellable** — an interrupt during a long trace replay
could not drain the active phase into partial results. Moving the listener into the shared
`drive_phases` (with `spawn_cancel_on_signal` relocated into
`timing::phase::orchestrator`) means the listener cannot be attached to one path and
forgotten on the other (`orchestrator.rs:425-433` doc). Graph now inherits the exact same
first-signal → cancellation-latch → `PhaseStats { was_cancelled: true }` behavior as
scheduled.

### The two call sites (verified)

Both capture `clock.is_virtual()` **before** the clock is moved into the runner factory,
then call `drive_phases`:

- **Scheduled:** `rust/runtime/src/phase_runtime.rs`
  - `let clock_is_virtual = clock.is_virtual();` — `phase_runtime.rs:757`
  - `ClockPhaseOrchestrator::new(configs, runner_factory, observer)` — `phase_runtime.rs:763`
  - `let phase_result = drive_phases(orchestrator, clock_is_virtual).await` — `phase_runtime.rs:770`

- **Graph:** `rust/runtime/src/engine/graph_phase_runtime.rs`
  - `let clock_is_virtual = clock.is_virtual();` — `graph_phase_runtime.rs:2066`
  - `ClockPhaseOrchestrator::new(phase_configs, runner_factory, phase_observer)?` — `graph_phase_runtime.rs:2072`
  - `let run_result = drive_phases(orchestrator, clock_is_virtual).await;` — `graph_phase_runtime.rs:2082`

```
  Scheduled entry                              Graph entry
  run_scheduled_phases_inner                   run_graph_phases
  phase_runtime.rs:717                         graph_phase_runtime.rs:1977
        │                                             │
        │  clock_is_virtual = clock.is_virtual()      │  clock_is_virtual = clock.is_virtual()
        │  :757                                        │  :2066
        │                                             │
        │  ScheduledPhaseExecutionFactory              │  GraphPhaseExecutionFactory
        │  :740  (impl PhaseExecutionFactory)          │  :2052  (impl PhaseExecutionFactory)
        │                                             │
        │  ClockPhaseRunnerFactory::new                │  ClockPhaseRunnerFactory::new
        │  :758                                        │  :2067
        │                                             │
        │  ClockPhaseOrchestrator::new                 │  ClockPhaseOrchestrator::new
        │  :763                                        │  :2072
        │                                             │
        └───────────────┐               ┌─────────────┘
                        ▼               ▼
              ╔══════════════════════════════════════╗
              ║  drive_phases(orchestrator,           ║   orchestrator.rs:434
              ║               clock_is_virtual)       ║
              ║   ├─ spawn_cancel_on_signal            ║   orchestrator.rs:451
              ║   │    (enabled = !clock_is_virtual)   ║   ← armed ONLY under wall clock
              ║   ├─ orchestrator.run_all().await      ║   orchestrator.rs:439
              ║   └─ drop(SignalCancelGuard) → abort   ║   orchestrator.rs:440 / 535
              ╚══════════════════════════════════════╝
                                │
                                ▼
                   ONE ClockPhaseOrchestrator
                   ONE ClockPhaseRunner per phase
                   ONE PhaseLifecycle escalation ladder
```

The scheduled path additionally waits on its record processors after `drive_phases`
returns and joins the two error channels (`phase_runtime.rs:773-783`); the graph path
consults the warmup-failure ledger and, if non-empty, aborts with the structured
`trajectory_warmup_failed` envelope before propagating the run result
(`graph_phase_runtime.rs:2083-2087`). These are post-drive concerns layered on top of the
identical orchestration — they do not fork the lifecycle.

---

## 6. The shared sidecar barriers

Barrier-synchronized, non-per-token control-plane work (GPU/network/server-metrics
side-channel accumulators, profilers) runs through the `ScheduledPhaseSidecar` trait
(`phase_runtime.rs:192`): `start()` before the first turn is issued, `finish()` after
dispatch fully drains, with `on_phase_start` / `on_phase_end` instants marked around them.

Both workloads share two helper functions rather than each carrying a byte-identical loop
(they were previously duplicated). The `label` parameter selects the workload word in the
error context (`"scheduled"` vs `"graph"`):

- `start_phase_sidecars(sidecars, clock, label)` — `phase_runtime.rs:210`
  starts every sidecar, then stamps `on_phase_start(clock.now_ns())` on each.
- `finish_phase_sidecars(sidecars, clock, label)` — `phase_runtime.rs:230`
  stamps `on_phase_end(clock.now_ns())` on each, then finishes every sidecar.

Call sites:

| Barrier | `PhaseExecution` hook | Scheduled | Graph |
|---|---|---|---|
| `start_phase_sidecars(.., "scheduled" / "graph")` | `setup()` | `phase_runtime.rs:1148-1152` | `graph_phase_runtime.rs:1525-1529` |
| `finish_phase_sidecars(.., "scheduled" / "graph")` | `finalize()` | `phase_runtime.rs:1258` | `graph_phase_runtime.rs:1641` |

```
   PhaseExecution::setup()   ──► start_phase_sidecars(&sidecars, clock, label)   phase_runtime.rs:210
        (runner calls setup before lifecycle.start — runner.rs:445-450)
          for s in sidecars: s.start().await          (error ⇒ "starting {label} phase sidecar")
          phase_start_ns = clock.now_ns()
          for s in sidecars: s.on_phase_start(phase_start_ns)

   PhaseExecution::finalize()  ──► finish_phase_sidecars(&sidecars, clock, label) phase_runtime.rs:230
        (runner calls finalize inside complete_phase — runner.rs:618)
          phase_end_ns = clock.now_ns()
          for s in sidecars: s.on_phase_end(phase_end_ns)
          for s in sidecars: s.finish().await          (error ⇒ "finishing {label} phase sidecar")
```

(The helper doc comment at `phase_runtime.rs:229` says "execute finish paths"; the actual
scheduled and graph call sites invoke `finish_phase_sidecars` from `finalize()` — after
`scheduler().wait_idle()` / the record-drain join — which is the true post-drain point.)

---

## 7. What each workload contributes (and what it does not)

A workload contributes exactly one `PhaseExecutionFactory` plus its `PhaseExecution`
adapter. It never implements the lifecycle, the escalation ladder, the ordering rule, or
the signal listener.

| Concern | Scheduled | Graph |
|---|---|---|
| Factory | `ScheduledPhaseExecutionFactory::create` — `phase_runtime.rs:907` | `GraphPhaseExecutionFactory::create` — `graph_phase_runtime.rs:1713` |
| Adapter | `ScheduledPhaseExecution` — `phase_runtime.rs:1135` | `GraphPhaseExecution` — `graph_phase_runtime.rs:1505` |
| `setup` | `start_phase_sidecars(.., "scheduled")` — `:1148` | `start_phase_sidecars(.., "graph")` — `:1525` |
| `execute` | drives `Workload::execute` + realtime block — `:1154` | single-pass workload or `GraphPressureRecycle` — `:1543` |
| `cancel_inflight` | `scheduler().cancel_all` + tracker — `:1215` | `placement.cancel_inflight` — `:1603` |
| `release_stuck_slots` | tracker + `resources.release_stuck` — `:1226` | (graph slot pools) |
| `finalize` | `finish_phase_sidecars(.., "scheduled")` + report — `:1244` | drain join + `finish_phase_sidecars(.., "graph")` — `:1626` |
| `PhaseContext` progress bridge | `PhaseDispatchTracker` (per-turn) — `:1281` | `GraphPhaseProgress` (per-node) — `graph_phase_runtime.rs:1053` |

Both adapters mutate the *same* runner-owned `PhaseProgress` through the *same*
`PhaseContext` (`runner.rs:110`) — scheduled per accepted turn (`PhaseDispatchTracker`,
`phase_runtime.rs:1288`), graph per admitted trace and per returned node
(`GraphPhaseProgress::admit`/`record`, `graph_phase_runtime.rs:1127`/`:1168`). The runner
reads that progress to decide the escalation ladder identically for both.

---

## 8. Extension discipline

Per the repository's non-negotiable trait-seam rule, adding a third execution mode (e.g. a
future dynosim-native phase driver or a new arrival family) means implementing
`PhaseExecutionFactory` + `PhaseExecution` and reusing:

- `ClockPhaseOrchestrator` / `ClockPhaseRunnerFactory` / `ClockPhaseRunner` unchanged
  (`orchestrator.rs:136` / `:37`, `runner.rs:356`);
- `drive_phases` for signal cancellation (`orchestrator.rs:434`) — never re-wire the
  listener into a new entry;
- `start_phase_sidecars` / `finish_phase_sidecars` for any barrier-synchronized
  control-plane work (`phase_runtime.rs:210` / `:230`).

The only per-mode code is the factory, the adapter, and the progress bridge that maps the
mode's dispatch events onto `PhaseContext`. Nothing in the lifecycle, the ordering rule,
the escalation ladder, or the cancellation discipline is duplicated.

---

## 9. Source index

| Item | File:line |
|---|---|
| `PhaseState` machine | `rust/runtime/src/timing/phase/lifecycle.rs:22` |
| `PhaseCompletionReason` | `rust/runtime/src/timing/phase/lifecycle.rs:39` |
| `PhaseLifecycle` transitions | `lifecycle.rs:110` (start), `:120` (sending), `:140` (complete), `:162` (cancel), `:167` (drain-timeout) |
| `time_left_ns` (duration/grace deadline) | `lifecycle.rs:175` |
| `ClockPhaseRunner::run_inner` (ladder) | `rust/runtime/src/timing/phase/runner.rs:439` |
| `finish_returning` (grace→cancel→drain→force) | `runner.rs:556` |
| `complete_phase` / `finalize_failure` | `runner.rs:607` / `:629` |
| `PhaseExecution` / `PhaseExecutionFactory` traits | `runner.rs:231` / `:287` |
| `PhaseOrchestrator` / `ClockPhaseOrchestrator` | `orchestrator.rs:75` / `:136` |
| `run_all_entry` / `cancel_active` | `orchestrator.rs:161` / `:296` |
| `validate_phase_order` (warmup→profiling) | `orchestrator.rs:339` |
| `drive_phases` | `orchestrator.rs:434` |
| `spawn_cancel_on_signal` / `SignalCancelGuard` (relocated here) | `orchestrator.rs:451` / `:529` |
| Scheduled `drive_phases` call site | `rust/runtime/src/phase_runtime.rs:770` |
| Graph `drive_phases` call site | `rust/runtime/src/engine/graph_phase_runtime.rs:2082` |
| `start_phase_sidecars` / `finish_phase_sidecars` | `phase_runtime.rs:210` / `:230` |
| Scheduled sidecar calls (setup/finalize) | `phase_runtime.rs:1148` / `:1258` |
| Graph sidecar calls (setup/finalize) | `graph_phase_runtime.rs:1525` / `:1641` |
