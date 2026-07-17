<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Phase orchestration

## Purpose

A benchmark runs as an ordered list of phases: zero or more warmup phases
followed by at least one profiling phase. Each phase issues load, closes
issuance at a duration or count bound, waits for outstanding requests within a
grace window, then escalates to cancellation and force completion. One
`Clock`-native lifecycle drives both scheduled and graph runs through a single
seam, so a change to the escalation ladder, the warmup→profiling ordering, or
the Ctrl-C latch happens once and both workloads inherit it.

## Built

### The seam

`aiperf_runtime::timing::phase` owns the workload-neutral orchestration. The
workload's only extension point is a single object-safe trait pair:

- `PhaseExecutionFactory::create(config, context) -> Rc<dyn PhaseExecution>`.
- `PhaseExecution`: the `!Send`, object-safe
  configure/setup/start_ramps/execute/stop/cancel/finalize adapter; all default
  hooks are no-ops.

The two implementations are `ScheduledPhaseExecution` (built by
`ScheduledPhaseExecutionFactory` in `phase_runtime`) and `GraphPhaseExecution`
(built by `GraphPhaseExecutionFactory` in `graph_phase_runtime`). Everything
above the trait — lifecycle transitions, deadlines, escalation, progress
emission, ordered multi-phase sequencing, and signal cancellation — is written
once.

### Lifecycle state machine

`PhaseLifecycle` is a strict transition-validated state machine over one injected
`Clock`; every timestamp comes from the same clock (no wall-clock/perf-clock
duality). States are `Created → Started → SendingComplete → Complete`.
Cancellation is an orthogonal boolean latch (`was_cancelled`), not a state, set
at any point and disambiguated from a grace timeout by a distinct
`PhaseCompletionReason` (`Completed`, `GraceTimeout`, `Cancelled`,
`ForceCompleted`, `Failed`).

`ClockPhaseOrchestrator` (`PhaseOrchestrator`) owns the phase list, active
runners, the seamless-failure signal, and the run-started/cancelled latches; it
validates warmup→profiling ordering. `ClockPhaseRunnerFactory` builds a fresh
`ClockPhaseRunner` per phase from the shared clock, observer, and execution
factory. Each runner drives the escalation ladder
(duration → grace → cancel → drain → force), the progress loop, and the return
task around the lifecycle machine. Cancellation cannot advance phases; failures
finalize the run; background returns are seamless; cross-phase request debt
drains.

### Shared entry and barriers

`drive_phases(orchestrator, clock_is_virtual)` is the shared signal-cancel entry
both the scheduled and graph paths call; it installs
`spawn_cancel_on_signal` / `SignalCancelGuard` so a graph run
(`dag_jsonl`/`weka_trace`/`dynamo_trace`) is Ctrl-C-cancellable through the same
listener as scheduled runs. `start_phase_sidecars` / `finish_phase_sidecars` are
the shared barrier helpers both `PhaseExecution` implementations use to bound
server-metrics, GPU, and network sidecars against the run window.

## Source anchors

- `rust/runtime/src/timing/phase/` (`orchestrator.rs`, `runner.rs`,
  `lifecycle.rs`): the shared driver, `drive_phases`, `PhaseExecution*` seam,
  `PhaseLifecycle`.
- `rust/runtime/src/phase_runtime.rs` (`ScheduledPhaseExecution*`,
  `start_phase_sidecars`/`finish_phase_sidecars`).
- `rust/runtime/src/engine/graph_phase_runtime.rs` (`GraphPhaseExecution*`).
