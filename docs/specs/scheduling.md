<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Scheduling

## Purpose

Define the scheduled workload shapes and their shared runtime. Scheduled
workloads provide request-rate, concurrency, user-centric, and fixed-schedule
operation over one `Clock`-backed `ScheduledRuntime`, so the identical code runs
online-real, online-mock, and offline. A workload never knows which clock or
transport it drives.

## Built

### Shared runtime

`aiperf_runtime::scheduled::ScheduledRuntime` and the `Workload` trait host the
scheduled strategies over shared Clock-backed seams; synthetic and
dataset-backed runner requests materialize turns from the segment store and share
the adaptive/ancillary actuators, native metrics, and reports. Admission uses
`aiperf_runtime::timing::SlotPool` (a semaphore plus debt tracking for graceful
ramp-down, layered global plus per-phase); bounds use
`aiperf_runtime::timing::StopChecker` (first-reached-wins: lifecycle,
request-count, session-count).

### Request-rate and concurrency (`RequestRateWorkload`)

`RequestRateWorkload` is a single-loop credit issuer: one credit (permission to
send one turn) per rate interval. `--request-rate` paces **turns**, not
conversation arrivals — a conversation's turns interleave with every other
conversation's, and continuation turns have priority over new-session starts. The
next interval is drawn before issuing. Local and `sharded` pacing retains an
absolute target while lag stays within `AIPERF_TIMING_MAX_CATCHUP_SECONDS`
(default 0.01 seconds, finite `0..=10`) and re-anchors only beyond that bounded
window. The value is resolved once at workload construction, so the issue loop
does only integer arithmetic; `0` restores strict no-burst behavior. Linux
`RealClock` waits use `CLOCK_MONOTONIC` timerfd through Tokio `AsyncFd`, while
`SimClock` drives the identical policy deterministically. Global dispatch modes
retain every `GlobalRateGate` slot for aggregate and corpus-position ordering and
do not apply the local re-anchor policy. Two concurrency dimensions gate
issuance:

- **Session slot** — one per conversation, acquired on turn 0 only, released on
  the root final turn plus in-flight cleanup at phase end; caps concurrent
  conversations.
- **Prefill slot** — one per request, acquired on every turn, released at TTFT
  (or on a return that never produced a first token); caps concurrent prefill.

Think time defers the continuation turn's queue insertion (Clock-scheduled), not
the rate loop. Closed-loop concurrency is the same runtime with the concurrency
bound as the pacing gate.

### User-centric (`UserCentricWorkload`)

Implements virtual-history seeding, per-user cadence, churn/replacement, session
caps, drain, and live user targets.

### Fixed-schedule (`FixedScheduleWorkload`)

Replays a trace with stable absolute ordering, auto or manual zeroing, and
timestamp/delay/immediate precedence. Fixed schedules reject ramps.

### Partitioning across sub-cells

For `workers > 1` and cellular runs, each shape splits so the union across all
threads and cells is a permutation of the whole run (see
[execution-model.md](execution-model.md) and [cellular.md](cellular.md)):

- Request-rate and concurrency partition by request budget
  (`slice_phase_for_thread`, `two_level_partition`, `ModuloCellPartition`); the
  static rate/concurrency/prefill caps are sliced per shard so aggregate offered
  load matches one shard under `runtime.dispatch: sharded`, or drawn from a
  shared per-cell `GlobalAdmission` gate under the `global`/`global-hop`
  dispatch modes (`global` is the default for `workers > 1`; see
  [global-exact-dispatch.md](global-exact-dispatch.md)).
- `user_centric` and `fixed_schedule` partition per conversation
  (`multiturn::…new_with_endpoint` filters the enumerated conversations to the
  shard's owned authored indices).

## Source anchors

- `rust/runtime/src/scheduled.rs` (`ScheduledRuntime`, `Workload`,
  `SingleTurnDatasetWorkload`), `rust/runtime/src/request_rate.rs`
  (`RequestRateWorkload`), `rust/runtime/src/user_centric.rs`,
  `rust/runtime/src/fixed_schedule.rs`, `rust/runtime/src/multiturn.rs`.
- `rust/runtime/src/timing/{slots.rs,stop.rs,intervals.rs}`.
- `rust/runtime/src/engine/sharded_scheduled.rs` (partition math).
