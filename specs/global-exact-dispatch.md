<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Global-exact dispatch for `workers>1`

## Purpose

`workers>1` scheduled execution spawns `W` self-contained sub-cell OS threads
(`rust/runtime/src/engine/sharded_scheduled.rs`). Left to a static per-thread
partition alone, concurrency and rate targets are sliced `1/W` up front and
each thread paces and admits requests against its own local share with no
runtime coordination — only approximating a single global concurrency limit
and a single global request rate in expectation, not reproducing Python's
single shared admission gate. This spec describes the `runtime.dispatch`
selector (`sharded` | `global` | `global-hop`) that closes that gap: `global`
(the default for `workers>1`) admits from a shared per-cell gate so aggregate
concurrency and rate are byte-exact against a single global limiter;
`global-hop` additionally reproduces exact global issuance order for cases
where shared admission alone is insufficient.

## Built

- `DispatchMode` (`rust/runtime/src/engine/protocol.rs`) is the `Sharded` |
  `Global` | `GlobalHop` enum, `#[serde(rename_all = "kebab-case")]`, with
  `Global` as `#[default]`. It is configurable through `runtime.dispatch` in
  Config v2 YAML, the `--dispatch` CLI flag (`rust/cli/src/flags.rs`,
  `Flags::dispatch_mode`), and the protocol-v2 wire request. An explicit
  `--dispatch` wins over an authored `runtime.dispatch`
  (`rust/cli/src/yaml.rs`), matching the `--cells`/`runtime.cells`
  precedence. `runtime.dispatch`/`--dispatch` are config-surface fields only;
  there is no separate `--workers` CLI flag (see
  `rust/e2e/tests/global_dispatch_real_clock.rs`).
- `Sharded` is today's static per-thread partition: `owned_positions`,
  `two_level_partition`, `slice_phase_for_thread`
  (`rust/runtime/src/engine/sharded_scheduled.rs`) slice concurrency, rate,
  and request budget `1/W` up front per worker thread, retained as an
  explicit throughput-oriented opt-in where byte-exact parity does not
  matter.
- `Global` (default for `workers>1`) keeps each worker thread's own
  transport, capture, and measurement, but draws concurrency and
  request-rate admission from a shared `GlobalAdmission` gate
  (`rust/runtime/src/engine/execute.rs`) built once per cell, on the main
  thread, before worker threads spawn, from the cell-local (already
  `owned_positions`-sliced, not further thread-sliced) phase budgets:
  - `GlobalSlotPool` (`rust/runtime/src/timing/slots.rs`) is the
    `Send`+`Sync` cross-thread concurrency admission gate — a semaphore with
    a runtime-adjustable limit (debt-tracked decreases, immediate-capacity
    increases) that every worker thread in the cell shares as one
    `Arc<GlobalSlotPool>` per concurrency-capped phase.
  - `GlobalRateGate` (`rust/runtime/src/timing/rate_gate.rs`) is the
    `Send`+`Sync` cross-thread rate-pacing gate: a single atomic
    next-fire-time counter modeling a fixed-interval base grid
    (`claim_offset_ns` hands out `0`, `interval_ns`, `2*interval_ns`, ...
    gaplessly across every calling thread). Each caller still draws its own
    mean-zero jitter offset from its local `IntervalGenerator` and adds it to
    its claimed base slot. This keeps the **aggregate rate** exact but does
    **not** reproduce true Poisson/Gamma arrival-process statistics (the
    resulting inter-arrival times are grid-plus-offset, not a renewal
    process); exact arrival-*pattern* parity is `global-hop`'s job.
  - `GlobalAdmission` is `Some` only under `Global`; `None` under `Sharded`
    (per-thread `1/W` slicing needs no shared gate) and under `GlobalHop`
    (its single coordinator loop enforces the full cap through one local
    `SlotPool`, so no cross-thread gate is needed — see
    `rust/runtime/src/engine/global_hop.rs`).
  - Conversation partitioning for `fixed_schedule`/`user_centric` is
    unaffected by `Global`; only concurrency/rate admission moves to the
    shared gate.
- `GlobalHop` is a single-coordinator hop executor
  (`rust/runtime/src/engine/turn_execution.rs`,
  `rust/runtime/src/engine/global_hop.rs`): one logical dispatcher
  (`ThreadPerCoreExecutor`) owns the full, un-thread-sliced schedule on the
  coordinator thread and hops individual prepared turns to worker OS threads
  over a bounded mpsc command queue, awaiting a oneshot reply. This
  reproduces exact request-to-thread assignment order (turn `i` -> worker
  `i % W`), not just exact aggregate concurrency/rate — the gap `Global`'s
  shared-admission-only fix cannot close because its `W` independent
  scheduling loops still race. `GlobalHop` does not consume
  `GlobalAdmission`; its exactness comes from "one loop, one full-cap local
  `SlotPool`", not a cross-thread gate.
- `--cells` cellular tiling composes unchanged under every dispatch mode:
  `GlobalAdmission` is built from the cell-local phase budgets (already
  narrowed from the global run by `owned_positions(global, cell_id, cells)`
  upstream in the cellular controller), never further sliced across cells.
  Each cell process gets its own independent `GlobalAdmission`; cells remain
  separate processes and never share a gate with each other.
- Verification:
  `rust/runtime/src/engine/workers_characterization.rs` is the oracle
  covering `Sharded`/`Global`/`GlobalHop` phase-shape parity, a
  `Sharded`-vs-`Global` divergence regression test, and SimClock-adjacent
  (RealClock-based) byte-exact determinism tests.
  `rust/e2e/tests/global_dispatch_real_clock.rs` is a real-binary end-to-end
  `RealClock` spot-check proving `Global` mode's aggregate concurrency cap
  against a live `aiperf-mock-server` process across `workers=4` OS-thread
  sub-cells, with deterministic TTFT/ITL and raw per-record assertions per
  the CLAUDE.md generated-token-timing test requirements.

### Boundary: `SimClock` is single-worker and Graph-only

`SimClock` unconditionally forces `workers = 1`
(`execute_prepared_native_plan_uncommitted_with_runtime_factories` in
`rust/runtime/src/engine/execute.rs`): a virtual-time run can only advance the
single reactor its idle-pump drives, while thread-per-core workers each own a
private reactor the pump cannot reach. `SimClock` is selected only by
transports whose `uses_virtual_clock()` binding says so (currently `dry_run`
with `clock: sim`) and is not a configuration `PreparedLinear` scheduled
workloads (concurrency, request-rate, user-centric, fixed-schedule) select.
`runtime.dispatch`'s `Global`/`GlobalHop` cross-thread coordination is
therefore inert for `SimClock` runs: there is exactly one worker thread and no
cross-thread admission to coordinate. This is a permanent architectural
boundary of the clock seam, not a gap in dispatch-mode coverage — "SimClock-
driven multi-worker dispatch" is not a real configuration.

## Source anchors

- `rust/runtime/src/engine/protocol.rs` — `DispatchMode`.
- `rust/runtime/src/timing/slots.rs` — `GlobalSlotPool`.
- `rust/runtime/src/timing/rate_gate.rs` — `GlobalRateGate`.
- `rust/runtime/src/engine/execute.rs` — `GlobalAdmission`,
  `ShardedShared::dispatch_mode`/`global_admission`, the
  `virtual_clock`/`workers = 1` SimClock gate.
- `rust/runtime/src/engine/sharded_scheduled.rs` — `Sharded` static partition
  (`owned_positions`, `two_level_partition`, `slice_phase_for_thread`,
  `run_sharded_scheduled`).
- `rust/runtime/src/engine/turn_execution.rs`,
  `rust/runtime/src/engine/global_hop.rs` — `GlobalHop`'s
  `ThreadPerCoreExecutor`-shaped single-coordinator dispatcher.
- `rust/runtime/src/engine/workers_characterization.rs` — parity oracle.
- `rust/e2e/tests/global_dispatch_real_clock.rs` — real-binary `RealClock`
  aggregate-concurrency e2e spot-check.
- `rust/cli/src/flags.rs`, `rust/cli/src/yaml.rs` — `--dispatch` flag and
  `runtime.dispatch` YAML surface and precedence.
- `rust/runtime/src/multiturn.rs` — conversation enumeration/partitioning for
  `fixed_schedule`/`user_centric`, unaffected by dispatch mode.
