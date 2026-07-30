<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Dry-run virtual workers for sharded scheduling

## Purpose

Define a virtual-clock equivalent of `runtime.dispatch: sharded` for the
analytic dry-run transport. It must model production sharding at the workload
and admission layer, not merely assign an already-issued request to a worker.

## Built

Production `sharded` partitions request budget, concurrency, and request rate
across real worker threads. `run_sharded_scheduled` rejects a virtual coordinator
clock because worker threads require reactor-local `RealClock`s. Virtual-clock
runs therefore collapse to one physical worker today.

The initial virtual-worker design rejects `sharded`; ordinal placement below one
shared scheduler would not reproduce production's per-worker partitions.

## Future requirements

### Virtual sharded scheduler

Add `VirtualShardedScheduledRunner`, a single-reactor implementation that builds
one logical scheduled runner per virtual worker. Each logical runner has its own
workload partition, interval generator, and session/prefill admission state.
All runners share one `SimClock`, one coordinator-side record merger, and one
LocalSet-local analytic state for global dry-run contention and jitter.

```text
authored workload + sharded policy
               |
       deterministic partitioner
               |
    +----------+----------+
    |          |          |
    v          v          v
logical 0  logical 1  logical N
workload   workload   workload
rate/slots rate/slots rate/slots
    \          |          /
     +---------+---------+
               |
               v
       one LocalSet + SimClock
               |
               v
      merged profile_export.jsonl
```

The first release supports only `cells: 1`, request-bounded concurrency and
request-rate scheduled workloads, and multi-turn conversation ownership. It
uses the same partition rules as production for those shapes:

- Request-bounded rate and concurrency workloads partition request budget and
  slot/rate limits deterministically.
- Multi-turn workloads partition conversations, preserving conversation
  ownership for all turns.
- Fixed schedules, graph workloads, agentic replay, user-centric workloads,
  actuator ramps, and multi-cell runs are rejected in the first version. Their
  ownership or aggregate-equivalence semantics require separate designs.

Later multi-cell support must use production's two-level partition identity
`cell_id + cells * thread_id` over `cells * workers`; it must not treat a local
worker index as globally sufficient.

No virtual worker queue is introduced. A logical worker may have multiple
in-flight futures, as a production async worker does.

### Timing and deterministic merge

Each logical runner issues through its own workload state, but a coordinator
arbitrates every virtual issuance before it registers a `SimClock` sleeper. The
coordinator assigns a dense `request_index` and virtual assignment index in
`(target time, worker index, local issuance ordinal)` order. The record merger
uses those preassigned values, never LocalSet polling or completion order.

Continuation turns receive a new target time and local issuance ordinal when
their owning logical runner admits them, then pass through the same coordinator
arbitration. Equal-time `SimClock` ordering is therefore a consequence of the
documented registration order rather than an accidental scheduler detail.

The run-wide seeded jitter ordinal and default contention counter live in one
shared LocalSet-local analytic state, not one `FakeFabricator` per logical
worker. They remain global to preserve dry-run's analytic contract. Worker-local
contention is an explicit test mode only.

### Validation and compatibility

Run-level validation occurs after the transport and workload are resolved but
before virtual-clock execution normalizes physical workers to one. It accepts
`runtime.dispatch: sharded` only after `VirtualShardedScheduledRunner` is
registered. Until then, current behavior must be replaced with a validation
failure rather than silently collapsing workers. The first release rejects every
unsupported workload and option named above rather than falling back to shared
scheduling.

The runner must not use `ThreadPerCoreExecutor`, cross-thread channels, or
`RealClock`. It runs on the coordinator `LocalSet` and is driven by `SimClock`.

### Acceptance tests

Integration assertions use `profile_export.jsonl` for terminal behavior. A
test-only logical-worker event trace or instrumented observer is also required
to assert local admission/rate/slot state and ownership; those facts are not in
the current JSONL schema.

- Four virtual workers partition a request-bounded concurrency workload into the
  production-equivalent request budgets.
- Per-worker request budgets, rate limits, and slot limits equal production's
  partition formulas. Aggregate concurrency may exceed the authored cap when
  production's per-shard minimum-one rule requires it; exact aggregate assertions
  use caps divisible by worker count.
- The logical-worker trace proves multi-turn conversations never move between
  their partition owners.
- Equal-time records merge in the documented preassigned order across repeated
  runs, including continuation turns.
- Post-send request cancellation and phase/external cancellation each release
  the appropriate logical worker, prefill/session state, and scheduler task;
  later work proves that released local capacity is reusable.
- Unsupported graph, fixed-schedule, agentic, user-centric, ramped, and multi-cell
  sharded configurations fail validation.

## Source anchors

- `rust/runtime/src/config/model/dispatch.rs` — production dispatch-mode
  semantics.
- `rust/runtime/src/engine/sharded_scheduled.rs` — production sharded runner and
  its virtual-clock rejection.
- `rust/runtime/src/engine/execute/plan.rs` — selected dispatch mode in the
  native plan.
- `rust/runtime/src/scheduled.rs` and `rust/runtime/src/phase_runtime.rs` —
  workload issuance, admission, and lifecycle contracts.
- `rust/runtime/src/clock/sim_clock.rs` — shared virtual-time ordering.
