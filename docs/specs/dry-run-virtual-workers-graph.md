<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Dry-run virtual workers for graph execution

## Purpose

Define virtual-worker placement for graph workloads under the analytic
`dry_run` transport. It extends the scheduled virtual-worker design in
[dry-run-virtual-workers.md](dry-run-virtual-workers.md) without changing graph
causality, `SimClock` ownership, or node-level record semantics.

## Built

Virtual-clock graph execution uses `InlineGraphPlacementFactory`, which hosts
all concurrently spawned traces on one `LocalSet` and one `SimClock`. Its
`worker_count` is intentionally ignored. A graph phase prepares a
transport-specific dispatcher through `NativeTransportExecution::build_graph_dispatcher`
for every prepared graph endpoint profile; dry-run constructs a `FakeDispatcher`
and `FakeFabricator` for each such inline backend. Graph worker metadata is
currently assigned by the graph backend, not by the dry-run executor factory.

Consequently, `virtual_workers.enabled` must be rejected for graph datasets
until this design is built. Silently accepting it would emit the ordinary inline
placement behavior, not virtual-worker attribution.

## Future requirements

### Placement boundary

Add a `VirtualGraphPlacementFactory` selected only when all of these hold:

- the transport is dry-run with virtual workers enabled;
- the clock is `SimClock`;
- the dataset is a graph workload.

It remains single-reactor and local: it creates no OS threads and does not reuse
`NativeRunnerGraphPlacementFactory`. Placement alone is insufficient because the
current `TracePlacement` seam assigns an entire trace to one `GraphWorkerBackend`.
The factory must therefore install a node-dispatch routing seam (a graph-sink or
endpoint-runtime wrapper) inside causal execution. That seam selects a virtual
worker immediately before each node's transport dispatch.

```text
Graph admission / causal firing gate
                 |
                 v
     VirtualGraphPlacementFactory
                 |
  node-dispatch routing seam at causal gate
                 |
                 v
worker-local fake dispatcher / fabricator state
                 |
                 v
GraphExecutionEvent + node CapturedRecord
```

### Assignment semantics

Graph placement is per **node dispatch**, not per whole trace. A trace may fan
out and join; assigning one worker to a trace would hide the worker behavior the
test seam exists to exercise.

The assignment key is the graph execution instance, node identifier, and a
monotonic dispatch ordinal. A single causal-gate queue assigns an ordinal and
registers the node's first simulated event in that order before dispatch futures
run. This is required because `SimClock` breaks equal deadlines by sleeper
registration, not by assignment ordinal alone.

Version one supports only `global-hop` semantics:

- `round-robin`: ordinal modulo virtual placement width;
- `sticky`: hash the trace-instance correlation identity, then keep all nodes of
  that trace on the same worker;
- `least-loaded`: select the lowest virtual in-flight count, then record a
  trace-instance binding for later nodes.

`sharded` and ordinary `global` require graph-specific admission semantics and
remain rejected. This explicitly differs from scheduled execution: graph
causality, not a request-rate partition, controls when a node becomes eligible.

### Timing, cancellation, and records

The implementation shares one LocalSet-local `Rc<RefCell<_>>` analytic state:
the run-wide in-flight counter and global jitter ordinal required by the
scheduled virtual-worker design. Every selected worker uses that state through
an exactly-once release guard; constructing independent `FakeFabricator`s would
incorrectly make both values worker-local. Worker-local contention is opt-in; it
changes only the analytic contention input, never graph firing causality.

Graph cancellation may occur before a node dispatches, while its dispatch is in
flight, or after it reaches terminal state. A node that never entered dispatch
does not acquire virtual in-flight state and produces no `CapturedRecord`; graph
phase accounting returns its phase state without synthesizing a request record.
A dispatched node releases exactly one virtual in-flight entry and produces its
normal terminal record. Graph phase accounting remains the owner of graph
admission and completion; virtual placement never returns graph credits.

Each dispatched graph-node record exports `worker_id` and
`worker_assignment_index`. This requires DTO additions plus normal and raw JSONL
projection changes; the projections carry the record's own stored `worker_id` and
fall back to `"rust-0"` only when nothing attributed it, and neither JSONL nor
`RecordIngest` currently contains assignment index. The worker identity
is the selected virtual worker, not the inline reactor. Parquet projection policy
must be decided alongside the record-schema change. No raw-record assertions are
required for the integration contract.

The strict transport DTO alone cannot make graph-aware decisions. A run-level
validation step after transport, dataset kind, clock, and dispatch mode are
resolved rejects unsupported graph virtual-worker combinations.

### Acceptance tests

- A same-time fan-out receives increasing assignment ordinals and deterministic
  round-robin workers.
- A join does not fire before every parent terminal record is observed, even when
  parents use different virtual workers.
- Sticky and least-loaded runs keep all nodes of one trace on their documented
  worker; round-robin demonstrates node movement.
- Cancellation before dispatch creates no record and acquires no virtual state;
  cancellation during dispatch releases one virtual entry and produces the
  expected terminal record.
- A fixed seed produces byte-identical `profile_export.jsonl` projections across
  repeated runs.

## Source anchors

- `rust/runtime/src/engine/graph_execution.rs` — graph placement factories,
  graph events, node record metadata, and inline virtual-clock placement.
- `rust/runtime/src/engine/execute/graph_backend.rs` — graph-phase backend
  assembly.
- `rust/runtime/src/engine/dry_run.rs` — `build_graph_dispatcher` and the fake
  graph dispatcher.
- `rust/runtime/src/engine/graph_phase_runtime.rs` — graph admission,
  cancellation, and phase accounting.
- `rust/runtime/src/clock/sim_clock.rs` — simulated-time event ordering.
