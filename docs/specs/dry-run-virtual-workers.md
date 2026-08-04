<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Dry-run virtual workers

## Purpose

Extend the socket-free `dry_run` transport with an opt-in, deterministic model
of AIPerf worker placement. The model must exercise multi-worker scheduling,
session affinity, routing, cancellation, and timing assertions under `SimClock`
without creating OS threads, sockets, or a second serving-system simulator.

The feature is a test seam. It models observable AIPerf execution behavior and
writes that behavior to `profile_export.jsonl`; it does not model GPU batching,
KV allocation, network contention, or physical worker queues. Those remain the
responsibility of [offline-cosimulation.md](offline-cosimulation.md).

## Built

The current `dry_run` transport is a single analytic fabrication leaf. Its
`FakeRequestExecutorFactory` validates `workers > 0`, then deliberately ignores
the worker count and creates one coordinator-local `FakeFabricator`. That
fabricator shares the native scheduler, phase runtime, cancellation policy,
observer, capture, metrics, and artifact pipeline with real transports, but it
has no worker-placement state.

It supports analytic linear, Aiconfigurator-polynomial, and recorded latency
models; seeded jitter; virtual or real clock driving; and post-send
cancellation. The scheduler selects `turn.cancel_after_ns`; dry-run uses it to
produce a terminal cancelled outcome with an HTTP-499-compatible record error.

The native entrypoint forces `plan.workers = 1` whenever a selected transport
uses a virtual clock. `ThreadPerCoreExecutor` is the production thread-per-core
backend and constructs worker `RealClock`s on OS threads. It remains the
authority for real-thread behavior; a virtual-worker implementation cannot reuse
it or alter it.

## Future requirements

### Activation and compatibility

Virtual workers are opt-in through a strict dry-run transport configuration:

```yaml
runtime:
  workers: 4
  dispatch: global-hop

transport:
  type: dry_run
  virtual_workers:
    enabled: true
    width: 4 # default: the authored runtime.workers value
```

`enabled: false` is the default and retains the current one-fabricator behavior,
including its treatment of `runtime.workers`. It emits no *virtual* worker
identity: `worker_id` then names the real executing worker (`rust-{n}` over the
`(cell × thread)` grid), which is `rust-0` for a single-worker run and one id per
worker thread otherwise. A virtual run replaces that with its modeled
`dry-run-{n}` placement.

Virtual workers require `clock: sim`. Config resolution must capture explicit
`width` or the authored `runtime.workers` into a new virtual placement-width
field *before* execution caps physical workers to dataset conversations, pins
virtual-clock execution to one reactor, or applies any other physical-worker
adjustment. The physical `plan.workers` remains one; the captured width is the
number of virtual workers. This preserves the current virtual-clock safety
invariant while making authored placement width independent of dataset size.

The Config-v2 `DryRunConfig`, YAML projection, and runtime
`DryRunTransportConfigV2` all carry the same strict nested DTO. Unknown
`virtual_workers` fields fail at the authored config boundary rather than being
accepted and discarded before transport validation. Validation rejects a zero
width, real-clock virtual workers, duplicate or out-of-range profile indices,
non-finite/non-positive multipliers, and `worker_local` contention while virtual
workers are disabled.

### Placement model

`VirtualWorkerDispatcher` is a single-reactor, `LocalSet`-local execution
component. It sits below shared issuance/admission and above the analytic
fabricator; it does not use channels, worker threads, or the production
`ThreadPerCoreExecutor`:

```text
shared scheduler and admission
  concurrency / rate / sessions / cancellation
                   |
                   v
       VirtualWorkerDispatcher
                   |
        +----------+----------+
        |          |          |
        v          v          v
    worker 0    worker 1    worker N
     local        local       local
     state        state       state
        \          |          /
         +---------+---------+
                   |
                   v
       FakeFabricator + shared SimClock
                   |
                   v
        observer / profile_export.jsonl
```

The plan layer must preserve the selected `runtime.dispatch` mode and virtual
placement width while it selects inline virtual-clock execution. Version one
supports `global` and `global-hop`; it rejects `sharded` rather than claiming
parity with production's per-worker workload/admission partition. A later
virtual sharded scheduler must sit above the dispatcher and partition request
budget, rate, and concurrency before placement.

As in the production configuration, `runtime.hop_routing` applies only to
`global-hop`:

| Mode | Requirement |
|---|---|
| `sharded` | Rejected in version one; future support requires virtual workload/admission partitioning, not ordinal placement alone. |
| `global` | Preserve shared global admission before deterministic placement. |
| `global-hop` | Preserve one globally ordered placement sequence. |
| `runtime.hop_routing: round-robin` | Assign issued turns by virtual-worker index in issuance order. |
| `runtime.hop_routing: sticky` | Match production semantics: hash `correlation_id`; a missing correlation id falls back to round-robin. |
| `runtime.hop_routing: least-loaded` | Select the smallest virtual in-flight count, break ties by index, and bind continuations to that initial selection. |

Worker placement and endpoint routing are distinct. A virtual worker executes a
request selected by the existing endpoint policy; it does not select an endpoint
or change global issue order. Phase one exports worker placement only: current
dry-run has neither an endpoint URL nor endpoint identity to export. A later
synthetic-endpoint model must define its own stable endpoint identifier and
failure/routing policy.

### Timing model

All workers share the run's `SimClock`, so their request futures advance
concurrently and deterministically. Equal-deadline observable order is the
single dispatcher assignment order followed by `SimClock` sleeper-registration
order; tests must cover that tie rule.

The compatibility default preserves today's dry-run analytic contention input
and seeded jitter sequence. `VirtualWorkerDispatcher` therefore owns one
run-wide in-flight counter and one global issuance ordinal, incrementing before
latency calculation and decrementing exactly once on completion, configured
cancellation, or dropped dispatch:

```text
TTFT / ITL contention = global in-flight count
```

An opt-in `virtual_workers.contention_scope: worker_local` may instead supply
the selected worker's in-flight count. It retains the global jitter ordinal but
changes the contention input, so it exists only for placement tests and is not
the default.

Optional worker profiles provide deterministic variation without pretending to
be a service queue:

```yaml
virtual_workers:
  profiles:
    - worker: 0
      ttft_multiplier: 1.0
      itl_multiplier: 1.0
    - worker: 1
      ttft_multiplier: 2.0
      itl_multiplier: 1.5
```

Profiles multiply the selected analytic result after the configured latency
model and before its event timeline is emitted. They do not impose an implicit
per-worker concurrency limit or queue.

### Records and cancellation

When virtual workers are enabled, `profile_export.jsonl` must include stable
assignment metadata for every terminal record:

```json
{
  "metadata": {
    "worker_id": "dry-run-1",
    "worker_assignment_index": 17
  }
}
```

`worker_id` remains present when virtual workers are disabled — it then names the
real executing worker rather than a modeled one — and only
`worker_assignment_index` is omitted. `RecordIngest` already has an optional
`worker_id`; the change adds `worker_assignment_index` and updates the JSONL
projection to use the stored worker ID. `"rust-0"` survives only as the projection's
fallback for a record no executing worker attributed. Tests assert only this
per-record export, never the raw-record artifact.

Cancellation stays scheduler-owned. The scheduler's selected
`turn.cancel_after_ns` is delivered to the assigned worker, which emits the
same cancelled terminal state as single-worker dry-run: error code 499,
`RequestCancellationError`, `was_cancelled: true`, and
`cancellation_time_ns`. Terminal completion, configured cancellation, and a
dropped dispatch release virtual in-flight state exactly once. Existing scheduler
completion owns the shared workload admission credit; the dispatcher must not
release that credit.

### Delivery order

1. Add the Config-v2/YAML/runtime strict DTO and a plan-level virtual placement
   width captured before dataset or physical-worker caps. Reject graph workloads
   and `runtime.dispatch: sharded` while their separate virtual execution designs
   are absent.
2. Add the single-reactor scheduled `VirtualWorkerDispatcher`, shared timing
   state, and the `RecordIngest`/JSONL assignment projection; retain
   global-contention timing.
3. Add `runtime.hop_routing` parity, worker profiles, and local-contention mode.
4. Design a virtual sharded scheduler above placement, then admit `sharded`;
   see [dry-run-virtual-workers-sharded.md](dry-run-virtual-workers-sharded.md).
5. Design graph node placement, graph cancellation/drop cleanup, and graph
   record attribution separately before admitting virtual workers for graph runs;
   see [dry-run-virtual-workers-graph.md](dry-run-virtual-workers-graph.md).
6. Add a separate declarative outcome-provider layer for synthetic endpoint
   errors, retries, timeout, failover, and scripted token timelines. That layer
   is independent of worker placement.

### Acceptance tests

All integration assertions read `profile_export.jsonl` only:

- `global` and `global-hop` never exceed the authored global concurrency cap;
  `global-hop` exports one increasing assignment sequence.
- An authored worker width larger than the dataset still creates the requested
  virtual placement width while physical execution remains one virtual-clock
  reactor.
- Under `sticky` and `least-loaded`, every multi-turn session keeps one
  `worker_id`; round-robin has a counterexample proving that turns may move.
- A 100% cancellation run emits terminal 499 records and balances credits. A
  follow-up request must acquire the released virtual placement capacity; a unit
  test separately exercises completion, cancellation, and dropped-dispatch guards
  for exactly-once virtual in-flight cleanup.
- A slower worker profile produces the expected TTFT and ITL increase.
- Repeated runs with the same seed have byte-identical stable record projections.
- Equal-deadline requests retain the documented assignment and sleeper tie order.
- Disabled virtual workers retain today's dry-run artifact projection.
- An enabled virtual-worker graph or sharded run fails validation until the
  corresponding execution design is implemented.

## Source anchors

- `rust/runtime/src/engine/dry_run.rs` — current analytic fabricator and the
  future virtual-worker dispatcher boundary.
- `rust/runtime/src/engine/turn_execution.rs` — production worker-placement
  contract and `ThreadPerCoreExecutor` authority.
- `rust/runtime/src/engine/execute/entrypoints.rs` — virtual-clock physical
  worker pinning and the future virtual placement-width handoff.
- `rust/runtime/src/config/model/transport.rs` and `rust/cli/src/yaml.rs` —
  Config-v2 and YAML dry-run transport projection.
- `rust/runtime/src/engine/records.rs` and `rust/runtime/src/metrics_core/ingest.rs`
  — per-record worker attribution and JSONL projection.
- `rust/runtime/src/scheduled.rs` — issuance, session affinity, cancellation
  selection, and admission-credit lifecycle.
- `rust/runtime/src/clock/` — `SimClock` event ordering.
- `rust/dry-run-tests/tests/` — socket-free integration harness and
  `profile_export.jsonl` assertions.
