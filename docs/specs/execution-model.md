<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Execution model

## Purpose

Define the single thread-per-core execution mechanism, the transport seam a
sink implements, worker-local accumulation, and the shared reduction and
measurement layers. There is exactly one hot path: a request is scheduled,
admitted, dispatched, measured, and captured entirely thread-local. Transports
contribute only wire decode and terminal mapping; parallelism, measurement, and
partitioning live above them.

## Built

### One thread-per-core model

`workers == 1` runs the byte-unchanged single-coordinator path in
`execute_native_inner`: one clock, one `RunCapture`, one co-located `WorkerSink`
on the coordinator's current-thread runtime and `LocalSet`, with no cross-thread
hop.

`workers > 1` runs `sharded_scheduled::run_sharded_scheduled`, which spawns `W`
self-contained sub-cell OS threads. Each thread owns a fresh `current_thread`
runtime and `LocalSet`, builds its own `workers == 1` transport sink on a
reactor-local clock derived from a shared `real_clock_anchor`, runs the unchanged
`run_scheduled_phases` engine over a `1/W` partition, and returns a
`ScheduledShardOutcome`. The `runtime.dispatch`/`--dispatch` selector
(`sharded` | `global` | `global-hop` | `global-push`, `global` default for `workers > 1`)
governs whether that `1/W` partition's concurrency and rate admission stays
purely thread-local (`sharded`) or draws from a shared per-cell
`GlobalAdmission` gate (`global`) or a single coordinator-owned dispatcher
(`global-hop`), or that dispatcher's issuance order carried by identity-only
credits the worker materializes out of band (`global-push`); see
[global-exact-dispatch.md](global-exact-dispatch.md). The
coordinator merges shards (`merge_shards`) and finalizes once. `shardable ==
request.workers > 1`; there is no per-request transport thread hop outside
`global-hop`. `exact_fold` (fold-and-drop memory retention) is an orthogonal
axis, not tied to `shardable`.

Sidecars (server-metrics, GPU, network) and artifacts are once-per-cell on the
main thread; worker phase plans install no live sink, heartbeat, or phase
observer. All shard timestamps sit on one monotonic origin captured once on the
main thread. Global-dense ordinals come from `issuance_authority_for(partition)`
plus shared `phase_ordinal_bases`, so the union of stamped ordinals is a
permutation of `0..total` and `merge_shards` needs only a sort, not a renumber.
See [cellular.md](cellular.md) for the `(cell × thread)` grid and
[scheduling.md](scheduling.md) for how each workload shape partitions.

### The transport seam

A transport implements exactly two traits; everything else is shared:

- `WorkerSink` (`#[async_trait(?Send)]`): `set_run_origin(origin_ns)`,
  `inference_dimensions(turn)`, `supports_response_streaming()`,
  `dispatch_measured(observer, turn, context, on_first_token, responses)`, and an
  optional `prewarm(turn)`. HTTP's `TransportSink` streams
  (`supports_response_streaming() == true`); gRPC's `GrpcTransportSink` does not.
- `ExecutionSinkBuilder` (`Send + Sync + 'static`): carries
  `type Sink: WorkerSink + RequestExecutor`, a `label()`, and
  `build_sink(clock, worker_id)`. The builder moves onto each sub-cell thread and
  constructs the `!Send` sink inside that thread's reactor — the sink never
  crosses a thread, only the builder does.

`turn_execution::build_native` is the shared worker loop, measurement, drain,
cancellation, and streaming relay. It asserts `workers == 1` and fails closed on
`> 1`, so a transport cannot reintroduce its own parallelism. A
`RequestExecutorFactory` (resolved per transport, e.g. `HttpExecutionFactory` /
`GrpcExecutionFactory`) validates prerequisites and hands its builder to
`build_native` with the resolved `ExecutionBackendConfig`. gRPC shares
`build_native` with no parallel path or worker loop. Adding a transport is
writing a builder and registering a factory.

### Transport-neutral dispatch vocabulary

`transport::core` holds the transport-neutral vocabulary with no dependency on
`transport::http` or `transport::grpc`: `PreparedTurn`, `Request`,
`MeasuredContext`, `MeasuredOutcome`, `DispatchResult`, `RequestRecord`,
`Response`, `RequestTrace`, `ErrorDetails`/`ErrorKind`, `ConnectionReuseStrategy`,
the SSE `SseMessage`/`SseField` types, and the `RequestExecutor` and `Dispatcher`
traits. `Dispatcher` extracts `dispatch_collect(PreparedTurn) +
inference_dimensions`; `impl Dispatcher for TransportSink` (HTTP) and
`impl Dispatcher for GrpcTransportSink` are thin, so the graph sink holds
`Rc<dyn Dispatcher>` and dispatches nodes over HTTP or gRPC without matching on
transport kind. `transport::http` and `transport::grpc` depend on
`transport::core`; the reverse never holds.

### Worker-local accumulation

Each worker owns a `NativeMetricsObserver` and accumulates locally with no
per-request cross-thread contention. The measured seam
(`configure_measurement` / `execute_turn_measured` / `drain_records`) is the sole
dispatch path and returns a flat `Vec<(Uuid, RecordIngest)>`; the static-accuracy
adapter uses it too. The global `request_index` is `RunCapture::begin` order;
phase/session/admit fields are patched at finish. `request_index` is the
accumulator ROW, which a fold-and-drop shard keeps dense per store (`0..N_shard`)
so the shard stores concatenate; the run-wide ordinal is the separate
`global_dispatch_index`, assigned by the `IssuanceAuthority` and the field the
per-record artifacts export. Records re-ingest in global
dispatch-index order, so `worker_count` 1-vs-N is byte-identical
(`worker_local_accumulation_parity.rs`).

### Shared reduce and measure

Both the HTTP and gRPC sinks feed their decoded `ServerResponse` iterator through
one `transport::reduce::reduce_parsed_response` (absorb usage, data, and
endpoint-metrics; emit first-token, output/classified token, usage, and terminal
observer events) and share `transport::measure::{WorkerMeasurement,
measure_dispatch}`. A transport therefore contributes only its wire decode plus
its error-enum→terminal map, not the reduction or measurement loop.

## Future requirements

- Consolidate scheduled and graph placement behind one `WorkerPool` and one
  workload-driver interface. The pool must own worker lifecycle, thread-local
  runtime construction, partition placement, and result merge; workload drivers
  must supply scheduling and admission policy over the same worker-local
  `Dispatcher`/`WorkerSink` seam. Transport factories must remain limited to
  constructing worker-local sinks, and the request hot path must remain free of
  per-request cross-thread hops and shared locks.

## Source anchors

- `rust/runtime/src/engine/turn_execution.rs` (`WorkerSink`,
  `ExecutionSinkBuilder`, `RequestExecutorFactory`, `ExecutionBackendConfig`,
  `build_native`).
- `rust/runtime/src/engine/sharded_scheduled.rs` (`run_sharded_scheduled`,
  `merge_shards`), `rust/runtime/src/engine/execute.rs` (`execute_native_inner`,
  `execute_scheduled_shard`, `ShardedShared`, `ScheduledShardOutcome`).
- `rust/runtime/src/engine/grpc_turn_execution.rs` (gRPC shares `build_native`).
- `rust/runtime/src/transport/{core,reduce.rs,measure.rs}`.
- `rust/cli/tests/worker_local_accumulation_parity.rs`,
  `rust/cli/tests/thread_per_core_product.rs`.
- Oracle: `rust/runtime/src/engine/workers_characterization.rs`.
