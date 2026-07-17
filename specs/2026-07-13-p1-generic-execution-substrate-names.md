<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# P1 — Generic shared names for the execution substrate

**Date:** 2026-07-13
**Status:** built. The scheduled and graph online paths share one generically-named
execution substrate reached through a small dispatch surface; the `Http*`/`Turn*`
path-specific names are gone, the leaf DTOs live in a transport-neutral module,
and the per-request executor collapsed to a single sharded thread-per-core model.
**Scope:** the naming, the dispatch-surface consolidation, the transport module
layout, and the one execution model that consumes the seam.

## 1. Problem this solved

The scheduled worker and the runner graph sink (`RunnerGraphSink`) always shared
the same transport dispatch, the same measured DTOs, and the same
`NativeMetricsObserver` — but under names that hid the convergence (`Http*` on
types gRPC and graph also used; `Turn*` on the leaf request DTO) and behind a
sprawl of ~6 near-duplicate `TransportSink` dispatch methods. The shared substrate
was invisible in the code. P1 gave it one generic vocabulary, trimmed the dispatch
sprawl to a small primitive set, relocated the transport-neutral types out of the
wire modules, and — as the execution model settled — replaced the per-request
cross-thread executor with a single sharded model.

## 2. The naming (final)

### Shared dispatch DTOs (transport-neutral)

| Type | Role |
|---|---|
| `PreparedTurn` | the transport-neutral dispatch unit both paths build and dispatch verbatim |
| `MeasuredContext` | the register-metadata → dispatch → record wiring context |
| `MeasuredOutcome` | the measured terminal outcome |
| `DispatchResult` | `{ outcome, request_payload, record }` |
| `Request` | the leaf dispatch DTO shared by http/grpc/dynosim/dry_run (`dispatch(&Request) -> Outcome`) |
| `PreparedEndpoint` | the prepared per-endpoint binding (was `PreparedHttpEndpoint`) |
| `RequestRecord` / `Response` / `TextResponse` | generic per-request record + parsed response |
| `TraceData` / `TraceExport` / `TraceReference` | transport-neutral per-request trace timing |
| `ErrorDetails` / `ErrorKind`, `ConnectionReuseStrategy` | shared error + reuse vocabulary |
| `SseMessage` / `SseField` / `SseFieldName` | SSE data types |

`metrics_core::RequestTrace` (was `HttpTrace`) is the per-request derived-metrics
trace filled by every transport (http/grpc/dynosim/dry_run), matching Python's
generic `BaseTraceData`/`TraceDataExport`. The raw http-client trace stays
`transport::http::models::TraceData` (genuinely http).

Names that stay path-specific because they *are* path-specific: `HttpRequestDispatcher`
(genuinely http), `GraphSink` / `GraphReply` (per-node DAG splice), `TransportSink`
(the http sink type), `GrpcTransportSink` (the grpc sink type).

### The dispatch surface on `TransportSink`

The ~6 dispatch methods collapsed to a small primitive set, on a clear level
convention — **`dispatch_*`** = transport-level (send bytes + measure):

- `dispatch_measured` — context-wired: register metadata → dispatch → record.
- `dispatch_collect` — the `None`-observer convenience over the streaming primitive.
- `dispatch_collect_streaming(…, Option<&dyn TurnResponseObserver>)` — the primitive
  (`Some` = forward live frames, `None` = terminal-only).

### The transport-dispatch trait

`Dispatcher` is the object-safe transport-dispatch trait extracting the method HTTP
and gRPC already shared. `impl Dispatcher for TransportSink` (http) and
`impl Dispatcher for GrpcTransportSink` (grpc) are thin — both bodies already
existed. The runner graph sink holds `Rc<dyn Dispatcher>` (not a concrete
`Rc<TransportSink>`), so graph nodes dispatch over http *or* grpc through the one
trait, and the placement never matches on a transport kind. `M` (the wire dialect,
`GraphSink<M>` OpenAI vs Anthropic) stays orthogonal to transport — only the
embedded transport handle is `dyn`.

## 3. The one execution model

There is exactly **one** thread-per-core execution mechanism, and it lives ABOVE
the transport:

- **`WorkerSink`** — the worker-facing contract a transport sink implements (the
  Rust analogue of Python `BaseTransport.send_request`): `set_run_origin` /
  `inference_dimensions` / `supports_response_streaming` / `dispatch_measured` /
  `prewarm`. `TransportSink` and `GrpcTransportSink` implement it.
- **`ExecutionSinkBuilder`** — the transport-specific half: one builder per transport
  (`HttpSinkBuilder`, `GrpcSinkBuilder`) constructs the worker-local `!Send` sink
  inside its thread's reactor. A transport contributes only a builder — never its
  own worker loop, measurement, drain, or streaming relay.
- **`build_native<B: ExecutionSinkBuilder>`** — the single entry the http/grpc
  factories (`HttpExecutionFactory` / `GrpcExecutionFactory`) share. It builds one
  **co-located** sink and asserts `workers == 1`; a `workers > 1` request is a
  wiring bug, not a second execution model, so it fails closed.
- **`run_sharded_scheduled`** (`engine::sharded_scheduled`) provides all
  thread-per-core parallelism: for `workers > 1` it tiles that same single-worker
  sink across `W` self-contained sub-cell OS threads, each running the *whole*
  scheduled pipeline (arrival pacing, `SlotPool` admission, dispatch, transport,
  record capture) over a `1/W` partition, then merges the record shards. Each
  sub-cell owns its own `current_thread` runtime + `LocalSet` and a `workers == 1`
  transport, so scheduler and transport are co-located and there is **no**
  cross-thread per-request hop.

The former generic per-request executor — `ThreadPerCoreExecutor` and its
`mpsc`/`oneshot`/`Notify` cross-thread hop, and the earlier duplicate
`ThreadPerCoreHttpExecutionBackend` / `ThreadPerCoreGrpcExecutionBackend` — is
**deleted**. gRPC has no parallel execution path: it shares `build_native` through
`GrpcSinkBuilder`.

Two shared helpers fell out of the sinks and live once, used by both:

- **`transport::reduce`** — decoded-response reduction (`reduce_parsed_response`,
  `absorb_usage` / `absorb_response_data` / `absorb_endpoint_metrics`,
  `assistant_message`). A transport contributes only the wire-decode that produces
  the `ServerResponse` iterator; the fold into observer facts + model/usage state
  is identical regardless of how the bytes arrived.
- **`transport::measure`** — `WorkerMeasurement` (the worker-local
  `NativeMetricsObserver` + drain) and `measure_dispatch` (the
  register/arrival/record-response/failed-terminal envelope). Neither sink
  re-implements measurement.

### Every workload shape shards

`slice_phase_for_thread` and the two-level `ModuloCellPartition::new(c + cells*t,
cells*W)` nesting make sharding cover everything:

- **rate / concurrency** phases shard by request-budget partition
  (`owned_positions`), splitting caps by `1/W` (floored to 1) and rate by `1/W`;
- **`user_centric` / `fixed_schedule`** shard per conversation, reusing the cellular
  `ModuloCellPartition` conversation ownership;
- **static-accuracy** shards its dispatch+capture: each shard captures
  `CapturedResponse`s over `Arc`-shared read-only `ProblemAssociation`s, and the
  disjoint per-shard captures are concatenated (order-independent, `problem_id`-keyed)
  and graded **once on the main thread at finalize** — the `!Send` Python evaluator
  never crosses the spawn boundary.

The `shardable` predicate is simply `workers > 1` — there is no static-accuracy
`workers == 1` clamp. Byte-identical records/tally across worker counts for every
workload shape are pinned by a committed characterization oracle
(`engine::workers_characterization`).

## 4. The two placement seams (still two)

The per-request and per-trace placement seams remain **two** traits at parallel
generic names:

- **`RequestExecutor`** — per request (the sink implements it alongside `WorkerSink`);
  factory `RequestExecutorFactory` / `HttpExecutionFactory` / `GrpcExecutionFactory`.
- **`TracePlacement`** — per trace (graph); factory `TracePlacementFactory`.

Merging them into one `WorkerPool` is the deferred aiperf-v2 structural work
(`2026-07-14-unified-execution-substrate-design.md` Stage 2), not part of this pass.
`LocalGraphTraceExecutionBackend` is intentionally not renamed: it owns the graph
`TraceExecutor` and executes a trace locally; it is not a placement.

Level convention this establishes: **`execute_*`** = placement-level (route to a
worker); **`dispatch_*`** = transport-level (send bytes + measure). `RequestExecutor`
(measured placement) and `RequestSink` (loadgen-core raw transport) are deliberately
distinct levels.

## 5. Transport module layout

The transport code lives under one `crate::transport` parent, with an honest
dependency direction:

- `crate::transport::http` — the hyper client + its `sink` (`TransportSink`).
- `crate::transport::grpc` — the tonic client + its `sink` (`GrpcTransportSink`),
  `#[cfg(feature = "grpc")]`.
- `crate::transport::core` — the transport-neutral dispatch vocabulary (all the
  types in §2's table, the `RequestExecutor` and `Dispatcher` traits). It has **no**
  dependency on `transport::http` or `transport::grpc`, so a future transport takes
  the shared vocabulary without pulling in an existing wire client.

`transport::http` and `transport::grpc` depend on `transport::core`; the reverse
does not hold.

## 6. Failure-policy divergence (untouched)

Graph fail-fast vs scheduled resilient failure policy is untouched by this pass — it
is a separate concern resolved by the shared `OnFailure { Continue, Abort }` enum;
the graph node/run failure traits stay the extension seam.

## 7. Parity & testing

The whole pass is naming + dispatch-surface consolidation + backend de-duplication
with **no** metric or dispatch-event change: the graph byte-exact parity tests, the
scheduled dispatch tests, the sim/online integration tests, and the
`workers_characterization` oracle stay green unmodified — that is the correctness
argument.

## 8. Related

- `2026-07-14-unified-execution-substrate-design.md` — the `Dispatcher` trait +
  pair-layer deletion (Stage 1, built) and the deferred `WorkerPool` merge (Stage 2).
- `2026-07-11-aiperf-runner-only-execution-surface-design.md` — the runner surface
  whose composition this substrate sits under.
