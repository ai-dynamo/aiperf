<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AIPerf-Rust: the unified thread-per-core execution model — one model, "a thread is a sub-cell"

**Date:** 2026-07-16
**Author:** Anthony Casagrande (Tech Lead) + Claude
**Status:** built — documents the final state after the cross-thread `ThreadPerCoreExecutor` per-request hop was deleted and folded into the sharded sub-cell runtime
**Grounding:** end-to-end read of `rust/runtime/src/engine/turn_execution.rs`,
`rust/runtime/src/engine/sharded_scheduled.rs`, `rust/runtime/src/engine/execute.rs`
(`execute_native_inner`, `execute_scheduled_shard`), and
`rust/runtime/src/engine/grpc_turn_execution.rs`.

One sentence: there is now exactly ONE thread-per-core execution mechanism —
`run_sharded_scheduled`, which runs the whole scheduled pipeline independently on
`W` self-contained sub-cell OS threads over a `1/W` partition — and the transport
factory only ever builds a single co-located `WorkerSink`, so no per-request work
ever crosses a thread boundary.

---

## 1. The problem: two coexisting thread-per-core models

Before this change, "thread-per-core" was implemented **twice**, at two different
layers, and the two fought each other:

1. **The transport-owned `ThreadPerCoreExecutor` hop (deleted).** The transport
   factory, when asked for `workers > 1`, spun up a pool of `W` transport-worker
   OS threads and handed each request from the single coordinator scheduler across
   an `mpsc`/`oneshot`/`Notify` hop to a worker thread, which ran the sink and
   handed the measured result back. The scheduler, admission (`SlotPool`), and
   record capture stayed on the coordinator reactor; only the *transport call*
   was parallel. Every request paid a cross-thread queue + wakeup on the hot path,
   and the `!Send` per-trace state had to be split awkwardly across the seam.

2. **The sharded sub-cell model (kept, now the only one).** Separately, design P3
   ("a thread is a sub-cell",
   `sharded_scheduled.rs:4-18`) runs the **entire** scheduled pipeline — arrival
   pacing, `SlotPool` admission, per-request dispatch, transport, and record
   capture — independently on `W` OS threads over a `1/W` partition of the run,
   then merges the per-thread record shards. Each thread owns a fresh
   `current_thread` runtime + `LocalSet`, its own `workers == 1` transport sink,
   its own `RunCapture`, and its own issuer stamping global ordinals.

Model (1) is strictly dominated by model (2): (2) has no per-request hop, keeps the
whole `!Send` stack thread-local (`sharded_scheduled.rs:365-378`), and contends
nothing on the hot path. Keeping both meant the transport had to carry an execution
model it no longer needed, and a run could in principle double-parallelize.

## 2. The final design: ONE model

The resolution is to **delete the transport's parallelism entirely** and make the
transport factory build a single co-located sink, always. Thread-per-core
parallelism moves *above* the transport, into the sharded runtime.

- **`workers == 1`** → the byte-unchanged single-coordinator-reactor path in
  `execute_native_inner` (`execute.rs:2810` — the `!shardable` branch). One clock,
  one `RunCapture`, one co-located sink on the coordinator reactor.

- **`workers > 1`** → `run_sharded_scheduled` (`sharded_scheduled.rs:274`) spawns
  `W` self-contained sub-cell threads (`sharded_scheduled.rs:291-306`), each of
  which builds its own `workers == 1` transport sink
  (`execute.rs:2359-2369`, note `workers: 1`) on its own reactor-local clock
  (`execute.rs:2345`), runs the **unchanged** `run_scheduled_phases` engine
  (`execute.rs:2523`), and returns a `ScheduledShardOutcome`. The coordinator
  merges the shards (`sharded_scheduled.rs:362` → `merge_shards`) and finalizes
  once (`execute.rs:3217-3235`).

The transport factory's `build_native` therefore fails closed on `workers > 1`
(`turn_execution.rs:213-227`): a `workers > 1` request to the factory is a wiring
bug (the sharded runtime always hands its threads `workers == 1`), not a second
execution model. There is no cross-thread transport hop anywhere in the tree.

### The `shardable` predicate: now just `workers > 1`

`execute_native_inner` routes on a single boolean (`execute.rs:2750`):

```rust
let shardable = request.workers > 1;
```

Everything that used to disqualify a run from sharding is gone. Every scheduled
phase shape now partitions (`execute.rs:2741-2749`):

- request-bounded phases (concurrency / poisson / constant / gamma) partition their
  request budget;
- trace-driven `user_centric` / `fixed_schedule` phases partition per conversation;
- **static accuracy** shards too — the `!Send` Python evaluator/grader stays on the
  main thread, but the per-record *capture* is pure `Send` data (a `problem_id`
  lookup pushing a `CapturedResponse`), so each shard owns a capture processor over
  the shared read-only associations (`execute.rs:2457-2460`) and the disjoint
  captures concatenate at the coordinator for one main-thread grade
  (`execute.rs:3223-3225`, `execute.rs:3437`).

The only non-sharded shape left is `workers <= 1` (`execute.rs:2727-2728`). Note
`shardable` is orthogonal to memory retention: `exact_fold` (fold-and-drop) is a
separate axis that no longer reads `shardable` at all (`execute.rs:962-963`,
`execute.rs:1012-1023`) — a `workers > 1` metrics-only run selects exact-fold, and
each shard folds into its own bounded accumulator (`execute.rs:2560-2570`).

### BEFORE / AFTER

```text
BEFORE — two thread-per-core models, one on top of the other
════════════════════════════════════════════════════════════

   ┌──────────────────────── coordinator reactor ────────────────────────┐
   │  scheduler · SlotPool admission · RunCapture · issuer                │
   │                                                                      │
   │   per request:  ──mpsc──▶ ┌─ transport worker 0 ─┐ ──oneshot──▶      │
   │                  ──mpsc──▶ ├─ transport worker 1 ─┤ ──oneshot──▶     │  ◀── MODEL 1
   │                  ──mpsc──▶ └─ transport worker W-1┘ ──oneshot──▶     │      ThreadPerCoreExecutor
   │                            (Notify wakeups, !Send state split)       │      (transport-owned hop)
   └──────────────────────────────────────────────────────────────────────┘
                    ▲  cross-thread hop on the HOT PATH, every request

   ── and SEPARATELY, unused-together, the sub-cell model (MODEL 2) ──


AFTER — ONE model: workers==1 co-located, workers>1 = W self-contained sub-cells
════════════════════════════════════════════════════════════════════════════════

  workers == 1                          workers > 1  (run_sharded_scheduled)
  ────────────                          ────────────────────────────────────
  ┌── coordinator reactor ──┐           ┌──────────── main / cell thread ────────────┐
  │ scheduler               │           │  build ShardedShared · artifacts · sidecars│
  │ SlotPool admission      │           └───────┬───────────┬───────────────┬────────┘
  │ RunCapture              │                   │spawn       │spawn          │spawn
  │ issuer                  │             ┌──────▼─────┐ ┌────▼───────┐ ┌─────▼──────┐
  │ ┌───────────────────┐   │             │ sub-cell 0 │ │ sub-cell 1 │ │sub-cell W-1│
  │ │ WorkerSink        │   │             │ full       │ │ full       │ │ full       │
  │ │ (co-located,      │   │             │ pipeline   │ │ pipeline   │ │ pipeline   │
  │ │  no hop)          │   │             │ over 1/W   │ │ over 1/W   │ │ over 1/W   │
  │ └───────────────────┘   │             └──────┬─────┘ └────┬───────┘ └─────┬──────┘
  └─────────────────────────┘                    │shard       │shard          │shard
                                                 └──────▶ merge_shards ◀───────┘
                                                          then finalize ONCE
```

No arrow crosses a thread boundary per-request in either arm. In `workers > 1`,
the only cross-thread traffic is one `ScheduledShardOutcome` per worker at the very
end, delivered over an unbounded channel (`sharded_scheduled.rs:288-300`).

## 3. Detailed control/data flow of the `workers > 1` sub-cell run

```text
                    execute_native_inner   (execute.rs:2621)
                              │
                    shardable = request.workers > 1        (execute.rs:2750)
                              │  true
                              ▼
   ┌───────────────────── main / cell thread (once-per-cell, D5) ───────────────────┐
   │ create_run_artifacts                                    (execute.rs:3122)       │
   │ (cell_id, cells) = ModuloCellPartition::from_env() | (0,1) (execute.rs:3135)    │
   │ phase_ordinal_bases = env  OR  compute_phase_ordinal_bases (execute.rs:3141-46) │
   │ build ShardedShared { transport_factory, table_factory, dataset,               │
   │        phases (UNSLICED), metrics_config, exact_fold, cell_id, cells,           │
   │        workers, phase_ordinal_bases, real_clock_anchor, start_ns, … }           │
   │                                                          (execute.rs:3159-3200) │
   │ build profiling-phase sidecars (server-metrics / GPU / net) (execute.rs:3204-16)│
   └───────────────────────────────────┬─────────────────────────────────────────────┘
                                        │ run_sharded_scheduled(shared, sidecars, clock)
                                        ▼                       (execute.rs:3217)
   ┌─────────────────────── run_sharded_scheduled (sharded_scheduled.rs:274) ────────┐
   │ for worker_id in 0..W:  std::thread::Builder::spawn(run_worker_thread)          │
   │                                                        (sharded_scheduled.rs:291)│
   │ start each profiling sidecar on the MAIN thread over the run window             │
   │                                                        (sharded_scheduled.rs:317)│
   │ while received < W:  result_rx.recv().await            (sharded_scheduled.rs:328)│
   └───────────┬───────────────────┬───────────────────────────────┬────────────────┘
   spawn       │                   │                               │
   ┌───────────▼──────────┐ ┌──────▼───────────────┐   …   ┌───────▼──────────────┐
   │ run_worker_thread    │ │ run_worker_thread    │       │ run_worker_thread    │
   │ (sharded:371)        │ │  worker 1            │       │  worker W-1          │
   │ fresh current_thread │ │                      │       │                      │
   │ runtime + LocalSet   │ │                      │       │                      │
   │        ▼             │ │                      │       │                      │
   │ execute_scheduled_   │ │                      │       │                      │
   │   shard (execute:2339)│ │                     │       │                      │
   │  • reactor-local clock│ │  each thread is a   │       │  each thread is a    │
   │    from anchor (2345) │ │  self-contained     │       │  self-contained      │
   │  • two_level_partition│ │  SUB-CELL over a    │       │  SUB-CELL over a     │
   │    (2351)             │ │  1/W partition:     │       │  1/W partition       │
   │  • transport_factory  │ │   sampler + issuer  │       │                      │
   │    .build(workers: 1) │ │   share the SAME    │       │                      │
   │    → co-located sink   │ │  partition object   │       │                      │
   │    (2359-2369)        │ │                      │       │                      │
   │  • slice_phase_for_   │ │                      │       │                      │
   │    thread per phase   │ │                      │       │                      │
   │    (2441-2451)        │ │                      │       │                      │
   │  • issuance_authority_│ │                      │       │                      │
   │    for(partition)     │ │                      │       │                      │
   │    + phase_ordinal_   │ │                      │       │                      │
   │    bases (2425-2426)  │ │                      │       │                      │
   │  • run_scheduled_     │ │                      │       │                      │
   │    phases (UNCHANGED) │ │                      │       │                      │
   │    (execute.rs:2523)  │ │                      │       │                      │
   │        ▼              │ │                      │       │                      │
   │ ScheduledShardOutcome │ │ ScheduledShardOutcome│       │ ScheduledShardOutcome│
   │  {records, input_     │ │                      │       │                      │
   │   sessions, accuracy_ │ │                      │       │                      │
   │   captures,           │ │                      │       │                      │
   │   was_cancelled,      │ │                      │       │                      │
   │   has_warmup}         │ │                      │       │                      │
   │  (execute.rs:2293)    │ │                      │       │                      │
   └───────────┬──────────┘ └──────┬───────────────┘       └───────┬──────────────┘
               │ send((id,outcome))│ over unbounded channel        │
               └───────────────────┴───────────────┬───────────────┘
                                                    ▼
   ┌──────────────────── merge_shards (sharded_scheduled.rs:388) ────────────────────┐
   │ base = first delivered shard; for each other: combined.absorb(shard)            │
   │        (execute.rs:2314 ScheduledShardOutcome::absorb → ShardRecords::absorb)    │
   │  · Retained: concatenate record Vecs, then sort_by request_index (global ordinal)│
   │        (sharded_scheduled.rs:410-412)                                            │
   │  · Folded : merge accumulator-to-accumulator (append_store / t-digest) +        │
   │        concatenate errored records            (execute.rs:2263-2287)            │
   │  · input_sessions unioned, then re-sorted by session_id (sharded:414)           │
   └───────────────────────────────────┬─────────────────────────────────────────────┘
                                        ▼
   ┌──────────── back on the main thread (execute.rs:3223-3235, finalize) ───────────┐
   │ accuracy_captures = outcome.accuracy_captures  (graded ONCE, main thread)       │
   │ Retained → ingest merged records into the report accumulator                    │
   │ Folded   → merge shard accumulators into the report accumulator                 │
   │ build NativeReport · write artifacts · ship (all once-per-cell)                 │
   └─────────────────────────────────────────────────────────────────────────────────┘
```

Key invariants the diagram encodes, each verifiable in code:

- **No sidecar/artifact on a worker.** Server-metrics / GPU / network sidecars are
  started, driven, and finished on the main thread only
  (`sharded_scheduled.rs:317-353`); the worker's phase plan installs
  `live_sink: None, heartbeat: None` (`execute.rs:2502-2503`) and a
  `NoopPhaseObserver` (`execute.rs:2522`). This is the D5 "once-per-cell vs
  per-thread" split (`sharded_scheduled.rs:76-83`).
- **One timeline.** Every worker builds its clock from the shared
  `real_clock_anchor` (`execute.rs:2345`, `RealClock::from_anchor`), so all shard
  timestamps sit on one monotonic origin captured once on the main thread
  (`execute.rs:3131`, `ShardedShared.start_ns` at `execute.rs:2221`).
- **Global-dense ordinals despite independent threads.** Each worker's issuer is
  `issuance_authority_for(partition)` plus the shared `phase_ordinal_bases`
  (`execute.rs:2425-2426`); the two-level partition makes the union of every shard's
  stamped ordinals a permutation of `0..total`, so `merge_shards` needs only a sort,
  not a renumber (`sharded_scheduled.rs:382-412`). The partitioning math is the
  subject of the companion spec, `2026-07-16-sub-cell-partitioning.md`.

## 4. gRPC shares `build_native` — no parallel path

gRPC is a transport, not an execution model. It contributes only a `WorkerSink`
(its `GrpcTransportSink`) and the `GrpcSinkBuilder` that constructs one per worker
(`grpc_turn_execution.rs:4-12`). `GrpcExecutionFactory::build` validates
gRPC-specific prerequisites (positive workers, prepared endpoints present) and then
hands a `GrpcSinkBuilder` straight to the shared `build_native`
(`grpc_turn_execution.rs:65-90`) — the *same* function HTTP's `HttpExecutionFactory`
calls (`turn_execution.rs:237-249`). There is explicitly "no gRPC worker loop"
(`grpc_turn_execution.rs:42-43`).

Consequently a `workers > 1` gRPC run travels the identical sharded path: the
`ShardedShared.transport_factory` is the gRPC factory, each sub-cell thread calls it
with `workers: 1` (`execute.rs:2359-2362`), and gets a co-located `GrpcTransportSink`
on its own reactor. The cell issuer and record shipper live *above* the transport,
so gRPC cellular and sharded runs reuse the same executor as HTTP (matching the
CLAUDE.md claim "gRPC runs the SAME cell executor as http"). The only gRPC-specific
difference is at the sink seam: `supports_response_streaming()` returns `false`
(`grpc_turn_execution.rs:144-147`), so the shared worker loop never opens a live
response channel for gRPC.

```text
        HTTP path                              gRPC path
   HttpExecutionFactory                   GrpcExecutionFactory
   (turn_execution.rs:237)                (grpc_turn_execution.rs:65)
          │                                       │  validate: workers>0,
          │                                       │  prepared_endpoints present
          ▼                                       ▼
   HttpSinkBuilder                         GrpcSinkBuilder
   (turn_execution.rs:163)                 (grpc_turn_execution.rs:93)
          │                                       │
          └──────────────┬────────────────────────┘
                         ▼
              build_native<B: ExecutionSinkBuilder>   (turn_execution.rs:213)
              ensure!(workers == 1)  ← fails closed on > 1
                         ▼
              builder.build_sink(clock, 0)
              → Rc<dyn RequestExecutor>  (one co-located sink)
```

## 5. The `WorkerSink` / `ExecutionSinkBuilder` seam a transport implements

A transport implements exactly two traits and nothing else. Everything above —
measurement, drain, cancellation, streaming relay, the worker loop — is written once
in `turn_execution.rs` and shared (`turn_execution.rs:71-80`).

- **`WorkerSink`** (`turn_execution.rs:81-111`), `#[async_trait(?Send)]` — the Rust
  analogue of Python's `BaseTransport.send_request`. Its methods:
  - `set_run_origin(origin_ns)` — anchor the sink's timestamp origin to the run
    origin so TTFT/ITL are not offset by setup duration;
  - `inference_dimensions(turn)` — report coordinator-known dims (no IO);
  - `supports_response_streaming()` — whether it can stream intermediate responses
    (HTTP `true` at `turn_execution.rs:123-125`; gRPC `false`);
  - `dispatch_measured(observer, turn, context, on_first_token, responses)` — drive
    one prepared turn to terminal, recording into the worker-local observer
    (`turn_execution.rs:97-104`);
  - `prewarm(turn)` — optional warm round-trip, default no-op
    (`turn_execution.rs:108-110`).

  `TransportSink` (HTTP) implements it at `turn_execution.rs:113-142`;
  `GrpcTransportSink` at `grpc_turn_execution.rs:134-161`.

- **`ExecutionSinkBuilder`** (`turn_execution.rs:151-160`), `Send + Sync + 'static`
  — the worker-local construction half. It carries an associated
  `type Sink: WorkerSink + RequestExecutor`, a `label()` (`"http"` / `"grpc"`), and
  `build_sink(clock, worker_id) -> Result<Self::Sink>`. The builder is moved onto
  each sub-cell OS thread and constructs the `!Send` sink *inside* that thread's
  reactor (`turn_execution.rs:148-150`) — this is precisely why the sink can be
  `!Send`: it never crosses a thread, only the `Send + Sync` builder does.

The two built-in builders are `HttpSinkBuilder` (`turn_execution.rs:163-199`) and
`GrpcSinkBuilder` (`grpc_turn_execution.rs:93-132`); each is fed into `build_native`
by its factory. The `RequestExecutorFactory` trait (`turn_execution.rs:66-69`) is
the composition seam the registry resolves per transport
(`GrpcExecutionFactory` / `HttpExecutionFactory`), and `ExecutionBackendConfig`
(`turn_execution.rs:35-53`) is the resolved per-run input bundle every factory
receives (workers, clocks, base URLs, model, transport policy, optional
worker-local prepared endpoint factory).

**Adding a transport is writing a builder and registering a factory** — never a
second worker loop, measurement path, drain, or cancellation. The thread-per-core
parallelism it inherits for free comes entirely from the sharded runtime above it
(`turn_execution.rs:200-212`).

## 6. Why this is the correct final shape

- **One hot path.** With the transport hop deleted, there is exactly one place where
  a request is scheduled, admitted, dispatched, measured, and captured — and it is
  always thread-local. Peak parallelism = `W` fully independent pipelines, not one
  pipeline feeding `W` transport threads.
- **Transports stay dumb.** The `build_native` guard (`turn_execution.rs:219-225`)
  makes it structurally impossible for a transport to reintroduce its own
  parallelism: `workers > 1` fails closed with a message pointing at the sharded
  runtime.
- **Uniformity across HTTP, gRPC, cellular, and future sockets.** Because the issuer
  and record shipper sit above the transport, the `(cell × thread)` grid, the
  cellular controller, and any future cross-node placement all compose over the same
  sink seam with no per-transport execution branch.

---

## Appendix — primary source map

| Claim | Source |
|---|---|
| `build_native` builds ONE co-located sink, fails on `workers > 1` | `turn_execution.rs:213-227` |
| `WorkerSink` trait (the only per-transport difference) | `turn_execution.rs:81-111` |
| `ExecutionSinkBuilder` trait + `build_sink` on the worker thread | `turn_execution.rs:151-160` |
| `HttpSinkBuilder` / `HttpExecutionFactory` | `turn_execution.rs:163-249` |
| `shardable = request.workers > 1` | `execute.rs:2750` |
| `!shardable` single-thread arm (workers==1) | `execute.rs:2810-2815` |
| sharded arm spawns sub-cells, finalizes once | `execute.rs:3109-3235` |
| `run_sharded_scheduled` (spawn / recv / merge) | `sharded_scheduled.rs:274-363` |
| `run_worker_thread` — fresh runtime + LocalSet | `sharded_scheduled.rs:371-378` |
| `execute_scheduled_shard` — full pipeline, `workers: 1` sink | `execute.rs:2339-2369` |
| `ShardedShared` (Send+Sync shard inputs) | `execute.rs:2146-2231` |
| `ScheduledShardOutcome` / `ShardRecords` / `absorb` | `execute.rs:2237-2322` |
| `merge_shards` (concat + sort by global ordinal) | `sharded_scheduled.rs:388-417` |
| gRPC shares `build_native`, no worker loop | `grpc_turn_execution.rs:4-12, 65-90` |
| D5 once-per-cell vs per-thread split | `sharded_scheduled.rs:76-83`, `execute.rs:2502-2522` |

---

## Naming vocabulary (the P1 substrate names)

The scheduled worker and the runner graph sink (`RunnerGraphSink`) always shared one
transport dispatch, one set of measured DTOs, and one `NativeMetricsObserver`, but the
old names hid that convergence (`Http*` on types gRPC and graph also used; `Turn*` on
the leaf request DTO). The P1 pass gave the shared substrate one generic vocabulary,
relocated the transport-neutral types into a `transport::core` module with no `http`
dependency, and collapsed the dispatch sprawl to a small primitive set. This appendix
records only those naming decisions; the execution model above is the substrate they
name. Grounded in `transport/core/dispatch.rs`, `transport/core/mod.rs`,
`transport/http/sink.rs`, `transport/grpc/sink.rs`, `endpoints/registry.rs`.

### Old → new names (transport-neutral DTOs)

| Old (path-specific) | New (generic) | Role |
|---|---|---|
| `PreparedHttpTurn` | `PreparedTurn` | the transport-neutral dispatch unit both paths build and dispatch verbatim (`transport/core/dispatch.rs:187`) |
| `MeasuredTurnContext` | `MeasuredContext` | register-metadata → dispatch → record wiring context (`dispatch.rs:137`) |
| `MeasuredTurnOutcome` | `MeasuredOutcome` | the measured terminal outcome (`dispatch.rs:167`) |
| `HttpTurnDispatchResult` | `DispatchResult` | `{ outcome, request_payload, record }` (`dispatch.rs:116`) |
| `TurnRequest` | `Request` | the leaf dispatch DTO shared by http/grpc/dynosim/dry_run (`dispatch.rs:36`, `dispatch(&Request)`) |
| `PreparedHttpEndpoint` | `PreparedEndpoint` | the prepared per-endpoint binding trait (`endpoints/registry.rs:456`) |
| `HttpTrace` | `RequestTrace` | per-request derived-metrics trace filled by every transport (matches Python `BaseTraceData`) |
| — | `RequestRecord` / `Response` / `TextResponse` | generic per-request record + parsed response (`transport/core/record.rs`, `response.rs`) |
| — | `TraceData` / `TraceExport` / `TraceReference` | transport-neutral per-request trace timing (`transport/core/trace.rs`) |
| — | `ErrorDetails` / `ErrorKind`, `ConnectionReuseStrategy` | shared error + reuse vocabulary (`transport/core/{error,reuse}.rs`) |

Names that stay path-specific because they *are* path-specific: `HttpRequestDispatcher`
(genuinely http), `GraphSink` / `GraphReply` (per-node DAG splice), `TransportSink`
(the http sink type), `GrpcTransportSink` (the grpc sink type). The raw http-client
trace stays `transport::http::models::TraceData` (genuinely http).

### The collapsed dispatch surface

The ~6 near-duplicate `TransportSink` dispatch methods collapsed to a small primitive
set on a clear level convention — **`dispatch_*`** = transport-level (send bytes +
measure):

- `dispatch_measured` — context-wired: register metadata → dispatch → record
  (`transport/http/sink.rs:972`, `transport/grpc/sink.rs:222`).
- `dispatch_collect` — the `None`-observer convenience over the streaming primitive
  (`http/sink.rs:944`, `grpc/sink.rs:240`).
- `dispatch_collect_streaming(…, Option<&dyn TurnResponseObserver>)` — the primitive
  (`Some` = forward live frames, `None` = terminal-only) (`http/sink.rs:996`).

### The `Dispatcher` trait

`Dispatcher` (`transport/core/dispatch.rs:383`) is the object-safe transport-dispatch
trait extracting the `dispatch_collect(PreparedTurn) + inference_dimensions` method HTTP
and gRPC already shared. `impl Dispatcher for TransportSink` (http) and
`impl Dispatcher for GrpcTransportSink` (`transport/grpc/sink.rs:481`) are thin — both
bodies already existed. The runner graph sink holds `Rc<dyn Dispatcher>` (not a concrete
`Rc<TransportSink>`), so graph nodes dispatch over http *or* grpc through the one trait
and the placement never matches on a transport kind.

### The `transport::{core, http, grpc}` layout

All transport code lives under one `crate::transport` parent with an honest dependency
direction:

- `transport::core` — the transport-neutral dispatch vocabulary (every type in the table
  above, plus the `RequestExecutor` and `Dispatcher` traits and the SSE
  `SseMessage`/`SseField`/`SseFieldName` types). It has **no** dependency on
  `transport::http` or `transport::grpc`, so a future transport takes the shared
  vocabulary without pulling in an existing wire client.
- `transport::http` — the hyper client + its `sink` (`TransportSink`).
- `transport::grpc` — the tonic client + its `sink` (`GrpcTransportSink`),
  `#[cfg(feature = "grpc")]`.

`transport::http` and `transport::grpc` depend on `transport::core`; the reverse does
not hold.
