<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Unified execution substrate — one `Dispatcher`, one `WorkerPool`, any workload over any transport

**Status:** Stage 1 **built**; Stage 2 (the `WorkerPool`/`Session` merge) **not built** — the
deferred aiperf-v2 endgame. Stage 1 shipped the `Dispatcher` trait, taught the graph placement to
build a gRPC dispatcher (`grpc + graph` now runs), and deleted the transport×workload pair layer.
Stage 2 collapses the two placement backends and two workload drivers into one `WorkerPool` over one
`Workload` trait; it is designed here but not implemented.

**Relationship.** This is the concrete realization of the "unify scheduled + graph under one
substrate" endgame in `2026-07-12-cellular-ready-seams-and-roadmap.md` §S5. It adopts the target
vocabulary of `2026-07-13-greenfield-execution-vocabulary.md` and builds on
`2026-07-13-p1-generic-execution-substrate-names.md` (`built`, the generic naming + the single
sharded execution model). It reuses the transport-neutral body plane
(`2026-07-13-segment-unification-design.md`, `2026-07-13-endpoint-body-construction-design.md`) and
the `OnFailure` enum. It is the seam a future WebSocket transport and the offline dynosim path slot
into for free.

---

## 0. What "everything I always wanted" means

The full product matrix, with no missing cells:

```
              scheduled   graph   (multi-turn / agentic traces)
   http          ✓          ✓
   grpc          ✓          ✓   ← Stage 1 closed the former gap
   ws            —          —   (once WS lands)
   dynosim       ✓          ✓
```

The unification erases the O(transport × workload) cross-product: **any `Workload` yields
`Trace`s; any `Trace` runs as a `Session` on one placement substrate; every `Request` in it
dispatches through one `Dispatcher` trait implemented once per transport.** Adding a transport adds
one `Dispatcher` impl and lights up every workload; adding a workload adds one `Trace` source and
lights up every transport. Stage 1 delivered the transport axis and the registry flatten; Stage 2 is
the remaining placement/workload merge.

## 1. What is already true (the built base)

Grounded in `rust/runtime/src` (`transport/http/sink.rs`, `transport/grpc/sink.rs`,
`engine/graph_execution.rs`, `engine/sharded_scheduled.rs`, `scheduled.rs`):

1. **The dispatch unit is transport-neutral.** `PreparedTurn` is consumed verbatim by HTTP
   (`TransportSink::dispatch_collect`) and gRPC (`GrpcTransportSink::dispatch_collect`) — identical
   signatures. Graph builds a `PreparedTurn` and dispatches it the same way. The leaf `Request` DTO
   lives in the transport-neutral `transport::core` module.
2. **Body materialization is already transport-agnostic.** A graph node's `format_payload →
   BodyPlan → materialize_standalone()` yields canonical bytes into `Request.request_body_bytes`;
   `GrpcTransportSink` consumes exactly those bytes. The registry/endpoint path does not know or
   care about transport.
3. **Measurement is one type.** `NativeMetricsObserver` is driven with identical calls by HTTP,
   gRPC, and graph, through the shared `transport::measure` (`WorkerMeasurement` + `measure_dispatch`)
   and `transport::reduce` (decoded-response reduction) helpers. The cellular measurement seams
   (`RecordsShard`, `MetricsHeartbeat`/t-digest, `IssuanceAuthority`) are decoupled from the
   execution model by design and are fed unchanged.
4. **Pacing is one function.** `next_arrival_target(prev, start, now, FirstArrival, WhenBehind,
   draw)` (`timing/arrival.rs`) is a pure arrival function shared in vocabulary by
   scheduled/dynosim/graph. Scheduled `(AfterInterval, Reanchor)` vs graph `(AtStart,
   KeepAbsolute)` are the two named policy axes.
5. **There is ONE thread-per-core execution model — the sharded sub-cell model.**
   `run_sharded_scheduled` (`sharded_scheduled.rs`) runs `W` self-contained threads, each executing
   the **whole** scheduled pipeline over a `ModuloCellPartition` `1/W` slice with a co-located
   `workers == 1` transport (no per-request channel hop), then merges record shards
   (`ScheduledShardOutcome::absorb`). The former generic per-request executor (the
   `mpsc`/`oneshot`/`Notify` cross-thread hop) is **deleted**. Graph's `ThreadPerCoreTracePlacement`
   (`graph/placement.rs`) is the per-trace analogue this model was aligned with.
6. **The failure divergence is a value, not a bug.** `OnFailure { Continue, Abort }` (`failure.rs`,
   built) is the shared `Copy` enum resolving the one genuine scheduled-vs-graph behavioral
   difference; the graph node/run failure traits stay the extension seam.
7. **The names are generic.** The two placement seams have parallel generic names (`RequestExecutor`
   per-request, `TracePlacement` per-trace); `TransportSink`'s dispatch surface is the small
   `dispatch_measured` / `dispatch_collect[_streaming]` primitive set.

## 2. Stage 1 (built) — the `Dispatcher` trait + the flattened registry

### 2.1 The `Dispatcher` trait

One object-safe transport-dispatch trait extracts the method HTTP and gRPC share:

```rust
/// Transport-level send-and-measure of one prepared request. Implemented once
/// per transport; the placement/workload layers above are transport-blind.
pub trait Dispatcher {
    async fn dispatch_collect(
        &self,
        turn: PreparedTurn,
        observer: &dyn RequestObserver,
        on_first_token: &dyn Fn(i64),
    ) -> Result<DispatchResult>;

    fn inference_dimensions(&self, request: &Request) -> InferenceDimensions;

    fn supports_response_streaming(&self) -> bool { false }
}
```

`impl Dispatcher for TransportSink` and `impl Dispatcher for GrpcTransportSink` are thin — both
bodies already existed. The graph sink holds `Rc<dyn Dispatcher>` (not a concrete
`Rc<TransportSink>`) at its transport sites (`GraphEndpointDispatch.transport`,
`PreparedRunnerGraphEndpointRuntime.transport`, and the two `RunnerGraphSink` call sites). `M`
(`GraphSink<M>`, the OpenAI vs Anthropic wire dialect) stays orthogonal to transport — only the
embedded transport handle is `dyn`. Each graph worker owns its sink `!Send` on a thread-local
`LocalSet`.

### 2.2 The graph placement builds a gRPC dispatcher

`PreparedRunnerGraphEndpointRuntimeFactory::prepare_worker` builds an HTTP or a gRPC dispatcher from
the same inputs. The gRPC arm assembles a `GrpcTransportSink` (clock, base URLs, model, transport
config, the `PreparedEndpointTable` the factory already builds, plus a `GrpcBindingRegistry` and
`sink.prepare_bindings(&table)` for the dense per-endpoint bindings). The materialized body bytes are
already gRPC-valid; only the binding step is transport-specific.

### 2.3 The `*Pair` abstraction is deleted (two registries, no gate)

There is no `RunnerPairFactory`, no `pairs: BTreeMap<(transport_id, workload_id), …>` map, no
`register_pair`, no `validate_descriptor_compatibility` predicate, and no `supported_pairs` catalog
field — all removed from the tree. The runner exposes a transport registry and a workload registry
as **orthogonal axes with no admission gate between them**: every registered workload runs over
every registered transport. `prepare` / `validate_run` live on the workload factory
(`RunnerWorkloadFactory`), which resolves the transport's execution factory from
`RunnerExecutionFactories` keyed by `transport_id` and is otherwise transport-blind. Selection is
map lookups by id — never a `match` on transport/workload strings — so the "coordinator never
string-matches" invariant is preserved with no reified cell. A constraint a transport genuinely
cannot satisfy (e.g. a token-native gRPC body for a non-streaming endpoint) surfaces where it is
exercised, not as a registry-time compatibility rejection.

**Transitional wrinkle (until Stage 2b):** "the transport's execution factory" resolves to two
factory *types* — `RequestExecutorFactory` (scheduled, per-request) and `TracePlacementFactory`
(graph, per-trace) — so the workload's `prepare` keys its factory lookup by `transport_id` **and**
its placement kind. That is a map lookup, not a pair object; after the placement merge (§3.2) it is
one `Dispatcher` per transport and the keying vanishes.

**Stage-1 result:** `aiperf profile` with `transport.type: grpc` + a `dag_jsonl`/`weka_trace`/
`dynamo_trace` dataset runs, dispatching graph nodes over Tonic, with **no `*Pair` type anywhere in
the tree**. Proven by `rust/cli/tests/test_graph_grpc.rs`.

### 2.4 Composition location and the unified registry

Two structural facts about where this composition lives:

1. **The v2 layer lives in `aiperf_runtime::engine` behind the `engine` Cargo feature.** The whole
   protocol / registry / execution-factory / `*_execution`-driver / coordinator / `RunnerApplication`
   / cellular-controller+cell / control-plane-HTTP / side-channel surface lives under
   `rust/runtime/src/engine/`, gated by the `engine` feature. Only `aiperf-cli` enables it (via
   `rust/cli/src/execute.rs`/`exec_bin.rs`, the `aiperf --execute` process shell);
   `aiperf-mock-server`, `e2e`, and other library consumers pull `aiperf-runtime` with default
   features and never compile the v2 layer.
2. **The category registries are ONE `AIPerfRegistry` / `AIPerfExtension` seam
   (`aiperf_runtime::extensions`).** The single registry owns endpoints, dataset loaders, samplers,
   transports, workloads, exporters, and actuators — all registered through the one
   `AIPerfExtension::register(&mut AIPerfRegistry)` seam, each category backed by a shared
   `TransactionalRegistry<T>`. The stock composition is one ordered
   `AIPerfRegistry::with_builtin_extensions([...])` list whose only `#[cfg]` is feature-gate lines,
   and `--capabilities` auto-derives its catalog from the registered set.

## 3. Stage 2 (not built) — the `WorkerPool` merge

Stage 1 gave `grpc + graph` but still leaves two placement backends and two workload drivers. Stage
2 collapses them per the greenfield model, so that **scheduled is the degenerate case of graph, not
a parallel world.**

### 3.1 The definition/instance model

Adopt the greenfield four-level split:

```
Replay ─▶ Trace ─▶ Session ─▶ Request ─▶ Outcome
(file)   (defn:    (in-flight  (one       (result)
          DAG of    instance    dispatch)
          Requests) of a Trace)
```

- A **flat scheduled turn is a 1-node `Trace`**; multi-turn is a linear `Trace`; agentic is a
  branching `Trace`. "Everything the runtime drives is a `Trace`."
- A **`Workload`** yields the `Trace` stream: `Rate` / `Concurrency` / `Users` (synthetic, mostly
  1-node traces) or `Replay` (recorded DAGs), replacing both the scheduled `Workload` driver and the
  graph trace source under one `Workload::next_trace()`.
- A **`WorkerPool`** of thread-per-core `Worker`s each runs a `Session`, sending every `Request`
  through a `Dispatcher` for an `Outcome`. **There is no separate executor tier** — `RequestExecutor`
  and `TracePlacement` both disappear into `WorkerPool`/`Worker`.

### 3.2 The one placement substrate (built on the sharded model)

Fold graph's per-trace placement into the sharded template that already exists. Each `Worker`
thread owns a `current_thread` runtime + `LocalSet` + a co-located `Rc<dyn Dispatcher>` (Stage 1),
an injected `IssuanceAuthority`, and an observer; pulls `Trace`s from its partition of the
`Workload` and runs each as a `Session` to terminal (the existing graph `TraceExecutor`/`GraphSink`
DAG driver retained as the *Session driver* — a 1-node trace exits after one dispatch, matching
flat-scheduled overhead); paces via the `Pacer`; admits via `SlotPool`; bounds via `StopCondition`;
merges its record shard on the join. The partition seam is the two-level
`ModuloCellPartition::new(c + cells*t, cells*W)` nesting already proven. Do **not** reintroduce the
per-request channel hop; do **not** add a fourth placement backend.

### 3.3 Failure and pacing unification

- **Failure:** the `Session` driver consults `OnFailure` (already built). `Abort` reproduces graph
  fail-fast; `Continue` reproduces scheduled resilience — selected by `cfg.failure_policy`. No new
  code.
- **Pacing:** the `Pacer` is `next_arrival_target` with the workload's `(FirstArrival, WhenBehind)`
  policy. Scheduled keeps `(AfterInterval, Reanchor)` with its closed-loop backpressure peek;
  replay/graph keeps `(AtStart, KeepAbsolute)`. Both draw through the same live `IntervalGenerator`
  handle, so ramp/adaptive actuators mutate one object on every path.

### 3.4 The registry after the merge

The pair layer is already deleted (Stage 1). What Stage 2b completes is the axis reduction: once the
two placement backends merge into one `WorkerPool`, the transport axis becomes a single `Dispatcher`
registry (one entry per transport: `http`, `grpc`, `dynosim_*`, later `ws`) and the workload's
`prepare` reduces to a trivial transport-blind join — the `transport_id`+placement-kind keying of
Stage 1's transitional wrinkle vanishes. Compatibility that is genuinely transport-specific
(streaming support, token-native bodies) stays expressed as transport `features` vs
workload/endpoint `requirements`, surfaced at point of use — never a runtime string switch.

## 4. Staging

1. **Stage 1 — built.** `Dispatcher` trait, graph gRPC dispatcher, `*Pair` layer deleted; `grpc +
   graph` falls out of the two-registry cross-product.
2. **Stage 2a — not built.** One `Trace`/`Session` vocabulary: rename graph's
   `TraceExecutor`/`GraphSink` to the `Session` driver; model a flat scheduled turn as a 1-node
   `Trace` behind the existing scheduled driver.
3. **Stage 2b — not built.** One `WorkerPool`: replace `sharded_scheduled` +
   `ThreadPerCoreTracePlacement` with a single thread-per-core `WorkerPool` whose `Worker` runs any
   `Session`; delete `RequestExecutor`/`TracePlacement` as separate traits.
4. **Stage 2c — not built.** One `Workload` trait: merge the scheduled `Workload` driver and the
   graph trace source into `Workload::next_trace()` partitioned at trace-ordinal granularity.

## 5. Non-goals / preserved invariants

- **The `{Clock} × {Dispatcher}` seam is sacred:** virtual/real and http/grpc/mock/sim stay
  orthogonal; `drive_sim`/`drive_real` dispatch on `is_virtual()` unchanged. A `Worker` is
  clock-agnostic.
- **No new measurement seam.** `RecordsShard`/`MetricsHeartbeat`/`IssuanceAuthority`/
  `ColumnStorePartition` are fed unchanged.
- **Body plane unchanged.** `BodyPlan` + per-wire materializers produce gRPC-ready bytes; the
  one-`Full<Bytes>` rule stays HTTP-local. Token-native gRPC bodies (`raw_input_contents`) remain a
  proven exclusion until a token-native gRPC endpoint exists.
- **`M` (wire dialect) stays orthogonal to transport** — not merged into the `Dispatcher` axis.
- **Byte-parity where it exists is preserved:** `workers == 1` and non-shardable shapes stay
  byte-identical; the two-level partition nesting math is load-bearing and unit-proven.
- Cross-host cell transport, the offline/dynosim cell wiring, and graph weighted-sampling partition
  remain out of scope.

## 6. Resolved design questions (verified in code)

1. **Session output capture over gRPC — WORKS, no new seam.** Graph feeds channel dependencies (turn
   N+1 references turn N's generated text) from `collected.outcome.response_text` and
   `outcome.model_response.{content,reasoning,assistant_message}` returned by `dispatch_collect` —
   not from the streaming observer. `GrpcTransportSink::dispatch_collect` builds those exact fields
   through the shared `transport::reduce` path and returns the same
   `TurnDispatchOutcome { response_text, model_response, … }` HTTP does. So graph channel deps flow
   over gRPC with zero extra plumbing; the `Dispatcher` trait needs only `dispatch_collect`.
2. **Stage 2b partitions at trace-ordinal granularity — reuses the primitive, not the function.**
   `run_sharded_scheduled` partitions request positions / budget / concurrency caps (a
   scheduled-specific unit); graph already has the general form — `PartitionedGraphTraceSource` owns
   global session ordinals `cell_id + k·cell_count` via the same `ModuloCellPartition`, and its tests
   drive 1-node traces. So the unified `WorkerPool` partitions the `Workload`'s trace-ordinal stream
   (general = graph's partitioned source; degenerate = scheduled's position partition, which
   coincides for 1-node traces). What carries over verbatim is the *structure* (per-thread
   whole-pipeline `current_thread`+`LocalSet`, co-located transport, `ScheduledShardOutcome::absorb`
   shard-merge) and the two-level nesting math `(c + cells*t, cells*W)`.
3. **gRPC per-node ITL — supported when the endpoint streams.** `transport::grpc` implements
   server-streaming and bidi-streaming dispatch, decoding chunks incrementally with per-chunk
   `perf_ns` and a first-response TTFT filter; it selects streaming iff the binding exposes a
   streaming method, else unary (TTFT = terminal). So graph-over-gRPC gets real per-node TTFT/ITL for
   streaming-capable KServe/Riva endpoints — the same streaming-vs-not split HTTP has with SSE.
