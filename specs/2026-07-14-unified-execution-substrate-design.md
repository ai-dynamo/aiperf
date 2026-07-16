# Unified execution substrate — one `Dispatcher`, one `WorkerPool`, any workload over any transport

**Status:** design — not built. Stage 1 (`Dispatcher` trait + `grpc + graph`) is a small, self-contained increment shippable immediately on top of `ajc/rust-threaded`; Stage 2 (the `WorkerPool`/`Session` merge) is the S5 endgame.

**Relationship / prerequisites.** This is the concrete realization of the "unify scheduled + graph under one substrate" endgame recorded in `2026-07-12-cellular-ready-seams-and-roadmap.md` §S5 (deferred to aiperf-v2, the single unbuilt cellular seam) and audited in `2026-07-13-scheduled-graph-production-convergence.md`. It adopts the target vocabulary of `2026-07-13-greenfield-execution-vocabulary.md` and lands on top of the incremental step already taken by `2026-07-13-p1-generic-execution-substrate-names.md` (`built`). It **assumes the `ajc/rust-threaded` P1–P5 series** (sharded scheduled runtime, injectable `ModuloCellPartition` seams, the P5 `next_arrival_target` pacer) is the base — Stage 2 folds graph into P3's sharded model rather than adding a fourth placement backend. It reuses the already-transport-neutral body plane (`2026-07-13-segment-unification-design.md`, `2026-07-13-endpoint-body-construction-design.md`) and the `OnFailure` enum (`2026-07-13-scheduled-graph-convergence-implementation.md`, `built`). It is the seam a future WebSocket transport (`2026-07-13-websocket-transport-design.md`) and the offline dynosim path slot into for free.

---

## 0. What "everything I always wanted" means

The full product matrix, with no missing cells:

```
              scheduled   graph   (multi-turn / agentic traces)
   http          ✓          ✓
   grpc          ✓          ✗  ← the visible symptom
   ws            —          —   (once WS lands)
   dynosim       ✓          ✓
```

`grpc + graph` fails closed today (`registry.rs` never composes the pair) not because graph scheduling is transport-specific, but because the **one** graph execution backend that exists is welded to the concrete HTTP `TransportSink`. The deeper cost is structural: there are **two thread-per-core placement backends** (per-request vs per-trace) and **two workload drivers** (`RequestRateWorkload`/`ScheduledRuntime` vs `GraphWorkload::execute`), so every new transport or workload variant must be re-crossed by hand. The unification erases the cross-product: **any `Workload` yields `Trace`s; any `Trace` runs as a `Session` on one `WorkerPool`; every `Request` in it dispatches through one `Dispatcher` trait implemented once per transport.** Adding a transport adds one `Dispatcher` impl and lights up every workload; adding a workload adds one `Trace` source and lights up every transport.

## 1. What is already true (do not relitigate)

Grounded in code (`rust/runtime/src/http.rs`, `grpc.rs`, `runner_protocol/graph_execution.rs`, `scheduled.rs`; the P3/P5 commits on `ajc/rust-threaded`):

1. **The dispatch unit is transport-neutral.** `PreparedTurn` (`http.rs:209`) is consumed verbatim by HTTP (`TransportSink::dispatch_collect`, `http.rs:1283`) and gRPC (`GrpcTransportSink::dispatch_collect`, `grpc.rs:271`) — **identical signatures**. Graph already builds a `PreparedTurn` and dispatches it (`graph_execution.rs:830`).
2. **Body materialization is already transport-agnostic.** A graph node's `format_payload → BodyPlan → materialize_standalone()` yields canonical bytes into `HttpRequest.request_body_bytes` (`graph_execution.rs:411`); `GrpcTransportSink` consumes exactly those bytes (`grpc.rs:366`). The registry/endpoint path does not know or care about transport (`2026-07-13-endpoint-body-construction-design.md` §2–3).
3. **Measurement is one type.** `NativeMetricsObserver` is driven with identical calls by HTTP, gRPC, and graph. The cellular measurement seams (`RecordsShard`, `MetricsHeartbeat`/t-digest, `IssuanceAuthority`) are **decoupled from the execution model by design** (§S5 "Freeze now") — a unified runner feeds them unchanged.
4. **Timing/pacing is converging.** P5's `next_arrival_target(prev, start, now, FirstArrival, WhenBehind, draw)` (`timing/arrival.rs`) is a pure arrival function already shared in vocabulary by scheduled/dynosim/graph and wired into graph. Scheduled `(AfterInterval, Reanchor)` vs graph `(AtStart, KeepAbsolute)` are the two named policy axes — this is the `Pacer` core.
5. **The thread-per-core template exists.** P3's `run_sharded_scheduled` (`sharded_scheduled.rs`) runs W self-contained threads, each executing the **whole** scheduled pipeline over a `ModuloCellPartition` 1/W slice with a co-located `workers==1` transport (no per-request channel hop), then merges record shards (`ScheduledShardOutcome::absorb`). Graph's `ThreadPerCoreTracePlacement` (`graph/placement.rs`) is the per-trace analogue P3 was explicitly modeled on.
6. **The failure divergence is already a value.** `OnFailure { Continue, Abort }` (`failure.rs`, `built`) is the shared `Copy` enum that resolves the one genuine scheduled-vs-graph behavioral difference; the graph node/run failure traits stay as the extension seam.
7. **The names are ready.** P1 renamed the two placement seams to parallel generic names (`RequestExecutor` per-request, `TracePlacement` per-trace) and folded `TransportSink`'s dispatch sprawl to `dispatch_measured` / `dispatch_collect[_streaming]`, "to lay the vocabulary groundwork for the eventual structural merge without committing to it now."

The only genuinely transport-coupled code is **the concrete `Rc<TransportSink>` field on the runner graph sink and its two paired call sites** (`graph_execution.rs:243`, `:800`, `:830`). Everything above the bytes is already shared.

## 2. Stage 1 — the `Dispatcher` trait + collapse the pair layer (ships `grpc + graph` now)

Stage 1 does **not** add a `(grpc, graph)` `RunnerPairFactory`. Hand-composing that cell perpetuates the O(transport × workload) matrix the unification exists to delete. Instead Stage 1 (a) introduces the `Dispatcher` trait, (b) teaches the graph placement to build a gRPC dispatcher, and (c) **collapses the per-cell pair factories into descriptor-gated auto-composition** — after which `grpc + graph` (and every other compatible cell) falls out for free. This pulls the spec's Stage-2c registry collapse (§3.4) forward; it is safe to do before the placement merge (§3.2) because the transport axis is already factored out of the pair objects.

### 2.1 The `Dispatcher` trait

Introduce one object-safe transport-dispatch trait — the greenfield `Dispatcher` — extracting the method HTTP and gRPC already share:

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

    fn inference_dimensions(&self, request: &HttpRequest) -> InferenceDimensions;

    fn supports_response_streaming(&self) -> bool { false }
}
```

`impl Dispatcher for TransportSink` and `impl Dispatcher for GrpcTransportSink` are thin — both method bodies already exist. (Object-safety: `dispatch_collect` is `async fn` in trait; box it via `async-trait` as the codebase already does for `GraphSink`, or return `Pin<Box<dyn Future>>`. It crosses a `dyn` boundary, so object-safe per the extensibility discipline. Use `?Send`: each graph worker owns its sink `!Send` on a thread-local `LocalSet`.)

The graph sink then holds `Rc<dyn Dispatcher>` in place of the concrete `Rc<TransportSink>` at three sites — `GraphEndpointDispatch.transport` (`graph_execution.rs:243`), `PreparedRunnerGraphEndpointRuntime.transport` (`:364`), and the two `RunnerGraphSink` call sites (`:800` `inference_dimensions`, `:830` `dispatch_collect`) that now go through the trait. No change to graph scheduling, workload driver, materialization, or metrics. **`M = OpenAiChatMessage` stays** — `GraphSink<M>` is generic over the wire *dialect* (OpenAI vs Anthropic), which is **orthogonal to transport**; only the embedded transport handle becomes `dyn`.

### 2.2 Teach the graph placement to build a gRPC dispatcher

This is the one genuinely transport-specific addition, and it survives the placement merge (§3.2), so it is not throwaway. Today `PreparedRunnerGraphEndpointRuntimeFactory::prepare_worker` (`graph_execution.rs:282`) builds `TransportSink::new_multi_configured` (`:303`). Give the factory a transport kind and a gRPC arm that assembles a `GrpcTransportSink` from the same inputs the scheduled gRPC path uses (`grpc_turn_execution::prepare_grpc_sink`) — clock, base URLs, model, transport config, the `PreparedEndpointTable` the factory already builds, plus a `GrpcBindingRegistry` (`GrpcBindingRegistry::builtin()`), then `sink.prepare_bindings(&table)` (`grpc.rs:174`) for the dense per-endpoint bindings. The materialized body bytes are already gRPC-valid (`grpc.rs:366` consumes them verbatim); only the binding step is new.

### 2.3 Delete the `*Pair` abstraction entirely

A `RunnerPairFactory` reifies a **cell of the transport × workload matrix** as a runtime object — that reification *is* the anti-pattern, and a generic auto-generated pair object keeps the same `pairs: BTreeMap<(String,String), …>` map, `(transport_id, workload_id)` key, and `selection.pair`, so it does not fix it. Remove the abstraction, do not genericize it.

The selection path already resolves both axes independently: `transports.get(id)` and `workloads.get(id)`, each validated, `validate_requirements` checked (`registry.rs:582-600`). It then does a **redundant** `pairs.get((transport_id, workload_id))` lookup (`:604`) whose only jobs are (a) compat validation — already the free predicate `validate_descriptor_compatibility(transport.descriptor, workload.descriptor)` called in `freeze()` (`:513`) — and (b) `prepare`/`validate_run`, which decompose into (workload lowering) + (transport dispatcher construction), each already owned by the workload adapter and the transport factory. The pair is pure glue.

Target shape — two registries, one predicate, inline composition:

1. **Delete** `RunnerPairFactory`, the `pairs` map, `register_pair`, `pair_key`, `selection.pair`, `OnlineHttpPairFactory`, `OnlineGrpcScheduledPair`, and the dynosim pair objects. No `ComposedPair` replaces them.
2. **Move `prepare_with_context` / `validate_run` onto `RunnerWorkloadFactory`** (the `OnlineWorkloadAdapter` scheduled/graph impls *become* the workload factory's prepare). It receives the validated transport and `transport_id`, and resolves the transport's dispatcher/placement factory from `RunnerExecutionFactories` — the workload is otherwise transport-blind. This folds the duplicated inline `OnlineGrpcScheduledPair` (`grpc_execution.rs:56`, which bypasses the adapter today) onto the one scheduled prepare, and exposes the graph placement's HTTP/gRPC arms (§2.2) through `RunnerExecutionFactories` keyed by `transport_id`.
3. **Admit at freeze and at selection with the descriptor predicate** over the full transport × workload cross-product — no cell is registered, hand-listed, or auto-generated. This *is* what `2026-07-11-aiperf-runner-only-execution-surface-design.md` §7 already specifies ("derived from workload requirements and transport descriptors at registry freeze… not a handwritten runtime switch"), realized without a pair object. The coordinator still never `match`es on transport/workload strings — it does map lookups by id plus the predicate — so that spec's invariant is preserved; only its `RunnerPairFactory` *mechanism* is superseded (record via a dated `## Addendum` on that spec).

`grpc + graph` needs nothing hand-added: gRPC + the graph workload are descriptor-compatible (graph requires `features: &[]`, `requires_semantic_responses: false`, `clock_kinds: [Real, Sim]`; gRPC provides `semantic_responses: true`, `clock: Real` — verified `registry.rs:1370-1394`). Genuinely "pair-specific" constraints already live on descriptors (the `control_plane_http` gate is a transport-feature check at `registry.rs:1419`; dynosim's clock is `clock_kinds`), so nothing is lost by deleting the pair.

**Transitional wrinkle (not a pair):** until the placement merge (§3.2), "the transport's dispatcher" is two factory *types* — `RequestExecutorFactory` (scheduled) vs `RunnerGraphPlacementFactory` (graph). So `workload.prepare(…, transport_id)` resolves the right factory from `RunnerExecutionFactories` keyed by `transport_id` **and** the workload's placement kind. That is a map lookup, not a pair object; after Stage 2b it is one `Dispatcher` per transport and the keying vanishes.

**Stage-1 deliverable:** `aiperf profile` with `transport.type: grpc` + a `dag_jsonl`/`weka_trace`/`dynamo_trace` dataset runs, dispatching graph nodes over Tonic — with **no `*Pair` type anywhere in the tree**. The `supported_pairs` inventory (`registry.rs:2030`) is derived from the transport × workload compat cross-product. Proven by an e2e against a gRPC target mirroring `test_graph_cellular.rs`.

## 3. Stage 2 — the `WorkerPool` merge (the S5 endgame)

Stage 1 gives `grpc + graph` but still leaves two placement backends and two workload drivers. Stage 2 collapses them per the greenfield model, so that **scheduled is the degenerate case of graph, not a parallel world.**

### 3.1 The definition/instance model

Adopt the greenfield four-level split (`greenfield-execution-vocabulary.md`):

```
Replay ─▶ Trace ─▶ Session ─▶ Request ─▶ Outcome
(file)   (defn:    (in-flight  (one       (result)
          DAG of    instance    dispatch)
          Requests) of a Trace)
```

- A **flat scheduled turn is a 1-node `Trace`**; multi-turn is a linear `Trace`; agentic is a branching `Trace`. "Everything the runtime drives is a `Trace`" (`unified-graph-runtime-design.md` §4).
- A **`Workload`** yields the `Trace` stream: `Rate` / `Concurrency` / `Users` (synthetic, mostly 1-node traces) or `Replay` (recorded DAGs). This replaces both `RequestRateWorkload` and `GraphWorkload` as the single trait — `Workload::next_trace()`.
- A **`WorkerPool`** of thread-per-core `Worker`s each runs a `Session`. A `Worker` drives its `Session`'s `Trace` fire-on-ready, sending every `Request` through a `Dispatcher` for an `Outcome`. **There is no executor tier** — `RequestExecutor` and `TracePlacement` both disappear into `WorkerPool`/`Worker` (greenfield §"Deleted, not renamed").

### 3.2 The one placement substrate (built on P3, not beside it)

Fold graph's per-trace placement into P3's sharded template. Each `Worker` thread:

- owns a `current_thread` runtime + `LocalSet` + a co-located `Rc<dyn Dispatcher>` (Stage 1), an injected `IssuanceAuthority`, and an observer — the `ShardedShared` pattern (`execute.rs:155`);
- pulls `Trace`s from its partition of the `Workload` and runs each as a `Session` to terminal, firing nodes on-ready (the existing graph `TraceExecutor`/`GraphSink` DAG driver, `graph/executor.rs:58`, is retained as the *Session driver* — a 1-node trace exits after one dispatch, matching flat-scheduled overhead, "flat-graph fast path", aiperf-v2 REQ 8/9);
- pacts arrivals via the `Pacer` (P5 `next_arrival_target`), admits via `SlotPool`, bounds via `StopCondition`;
- merges its record shard on the join (`absorb` / `merge_records_in_global_order`).

The partition seam is the **two-level `ModuloCellPartition::new(c + cells*t, cells*W)`** nesting P3 proved (cell × thread). It already tiles `0..total` and is workload-kind-agnostic: P4's `runtime.workers / cell_count` line is the one place scheduled (reads W) and graph (reads worker_count) already compose uniformly. **Do not reintroduce the per-request channel hop P3 removed; do not add a fourth placement backend.**

### 3.3 Failure and pacing unification

- **Failure:** the `Session` driver consults `OnFailure` (already built). `Abort` reproduces graph's fail-fast (`ensure!(failed == 0)`); `Continue` reproduces scheduled resilience — selected by `cfg.failure_policy`, defaulting per today's historical per-path behavior. No new code.
- **Pacing:** the `Pacer` is `next_arrival_target` with the workload's `(FirstArrival, WhenBehind)` policy. Scheduled keeps `(AfterInterval, Reanchor)` with its closed-loop backpressure peek; replay/graph keeps `(AtStart, KeepAbsolute)`. Both draw through the same live `IntervalGenerator` handle, so ramp/adaptive actuators mutate one object on every path (byte-exact, P5's 9 parity tests generalize).

### 3.4 The registry (collapsed in Stage 1; completed in Stage 2b)

The `*Pair` abstraction is **deleted** in **Stage 1** (§2.3): no `RunnerPairFactory`, no `pairs` map, no `ComposedPair` — just a transport registry, a workload registry, and the descriptor predicate composing them inline. What Stage 2b *completes* is the axis reduction underneath. Until the placement merge, "transport" resolves to **two** factory *types* — `RequestExecutorFactory` (per-request, scheduled) and `RunnerGraphPlacementFactory` (per-trace, graph) — so the workload's `prepare` keys its factory lookup by both `transport_id` and its placement kind. Once Stage 2b merges the two placement backends into one `WorkerPool`, the transport axis becomes a single `Dispatcher` registry:

- a **`Dispatcher` registry** (one entry per transport: `http`, `grpc`, `dynosim_*`, later `ws`), and
- a **`Workload` registry** (one entry per trace source: `rate`/`concurrency`/`users`/`replay`),

with **any `Workload` runnable over any `Dispatcher`** and the workload's `prepare` reduced to a trivial transport-blind join. Compatibility that is genuinely transport-specific (streaming support, token-native bodies) stays expressed as transport `features` vs workload/endpoint `requirements`, admitted at freeze — never a runtime string switch (`runner-only-execution-surface` Invariant 5/12). The universal `Dispatcher` trait is precisely the "aspirational universal transport trait" that §5.1 of that spec deferred; this spec is where it lands.

## 4. Staging (each step keeps the suite green)

1. **Stage 1 — `Dispatcher` trait + graph gRPC dispatcher + delete `*Pair`.** Extract the trait (§2.1), `dyn`-ify the graph sink's three transport sites, add the gRPC arm to the graph endpoint runtime (§2.2), and **delete the `RunnerPairFactory` abstraction** — `pairs` map, `register_pair`, `OnlineHttpPairFactory`, `OnlineGrpcScheduledPair`, dynosim pairs — moving `prepare` onto the workload factory and admitting via the descriptor predicate (§2.3). `grpc + graph` (and `grpc + scheduled` deduplicated) fall out of the transport × workload cross-product — **no cell object exists**. Independent, shippable; net-negative lines.
2. **Stage 2a — one `Trace`/`Session` vocabulary.** Rename graph's `TraceExecutor`/`GraphSink` to the `Session` driver; model a flat scheduled turn as a 1-node `Trace` behind the existing scheduled driver (no behavior change yet).
3. **Stage 2b — one `WorkerPool`.** Replace `sharded_scheduled` + `ThreadPerCoreTracePlacement` with a single thread-per-core `WorkerPool` whose `Worker` runs any `Session`; delete `RequestExecutor`/`TracePlacement` as separate traits. The transport axis reduces to one `Dispatcher` registry and the workload's `prepare` becomes a trivial transport-blind join (§3.4); `grpc + multi-turn` and (later) `ws + anything` fall out.
4. **Stage 2c — one `Workload` trait.** Merge `RequestRateWorkload`/`GraphWorkload` into `Workload::next_trace()` yielding a trace stream partitioned at trace-ordinal granularity (§6, resolved). The registry collapse is already done (Stage 1); this removes the last driver duplication.

Stages 2a–2c are the aiperf-v2 substrate merge; Stage 1 is a v1 increment that does not block on them — and it leaves the tree with *fewer* registry objects than today, not more.

## 5. Non-goals / preserved invariants

- **The `{Clock} × {Dispatcher}` seam is sacred** ("the crown jewel"): virtual/real and http/grpc/mock/sim stay orthogonal; `drive_sim`/`drive_real` dispatch on `is_virtual()` unchanged. A `Worker` is clock-agnostic.
- **No new measurement seam.** `RecordsShard`/`MetricsHeartbeat`/`IssuanceAuthority`/`ColumnStorePartition` are fed unchanged (S5 "Freeze now").
- **Body plane unchanged.** `BodyPlan` + per-wire materializers already produce gRPC-ready bytes; the one-`Full<Bytes>` rule stays HTTP-local. Token-native gRPC bodies (`raw_input_contents`) remain a proven exclusion until a token-native gRPC endpoint exists (`endpoint-body-construction` §10).
- **`M` (wire dialect) stays orthogonal to transport** — not merged into the `Dispatcher` axis.
- **Byte-parity where it exists is preserved**: `workers == 1` and non-shardable shapes stay byte-identical; the two-level partition nesting math is load-bearing and unit-proven.
- Cross-host cell transport, the offline/dynosim cell wiring, and graph weighted-sampling partition remain out of scope (cellular §"Out of scope").

## 6. Resolved design questions (verified in code, `ajc/rust-threaded`)

1. **Session output capture over gRPC — WORKS, no new seam.** Graph feeds channel dependencies (turn N+1 references turn N's generated text) from `collected.outcome.response_text` and `outcome.model_response.{content,reasoning,assistant_message}` returned by `dispatch_collect` (`graph_execution.rs:884-898`) — **not** from the streaming observer. `GrpcTransportSink::dispatch_collect` builds those exact fields by walking `record.responses` through `endpoint.parse_response` / `absorb_response_data` / `build_assistant_turn` and returns the **same `TurnDispatchOutcome { response_text, model_response, … }`** HTTP does (`grpc.rs:154-155, 410-495`). So graph channel deps flow over gRPC with zero extra plumbing; `dispatch_collect_streaming` is **not** required for capture. The Stage-1 `Dispatcher` trait needs only `dispatch_collect`.
2. **Stage 2b partitions at trace-ordinal granularity — reuses the primitive, not the function.** `run_sharded_scheduled` partitions **request positions / the request budget / concurrency caps** (`owned_positions(cell_requests, t, W)`, `sharded_scheduled.rs:144-148`) — a scheduled-specific unit. Graph already has the general form: `PartitionedGraphTraceSource` (`graph/workload.rs:197`) owns global **session ordinals** `cell_id + k·cell_count` via the same `ModuloCellPartition`, and its tests drive **1-node traces** (`one_node_plan`, `workload.rs:943-974`) — i.e. the "scheduled turn = 1-node `Trace`" unit already runs through it. So the unified `WorkerPool` partitions the `Workload`'s **trace-ordinal** stream (general case = graph's `PartitionedGraphTraceSource`; degenerate case = scheduled's position partition, which coincides for 1-node traces). What carries over from `run_sharded_scheduled` verbatim is the **structure** (per-thread whole-pipeline `current_thread`+`LocalSet`, co-located transport, `ScheduledShardOutcome::absorb` shard-merge) and the **two-level nesting math** `(c + cells*t, cells*W)`. Per-thread `rate/W` and `cap.max(1)` slicing stays as an orthogonal `SlotPool` concern, unchanged by trace granularity. **No fourth placement seam; no verbatim reuse of the function.**
3. **gRPC per-node ITL — supported when the endpoint streams.** `transport_grpc/transport.rs` implements `dispatch_streaming` (server-streaming) and `dispatch_bidi_streaming` (`:330-363`), decoding chunks incrementally with per-chunk `perf_ns` and a `first_response_filter(ttft_ns, …)` (`:562-604`); it selects streaming iff `binding.streaming_method()`/`bidi_streaming_method()` exists, else `dispatch_unary`. So graph-over-gRPC gets real per-node TTFT/ITL for streaming-capable KServe/Riva endpoints and a single-response (TTFT = terminal) unary path otherwise — **the same streaming-vs-not split HTTP already has with SSE**, not a non-streaming-only limitation. Graph ITL metrics need no gRPC-specific handling.

---

*Cited code is on `ajc/rust-threaded` (P1–P5 base); `graph_execution.rs`/`grpc.rs`/`http.rs` line numbers are from that branch's `rust/runtime/src/runner_protocol` and `rust/runtime/src`.*

---

## Addendum — 2026-07-15

*This addendum is authoritative where it conflicts with the body above; the body is preserved as the original append-only record.*

Two structural changes landed on top of Stage 1 (the `Dispatcher` trait + pair-layer deletion recorded in the body already shipped). Neither changes the execution model; both change where code lives and how it is composed.

1. **The v2 layer was relocated into `aiperf_runtime::runner_protocol` behind a `runner-protocol` Cargo feature.** The entire protocol / registry / execution-factory / `*_execution`-driver / coordinator / `RunnerApplication` / cellular-controller+cell / control-plane-HTTP / GPU+network+server side-channel surface (~30k lines) lives under `rust/runtime/src/runner_protocol/` (`mod.rs` + siblings), gated by the `runner-protocol` feature on the `aiperf-runtime` crate. The `aiperf-cli` crate is the thin process shell that drives it: `rust/cli/src/execute.rs`/`exec_bin.rs` re-export the relocated composition root and run the `aiperf --execute` process/stdio/signal harness (mimalloc install, one-request read, `RunnerApplication` drive, v2-report write). Only `aiperf-cli` enables `runner-protocol`; `aiperf-mock-server`, `e2e`, and other library consumers pull `aiperf-runtime` with default features and never compile the v2 layer or its dependency surface.

2. **The category registries were unified into ONE `AIPerfRegistry` / `AIPerfExtension` seam (`aiperf_runtime::extensions`).** The single registry now owns endpoints, dataset loaders, samplers, transports, workloads, **exporters, and actuators** — all registered through the one `AIPerfExtension::register(&mut AIPerfRegistry)` seam, each category backed by a shared `TransactionalRegistry<T>`. The stock composition is one ordered `AIPerfRegistry::with_builtin_extensions([...])` list whose only `#[cfg]` is feature-gate lines (the `runner-protocol` HTTP/gRPC extensions and the `runner-protocol`+`dynosim` extension), and `--capabilities` auto-derives its catalog from the registered component set. The `Aiperf*` PascalCase type family was renamed to `AIPerf*` in the same change. The transport×workload pair map and `validate_descriptor_compatibility` / `supported_pairs` predicate were already removed in Stage 1 (see body §Stage 1 and the deleted-pair-layer note); this addendum only folds exporters and actuators into the same single seam.

No behavioral or wire-format change: the same transports, workloads, endpoints, exporters, and actuators are registered and reachable; capabilities output is unchanged except that it derives from the unified registered set. Cited reality: `rust/runtime/src/extensions/mod.rs`, `rust/runtime/src/runner_protocol/mod.rs`, `rust/cli/src/execute.rs`, `rust/runtime/Cargo.toml` (`[features] runner-protocol`).
