<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# AIPerf Rust Runtime — Architecture

> Scope: the entire Rust implementation under `rust/` (crates `aiperf-cli` — the `aiperf`
> binary, `aiperf-runtime`, `loadgen-core`, `aiperf-mock-server`). This document traces the whole path **from the
> command line to results**, annotating every component along the way, its structure, and its
> configuration surface. Every claim is grounded in `crate/src/file.rs`; specs are design
> intent, code is truth.

---

## 0. One-paragraph summary

AIPerf is a **Rust-executed** load generator and measurement front end for
inference servers. The human-facing `aiperf` binary (crate `aiperf-cli`) is BOTH the native
entry point — it owns Config-v2, sweep/trial planning, and presentation natively — AND the
execution engine. For each run it projects exactly one **protocol-v2 JSON envelope** and pipes
it to a fresh `aiperf --execute` child over stdin (with `AIPERF_NATIVE=0`, the pure-Python
frontend spawns the same `aiperf --execute` child). The `aiperf` binary is the **only Rust
executable on the product path**: in `--execute` mode it validates the envelope against frozen
registries, selects a `{transport, workload}` pair, builds a dataset/graph, drives a phased load test over a single injected
`Clock` and dispatch seam, accumulates fine-grained timing metrics, writes the authoritative
`native-v2.json` report, fans out native exporters, and emits one terminal JSONL line on stdout.
The whole execution substrate is written against two seams — `{clock}` (real vs virtual time)
and `{transport}` (`RequestSink`/`RequestObserver`/`Dispatchable`) — so the identical code runs
in three interchangeable modes: online-real, online-mock, and offline deterministic
co-simulation.

---

## 1. The two seams (the whole architecture)

Everything else is built on two trait seams. If you understand these, the rest is composition.

### 1.1 `{clock}` — `rust/runtime/src/clock/`

One `Clock` trait (`clock/clock.rs:20`) abstracts wall-vs-virtual time so the **same** tokio
executor runs live or as a deterministic discrete-event simulation.

- `now_ns() -> i64`, `sleep(self: Rc<Self>, duration_ns) -> Pin<Box<dyn Future>>`,
  `is_virtual() -> bool` (default `false`).
- **`RealClock`** (`clock/real_clock.rs`) — monotonic `Instant`; on Linux `sleep` arms a
  one-shot `CLOCK_MONOTONIC` **timerfd** (ns resolution) awaited via `AsyncFd`, *not*
  `tokio::time`'s 1 ms timer wheel (which would destroy µs/ns firing gates). `RealClockAnchor`
  is a copyable origin so per-reactor clocks in thread-per-core share one timeline without a
  shared object.
- **`SimClock`** (`clock/sim_clock.rs`) — integer-ns discrete-event clock: `now_ns: Cell`,
  `seq: Cell`, `heap: BinaryHeap<Sleeper>`. `Sleeper{at_ns, seq_no, waker}` with **reversed
  `Ord`** so the earliest `(at_ns, seq_no)` fires first; the monotonic `seq_no` (registration
  order) is the deterministic tie-break. `next_event_time()` / `advance_to(ns)` are deliberately
  **not** on the `Clock` trait (meaningful only for sim; driven through `Rc<SimClock>`).
- **Drivers** dispatch on `is_virtual()`: `drive_real` lets the tokio reactor drive;
  `drive_sim` is an **idle-pump** — poll the `LocalSet` to quiescence (a `FlagWaker` detects
  same-instant wakes) draining all same-timestamp work, then `advance_to(next_event_time())`
  waking heap-ordered sleepers, repeat; panics on deadlock (idle with no clock event). The
  graph engine's `drive_sim_with_source` / `drive_real_with_source` also merge an external
  `SimEventSource` (e.g. the in-process dynosim engine).

### 1.2 `{transport}` — `loadgen-core/src/sink.rs`

The transport-neutral dispatch/measure seam (extracted so it has zero engine/HTTP/KV deps).

- **`Dispatchable: Send + Sync`** — every concrete request implements `uuid()`,
  `input_length()`, `max_output_tokens()`. The crate never names concrete request types.
- **`RequestSink<R: Dispatchable>`** (`#[async_trait(?Send)]`) — `dispatch(req, obs)`. `?Send`
  because the hyper/gRPC sinks are `!Send` (hold `Rc<dyn Clock>`), driven on a `LocalSet`. A
  request that finishes with an *error terminal* still returns `Ok(())` after
  `obs.on_terminal(..)`; `Err` is reserved for a transport failure the caller must surface.
- **`RequestObserver`** — **no `Send`/`Sync` supertraits**, so each thread-per-core worker owns
  a local observer graph in `Rc`/`RefCell` with no per-token mutex. Callback sequence per
  request:
  1. `on_arrival(uuid, arrival_ms, input_length, requested_output_length)`
  2. `on_admit(uuid, admit_ms, reused_input_tokens)`
  3. tokens — `on_token(uuid, at_ms)` **or** `on_classified_token(uuid, at_ms,
     ObservedTokenKind::{Output,Reasoning})` (default falls back to `on_token`) **or**
     `on_output_tokens(uuid, &[f64])` batch (default replays classified Output).
  4. `on_usage(uuid, ObservedUsage)` — terminal authoritative server usage, 13 optional fields;
     default no-op, absent fields stay absent.
  5. `on_endpoint_metrics(uuid, ObservedEndpointMetrics)` — image/video facts; default no-op.
  6. `on_terminal(uuid, ReplayTerminalStatus::{Completed,Rejected,Canceled,Failed})`.
- TTFT is *derived* as the first token callback; sinks emit no separate first-token event.
- **`CollectorObserver`** / **`TraceCollector`** (`observer.rs`, `collector.rs`) — a pure,
  clock-free recorder (callers supply Clock-derived ms) producing a `TraceSimulationReport`;
  used by dynosim and as a byte-exact parity reference.

Real HTTP, native gRPC, mock HTTP, and the in-process co-sim **all** feed this same observer
seam, so scheduling, pacing, admission, adaptive control, and reporting **never branch on the
backend**.

---

## 2. Crate topology

| Crate | Role | Depends on |
|---|---|---|
| `loadgen-core` | The `{transport}` seam + trace collector. Zero engine/HTTP deps. | — |
| `aiperf-runtime` | Library-only runtime: clocks, transports, endpoints, datasets, RNG, timing/scheduling, graph engine, metrics, exporters, adaptive, accuracy, side-channel telemetry, cellular, dynosim. 16 former `aiperf-*` crates are now modules. Owns the single unified `AIPerfRegistry`/`AIPerfExtension` seam (`extensions`) and — behind the `engine` Cargo feature — hosts the v2 layer `engine` (protocol/registry, execution factories and `*_execution` drivers, `RunnerV2Coordinator`/`RunnerApplication`, cellular controller/cell, control-plane HTTP, GPU/network/server side-channels). **No binary.** | `loadgen-core` (+ optional `dynamo-mocker` under `dynosim`) |
| `aiperf-cli` | The ONE product binary `aiperf`: BOTH the native entry point (owns `profile`/`config` natively) AND the execution engine. It re-execs ITSELF (`aiperf --execute`, an internal hidden mode) once per run/cell; the execution child is the strict process/stdio/signal harness that composes the v2 layer `aiperf_runtime::engine` (feature `engine`). Protocol-v2 only. | `aiperf-runtime` (with `engine`), `loadgen-core` |
| `aiperf-mock-server` | Standalone online test/benchmark inference target (OpenAI/Anthropic/TGI/…); launched independently, **not** in the execution-engine dep graph. | `aiperf-runtime` (only for `aiperf_runtime::rng`) |

`rust/runtime/src/lib.rs` declares the module universe: `clock`, `transport_http`, `transport_grpc`,
`endpoints`, `dataset`, `rng`, `timing`, `graph`, `metrics_core`, `adaptive_core`,
`accuracy_core`, `gpu_telemetry`, `network_latency`, `server_metrics`, `content_server`,
`cellular`, `extensions`, plus the composition modules `http`, `grpc`, `run`, `scheduled`,
`scheduler`, `phase_runtime`, `workload`, `request_rate`, `user_centric`, `multiturn`,
`fixed_schedule`, `body_plan`, `failure`, `report`, `metrics`, `export`, `adaptive`, `accuracy`,
`ancillary`, the feature-gated `dynosim` and `aic_runtime`, and the feature-gated
`engine` (the relocated v2 protocol/registry/execution layer, gated on
`engine`; only `aiperf-cli` enables it).

> **Note:** §A.5 and the double-dispatch / `RunnerPairFactory` / `supported_pairs` narrative
> below describe an older shape. Today there is no transport×workload pair layer — any workload
> runs over any transport, with no compatibility gate — and the v2 layer lives in
> `aiperf_runtime::engine` (feature `engine`) with all category registries
> unified under one `AIPerfRegistry`/`AIPerfExtension` seam (endpoints, loaders, samplers,
> transports, workloads, exporters, actuators). Ground registry claims in
> `rust/runtime/src/extensions/mod.rs` and `rust/runtime/src/engine/`.

---

## 3. End-to-end flow: command line → results

```
┌─ `aiperf profile --config x.yaml`  (entry point, crate aiperf-cli) ────────────────────┐
│  Config-v2 → BenchmarkRun → ONE protocol-v2 JSON envelope                             │
└───────────────────────────────┬───────────────────────────────────────────────────────┘
                                 │  stdin (JSONL)  →  self-exec `aiperf --execute`
                                 ▼
┌─ aiperf --execute  (execution engine, crate aiperf-cli) ──────────────────────────────┐
│  • mimalloc global allocator; stderr tracing (AIPERF_RUNNER_LOG); stdout = protocol    │
│  • arg dispatch: `--execute` | `--cell` | `--aggregator` | controller (validate too)   │
│  • compose_stock_application() → RunnerApplication::stock(distribution_id)              │
│  • run_v2(): parse EnvelopeBootstrapV2 → RunnerEnvelopeV2 (strict, deny_unknown_fields) │
│    catch_unwind around handle_v2 → exactly ONE terminal/validation JSONL + exit code   │
└───────────────────────────────┬───────────────────────────────────────────────────────┘
                                 ▼
┌─ RunnerV2Coordinator::handle(envelope)  (coordinator.rs) ──────────────────────────────┐
│  1. validate_outer()            protocol==2, benchmark_id/artifact_dir/datasets present │
│  2. into_authored()             lower cfg → AuthoredRunSpecV2; DECIDE scheduled vs graph│
│                                 (dataset format dag_jsonl|weka_trace|dynamo_trace→graph)│
│  3. validate_selection_for_run  workload requirements → resource presence → transport   │
│                                 clock/feature/semantic compat → pair validate_pair       │
│  4. validate_endpoint_profiles  strict per-profile decode; build ClientConfig/readiness │
│  5. sidecar_inputs.prepare      strict side-effect-free decode of 5 sidecars            │
│  6. RunnerRunContext::new        frozen per-run context (product registry, factories,…) │
│  7. pair validate_run                                                                   │
│  8. report_provenance                                                                   │
│  ── if operation == Validate ──►  RunValidationV2{success, completeness:Static}, exit 0 │
│  9. prepare_with_context → Box<dyn PreparedRunnerOperation>   (pair-owned preparation)  │
│ 10. operation.execute()  ───────────────────────────────────────────────────────────┐  │
└──────────────────────────────────────────────────────────────────────────────────────┼──┘
                                                                                        ▼
┌─ execute.rs / *_execution.rs  (the native run) ───────────────────────────────────────┐
│  new_current_thread runtime + LocalSet; block_on(prepare_and_execute_native)           │
│    • prepare static-accuracy evaluator (if any) + sidecar resources                    │
│    • build transport on the run Clock; readiness.wait() BEFORE artifact dir is created  │
│    • branch: NativeDatasetPlan::Graph → run_graph_phases                                │
│              else                     → run_scheduled_phases                            │
│  Scheduled: ClockPhaseOrchestrator drives warmup→profiling phases; per phase a          │
│    ScheduledRuntime paces arrivals → SlotPool admission → TurnDispatcher (HTTP/gRPC/    │
│    dynosim sink) → NativeMetricsObserver ingests callbacks → MetricsAccumulator         │
│  Graph:     DAG placement across worker threads (thread-per-core) → GraphSink dispatch  │
│  Side-channels (gpu/network/server-metrics) run as ScheduledPhaseSidecars at barriers   │
│  → PreparedRunOutcome { native_report, report_facts, provenance, report_commit }        │
└───────────────────────────────┬───────────────────────────────────────────────────────┘
                                 ▼
┌─ persist + export  (coordinator.rs persist_prepared_report → report.rs / export/) ─────┐
│  • refuse if native-v2.json exists → finalize_and_write_native_report_json (atomic)     │
│  • run_exporters(finalized, dir, export)  best-effort native exporter plane             │
│  • report_commit.commit() exactly once                                                  │
│  → RunTerminalV2 { success:true, report_path, provenance{transport,workload,…} } exit 0 │
└───────────────────────────────┬───────────────────────────────────────────────────────┘
                                 ▼
        The entry point (or the Python frontend when AIPERF_NATIVE=0) reads the terminal
        line + opens native-v2.json + artifact files
```

Every failure funnels to a **typed** `RunTerminalV2`/`RunValidationV2` with a
`RunnerFailureStageV2 ∈ {Protocol, Validation, Preparation, Execution, Reporting}`, redacted
diagnostic, and exit code (0 ok / 1 validated failure / 2 protocol failure). A caught panic
becomes a typed failure, never a bare crash — the entry point always sees one JSONL line.

---

## 4. Component reference

### A. Process boundary & protocol v2 (`aiperf --execute`)

#### A.1 `main.rs` — stdio entry point
- Installs **mimalloc** as the global allocator (top profiled hotspot was per-request alloc
  churn); a Linux `.init_array` hook tunes `arena_eager_commit`/`purge_delay` before any Rust
  heap allocation.
- Tracing subscriber writes only to **stderr** (stdout is the protocol channel); default filter
  `warn`, overridable with `AIPERF_RUNNER_LOG`.
- Argument dispatch: `--execute` runs one run/probe/cell; `--cell` runs one cell of a multi-cell
  run; `--aggregator` runs the velo-gated aggregator; a plain `execute` envelope with
  `runtime.cells > 1` promotes this process to the **cellular controller**; otherwise the normal
  single-process path. (Capabilities is not an argv mode — it is the in-process
  `aiperf_cli::execute_mode::capabilities_catalog` function.)
- `configure_dynosim_process_defaults` pins OpenBLAS/OMP/MKL/Rayon thread counts for the
  deterministic dynosim event loop.
- `run_v2` parses a two-stage `EnvelopeBootstrapV2` (version + operation + raw run) then the
  full strict `RunnerEnvelopeV2`, wraps `handle_v2` in `catch_unwind`, and writes exactly one
  JSONL line via `write_json_line` (which also sets the process exit code).

#### A.2 `protocol_v2.rs` — the wire contract
- `RUNNER_PROTOCOL_V2 = 2`. `RunnerEnvelopeV2 {protocol_version, operation:
  RunnerOperationV2::{Validate,Execute}, run: BenchmarkRunWireV2}` — `#[serde(deny_unknown_fields)]`.
- `BenchmarkRunWireV2` — `benchmark_id`, `artifact_dir`, `cfg: BenchmarkConfigWireV2`, plus
  retained-but-uninterpreted `resolved`, `sweep_id`, `variation`, `trial`, `label`,
  `cli_command`, `random_seed`, `variables`.
- `BenchmarkConfigWireV2` — deliberately **not** `deny_unknown_fields` (the entry point dumps the whole
  BenchmarkConfig): `models`, `endpoint`, `endpoint_profiles`, `datasets`, `phases`,
  `tokenizer`, `transport`, `runtime`, `artifacts`, `metrics`, `failure_policy`, `slos`,
  `goodput`, `gpu_telemetry`, `server_metrics`, `network_latency`, `content_server`, `sidecars`,
  `export`.
- `into_authored()` — the lowering adapter. **This is where scheduled-vs-graph is decided**: if
  the single dataset's `format`/`type` is `dag_jsonl|weka_trace|dynamo_trace` → workload id
  `"graph"`, else `"scheduled"`. Transport id from `cfg.transport.type`; workers from
  `cfg.runtime.workers`; `failure_policy` folded into the workload config. Produces
  `AuthoredRunSpecV2`.
- `RunnerComponentId` — open but wire-safe grammar `[a-z][a-z0-9_]*`: keeps transport / workload
  / endpoint identities open (registry keys) while the outer contract stays strict.
- Factory-owned config stays `Box<RawValue>` until the selected factory decodes it
  (`NamedRunnerComponentSpecV2`, `EndpointProfilesSpecV2`, `SidecarSpecV2`).
- Responses: `RunValidationV2 {completeness: Static|Complete, deferred_checks, errors}` and
  `RunTerminalV2 {success, report_path, stage, errors, diagnostic_artifacts, provenance}`.
  `RunnerDiagnosticV2 {code, message(redacted), path}`.

#### A.3 `application.rs` / `coordinator.rs` — frozen composition & the `handle()` pipeline
- `RunnerApplication::stock(distribution_id)` composes the built-in universe once:
  `BuiltinRunnerRegistryFactory`, `BuiltinAIPerfRegistryFactory`, `native_execution_factories()`,
  and the three built-in input-adapter resolvers (graph/dataset/sidecar).
- `RunnerV2Coordinator` holds the frozen runner registry, `Arc<AIPerfRegistry>` product
  registry, `RunnerExecutionFactories`, and the input resolvers. `handle()` runs the 10-step
  pipeline in §3, returning `RunnerProcessResultV2 {response, exit_code}`.
- `persist_prepared_report` — refuses an existing `native-v2.json`, writes atomically via
  `finalize_and_write_native_report_json`, runs `aiperf_runtime::export::run_exporters` (best-effort),
  and invokes the one-shot `PreparedReportCommit` exactly once. `AIPERF_EXPORT_SUBDIR` redirects
  native sink outputs for byte-diff parity proofs.

#### A.4 `distribution_identity.rs` / `redaction.rs`
- `current_distribution_id()` = `blake3:` + 64 hex of `DOMAIN || executable_bytes` (reads
  `/proc/self/exe` on Linux — immune to post-launch path swap). Stamped into report provenance;
  the coordinator validates the `blake3:` + 64-lowercase-hex shape.
- `redact_diagnostic()` strips URL userinfo, Authorization/bearer/basic, API-key headers, and
  structured secret assignments before any diagnostic reaches stdout; idempotent.

#### A.5 `registry.rs` — the frozen transport/workload/pair registry (open double-dispatch)
- Extension seams: `RunnerTransportFactory`, `RunnerWorkloadFactory`, and the central
  **`RunnerPairFactory`** (`transport_id()`, `workload_id()`, `validate_pair`, `validate_run`,
  `prepare`, `prepare_with_context`). Type-erased `ValidatedTransportConfig` /
  `ValidatedWorkloadConfig` let a pair downcast only its own startup value.
  `PreparedRunnerOperation::execute()` has **no `Send` bound** (owns `Rc`/`RefCell`).
- Descriptors carry capability facts: `RunnerTransportDescriptor {id, clock:
  RunnerClockKind::{Real,Sim}, semantic_responses, features}`; `RunnerWorkloadDescriptor
  {clock_kinds, requires_semantic_responses, required_transport_features}`. `freeze()` verifies
  every pair references registered components and that clock/feature/semantic requirements hold.
- Built-in transports: `http` (Real, semantic, features `control_plane_http, h1, h2c, http, tls,
  uds`), `grpc` (features `grpc, h2, tls`). Built-in workloads: `scheduled` (Real+Sim), `graph`
  (Real+Sim). Verified supported pairs in the **base** build: `(grpc, scheduled)`, `(http,
  graph)`, `(http, scheduled)`; with `dynosim`: add `dynosim_offline|dynosim_online` ×
  `scheduled|graph`.
- `validate_endpoint_profiles_v2` strictly decodes each profile through the product
  `EndpointRegistry`, compiles per-model readiness, and builds the HTTP `ClientConfig`
  (http2-prior-knowledge, ssl_verify, keepalive/timeout ns, per-origin connection limit).
  `RunnerRunContext` is the cheap-cloneable immutable per-run context shared by validation and
  preparation, and the sole bridge into `ReportRunProvenance`.

#### A.6 `execution_factories.rs` — replaceable placement, not a mode enum
`RunnerExecutionFactories` bundles six object-safe factories as *values*: `http`/`grpc`
`RequestExecutorFactory`, `graph` `RunnerGraphPlacementFactory`, readiness plan + readiness
transport factories, and `control_plane_http`. `native_execution_factories()` installs the stock
in-process implementations; a custom distribution can swap any of them (e.g. a remote/RPC
executor) without touching the coordinator.

### B. Input resolution (`aiperf_runtime::engine`)

Three resolver traits injected into the coordinator; all do the **sole** strict decode of their
input and are side-effect-free at validation time.

- `RunnerGraphInputAdapterResolver` (`graph_input.rs`) — identity-only format lookup then one
  strict decode compiling directly to a `GraphInputBundle`. Adapters: `dag_jsonl`, `weka_trace`,
  `dynamo_trace` (path-only), `aiperf_trace`. Bypasses the linear loader registry entirely.
- `RunnerDatasetInputAdapterResolver` (`dataset_input.rs`) — linear datasets: `synthetic`,
  `file`, `public`. Owns the Config-v2 dataset DTOs (`DistributionSpec::{Fixed, Normal,
  LogNormal, Multimodal, Empirical}`, synthetic media specs). Returns `PreparedDatasetInput`.
  `RunnerDatasetInputContext` carries the two backend-selected policies:
  `MaterializedTracePromptStorage` (online) vs `HashIdentityTracePromptStorage` (dynosim).
- `RunnerSidecarInputAdapterResolver` (`sidecar_input.rs`) — five sidecars (`content_server`,
  `gpu_telemetry`, `live_streaming`, `network_latency`, `server_metrics`) strict-decoded into a
  type-erased `PreparedSidecarInputs` store; runtime resource preparation (sockets, subprocess)
  is a separate later seam.

### C. Execution paths (`aiperf_runtime::engine`)

`execute.rs` (the ~4.5k-line native driver) lowers a protocol-neutral `NativeRunSpec`
(`NativeEndpointPlan`, `NativeDatasetPlan::{PreparedLinear, StaticAccuracy, Graph}`,
`NativeSidecarPlan`) onto a `current_thread` runtime + `LocalSet` and runs
`prepare_and_execute_native`: prepare accuracy + sidecar resources → **readiness wait before
artifact dir creation** → branch graph vs scheduled.

- **`online_execution.rs`** — HTTP pairs (`scheduled`, `graph`, `static_accuracy`). The
  `OnlineWorkloadAdapter` seam has three impls; `prepare_with_context` validates, builds
  readiness, and lowers to a `NativeRunSpec` executed on the **real clock** with the HTTP
  `TransportSink` and `NativeMetricsObserver`. Owns the tokenizer-acquisition seam
  (`OnlineTokenizerSourceResolver`: native HF-commit fetch with blake3 cache, or `hf-hub`;
  `trust_remote_code=true` always rejected; env `AIPERF_CACHE_DIR`, `HF_ENDPOINT`, `HF_TOKEN`,
  `HF_HUB_OFFLINE`). `lower_scheduled` is shared with the gRPC pair.
- **`grpc_execution.rs` / `grpc_turn_execution.rs`** — the `(grpc, scheduled)` pair; requires
  `grpc://`/`grpcs://` URLs, rejects all sidecars, reuses `lower_scheduled` and the same
  `RequestExecutorFactory`/`HttpExecutionBackendConfig` seam as HTTP.
- **`turn_execution.rs`** — HTTP turn *placement* below one logical `TurnDispatcher`:
  `NativeRequestExecutorFactory` picks a single `TransportSink` on the coordinator reactor
  (`workers==1`) or a `ThreadPerCoreRequestExecutor` (one `current_thread` runtime + sink per OS
  thread, bounded mpsc command channels, `Configure`/`Command`/`Prewarm`/`Drain` control
  protocol, per-worker record concatenation at drain).
- **`graph_execution.rs` / `graph_phase_runtime.rs`** — worker-local whole-trace HTTP execution
  behind the graph placement seam, and the **backend-neutral** graph phase orchestration
  (`run_graph_phases`) shared by online HTTP and offline dynosim. Workers emit
  `RunnerGraphExecutionEvent::{FirstToken, Record, TraceComplete}` into `GraphPhaseProgress`.
- **`offline_execution.rs`** (`#[cfg(feature="dynosim")]`) — registers `dynosim_offline`
  (Sim clock) and `dynosim_online` (Real clock) transports × scheduled/graph. `DynosimExecutor`
  selects online/offline library entry points and verifies parity (byte-exact offline; relaxed
  online). Requires `worker_count == 1`. `DynamoBuildFeature` bridges to the `dynamo-*` Cargo
  features and fails static validation when an uncompiled feature is requested.
- **Side channels & live streaming** — `live_streaming.rs` bridges to a supervised Python
  OTel/MLflow child over versioned stdio (`LIVE_STREAMING_PROTOCOL_VERSION=1`, bounded queue
  drops oldest, never backpressures). `network_latency.rs`, `gpu_telemetry.rs`,
  `server_metrics.rs` are `ScheduledPhaseSidecar`s barrier-synchronized to phases.

### D. HTTP transport (`rust/runtime/src/http.rs`, `rust/runtime/src/transport_http/`)

- **`TransportSink`** (`http.rs`) — the online sink over the hyper client + `Clock`; `!Send`,
  `Rc`-based. `HttpRequest` (implements `Dispatchable`) carries either `request_body: Value`
  (accuracy) **xor** `request_body_bytes: Bytes` (byte-exact fast path). Maps observer callbacks
  to hyper streaming + SSE parsing: `on_admit` at dispatch, TTFT released only on a *meaningful*
  chat token (`is_meaningful_chat_token` — role-only/usage-only/finish-only/`[DONE]` do not
  release, mirroring the Python worker), `on_classified_token` per non-empty delta (Output vs
  Reasoning), terminal classification, `on_usage`, then a k6-style `HttpTrace`. Implements the
  `RequestExecutor`, `HttpRequestDispatcher`, and `TurnDispatcher` seams; `prewarm` is the
  "workers ready, go" barrier (one discarded round-trip, never recorded).
- **`http/endpoint_dispatch.rs`** — endpoint-aware dispatch: endpoint adapters own decode
  semantics, `transport_http` owns URL/body/lifecycle via `HttpEndpointBinding`; absorbs 13
  usage fields (incl. Anthropic cache aliases), classifies tokens, reconstructs the assistant
  turn.
- **`transport_http/`** — a behavioral port of the Python aiohttp transport, entirely
  Clock-driven. `HttpTransport` facade → `HttpClient` request path (origin-form URI + explicit
  Host so h1 and h2c both work; per-chunk body-size enforcement; `ChunkTiming`). `connection.rs`
  does DNS→TCP(+socketopts)→optional TLS/ALPN→httpN, every phase timestamped;
  **`SendCompletion`/`TimedBody`** capture the true send-complete instant so post-send
  cancellation (HTTP **499**) is anchored to send completion, immune to executor lag. **UDS**
  path (`#[cfg(unix)]`, h1 over `UnixStream`). TLS via rustls/aws-lc-rs; `ssl_verify=false`
  disables only chain/hostname (signatures still verified). `pool.rs` — per-origin H1 idle pool +
  sticky sessions + shared H2 multiplex, RAII `ConnectionLease` (`mark_reusable` only after a
  fully-drained response), `ConnectionReuseStrategy::{Pooled, Never, StickyUserSessions}`.
  `resolver.rs` — Clock-injected caching DNS. `sse/reader.rs` — incremental SSE parser with a
  3-byte cross-chunk back-scan and multibyte-safe UTF-8 (decode only complete lines). The client
  never consults ambient `HTTP_PROXY` (loopback benchmarking stays direct).
- **`ClientConfig`** knobs: `connect/request/total_timeout_ns` (one absolute deadline that can't
  restart), `max_response_body_bytes`, `ssl_verify` (default true), `prepared_tls`,
  `http_version: Auto|Http1Only|Http2PriorKnowledge`, `keepalive_ns` (300 s),
  `max_connections_per_origin` (2500), `use_dns_cache`, `dns_cache_ttl_ns`, `uds_path`.

### E. gRPC transport (`rust/runtime/src/grpc.rs`, `rust/runtime/src/transport_grpc/`)

- **`GrpcTransportSink`** (`grpc.rs`) — protocol-v2-only scheduled sink over Tonic + `Clock`;
  accepts only worker-local prepared endpoints; produces the **same** observer callbacks and a
  compatibility `RequestRecord`/`HttpTrace` so downstream metrics don't branch on transport.
- **`GrpcTransport`** (`transport_grpc/transport.rs`) — Clock-injected Tonic; all three call
  shapes (unary, server-streaming, bidi) over an identity `RawBytesCodec`; per-authority pooled
  / sticky / fresh channels; a `DeadlineKind::{ChannelReady, Cancellation, Total}` model races
  the RPC against `clock.sleep`; gRPC status → HTTP-equivalent mapping (Cancelled→499,
  DeadlineExceeded→504, …).
- **Bindings** — `GrpcEndpointBinding`/`GrpcEndpointBindingFactory` with transactional
  duplicate-rejecting registration. **KServe** OIP v2 (5 endpoints: embeddings/images/infer/
  rankings/vlm) with a checked-in Prost mirror of `grpc_predict_v2.proto` (no build-time
  `protoc`) and canonical-JSON↔protobuf codec (typed + raw little-endian tensor decode, FP16 via
  `half`). **Riva** — 9 endpoints (ASR unary+bidi, TTS unary+server-stream, 7 NLP) generated by
  a `binding_factory!` macro over checked-in Prost messages.
- `GrpcClientConfig`: `max_receive/send_message_size` (256 MiB), `channel_ready_timeout_ns`
  (30 s), `total_timeout_ns`, `trace_chunks`.

### F. Endpoints (`rust/runtime/src/endpoints/`)

Owns request-body construction and response/usage parsing per provider dialect; transport is out
of scope; **auth headers are a dialect property**.

- Two adapter contracts coexist: the legacy `Endpoint` trait (v1) and the open-registry
  `EndpointFactory` → `PreparedEndpoint` path (v2). The allocation-free formatter seam is
  `PreparedEndpointBehavior::format_prepared_payload(&PreparedRequest, &RawEndpointConfig) ->
  BodyPlan` — note it returns a **`BodyPlan`** (from `crate::body_plan`), not a raw `Value`, so
  the dataset materializer splices pre-serialized message wires without re-serializing.
- Built-in dialects: `chat`, `completions`, `responses`, Anthropic `messages` (x-api-key +
  anthropic-version, cache-usage reconciliation), `embeddings`/`chat_embeddings`, NIM
  embeddings, three ranking flavors (`Nim`/`Cohere`/`HfTei`), HF `generate`, image
  generation/edit (multipart), video generation, image retrieval, Solido RAG, token-native
  `vllm_generate` (`requires_raw_token_ids`, non-precomputable body), the flexible `raw` and
  minijinja `template` endpoints (JMESPath response selectors), 9 KServe + 9 Riva open-registry
  factories, and the materialization-only `dynosim` dialect (unconditionally registered — no
  `dynamo-mocker` dep — mirroring `chat` composition so trace-hash datasets validate identically).
- `EndpointDescriptor` carries static capability facts (`produces_tokens`, `tokenizes_input`,
  `requires_raw_token_ids`, `requires_form_data/polling/inline_media`, input/output `Modality`).
  `RawEndpointConfig` (identity-free authored policy) → `EffectiveEndpointConfig` (validated,
  bound to one factory, only the frozen registry constructs it). `EndpointKey` +
  `PreparedEndpointTable` keep string lookup off the request path.
- Extraction: `extract_inputs` single-pass walk counts image/audio/video parts, collects text +
  pretokenised counts; `UsageView` normalizes disjoint provider usage (Anthropic/Bedrock cache
  fields, Gemini/Cohere envelopes, nested detail objects); `chat_chunk` is the typed OpenAI SSE
  codec.

### G. Dataset pipeline (`rust/runtime/src/dataset/`)

The linear `load → compose → store → sample → materialize` path.

- **Content-addressed segment store** (`segment.rs`): `SegmentPool`/`InMemorySegmentStore`
  behind the `SegmentStore` trait; `Handle(u32)` dense arena index; `SegmentId` = BLAKE3 that
  folds the **prefix parent id** and the message wire bytes into the identity (shared prefixes
  dedup; identical text under different prefixes/media stays distinct). `Payload::{Message, Text,
  Raw, TokenIds, Media, TraceHashIds}`.
- **Model** (`model.rs`): `Conversation`/`Turn` hold only dense `Handle`s so sharing a `Dataset`
  across worker threads shares payload bytes. `Turn::dispatch_body` encodes precedence: raw →
  token-ids → messages. Four `ConversationContextMode`s (deltas/message-array × with/without
  responses); DAG fields for the graph path.
- **`Dataset`** (`dataset.rs`): `Arc<[Conversation]>` + `Arc<dyn SegmentStore>` + precomputed
  `body_plans`. `lower_messages_for_endpoint(ShapeLowerer)` pre-serializes eligible static turns
  to message wires; `precompute_body_plans` caches profiling-phase `BodyPlan`s for zero-reserialize
  dispatch. Full DAG validation (lineage, branch/spawn/fork rules, cycles).
- **Loaders** (`loader/`): `simple` (single/multi-turn JSONL), `synthetic` (paired ISL/OSL,
  prefix pools, multimodal, token-native), `trace` (Mooncake/Bailian/BurstGPT/SageMaker),
  `public` (Accuracy/ShareGPT/HF/MT-Bench/… with remote HF fetch), `random_pool`, `raw_payload`,
  `asr`, `exgentic`. `DatasetLoader` + `Composer` paired via `DatasetFormatRegistration` in a
  `LoaderRegistry` (explicit lookup + structural auto-detect).
- **Media** (`generator/`): `SyntheticMediaGenerator` (image/audio/video) + a separate
  `SyntheticMediaPublisher` delivery seam (inline data-URI vs content-server URL). FFmpeg used
  for MP3/MP4/WebM.
- **Tokenizers** (`tokenizer.rs`): `TiktokenTokenizer` (o200k/harmony/cl100k/…),
  `HuggingFaceTokenizer` (local `tokenizer.json` + chat template).

### H. RNG substrate (`rust/runtime/src/rng/`)

Hash-derived, **order-independent** randomness: a component names its stream and the stream
depends only on `(root_seed, identifier)` via BLAKE3 seed algebra → `rand_pcg::Pcg64`.

- `RngRoot::derive(identifier) -> RandomGenerator`, `derive_seed` = first 8 bytes of BLAKE3 of
  `"{root}:{identifier}"`; hierarchical `derive_root`/`derive_indexed_root`;
  `derive_variation_seed`. Adding a consumer cannot perturb existing streams.
- `RandomGenerator` — Python-compatible sampling (choice/choices/sample/shuffle/expovariate/
  gammavariate/normal…). `HashIdRandomGenerator` reseeds per `(trace_id, hash_id)` for
  order-independent parallel trace synthesis. `dist.rs` distribution samplers (`SamplingRng`,
  `DistributionSampler`, `SequenceSampler`, `SequenceLengthDistribution`). `namespace.rs` — ~44
  canonical dotted stream names, asserted sorted/unique/collision-free.
- Cross-language parity: `AIPERF_RNG_BACKEND=rust_parity` swaps Python onto a byte-exact port of
  this Pcg64+BLAKE3 substrate.

### I. Timing & scheduling (`rust/runtime/src/timing/`, `run.rs`, `scheduled.rs`, `request_rate.rs`, …)

- **`ScheduledRuntime`** (`scheduled.rs`) — the policy-neutral bridge from a `Workload` schedule
  generator to a `TurnDispatcher`. `issue_turn_internal` synchronously mutates counters/URL/
  cancel/metadata then spawns the dispatch task (races dispatch against a cancellation latch,
  records TTFT/terminal/native response, runs the workload continuation callback). Seams:
  `TurnDispatcher` (transport-neutral backend), `Workload`, `IssuanceGate` (adaptive admission),
  `DispatchCancellation`, `TurnRecordProcessor`, `TurnLifecycleObserver`.
- **Arrivals** (`timing/intervals.rs`): `IntervalGenerator` trait with `set_rate` (mid-run
  ramp); `ArrivalPattern::{Constant, Poisson, Gamma, ConcurrencyBurst}` from a seeded RNG,
  integer-ns intervals.
- **Admission** (`timing/slots.rs`): `SlotPool` dynamic-capacity semaphore with **debt-drain**
  (a limit decrease records shortfall as debt so `effective_slots = available − debt`); RAII
  `SlotGuard`. "Prefill released on first token" is a caller policy (drop the guard at TTFT).
- **Stop bounds** (`timing/stop.rs`): ordered `StopChecker` chain `[Lifecycle, RequestCount,
  SessionCount, Duration]`, first-no wins.
- **Ramps** (`timing/ramping.rs`): the `RampStrategy` trait with concrete `LinearRamp` /
  `ExponentialRamp` / `PoissonRamp` impls, driven by an injected `Clock`; forces exact target on
  completion.
- **Cancellation** (`timing/cancellation.rs`): `BernoulliFixedDelay` — one Bernoulli draw per
  request at issuance, fixed post-send delay; warmup returns before drawing RNG so the profiling
  stream is reproducible.
- **Phase state machine** (`timing/phase/`): `PhaseKind::{Warmup, Profiling}`,
  `GracePeriod::{Disabled, Finite, Infinite}`, `PhaseState::{Created, Started, SendingComplete,
  Complete}`, `PhaseCompletionReason::{Completed, GraceTimeout, Cancelled, ForceCompleted,
  Failed}`. `ClockPhaseRunner` drives the escalation ordering **configure → setup → start →
  ramps → issuance → sending-timeout/freeze → grace → cancel-inflight → bounded drain → force
  completion**. `ClockPhaseOrchestrator` enforces warmup-before-profiling ordering and seamless
  phase overlap; SIGINT/SIGTERM lower a cancellation latch. The "AIPerf System is PROFILING"
  banner is emitted on stderr.
- **Workloads**: `RequestRateWorkload` (one turn per tick, **continuation-FIFO priority** over
  new sessions, nonblocking new-session admission), `UserCentricWorkload` (open-loop per-user
  pacing with steady-state seeding + adaptive user pool), `FixedScheduleWorkload` (absolute
  timestamp trace replay, ignores stop bounds), `SingleTurnDatasetWorkload`. Multi-turn
  continuations resolve through the `ConversationSource` seam (`multiturn.rs`). `body_plan.rs`
  materializes request bodies by concatenating pre-serialized segment bytes (zero content
  re-serialize). `failure.rs` — `OnFailure::{Continue (scheduled default), Abort (graph
  default)}`.

### J. Graph-IR engine (`rust/runtime/src/graph/`)

A thread-per-core async-dataflow engine for recorded traces / authored DAGs.

- **Model**: a `GraphRecord` is a DAG of `LlmNode`s connected by `StaticEdge`s over named
  channels (`ChannelType::{Text, Messages}`, `ReducerName::{Overwrite, AddMessages}`). Edges
  carry four firing-gate delays (after-predecessor-completion, min-start, after-predecessor-
  start, after-predecessor-first-token). `GraphTracePlan` (graph + trace + arrival offset) is the
  **whole-trace placement unit** — node turns never cross a worker boundary.
- **Lowering**: authored `dag_jsonl` → `GraphBuilder` (FORK/SPAWN/JOIN → nodes/edges/channels);
  recorded traces → the shared **LCP-trie lowerer** (`recorded/trie/`): content-parent trie
  (linear-time longest-common-prefix), idle-warp (compress idle gaps, never active intervals),
  interval-order fan-in with async-ancestor exclusion, start-anchors + first-token delays,
  block-role planning, cache-keyed prompt emission. All content interned into a blake3 segment
  pool.
- **Recorded adapters** (`recorded/`): **WEKA** (recursive JSON, heuristic message boundaries,
  honors `max_osl`), **Dynamo** (JSONL sessions→chains→trees, virtual negative hashes for
  non-replay turns, ignores `max_osl`, subagent depth guard), **aiperf_trace** (strict
  `aiperf.trace.v1` with **ground-truth** `explicit_tags`/`block_lens`). Content synthesized from
  a deterministic coding corpus or the Shakespeare corpus via the canonical BLAKE3/PCG64 RNG.
- **Execution**: `TraceExecutor<M>` per trace fires each node — `await_inputs` on the
  `VersionedChannelStore` fan-in gate → firing-gate delay → snapshot splice channels →
  materialize prompt → `NodeDispatchPolicy` admission → `GraphSink::dispatch` → publish reply to
  the output channel (reducer) → schedule successors. Runtime is `current_thread` + `LocalSet`,
  `Rc`/`RefCell`, `spawn_local`; parallelism = many traces across threads via
  `ThreadPerCoreTracePlacement`.
- **Policy seams** (`policy.rs`, `workload.rs`): `NodeDispatchPolicy` (Noop/PrefillSlot/
  Cancellation/Composite), `NodeFailurePolicy` (Resilient/AbortTrace), `RunFailurePolicy`
  (Continue/FailFast), `GraphTraceSource` (Vec/Cycling/**Partitioned** for cellular),
  `GraphArrivalPolicy` (Immediate/Scheduled/Interval), `TraceAdmissionPolicy`, `GraphStopPolicy`.
  Live transport is `TransportChatSink` (`GraphSink<OpenAiChatMessage>` over hyper/SSE feeding a
  `RequestObserver`). `validate.rs` runs a fireability-fixpoint deadlock check; the Sim dry-run
  is the backstop.

### K. Metrics core (`rust/runtime/src/metrics_core/`) and runtime adapter (`metrics.rs`)

- **Catalog** (`catalog.rs`): a declarative `MetricSpec` table (`MetricTag` names,
  units/flags/groups/dependencies), `MetricType::{Record, Aggregate, Derived}`, validated for
  uniqueness, dependency tiering, and acyclicity (toposort). Pinned at **119** rows with a stable
  `catalog_fingerprint` (103 Python identities + 16 native sweep identities).
- **Storage** (`store.rs`): NaN-sparse columnar `ColumnStore` aligned by **absolute request
  index**; `NumericColumn` (NaN = absence), `RaggedSeries` (CSR list backend for ICL),
  `CategoryInterner`. Worker merge (`append_store`) re-interns categoricals once per unique value.
- **Ingest** (`metrics.rs` + `ingest.rs`): `NativeMetricsObserver` implements `RequestObserver`,
  joins UUID-addressed events into a `RecordIngest` row (k6-style `HttpTrace`, `TokenCounts`,
  `UsageMetrics`, dimensions), addressed by absolute slot. Finalizer is runtime-neutral
  (moveable to a reduction worker). `ObserverTee` fans one event stream to multiple observers
  (e.g. the adaptive sampler).
- **Accumulation & derivation** (`accumulator.rs`): per-record TTFT/ITL/TPOT/e2e/throughput
  (ITL = `(latency − ttft)/(osl − 1)`; endpoint `usage.completion_tokens` authoritative over
  locally-tokenized OSL); SLO goodput; usage-diff and OSL-mismatch checks; then derived scalars
  in topo order (RequestThroughput, OutputTokenThroughput, Goodput, …). Rayon-parallel above
  4096 rows.
- **Sweep-lines** (`sweepline/`): right-continuous step functions with end-before-start
  tie-break; concurrency / decode & prefill throughput / tokens-in-flight curves (ICL-aware),
  duration-weighted effective vs active stats.
- **Phase windows** (`window.rs`): `ExportContext {start_ns, end_ns, phase}` where **phase is
  authoritative over wall-clock bounds**; timeslicing by record start.

### L. Reporting & export (`report.rs`, `rust/runtime/src/report.rs`, `rust/runtime/src/export/`)

- **Reporter** (`metrics_core/report.rs`): the `Reporter` trait → `NativeReport`
  (`schema_version = "2.0"`) with `run` (provenance + facts), `summary`, `metrics` (distribution/
  scalar/counter/histogram entries with labeled per-model/endpoint series + timeslices),
  `server_metrics`, `accuracy`, `errors`, and a transient in-memory `otel_per_record` side
  channel (never in committed bytes). All values are finite-or-absent (`ReportValue`).
- **Persistence** (`rust/runtime/src/report.rs`, invoked from the engine `coordinator.rs`):
  `finalize_and_write_native_report_json` joins coordinator provenance + pair facts and does the
  sole atomic `create_new` temp + rename, refusing to overwrite an existing authoritative report.
- **Export plane** (`export/`): the `Exporter` trait (`name`, `enabled(&ExportConfig)`,
  `export`), a frozen `registry()` ordering **local-file writers before network uploaders**, run
  best-effort (`run_exporters` logs and never aborts the run). Concrete exporters:
  `GenaiPerfV1` (aiperf-v1 JSON+CSV, byte-for-byte vs retired Python), `ConsoleTxt`
  (fixed-width, reproduces Rich box geometry), `Timeslice`, `AccuracyCsv`, `ServerMetrics`
  (JSON/CSV), `Parquet` (arrow-rs), `Otel` (OTLP GenAI-semconv histograms over a throwaway
  runtime), `Mlflow` (REST or file store), `Wandb` (offline `.wandb` LevelDB framing). Selected by
  `AIPERF_RUNTIME_NATIVE_EXPORT` (default `1`; `0` restores legacy Python emitters).

### M. Adaptive scale (`rust/runtime/src/adaptive.rs`, `adaptive_core/`)

Object-safe seam: `ControlActuator` (the load knob), `SlaEvaluator`, `StepPolicy`, `Controller`,
`WindowSampler`. Four live actuators — session concurrency, prefill concurrency, request rate
(verifies the interval generator honored the rate), users. `AdaptiveControlVariable::{Concurrency,
PrefillConcurrency, RequestRate, Users}` selects the actuator. The `RampUntilFailController`
monotonically ramps the knob (**Discover**) by SLA-margin-scaled or fixed-percent steps each
assessment window until an SLA breach, then steps back to the last-passing boundary and holds
(**Sustain**) for a duration with one recovery step-down; the "knee" is the conservative
boundary value. Terminal `Completed`/`Incomplete`/`Failed`. Schema-v2 JSONL event +
summary artifacts (`adaptive_scale_events.jsonl`, `adaptive_scale_summary.json`). Same futures
online and offline (everything behind `Clock` + `ControlActuator`).

### N. Accuracy (`rust/runtime/src/accuracy.rs`, `accuracy_core/`)

Rust owns scheduling/IO/timing/metrics; a long-lived **Python/Lighteval worker** owns dataset
prep, prompts, hidden tests, and grading over a versioned JSONL stdio protocol
(`EVALUATOR_PROTOCOL_VERSION = 1`). The worker is env-hardened (`env_clear()` + curated
allowlist so host secrets never reach the third-party evaluator) and identity-validated
(worker/python versions, source sha256, dependency-lock or container digest, required
capabilities). Data-hiding is enforced by `deny_unknown_fields` — hidden fields
(`ground_truth`, `private_tests`) are rejected on the wire. Problems flow through the **ordinary**
online scheduler/transport/observer (accuracy is *not* a dispatch workload); captures are graded
in batches, joined by `AccuracyAccumulator` + `AccuracyResultsAnalyzer` (Wilson CI, accuracy-at-
load, correct-answers-per-kWh from the telemetry sidecar's injected energy metric).

### O. Side-channel telemetry (`gpu_telemetry/`, `network_latency/`, `server_metrics/`, `content_server/`)

Every source/decoder/parser is a Clock-injected trait; all feed the shared `SidecarMetric`
series into the native report. Continuous scrapes dedup identical bodies → gauge distributions;
boundary scrapes bypass dedup → exact reset-clamped counter/histogram deltas.

- **`gpu_telemetry`** — DCGM-first `GpuTelemetrySource` (HTTP) or supervised Python source;
  `DcgmPrometheusDecoder` applies exact per-field scaling (`DCGM_METRICS` table); per-GPU series;
  energy/power/efficiency joins (`TotalGpuEnergy`, `OutputTokensPerJoule`, `EnergyPerUser`).
- **`network_latency`** — fresh-TCP-connect RTT calibration with DNS resolution/fallback,
  per-target population stats + a run-level mean, nonfatal structured failures.
- **`server_metrics`** — Prometheus/OpenMetrics parser (rejects JSON, auto-disables after a
  terminal error), classic + estimated histogram percentiles, a `VllmSglangMetricAtlas` deriving
  prefix-cache hit rate / KV usage / queue depth / token throughput from exact boundary deltas,
  LRU unit inference. JSON/CSV/parquet artifacts.
- **`content_server`** — an Axum server (`AIPERF_CONTENT_SERVER_ENABLED` +
  `AIPERF_CONTENT_SERVER_CONTENT_DIR`) that turns generated images/videos into stable HTTP URLs
  (audio stays inline); path-traversal-confined `ServeDir` with per-request lifecycle tracking;
  implements the `SyntheticMediaPublisher` seam.

### P. Cellular multi-process mode (`rust/runtime/src/cellular/`, `rust/runtime/src/engine/`)

`--cells N` (or `runtime.cells: N`) makes the launched `aiperf` a **controller** that spawns N
`aiperf --cell` children over a budget partition and merges their records into one report.
Five seams, each with a "Direct" single-process impl:

- **Partition** (`ModuloCellPartition`) — cell k owns `i % cell_count == cell_id`; env
  `AIPERF_CELL_ID`/`AIPERF_CELL_COUNT`; `derive_cell_root` isolates streams so adding a cell
  can't perturb existing ones.
- **Issuance** (`CellularAutonomousIssuer`) — global ordinal = `phase_ordinal_base +
  within_phase_local*cell_count + cell_id` (a single central assignment, never a shared atomic),
  so merged reports tile `0..total` densely.
- **Shard** — the authoritative path ships raw `RecordIngest` records; the controller validates
  the union of ordinals is a permutation of `0..total`, stable-sorts, and re-ingests → **byte-
  identical to a single-cell run** including last-ULP reductions. MessagePack wire (preserves
  NaN/+inf sentinels).
- **Heartbeat/Sketch** — live t-digest (Dunning) percentiles for a progress lane; report
  percentiles stay exact from the shard.
- **Transport** — length-prefixed MessagePack over TCP (`CellClient`/`ControllerTransport`).
- Fail-closed gates: HTTP transport only, synthetic single-turn, request-bounded phases
  (`concurrency|poisson|gamma|constant`), `requests >= cell_count`; rejects duration/sessions/
  adaptive/multi-turn. Seedless auto-derives a shared seed; multi-URL round-robins cell-locally;
  ramps/rate/cancellation are aggregate-equivalent (warned). Merged report omits coordinator
  finalize provenance, grouped per-error detail, and side-channel sidecar data (warned).

### Q. Dynosim co-simulation (`rust/runtime/src/dynosim.rs`, `#[cfg(feature="dynosim")]`)

In-process co-simulation of Dynamo's passive mock engine over AIPerf's `SimClock` — no sockets/
subprocesses. AIPerf owns clock/workload/observers/report; `dynamo-mocker` owns scheduler +
perf-model behind the `SteppableReplay` trait. `DynosimSink` implements the same
`RequestSink`/`HttpRequestDispatcher`/`TurnDispatcher` seams (token source priority: authored
`raw_token_ids` → `trace_hash_ids` → chat body); `DynamoGraphSink` implements `GraphSink`.
`EngineHost` bridges the DES/wall clock as a `SimEventSource`. `dynosim_offline` (Sim clock,
`drive_sim_with_source`, **enforced byte-exact parity** vs Dynamo) vs `dynosim_online` (Real
clock, `drive_real_with_source`, relaxed parity) differ **only** in the clock/driver axis.
`dynamo-full` adds the router/ZMQ/KV/AIC surface.

### R. Mock server (`aiperf-mock-server`)

A standalone OpenAI/Anthropic/TGI-compatible inference target (depends on `aiperf-runtime` only for
`aiperf_runtime::rng`); launched independently and supplies an ordinary online URL — the frontend does **not**
supervise it. A manual hyper accept loop sets `TCP_NODELAY` per socket (Nagle otherwise held
small SSE chunks). Deterministic char-based token generation (~4 chars/token, per-prompt seeded
output counts, optional Shakespeare corpus). Two latency models: **analytic** (closed-form TTFT/
ITL from ISL/OSL/concurrency + lognormal jitter) vs a global **batch scheduler** (prefill/decode
admission, goodput collapse past a queue threshold). Content-addressed **prefix cache**
(SGLang-style eviction policies, chain-hash block matching). Seven independent Prometheus
dialects (`vllm:*`/`sglang:*`/`trtllm:*`/`dynamo_*`/…) and a **DCGM faker** auto-driven by
measured throughput (100 Hz sampler → per-GPU synthetic telemetry). `--fast` zeros latencies and
disables the scheduler + prefix cache.

---

## 5. Configuration & options catalog

| Axis | Options (Config-v2 → protocol-v2) |
|---|---|
| **Transport** (`cfg.transport.type`) | `http`, `grpc`, `dynosim_offline`, `dynosim_online` (last two need a `dynosim`-feature build) |
| **Workload** (derived from dataset format) | `scheduled` (linear datasets), `graph` (`dag_jsonl`/`weka_trace`/`dynamo_trace`) |
| **Supported pairs (base build)** | `(http, scheduled)`, `(http, graph)`, `(grpc, scheduled)` (+ dynosim × scheduled/graph with the feature) |
| **Phase types** (`cfg.phases[].type`) | `concurrency`, `poisson`, `gamma`, `constant`, `user_centric`, `fixed_schedule`; each with `warmup|profiling`, stop bounds (`requests`/`sessions`/`duration`), grace, ramps (`concurrency_ramp`/`prefill_ramp`/`rate_ramp`), `cancellation`, `adaptive_scale` |
| **Arrival distributions** | `Constant`, `Poisson`, `Gamma`, `ConcurrencyBurst` |
| **Ramp strategies** | `Linear`, `Exponential`, `Poisson` |
| **Adaptive control variable** | `concurrency`, `prefill_concurrency`, `request_rate`, `users` (strategy `ramp_until_fail`; step `sla_margin`/`fixed_percent`) |
| **Model selection** | `round_robin` (default), `random`, `weighted` |
| **Connection reuse** | HTTP `pooled`/`never`/`sticky-user-sessions`; gRPC pooled/never/sticky |
| **Endpoints** | chat, completions, responses, messages (Anthropic), embeddings, chat_embeddings, nim/cohere/hf_tei rankings, huggingface_generate, image_generation/edit, video_generation, image_retrieval, solido_rag, raw, template, vllm_generate, 9× kserve_*, 9× riva_*, dynosim |
| **Sequence-length distributions** | `Fixed`, `Normal`, `LogNormal`, `Multimodal`, `Empirical` |
| **Failure policy** | `continue` (scheduled default), `abort` (graph default) |
| **Cellular** | `--cells N` / `runtime.cells` (HTTP + synthetic + request-bounded only) |

**Key environment variables** (runtime switches):

| Var | Effect |
|---|---|
| `AIPERF_RUNTIME_ENGINE` | `rust` (default) native execution engine vs `python` legacy A/B fallback service mesh |
| `AIPERF_RUNTIME_NATIVE_EXPORT` | `1` (default) native exporter plane vs `0` legacy Python emitters |
| `AIPERF_RNG_BACKEND` | `legacy` vs `rust_parity` (byte-exact Pcg64+BLAKE3 port in Python) |
| `AIPERF_RUNNER_LOG` | execution-engine stderr tracing filter |
| `AIPERF_CONTENT_SERVER_ENABLED` / `_CONTENT_DIR` | enable the run-owned content server |
| `AIPERF_CELL_ID` / `AIPERF_CELL_COUNT` / `AIPERF_CELL_CONTROLLER_ADDR` / `AIPERF_CELL_PHASE_ORDINAL_BASES` | cellular child wiring |
| `AIPERF_CACHE_DIR`, `HF_ENDPOINT`, `HF_TOKEN`, `HF_HUB_OFFLINE` | tokenizer/dataset fetch |
| `AIPERF_EXPORT_SUBDIR` | redirect native sink outputs for parity byte-diff |
| `AIPERF_CELLULAR_HEARTBEAT_LOG` | single-process live heartbeat NDJSON lane |
| `AIPERF_ACCURACY_PYTHON`, `AIPERF_DYNAMO_MAX_SUBAGENT_DEPTH` | accuracy worker / Dynamo trace tuning |

**Cargo features**: `dynosim` (offline/online Dynamo replay), `dynamo-aic-forward-pass`,
`dynamo-router-runtime`, `dynamo-zmq-events`, `dynamo-kvbm-offload`, `dynamo-profile`,
`dynamo-full`, `dynamo-parity`.

---

## 6. Extension-seam index ("everything is a trait")

Every conceivable variation point is an implementable trait with ≥1 concrete impl; where it
crosses a `dyn` boundary it is object-safe, where it is hot-path it is monomorphized.

| Layer | Seam(s) |
|---|---|
| Time | `Clock` (RealClock / SimClock) |
| Dispatch/measure | `Dispatchable`, `RequestSink<R>`, `RequestObserver` |
| Runner composition | `RunnerRegistryFactory`, `RunnerTransportFactory`, `RunnerWorkloadFactory`, `RunnerPairFactory`, `PreparedRunnerOperation`, `PreparedReportCommit`, `AIPerfRegistryFactory`, `AIPerfExtension` |
| Execution placement | `RequestExecutorFactory`/`RequestExecutor`, `RunnerGraphPlacementFactory`, `OnlineReadinessPlanFactory`, `ReadinessTransportFactory`, `ControlPlaneHttpProviderFactory` |
| Inputs | `RunnerGraphInputAdapterResolver`, `RunnerDatasetInputAdapterResolver`, `RunnerSidecarInputAdapterResolver`, `OnlineTokenizerSourceResolver` |
| HTTP | `ConnectionManager`, `DnsResolver`/`HostLookup`, `SseMessageFilter`/`SseMessageHandler`, `HttpEndpointBinding`, `RequestExecutor`, `TurnDispatcher` |
| gRPC | `GrpcEndpointBinding`/`GrpcEndpointBindingFactory`, `RivaWireBehavior`, `RawBytesCodec` |
| Endpoints | `Endpoint`, `PreparedEndpointBehavior`, `EndpointFactory`, `PreparedEndpoint`, `EndpointResolver`, `TurnMessageLowerer` |
| Dataset | `DatasetLoader`, `Composer`, `Sampler(Factory)`, `ModelSelector(Factory)`, `TracePromptStoragePolicy`, `PromptGenerator(Factory)`, `TextTokenizer`, `TraceSynthesizer`, `MediaResolver`, `SyntheticMediaGenerator(Factory)`, `SyntheticMediaPublisher`, `DatasetFetcher`, `SegmentStore`, `RequestMaterializer` |
| RNG | `SamplingRng`, `DistributionSampler<R>`, `SequenceSampler<R>` |
| Scheduling/timing | `Workload`, `TurnDispatcher`, `IntervalGenerator`, `StopCondition`, `RampStrategy`, `CancellationPolicy`, `UrlSelector`, `ConversationSource`, `FixedScheduleSource`, `IssuanceGate`, `LocalTaskScheduler`, `TurnRecordProcessor`, `TurnLifecycleObserver` |
| Phase lifecycle | `PhaseRunner(Factory)`, `PhaseOrchestrator`, `PhaseExecution(Factory)`, `PhaseObserver`, `ScheduledPhaseController`, `ScheduledPhaseResources`, `ScheduledPhaseSidecar`, `ScheduledRuntimeExtension` |
| Graph | `GraphSink<M>`, `WireMessage`, `PromptMaterializer`, `TracePlacement(Factory)`, `NodeDispatchPolicy`, `NodeFailurePolicy`, `RunFailurePolicy`, `GraphTraceSource`, `GraphArrivalPolicy`, `TraceAdmissionPolicy`, `GraphStopPolicy`, `SimEventSource`, `RecordedContentSynthesizer` |
| Metrics/report | `Accumulator<Record>`, `ListMetricBackend`, `MetricSpec`/catalog, `Reporter`, `Analyzer`, `Exporter` |
| Adaptive | `ControlActuator`, `SlaEvaluator`, `StepPolicy`, `Controller`, `WindowSampler`, `AdaptiveArtifactSink`, `UserTargetController` |
| Accuracy | `AccuracyEvaluator` (Python worker) |
| Telemetry | `GpuTelemetrySource`/`GpuTelemetryDecoder`, `NetworkLatencyProbe`, server-metrics source/parser/atlas/unit, `ContentServerFactory` |
| Cellular | `CellPartition`, `IssuanceAuthority`, `RecordsShard`, `CellClient`/`ControllerTransport` |

---

## 7. Cross-cutting invariants (from the source, not the docs)

- **All time through `Clock`** in clock-aware code — never `Instant::now()`, `SystemTime::now()`,
  or raw `tokio::time` for measurement/firing gates. Virtual time is integer ns with `(at_ns,
  seq_no)` deterministic tie-break.
- **Thread-per-core, not work-stealing** on the hot path: N `current_thread` runtimes +
  `LocalSet`, per-trace state in `Rc`/`RefCell`, `!Send` futures, `spawn_local`; parallelism is
  many traces across threads.
- **No `Arc`/`Mutex` on hot paths** — accumulate lock-free per worker, merge once at the join.
- **Content-addressed segments** — blake3, prefix-dependent; materialize = concat pre-serialized
  bytes, never re-serialize.
- **NaN/Inf discipline** — every metric value crossing a serialization boundary is finite or
  explicitly absent; cellular wire uses MessagePack because NaN-sparsity + `+inf` need a
  self-describing binary format.
- **One product entry path** — the `aiperf` entry point projects exactly one side-effect-free
  protocol-v2 request; the `aiperf --execute` engine is v2-only and fails closed on any
  unregistered transport/workload/endpoint id.
- **Redaction at the boundary** — every diagnostic is credential-scrubbed before stdout; secrets
  never enter DTOs and zeroize on drop.
