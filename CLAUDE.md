<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AIPerf

> ✅ **CANONICAL PRODUCT ARCHITECTURE.** The Python `aiperf` command owns the only human-facing CLI, Config v2, outer orchestration, and presentation. A fresh `aiperf-runner` child is the **only Rust executable on the product execution path** and owns exactly one run's high-performance path; `aiperf-mock-server` is a standalone developer/test inference target, not an orchestrated backend process. The `rust/aiperf` package is a runtime library with no binary target. There is no ZMQ, service mesh, multiprocess credit protocol, mmap dataset cache, or `plugins.yaml` on the native path. The `aiperf-rs` and `~/projects/aiperf-rust` trees remain **DEPRECATED**. Design record: [`specs/`](specs/); start-here index: [`llms.txt`](llms.txt).
>
> Rust execution truth is in `rust/`; canonical Python frontend truth includes `src/aiperf/config/`, `src/aiperf/orchestrator/`, and `src/aiperf/cli_runner/`. Other inherited Python controller/service code is legacy, not an alternate hot path.

## What this is

A Python-orchestrated, Rust-executed load generator + measurement front-end for inference servers. The runner dispatches OpenAI-compatible, Anthropic Messages, KServe, and NVIDIA Riva ASR/TTS/NLP requests over native HTTP/SSE or gRPC, records fine-grained timing (TTFT / ITL / TPOT / e2e / throughput / goodput), and serializes native results. The execution substrate is designed for **three interchangeable modes** over a single `{transport, clock}` seam:

1. **ONLINE-real** — real HTTP or gRPC to a real server, wall clock. *(built)*
2. **ONLINE-mock** — real HTTP to a mock server (`aiperf-mock-server`), wall clock. *(built — same code as (1), different target)*
3. **OFFLINE-mock** — in-process virtual-clock co-simulation of the Dynamo mocker engine, no sockets, deterministic. *(built behind the non-default `dynosim` Cargo feature; requires the sibling `dynamo-aiperf-native` checkout.)*

Online-real and online-mock are product-reachable through Python + the base runner. The workspace-owned `aiperf-mock-server` server is launched independently and supplies an ordinary online target URL; Python does not supervise it as part of a run. Offline-mock is product-reachable through the same Python Config-v2 path when the selected `aiperf-runner` is built with `dynosim`; exact-image capabilities omit those pairs from the base build.

## Canonical vs aspirational — the code is a walking skeleton

Ground every claim in `crate/src/file.rs`, not the specs: specs are design intent, **code is truth**. The current gaps between the north-star design and today's code:

- The north-star's `Backend` / `Engine` / `Harness` vocabulary is **aspirational**. Today's seam is `Clock` + `RequestSink<R>` / `RequestObserver` / `Dispatchable`.
- **One product entry path.** Python projects one BenchmarkRun protocol-v2 request and launches `aiperf-runner`; runner is v2-only, rejects non-v2. Selects scheduled vs graph from dataset format; binds `cfg.transport.type`. `--capabilities` emits a plugins.yaml-shaped catalog. Unregistered transport/endpoint ids fail closed.
- **Native online transports are built.** HTTP: Clock-injected hyper (`transport_http`), h1/h2c/UDS/TLS/SSE, post-send cancellation. gRPC: Clock-injected Tonic (`transport_grpc`), KServe OIP + Riva ASR/TTS/NLP. Both on `current_thread` + `LocalSet`; no Python gRPC plugin.
- **Open endpoint registry, v2 validation, and frozen linked application are built.** `aiperf::endpoints`: open registry with 9 KServe + 9 Riva + `vllm_generate` factories, raw/effective config, worker-local prepared bindings. `aiperf::extensions` transactionally composes new dialects. `RunnerApplication` freezes the linked registry, input resolvers, pair factories, and v2 coordinator at bootstrap.
- **Anthropic Messages parity is built.** `aiperf::endpoints::MessagesEndpoint`: exact PR 731 `/v1/messages` body, auth headers, all content shapes, streaming/non-streaming parsing, cache-usage reconciliation, thinking/signature replay. Graph transport remains Chat-shaped.
- **Offline and wall-clock Dynamo co-simulation are built** (behind `dynosim` feature). `aiperf::dynosim`: same `RequestSink` / `TurnDispatcher` / `GraphSink` over Dynamo's `SteppableReplay`. `dynosim_offline` = virtual clock; `dynosim_online` = wall clock, apples-to-apples with Dynamo's live driver. `dynamo-full` adds router/ZMQ/KV/AIC.
- **Dynamo replay is authored through `aiperf profile`**; no `aiperf dynosim` command. Select `transport.type: dynosim_offline|dynosim_online` in Config v2; trace file and concurrency/rate axes reuse the shared `dataset`/`phases` surface. Live mocker and replay-optimize sweep remain `python -m dynamo.*` tools.
- **Online scheduling policy is built.** `ScheduledRuntime` paces arrivals (Poisson/Gamma/constant/burst) with `SlotPool` admission and `StopChecker` bounds. `RequestRateWorkload`: one turn per tick, FIFO continuation priority. Graph mode: trait-backed root/arrival/admission/placement/failure policy.
- **Phase orchestration is built.** `aiperf::timing::phase`: `PhaseLifecycle`, `ClockPhaseRunner`/`Orchestrator`, duration→grace→cancel→drain→force escalation, warmup→profiling ordering, cancellation latch. Graph adapts through `PhaseExecutionFactory`.
- **Ancillary timing policy is built.** `aiperf::timing`: `RampStrategy`/`RampDriver`, seeded `CancellationPolicy`, `UrlSelector`. HTTP 499 anchored to send completion; in-process endpoint rejects URL selection; fixed schedules reject ramps.
- **Adaptive scale is built** as `aiperf::adaptive_core`: object-safe actuator/evaluator/step/window/controller; all four live actuators; `ramp_until_fail` controller; schema-v2 artifacts. Same futures online and offline.
- **Hash-derived RNG substrate is built** as `aiperf::rng`: `RngRoot::derive` / BLAKE3 seed derivation, `RandomGenerator`, `HashIdRandomGenerator`, sampler seams. Non-graph scheduler integration remains future work.
- **WEKA and Dynamo recorded-graph adapters are built.** `aiperf::graph::recorded`: strict WEKA/Dynamo decode, shared LCP-trie lowerer, dense segment interning. All content via `aiperf::rng` BLAKE3/PCG64; Python never parses or lowers either input.
- **Performance metrics are built** as `aiperf::metrics_core`: 119-row catalog, NaN-sparse storage, ragged ICL, all sweep-lines, phase windows, worker merge, typed native-v2 `Reporter`. The native-Rust exporter plane (`aiperf::export`) is built and is the default sole emitter: aiperf-v1 (genai-perf-v1) JSON+CSV, timeslice, server-metrics JSON/CSV/parquet, accuracy CSV, console.txt, OTLP per-record metrics, MLflow, and W&B — parity-verified byte-for-byte vs the retired Python exporters (`AIPERF_RUNTIME_NATIVE_EXPORT=0` restores legacy Python).
- **Legacy static/stateful accuracy uses canonical Python evaluators.** Rust: scheduling, HTTP I/O, timing, metrics. Python worker: prompts, hidden tests, execution, scoring over JSONL stdio. Pinned: Harbor 0.18, AgentLab 0.4.2 + BrowserGym 0.14.3, MCPMark `cd45b7f57923b9b3985467f5139927575f83141c`. No Rust prompt builders, graders, or model clients. OSWorld/AppWorld still need canonical providers.
- **Compile-time extensibility is built.** `AiperfRegistry` / `AiperfExtension`: static registration, duplicate rejection. `RunnerApplication` freezes at bootstrap. No `plugins.yaml`, runtime discovery, or dynamic ABI.
- **Dataset/segment unified store is built end to end** as `aiperf::dataset`. `Payload::TokenIds` + `Turn::raw_token_ids` for exact token arrays. `dag_jsonl` / `weka_trace` / `dynamo_trace` bypass the linear loader registry: runner-owned resolver → one compiler → frozen `SegmentStore`; no `Dataset` / `DagMetadata` intermediate.
- **Native content serving is built** (`aiperf::content_server`). `AIPERF_CONTENT_SERVER_ENABLED=true` + non-empty dir: stable image/video URLs, audio inline. gRPC, offline, and agentic/evaluation pairs reject the sidecar.
- **Telemetry archive/watch was removed; side-channel telemetry is built.** The legacy `aiperf-prometheus`, `aiperf-telemetry-archive`, `aiperf watch`, and `telemetry_watch` workload/pair are deleted. `aiperf::server_metrics` owns its own Prometheus/OpenMetrics parser. **GPU telemetry and network-latency calibration are built** as Clock-injected side-channel accumulator modules: `aiperf::gpu_telemetry` (DCGM-first source/decoder traits, canonical field scaling, boundary counter snapshots, cadence gauge distributions, supervised Python source, per-GPU series, energy/power/efficiency joins) and `aiperf::network_latency` (fresh TCP-connect RTT calibration with DNS resolution/fallback, per-target population stats, run-level mean, nonfatal structured failures). Both feed the shared side-channel accumulator seam. The feature-gated `aiperf::aic_runtime` builds an aiconfigurator timing engine onto the mocker's `perf_model` for the DynoSim path (part of the `dynamo-full` AIC surface).

When something is designed-but-not-built, this file says so. Do not assume a spec feature exists in the code.

## Crate workspace (`rust/`)

Workspace: `edition = "2024"`, `resolver = "3"`. Four crates; 16 former `aiperf-*` library crates are now modules of `aiperf` (see §Module organization below).

| Crate | Purpose | Key files |
|---|---|---|
| `loadgen-core` | Transport-neutral dispatch/measure seam + the collector. `Dispatchable`, `RequestSink<R>`, local-loop `RequestObserver` (no `Send`/`Sync`; f64-ms timestamps; optional `ObservedTokenKind` classification; terminal `ObservedUsage` with optional fields), `TraceCollector` → `TraceSimulationReport`; the `CollectorObserver` pure recorder. Zero engine/KV/HTTP deps. | `sink.rs`, `collector.rs`, `observer.rs` |
| `aiperf` | Library-only runtime composition used by `aiperf-runner`; there is no `src/main.rs` or native `aiperf` executable. It owns scheduled/transport composition, HTTP and gRPC prepared sinks, online pacing, datasets, exact token-array response observation/usage, the provider-neutral evaluation workload/typed host registry/fair arbiter/ledger/retry/scoped proxy/report join, legacy static and stateful accuracy seams, adaptive/ancillary policy, native report persistence, and the feature-gated direct raw-token Dynamo adapter. Sixteen former library crates are inlined as modules (see §Module organization). | `lib.rs`, `evaluation.rs`, `evaluation/`, `dynosim.rs`, `ancillary.rs`, `metrics.rs`, `accuracy.rs`, `adaptive.rs`, `http.rs`, `grpc.rs`, `run.rs`, `phase_runtime.rs`, `scheduled.rs`, `report.rs` |
| `aiperf-runner` | Sole strict Rust executable used by the Python orchestrator. Python always sends exact-image-bound protocol-v2 `validate`/`execute` operations; the runner is protocol-v2-only, advertises `protocol_versions: [2]`, and rejects any non-v2 request as a v2 failure envelope. Frozen transport/workload/pair registries direct-load one prepared operation and derive executable capabilities. One injected runner-owned graph-input resolver performs identity-only selection, strict decode, and direct compilation. The base build registers HTTP scheduled/graph/static-accuracy/agentic plus native gRPC scheduled; `dynosim` adds `dynosim_offline`/`dynosim_online` scheduled/graph. | `main.rs`, `protocol_v2.rs`, `registry.rs`, `graph_input.rs`, `execute.rs`, `online_execution.rs`, `offline_execution.rs`, `agentic_execution.rs`, `grpc_execution.rs`, `grpc_turn_execution.rs`, `graph_phase_runtime.rs`, `graph_execution.rs`, `records.rs` |
| `aiperf-mock-server` | Standalone online test/benchmark inference target: OpenAI chat/completions/embeddings, TGI, rerank, image, multimodal, and RAG routes; real SSE; analytic or batch-scheduler latency; deterministic token generation; prefix-cache policy; Prometheus backend dialects; and synthetic DCGM telemetry. It exports an Axum router for tests and a tuned Hyper server binary, but is not part of the runner dependency graph. | `app.rs`, `config.rs`, `handlers.rs`, `tokens.rs`, `latency.rs`, `scheduler.rs`, `prefix_cache.rs`, `metrics.rs`, `prom.rs`, `dcgm.rs`, `main.rs` |

Dependency direction: `aiperf-runner` → {`aiperf`, `loadgen-core`}; `aiperf` → {`loadgen-core`} plus optional `dynamo-mocker` only under `dynosim`; `aiperf-mock-server` → {`aiperf`}. The runner and mock-rs do not depend on each other; real-network integration tests spawn the mock binary as an ordinary target.

## Module organization (`rust/aiperf/src/`)

Sixteen former `aiperf-*` library crates are now `aiperf::<module>::` namespaces. All inter-module imports use `crate::<module>::` within `aiperf`; runner and mock-rs use `aiperf::<module>::`. Five modules use a `_core` or `transport_` prefix to avoid name conflicts (`metrics_core`, `adaptive_core`, `accuracy_core`, `transport_http`, `transport_grpc`). Full module table with purposes and key files: [`docs/module-organization.md`](docs/module-organization.md).

## The two seams (the whole architecture)

- **`{clock}`** (`aiperf-clock`): `RealClock` vs `SimClock` behind one `Clock` trait. `is_virtual()` selects the `drive_real` (tokio reactor drives) vs `drive_sim` (idle-pump: poll the `LocalSet` to quiescence draining all same-instant work → `advance_to(next_event_time)` waking heap-ordered sleepers → repeat) driver over the *same* executor. Virtual time is integer ns with an `(at_ns, seq_no)` deterministic tie-break — **never `tokio::time`** (its 1 ms timer wheel destroys µs/ns firing gates).
- **`{transport}`** (`loadgen-core::sink`): `RequestSink<R>::dispatch` drives a `Dispatchable` request to terminal and emits `on_arrival` / `on_admit` / `on_token` (or `on_classified_token` with `ObservedTokenKind::{Output,Reasoning}`) / terminal `on_usage(ObservedUsage)` / `on_terminal` through a `RequestObserver`. Classification defaults to `on_token`; usage defaults to a no-op and keeps unreported counts as absent fields. `RequestObserver` has no `Send`/`Sync` supertraits: each thread-per-core worker owns a local observer graph in `Rc`/`RefCell`; cross-thread consumers may still provide a thread-safe implementation. Real HTTP, native gRPC, mock HTTP, and the feature-gated in-process engine co-sim all feed this observer seam; `GrpcRequest` retains its prepared endpoint reference for `RequestSink` dispatch. TTFT is the first token callback; sinks emit no separate first-token event.

## Extensibility & porting discipline (non-negotiable)

- **Every extension point is a trait.** Anything that could ever have a second implementation — a transport, a clock, an accuracy evaluator, a request/response shape, an arrival pattern, a dataset loader, a sampler, a segment store, a metric accumulator, an analyzer, an exporter, an endpoint dialect, a tokenizer, a scheduling policy — MUST be an implementable `trait` (object-safe where it crosses a `dyn` boundary; generic where it is hot-path monomorphized) with at least one concrete impl behind it. Never hardcode a concrete type where a future variant is conceivable. If you are `match`-ing on an enum of "kinds" or branching on a string mode, that is a trait waiting to be extracted. In-tree precedent: `Clock`, `RequestSink<R>` / `RequestObserver` / `Dispatchable`, `AccuracyEvaluator`, `SegmentStore` / `PromptMaterializer`, `GraphSink`.
- **Always design ahead.** When you add code, add the seam the next plausible requirement will need — name the trait, take the trait (not the concrete) in signatures, thread the injection point — even if you ship exactly one impl today. The three-modes-for-free property only survives if features are written against the `{transport, clock}` seams, never against a specific backend/clock/transport; a feature that works in only one mode is a design bug. Note the extension you are leaving open in a `//!` / `///` doc comment.
- **Read the ENTIRE Python source before porting ANYTHING.** Before porting a behavior, read the WHOLE Python file end-to-end AND every file it meaningfully touches (its imports, the models it builds, the callers that consume its output, the tests that pin it). Never port from a docstring, a grep hit, a snippet, or a spec paragraph — those omit the earned-in-blood edge cases (SSE fast-paths, backward usage walks, finish/usage reconciliation, firing-gate rounding), which live in the parts of the file you assumed you could skip. Cite the exact Python `path:line` you ported from in the Rust `//!` / `///` docs, and guard it with a parity test wherever byte-exactness matters.

## Coding standards (Rust)

- **SPDX header** (`// SPDX-FileCopyrightText …` + `// SPDX-License-Identifier: Apache-2.0`) atop every source file. `//!` module docs; `///` doc comment on every public item.
- **Thread-per-core**, not work-stealing, on the hot path: N OS threads, each a `current_thread` tokio runtime + `LocalSet`; per-trace state is `Rc`/`RefCell`, futures are `!Send`, tasks are `spawn_local`; parallelism = many traces across threads. See `aiperf-graph/src/runtime.rs`, `transport_bench.rs`. The runner's online `run.rs` path follows the same `!Send` model on a single `current_thread` runtime + `LocalSet`, with `Rc` observers and dynamic `SlotPool` admission.
- **All time through `Clock`** in the clock-aware crates (`aiperf-transport-http`, `aiperf-transport-grpc`, `aiperf-graph`, and the runner/library online path): never `Instant::now()`, `SystemTime::now()`, or raw `tokio::time` for measurement or firing gates. The relocated OpenAI SSE chunk types (`aiperf-transport-http::sse`) and the `CollectorObserver` recorder (`loadgen-core::observer`) own no clock; callers supply Clock-derived timestamps.
- **No `Arc`/`Mutex` on hot paths.** Accumulate lock-free per-thread / per-worker and merge once at the end (the graph bench keeps a per-worker accumulator and merges at the join) — never contend a shared collector lock per token on the throughput path.
- **Content-addressed segments**: blake3, prefix-dependent (fold the parent id into the hash so shared prefixes dedup and identical text under different prefixes stays distinct); materialize = clone/concat pre-serialized bytes, never re-serialize.
- **mimalloc** is installed by the `aiperf-runner` executable; per-request allocation churn in the graph executor and streaming client was the top profiled hotspot.
- **Loopback benchmarking**: the hyper transport never consults the ambient `HTTP_PROXY`, so localhost traffic is not proxied (an ambient proxy 405s localhost and tanks throughput).
- **SSE**: buffer raw bytes, split lines on the byte buffer, UTF-8-decode only complete lines (a multibyte char may straddle a TCP chunk boundary).
- **Authoritative token counts**: request `stream_options.include_usage` so `usage.completion_tokens` is returned; the HTTP sink emits it through `RequestObserver::on_usage`. Adaptive windows and the collector-wide native accumulator reconcile OSL and the `(last−first)/(osl−1)` ITL denominator to authoritative completion usage while preserving observed chunk timings.
- **Errors**: `anyhow` in the runner/app layer; library crates use plain error enums with hand-written `Display` (no `thiserror`).
- **Python owns the CLI schema.** Rust runner requests use strict `serde` / `serde_json` DTOs with unknown-field rejection. Prefer a direct-serialized request body on the hot path.
- **NaN/Inf discipline**: numeric metric values crossing a serialization boundary must be finite or explicitly absent.
- **Comments explain *why*, never *what*.** No emojis in code. Read the actual code, never trust the docstrings or comments.

## Build, test, run

The (inherited-Python) `Makefile` has **no** cargo targets — use cargo directly:

```bash
cargo build                  # debug build of the whole workspace
cargo build --release        # optimized — use for any throughput number
cargo test                   # all unit tests (self-contained: in-process axum mock, no external server)
cargo test -p aiperf-graph   # one crate
cargo test -p aiperf-mock-server # standalone mock-server unit + HTTP integration suite
cargo clippy --all-targets   # lints
cargo fmt                    # format (rustfmt)

# Product runner with offline scheduled + graph pairs; expects the sibling checkout.
cargo build -p aiperf-runner --features dynosim
# Focused library algorithms remain independently testable.
cargo test -p aiperf --features dynosim --lib
# Complete native build: router runtime, ZMQ events, KV offload, AIC, and profile support.
cargo build -p aiperf --features dynamo-full

# Standalone online mock target for local integration runs.
cargo run -p aiperf-mock-server -- --fast
```

Run the product:

```bash
# Generate, validate, and run through the only human-facing frontend.
aiperf config init --template minimal --output benchmark.yaml
aiperf config validate benchmark.yaml
aiperf profile --config benchmark.yaml

# Externalize generated images/videos through the run-owned native server.
mkdir -p /tmp/aiperf-content
AIPERF_CONTENT_SERVER_ENABLED=true \
  AIPERF_CONTENT_SERVER_CONTENT_DIR=/tmp/aiperf-content \
  aiperf profile --config benchmark.yaml

# Cellular (multi-process) mode: `--cells N` (or `runtime.cells: N`) makes the launched
# runner a controller that spawns N `aiperf-runner --cell` children over a
# (cell_id, cell_count) budget partition and merges their records into one report.
# `--cells 1` (default) is the unchanged single-process path. Supported for seeded,
# synthetic, single-turn HTTP runs with request-bounded phases and a single endpoint URL;
# fails closed otherwise. E2e from the frontend: rust/e2e/tests/test_cellular.rs.
aiperf profile --config benchmark.yaml --cells 4

# Developer-only protocol inventory; normal children are launched by Python.
cargo run --release -p aiperf-runner -- --capabilities

# Offline/online Dynamo replay is a Config-v2 transport, authored through `aiperf profile`.
# (transport.type: dynosim_offline|dynosim_online). The live mocker server and the
# replay-optimize sweep stay native tools: `python -m dynamo.mocker`, Dynamo's profiler.
aiperf profile --config dynosim_offline_replay.yaml
```

`cargo run -p aiperf` is invalid: the `aiperf` package has no binary. Python projects one side-effect-free authored-v2 request; an absent pair fails closed without conversion to v1. `dag_jsonl`/`weka_trace`/`dynamo_trace` enter the runner-owned graph-input resolver once, call exactly one compiler, and never pass through a second registry. gRPC endpoints require `transport.type: grpc` with `grpc://`/`grpcs://` URLs. Offline/online replay requires `dynosim_offline`/`dynosim_online` and a feature-bearing runner.

Content server: `AIPERF_CONTENT_SERVER_ENABLED=true` + non-empty `AIPERF_CONTENT_SERVER_CONTENT_DIR`. See `docs/tutorials/content-server.md`.

Execution-engine off switch: `AIPERF_RUNTIME_ENGINE=python` (default `rust`) routes a single `aiperf profile` run through the legacy pure-Python service mesh (`SystemController` + Worker/TimingManager/RecordsManager children) instead of the native `aiperf-runner`, for A/B benchmarking the old hot path against the Rust core on an identical `BenchmarkRun`. Enum field `Environment.RUNTIME.ENGINE` in `src/aiperf/common/environment.py`; `rust` (default) rejects unknown values.
RNG backend switch: `AIPERF_RNG_BACKEND=rust_parity` (default `legacy`) swaps the seeded random substrate from Python`\s Mersenne Twister + NumPy (SHA-256 seed derivation) to a pure-Python byte-exact port of the Rust `aiperf::rng` `Pcg64` + BLAKE3 substrate (`src/aiperf/common/rng_parity/`), so seeded Python and Rust produce identical streams in tests. Enum field `Environment.RNG.BACKEND` in `src/aiperf/common/environment.py`; `legacy` (default) is unchanged. Parity is proved against committed Rust golden vectors (`rust/aiperf/examples/rng_parity_vectors.rs` -> `rust/aiperf/tests/data/rng_parity_vectors.json`, replayed by `tests/unit/common/test_rng_parity.py`).

Export-plane off switch: `AIPERF_RUNTIME_NATIVE_EXPORT=0` (default `1`) restores the legacy Python emitters (the `ExporterManager` data/console exporters, the mlflow/wandb post-run uploaders, and the OTel live-streaming sidecar) instead of the native `aiperf::export` sink plane, for A/B verification (mirrors `AIPERF_RUNTIME_ENGINE=python`). By default the native Rust sinks are the sole emitter of `profile_export_aiperf.{json,csv}`, timeslices, `server_metrics.{json,csv,parquet}`, `accuracy_results.csv`, `profile_export_console.txt`, and the OTel/MLflow/W&B network sinks: the frontend projects `cfg.export` (`rust_wire._export`) whenever the config signal is present, suppresses the live-streaming sidecar (`rust_wire._live_streaming`), and skips `native_report.export_python_compatibility_reports`. Bool field `Environment.RUNTIME.NATIVE_EXPORT` in `src/aiperf/common/environment.py`.

Unit tests use in-process axum endpoints. `tests/scheduled_real_mock.rs` retains real wall-clock library coverage; runner product coverage lives in `aiperf-runner/tests/`.

## Design specs (`specs/`)

Read for intent; verify against `rust/` for reality. Full index with status and one-liners: [`specs/README.md`](specs/README.md).

## Adding things

- **A new transport** → implement `RequestSink<YourReq>` (+ `Dispatchable` for `YourReq`) in its own crate; emit `on_classified_token` when output versus reasoning is known (otherwise `on_token`) and one terminal `on_usage` observation with optional fields; nothing in `loadgen-core` changes.
- **A new clock / execution mode** → implement `Clock`; `drive_sim` / `drive_real` already dispatch on `is_virtual()`.
- **A new graph feature** → the `executor` / `segment` / `channel_store` modules in `aiperf-graph`; keep firing-gate arithmetic byte-exact (see the graph-IR spec's parity contract).
- **A new accuracy benchmark** → for provider-neutral evaluation, implement semantics inside a pinned evaluator provider and register only an exact immutable distribution/task manifest, factory-owned public projections, required typed host operations, isolation proof, parity evidence, and product subprocess proof. Never add a Rust prompt builder, grader, or evaluator inference client. Until the benchmark's exact migration/deletion gates pass, static tasks remain in the pinned Python/Lighteval worker and stateful families remain behind their pinned Harbor, AgentLab/BrowserGym, or MCPMark `AgenticHarnessProvider` / `AgenticHarness` path.
- **A new metric** → add its `MetricSpec` in `aiperf-metrics::catalog`, implement its record/aggregate/derived computation in `store.rs` / `accumulator.rs`, and extend `RecordIngest` plus the runtime adapter only when a new raw fact is required. The native reporter consumes accumulator results without a per-metric serializer branch.
- **A new synthetic-media delivery method** → implement `SyntheticMediaPublisher`; keep codec/generation in `SyntheticMediaGenerator`, select the publisher through `SyntheticMediaGeneratorFactory`, and return exact endpoint-ready bytes to the composer.

## Keeping these docs current (MANDATORY)

These four agent files, `specs/README.md`, and root `llms.txt` are the architecture map. They go stale the instant code or specs change. **Whenever you add, modify, remove, or implement any architecture, update the map IN THE SAME CHANGE — it is part of the task, not optional follow-up.** Explicit triggers -> required edits -> verify:

| When you… | Edit | Verify |
|---|---|---|
| Add a spec to `specs/` | `specs/README.md` index row + `llms.txt` specs index + the "Design specs" section (all four agent files) | sync check |
| Modify / rename / remove a spec | the same three places (fix row, links, filename) | sync check |
| **Implement** a designed feature (designed -> built) | flip its flag in "Canonical vs aspirational" + the crate-table built/designed note + `specs/README.md` status column; delete the stale "not built" caveat | `cargo build`, sync check |
| Add / remove / rename a crate | crate topology table + the dependency-direction line (all four) + `llms.txt` crate table | `cargo build`, sync check |
| Change a seam (`Clock` / `RequestSink` / `RequestObserver` / `Dispatchable`) or a trait method | "The two seams" section + "Adding things" + `llms.txt` seam summary | `cargo build`, sync check |
| Add / change a Python CLI/config surface, runner wire feature, or build/run command | "Build, test, run" (all four) + `llms.txt` | run the command or runner subprocess proof |
| Deprecate / un-deprecate a sibling tree | the CANONICAL banner (all four + `README.md`) | sync check |
| Contradict / supersede a decision in an existing spec | append a dated `## Addendum` at the END of that spec (NEVER edit its body) + note the supersession in `specs/README.md` | — |

**Rules:**
- Edit ALL FOUR agent files together (identical body) and ALWAYS finish with `python tools/check_agent_files_sync.py` (or `make check-agent-files-sync`) — non-zero exit = bodies diverged; fix before committing.
- `specs/README.md` and `llms.txt` are NOT sync-checked but MUST move in lockstep — a spec/crate/seam change that leaves them stale is an INCOMPLETE change.
- Ground every claim in `crate/src/file.rs`. State designed-but-not-built explicitly; never describe intent as reality.
- **Never rewrite a shipped spec to contradict it.** Specs are an append-only historical record. If a decision or implementation supersedes, revises, or contradicts an already-written spec, do NOT edit that spec's body — append a dated `## Addendum — YYYY-MM-DD` at the END of the spec stating what changed, why, and which section/claim it supersedes. The original text stays; the addendum is authoritative where they conflict. Record the supersession in the `specs/README.md` status column.
- Put the doc updates in the SAME commit as the code/spec change they describe.
- **Enforced by tooling:** `tools/check_docs_current.py` fails a change that touches `specs/` or adds/removes a crate without also moving `specs/README.md` / `llms.txt` (and, for crates, the four agent files). It runs as the `check-docs-current` pre-commit hook and the "Rust Docs Guard" CI workflow; run `python tools/check_docs_current.py` locally before committing. Bypass only with `DOCS_GUARD_SKIP=1`, and justify it in the commit message.

### Agent-instruction file sync (mechanics)

`AGENTS.md`, `CLAUDE.md`, `.github/copilot-instructions.md`, and `.cursor/rules/python.mdc` (name is Python-legacy; kept so the checker's target list matches) share a **byte-identical body** below their per-tool headers. Only the header differs: the cursor file keeps its YAML frontmatter (`alwaysApply: true`) then the SPDX comment; the other three start with the SPDX comment. The body begins at the first `# AIPerf` H1. Edit all four together and verify with `python tools/check_agent_files_sync.py` (or `make check-agent-files-sync`).
