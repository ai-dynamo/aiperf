<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AIPerf

> ✅ **CANONICAL — this is THE native Rust AIPerf.** This `crates/` workspace (branch `ajc/rust`) is a from-scratch, single-process, multi-threaded tokio rewrite of the Python AIPerf LLM-inference benchmarking tool. There is **no ZMQ, no service mesh, no multiprocess credit protocol, no mmap dataset cache, no `plugins.yaml`** — those were GIL/multiprocess workarounds, not benchmarking features, and are deliberately deleted. The `aiperf-rs` and `~/projects/aiperf-rust` trees are **DEPRECATED** (mechanical 1:1 ports that reproduce the Python complexity this design removes). Design record: [`specs/`](specs/); start-here index: [`llms.txt`](llms.txt).
>
> ⛔️ The rest of this repo (`src/`, `docs/`, `Makefile`, `pyproject.toml`, `tests/`) is inherited Python-AIPerf scaffolding and is **not** the Rust tool. The Rust truth is in `crates/` + `specs/`.

## What this is

A load generator + measurement front-end for LLM inference servers: it dispatches OpenAI-compatible chat/completions requests, streams SSE, records fine-grained timing (TTFT / ITL / TPOT / e2e / throughput / goodput), and prints or serializes an aggregate report. One front-end is designed to drive **three interchangeable execution modes** over a single `{transport, clock}` seam:

1. **ONLINE-real** — real HTTP to a real server, wall clock. *(built)*
2. **ONLINE-mock** — real HTTP to a mock server (`aiperf-mock-rs`), wall clock. *(built — same code as (1), different target)*
3. **OFFLINE-mock** — in-process virtual-clock co-simulation of the mocker engine, no sockets, deterministic. *(designed — `specs/2026-07-10-steppable-clock-injected-engine-design.md`. The DES `SimClock` + `drive_sim` pump exist and are unit-tested, but the steppable-engine sink is not yet wired; `crates/aiperf-graph/src/lib.rs` says so.)*

Because it is ONE front-end, every feature (arrival patterns, datasets, multi-turn, metrics, exporters) is designed to work across all three modes for free — build once, zero drift.

## Canonical vs aspirational — the code is a walking skeleton

Ground every claim in `crate/src/file.rs`, not the specs: specs are design intent, **code is truth**. The current gaps between the north-star design and today's code:

- The north-star's `Backend` / `Engine` / `Harness` / `harness-contract` vocabulary is **aspirational**. Today's seam is `Clock` (`aiperf-clock`) + `RequestSink<R>` / `RequestObserver` / `Dispatchable` (`loadgen-core`).
- **One HTTP stack.** Both the **online CLI** path and the **graph benchmark** now dispatch over the **Clock-injected `aiperf-transport`** (hyper) client. The online path (`aiperf::http::TransportSink`, implementing `RequestSink<HttpRequest>`) runs on a `current_thread` runtime + `LocalSet` with `spawn_local`; a shared `RealClock` is the single time authority, so arrival/admit/token timestamps sit on one timeline. The legacy reqwest sink was deleted and reqwest dropped from the workspace — the former "PR2.5" gap is closed.
- **Online scheduling policy** is built for Clock-paced Poisson/Gamma/constant/request-rate and concurrency-burst arrivals, dynamic session and prefill `SlotPool` admission (prefill releases at TTFT with terminal fallback), and `StopChecker` request/duration bounds in `aiperf/src/run.rs`. User-centric, fixed-schedule, and one-pass single-turn datasets use the shared `ScheduledRuntime`; its generic `TurnRecordProcessor` hook runs downstream consumers only after normal dispatch, measurement, and credit return. Graph mode currently consumes only the shared duration gate. The full phase-runner/orchestrator remains designed.
- **Ancillary timing policy** is built: `aiperf-timing` owns Clock-driven Linear/Exponential/Poisson `RampStrategy` implementations plus `RampDriver`, warmup-aware seeded `CancellationPolicy`, and round-robin `UrlSelector`; `aiperf-transport` anchors HTTP 499 cancellation to captured request-body send completion; online issuers wire live ramps, per-turn cancellation, comma-separated endpoint selection, and turn-zero-only sticky session routing. Fixed authored schedules reject actuator ramps; user-centric supports its owned session-concurrency actuator but rejects authored-rate and absent-prefill ramps. The separate phase orchestrator and graph arrival/slot consumer remain designed.
- **Adaptive scale** is built as `aiperf-adaptive`: object-safe actuator/evaluator/step/window/controller seams, live session/prefill/rate/users actuators, Python-grounded SLA evaluation with authoritative completion-token OSL/ITL reconciliation, the `ramp_until_fail` discover/sustain/single-recovery controller, Clock-paced assessment, and schema-v2 artifacts. The online CLI wires all four controls into live issuer state. `SimClock` control is unit-tested; offline end-to-end remains blocked by the still-unwired in-process engine sink.
- **Hash-derived RNG substrate** is built as `aiperf-rng`: `RngRoot::derive` / BLAKE3 seed derivation, canonical stream-name constants, `RandomGenerator`, `HashIdRandomGenerator`, and generic `SamplingRng` / `DistributionSampler` / `SequenceSampler` seams over the five sampling distributions and sequence-length distribution. It is consumed by `aiperf-dataset` composition/samplers and the ancillary timing ramp/cancellation policies; broader scheduler/graph RNG integration remains future work.
- **Performance metrics accumulator/sweep-lines** are built in `aiperf-metrics`: 103 source-grounded metric identities plus 16 native sweep identities, NaN-sparse column storage, exact ragged ICL replay, record/aggregate/derived kernels, SLO goodput, all effective/active sweep curves, authoritative phase windows/timeslices, worker-local merge, and the typed native-v2 `Reporter`. `aiperf::metrics::NativeMetricsObserver` feeds observer timing/classification/usage and real HTTP traces from online, scheduled, adaptive, and accuracy runs; fixed schedules omit credit-relative metrics. Graph workers feed lean request/token/usage facts directly and merge in worker order. Telemetry producers and the genai-perf-v1 compatibility sink remain unbuilt.
- **Accuracy** is built end to end for the 11-benchmark native catalog: MMLU-Pro, MMLU, AIME, HellaSwag, BigBench-Hard, AIME24/25, MATH-500, GSM8K, GPQA-Diamond, and LiveCodeBench code generation. `aiperf-accuracy` owns row-independent config preflight, benchmark/source/grader registries, source-faithful prompt builders, native math/extractive graders, and sandboxed code execution; official Hugging Face/DeepEval providers feed the unified dataset/segment/tokenizer path. Accuracy owns **no dispatcher or HTTP path**: `aiperf::accuracy::AccuracyDataset` lowers problems into ordinary conversations and `AccuracyRecordProcessor` grades terminal results through the generic `ScheduledRuntime::TurnRecordProcessor` hook after the normal `TurnDispatcher`/observer/metrics/credit-return path. It accumulates real correlation ids, joins performance metrics, and emits native-v2 per-record JSON plus the inherited accuracy-summary CSV. Telemetry-backed energy joins remain unavailable until telemetry producers exist.
- **Dataset/segment unified store is built end to end.** `aiperf-dataset` owns the full loader registry, composition, token-keyed prefix-dependent dense-handle storage, exact raw/message materialization, decoded HF and synthetic media, context reconstruction, sampler factories, remote fetch/cache, and tiktoken/Hugging Face tokenizer traits. Native fixed-schedule, user-centric, one-pass accuracy, and Graph-IR paths share the store; loader-preferred random/sequential/shuffle policy is honored online. The default paced `SkeletonWorkload` remains a fixed synthetic single-turn source, and ordinary `--input-file` use is currently scoped to fixed/user-centric CLI paths.

When something is designed-but-not-built, this file says so. Do not assume a spec feature exists in the code.

## Crate workspace (`crates/`)

| Crate | Purpose | Key files |
|---|---|---|
| `aiperf-clock` | The `{clock}` seam. `Clock` trait (`now_ns` / `sleep` / `is_virtual`; the virtual-time controls `next_event_time`/`advance_to` are inherent methods on `SimClock`, not the trait, so real clocks carry no no-op stubs); `SimClock` = integer-ns discrete-event `BinaryHeap` keyed `(at_ns, seq_no)` (deterministic same-instant tie-break); `RealClock` = monotonic `Instant` + Linux `timerfd`/`AsyncFd` ns sleeps (tokio fallback off-Linux and on syscall failure). | `clock.rs`, `sim_clock.rs`, `real_clock.rs` |
| `loadgen-core` | Transport-neutral dispatch/measure seam + the collector. `Dispatchable`, `RequestSink<R>`, local-loop `RequestObserver` (no `Send`/`Sync`; f64-ms timestamps; optional `ObservedTokenKind` classification; terminal `ObservedUsage` with optional fields), `TraceCollector` → `TraceSimulationReport`. Zero engine/KV/HTTP deps. | `sink.rs`, `collector.rs` |
| `aiperf-timing` | Clock-native scheduling policy. `IntervalGenerator` arrivals; debt-draining `SlotPool`; `StopChecker`; user-centric schedule math; Clock-driven `RampStrategy`/`RampDriver` with Linear/Exponential/Poisson curves; seeded warmup-aware `CancellationPolicy`; and pluggable `UrlSelector`. The online issuer consumes all applicable policies; graph currently consumes the duration gate, with its arrival/slot consumers next. | `intervals.rs`, `slots.rs`, `stop.rs`, `user_centric.rs`, `ramping.rs`, `cancellation.rs`, `url_selection.rs` |
| `aiperf-adaptive` | Transport-neutral SLA-control leaf. `ControlActuator` adapters mutate live session/prefill slots, request rate, or user target; `TumblingWindowSampler` joins arrival/admit/token/usage/terminal observations; `SlaEvaluator` feeds `StepPolicy` and `RampUntilFailController`; `AdaptiveScale` is Clock-paced; typed schema-v2 events/summary serialize through `AdaptiveArtifactSink`. | `actuator.rs`, `window.rs`, `sla.rs`, `step.rs`, `controller.rs`, `runtime.rs`, `artifacts.rs` |
| `aiperf-rng` | Hash-derived reproducibility substrate: `RngRoot::derive` + BLAKE3 seed derivation, canonical stream names, one `Pcg64`-backed `RandomGenerator`, alloc-free `HashIdRandomGenerator` reseeding, and generic `SamplingRng` / `DistributionSampler` / `SequenceSampler` extension seams. Weighted distributions cache validated cumulative weights once; the fixed-seed profile canary pins internal reproducibility. Leaf crate consumed by `aiperf-dataset` and ancillary timing policies; broader scheduler/graph consumers remain. | `derive.rs`, `namespace.rs`, `generator.rs`, `hash_id.rs`, `dist.rs` |
| `aiperf-dataset` | Unified dataset/segment substrate: loader/composer/sampler/tokenizer/fetch/request traits; token-keyed prefix-dependent blake3 segments behind dense `Handle`s; exact message/raw-body/tool/header/query/media preservation; decoded HF + synthetic media; context reconstruction; sequential/shuffle/random factory registry. Shared by CLI conversations, accuracy, and Graph-IR. | `loader/`, `compose.rs`, `segment.rs`, `materialize.rs`, `request.rs`, `sampler.rs`, `tokenizer.rs`, `fetch.rs` |
| `aiperf-metrics` | IO-free performance/accuracy engine: 119-row validated catalog; NaN-sparse `ColumnStore` + exact `RaggedSeries`; `MetricsAccumulator` formulas, SLOs, phase windows, timeslices, worker merge; effective/active/ICL sweep-lines and duration-weighted stats; accuracy accumulator/analyzer; typed native-v2 `Reporter`. Telemetry collectors and genai-perf-v1 compatibility export remain external consumers. | `store.rs`, `accumulator.rs`, `sweepline/`, `catalog.rs`, `accuracy.rs`, `report.rs` |
| `aiperf-accuracy` | IO-light accuracy benchmark layer: dataset-source, benchmark/config-preflight, prompt/generation, grader, and code-executor traits; 11 native benchmark plugins and 9 native graders, including pinned MMLU-Pro and sandboxed LiveCodeBench execution. Runtime dispatch and report joins live in `aiperf`. | `benchmark.rs`, `benchmarks/`, `source.rs`, `grader/`, `mmlu_pro.rs`, `registry.rs` |
| `aiperf-core` | Shared measurement layer (no HTTP client of its own). OpenAI SSE chunk types (`sse::ChatChunk` + `delta_text`), the shared chat request-body builder (`chat::chat_request_body`), and `CollectorObserver` — a pure recorder into `TraceCollector`; callers supply Clock-derived ms timestamps. | `sse.rs`, `chat.rs`, `observer.rs` |
| `aiperf-transport` | Rust-native, **Clock-injected** HTTP transport on hyper 1.x's low-level conn API (not reqwest). h1 + h2c prior-knowledge + UDS; rustls; fine-grained DNS/TCP/TLS/send/recv trace timing; source-faithful live SSE first-token filtering; captured full-body `SendCompletion`; Clock-scheduled HTTP 499 cancellation after send; connection-reuse strategies. Every timestamp via `Clock`. | `client/`, `transport/`, `sse/`, `models/`, `config/` |
| `aiperf-graph` | Graph-IR async-dataflow engine: multi-turn DAG conversations with fan-in/out + firing-gate timing over the unified dataset store; `runtime` (`drive_sim` / `drive_real` on `current_thread` + `LocalSet`); and transport throughput benches. Each graph worker owns a native `MetricsAccumulator`; scoped workers return local stores that merge once in worker order. Offline co-sim sink intentionally not wired yet. | `runtime.rs`, `executor.rs`, `materialize.rs`, `transport_bench.rs` |
| `aiperf` | The CLI binary. `--mode online` supports Clock-paced concurrency/request-rate, prefill admission, user-centric/fixed/single-turn-dataset schedules, the registry-driven native accuracy catalog, adaptive scale, linear phase ramps, post-send request cancellation, and comma-separated sticky endpoint routing over `aiperf-transport`; `--mode graph` runs Graph-IR throughput. `ScheduledRuntime` owns generic terminal record processors; accuracy only prepares data and consumes returned records. Every path's `--json` writes the unified native-v2 report; accuracy additionally supports the inherited summary CSV. `clap`; mimalloc; tracing. | `main.rs`, `ancillary.rs`, `metrics.rs`, `accuracy.rs`, `accuracy_dataset.rs`, `adaptive.rs`, `http.rs`, `run.rs`, `scheduled.rs`, `scheduler.rs`, `user_centric.rs`, `fixed_schedule.rs`, `report.rs` |

Dependency direction: `aiperf` → {`aiperf-accuracy`, `aiperf-adaptive`, `aiperf-core`, `aiperf-dataset`, `aiperf-graph`, `aiperf-metrics`, `aiperf-timing`, `aiperf-transport`, `aiperf-clock`, `loadgen-core`}; `aiperf-accuracy` → `aiperf-metrics`; `aiperf-adaptive` → {`aiperf-clock`, `aiperf-metrics`, `aiperf-timing`, `loadgen-core`}; `aiperf-dataset` → {`aiperf-clock`, `aiperf-endpoints`, `aiperf-rng`, `aiperf-transport`}; `aiperf-graph` → {`aiperf-core`, `aiperf-dataset`, `aiperf-metrics`, `aiperf-timing`, `aiperf-transport`, `aiperf-clock`, `loadgen-core`}; `aiperf-timing` → {`aiperf-clock`, `aiperf-rng`}; `aiperf-core` → `loadgen-core`; `aiperf-transport` → `aiperf-clock`. `loadgen-core`, `aiperf-clock`, `aiperf-timing`, `aiperf-rng`, `aiperf-metrics`, `aiperf-accuracy`, and `aiperf-adaptive` have no transport/backend dependency; `aiperf-metrics` remains IO-free and runtime-neutral. Workspace: `edition = "2024"`, `resolver = "3"`.

## The two seams (the whole architecture)

- **`{clock}`** (`aiperf-clock`): `RealClock` vs `SimClock` behind one `Clock` trait. `is_virtual()` selects the `drive_real` (tokio reactor drives) vs `drive_sim` (idle-pump: poll the `LocalSet` to quiescence draining all same-instant work → `advance_to(next_event_time)` waking heap-ordered sleepers → repeat) driver over the *same* executor. Virtual time is integer ns with an `(at_ns, seq_no)` deterministic tie-break — **never `tokio::time`** (its 1 ms timer wheel destroys µs/ns firing gates).
- **`{transport}`** (`loadgen-core::sink`): `RequestSink<R>::dispatch` drives a `Dispatchable` request to terminal and emits `on_arrival` / `on_admit` / `on_token` (or `on_classified_token` with `ObservedTokenKind::{Output,Reasoning}`) / terminal `on_usage(ObservedUsage)` / `on_terminal` through a `RequestObserver`. Classification defaults to `on_token`; usage defaults to a no-op and keeps unreported counts as absent fields. `RequestObserver` has no `Send`/`Sync` supertraits: each thread-per-core worker owns a local observer graph in `Rc`/`RefCell`; cross-thread consumers may still provide a thread-safe implementation. Real HTTP, mock HTTP, and (designed) in-process engine co-sim are all `RequestSink` impls behind one observer. TTFT is the first token callback; sinks emit no separate first-token event.

## Extensibility & porting discipline (non-negotiable)

- **Every extension point is a trait.** Anything that could ever have a second implementation — a transport, a clock, a request/response shape, an arrival pattern, a dataset loader, a sampler, a segment store, a metric accumulator, an analyzer, an exporter, an endpoint dialect, a tokenizer, a scheduling policy — MUST be an implementable `trait` (object-safe where it crosses a `dyn` boundary; generic where it is hot-path monomorphized) with at least one concrete impl behind it. Never hardcode a concrete type where a future variant is conceivable. If you are `match`-ing on an enum of "kinds" or branching on a string mode, that is a trait waiting to be extracted. In-tree precedent: `Clock`, `RequestSink<R>` / `RequestObserver` / `Dispatchable`, `SegmentStore` / `PromptMaterializer`, `GraphSink`.
- **Always design ahead.** When you add code, add the seam the next plausible requirement will need — name the trait, take the trait (not the concrete) in signatures, thread the injection point — even if you ship exactly one impl today. The three-modes-for-free property only survives if features are written against the `{transport, clock}` seams, never against a specific backend/clock/transport; a feature that works in only one mode is a design bug. Note the extension you are leaving open in a `//!` / `///` doc comment.
- **Read the ENTIRE Python source before porting ANYTHING.** Before porting a behavior, read the WHOLE Python file end-to-end AND every file it meaningfully touches (its imports, the models it builds, the callers that consume its output, the tests that pin it). Never port from a docstring, a grep hit, a snippet, or a spec paragraph — those omit the earned-in-blood edge cases (SSE fast-paths, backward usage walks, finish/usage reconciliation, firing-gate rounding), which live in the parts of the file you assumed you could skip. Cite the exact Python `path:line` you ported from in the Rust `//!` / `///` docs, and guard it with a parity test wherever byte-exactness matters.

## Coding standards (Rust)

- **SPDX header** (`// SPDX-FileCopyrightText …` + `// SPDX-License-Identifier: Apache-2.0`) atop every source file. `//!` module docs; `///` doc comment on every public item.
- **Thread-per-core**, not work-stealing, on the hot path: N OS threads, each a `current_thread` tokio runtime + `LocalSet`; per-trace state is `Rc`/`RefCell`, futures are `!Send`, tasks are `spawn_local`; parallelism = many traces across threads. See `aiperf-graph/src/runtime.rs`, `transport_bench.rs`. The online `run.rs` path follows the same `!Send` model on a single `current_thread` runtime + `LocalSet`, with `Rc` observers and dynamic `SlotPool` admission.
- **All time through `Clock`** in the clock-aware crates (`aiperf-transport`, `aiperf-graph`, and the CLI's online path): never `Instant::now()`, `SystemTime::now()`, or raw `tokio::time` for measurement or firing gates. `aiperf-core` is now a pure measurement layer and owns no clock.
- **No `Arc`/`Mutex` on hot paths.** Accumulate lock-free per-thread / per-worker and merge once at the end (the graph bench keeps a per-worker accumulator and merges at the join) — never contend a shared collector lock per token on the throughput path.
- **Content-addressed segments**: blake3, prefix-dependent (fold the parent id into the hash so shared prefixes dedup and identical text under different prefixes stays distinct); materialize = clone/concat pre-serialized bytes, never re-serialize.
- **mimalloc** global allocator (per-request alloc churn in the graph executor + streaming client was the top profiled hotspot).
- **Loopback benchmarking**: the hyper transport never consults the ambient `HTTP_PROXY`, so localhost traffic is not proxied (an ambient proxy 405s localhost and tanks throughput).
- **SSE**: buffer raw bytes, split lines on the byte buffer, UTF-8-decode only complete lines (a multibyte char may straddle a TCP chunk boundary).
- **Authoritative token counts**: request `stream_options.include_usage` so `usage.completion_tokens` is returned; the HTTP sink emits it through `RequestObserver::on_usage`. Adaptive windows use the authoritative completion count for OSL and the `(last−first)/(osl−1)` ITL denominator. Aggregate per-token timing still counts one output token per non-empty SSE delta; full collector-wide reconcile-to-usage remains unimplemented.
- **Errors**: `anyhow` in the binary/app layer; library crates use plain error enums with hand-written `Display` (no `thiserror`).
- **`clap` derive** for CLI; `serde` / `serde_json` for wire. Prefer a direct-serialized request body on the hot path (skip an intermediate `serde_json::Value`).
- **NaN/Inf discipline**: numeric metric values crossing a serialization boundary must be finite or explicitly absent.
- **Comments explain *why*, never *what*.** No emojis in code. Read the actual code, never trust the docstrings or comments.

## Build, test, run

The (inherited-Python) `Makefile` has **no** cargo targets — use cargo directly:

```bash
cargo build                  # debug build of the whole workspace
cargo build --release        # optimized — use for any throughput number
cargo test                   # all unit tests (self-contained: in-process axum mock, no external server)
cargo test -p aiperf-graph   # one crate
cargo clippy --all-targets   # lints
cargo fmt                    # format (rustfmt)
```

Run the tool:

```bash
# online (default): closed-loop concurrency
cargo run --release -p aiperf -- [BASE_URL] [MODEL] \
  --concurrency 16 --requests 100 --isl 128 --osl 128 [--json out.json]

# native accuracy (official cached dataset or --accuracy-dataset DIR)
cargo run --release -p aiperf -- [BASE_URL] [MODEL] \
  --accuracy-benchmark mmlu-pro --accuracy-tasks math \
  --accuracy-max-problems 100 --accuracy-tokenizer builtin \
  [--accuracy-csv accuracy.csv] [--json out.json]

# online ancillary timing: comma-separated endpoints, live ramps, post-send cancellation
cargo run --release -p aiperf -- http://host-a:8000,http://host-b:8000 [MODEL] \
  --request-rate 100 --concurrency 32 --prefill-concurrency 8 \
  --request-rate-ramp-duration 30 --concurrency-ramp-duration 20 \
  --prefill-concurrency-ramp-duration 20 \
  --request-cancellation-rate 5 --request-cancellation-delay 0.5

# online user-centric: per-user pacing, churn, and multi-turn continuations
cargo run --release -p aiperf -- [BASE_URL] [MODEL] \
  --user-centric-rate 8 --num-users 4 --turns 4 --sessions 20 \
  --think-time-ms 50 [--input-file conversations.jsonl] \
  [--input-format multi_turn] [--tokenizer builtin] \
  [--dataset-option KEY=JSON] \
  [--timing-json timing.json] [--json out.json]

# online fixed schedule: absolute trace replay (auto-anchored by default)
cargo run --release -p aiperf -- [BASE_URL] [MODEL] \
  --fixed-schedule --input-file trace.jsonl \
  [--input-format mooncake_trace] [--tokenizer builtin] \
  [--dataset-option KEY=JSON] \
  [--timing-json timing.json] [--json out.json]

# online adaptive concurrency (repeat --adaptive-scale-sla to AND filters)
cargo run --release -p aiperf -- [BASE_URL] [MODEL] \
  --duration 300 --concurrency 64 --adaptive-scale \
  --adaptive-scale-strategy-type ramp_until_fail \
  --adaptive-control-variable concurrency --adaptive-control-min 1 \
  --adaptive-control-max 64 --adaptive-assessment-period 30 \
  --adaptive-sustain-duration 60 \
  --adaptive-scale-sla request_latency:p95:le:1000

# graph: Graph-IR E2E streaming throughput (multi-turn DAG conversations)
cargo run --release -p aiperf -- --mode graph [BASE_URL] [MODEL] \
  --turns 4 --instances 400000 --workers <cores> --concurrency 64 --osl 1 [--http2]
```

In every mode, `--json PATH` writes one unified native-v2 report with
metrics keyed by name and type-specific distribution/scalar/counter series.

Unit tests spin up an in-process axum OpenAI-SSE mock (`test_util::spawn_mock`). `tests/scheduled_real_mock.rs` launches the external `aiperf-mock-rs` binary when `AIPERF_MOCK_RS_BIN` (or a discoverable sibling/PATH binary) is available and validates both scheduled library APIs and the compiled CLI with real wall-clock timing. The graph `transport_bench` throughput path also validates against that external binary (a `unix:/path` base URL uses a Unix-domain socket, which is what pushes co-located throughput past 1M req/s).

## Design specs (`specs/`)

Read for intent; verify against `crates/` for reality. Full index + one-liners: [`specs/README.md`](specs/README.md).

- **North star & rationale**
  - `2026-07-10-shared-rust-architecture-northstar.md` — the target abstraction: one front-end, three orthogonal axes (time / backend / workload), a ~120-line neutral contract. *Aspirational vocabulary; not the current symbol set.* Addendum (2026-07-11): current built symbols are `Clock` + `RequestSink<R>` / `RequestObserver` / `Dispatchable`; virtual controls are inherent on `SimClock`.
  - `2026-07-10-aiperf-rust-port-exact-vs-redo-ledger.md` — **read first for scope.** Per-concept port-exact vs redo-cleaner vs throw-away rulings; the credit-*policy* trap (delete the protocol, keep the policy). Addendum (2026-07-11): online/offline parity means shared code path + report schema, not byte-identical real-vs-sim metric values; policy is realized through `Workload`/`SlotPool`/`RatePool`/`Gate`.
  - `2026-07-10-unified-graph-runtime-design.md` — **the realization design (read after the ledger).** Every load mode (rate/concurrency/user-centric/fixed-schedule/adaptive/DAG) reduces to one dispatch verb on the clock-scheduled graph executor; strategies become `Workload` schedule generators, not loops. Supersedes the scheduling-policy sketch. Addendum (2026-07-11): RNG seed derivation is BLAKE3, and implementation against today's crates should translate north-star seam names to current `RequestSink` / `RequestObserver` symbols.
  - `2026-07-10-aiperf-rust-coverage-gap-ledger.md` — research synthesis of unspec'd bodies (endpoint/exporter zoo, config-v2 hidden algorithms, timing-engine depth, presentation/API/plot) + cross-cutting scope decisions. Addendum (2026-07-11): metrics, telemetry, and RNG gaps are now covered by dedicated specs/addenda.
- **Architecture seams**
  - `2026-07-10-steppable-clock-injected-engine-design.md` — the `{clock}` seam + the OFFLINE-mock steppable-engine boundary (the missing third mode). Addendum (2026-07-11): its `lib/aiperf` + dynamo `lib/mocker` framing is historical lineage; translate concepts to the standalone `crates/` workspace and current `Clock` + `RequestSink` seam.
  - `2026-07-10-aiperf-transport-rust-port-design.md` — the Clock-injected hyper transport (realized in `aiperf-transport`). Addendum (2026-07-11): cancellation-after-send, full h2 reuse semantics, and the full aiohttp-style trace field set are design targets where current code is narrower.
  - `2026-07-10-aiperf-rust-dataset-segment-seam-design.md` — **built end to end + implementation addendum** as `aiperf-dataset`: the complete loader→compose→dense-handle store→sampler→materializer pipeline is shared by native CLI/accuracy and Graph-IR, including exact raw replay, decoded/synthetic media, all context modes, dispatch fields, real correlation ids, and DAG metadata. The addendum resolves the four open decisions and records executable proof.
- **Subsystem designs**
  - `2026-07-10-aiperf-rust-scheduling-policy-sketch.md` — early sketch of the credit-*policy* `Scheduler`. **Superseded by `2026-07-10-unified-graph-runtime-design.md`**, which realizes the policy as `Workload`/`SlotPool`/`RatePool`/`Gate` on the graph executor.
  - `2026-07-11-aiperf-rust-request-rate-multiturn-design.md` — source-grounded realization of request-rate multi-turn: a **single-loop credit issuer** emitting **one turn per rate interval, continuation-priority** (NOT conversation arrivals), gated by a session `SlotPool` (turn-0→final) + prefill `SlotPool` (every turn→TTFT), bounded by `StopChecker`, turns materialized from the segment pool, think-time deferred via `Clock::sleep`. Read end-to-end from `request_rate.py`/`issuer.py`/`concurrency.py`/`stop_conditions.py`/`callback_handler.py`/`credit_counter.py`. Carries the two-plane throughput framing (control-loop 6.5–20 M/s never the bottleneck; HTTP data plane fans across cores; handoff ~1.7 M/s ≫ any policy rate). Most primitives exist in `aiperf-timing`; unbuilt core = continuation queue + two-source issue loop + conversation source over the segment pool.
  - `2026-07-11-aiperf-rust-user-centric-fixed-schedule-design.md` — **built + addendum**: `ConversationSource`, the Clock-backed `LocalTaskScheduler`, shared `ScheduledRuntime`, `UserPool`/`UserCentricWorkload`, and `FixedScheduleSource`/`FixedScheduleWorkload` implement per-user steady-state seeding/churn/continuation pacing and absolute trace replay. The CLI exposes dataset or synthetic multi-turn user-centric runs plus dataset-required fixed replay and detailed timing JSON. Exact `SimClock` tests and real `aiperf-mock-rs` library/CLI tests cover ordering, pacing, drain behavior, timing error, TTFT, and reply splicing; the global offline engine sink remains unwired.
  - `2026-07-11-aiperf-rust-phase-runner-orchestrator-design.md` — the phase driver ABOVE the credit issuer: `PhaseLifecycle` state machine (CREATED→STARTED→SENDING_COMPLETE→COMPLETE + orthogonal cancel flag), grace/duration-timeout/cancel-drain/force-complete escalation, warmup→profiling sequencing with **seamless** overlap + cross-phase debt-drain. Source-grounded from `phase/runner.py` (786 lines) + `phase_orchestrator.py` + `phase/publisher.py` + `manager.py` + `config.py`. **Deletes the ZMQ/IPC scaffolding** (`PhasePublisher`, the `TimingManager` service, `wait_for_workers`) → direct `PhaseObserver`/`RequestObserver` trait calls on one `!Send` loop. New seams: `PhaseRunner`, `PhaseOrchestrator`, `PhaseObserver`.
  - `2026-07-11-aiperf-rust-adaptive-scale-design.md` — **built in `aiperf-adaptive` + addendum**: object-safe actuator/evaluator/step/window/controller traits, all four live actuators, Python-grounded SLA math including authoritative completion-token OSL/ITL reconciliation, monotone `ramp_until_fail` discover/sustain/single-recovery control, Clock-paced assessment, and schema-v2 artifacts. The online CLI wires concurrency, prefill concurrency, request rate, and user-centric users into live issuer state. `SimClock` control is unit-tested; offline end-to-end remains gated by the unwired engine sink.
  - `2026-07-11-aiperf-rust-ancillary-timing-policy-design.md` — **built + addendum**: Clock-driven Linear/Exponential/Poisson ramps, seeded warmup-aware Bernoulli fixed-delay cancellation, captured send-complete HTTP 499 aborts, and turn-zero round-robin with sticky session pinning are implemented and wired into online issuers. The CLI exposes session/prefill/rate ramp durations, cancellation rate/delay, and comma-separated endpoints. Fixed schedules reject ramps; user-centric accepts its session ramp but rejects authored-rate/absent-prefill actuators. The phase orchestrator and graph arrival/slot consumer remain companion-spec work.
  - `2026-07-11-aiperf-rust-dag-branch-orchestrator-design.md` — **superseded by the `aiperf-graph` async-dataflow engine** (lineage + reconciliation, not a build plan). The Python ~1000-line FORK/SPAWN credit-side orchestrator collapses into the graph executor: fan-out = out-edges, join gating = `ChannelRequirement.count` with static producer accounting, and sticky-routing / drain-observer / future-active-gate / the spawn-first + drain-after-return races are **deleted credit-protocol artifacts, not ported**. Residual = graph-build lowering (branch metadata → nodes/edges) + FORK/SPAWN materialization + session-cap wiring + whole-run FAIL_FAST; no orchestrator to rebuild.
  - `2026-07-10-aiperf-rust-accuracy-accumulator-design.md` — **built + addenda**: first-class records/accumulator/analyzer in `aiperf-metrics`; 11 native benchmark plugins, 9 graders, official cached sources, unified dataset/tokenizer lowering, downstream grading on the normal scheduled runtime (no accuracy dispatcher), native-v2 per-record output, and accuracy CSV through `aiperf-accuracy` + `aiperf::accuracy`. Telemetry energy joins await producers.
  - `2026-07-10-aiperf-rust-metrics-accumulator-sweepline-design.md` — **built + addenda** in `aiperf-metrics`: columnar/ragged storage, full record/aggregate/derived engine, SLO goodput, effective/active and ICL sweep-lines, duration-weighted statistics, phase windowing/timeslicing, per-worker merge, runtime adapters in every built mode, and native-v2 `Reporter`. Telemetry producers and genai-perf-v1 compatibility remain separate consumers.
  - `2026-07-10-aiperf-rust-metric-catalog-appendix.md` — **built + addendum**: 103 inherited Python metric identities plus 16 native sweep identities with exact metadata, dependency validation/fingerprint, formulas, and the per-metric scars. Telemetry-injected rows remain absent until their producers supply values.
  - `2026-07-10-aiperf-rust-telemetry-accumulators-design.md` — GPU / server-metrics / network-RTT as side-channel `Accumulator`s reusing the metrics seam: DCGM fields, server fallback/auto-disable, polynomial histogram percentiles, TCP-connect RTT. Addenda: 2026-07-10 phase-boundary snapshots supersede scrape reconstruction; 2026-07-11 keeps dependency direction from telemetry producers toward the metrics seam, not from core metrics into telemetry collectors.
  - `2026-07-11-aiperf-rust-exporters-overhaul-design.md` — **partially built + addendum**: the IO-free native-v2 `Reporter` and application JSON writer are the default across online, scheduled, accuracy, and graph modes; the inherited task/overall accuracy-summary CSV is built. General metric CSV, genai-perf-v1 compatibility, warnings/insights, console replay, and timed uploaders remain designed.
  - `2026-07-11-aiperf-rust-endpoints-design.md` — **faithful port** of the endpoint layer (crate `aiperf-endpoints`): the `Endpoint` trait (build request body + parse response into records) + input-side ISL accounting. Carries the parse scars — chat response precedence `reasoning>content+tool_calls>tool_calls>content` (~18% OSL-undercount mixed-emit fix), tool-call streaming reassembly (missing-index→`len(dict)`, modern name-overwrite/args-concat vs legacy), the responses SSE event map (`function_call_arguments.delta`→ToolCall, ~64% of agentic turns) + replay-unsafe filter + dedup-by-id union, the three malformed-response policies (embeddings raises; chat/completions degrade). The input-ISL walk's tool-schema `orjson.dumps(parameters)` byte-parity is the #1 risk. Capability-flag→lifecycle table + 16-type registry; tier-2 vendor endpoints deferred.
  - `2026-07-10-aiperf-rust-rng-derive-system-design.md` — hash-derived RNG substrate: order-independent `blake3(f"{root}:{id}")[:8]`→u64 seed derivation (BLAKE3 locked, not Python-matching) + canonical namespace constants + `HashIdRandomGenerator` (per-`(trace_id, hash_id)` reseed) + the `RandomGenerator` contract and generic sampler traits. Ruling: NO cross-language byte parity — one deterministic PRNG (`Pcg64`) + `rand_distr` with faithful distribution/edge-case semantics; internal reproducibility + order-independence, lock-free per-thread + alloc-free reseed. Built as leaf crate `aiperf-rng`; the Rust-internal profile canary pins its stream; dataset composition/samplers and ancillary timing policies consume it, while broader scheduler/graph integration remains.
  - `2026-07-09-graph-ir-rust-port-design.md` — byte-exact Graph-IR dataflow port (partly realized in `aiperf-graph`). Addendum (2026-07-11): standalone/offline-only framing is superseded by the current `LocalSet` + `drive_sim`/`drive_real` + HTTP-dispatch path.
- **Historical precursors** — describe a *different* working tree built *on* dynamo's `lib/mocker`; the current workspace is standalone (`loadgen-core` extracted, no dynamo dependency). Kept for lineage.
  - `2026-07-09-dynamo-aiperf-shared-core-design.md`, `2026-07-09-dynamo-aiperf-request-rate-tokenizer-design.md`.

## Adding things

- **A new transport** → implement `RequestSink<YourReq>` (+ `Dispatchable` for `YourReq`) in its own crate; emit `on_classified_token` when output versus reasoning is known (otherwise `on_token`) and one terminal `on_usage` observation with optional fields; nothing in `loadgen-core` changes.
- **A new clock / execution mode** → implement `Clock`; `drive_sim` / `drive_real` already dispatch on `is_virtual()`.
- **A new graph feature** → the `executor` / `segment` / `channel_store` modules in `aiperf-graph`; keep firing-gate arithmetic byte-exact (see the graph-IR spec's parity contract).
- **A new accuracy benchmark** → implement `AccuracyBenchmark` (including row-independent `validate_config`), register its benchmark and default `Grader` factories in `aiperf-accuracy`, and add a `BenchmarkDatasetProvider` only when official remote acquisition is available. Prompt/data parsing stays transport-neutral; `AccuracyDataset` lowers it into ordinary conversations and `AccuracyRecordProcessor` consumes terminal results. Never add an accuracy dispatcher or HTTP client.
- **A new metric** → add its `MetricSpec` in `aiperf-metrics::catalog`, implement its record/aggregate/derived computation in `store.rs` / `accumulator.rs`, and extend `RecordIngest` plus the runtime adapter only when a new raw fact is required. The native reporter consumes accumulator results without a per-metric serializer branch.

## Keeping these docs current (MANDATORY)

These four agent files, `specs/README.md`, and root `llms.txt` are the architecture map. They go stale the instant code or specs change. **Whenever you add, modify, remove, or implement any architecture, update the map IN THE SAME CHANGE — it is part of the task, not optional follow-up.** Explicit triggers -> required edits -> verify:

| When you… | Edit | Verify |
|---|---|---|
| Add a spec to `specs/` | `specs/README.md` index row + `llms.txt` specs index + the "Design specs" section (all four agent files) | sync check |
| Modify / rename / remove a spec | the same three places (fix row, links, filename) | sync check |
| **Implement** a designed feature (designed -> built) | flip its flag in "Canonical vs aspirational" + the crate-table built/designed note + `specs/README.md` status column; delete the stale "not built" caveat | `cargo build`, sync check |
| Add / remove / rename a crate | crate topology table + the dependency-direction line (all four) + `llms.txt` crate table | `cargo build`, sync check |
| Change a seam (`Clock` / `RequestSink` / `RequestObserver` / `Dispatchable`) or a trait method | "The two seams" section + "Adding things" + `llms.txt` seam summary | `cargo build`, sync check |
| Add / change a CLI mode, flag, or build/run command | "Build, test, run" (all four) + `llms.txt` | run the command |
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
