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
- **Scheduling policy** (arrival patterns, session-vs-request slots, prefill-release-on-TTFT, phase-scoped metrics, `--request-count` recycle) is **designed** (`specs/…scheduling-policy-sketch.md`) but **not built** — `run.rs` is still a naive `Semaphore` loop (`admit == dispatch`).
- **Dataset/segment unified store, accuracy accumulator, tokenizer-exact prompts, request-rate mode**: designed, not implemented. `SkeletonWorkload` emits fixed synthetic requests; there is no tokenizer and no request-rate mode.

When something is designed-but-not-built, this file says so. Do not assume a spec feature exists in the code.

## Crate workspace (`crates/`)

| Crate | Purpose | Key files |
|---|---|---|
| `aiperf-clock` | The `{clock}` seam. `Clock` trait (`now_ns` / `sleep` / `is_virtual`; the virtual-time controls `next_event_time`/`advance_to` are inherent methods on `SimClock`, not the trait, so real clocks carry no no-op stubs); `SimClock` = integer-ns discrete-event `BinaryHeap` keyed `(at_ns, seq_no)` (deterministic same-instant tie-break); `RealClock` = monotonic `Instant` + Linux `timerfd`/`AsyncFd` ns sleeps (tokio fallback off-Linux and on syscall failure). | `clock.rs`, `sim_clock.rs`, `real_clock.rs` |
| `loadgen-core` | Transport-neutral dispatch/measure seam + the collector. `Dispatchable`, `RequestSink<R>`, `RequestObserver` (f64-ms timestamps), `TraceCollector` → `TraceSimulationReport`. Zero engine/KV/HTTP deps. | `sink.rs`, `collector.rs` |
| `aiperf-core` | Shared measurement layer (no HTTP client of its own). OpenAI SSE chunk types (`sse::ChatChunk` + `delta_text`), the shared chat request-body builder (`chat::chat_request_body`), and `CollectorObserver` — a pure recorder into `TraceCollector`; callers supply Clock-derived ms timestamps. | `sse.rs`, `chat.rs`, `observer.rs` |
| `aiperf-transport` | Rust-native, **Clock-injected** HTTP transport on hyper 1.x's low-level conn API (not reqwest). h1 + h2c prior-knowledge + UDS; rustls; fine-grained DNS/TCP/TLS/send/recv trace timing; cancellation; connection-reuse strategies. Every timestamp via `Clock`. | `client/`, `transport/`, `sse/`, `models/`, `config/` |
| `aiperf-graph` | Graph-IR async-dataflow engine: multi-turn DAG conversations with fan-in/out + firing-gate timing. Content-addressed `SegmentStore` (blake3, prefix-dependent); `materialize` (static segment + predecessor-reply splice); `executor` / `scheduler` / `channel_store` / `reducers`; `runtime` (`drive_sim` / `drive_real` on `current_thread` + `LocalSet`); `bench` (shared workload scaffolding — segment pool, `BenchConfig`, server resolution) + `transport_bench` (aiperf-transport; UDS → >1M req/s). Offline co-sim sink intentionally not wired yet. | `runtime.rs`, `executor.rs`, `segment.rs`, `transport_bench.rs` |
| `aiperf` | The CLI binary. `--mode online` (default: closed-loop concurrency over `aiperf-transport`, on a `current_thread` runtime + `LocalSet`) and `--mode graph` (Graph-IR throughput). `clap` args; mimalloc global allocator; ai-dynamo-style `tracing` logging. | `main.rs`, `http.rs`, `run.rs`, `workload.rs`, `report.rs`, `logging.rs` |

Dependency direction: `aiperf` → {`aiperf-core`, `aiperf-graph`, `aiperf-transport`, `aiperf-clock`, `loadgen-core`}; `aiperf-graph` → {`aiperf-core`, `aiperf-transport`, `aiperf-clock`, `loadgen-core`}; `aiperf-core` → `loadgen-core`; `aiperf-transport` → `aiperf-clock`. `loadgen-core` and `aiperf-clock` are leaves. Workspace: `edition = "2024"`, `resolver = "3"`.

## The two seams (the whole architecture)

- **`{clock}`** (`aiperf-clock`): `RealClock` vs `SimClock` behind one `Clock` trait. `is_virtual()` selects the `drive_real` (tokio reactor drives) vs `drive_sim` (idle-pump: poll the `LocalSet` to quiescence draining all same-instant work → `advance_to(next_event_time)` waking heap-ordered sleepers → repeat) driver over the *same* executor. Virtual time is integer ns with an `(at_ns, seq_no)` deterministic tie-break — **never `tokio::time`** (its 1 ms timer wheel destroys µs/ns firing gates).
- **`{transport}`** (`loadgen-core::sink`): `RequestSink<R>::dispatch` drives a `Dispatchable` request to terminal and emits `on_arrival` / `on_admit` / `on_token` / `on_terminal` through a `RequestObserver` into the `TraceCollector`. Real HTTP, mock HTTP, and (designed) in-process engine co-sim are all just `RequestSink` impls behind one observer. TTFT is derived by the collector as the first `on_token` — sinks emit no separate first-token event.

## Extensibility & porting discipline (non-negotiable)

- **Every extension point is a trait.** Anything that could ever have a second implementation — a transport, a clock, a request/response shape, an arrival pattern, a dataset loader, a sampler, a segment store, a metric accumulator, an analyzer, an exporter, an endpoint dialect, a tokenizer, a scheduling policy — MUST be an implementable `trait` (object-safe where it crosses a `dyn` boundary; generic where it is hot-path monomorphized) with at least one concrete impl behind it. Never hardcode a concrete type where a future variant is conceivable. If you are `match`-ing on an enum of "kinds" or branching on a string mode, that is a trait waiting to be extracted. In-tree precedent: `Clock`, `RequestSink<R>` / `RequestObserver` / `Dispatchable`, `SegmentStore` / `PromptMaterializer`, `GraphSink`.
- **Always design ahead.** When you add code, add the seam the next plausible requirement will need — name the trait, take the trait (not the concrete) in signatures, thread the injection point — even if you ship exactly one impl today. The three-modes-for-free property only survives if features are written against the `{transport, clock}` seams, never against a specific backend/clock/transport; a feature that works in only one mode is a design bug. Note the extension you are leaving open in a `//!` / `///` doc comment.
- **Read the ENTIRE Python source before porting ANYTHING.** Before porting a behavior, read the WHOLE Python file end-to-end AND every file it meaningfully touches (its imports, the models it builds, the callers that consume its output, the tests that pin it). Never port from a docstring, a grep hit, a snippet, or a spec paragraph — those omit the earned-in-blood edge cases (SSE fast-paths, backward usage walks, finish/usage reconciliation, firing-gate rounding), which live in the parts of the file you assumed you could skip. Cite the exact Python `path:line` you ported from in the Rust `//!` / `///` docs, and guard it with a parity test wherever byte-exactness matters.

## Coding standards (Rust)

- **SPDX header** (`// SPDX-FileCopyrightText …` + `// SPDX-License-Identifier: Apache-2.0`) atop every source file. `//!` module docs; `///` doc comment on every public item.
- **Thread-per-core**, not work-stealing, on the hot path: N OS threads, each a `current_thread` tokio runtime + `LocalSet`; per-trace state is `Rc`/`RefCell`, futures are `!Send`, tasks are `spawn_local`; parallelism = many traces across threads. See `aiperf-graph/src/runtime.rs`, `transport_bench.rs`. The online `run.rs` path follows the same `!Send` model on a single `current_thread` runtime + `LocalSet` (a shared `Arc` observer + `Semaphore` gate the closed loop).
- **All time through `Clock`** in the clock-aware crates (`aiperf-transport`, `aiperf-graph`, and the CLI's online path): never `Instant::now()`, `SystemTime::now()`, or raw `tokio::time` for measurement or firing gates. `aiperf-core` is now a pure measurement layer and owns no clock.
- **No `Arc`/`Mutex` on hot paths.** Accumulate lock-free per-thread / per-worker and merge once at the end (the graph bench keeps a per-worker accumulator and merges at the join) — never contend a shared collector lock per token on the throughput path.
- **Content-addressed segments**: blake3, prefix-dependent (fold the parent id into the hash so shared prefixes dedup and identical text under different prefixes stays distinct); materialize = clone/concat pre-serialized bytes, never re-serialize.
- **mimalloc** global allocator (per-request alloc churn in the graph executor + streaming client was the top profiled hotspot).
- **Loopback benchmarking**: the hyper transport never consults the ambient `HTTP_PROXY`, so localhost traffic is not proxied (an ambient proxy 405s localhost and tanks throughput).
- **SSE**: buffer raw bytes, split lines on the byte buffer, UTF-8-decode only complete lines (a multibyte char may straddle a TCP chunk boundary).
- **Authoritative token counts**: request `stream_options.include_usage` so `usage.completion_tokens` is returned; the current sinks count one output token per non-empty SSE delta (real per-token arrival times). The Python port's reconcile-to-usage-count step is not yet re-implemented.
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

# graph: Graph-IR E2E streaming throughput (multi-turn DAG conversations)
cargo run --release -p aiperf -- --mode graph [BASE_URL] [MODEL] \
  --turns 4 --instances 400000 --workers <cores> --concurrency 64 --osl 1 [--http2]
```

Unit tests spin up an in-process axum OpenAI-SSE mock (`test_util::spawn_mock`); the graph `transport_bench` throughput path validates against the external `aiperf-mock-rs` binary (a `unix:/path` base URL uses a Unix-domain socket, which is what pushes co-located throughput past 1M req/s).

## Design specs (`specs/`)

Read for intent; verify against `crates/` for reality. Full index + one-liners: [`specs/README.md`](specs/README.md).

- **North star & rationale**
  - `2026-07-10-shared-rust-architecture-northstar.md` — the target abstraction: one front-end, three orthogonal axes (time / backend / workload), a ~120-line neutral contract. *Aspirational vocabulary; not the current symbol set.*
  - `2026-07-10-aiperf-rust-port-exact-vs-redo-ledger.md` — **read first for scope.** Per-concept port-exact vs redo-cleaner vs throw-away rulings; the credit-*policy* trap (delete the protocol, keep the policy).
  - `2026-07-10-unified-graph-runtime-design.md` — **the realization design (read after the ledger).** Every load mode (rate/concurrency/user-centric/fixed-schedule/adaptive/DAG) reduces to one dispatch verb on the clock-scheduled graph executor; strategies become `Workload` schedule generators, not loops. Enumerates every seam as a trait (28), deletes `BranchOrchestrator` + most of `CreditCounter`, specs the OFFLINE dynosim path. Grounded in a line-by-line read of all 37 `timing/` files.
  - `2026-07-10-aiperf-rust-coverage-gap-ledger.md` — research synthesis of the five large unspec'd bodies (sweep-line metrics + columnar accumulator, endpoint/exporter zoo, config-v2 hidden algorithms, timing-engine depth, telemetry) + cross-cutting scope decisions.
- **Architecture seams**
  - `2026-07-10-steppable-clock-injected-engine-design.md` — the `{clock}` seam + the OFFLINE-mock steppable-engine boundary (the missing third mode).
  - `2026-07-10-aiperf-transport-rust-port-design.md` — the Clock-injected hyper transport (realized in `aiperf-transport`).
  - `2026-07-10-aiperf-rust-dataset-segment-seam-design.md` — one content-addressed segment store for datasets + graph (designed).
- **Subsystem designs**
  - `2026-07-10-aiperf-rust-scheduling-policy-sketch.md` — early sketch of the credit-*policy* `Scheduler`. **Superseded by `2026-07-10-unified-graph-runtime-design.md`**, which realizes the policy as `Workload`/`SlotPool`/`RatePool`/`Gate` on the graph executor.
  - `2026-07-10-aiperf-rust-accuracy-accumulator-design.md` — accuracy as a first-class accumulator + analyzer pair (designed).
  - `2026-07-10-aiperf-rust-metrics-accumulator-sweepline-design.md` — **new-code** metrics engine: `aiperf-metrics` leaf crate (columnar accumulator + sweep-line time-weighted curves + percentile/derived kernels + phase windowing/timeslicing + genai-perf `Reporter`). Carries the scars (sweep tie-break/FP-snap, decode-rate `−1`, ICL `nextafter` clamp, ddof split, `adj_*` `+inf`/nearest, phase-tag-authoritative mask, observation_duration); deletes the ZMQ/plugin/dual-path accidental complexity; realizes the unified-runtime `Collector`/`Reporter` seams (designed).
  - `2026-07-10-aiperf-rust-rng-derive-system-design.md` — hash-derived RNG substrate: order-independent `blake3(f"{root}:{id}")[:8]`→u64 seed derivation (blake3 locked, not Python-matching) + `HashIdRandomGenerator` (per-`(trace_id, hash_id)` reseed) + the `RandomGenerator` contract. Ruling: NO cross-language byte parity — one deterministic PRNG (`Pcg64`) + `rand_distr` with faithful distribution/edge-case semantics; internal reproducibility + order-independence, lock-free per-thread + alloc-free reseed. New leaf crate `aiperf-rng` (designed, not built).
  - `2026-07-09-graph-ir-rust-port-design.md` — byte-exact Graph-IR dataflow port (partly realized in `aiperf-graph`).
- **Historical precursors** — describe a *different* working tree built *on* dynamo's `lib/mocker`; the current workspace is standalone (`loadgen-core` extracted, no dynamo dependency). Kept for lineage.
  - `2026-07-09-dynamo-aiperf-shared-core-design.md`, `2026-07-09-dynamo-aiperf-request-rate-tokenizer-design.md`.

## Adding things

- **A new transport** → implement `RequestSink<YourReq>` (+ `Dispatchable` for `YourReq`) in its own crate; nothing in `loadgen-core` changes.
- **A new clock / execution mode** → implement `Clock`; `drive_sim` / `drive_real` already dispatch on `is_virtual()`.
- **A new graph feature** → the `executor` / `segment` / `channel_store` modules in `aiperf-graph`; keep firing-gate arithmetic byte-exact (see the graph-IR spec's parity contract).
- **A new metric** → `loadgen-core::collector` (`TraceCollector` + the `Trace*Stats` structs); the report is a plain struct with a custom `Serialize`.

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
