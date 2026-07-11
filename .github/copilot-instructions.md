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
- **Two HTTP stacks coexist.** The **online CLI** path (`aiperf-core::http_sink::HttpSink`) is **reqwest + `std::time::Instant`**, `Send`/`Arc`/`Mutex` on a multi-thread tokio runtime — it does **not** yet go through `Clock`. The **Clock-injected, `!Send`, thread-per-core** stack (`aiperf-transport`, a hyper client) is used only by the graph benchmark. Moving the online path onto `Clock` is the "PR2.5" gap in the steppable spec.
- **Scheduling policy** (arrival patterns, session-vs-request slots, prefill-release-on-TTFT, phase-scoped metrics, `--request-count` recycle) is **designed** (`specs/…scheduling-policy-sketch.md`) but **not built** — `run.rs` is still a naive `Semaphore` loop (`admit == dispatch`).
- **Dataset/segment unified store, accuracy accumulator, tokenizer-exact prompts, request-rate mode**: designed, not implemented. `SkeletonWorkload` emits fixed synthetic requests; there is no tokenizer and no request-rate mode.

When something is designed-but-not-built, this file says so. Do not assume a spec feature exists in the code.

## Crate workspace (`crates/`)

| Crate | Purpose | Key files |
|---|---|---|
| `aiperf-clock` | The `{clock}` seam. `Clock` trait (`now_ns` / `sleep` / `next_event_time` / `advance_to` / `is_virtual`); `SimClock` = integer-ns discrete-event `BinaryHeap` keyed `(at_ns, seq_no)` (deterministic same-instant tie-break); `RealClock` = monotonic `Instant` + Linux `timerfd`/`AsyncFd` ns sleeps (tokio fallback off-Linux). | `clock.rs`, `sim_clock.rs`, `real_clock.rs` |
| `loadgen-core` | Transport-neutral dispatch/measure seam + the collector. `Dispatchable`, `RequestSink<R>`, `RequestObserver` (f64-ms timestamps), `TraceCollector` → `TraceSimulationReport`. Zero engine/KV/HTTP deps. | `sink.rs`, `collector.rs` |
| `aiperf-core` | Shared online HTTP client + measurement. `HttpSink` (reqwest streaming chat, SSE parse, authoritative `usage` counts, first-token cb, wire trace); `CollectorObserver` (batched `submit`, `reconcile_output_times`); `sse`, `wire`. **Uses `Instant` directly, not `Clock`.** | `http_sink.rs`, `observer.rs`, `sse.rs`, `wire.rs` |
| `aiperf-transport` | Rust-native, **Clock-injected** HTTP transport on hyper 1.x's low-level conn API (not reqwest). h1 + h2c prior-knowledge + UDS; rustls; fine-grained DNS/TCP/TLS/send/recv trace timing; cancellation; connection-reuse strategies. Every timestamp via `Clock`. | `client/`, `transport/`, `sse/`, `models/`, `config/` |
| `aiperf-graph` | Graph-IR async-dataflow engine: multi-turn DAG conversations with fan-in/out + firing-gate timing. Content-addressed `SegmentStore` (blake3, prefix-dependent); `materialize` (static segment + predecessor-reply splice); `executor` / `scheduler` / `channel_store` / `reducers`; `runtime` (`drive_sim` / `drive_real` on `current_thread` + `LocalSet`); `bench` (reqwest) + `transport_bench` (aiperf-transport; UDS → >1M req/s). Offline co-sim sink intentionally not wired yet. | `runtime.rs`, `executor.rs`, `segment.rs`, `transport_bench.rs` |
| `aiperf` | The CLI binary. `--mode online` (default: closed-loop concurrency via `aiperf-core`) and `--mode graph` (Graph-IR throughput). mimalloc global allocator. | `main.rs`, `run.rs`, `workload.rs`, `report.rs` |

Dependency direction: `aiperf` → {`aiperf-core`, `aiperf-graph`}; `aiperf-graph` → {`aiperf-core`, `aiperf-transport`, `aiperf-clock`, `loadgen-core`}; `aiperf-core` → `loadgen-core`; `aiperf-transport` → `aiperf-clock`. `loadgen-core` and `aiperf-clock` are leaves. Workspace: `edition = "2024"`, `resolver = "3"`.

## The two seams (the whole architecture)

- **`{clock}`** (`aiperf-clock`): `RealClock` vs `SimClock` behind one `Clock` trait. `is_virtual()` selects the `drive_real` (tokio reactor drives) vs `drive_sim` (idle-pump: poll the `LocalSet` to quiescence draining all same-instant work → `advance_to(next_event_time)` waking heap-ordered sleepers → repeat) driver over the *same* executor. Virtual time is integer ns with an `(at_ns, seq_no)` deterministic tie-break — **never `tokio::time`** (its 1 ms timer wheel destroys µs/ns firing gates).
- **`{transport}`** (`loadgen-core::sink`): `RequestSink<R>::dispatch` drives a `Dispatchable` request to terminal and emits `on_arrival` / `on_admit` / `on_token` / `on_terminal` through a `RequestObserver` into the `TraceCollector`. Real HTTP, mock HTTP, and (designed) in-process engine co-sim are all just `RequestSink` impls behind one observer. TTFT is derived by the collector as the first `on_token` — sinks emit no separate first-token event.

## Coding standards (Rust)

- **SPDX header** (`// SPDX-FileCopyrightText …` + `// SPDX-License-Identifier: Apache-2.0`) atop every source file. `//!` module docs; `///` doc comment on every public item.
- **Thread-per-core**, not work-stealing, on the hot path: N OS threads, each a `current_thread` tokio runtime + `LocalSet`; per-trace state is `Rc`/`RefCell`, futures are `!Send`, tasks are `spawn_local`; parallelism = many traces across threads. See `aiperf-graph/src/runtime.rs`, `bench.rs`. (The online `run.rs` path is the exception — still `Arc`/`Semaphore` on a multi-thread runtime.)
- **All time through `Clock`** in clock-aware crates (`aiperf-transport`, `aiperf-graph`): never `Instant::now()`, `SystemTime::now()`, or raw `tokio::time` for measurement or firing gates. (`aiperf-core`'s online path currently violates this — a known, tracked gap.)
- **No `Arc`/`Mutex` on hot paths.** Accumulate lock-free per-thread / per-worker and merge once at the end; batch a request's token times and take the collector lock once per request (`observer.rs::submit`), never per token.
- **Content-addressed segments**: blake3, prefix-dependent (fold the parent id into the hash so shared prefixes dedup and identical text under different prefixes stays distinct); materialize = clone/concat pre-serialized bytes, never re-serialize.
- **mimalloc** global allocator (per-request alloc churn in the graph executor + streaming client was the top profiled hotspot).
- **reqwest/hyper**: always disable the ambient proxy for loopback benchmarking (`.no_proxy()`) — an ambient `HTTP_PROXY` 405s localhost and tanks throughput.
- **SSE**: buffer raw bytes, split lines on the byte buffer, UTF-8-decode only complete lines (a multibyte char may straddle a TCP chunk boundary).
- **Authoritative token counts** from `usage.completion_tokens` (`stream_options.include_usage`); reconcile per-chunk arrival times to that count (`observer.rs::reconcile_output_times`) — keeps TTFT/e2e exact while making output-token throughput correct. Port the Python behavior faithfully.
- **Errors**: `anyhow` in the binary/app layer; `thiserror` for library error enums (`aiperf-transport`).
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
  --turns 4 --instances 400000 --workers <cores> --concurrency 64 --osl 1 [--http2] [--reqwest]
```

Unit tests spin up an in-process axum OpenAI-SSE mock (`test_util::spawn_mock`); the graph `transport_bench` throughput path validates against the external `aiperf-mock-rs` binary (a `unix:/path` base URL uses a Unix-domain socket, which is what pushes co-located throughput past 1M req/s).

## Design specs (`specs/`)

Read for intent; verify against `crates/` for reality. Full index + one-liners: [`specs/README.md`](specs/README.md).

- **North star & rationale**
  - `2026-07-10-shared-rust-architecture-northstar.md` — the target abstraction: one front-end, three orthogonal axes (time / backend / workload), a ~120-line neutral contract. *Aspirational vocabulary; not the current symbol set.*
  - `2026-07-10-aiperf-rust-port-exact-vs-redo-ledger.md` — **read first for scope.** Per-concept port-exact vs redo-cleaner vs throw-away rulings; the credit-*policy* trap (delete the protocol, keep the policy).
- **Architecture seams**
  - `2026-07-10-steppable-clock-injected-engine-design.md` — the `{clock}` seam + the OFFLINE-mock steppable-engine boundary (the missing third mode).
  - `2026-07-10-aiperf-transport-rust-port-design.md` — the Clock-injected hyper transport (realized in `aiperf-transport`).
  - `2026-07-10-aiperf-rust-dataset-segment-seam-design.md` — one content-addressed segment store for datasets + graph (designed).
- **Subsystem designs**
  - `2026-07-10-aiperf-rust-scheduling-policy-sketch.md` — the `Scheduler` that re-surfaces credit *policy* (designed, not built).
  - `2026-07-10-aiperf-rust-accuracy-accumulator-design.md` — accuracy as a first-class accumulator + analyzer pair (designed).
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

**Rules:**
- Edit ALL FOUR agent files together (identical body) and ALWAYS finish with `python tools/check_agent_files_sync.py` (or `make check-agent-files-sync`) — non-zero exit = bodies diverged; fix before committing.
- `specs/README.md` and `llms.txt` are NOT sync-checked but MUST move in lockstep — a spec/crate/seam change that leaves them stale is an INCOMPLETE change.
- Ground every claim in `crate/src/file.rs`. State designed-but-not-built explicitly; never describe intent as reality.
- Put the doc updates in the SAME commit as the code/spec change they describe.

### Agent-instruction file sync (mechanics)

`AGENTS.md`, `CLAUDE.md`, `.github/copilot-instructions.md`, and `.cursor/rules/python.mdc` (name is Python-legacy; kept so the checker's target list matches) share a **byte-identical body** below their per-tool headers. Only the header differs: the cursor file keeps its YAML frontmatter (`alwaysApply: true`) then the SPDX comment; the other three start with the SPDX comment. The body begins at the first `# AIPerf` H1. Edit all four together and verify with `python tools/check_agent_files_sync.py` (or `make check-agent-files-sync`).
