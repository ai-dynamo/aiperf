<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# `specs/` — native Rust AIPerf design record

This folder is the design record for **the native Rust AIPerf** (the `crates/`
workspace on branch `ajc/rust`): a from-scratch, single-process, multi-threaded
tokio rewrite of the Python AIPerf LLM-inference benchmarking tool. The thesis
across every spec is the same: keep AIPerf's *external contracts* and its
*earned-in-blood algorithms* (SSE parsing, timing breakdown, metric formulas,
firing-gate arithmetic), keep the **`{clock}` + `{transport}` trait seam** as the
crown jewel (it is what makes real / mock / offline execution modes free), and
throw away every internal artifact of the Python multiprocess/GIL model (ZMQ bus,
services, credit protocol, `plugins.yaml`, shard export, mmap cache). The specs
are **design intent** — the code in `crates/` is a walking skeleton that is ahead
of and behind them in places. When they disagree, the code wins; verify before
relying on any spec feature (see [`../llms.txt`](../llms.txt) and the four agent
files for the code-vs-spec gaps).

Reading order for a newcomer: the **ledger** first (it frames scope), then the
**north star** (the target shape), then whichever subsystem you are touching.

## Conventions

- **Specs are append-only history.** Never rewrite a spec's body to reflect a later
  decision. When a decision or implementation supersedes, revises, or contradicts a
  shipped spec, append a dated `## Addendum — YYYY-MM-DD` section at the END of that
  spec stating what changed, why, and which section/claim it supersedes. The original
  text stays as the record; the addendum is authoritative where they conflict.
- **Status column** reflects the whole spec's current standing: `decided` (the
  design holds), `design` / `sketch` (proposed, not built), `partly built` (code
  exists, verify per-claim), `superseded` (a newer spec or an addendum overrides it —
  the row says which). Bump the status here whenever an addendum lands.

## Index

### North star & rationale

| Spec | Status | Purpose |
|---|---|---|
| `2026-07-10-aiperf-rust-port-exact-vs-redo-ledger.md` | decided | **Start here.** Per-concept port-exact vs redo-cleaner vs throw-away rulings; the credit-*policy* trap (delete the ZMQ protocol, keep the scheduling policy); the "ONE front-end, THREE modes" framing. |
| `2026-07-10-shared-rust-architecture-northstar.md` | decided (aspirational) | The cleanest end-state abstraction: three orthogonal axes (time / backend / workload), a ~120-line neutral contract, one `dispatch` verb. Vocabulary (`Backend`/`Engine`/`Harness`) is the target, not the current symbol set. |

### Architecture seams

| Spec | Status | Purpose |
|---|---|---|
| `2026-07-10-steppable-clock-injected-engine-design.md` | decided | The `{clock}` seam and the OFFLINE-mock steppable-engine boundary (`submit` / `step_to` / `next_event_ms` / injected time). The missing third execution mode; canonical time = `i64` ns. |
| `2026-07-10-aiperf-transport-rust-port-design.md` | decided / partly built | The Clock-injected hyper HTTP transport (streaming SSE, fine-grained trace timing, cancellation, reuse strategies, h1/h2c/UDS). Realized as the `aiperf-transport` crate. |
| `2026-07-10-aiperf-rust-dataset-segment-seam-design.md` | design | Unify the graph segment store and the multi-modal dataset cache into one content-addressed segment/blob store; `Conversation`/`Turn` carry handles, not bytes. Designed; not yet built. |

### Subsystem designs

| Spec | Status | Purpose |
|---|---|---|
| `2026-07-09-graph-ir-rust-port-design.md` | decided / partly built | Byte-exact port of the Graph-IR runtime/dataflow plane (segment store, channels, reducers, firing-gate executor, DES clock). Partly realized in `aiperf-graph`. |
| `2026-07-10-aiperf-rust-scheduling-policy-sketch.md` | sketch | A single-threaded `Scheduler` that re-surfaces the credit *policy* deleted with ZMQ: arrival patterns, session-vs-request slots, prefill-release-on-TTFT, absolute-schedule pacing, phase handoff. Designed; `run.rs` is still a naive `Semaphore` loop. |
| `2026-07-10-aiperf-rust-accuracy-accumulator-design.md` | design | Make accuracy a first-class accumulator + analyzer pair (like energy), keyed by a real correlation id, so accuracy-over-time / under-load / per-watt joins fall out. Designed. |

### Historical precursors

These predate the standalone `crates/` workspace and describe a **different**
working tree (`dynamo-aiperf-native`) built *on* ai-dynamo's `lib/mocker`. The
current workspace extracted `loadgen-core` and dropped the dynamo dependency, so
these are lineage, not current architecture.

| Spec | Status | Purpose |
|---|---|---|
| `2026-07-09-dynamo-aiperf-shared-core-design.md` | superseded | Increment-1 walking skeleton sharing DynoSim's collector/driver via a curated facade; origin of the `RequestSink` / `RequestObserver` seam. |
| `2026-07-09-dynamo-aiperf-request-rate-tokenizer-design.md` | superseded | Increment-2 tokenizer-exact prompts + Poisson request-rate through the shared `WorkloadDriver`. |
