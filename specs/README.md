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
| `2026-07-10-aiperf-rust-port-exact-vs-redo-ledger.md` | decided + addendum | **Start here.** Per-concept port-exact vs redo-cleaner vs throw-away rulings; the credit-*policy* trap; the "ONE front-end, THREE modes" framing. **Addendum (2026-07-11):** online/offline parity means shared code path + report schema, not byte-identical real-vs-sim metric values; scheduling policy is realized through the unified-runtime `Workload`/`SlotPool`/`RatePool`/`Gate` seams. |
| `2026-07-10-shared-rust-architecture-northstar.md` | decided (aspirational) + addendum | The cleanest end-state abstraction: three orthogonal axes (time / backend / workload), a ~120-line neutral contract, one `dispatch` verb. North-star backend/engine/harness vocabulary is aspirational; **current built symbols are** `Clock` + `RequestSink<R>` / `RequestObserver` / `Dispatchable`, with virtual controls inherent on `SimClock`. |
| `2026-07-10-unified-graph-runtime-design.md` | decided + addendum | **The realization design.** Every load mode reduces to one dispatch verb on the clock-scheduled graph executor; strategies become `Workload` schedule generators. Supersedes the scheduling-policy sketch. **Addendum (2026-07-11):** RNG seed derivation is BLAKE3, and implementation against today's crates should translate north-star backend/sink terms to `RequestSink<R>` / `RequestObserver` / `Dispatchable`. |
| `2026-07-10-aiperf-rust-coverage-gap-ledger.md` | research synthesis + addendum | 7-pass read of the 720-file Python tree cataloguing large unspec'd bodies. **Addendum (2026-07-11):** metrics, telemetry, and RNG gaps are now covered by dedicated specs/addenda; remaining gap areas are endpoint/exporter, config-v2, timing-engine depth, and presentation/API/plot surfaces. |

### Architecture seams

| Spec | Status | Purpose |
|---|---|---|
| `2026-07-10-steppable-clock-injected-engine-design.md` | decided + addendum | The `{clock}` seam and OFFLINE-mock steppable-engine boundary. **Addendum (2026-07-11):** its `lib/aiperf` + dynamo `lib/mocker` framing is historical; translate concepts to the standalone `crates/` workspace and the current `Clock` + `RequestSink` seam. |
| `2026-07-10-aiperf-transport-rust-port-design.md` | decided / partly built + addendum | The Clock-injected hyper HTTP transport. Realized as `aiperf-transport`; **addendum (2026-07-11):** cancellation-after-send, full h2 reuse semantics, and the full aiohttp-style trace field set are design targets where current code is narrower. |
| `2026-07-10-aiperf-rust-dataset-segment-seam-design.md` | design + addendum | Unify the graph segment store and multi-modal dataset cache into one content-addressed segment/blob store; `Conversation`/`Turn` carry handles, not bytes. **Addendum (2026-07-11):** preserve raw payload/tool/header fields, audio duration, context mode, and DAG metadata needed for dispatch, metrics, and context reconstruction. |

### Subsystem designs

| Spec | Status | Purpose |
|---|---|---|
| `2026-07-09-graph-ir-rust-port-design.md` | decided / partly built + addendum | Byte-exact port of the Graph-IR runtime/dataflow plane. **Addendum (2026-07-11):** standalone/offline-only framing is superseded by `aiperf-graph` in the native workspace, with tokio `LocalSet`, `drive_sim`/`drive_real`, and live HTTP graph dispatch. |
| `2026-07-10-aiperf-rust-scheduling-policy-sketch.md` | superseded | Early sketch of the credit-*policy* `Scheduler` (arrival patterns, session-vs-request slots, prefill-release-on-TTFT, absolute-schedule pacing, phase handoff). **Superseded by `2026-07-10-unified-graph-runtime-design.md`**, which realizes the same policy as `Workload`/`SlotPool`/`RatePool`/`Gate` on the graph executor. Kept for lineage. |
| `2026-07-10-aiperf-rust-accuracy-accumulator-design.md` | design + addendum | Make accuracy a first-class accumulator + analyzer pair (like energy), keyed by a real correlation id. **Addendum (2026-07-11):** Python is no longer just a `NotImplementedError` stub, but the Rust goal remains integrating accuracy into the main accumulator/reporting pipeline. |
| `2026-07-10-aiperf-rust-metrics-accumulator-sweepline-design.md` | design + addendum | **New-code** metrics engine: `aiperf-metrics` leaf crate — columnar accumulator + sweep-line curves + percentile/derived kernels + phase windowing/timeslicing + genai-perf `Reporter`. **Addendum (2026-07-11):** categorical interning is dense first-appearance `FxHashMap`/reverse-vector interning, not BLAKE3; telemetry reuses the metrics seam without inverting dependencies. |
| `2026-07-10-aiperf-rust-metric-catalog-appendix.md` | design (appendix) | The ~120-metric `MetricSpec` catalog over the engine above: every metric's tag/header/unit/type/agg/flags/console-group/required + formula, the base-class→compute-shape mapping (RECORD/AGGREGATE/DERIVED/DerivedSum), and the per-metric scars (ITL `osl<2`+`osl−1`, TTFO first-non-reasoning, osl-mismatch `min()` cap, absent-vs-0, the zero-error trap, `ERROR_ONLY` gate inversion, the injected-metric pattern, wall-vs-perf clock, network-adjusted exclusions). |
| `2026-07-10-aiperf-rust-telemetry-accumulators-design.md` | design + addenda | GPU / server-metrics / network-RTT telemetry as side-channel `Accumulator`s reusing the metrics seam. Covers DCGM fields, server routing/fallback/auto-disable, histogram percentiles, and TCP-connect RTT. **Addenda:** 2026-07-10 phase-boundary snapshots supersede scrape-then-reconstruct windowing; 2026-07-11 clarifies telemetry depends on/reuses the metrics seam without making `aiperf-metrics` depend on telemetry collectors. |
| `2026-07-10-aiperf-rust-rng-derive-system-design.md` | design | Native hash-derived RNG substrate: `blake3(f"{root}:{id}")[:8]`→u64 order-independent seed derivation + `HashIdRandomGenerator` + `RandomGenerator` facade. Answers the coverage-gap ledger's RNG question: **NO cross-language byte parity anywhere** — internal reproducibility + order-independence + distributional/semantic parity via deterministic `Pcg64` + `rand_distr`. New leaf crate `aiperf-rng`. |

### Historical precursors

These predate the standalone `crates/` workspace and describe a **different**
working tree (`dynamo-aiperf-native`) built *on* ai-dynamo's `lib/mocker`. The
current workspace extracted `loadgen-core` and dropped the dynamo dependency, so
these are lineage, not current architecture.

| Spec | Status | Purpose |
|---|---|---|
| `2026-07-09-dynamo-aiperf-shared-core-design.md` | superseded | Increment-1 walking skeleton sharing DynoSim's collector/driver via a curated facade; origin of the `RequestSink` / `RequestObserver` seam. |
| `2026-07-09-dynamo-aiperf-request-rate-tokenizer-design.md` | superseded | Increment-2 tokenizer-exact prompts + Poisson request-rate through the shared `WorkloadDriver`. |
