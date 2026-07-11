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
| `2026-07-11-aiperf-rust-request-rate-multiturn-design.md` | design | **Faithful, source-grounded** realization of the `request-rate | chain` row: a **single-loop credit issuer** emitting **one turn per rate interval, continuation-priority** (not conversation arrivals), gated by a session `SlotPool` (turn-0→final) + prefill `SlotPool` (every turn→TTFT), bounded by `StopChecker`, turns materialized from the segment pool, think-time deferred via `Clock::sleep`. Read end-to-end from `request_rate.py` + `issuer.py` + `concurrency.py` + `stop_conditions.py` + `callback_handler.py` + `credit_counter.py`. Carries the two-plane throughput framing (control-plane single loop 6.5–20 M/s never the bottleneck; HTTP data plane fans across cores; handoff ~1.7 M/s ≫ any policy rate). Most primitives exist in `aiperf-timing`; unbuilt core = continuation queue + two-source issue loop + conversation source over the segment pool. |
| `2026-07-11-aiperf-rust-user-centric-fixed-schedule-design.md` | design (user-centric partly built) | The other load strategies. **User-centric**: per-user cadence (`stagger=1/rate`, `turn_gap=num_users/rate`), virtual-history steady-state seeding, open-loop churn + user replacement. **Fixed-schedule**: absolute-timestamp trace replay (no rate/slots/stop). Source-grounded from `strategies/{core,user_centric_rate,fixed_schedule}.py`. User-centric SETUP math is partly built (`aiperf-timing::plan_user_centric`, verified line-for-line); the per-user pacer/spawn-heap/churn + all of fixed-schedule are unbuilt. New seams: `UserPool`, `FixedScheduleSource`, `UserCentricWorkload`/`FixedScheduleWorkload`. |
| `2026-07-11-aiperf-rust-phase-runner-orchestrator-design.md` | design | The phase driver ABOVE the credit issuer: `PhaseLifecycle` state machine (CREATED→STARTED→SENDING_COMPLETE→COMPLETE + orthogonal cancel flag), grace/duration-timeout/cancel-drain/force-complete escalation, warmup→profiling sequencing with **seamless** overlap + cross-phase debt-drain. Source-grounded from `phase/runner.py` (786 lines) + `phase_orchestrator.py` + `phase/publisher.py` + `manager.py` + `config.py`. **Deletes the ZMQ/IPC scaffolding** (`PhasePublisher`, the `TimingManager` service, `wait_for_workers`) → direct `PhaseObserver`/`RequestObserver` trait calls on one `!Send` loop. New seams: `PhaseRunner`, `PhaseOrchestrator`, `PhaseObserver`. |
| `2026-07-11-aiperf-rust-adaptive-scale-design.md` | design | SLA-driven concurrency/rate autoscaling: the monotone **`ramp_until_fail`** step-ramp (step scaled by the tightest SLA filter's head-room) walking a knob via `AdaptiveControlBackend` → `SlotPool::set_limit`/`IntervalGenerator::set_rate`, watching a tumbling `WindowSampler` SLA over returned requests (`discover→sustain→complete`). Source-grounded from the 9 `adaptive_scale*`/`adaptive_*` files. **Closed-loop live-metrics-during-the-run** is the sharpest offline-parity constraint (the sim engine sink must deliver incremental completions or every window starves). Actuators built; controller/SLA/window designed. |
| `2026-07-11-aiperf-rust-ancillary-timing-policy-design.md` | design | Three ancillary knobs: **ramping** (`RampStrategy`/`RampDriver` Linear/Exponential-ease-in/Poisson curves driving `set_rate`/`set_limit`; discrete-vs-continuous by `update_interval`), **request cancellation** (Bernoulli-fraction **fixed-delay** client disconnect armed at **send-complete** not issuance, off during warmup → `cancel_after_ns` → Clock-scheduled transport abort), **URL sampling** (round-robin with **turn-0-only** advance + sticky-per-session pin). Source-grounded from `ramping.py`/`request_cancellation.py`/`url_samplers.py`. Actuators built; `RampStrategy`/`CancellationPolicy`/`UrlSelector` traits new. |
| `2026-07-11-aiperf-rust-dag-branch-orchestrator-design.md` | superseded (by `aiperf-graph` dataflow) | The Python ~1000-line FORK/SPAWN credit-side branch orchestrator is **superseded wholesale by the `aiperf-graph` async-dataflow engine** — fan-out = out-edges, join gating = `ChannelRequirement.count` (static producer accounting), and sticky-routing / drain-observer / future-active-gate + the spawn-first / drain-after-return races are **deleted credit-protocol artifacts, not ported**. Residual = graph-build lowering (branch metadata → nodes/edges) + FORK/SPAWN materialization + session-cap wiring + whole-run FAIL_FAST. Lineage + reconciliation doc, not a build plan. |
| `2026-07-10-aiperf-rust-accuracy-accumulator-design.md` | built + addenda | Native Rust `aiperf-metrics` now has `AccuracyRecord`/`GradingResult`, `AccuracyAccumulator`, phase/time-window summaries, real `CorrelationId` association, dependency-enforced `AccuracyResultsAnalyzer`, and optional quality-at-load / accuracy-per-energy joins. Grader plugins and runtime wiring remain future consumers. |
| `2026-07-10-aiperf-rust-metrics-accumulator-sweepline-design.md` | design + addendum | **New-code** metrics engine: `aiperf-metrics` leaf crate — columnar accumulator + sweep-line curves + percentile/derived kernels + phase windowing/timeslicing + genai-perf `Reporter`. **Addendum (2026-07-11):** categorical interning is dense first-appearance `FxHashMap`/reverse-vector interning, not BLAKE3; telemetry reuses the metrics seam without inverting dependencies. |
| `2026-07-10-aiperf-rust-metric-catalog-appendix.md` | design (appendix) | The ~120-metric `MetricSpec` catalog over the engine above: every metric's tag/header/unit/type/agg/flags/console-group/required + formula, the base-class→compute-shape mapping (RECORD/AGGREGATE/DERIVED/DerivedSum), and the per-metric scars (ITL `osl<2`+`osl−1`, TTFO first-non-reasoning, osl-mismatch `min()` cap, absent-vs-0, the zero-error trap, `ERROR_ONLY` gate inversion, the injected-metric pattern, wall-vs-perf clock, network-adjusted exclusions). |
| `2026-07-10-aiperf-rust-telemetry-accumulators-design.md` | design + addenda | GPU / server-metrics / network-RTT telemetry as side-channel `Accumulator`s reusing the metrics seam. Covers DCGM fields, server routing/fallback/auto-disable, histogram percentiles, and TCP-connect RTT. **Addenda:** 2026-07-11 phase-boundary snapshots supersede scrape-then-reconstruct windowing (GPU counters); 2026-07-11 (server metrics) extends boundary snapshots to server counters — unifies the two baseline pickers, sequential scrape deletes the auto-disable concurrency race (histograms stay hybrid: intra-phase series for mean-learning + boundary delta for totals); **and the follow-up correction deletes the flush/settle wait entirely** (`COLLECTION_FLUSH_PERIOD` gone — AIPerf's own metrics are authoritative; the server counters are a cross-check, so snapshot at the boundary and accept the bounded smear); 2026-07-11 clarifies telemetry depends on/reuses the metrics seam without making `aiperf-metrics` depend on telemetry collectors. |
| `2026-07-11-aiperf-rust-exporters-overhaul-design.md` | design | **Overhaul** (not port) of the exporters: one typed `Report` → a static set of `Exporter`s behind one trait; deletes the plugin registry / async fan-out / `outputs_json` shard-glob / mlflow-wandb subprocess apparatus. **Breaks genai-perf as the default** in favor of a **v2 native report** — one unified file using the type-specific-series model lifted from the server-metrics export (metrics-keyed-by-name; type-tagged `distribution`/`scalar`/`counter`/`histogram` stats so there is no `avg`-for-scalars lie; labeled `series[]`; per-type timeslices; `percentiles` map). genai-perf v1 kept as an opt-in compat sink (`--export-genai-perf`, frozen SCHEMA_VERSION 1.4 contract). Keeps the OSL-mismatch / usage-discrepancy / API-error warning intelligence + version lore. |
| `2026-07-11-aiperf-rust-endpoints-design.md` | design | **Faithful port** of the endpoint layer (crate `aiperf-endpoints`): the `Endpoint` trait (build request body + parse response into records) + input-side ISL accounting. Carries the parse scars — chat response precedence `reasoning>content+tool_calls>tool_calls>content` (the ~18% OSL-undercount mixed-emit fix), tool-call streaming reassembly (missing-index→`len(dict)`, modern name-overwrite/args-concat vs legacy), the responses SSE event map (`function_call_arguments.delta`→ToolCall, ~64% of agentic turns) + replay-unsafe filter + dedup-by-id union, the three malformed-response policies (embeddings raises; chat/completions degrade). The input-ISL walk's tool-schema `orjson.dumps(parameters)` byte-parity is the #1 risk. Capability-flag→lifecycle table + 16-type registry; tier-2 vendor endpoints deferred. |
| `2026-07-10-aiperf-rust-rng-derive-system-design.md` | built + addendum | Native hash-derived RNG substrate built as leaf crate `aiperf-rng`: `RngRoot` BLAKE3 seed derivation + `HashIdRandomGenerator` + `RandomGenerator` facade + sampling and sequence distributions. Answers the coverage-gap ledger's RNG question: **NO cross-language byte parity anywhere** — internal reproducibility + order-independence + distributional/semantic parity via deterministic `Pcg64` + `rand_distr`. Consumers are not wired yet. |

### Historical precursors

These predate the standalone `crates/` workspace and describe a **different**
working tree (`dynamo-aiperf-native`) built *on* ai-dynamo's `lib/mocker`. The
current workspace extracted `loadgen-core` and dropped the dynamo dependency, so
these are lineage, not current architecture.

| Spec | Status | Purpose |
|---|---|---|
| `2026-07-09-dynamo-aiperf-shared-core-design.md` | superseded | Increment-1 walking skeleton sharing DynoSim's collector/driver via a curated facade; origin of the `RequestSink` / `RequestObserver` seam. |
| `2026-07-09-dynamo-aiperf-request-rate-tokenizer-design.md` | superseded | Increment-2 tokenizer-exact prompts + Poisson request-rate through the shared `WorkloadDriver`. |
