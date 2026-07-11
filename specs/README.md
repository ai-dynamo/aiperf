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
| `2026-07-10-aiperf-rust-dataset-segment-seam-design.md` | built end to end + addenda | Realized as `aiperf-dataset`: the shared loader→compose→store→sampler→materializer flow now carries only opaque `{correlation_id, task}` accuracy association; Rust-held ground truth and hidden tests are superseded by the external evaluator boundary. |
| `2026-07-11-aiperf-rust-compile-time-extension-registry-design.md` | built + addendum | Statically linked dataset/sampler/endpoint composition remains in `aiperf-extensions`; accuracy benchmark/grader categories are removed because canonical accuracy is a directly injected `AccuracyEvaluator` process seam. |

### Subsystem designs

| Spec | Status | Purpose |
|---|---|---|
| `2026-07-09-graph-ir-rust-port-design.md` | decided / partly built + addendum | Byte-exact port of the Graph-IR runtime/dataflow plane. **Addendum (2026-07-11):** standalone/offline-only framing is superseded by `aiperf-graph` in the native workspace, with tokio `LocalSet`, `drive_sim`/`drive_real`, and live HTTP graph dispatch. |
| `2026-07-10-aiperf-rust-scheduling-policy-sketch.md` | superseded | Early sketch of the credit-*policy* `Scheduler` (arrival patterns, session-vs-request slots, prefill-release-on-TTFT, absolute-schedule pacing, phase handoff). **Superseded by `2026-07-10-unified-graph-runtime-design.md`**, which realizes the same policy as `Workload`/`SlotPool`/`RatePool`/`Gate` on the graph executor. Kept for lineage. |
| `2026-07-11-aiperf-rust-request-rate-multiturn-design.md` | design | **Faithful, source-grounded** realization of the `request-rate | chain` row: a **single-loop credit issuer** emitting **one turn per rate interval, continuation-priority** (not conversation arrivals), gated by a session `SlotPool` (turn-0→final) + prefill `SlotPool` (every turn→TTFT), bounded by `StopChecker`, turns materialized from the segment pool, think-time deferred via `Clock::sleep`. Read end-to-end from `request_rate.py` + `issuer.py` + `concurrency.py` + `stop_conditions.py` + `callback_handler.py` + `credit_counter.py`. Carries the two-plane throughput framing (control-plane single loop 6.5–20 M/s never the bottleneck; HTTP data plane fans across cores; handoff ~1.7 M/s ≫ any policy rate). Most primitives exist in `aiperf-timing`; unbuilt core = continuation queue + two-source issue loop + conversation source over the segment pool. |
| `2026-07-11-aiperf-rust-user-centric-fixed-schedule-design.md` | built + addendum | The two scheduled load strategies are built over the shared Clock-backed `ScheduledRuntime`: `ConversationSource`, `LocalTaskScheduler`, `UserPool`/`UserCentricWorkload`, and `FixedScheduleSource`/`FixedScheduleWorkload`. User-centric implements virtual-history seeding, per-user cadence, open-loop churn/replacement, optional session caps, stop-and-drain, and live user-target changes; fixed replay implements stable absolute ordering, auto/manual zero, timestamp/delay/immediate continuation precedence, and intentionally ignores normal stop bounds. The CLI supports synthetic or dataset-backed user-centric workloads and dataset-required fixed replay with detailed timing JSON. Exact `SimClock` tests plus real `aiperf-mock-rs` library/CLI tests validate timing and reply splicing. Offline engine dispatch remains gated by the separate unwired co-sim sink. **Addendum (2026-07-11)** supersedes the original designed-status and resolves the listed implementation questions. |
| `2026-07-11-aiperf-rust-phase-runner-orchestrator-design.md` | built + implementation addendum | `aiperf-timing::phase` implements validated lifecycle/config/progress, direct observers, Clock-driven duration→grace→cancel→drain→force escalation, failure finalization, ordered warmup→profiling execution, seamless background returns, cross-phase debt drain, and cancellation that cannot advance phases. `aiperf::phase_runtime` connects scheduled workloads, shared slot resources, phase-owned ramps, processors, `SimClock`, and real HTTP; ordinary scheduled entry points lower through one profiling phase. The ZMQ publisher/manager/worker-readiness layer is deleted. Arbitrary phase-list CLI authoring and the Graph-IR phase consumer remain separate composition work. |
| `2026-07-11-aiperf-rust-adaptive-scale-design.md` | built + addendum | SLA-driven autoscaling is built in `aiperf-adaptive`: object-safe actuator/evaluator/step/window/controller seams, all four live actuators, Python-grounded SLA math with authoritative completion-token OSL/ITL reconciliation, `ramp_until_fail` discover/sustain/single-recovery control, Clock-paced assessments, and schema-v2 events/summary artifacts. The online CLI wires session concurrency, prefill concurrency, request rate, and user-centric users into their live issuer state. `SimClock` control is unit-tested; offline end-to-end remains gated by the still-unwired in-process engine sink. **Addendum (2026-07-11)** supersedes the original designed-status and proposed actuator placement. |
| `2026-07-11-aiperf-rust-ancillary-timing-policy-design.md` | built + addendum | Three ancillary knobs are built: `aiperf-timing` owns Clock-driven `RampStrategy`/`RampDriver` with Linear/Exponential/Poisson curves, seeded warmup-aware `CancellationPolicy`, and round-robin `UrlSelector`; `aiperf-transport` anchors HTTP 499 disconnects to captured full-body send completion; online/scheduled issuers carry cancellation and sticky session endpoint indices. Ordinary online and user-centric paths wire applicable live actuators; explicit phase plans own prepared drivers through `RampScheduledPhaseController`. CLI flags expose the controls. Fixed/user-centric reject inapplicable ramps; the graph arrival/slot consumer remains companion-spec work. **Addendum (2026-07-11)** records the support matrix. |
| `2026-07-11-aiperf-rust-dag-branch-orchestrator-design.md` | superseded (by `aiperf-graph` dataflow) | The Python ~1000-line FORK/SPAWN credit-side branch orchestrator is **superseded wholesale by the `aiperf-graph` async-dataflow engine** — fan-out = out-edges, join gating = `ChannelRequirement.count` (static producer accounting), and sticky-routing / drain-observer / future-active-gate + the spawn-first / drain-after-return races are **deleted credit-protocol artifacts, not ported**. Residual = graph-build lowering (branch metadata → nodes/edges) + FORK/SPAWN materialization + session-cap wiring + whole-run FAIL_FAST. Lineage + reconciliation doc, not a build plan. |
| `2026-07-10-aiperf-rust-accuracy-accumulator-design.md` | built + addenda | Rust retains typed accumulation/analysis/reporting and owns normal inference. One supervised pinned Python/Lighteval worker owns canonical datasets, prompts, private tests, execution, and every grading decision over strict versioned JSONL with opaque ids. Worker failures are infrastructure errors; terminal text is batch-graded after dispatch drains; native v2 records exact evaluator/package/dataset identity. |
| `2026-07-10-aiperf-rust-metrics-accumulator-sweepline-design.md` | built + addenda | Built IO-free engine in `aiperf-metrics`: NaN-sparse column store, exact ragged replay, record/aggregate/derived metrics, SLO goodput, all effective/active and ICL-aware sweep curves, duration-weighted statistics, authoritative phase windows/timeslices, deterministic worker-local merge, and typed native-v2 `Reporter`. Online/scheduled/adaptive/accuracy adapters feed observer timing/classification/usage plus real HTTP traces; graph's lean workers feed request/token/usage facts directly and merge by worker; fixed schedules omit credit-relative metrics. Telemetry producers and genai-perf-v1 compatibility export remain separate unbuilt consumers. |
| `2026-07-10-aiperf-rust-metric-catalog-appendix.md` | built (appendix) + addendum | Built catalog of 103 inherited Python identities plus 16 native sweep identities, with exact metadata/dependencies and implementations for every record/aggregate/derived row whose source data exists. Validation and a deterministic metadata fingerprint pin the graph. Telemetry-owned injected rows intentionally stay absent until their producer supplies values. |
| `2026-07-10-aiperf-rust-telemetry-accumulators-design.md` | design + addenda | GPU / server-metrics / network-RTT telemetry as side-channel `Accumulator`s reusing the metrics seam. Covers DCGM fields, server routing/fallback/auto-disable, histogram percentiles, and TCP-connect RTT. **Addenda:** 2026-07-11 phase-boundary snapshots supersede scrape-then-reconstruct windowing (GPU counters); 2026-07-11 (server metrics) extends boundary snapshots to server counters — unifies the two baseline pickers, sequential scrape deletes the auto-disable concurrency race (histograms stay hybrid: intra-phase series for mean-learning + boundary delta for totals); **and the follow-up correction deletes the flush/settle wait entirely** (`COLLECTION_FLUSH_PERIOD` gone — AIPerf's own metrics are authoritative; the server counters are a cross-check, so snapshot at the boundary and accept the bounded smear); 2026-07-11 clarifies telemetry depends on/reuses the metrics seam without making `aiperf-metrics` depend on telemetry collectors. |
| `2026-07-11-aiperf-rust-exporters-overhaul-design.md` | partially built + addendum | The typed, IO-free native-v2 `Reporter` and application JSON writer are built and are the `--json` output across online, scheduled, accuracy, and graph modes. Metrics are name-keyed with type-specific distribution/scalar/counter series, metadata, timeslices, warmup/accuracy joins, absent omission, and non-finite nulls. Native CSV, genai-perf-v1 compatibility files, warnings/insights, console replay, and timed uploaders remain designed. |
| `2026-07-11-aiperf-rust-endpoints-design.md` | design | **Faithful port** of the endpoint layer (crate `aiperf-endpoints`): the `Endpoint` trait (build request body + parse response into records) + input-side ISL accounting. Carries the parse scars — chat response precedence `reasoning>content+tool_calls>tool_calls>content` (the ~18% OSL-undercount mixed-emit fix), tool-call streaming reassembly (missing-index→`len(dict)`, modern name-overwrite/args-concat vs legacy), the responses SSE event map (`function_call_arguments.delta`→ToolCall, ~64% of agentic turns) + replay-unsafe filter + dedup-by-id union, the three malformed-response policies (embeddings raises; chat/completions degrade). The input-ISL walk's tool-schema `orjson.dumps(parameters)` byte-parity is the #1 risk. Capability-flag→lifecycle table + 16-type registry; tier-2 vendor endpoints deferred. |
| `2026-07-10-aiperf-rust-rng-derive-system-design.md` | built + addenda | Native hash-derived RNG substrate built as leaf crate `aiperf-rng`: `RngRoot::derive` + BLAKE3 seed derivation, canonical namespace constants, alloc-free `HashIdRandomGenerator`, `RandomGenerator`, generic sampler seams, five sampling distributions, and sequence distributions. Weighted mixtures cache validated cumulative weights; a Rust-internal profile canary pins the stream. Answers the coverage-gap ledger's RNG question: **NO cross-language byte parity anywhere** — internal reproducibility + order-independence + distributional/semantic parity via deterministic `Pcg64` + `rand_distr`. Dataset composition/samplers and ancillary timing policies consume it; broader scheduler/graph integration remains. |

### Historical precursors

These predate the standalone `crates/` workspace and describe a **different**
working tree (`dynamo-aiperf-native`) built *on* ai-dynamo's `lib/mocker`. The
current workspace extracted `loadgen-core` and dropped the dynamo dependency, so
these are lineage, not current architecture.

| Spec | Status | Purpose |
|---|---|---|
| `2026-07-09-dynamo-aiperf-shared-core-design.md` | superseded | Increment-1 walking skeleton sharing DynoSim's collector/driver via a curated facade; origin of the `RequestSink` / `RequestObserver` seam. |
| `2026-07-09-dynamo-aiperf-request-rate-tokenizer-design.md` | superseded | Increment-2 tokenizer-exact prompts + Poisson request-rate through the shared `WorkloadDriver`. |
