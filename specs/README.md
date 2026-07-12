<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# `specs/` — Python-orchestrated, Rust-executed AIPerf design record

This folder is the design record for **Python-orchestrated, Rust-executed
AIPerf** on branch `ajc/rust`: Python owns the user CLI, Config v2, outer loops,
and presentation; the sole Rust executable is the single-run `aiperf-runner`;
and the `crates/aiperf` package is a runtime library, not a second CLI. The thesis
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
| `2026-07-10-shared-rust-architecture-northstar.md` | decided (aspirational) + addenda | The cleanest end-state abstraction: three orthogonal axes (time / backend / workload), a ~120-line neutral contract, one `dispatch` verb. North-star backend/engine/harness vocabulary is aspirational; **current built symbols are** `Clock` + `RequestSink<R>` / `RequestObserver` / `Dispatchable`, with virtual controls inherent on `SimClock`. The application-layer addendum replaces the native bin with Python Config v2 plus the strict runner. |
| `2026-07-10-unified-graph-runtime-design.md` | decided + addendum | **The realization design.** Every load mode reduces to one dispatch verb on the clock-scheduled graph executor; strategies become `Workload` schedule generators. Supersedes the scheduling-policy sketch. **Addendum (2026-07-11):** RNG seed derivation is BLAKE3, and implementation against today's crates should translate north-star backend/sink terms to `RequestSink<R>` / `RequestObserver` / `Dispatchable`. |
| `2026-07-10-aiperf-rust-coverage-gap-ledger.md` | research synthesis + addendum | 7-pass read of the 720-file Python tree cataloguing large unspec'd bodies. **Addendum (2026-07-11):** metrics, telemetry, and RNG gaps are now covered by dedicated specs/addenda; remaining gap areas are endpoint/exporter, config-v2, timing-engine depth, and presentation/API/plot surfaces. |

### Architecture seams

| Spec | Status | Purpose |
|---|---|---|
| `2026-07-10-steppable-clock-injected-engine-design.md` | library built behind feature; runner projection pending + addenda | The `{clock}` seam and OFFLINE-mock steppable-engine boundary are implemented with dynamic cancellation, all canonical trace formats/topologies/routers, exact cutoff, SLA/raw/timed artifacts, AIC/profile/KV-offload forwarding, fail-closed offload, and byte-exact common reports. AIPerf offline is not currently product-reachable; the runner-only execution-surface spec owns its feature-gated backend projection and subprocess gates. The separate Python `aiperf dynosim` facade remains canonical for Dynamo products. |
| `2026-07-10-aiperf-transport-rust-port-design.md` | decided / partly built + addendum | The Clock-injected hyper HTTP transport. Realized as `aiperf-transport`; **addendum (2026-07-11):** cancellation-after-send, full h2 reuse semantics, and the full aiohttp-style trace field set are design targets where current code is narrower. |
| `2026-07-10-aiperf-rust-dataset-segment-seam-design.md` | built end to end + addenda | Realized as `aiperf-dataset`: the complete loader→compose→dense-handle store→sampler→materializer pipeline is shared by `aiperf-runner`, evaluator-authored static accuracy, Graph-IR, and the library-only offline adapter. Addenda preserve all dispatch/metric/DAG fields, resolve the four open decisions, and replace Rust-held accuracy ground truth with an opaque `{correlation_id, task}` external-evaluator association. |
| `2026-07-11-aiperf-rust-compile-time-extension-registry-design.md` | built + addenda | Statically linked replacement for Python `plugins.yaml`: a cycle-free `aiperf-extensions` composition crate, aggregate `AiperfRegistry`, transactional `AiperfExtension` registration, and deterministic duplicate rejection for dataset formats, samplers, and endpoints. Addenda remove accuracy benchmark/grader categories and record that the runner-owned endpoint spec supersedes the current closed-enum/runner-composition limitation. |
| `2026-07-11-aiperf-runner-owned-endpoint-registry-design.md` | decided / not built + adjudication addendum | **Authoritative endpoint-ownership design.** The exact selected `aiperf-runner` binary owns one open-ID, descriptor-bearing, extension-aware endpoint registry used for capabilities, validation, and execution. The adjudication addendum pins authored preflight, validation completeness, the strict v2 operation envelope, executable-content identity, and worker-local prepared bindings. Backend/workload mode reachability is deliberately delegated to the separate runner-only execution-surface spec. |
| `2026-07-11-aiperf-runner-only-execution-surface-design.md` | decided / partly built | **Authoritative native product-reachability design.** `aiperf-runner` is the sole native executable. A strict authored v2 envelope composes registered `online_http` or feature-gated `dynamo_offline` backends with scheduled, Graph-IR, static-accuracy, or stateful-agentic workloads; capabilities, preparation, reports, packaging, migration, and subprocess gates are defined for every supported pair. |
| `2026-07-11-python-orchestrator-rust-single-run-design.md` | built v1 + runner-only v2 addenda | Canonical process architecture: Python `aiperf` owns structural Config v2, the only human CLI, and outer/presentation loops; a fresh strict `aiperf-runner` is the only Rust executable and owns one run's hot path. Protocol v2 replaces side-effecting Python pre-resolution with an authored projection followed by Rust validation/preparation. Online scheduled/static accuracy are currently runner-reachable; the runner-only execution-surface spec owns the remaining Graph/offline/agentic migration. |

### Subsystem designs

| Spec | Status | Purpose |
|---|---|---|
| `2026-07-09-graph-ir-rust-port-design.md` | decided / partly built + addenda | Byte-exact port of the Graph-IR runtime/dataflow plane. Addenda supersede standalone/offline-only framing with `aiperf-graph` using `LocalSet`, `drive_sim`/`drive_real`, and live HTTP dispatch, and assign its sole product projection to the runner-only `graph` workload over online or Dynamo-offline backends. |
| `2026-07-10-aiperf-rust-scheduling-policy-sketch.md` | superseded | Early sketch of the credit-*policy* `Scheduler` (arrival patterns, session-vs-request slots, prefill-release-on-TTFT, absolute-schedule pacing, phase handoff). **Superseded by `2026-07-10-unified-graph-runtime-design.md`**, which realizes the same policy as `Workload`/`SlotPool`/`RatePool`/`Gate` on the graph executor. Kept for lineage. |
| `2026-07-11-aiperf-rust-request-rate-multiturn-design.md` | built + implementation addendum | **Faithful, source-grounded** realization of the `request-rate | chain` row: `RequestRateWorkload` is a **single-loop credit issuer** emitting **one turn per rate interval, continuation-priority** (not conversation arrivals), gated by session and prefill `SlotPool`s, bounded by `StopChecker`, and materialized from the segment store with Clock-deferred think time. Synthetic and dataset-backed runner requests share `ScheduledRuntime`, adaptive/ancillary actuators, native metrics, and reports. SimClock, real-HTTP library, and runner subprocess tests pin the product path; offline remains library-only. |
| `2026-07-11-aiperf-rust-user-centric-fixed-schedule-design.md` | built + addendum | The two scheduled strategies are built over shared Clock-backed `ScheduledRuntime` traits. User-centric implements virtual-history seeding, per-user cadence, churn/replacement, session caps, drain, and live user targets; fixed replay implements stable absolute ordering, auto/manual zero, and timestamp/delay/immediate precedence. Python Config v2 lowers both to runner phase DTOs; exact SimClock, real-HTTP library, and runner tests validate timing and reply splicing. Offline consumers remain library-only. |
| `2026-07-11-aiperf-rust-phase-runner-orchestrator-design.md` | built + implementation addendum | `aiperf-timing::phase` implements validated lifecycle/config/progress, direct observers, Clock-driven duration→grace→cancel→drain→force escalation, failure finalization, ordered warmup→profiling execution, seamless background returns, cross-phase debt drain, and cancellation that cannot advance phases. `aiperf::phase_runtime` connects workloads and resources; Python Config v2 plus runner protocol v1 now author ordered warmup/profiling phase lists. The Graph-IR phase consumer remains separate work. |
| `2026-07-11-aiperf-rust-adaptive-scale-design.md` | built + addenda | SLA-driven autoscaling is built in `aiperf-adaptive`: object-safe actuator/evaluator/step/window/controller seams, all four live actuators, Python-grounded SLA math with authoritative completion-token OSL/ITL reconciliation, `ramp_until_fail` discover/sustain/single-recovery control, Clock-paced assessments, and schema-v2 events/summary artifacts. Online and feature-gated offline modes share the injected-backend futures for paced session/prefill concurrency, scheduled request rate, and user-centric target users. |
| `2026-07-11-aiperf-rust-ancillary-timing-policy-design.md` | built + addenda | Three ancillary knobs are built: Clock-driven Linear/Exponential/Poisson ramps, seeded warmup-aware cancellation, and sticky round-robin URL selection. HTTP anchors cancellation to body-send completion; offline calls the steppable engine terminal operation. Backend-neutral paced/request-rate/user-centric paths consume their applicable ramps and cancellation over either clock/dispatcher pair. Fixed authored schedules reject ramps, the single in-process endpoint rejects URL selection, and Graph-IR still owns no arrival/slot actuator. |
| `2026-07-11-aiperf-rust-dag-branch-orchestrator-design.md` | superseded (by `aiperf-graph` dataflow) | The Python ~1000-line FORK/SPAWN credit-side branch orchestrator is **superseded wholesale by the `aiperf-graph` async-dataflow engine** — fan-out = out-edges, join gating = `ChannelRequirement.count` (static producer accounting), and sticky-routing / drain-observer / future-active-gate + the spawn-first / drain-after-return races are **deleted credit-protocol artifacts, not ported**. Residual = graph-build lowering (branch metadata → nodes/edges) + FORK/SPAWN materialization + session-cap wiring + whole-run FAIL_FAST. Lineage + reconciliation doc, not a build plan. |
| `2026-07-10-aiperf-rust-accuracy-accumulator-design.md` | static accuracy built in runner; agentic library built / runner pending + addenda | Rust owns ordinary scheduled inference, metrics, analysis, and native-v2 reporting; pinned Python providers own canonical prompts, tasks, loops, environments, private tests, and scoring. Static Lighteval is runner-reachable and subprocess-tested. The runner-only execution-surface spec owns the pending stateful `agentic + online_http` projection, authenticated callbacks, native-v2 joins, and restored Harbor/BrowserGym/MCPMark subprocess canaries. No benchmark-specific scorer exists in Rust. |
| `2026-07-10-aiperf-rust-metrics-accumulator-sweepline-design.md` | built + addenda | Built IO-free engine in `aiperf-metrics`: NaN-sparse column store, exact ragged replay, record/aggregate/derived metrics, SLO goodput, all effective/active and ICL-aware sweep curves, duration-weighted statistics, authoritative phase windows/timeslices, deterministic worker-local merge, and typed native-v2 `Reporter`. Online/scheduled/adaptive/accuracy adapters feed observer timing/classification/usage plus real HTTP traces; graph's lean workers feed request/token/usage facts directly and merge by worker; fixed schedules omit credit-relative metrics. Built telemetry producers remain separate consumers; genai-perf-v1 compatibility export remains unbuilt. |
| `2026-07-10-aiperf-rust-metric-catalog-appendix.md` | built (appendix) + addendum | Built catalog of 103 inherited Python identities plus 16 native sweep identities, with exact metadata/dependencies and implementations for every record/aggregate/derived row whose source data exists. Validation and a deterministic metadata fingerprint pin the graph. Telemetry-owned injected rows intentionally stay absent until their producer supplies values. |
| `2026-07-10-aiperf-rust-telemetry-accumulators-design.md` | built + addenda | Built as `aiperf-gpu-telemetry`, `aiperf-server-metrics`, `aiperf-network-latency`, and their Clock-paced `aiperf-runner` sidecars: DCGM/Python GPU sources, exact boundary counters, cadence gauges, energy/power/efficiency joins, Prometheus routing/fallback/terminal auto-disable, unit inference, vLLM/SGLang atlas, hybrid histogram estimation, fresh TCP-connect calibration, per-target population stats, and pre-summary flat-mean network delivery. The shared `Accumulator` seam supplies half-open queries and merge behavior; native-v2 carries GPU joins and server blocks, while compatibility JSONL feeds Python exporters. Addenda replace scrape reconstruction and final grace/settle waits with runtime-owned phase snapshots and preserve the dependency direction toward the IO-free metrics seam. |
| `2026-07-11-aiperf-rust-exporters-overhaul-design.md` | partially built + addenda | The typed, IO-free native-v2 `Reporter` and runner JSON writer are built. Metrics are name-keyed with typed series, metadata, timeslices, warmup/accuracy joins, absent omission, and non-finite nulls. Native CLI tables, logger, accuracy CSV, and legacy report helpers were deleted with the binary; Python owns presentation/export. Native CSV, genai-perf-v1 compatibility, warnings/insights, console replay, and timed uploaders remain unbuilt in Rust. |
| `2026-07-11-aiperf-rust-endpoints-design.md` | built behavior + ownership addendum | Faithful `Endpoint` trait layer with every tier-1 and tier-2 dialect, input-side ISL accounting, vendor response parsing, raw/template JMESPath + Jinja, and the inherited parse scars. `aiperf-transport` owns multipart encoding, Clock-paced video polling/download, inline-media retrieval/deduplication, endpoint streaming paths, and post-send cancellation across the full lifecycle. Its static identity/table and Python-parity configuration ownership are superseded by the runner-owned endpoint-registry design. |
| `2026-07-10-aiperf-rust-rng-derive-system-design.md` | built + addenda | Native hash-derived RNG substrate built as leaf crate `aiperf-rng`: `RngRoot::derive` + BLAKE3 seed derivation, canonical namespace constants, alloc-free `HashIdRandomGenerator`, `RandomGenerator`, generic sampler seams, five sampling distributions, and sequence distributions. Weighted mixtures cache validated cumulative weights; a Rust-internal profile canary pins the stream. Answers the coverage-gap ledger's RNG question: **NO cross-language byte parity anywhere** — internal reproducibility + order-independence + distributional/semantic parity via deterministic `Pcg64` + `rand_distr`. Dataset composition/samplers and ancillary timing policies consume it; broader scheduler/graph integration remains. |

### Historical precursors

These predate the standalone `crates/` workspace and describe a **different**
working tree (`dynamo-aiperf-native`) built *on* ai-dynamo application internals.
The current workspace extracted `loadgen-core`; default builds have no Dynamo
dependency, while `dynamo-offline` uses only the curated public mocker boundary.
These are lineage, not current architecture.

| Spec | Status | Purpose |
|---|---|---|
| `2026-07-09-dynamo-aiperf-shared-core-design.md` | superseded | Increment-1 walking skeleton sharing DynoSim's collector/driver via a curated facade; origin of the `RequestSink` / `RequestObserver` seam. |
| `2026-07-09-dynamo-aiperf-request-rate-tokenizer-design.md` | superseded | Increment-2 tokenizer-exact prompts + Poisson request-rate through the shared `WorkloadDriver`. |
