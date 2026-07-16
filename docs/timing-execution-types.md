<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Timing & execution runtime — type catalog

Every major public type across the **timing** and **executing** layers of the Rust
runtime, extracted from the code (post-rename `rust/` tree, 2026-07-13) with each
type's own doc comment. **◆** marks a trait (an extension seam). Code is truth —
verify before relying on any row; names drift.

Layers: [1 Clock](#1-clock) · [2 Measurement foundation](#2-measurement--dispatch-foundation-loadgen-core) ·
[3 Timing policy](#3-timing-policy-aiperf_runtimetiming) · [4 Phase orchestration](#4-phase-orchestration-aiperf_runtimetimingphase) ·
[5 Scheduled execution](#5-scheduled-execution) · [6 HTTP/gRPC dispatch + measurement](#6-http--grpc-dispatch--native-measurement) ·
[7 Graph execution plane](#7-graph-execution-plane-aiperf_runtimegraph) · [8 Runner wiring](#8-runner-concrete-wiring-aiperf_runtimerunner_protocol) ·
[Addendum: recommended renames](#addendum--recommended-renames)

## 1. Clock — `aiperf_runtime::clock`
| Type | Kind | Description |
|---|---|---|
| `Clock` | trait ◆ | A sleepable time source (`now_ns`/`sleep`/`is_virtual`). |
| `RealClock` | struct | Monotonic wall clock with ns-precision `timerfd` sleeps. |
| `RealClockAnchor` | struct | Copyable monotonic origin shared by cooperating real-clock runtimes. |
| `SimClock` | struct | Virtual-time clock advanced by the runtime driver pump. |

## 2. Measurement / dispatch foundation — `loadgen-core`
| Type | Kind | Description |
|---|---|---|
| `RequestSink` | trait ◆ | Dispatch one request of type `R` to terminal, resolving on completion. |
| `Dispatchable` | trait ◆ | What every dispatchable request exposes to the sink and collector. |
| `RequestObserver` | trait ◆ | Measurement hook fed by any sink (`on_arrival`/`on_admit`/`on_token`/`on_usage`/`on_terminal`); ms timestamps. |
| `ObservedUsage` | struct | Authoritative server-reported token usage for one request. |
| `ObservedTokenKind` | enum | Semantic class of one streamed token-like delta (Output/Reasoning). |
| `ObservedEndpointMetrics` | struct | Endpoint-specific modality facts feeding native metrics. |
| `CollectorObserver` | struct | Collects measurement events from any sink into one `TraceCollector`. |
| `TraceCollector` | struct | Accumulates per-request events → `TraceSimulationReport`. |
| `TraceSimulationReport` | struct | Aggregate request/throughput/latency/goodput report (+ `Trace*Stats`). |
| `ReplayTerminalStatus` | enum | Terminal classification (Completed/Failed/Canceled). |
| `PerRequestRecord` | struct | Flat per-request record for `--report-jsonl`. |
| `SlaThresholds` / `TraceGoodputStats` | struct | SLA classification + goodput (throughput restricted to SLA-satisfying requests). |

## 3. Timing policy — `aiperf_runtime::timing`

**Arrival** (`intervals.rs`)
| Type | Kind | Description |
|---|---|---|
| `IntervalGenerator` | trait ◆ | Produces successive inter-arrival intervals in ns; `set_rate` for ramping. |
| `ArrivalPattern` | enum | Selects the inter-arrival distribution. |
| `Poisson` / `GammaProcess` / `Constant` / `ConcurrencyBurst` | struct | The four arrival impls (exponential / tunable-burstiness / fixed-period / zero-delay). |

**Admission** (`slots.rs`)
| Type | Kind | Description |
|---|---|---|
| `SlotPool` | struct | Dynamic-capacity concurrency semaphore with debt-tracked graceful drain. |
| `SlotGuard` | struct | RAII handle for one acquired slot; releases on drop. |
| `ConcurrencyManager` | struct | Bundle of two independent `SlotPool`s (session + prefill). |
| `ConcurrencyStats` | struct | Instrumentation counters for a `SlotPool`. |

**Stop** (`stop.rs`)
| Type | Kind | Description |
|---|---|---|
| `StopCondition` | trait ◆ | A single ordered stop predicate over `RunState` + `now_ns`. |
| `StopChecker` | struct | Evaluates the ordered stop-condition chain for a run. |
| `StopConfig` | struct | Configured stop thresholds (each `Some` activates a condition). |
| `RunState` | struct | Read-only snapshot of counters/lifecycle flags the conditions read. |
| `Lifecycle` / `RequestCount` / `SessionCount` / `Duration` | struct | The four stop conditions. |

**Ancillary** (`cancellation.rs` / `url_selection.rs` / `ramping.rs` / `user_centric.rs`)
| Type | Kind | Description |
|---|---|---|
| `CancellationPolicy` | trait ◆ | Pluggable decision policy for arming a post-send cancellation timer. |
| `BernoulliFixedDelay` | struct | Bernoulli selection with one constant post-send delay. |
| `Phase` | enum | Benchmark phase relevant to ancillary timing (Warmup/Profiling). |
| `UrlSelector` | trait ◆ | Pluggable endpoint-index selector. |
| `RoundRobinUrlSelector` | struct | Sequential endpoint selection with wraparound. |
| `RampStrategy` | trait ◆ | Object-safe curve seam used by `RampDriver`. |
| `LinearRamp` / `ExponentialRamp` / `PoissonRamp` | struct | The three ramp curves. |
| `RampDriver` / `RampHandle` / `RamperConfig` | struct | Async driver applying a strategy to a setter closure; control handle; params. |
| `UserCentricPlan` / `InitialUser` | struct | Deterministic steady-state seeding plan for the user-centric strategy. |

## 4. Phase orchestration — `aiperf_runtime::timing::phase`
| Type | Kind | Description |
|---|---|---|
| `PhaseExecution` | trait ◆ | Workload/backend adapter driven by `PhaseRunner` — the seam both scheduled & graph implement. |
| `PhaseExecutionFactory` | trait ◆ | Factory creating fresh execution state per phase. |
| `PhaseRunner` / `ClockPhaseRunner` | trait ◆ / struct | Object-safe single-phase driver; clock-driven default (owns duration→grace→cancel→drain→force). |
| `PhaseOrchestrator` / `ClockPhaseOrchestrator` | trait ◆ / struct | Multi-phase orchestration; ordered default (warmup→profiling). |
| `PhaseRunnerFactory` / `ClockPhaseRunnerFactory` | trait ◆ / struct | One fresh runner per phase. |
| `PhaseObserver` | trait ◆ | Local observer for phase lifecycle/progress (Noop/Recording/Console impls). |
| `PhaseLifecycle` | struct | CREATED→STARTED→SENDING_COMPLETE→COMPLETE state machine. |
| `PhaseContext` | struct | Local-loop context shared with one execution strategy. |
| `PhaseProgress` | struct | Cloneable progress handle shared by issuer + return callbacks. |
| `PhaseConfig` | struct | Validated policy for one bounded issuance phase. |
| `PhaseKind` / `PhaseState` / `PhaseCompletionReason` / `GracePeriod` | enum | Role / lifecycle state / why-complete / return-wait policy. |
| `NoopPhaseExecution(Factory)` | struct | Empty-plan reference impls. |
| `PhaseStats` / `PhaseBranchStats` / `ReleasedStuckSlots` | struct | Snapshots + DAG counters + recovered-slot count. |

## 5. Scheduled execution

**Core runtime & seams** (`scheduled.rs`)
| Type | Kind | Description |
|---|---|---|
| `ScheduledRuntime` | struct | Shared facilities injected into a `Workload` (issuer, stop-checker, ancillary policy). |
| `Workload` | trait ◆ | Schedule-generating workload seam shared across online & offline backends. |
| `TurnDispatcher` | trait ◆ | Transport/backend seam consumed by scheduled multi-turn workloads. |
| `TurnDispatchOutcome` | struct | Terminal result returned by a `TurnDispatcher`. |
| `ModelResponseMetadata` | struct | Endpoint-normalized assistant/terminal metadata (reasoning/cache/finish/token-ids…). |
| `DispatchCancellation` | trait ◆ | Replaceable cancellation latch for one admitted dispatch. |
| `TurnResponseObserver` | trait ◆ | Backpressured endpoint-normalized response-frame consumer. |
| `IssuanceGate` | trait ◆ | Optional external admission gate above ordinary stop conditions (adaptive uses it). |
| `TurnRecordProcessor` | trait ◆ | Post-dispatch record-processing seam. |
| `TurnLifecycleObserver` | trait ◆ | Synchronous per-turn lifecycle seam for phase/accounting. |
| `ScheduledRunReport` / `TurnTimingRecord` / `ScheduleTimingAnalysis` / `UserControlSnapshot` | struct | Report + per-turn timing + schedule fidelity + user-pool snapshot. |
| `SingleTurnDatasetWorkload` | struct | One-pass single-turn dataset workload. |

**Workload impls**
| Type | Kind | Description |
|---|---|---|
| `RequestRateWorkload` (`request_rate.rs`) | struct | Single-loop, continuation-priority request-rate `Workload`. |
| `UserCentricWorkload` + `UserPool` + `UserTargetController`◆ + `UserCentricControl` | struct | Virtual-history-seeded per-user workload with churn + adaptive target. |
| `FixedScheduleWorkload` + `FixedSchedule` + `FixedScheduleSource`◆ + `DatasetFixedScheduleSource` | struct | Absolute-timestamp replay workload. |
| `SkeletonWorkload` (`workload.rs`) | struct | Synthetic N-chat-request workload. |

**Phase runtime glue** (`phase_runtime.rs`)
| Type | Kind | Description |
|---|---|---|
| `ScheduledPhaseController`◆ (+ `Noop`/`Ramp`) | trait | Optional phase-owned actuator/ramp lifecycle. |
| `ScheduledPhaseResources`◆ (+ `Noop`/`SlotPoolPhaseResources`) | trait | Shared admission resources configured/cleaned at phase boundaries. |
| `ScheduledPhaseSidecar`◆ | trait | Async control-plane work synced to a phase's hard barriers. |
| `ScheduledRuntimeExtension`◆ (+ `…Parts`) | trait | Object-safe factory for one phase-local runtime policy extension. |
| `ScheduledPhasePlan` | struct | One prepared phase lowered into the shared scheduled runtime. |
| `PhasedScheduledRunReport` (+ Deferred/Aggregated variants) | struct | Ordered phased-run results. |

*Runner-internal (not `pub`): `ScheduledPhaseExecution` + `…Factory` — the `PhaseExecution` impl wrapping a `Workload` + `ScheduledRuntime`.*

## 6. HTTP / gRPC dispatch & native measurement
`aiperf_runtime::http` / `aiperf_runtime::grpc` / `aiperf_runtime::metrics`
| Type | Kind | Description |
|---|---|---|
| `TransportSink` | struct | Live sink over `transport_http` (hyper+Clock); shared by the scheduled worker & the graph sink. |
| `HttpTurnExecutionBackend` | trait ◆ | Pluggable execution *placement* behind the one logical turn dispatcher (reactor/thread-per-core/remote). |
| `HttpRequestDispatcher` | trait ◆ | Response-capturing request-dispatch seam used by the shared paced issuer. |
| `PreparedHttpTurn` | struct | Owned, scheduler-free execution command handed to a backend. |
| `PreparedHttpEndpoint` | enum | Endpoint selection retained by one prepared command. |
| `MeasuredTurnContext` / `MeasuredTurnOutcome` | struct | Coordinator-supplied measurement facts / worker-local execution result (+ optional live record). |
| `HttpTurnDispatchResult` | struct | Terminal result: `{ outcome, request_payload, record }`. |
| `HttpRequest` / `HttpDispatchResult` | struct | Slim online request / captured response. |
| `GrpcTransportSink` / `GrpcRequest` | struct | Native gRPC scheduled sink + request (also impls `HttpTurnExecutionBackend`). |
| `NativeMetricsObserver` | struct | Observer-backed native metrics collector — the shared measurement impl for both paths. |
| `RequestMetricMetadata` / `NativeResponseMetadata` | struct | Pre-arrival dimensions / post-terminal transport facts. |
| `NativeMetricsFinalizer` / `NativeMetricsCollection` / `ObserverTee` | struct | Post-drain reduction / records+aggregate / local observer fan-out. |

## 7. Graph execution plane — `aiperf_runtime::graph`

**Executor & driver** (`executor.rs`/`runtime.rs`/`execution.rs`/`scheduler.rs`/`context.rs`)
| Type | Kind | Description |
|---|---|---|
| `TraceExecutor` | struct | Async-dataflow trace executor for one resolved graph (generic over dialect msg). |
| `TraceContext` | struct | Per-trace mutable state passed into every node's fire path. |
| `GraphTraceExecutionBackend` | trait ◆ | Object-safe backend for one complete root trace. |
| `LocalGraphTraceExecutionBackend` | struct | Local impl backed by the canonical `TraceExecutor`. |
| `SimEventSource` | trait ◆ | Externally-clocked discrete-event source consumed by the virtual-time pump. |
| `Handle` | struct | Task handle: spawn, clock access, sleeping. |
| `Scheduler` | struct | Pure adjacency view over the graph's static edges (firing-gate topology). |
| `RunOutcome` / `SimStep` / `SimDriveError` | struct/enum | Drive-to-quiescence result / DES step / pump failure. |

**Workload & policy seams** (`workload.rs`/`policy.rs`)
| Type | Kind | Description |
|---|---|---|
| `GraphWorkload` | struct | Policy-composed coordinator delegating whole traces to one backend. |
| `GraphTraceSource` | trait ◆ | Root-trace selection (`VecGraphTraceSource` / `CyclingGraphTraceSource`). |
| `GraphArrivalPolicy` | trait ◆ | Arrival pacing (`Immediate` / `Scheduled`-offset / `Interval`-generator). |
| `TraceAdmissionPolicy` (+`Permit`) | trait ◆ | Root-session admission (`Unlimited` / `SlotPool`-backed). |
| `GraphStopPolicy` | trait ◆ | Run-level admission deadline (`Unlimited` / `Duration`). |
| `NodeDispatchPolicy` (+`Permit`) | trait ◆ | Per-node admission/ancillary (`Noop`/`PrefillSlot`/`Cancellation`/`Composite`). |
| `NodeFailurePolicy` | trait ◆ | Per-node failure disposition (`Resilient`=continue-with-empty / `AbortTrace`). |
| `RunFailurePolicy` | trait ◆ | Run-level admission after failures (`Continue` / `FailFast`). |
| `GraphWorkloadObserver` | trait ◆ | Phase/run hooks emitted by the workload. |
| `NodeFailureKind` / `NodeFailureDisposition` | enum | Failure classification / decision. |
| `GraphTracePlan` / `GraphWorkloadReport` / `GraphTraceRunResult` | struct | One root command / aggregate outcome / one drained trace. |

**Dispatch sink & channel store** (`sink.rs`/`transport_sink.rs`/`channel_store.rs`/`reducers.rs`)
| Type | Kind | Description |
|---|---|---|
| `GraphSink` | trait ◆ | The per-node dispatch seam (generic over dialect message). |
| `GraphReply` / `GraphReplyStatus` / `GraphDispatchOptions` | struct/enum | Splice value + status / terminal class / per-node directives. |
| `EchoSink` / `TransportChatSink` | struct | Test double / live OpenAI-chat sink (library — **not** the product sink; see §8). |
| `VersionedChannelStore` / `VersionCapture` | struct | Per-trace append-only channel store / firing snapshot. |
| `ChanVal` | enum | A channel value (sentinel or JSON). |

**Placement** (`placement.rs`)
| Type | Kind | Description |
|---|---|---|
| `GraphTraceExecutionBackendFactory` | trait ◆ | Builds one worker-local backend per OS thread. |
| `ThreadPerCoreGraphTraceExecutionBackend` | struct | Native thread-per-core **whole-trace** placement. |

**Graph IR data model** (`model.rs` — inputs the executor consumes): `GraphRecord`, `TraceRecord`, `ParsedGraph`, `LlmNode`, `StaticEdge`, `ChannelSpec`, `ChannelRequirement`, `Count`, `PromptItem`, `ChannelType`, `ReducerName`.

## 8. Runner concrete wiring — `aiperf_runtime::runner_protocol`
| Type | Kind | Description |
|---|---|---|
| `RunnerExecutionFactories` | struct | The exact execution-factory universe (HTTP + graph + gRPC + readiness). |
| `HttpExecutionBackendFactory` | trait ◆ | Composition seam for local/thread-per-core/remote request placement. |
| `NativeHttpExecutionBackendFactory` | struct | Native factory → `ThreadPerCoreHttpExecutionBackend` (per-request), or single-thread if `workers==1`. |
| `RunnerGraphPlacementFactory` | trait ◆ | Composition seam for whole-trace graph placement. |
| `NativeRunnerGraphPlacementFactory` | struct | Stock thread-per-core whole-trace placement. |
| `NativeGrpcExecutionBackendFactory` | struct | V2-only native gRPC execution factory. |
| `HttpPreparedEndpointTableFactory` | trait ◆ | Worker-local prepared endpoint-table construction. |

*Runner-internal (not `pub`) but central to executing:*
- `RunnerGraphSink` — **the product graph sink** (full-fidelity: endpoint registry, `PreparedHttpTurn`, `NativeMetricsObserver`). Judge graph capability from this, not the library `TransportChatSink`.
- `ConfiguredDispatcher` — the `TurnDispatcher` → `execute_turn_measured` adapter used by the scheduled path.
- `RunnerGraphWorkerBackend`, `ThreadPerCoreHttpExecutionBackend` — the two thread-per-core worker pools.

---

## Addendum — recommended renames

Grouped by priority. Renames are mechanical, no behavior change; sequence them so the suite stays green. The **High** set is worth doing standalone (correctness/clarity); the **Medium** set is the shared-substrate work already in `specs/2026-07-13-p1-generic-execution-substrate-names.md`; **Low** is polish. *(Revised 2026-07-13 after a two-reviewer naming pass: `Duration`→`DurationLimit` (was `DurationStop`), three-way `Phase` collapse, and the Medium picks avoid the already-taken `PreparedRequest`/`TraceExecutor` — now `PreparedTurn`/`TracePlacement`.)*

### High — hazards & misnomers (independent, do anytime)

| Now | → | Why |
|---|---|---|
| `timing::stop::Duration` | `DurationLimit` | A bare `Duration` in a timing module shadows the ubiquitous `std::time::Duration` — a genuine footgun. (`*Limit`, not `*Stop`: parallels the sibling conditions and the greenfield `Limits` family; both reviewers preferred it.) |
| `timing::stop::{RequestCount, SessionCount}` | `RequestLimit`, `SessionLimit` | Consistency with `DurationLimit`; makes their role legible without opening `stop.rs`. `Lifecycle` (the cancel/complete gate, not a count) stays as-is. |
| `graph::scheduler::Scheduler` | `GraphAdjacency` | Its own doc: *"a pure adjacency view over the parsed graph's static edges; holds no per-trace state."* It schedules nothing. (Not `Topology` — that collides with `ReportDynamoTopology` / `OfflineTopology`.) |
| `cancellation::Phase` + `phase::PhaseKind` + `metrics_core::window::Phase` | one shared `PhaseKind` | **Three** byte-identical `{Warmup, Profiling}` enums across modules. Keep `phase::PhaseKind`, delete the other two, re-export if needed. Removes two silent duplicates. |

### Medium — shared-substrate generic names (the P1 spec)

Covered in full by `specs/2026-07-13-p1-generic-execution-substrate-names.md`; summarized here for cross-reference:

| Now | → |
|---|---|
| `HttpTurnExecutionBackend` | `RequestExecutor` (serves gRPC too; per-request placement) |
| `GraphTraceExecutionBackend` | `TracePlacement` (per-trace placement — parallel name; not `TraceExecutor`, already the DAG driver) |
| `PreparedHttpTurn` | `PreparedTurn` (not `PreparedRequest`, already an `endpoints` type) |
| `MeasuredTurnContext` / `MeasuredTurnOutcome` | `MeasuredContext` / `MeasuredOutcome` |
| `HttpTurnDispatchResult` | `DispatchResult` |
| `TransportSink` dispatch methods (~6) | `dispatch_measured` + `dispatch_collect[_streaming]` (2 primitives) |
| `ThreadPerCoreHttpExecutionBackend` / `…GraphTrace…` | `ThreadPerCoreRequestExecutor` / `ThreadPerCoreTracePlacement` |

Plus one the P1 spec lists as deferred but I'd still recommend:

| Now | → | Why |
|---|---|---|
| `TurnDispatchOutcome` | `ResponseOutcome` | Kills the nested `DispatchResult.outcome` double-`outcome`. Deferred in P1 only because the read surface is wide, not because it's wrong. |

### Low — disambiguation & polish

| Now | → | Why |
|---|---|---|
| `intervals::{Poisson, Constant, GammaProcess, ConcurrencyBurst}` | `PoissonArrival`, `ConstantArrival`, `GammaArrival`, `BurstArrival` | Single-word `Poisson` (arrival) collides conceptually with `PoissonRamp` (ramp); the `*Arrival` suffix reads clearly at construction sites. |
| `graph::runtime::Handle` | `GraphTaskHandle` | `Handle` is maximally generic; the name says nothing about what it handles. |
| `graph::transport_sink::TransportChatSink` | *(doc-clarify, or `LibraryChatSink`)* | It is a library/bench sink, **not** the product path (that is the runner's `RunnerGraphSink`). Its current doc "Live OpenAI-chat sink" invites exactly the mistake of treating it as the graph's real dispatch. |

**Not recommended for rename** (intentional, leave alone): `GraphSink`/`GraphReply` (genuinely graph-specific per-node splice), `HttpRequest` (leaf transport DTO, wide blast radius), `NativeMetricsObserver`, `SlotPool`, `Clock`. And the two placement seams stay **two** traits — merging `RequestExecutor` + `TracePlacement` is the deferred structural work, not a rename.
