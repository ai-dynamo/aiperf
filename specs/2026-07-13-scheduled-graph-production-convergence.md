<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Scheduled ↔ graph execution paths: production convergence audit

**Date:** 2026-07-13
**Status:** analysis (code-grounded, current tree)
**Scope:** What the SCHEDULED and GRAPH online execution paths actually **share vs. duplicate in the product** (`rust/runtime`), grounded in code — not specs. Records reality and flags follow-ups; proposes **no** refactor.

> Motivated by a review of *"unify the scheduled and graph arrival/admission/failure policy seams, or write down why they're irreducibly different."* The answer, grounded in the runner: they are mostly **already unified** at the substrate; the arrival/admission "seams" are thin wrappers over shared primitives; the one genuine divergence is **failure behavior**.

## Why this exists (and a correction)

The module map ("graph owns trait-backed root/arrival/admission/placement/failure policy") and the library sinks (`graph/transport_sink.rs::TransportChatSink`) suggest the graph is a separate, lower-fidelity execution world. In the **product** it is not. The runner's graph sink is `RunnerGraphSink` (`rust/runtime/src/engine/graph_execution.rs:703`), which materializes through the endpoint registry and dispatches through the same `TransportSink` as the scheduled path. An earlier draft of this analysis (a "unify the dispatch seam" design) was **withdrawn**: it was built on the library `TransportChatSink` and stale spec claims and wrongly concluded the graph was metrics-lite. Code is truth.

## Convergence map (product paths)

| Concern | Scheduled (product) | Graph (product) | Verdict |
|---|---|---|---|
| Transport | `TransportSink` (`http.rs:466`) | `TransportSink` — same `new_multi_configured`, same low-level `dispatch_prepared_turn_collect_record_with_response_observer` | **shared** |
| Request struct | `PreparedHttpTurn` (`http.rs:209`) | `PreparedHttpTurn` (`graph_execution.rs:818`) | **shared** |
| Endpoints | registry + `format_payload` | registry + `format_payload` — multi-endpoint, **not** chat-only (`graph_execution.rs:763`) | **shared** |
| Measurement | `NativeMetricsObserver`, per-**worker**, drain-at-end (`turn_execution.rs:573`) | `NativeMetricsObserver`, per-**trace**, drain-per-node (`graph_execution.rs:689,857`) | **shared type**, diff lifecycle |
| Arrival | `IntervalGenerator` via `make_interval_generator` (`execute.rs:1818`) | **same handle**, wrapped in `IntervalGraphArrival` (`graph_phase_runtime.rs:1083`) | **shared primitive** + thin wrapper |
| Admission | `SlotPool` session/prefill (`execute.rs:169,186`) | **same `SlotPool`**, wrapped in `SlotPoolTraceAdmission`/`PrefillSlotNodePolicy` (`graph_phase_runtime.rs:1138`) | **shared primitive** + thin wrapper |
| Stop bounds | `StopChecker` + shared phase deadline | source budgets + shared phase deadline (`GraphStopPolicy` vestigial) | **shared deadline** |
| Phase orchestration | `ClockPhaseOrchestrator`/`Runner` | same | **shared** |
| Placement | `ThreadPerCoreHttpExecutionBackend`, per-request (`turn_execution.rs:202`); single-thread if `workers==1` | `ThreadPerCoreGraphTraceExecutionBackend`, per-trace (`graph/placement.rs:43`) | **duplicate impls**, diff granularity |
| Workload driver | `RequestRateWorkload`/`ScheduledRuntime` | `GraphWorkload::execute` | **different** |
| Failure | record + continue (**resilient**) | abort trace + stop run (**fail-fast**) | **genuinely different behavior** |

## Already shared (the substrate)

Transport, prepared-request struct, endpoint registry + payload formatting, the `NativeMetricsObserver` measurement type (both reconcile authoritative prompt/completion usage), the `IntervalGenerator` **handle**, the `SlotPool` **instances**, and the phase lifecycle are all shared. Arrival and admission "policies" on the graph side (`IntervalGraphArrival`, `SlotPoolTraceAdmission`, `PrefillSlotNodePolicy`) are **thin trait wrappers over the same timing primitives** the scheduled path uses directly — the ramp and adaptive actuators mutate the same objects on both paths.

## Genuinely different

1. **Failure behavior (the sharp one).** Graph online is **fail-fast**: `AbortTraceNodeFailurePolicy` (`graph_execution.rs:642`) + `FailFastRunFailurePolicy` (`graph_phase_runtime.rs:1137`) + coordinator `ensure!(phased.workload.failed == 0, …)` (`execute.rs:1168`) → the **whole benchmark errors** on the first failed node/trace. Scheduled online is **resilient**: transport failures are recorded and the run continues (`scheduled.rs:980-1004`); only issuance/materialization errors abort (`request_rate.rs:398-415`). Graph fail-fast mirrors Python DAG `FAIL_FAST` parity (see the dag-branch-orchestrator spec); the scheduled path does not. **The two product paths behave differently when a request fails** — worth confirming as intended vs. reconciling onto one consistent, configurable policy.
2. **Placement granularity.** Two thread-per-core worker pools, both injected by `native_execution_factories()` (`execution_factories.rs:147-148`): scheduled routes **per request** (round-robin over `WorkerCommand` mpsc), graph routes **per whole trace**. Duplicate worker-pool machinery; different unit.
3. **Workload driver.** Scheduled `RequestRateWorkload`/`ScheduledRuntime` issue loop vs. graph `GraphWorkload::execute` admission loop.
4. **Dispatch entry + observer locus.** Scheduled `dispatch_prepared_turn_measured` (observer wired internally, per-worker, drain-at-end) vs. graph `dispatch_prepared_turn_collect_record` (observer wired by hand in `RunnerGraphSink`, per-trace, drain-per-node). Same `TransportSink`, same low-level dispatch fn underneath.

## Answering the original question

- **Arrival:** already shared (same `IntervalGenerator` handle). "Unifying" = deleting a wrapper. Low value.
- **Admission:** already shared (same `SlotPool`). Same. Low value.
- **Failure:** genuinely different *behavior* — the one item worth a deliberate decision.

So the seams are **not irreducibly different**, and mostly **not even meaningfully duplicated**. The exception is failure, which differs by product choice, not necessity.

## Remaining real duplication

The only large remaining duplication is structural: two thread-per-core placement backends + two workload drivers. That is the "unify scheduled+graph under one substrate" endgame recorded in `2026-07-12-cellular-ready-seams-and-roadmap.md` §S5 (deferred to aiperf-v2). This audit does not change that deferral.

## Incidental cleanups surfaced (independent of any unification)

- **Discarded coordinator observer.** `ConfiguredDispatcher::dispatch_turn` ignores the `ScheduledRuntime`-built `NativeMetricsObserver` (`execute.rs:3573`; comment `:3585-3586`); the report comes only from drained per-worker observers, so the coordinator observer is dead work per request.
- **Vestigial `GraphStopPolicy`.** Never set in the runner (default `UnlimitedGraphStop`); duration comes from the shared phase deadline. Removal candidate.
- **Unused-in-runner graph policies.** `ScheduledGraphArrival`, `ContinueRunFailurePolicy`, `ResilientNodeFailurePolicy`, `DurationGraphStop` exist in the library but are never wired by the product.
- **Non-streaming dispatch on a streaming transport.** Scheduled uses `execute_turn_measured` (not `_streaming`) though the backend advertises streaming; SSE is parsed in-worker, no live frames forwarded to the coordinator.

## Method / trust note

Grounded in `rust/runtime/src/engine/{execute,turn_execution,graph_execution,graph_phase_runtime}.rs` and `rust/runtime/src/{http,metrics,scheduled,request_rate}.rs`, verified against code (the surprising `_observer`-discard and `ensure! failed==0` claims were re-read directly). Specs are intent; where they disagreed with the runner (graph "metrics-lite"; a retained `BufferedObserver`), the code won.

## Related

- `2026-07-12-cellular-ready-seams-and-roadmap.md` §S5 — unify scheduled+graph, deferred to aiperf-v2.
- `2026-07-12-scheduled-worker-local-accumulation.md` — Track A A1, the worker-local measured seam both paths share.
- `2026-07-11-aiperf-rust-dag-branch-orchestrator-design.md` — the Python DAG `FAIL_FAST` semantics the graph path mirrors.
