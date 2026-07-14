<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Greenfield execution vocabulary — the unified-substrate target names

**Date:** 2026-07-13
**Status:** design (north-star / greenfield ideal — **not** an incremental plan). **Revised 2026-07-13** after a two-reviewer naming pass: `Trace`→`RequestGraph` (≈600 in-tree/OTel overload), `Backend`→`Dispatcher` (SUT word + saturated `*Backend` namespace), kept `Observer` (dropped `Recorder`), `Bounds`→`Limits`, `Topology`→`GraphAdjacency`.
**Scope:** The names an execution runtime *would* use if written fresh, with no
migration constraints. Records the target vocabulary for the deferred "unify
scheduled + graph" v2 substrate (`2026-07-12-cellular-ready-seams-and-roadmap.md`
§S5). The **incremental** counterpart — what to actually rename in the live tree
today — is `2026-07-13-p1-generic-execution-substrate-names.md`.

> The honest framing: greenfield, the answer is not a rename table. Roughly half
> of today's names exist only because there are **two of everything** — a
> scheduled path and a graph path. Accept the one fact the convergence audit
> (`2026-07-13-scheduled-graph-production-convergence.md`) kept pointing at — *a
> flat request is a single-node trace* — and the duplication collapses. The clean
> names fall out of one model; the count drops because the model is one thing, not
> because of clever naming.

## The one model — three nouns

- **`Request`** — one dispatch to the server under test. Collapses `TurnToSend`,
  `PreparedHttpTurn`, `HttpRequest`, `GrpcRequest`, and "a graph node's dispatch."
- **`RequestGraph`** — a DAG of `Request`s. A flat benchmark is a **single-node**
  graph; multiturn is a **linear** graph; an agentic graph is the general case.
  Collapses "turn-chain / session / conversation" *and* "graph trace" into one word.
  (Named `RequestGraph`, not `Trace`: bare `Trace` is already saturated in-tree —
  replay-input files, `TraceCollector`/`TraceSimulationReport`, `HttpTrace` — and
  clashes with OTel's server-side span-tree meaning. `Request` stays the primary
  noun a flat run meets; the graph wrapper is only seen inside the DAG driver.)
- **`Outcome`** — the result of one `Request`: status + timing + usage + body.
  **One** type, replacing the five outcome-shaped types today (`TurnDispatchOutcome`,
  `GraphReply`, `HttpTurnDispatchResult`, `MeasuredTurnOutcome`, `DispatchResult`).

## The seams — one each, not scheduled-vs-graph pairs

| Greenfield | verb | Collapses today's… |
|---|---|---|
| **`Dispatcher`** | `dispatch(&Request) -> Outcome` | `TurnDispatcher` + `GraphSink` + `HttpTurnExecutionBackend` + `HttpRequestDispatcher` + `TransportSink`'s ~6 methods. Impls are the three-modes axis: `HttpDispatcher`, `GrpcDispatcher`, `MockDispatcher`, `SimDispatcher`. (Named `Dispatcher`, not `Backend`: "backend" means the server-under-test to load-test practitioners, and `*Backend` already names the worker pool + 3 other traits in-tree.) |
| **`Observer`** *(kept)* | `on_arrival` / `on_first_token` / `on_token` / `on_usage` / `on_terminal` | `RequestObserver` + `NativeMetricsObserver` + `CollectorObserver` + the graph's hand-rolled `register_metadata`/`record_response` wiring, unified to one `MetricsObserver` impl. Keep the name `Observer` — it is the textbook GoF pattern and already the dominant seam; do **not** rename to `Recorder` (collides with the `metrics` crate's `Recorder` trait). |
| **`Workload`** | `next_graph() -> Option<RequestGraph>` | `Workload` + `GraphWorkload` + `GraphTraceSource`. Impls: `RateWorkload`, `ConcurrencyWorkload`, `UserWorkload`, `ReplayWorkload`, `GraphWorkload`. |
| **`Pacer`** | `next_arrival() -> <Clock time>` | `IntervalGenerator` **and** `GraphArrivalPolicy` (the same concept). Impls `Poisson`/`Gamma`/`Constant`/`Burst`/`Replay`. The `ArrivalPattern` enum disappears — it *is* the trait. |
| **`Admission`** | `admit() -> Slot` | `TraceAdmissionPolicy` + `NodeDispatchPolicy` + direct `SlotPool` use. Whole-graph vs per-request are two call sites of `admit`. |
| **`Limits`** | `may_send()` / `may_start()` | `StopChecker` + `GraphStopPolicy` + `CyclingGraphTraceSource` budgets. Conditions `RequestLimit`/`SessionLimit`/`DurationLimit`/`Cancelled` — so the container is `Limits`, self-consistently (not `Bounds`, which is overloaded with generic/trait bounds). |
| **`FailurePolicy`** | `on_failure(&Outcome) -> Disposition` | `NodeFailurePolicy` + `RunFailurePolicy` + the scheduled `has_failed` latch. `Resilient`/`FailFast`, one seam — which also *forces* a single answer to the graph-aborts-vs-scheduled-tolerates inconsistency instead of hiding it in two places. |
| **`WorkerPool`** | `run(RequestGraph)` | Both `ThreadPerCoreHttpExecutionBackend` **and** `ThreadPerCoreGraphTraceExecutionBackend`. One thread-per-core pool; each worker owns a `Dispatcher` + a `GraphExecutor`. |
| **`GraphExecutor`** | `execute(&RequestGraph)` | Today's `TraceExecutor`, kept as the genuinely irreducible DAG driver (firing gates, splicing) — renamed off `Trace` for the same overload reason. A single-node graph runs through it trivially. |

~9 core seams replacing ~25 in the same space today.

## Kept almost verbatim (already right)

`Clock`; `SlotPool` + `Slot` (rename `SlotGuard`); the whole **phase** layer
(`Phase`, `PhaseRunner`, `PhaseLifecycle`, `PhaseOrchestrator`) — already clean and
already shared. `RampStrategy` → `Ramp`; `UrlSelector` → `EndpointSelector` (URL is
too HTTP-specific); `CancellationPolicy` → `Cancellation`.

## Pure-taste calls

- `RealClock`/`SimClock` → **`WallClock`/`VirtualClock`** — the actual distinction;
  "real vs sim" is jargon.
- `graph::scheduler::Scheduler` → **`GraphAdjacency`** — a static-edge adjacency
  view; it schedules nothing. (Not `Topology`: that collides with the deployment
  `ReportDynamoTopology` / `OfflineTopology`.)
- **Three** warmup/profiling enums collapse to one `PhaseKind`: `cancellation::Phase`
  + `phase::PhaseKind` + `metrics_core::window::Phase`.
- Arrival impls lose disambiguating suffixes (`Poisson`, not `PoissonArrival`)
  *because* `PoissonRamp` now lives under `Ramp::Poisson`.

## Caveat

This is the unconstrained ideal. It is **not** what to do to the live tree
incrementally — greenfield you would never write the second placement seam in the
first place, so `RequestExecutor` vs `TraceExecutor` would simply be `WorkerPool`.
The incremental, blast-radius-aware step is the P1 spec; this document is the
destination that step is walking toward.

## Related

- `2026-07-13-scheduled-graph-production-convergence.md` — the audit establishing that a flat request already shares the substrate with a graph trace.
- `2026-07-13-p1-generic-execution-substrate-names.md` — the incremental rename step toward this vocabulary.
- `2026-07-12-cellular-ready-seams-and-roadmap.md` §S5 — the deferred unify-scheduled+graph substrate this vocabulary names.
