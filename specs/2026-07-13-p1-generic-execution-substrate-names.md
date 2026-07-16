<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# P1 — Generic shared names for the execution substrate

**Date:** 2026-07-13
**Status:** built — see the 2026-07-14 addenda. Group A/B identifier + method renames and the method-count *fold* are implemented; the `inference_dimensions` signature decoupling is intentionally not done (it would double-build `PreparedTurn` on the hot path — see the 2026-07-14b addendum). **Revised 2026-07-13** after a naming review caught two collisions: `PreparedRequest` and `TraceExecutor` are already-taken names in crate `aiperf-runtime`, so the picks are now `PreparedTurn` and `TracePlacement` (see §3 note).
**Scope:** A **naming + dispatch-method-consolidation** pass over the execution substrate the scheduled and graph online paths *already share*. Pure rename + de-duplication — **no behavior change, no structural merge**. Premised on the production audit `2026-07-13-scheduled-graph-production-convergence.md`.

> This supersedes the withdrawn "unify the dispatch seam" P1 draft, which was built on the false premise that the graph path was metrics-lite. The graph path (`RunnerGraphSink`) is full-fidelity and already calls the same `TransportSink` with the same `PreparedHttpTurn` and `NativeMetricsObserver`. What's actually wrong is that the shared substrate is **named path-specifically** (`Http*` / `Turn*`) and reached through a **sprawl of ~6 near-duplicate `TransportSink` dispatch methods**, so the convergence is invisible in the code.

## 1. Problem

The scheduled and graph online paths share these exact types/instances (audit §"Already shared"), but under names that imply otherwise:

- **Shared DTOs, `Http*`/`Turn*`-named:** `PreparedHttpTurn`, `MeasuredTurnContext`, `MeasuredTurnOutcome`, `HttpTurnDispatchResult` (`rust/runtime/src/http.rs:190,209,306,329`). Both the scheduled worker and `RunnerGraphSink` build/consume these.
- **Shared transport dispatch, sprawling:** `TransportSink` exposes `execute_turn_measured[_streaming]`, `dispatch_prepared_turn_measured`, `dispatch_prepared_turn_collect_record[_streaming]`, `dispatch_prepared_turn_collect_record_with_response_observer` (`http.rs:1155-1405`). The scheduled worker calls `dispatch_prepared_turn_measured` (`turn_execution.rs:670`); `RunnerGraphSink` calls `dispatch_prepared_turn_collect_record` (`graph_execution.rs:816`). They bottom out in the **same** underlying function.
- **Two placement seams named non-parallel:** `HttpTurnExecutionBackend` (per-request, `http.rs:343`) and `GraphTraceExecutionBackend` (per-trace, `graph/placement.rs`). Both are "execute on thread-per-core workers" at different granularity, but nothing in the names says so, and `HttpTurnExecutionBackend` serves gRPC too (the `Http` is a lie).

P1 gives all of this **one generic vocabulary** and trims the dispatch sprawl. It does **not** merge the two placement seams or change any behavior — those are the deferred v2 structural work.

## 2. Naming

### Group A — shared substrate (used by BOTH paths; the main win)

| Now | New | Note |
|---|---|---|
| `PreparedHttpTurn` | `PreparedTurn` | both paths build it (`from_turn` stays a scheduled-only helper) |
| `MeasuredTurnContext` | `MeasuredContext` | pairs with `execute_measured` |
| `MeasuredTurnOutcome` | `MeasuredOutcome` | |
| `HttpTurnDispatchResult` | `DispatchResult` | `{ outcome, request_payload, record }` unchanged |
| `TransportSink::dispatch_prepared_turn_measured` | `TransportSink::dispatch_measured` | context-wired: register metadata → dispatch → record |
| `TransportSink::dispatch_prepared_turn_collect_record[_streaming]` | `TransportSink::dispatch_collect[_streaming]` | primitive: dispatch → return `DispatchResult`, caller wires the observer |
| `..._collect_record_with_response_observer` | fold into `dispatch_collect_streaming` | trims the sprawl: ~6 methods → 2 primitives + streaming |

### Group B — the two placement seams (parallel generic names; they stay separate)

| Now | New | Granularity |
|---|---|---|
| `HttpTurnExecutionBackend` | `RequestExecutor` | per **request** |
| `execute_turn_measured[_streaming]` (on it) | `execute_measured[_streaming]` | |
| `inference_dimensions(&TurnToSend)` | `inference_dimensions(&PreparedTurn)` | decouple from the scheduler type (an `&HttpRequest` variant already exists) |
| `HttpExecutionBackendFactory` / `NativeHttpExecutionBackendFactory` | `RequestExecutorFactory` / `NativeRequestExecutorFactory` | |
| `ThreadPerCoreHttpExecutionBackend` | `ThreadPerCoreRequestExecutor` | |
| `GraphTraceExecutionBackend` / `…Factory` | `TracePlacement` / `TracePlacementFactory` | per **trace** |
| `ThreadPerCoreGraphTraceExecutionBackend` | `ThreadPerCoreTracePlacement` | |

> **Collision note (naming review).** `PreparedTurn` not `PreparedRequest` — a
> `pub struct PreparedRequest<'a>` already exists at `endpoints/registry.rs:292`
> (re-exported). `TracePlacement` not `TraceExecutor` — `TraceExecutor` is already
> the DAG driver at `graph/executor.rs:58`; reusing it would collide *and*
> contradict the greenfield spec, which keeps that name for the driver.

Level convention this establishes: **`execute_*`** = placement-level (route to a worker); **`dispatch_*`** = transport-level (send bytes + measure). `RequestExecutor` vs the existing `RequestSink` (loadgen-core raw transport) are deliberately distinct levels; if that proximity reads as ambiguous, `MeasuredExecutor` is the fallback trait name.

### Not renamed in P1 (scope creep / wide read surface)

- `HttpRequest` (leaf transport DTO shared by http+grpc).
- `TurnDispatchOutcome` — the nested `DispatchResult.outcome` double-`outcome` smell; renaming to `ResponseOutcome` touches the whole scheduled read surface. Deferred.
- `TurnResponseObserver`, `TurnDispatcher`/`ConfiguredDispatcher`.
- `GraphSink` / `GraphReply` — genuinely graph-specific (per-node DAG splice). Kept; `RunnerGraphSink` just calls the renamed `dispatch_collect`. `TransportSink` keeps its name.

## 3. What stays as-is (explicitly)

- Both placement seams remain **two** traits (`RequestExecutor` per-request, `TracePlacement` per-trace). Merging them is the deferred structural unification, not P1.
- Failure-policy divergence (graph fail-fast vs scheduled resilient) is untouched — a separate decision (audit §"Genuinely different").
- The graph keeps per-node observer wiring (`register_metadata`/`record_response`/`drain_terminal_record` in `RunnerGraphSink`); it just calls the generically-named `dispatch_collect`.

## 4. Rollout (staged; each stage keeps the suite green)

1. **Rename Group A** (shared DTOs + `TransportSink` methods) and trim the dispatch-method sprawl to `dispatch_measured` + `dispatch_collect[_streaming]`. Update both callers (`turn_execution.rs`, `graph_execution.rs`, `http.rs`, `grpc.rs`).
2. **Rename Group B** (placement seams + factories + thread-per-core impls) and decouple `inference_dimensions` to `&PreparedTurn`.
3. Update `execution_factories.rs`, `registry.rs`, docs (`module-organization.md` crate/module table if any symbol is referenced), and the four agent files only if a renamed symbol appears in them.

Pure rename/consolidation: no control-flow or measurement change at any stage.

## 5. Parity & testing

Rename + method-consolidation only, so **no metric or dispatch-event change**. The existing graph byte-exact parity tests, scheduled dispatch tests, and sim/online integration tests must stay green **unmodified** — that is the entire correctness argument. No new tests are required beyond confirming the suite is green after each stage; add none for the sake of the diff.

## 6. Value

Modest but real: the shared substrate becomes **visibly** shared (one vocabulary), the ~6 `TransportSink` dispatch methods collapse to 2 primitives, and `Http`/`Turn` stop lying about what the code serves (gRPC, graph). It also lays the vocabulary groundwork for the eventual structural merge of `RequestExecutor` + `TracePlacement` without committing to it now.

## 7. Related

- `2026-07-13-scheduled-graph-production-convergence.md` — the audit this is premised on (what's shared vs. different).
- `2026-07-12-scheduled-worker-local-accumulation.md` — Track A A1, which built the worker-local measured seam being renamed here.
- `2026-07-13-scheduled-graph-convergence-implementation.md` — the change that carried these renames into the tree.

## Addendum — 2026-07-14 (built: renames; then fold, per the 2026-07-14b addendum below)

The rename half of this pass landed. Group A DTOs (`PreparedHttpTurn`→`PreparedTurn`,
`MeasuredTurnContext`→`MeasuredContext`, `MeasuredTurnOutcome`→`MeasuredOutcome`,
`HttpTurnDispatchResult`→`DispatchResult`) and Group B placement seams
(`HttpTurnExecutionBackend`→`RequestExecutor`, `ThreadPerCoreHttpExecutionBackend`→
`ThreadPerCoreRequestExecutor`, `Http`/`NativeHttpExecutionBackendFactory`→
`RequestExecutor`/`NativeRequestExecutorFactory`; `GraphTraceExecutionBackend`→
`TracePlacement`, `ThreadPerCoreGraphTraceExecutionBackend`→`ThreadPerCoreTracePlacement`,
`GraphTraceExecutionBackendFactory`→`TracePlacementFactory`) are renamed. The dispatch
methods are renamed to the level-generic names (`dispatch_measured`,
`dispatch_collect[_streaming]`, `execute_measured[_streaming]`, plus the private
`dispatch_collect_with_observer`). Pure rename, no behavior change; the full suite
(692 aiperf lib + runner stdio/graph/parity) stays green unmodified — the §5 correctness
argument.

## Addendum — 2026-07-14b (fold built; the other two items resolved with rationale)

1. **The method-count fold is now built.** `dispatch_collect_with_observer` was folded into a
   single public `dispatch_collect_streaming` taking `Option<&dyn TurnResponseObserver>`
   (`Some` = live frames, `None` = terminal-only); the previously-uncalled non-`Option`
   `dispatch_collect_streaming` wrapper was deleted. The collect surface is the two primitives
   §6 promised (`dispatch_collect` = the `None` convenience, `dispatch_collect_streaming` =
   the `Option` primitive) plus `dispatch_measured`. No behavior change (72 http unit tests
   green).
2. **`inference_dimensions(&TurnToSend)`→`(&PreparedTurn)` is intentionally NOT done** — it is
   net-negative, not a clean decouple. The caller (`ScheduledRuntime::issue_turn_internal`)
   holds a `TurnToSend` and needs the dimensions *at issue time*, before `PreparedTurn` is
   built at dispatch. Switching the backend seam to `&PreparedTurn` would force building a
   `PreparedTurn` twice per request (once for dimensions, once for dispatch) on the hot path,
   a real perf regression for zero functional gain. Keeping `&TurnToSend` is the correct
   signature; the aesthetic "decouple from the scheduler type" is subsumed by the deferred v2
   flow restructure (build `PreparedTurn` once at issue), not a standalone signature swap.

Two placement traits remain two traits, as §3 intends. `LocalGraphTraceExecutionBackend`
was intentionally **not** renamed: it owns the `TraceExecutor` and executes a trace locally;
it is not a placement.

## Addendum — 2026-07-16 (per-request executor collapsed to one generic backend; more `Http` lies removed)

A later sink-driven-executor pass took a step toward the greenfield §"Deleted, not
renamed" ideal (one `WorkerPool`, transport as a `Dispatcher` variant, no per-transport
executor). It did **not** merge the two *placement* traits (`RequestExecutor` per-request vs
`TracePlacement` per-trace — still deferred), but it did merge the two **per-request execution
backends** into one:

- **New seam `WorkerSink`** (the Rust analogue of Python `BaseTransport.send_request`): the
  worker-facing contract a transport sink implements (`set_run_origin` / `dispatch_measured` /
  `prewarm` / `inference_dimensions` / `supports_response_streaming`). `TransportSink` (http)
  and `GrpcTransportSink` (grpc) implement it.
- **`ThreadPerCoreExecutor<B: ExecutionSinkBuilder>`** replaces the former
  `ThreadPerCoreRequestExecutor` **and** the duplicate `ThreadPerCoreGrpcExecutionBackend`
  (~370 lines deleted): one worker loop, measurement, drain, cancellation, and streaming
  relay, generic over the sink. `build_native` is the single entry the http/grpc factories
  share. A transport contributes only an `ExecutionSinkBuilder` (`HttpSinkBuilder` /
  `GrpcSinkBuilder`) — never its own execution model. gRPC has no parallel execution path.
- **More `Http` lies removed:** `HttpExecutionBackendConfig`→`ExecutionBackendConfig`,
  `HttpPreparedEndpointTableFactory`→`PreparedEndpointTableFactory`, and — because the
  per-request backend is now genuinely per-transport rather than "http that also serves grpc"
  — `NativeRequestExecutorFactory`→`HttpExecutionFactory` and
  `NativeGrpcExecutionBackendFactory`→`GrpcExecutionFactory` (each honestly names the one
  transport it builds a sink for). The now-generic worker-loop error strings dropped their
  `HTTP` prefix.
- **`metrics_core::HttpTrace`→`RequestTrace`:** the per-request derived metrics trace is
  filled by every transport (http/grpc/dynosim/dry_run), matching Python's generic
  `BaseTraceData`/`TraceDataExport` (vs the http-specific `AioHttpTraceData`). The raw
  http-client trace stays `transport_http::models::TraceData`. The greenfield ideal still
  folds this timing into `Outcome`; the rename is the incremental de-`Http` step.

**`HttpRequest`→`Request` is now done too**, resolving the §2 "Not renamed in P1" deferral:
the leaf dispatch DTO shared by http/grpc/dynosim/dry_run takes the greenfield name
`Request` (`dispatch(&Request) -> Outcome`). ~64 references across `http.rs`, `grpc.rs`,
`dynosim.rs`, `dry_run.rs`, `workload.rs`, `graph_execution.rs`, and tests; word-boundary
rename left compounds like `HttpRequestDispatcher` (genuinely http) untouched. It still
lives in the `http` module (like `RequestRecord`/`Response`/`TraceData` — a generic type in
the http module; the greenfield relocation to a shared module is separate, structural work).

Pure rename + backend de-duplication; no metric/dispatch-event change (full suite green).

## Addendum — 2026-07-16 (transport module reorg: `transport::{core,http,grpc}` — the relocation follow-through)

The 2026-07-16 addendum above closed the `Http`-naming lies but explicitly left the
**location** open: `Request`/`RequestRecord`/`Response`/`TraceData` were generic types that
still *lived in* the `http` module, and it noted "the greenfield relocation to a shared
module is separate, structural work." That structural work is now **built**. The four flat
transport modules were consolidated under one `crate::transport` parent:

- `crate::transport_http` (hyper client) → `crate::transport::http`
- `crate::transport_grpc` (tonic client) → `crate::transport::grpc` (`#[cfg(feature = "grpc")]`)
- `crate::http` (the `http.rs` sink) → `crate::transport::http::sink`
- `crate::grpc` (the `grpc.rs` sink) → `crate::transport::grpc::sink`
- **NEW** `crate::transport::core` — the transport-neutral dispatch vocabulary, moved out of
  the http module so no generic type lives in a wire module any more: `Request`,
  `DispatchResult`, `MeasuredContext`/`MeasuredOutcome`, `PreparedTurn`, the `RequestExecutor`
  and `Dispatcher` traits, `PreparedEndpoint` (the location follow-through to this spec's
  earlier `PreparedHttpEndpoint`→`PreparedEndpoint` de-`Http` rename), `RequestRecord`,
  `Response`/`TextResponse`, `TraceData`/`TraceExport`/`TraceReference`, `ErrorDetails`/
  `ErrorKind`, `ConnectionReuseStrategy`, and the SSE data types `SseMessage`/`SseField`/
  `SseFieldName`.

Dependency direction is now honest: `transport::http` and `transport::grpc` depend on
`transport::core`, and `transport::core` has **no** dependency on `transport::http` (or on
`transport::grpc`) — a future transport takes the shared vocabulary without pulling in an
existing wire client. This supersedes the "still lives in the `http` module" caveat of the
2026-07-16 addendum for `Request`/`RequestRecord`/`Response`/`TraceData`; the de-`Http`
*naming* records above are otherwise unchanged. Pure relocation — no metric/dispatch-event
change (full suite green).
