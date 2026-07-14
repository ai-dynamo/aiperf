<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Greenfield execution vocabulary — the unified-substrate target names

**Date:** 2026-07-13
**Status:** design (north-star / greenfield ideal — **not** an incremental plan). Converged through an iterative naming review (two reviewers + refinement). Supersedes the earlier picks in this doc: `RequestGraph`/`Flow`/`Session`-as-the-unit, `Backend`, `Recorder`, `Bounds`/`Limits`/`Budget`, `Admission`, the executor tier, `FailurePolicy`, `WallClock`, and `Run`.

> The names an execution runtime would use if written fresh, no migration constraints. Records the target vocabulary for the deferred "unify scheduled + graph" v2 substrate (`2026-07-12-cellular-ready-seams-and-roadmap.md` §S5). The incremental counterpart — what to actually rename in the live tree — is `2026-07-13-p1-generic-execution-substrate-names.md`.
>
> Two principles did most of the work: (1) roughly half of today's names exist only because there are **two of everything** (scheduled + graph) — delete the duplication and the names collapse; (2) the **cleanest name is the one you don't need** — the entire executor tier below simply isn't a type.

## The work — four distinct levels

The modeling call that matters: a **definition** and an **in-flight instance** are different things (the code already splits them — `TraceRecord`/`GraphTracePlan` vs `TraceContext`). Don't force one word onto both.

- **`Replay`** — a recorded input file. Yields `Trace`s.
- **`Trace`** — the authored/recorded **definition**: a DAG of `Request`s. A flat run is a one-node trace; multiturn is linear; agentic is branching. Static, reusable, cycled by the workload.
- **`Session`** — one **in-flight instance** of a `Trace`. Mutable, unique id, runs on a worker. (`Session` is *not* the unit — it's a running one of it.)
- **`Request` → `Outcome`** — one dispatch and its result.

`Replay → Trace → Session → Request`: four levels, each a distinct thing, no overload. "Trace" is reserved for the definition — the recorded-file meaning moved to `Replay`, the http-timing meaning folds into `Outcome`.

## The seams

| Concept | Name | verb / variants |
|---|---|---|
| one benchmark run | **`BenchmarkRun`** | matches the Python wire type |
| a stage of it | **`Phase`** | `PhaseKind { Warmup, Measure }` |
| yields the trace stream | **`Workload`** | `Rate` / `Concurrency` / `Users` / `Replay` |
| arrival timing | **`Pacer`** | `Poisson` / `Constant` / `Gamma` / `Burst` |
| concurrency admission | **`SlotPool` → `Slot`** | `acquire() -> Slot` |
| when to stop | **`StopCondition`** | `RequestLimit` / `SessionLimit` / `DurationLimit` / `Cancelled`; aggregated by `StopChecker` |
| thread-per-core placement | **`WorkerPool` → `Worker`** | a `Worker` runs a `Session` |
| the client-side sender | **`Dispatcher`** | `dispatch(&Request) -> Outcome`; `Http` / `Grpc` / `Mock` / `Sim` |
| measurement | **`Observer`** | `on_arrival` / `on_token` / `on_usage` / `on_terminal` |
| failure behavior | **`OnFailure`** | `{ Continue, Abort }` |
| time | **`Clock`** | `MonotonicClock` / `VirtualClock` |

## Reads as a sentence

> A **`BenchmarkRun`** steps through **`Phase`**s. Each pulls **`Trace`**s from a **`Workload`**, paces them with a **`Pacer`**, admits them through a **`SlotPool`**, bounded by **`StopCondition`**s. Each admitted `Trace` becomes a **`Session`** on the **`WorkerPool`**; a **`Worker`** drives the `Session`, sending every **`Request`** through a **`Dispatcher`** for an **`Outcome`**, while an **`Observer`** records — on a **`Clock`**.

## Deleted, not renamed (the real cleanup)

Fewer types is most of the win:
- `RequestGraph` / `Flow` → **`Trace`** (definition) + **`Session`** (instance) — the definition/instance split is exactly what made those single words feel wrong.
- `Backend` → **`Dispatcher`** (the SUT word; `dispatch` is the verb).
- `Recorder` → **`Observer`** (kept; it's the GoF pattern and `Recorder` collides with the `metrics` crate).
- `Bounds` / `Limits` / `Budget` → **`StopCondition`** + `StopChecker` (incumbent — no new concept-word invented on top).
- `Admission` → **gone** — it's just `SlotPool.acquire()`; a one-method policy over a semaphore is ceremony.
- the executor tier — `RequestExecutor` / `TracePlacement` / `GraphExecutor` / `TraceEngine` → **gone**. A `Worker` runs a `Session`; a `Session` drives its `Trace`. No "executor" type exists.
- `FailurePolicy` → **`OnFailure { Continue, Abort }`** (an enum, not a trait with two impls).
- `GraphAdjacency` → folded into **`Trace`** (the DAG structure is the trace).
- `WallClock` → **`MonotonicClock`** ("wall clock" is the settable, jumping `CLOCK_REALTIME` — the opposite of a measurement clock; ours is `Instant`/monotonic).
- `Run` → **`BenchmarkRun`** (Python parity).

## Kept as-is (already right)

`Clock`, `SlotPool`/`Slot`, `Phase`, `Workload`, `Pacer`, `Request`, `Outcome`, `Observer`, `WorkerPool`/`Worker`, `Dispatcher`, `StopChecker`. `Ramp` (from `RampStrategy`), `EndpointSelector` (from `UrlSelector`), `Cancellation` stay.

## Caveat

Unconstrained ideal, not a tree change. Greenfield you never write the second placement seam, so `RequestExecutor` vs `TracePlacement` simply don't exist — there is one `WorkerPool`. The incremental, blast-radius-aware step is the P1 spec; this is the destination it walks toward.

## Related

- `2026-07-13-scheduled-graph-production-convergence.md` — why a flat request already shares the substrate with a graph trace.
- `2026-07-13-p1-generic-execution-substrate-names.md` — the incremental rename step.
- `2026-07-12-cellular-ready-seams-and-roadmap.md` §S5 — the deferred unify-scheduled+graph substrate.
- `2026-07-13-benchmarkrun-wire-and-runner-catalog-design.md` — the Python `BenchmarkRun` this aligns to.
