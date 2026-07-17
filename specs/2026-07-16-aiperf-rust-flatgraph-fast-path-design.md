<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AIPerf-Rust: Flat-Graph Fast Path (`FlatGraphActor`) — Increment A

**Date:** 2026-07-16
**Status:** Design, awaiting review. Scoped to **Increment A only**; Increments B and C
are named for sequencing but out of scope here.
**Author:** Anthony Casagrande (Tech Lead) + Claude
**Grounds on:** the authoritative v2 design record
`ajc/dag-v3/docs/deps/aiperf-v2-cellular-runtime.md` (REQ 9 + §"Flat-Graph Fast Path"),
and a four-agent adversarial verification of the current `ajc/rust` tree (verdicts inline).
**Supersedes:** the "Stage 2a = rename `TraceExecutor`" framing in
`specs/2026-07-14-unified-execution-substrate-design.md`. This increment is not a rename;
it is the flat-graph fast path that makes the trace substrate cheap enough for the
degenerate case, which is the actual prerequisite for folding scheduled into the substrate.

---

## 1. Problem

AIPerf-Rust runs two disjoint execution engines:

- **Scheduled** (rate / concurrency / fixed-schedule / user-centric) dispatches through the
  lean `TurnDispatcher` seam (`scheduled.rs:170`); selection is gated on
  `request.dataset.is_graph()` (`engine/execute.rs:1352`, `:388`) and a plain synthetic
  workload can never reach the graph engine (verified: the `Graph` plan variant is produced
  only by `lower_graph`/`GraphWorkloadFactoryV2`, `online_execution.rs:1121`, `:280`).
- **Graph** (dag_jsonl / weka_trace / dynamo_trace) dispatches through the async-dataflow
  `TraceExecutor` + `GraphSink` + `VersionedChannelStore`.

The graph engine pays large **fixed per-trace construction overhead**, incurred regardless of
node count (verified against the real code):

- `LocalGraphTraceExecutionBackend::execute_trace` rebuilds a fresh `TraceExecutor` **every
  trace** (`graph/execution.rs:124`); the constructor rebuilds the `Scheduler` (edge
  adjacency, `executor.rs:111`), recomputes `producers_per_channel` (`:113`), clones the
  channel-spec map (`:114`), and deep-clones every `LlmNode` into a fresh `node_index`
  (`:115-119`).
- `build_context` then allocates, per trace, an `Rc<VersionedChannelStore>` (four fresh
  `BTreeMap`s + one `Rc<Notify>` per channel, `channel_store.rs:97-112`) and an
  `Rc<TraceContext>` (2 `HashSet`s + 3 `BTreeMap`s + 2 `Notify` behind `RefCell`,
  `context.rs:49-64`).
- The **online** production path is heavier still: `GraphWorkerBackend::execute_trace`
  (`engine/graph_execution.rs:729-799`) additionally clones the entire node map into a fresh
  `EngineGraphSink` and builds a fresh `NativeMetricsObserver` **per trace**.
- There is **zero cross-trace pooling or reuse** of any of these objects anywhere in
  `rust/runtime` (verified).

For a single-LLM-node trace the only genuinely necessary work is **one sink dispatch** (plus, on
the current path, a terminal channel write nothing downstream consumes); everything above is
pure orchestration tax. This tax is the sole reason
"scheduled is the degenerate case of graph" would be a regression, and thus the reason the
two engines coexist today.

### Why this matters now (the two-ontology frame)

A **behavior graph** (Pinterest: conditional routing, race-cancel, `@channel` threading) is a
*generator of executions* and genuinely needs the channel/scheduler/versioning machinery. A
**trace** (ATIF/ATOF/SATF/weka/dynamo captures — and a flat scheduled request) is a *single
execution instance*: a fixed partial order with choices already made. It needs only
dependency readiness, timing gates, and content handoff. The heavyweight machinery is
Behavior-IR machinery; a pure trace should not pay for it. The flat-graph fast path is the
narrowest, highest-value slice of that principle: the 1-node trace.

The authoritative v2 record already specifies this as **REQ 9 / "Flat-Graph Fast Path"**: a
`FlatGraphActor` that, for a single-LLM-node graph, "bypasses channel store allocation,
scheduler allocation, the per-trace TaskGroup, and the readiness/fan-in machinery," with
per-trace overhead matching non-graph single-turn dispatch. It is **built and production-wired
in Python** (`ajc/dag-v3/src/aiperf/v2/runner.py`), but **not ported to Rust** (verified: zero
hits for `FlatGraphActor`/`is_flat_graph`/`VirtualTraceRunner`/`RunnerPool` in `ajc/rust` or
the sibling `dynamo-aiperf-native`).

## 2. Scope

**In scope (Increment A):**

1. A Rust `is_flat_graph` predicate.
2. A Rust `FlatGraphActor`: a straight-line executor for a 1-node trace that allocates **no**
   `Scheduler`, `VersionedChannelStore`, or `TraceContext`.
3. Routing: the graph placement detects flat at plan/parse time and drives `FlatGraphActor`
   instead of constructing a `TraceExecutor`.
4. A byte-parity oracle proving `FlatGraphActor`'s **`RequestRecord`** output is identical to
   the current `LocalGraphTraceExecutionBackend` path for 1-node graphs.

**Explicitly out of scope (named only for sequencing):**

- **Increment B** — rerouting flat *scheduled* workloads through `FlatGraphActor` (folding the
  two engines). A pilot proof of the substrate; not attempted here.
- **Increment C** — `VirtualTraceRunner` executor-reuse + context **pooling** for *multi-node*
  recorded traces. This is where the real porting risk lives (Python's structured-concurrency
  `asyncio.TaskGroup`, detached-spawn-outliving-trace, `ExceptionGroup`, and the
  monkeypatch-of-executor-slots binding trick — verified in the Python source). Deliberately
  deferred so Increment A stays low-risk.
- No `LoadExecutor` merge, no span-shape change (see §4.4), no cellular changes.

## 3. Non-negotiable invariants

- **RequestRecord parity by construction.** The flat path must emit the identical
  `RequestRecord` the current 1-node path emits. It achieves this by **reusing the same
  dispatch + measurement seam** (`GraphSink::dispatch_with_options` →
  `transport::reduce`/`transport::measure`), not by reimplementing a minimal dispatch. (In
  Python, `FlatGraphActor` emits a deliberately minimal stub record and parity holds only
  because both funnel through the same `HttpLocalDispatcher`; in Rust we get parity more
  cheaply and more safely by reusing the existing `GraphSink`.)
- **The `{Clock} × {transport}` seam is untouched.** The flat actor is clock-agnostic; no
  `Instant::now`/tokio timers. (Trivial here — a 1-node trace has no edge-delay gates.)
- **`workers == 1` and all current graph outputs stay byte-identical** when the flat path is
  disabled; the flat path is additive and behind the `is_flat_graph` gate.
- **Cancellation and observability preserved.** The flat path must still register with the
  backend's cancellation registry (`active`, `execution.rs:134`) and drive the same observer /
  record / metrics wiring the full path uses.

## 4. Design

### 4.1 `is_flat_graph` predicate

Port the Python predicate (`runner.py:74-103`) faithfully:

- Exactly **one** `LlmNode`, and
- **zero** disqualifying node kinds: Spawn, Subgraph, Loop, Barrier, Tool, ToolCall,
  ToolResult, Await.
- Replay-class scaffolding (Replay / Delay / Compact / Bootstrap) does **not** disqualify —
  but note (open item, §7) the Rust node-kind set must be enumerated against the actual
  `graph::model` types, which differ from the Python taxonomy.

Evaluated once at plan/parse time (where the `GraphTracePlan` graph is known), cached on the
plan/placement so `execute_trace` does not re-scan per trace. A defensive re-check in the flat
actor's bind path mirrors Python's hard guard (`runner.py:351`).

### 4.2 `FlatGraphActor`

A straight-line executor over a single node. For a 1-node trace the node's inputs come from
`trace.initial_state` directly (no upstream channel writes exist), and the node's reply is
terminal (nothing reads it), so **no channel store is needed**:

1. Resolve the single `LlmNode` and materialize its inputs from `initial_state` (reusing the
   existing `PromptMaterializer`/`materializer.build` path).
2. Admit via the graph path's existing admission (`node_policy.admit` / prefill slot) — **not**
   a new mechanism.
3. Dispatch via the **same** `GraphSink::dispatch_with_options` the full path uses → this drives
   `transport::reduce`/`measure` and emits the identical `RequestRecord` + metrics.
4. No successor scheduling, no `snapshot`. The terminal channel write **may** be skippable
   (nothing reads a 1-node terminal trace's output channel, and the response text reaches the
   `RequestRecord` via `transport::reduce`, not the channel snapshot) — but this must be
   confirmed against every graph artifact that reads channel state (§7, item 5), so it is an
   optimization to verify, not an assumed given.
5. Honor cancellation via the same abort latch the backend already exposes.

It allocates none of: `Scheduler`, `producers_per_channel`, channel-spec clone, `node_index`
map, `VersionedChannelStore`, `TraceContext`. The per-trace cost collapses to input
materialization + one dispatch — matching the flat scheduled path's overhead.

### 4.3 Routing / placement integration

`LocalGraphTraceExecutionBackend::execute_trace` (and the online `GraphWorkerBackend`) branch
on the cached flat flag: flat → `FlatGraphActor::run`; non-flat → today's `TraceExecutor` path,
unchanged. Both arms register with the `active` cancellation registry and use the same
`EngineGraphSink` (online) so records/metrics are produced by the identical seam. **Naming:**
disambiguate from the existing cellular `IssuanceAuthority` (`cellular/issuance.rs:32`), which
is a record-ordinal authority, not a credit acquire/release runner authority — do not overload
the name.

### 4.4 Spans — explicitly NOT a parity claim

Verified in the Python source: `FlatGraphActor` and `VirtualTraceRunner` emit **different**
span streams (a single `trace_complete` vs begin/end trace+node), and no Python test asserts
span parity. Therefore this increment asserts **RequestRecord** parity only. Because the Rust
flat path reuses the same `EngineGraphSink`/`transport::measure` seam, its spans/records match
the current 1-node path by construction; we still assert only the RequestRecord surface and do
not introduce a divergent minimal span.

## 5. Byte-parity oracle

Model on `engine/workers_characterization.rs` (verified: in-crate `#[cfg(test)]`, deterministic
in-process `FixedMock` hyper SSE server with fixed ISL/OSL/TTFT/ITL, sorted-`(ISL,OSL)`-multiset
assertions, `assert_pinned_records`, driven through
`execute_prepared_native_plan_uncommitted_selected`).

New oracle: run the **same seeded 1-node graph fixture** against the same `FixedMock` through
**two executor arms** — (i) the current `LocalGraphTraceExecutionBackend`/`TraceExecutor` path
and (ii) the new `FlatGraphActor` path — and assert:

- `sorted_data_keys` (sorted `(ISL, OSL)` multiset over `profile_export.jsonl`) equal, and
- `assert_pinned_records` invariants (exact record count, no errors, ISL/OSL pinned, positive
  `request_latency`/`time_to_first_token`, `inter_token_latency` present) hold on both arms.

The one wiring that does not exist today is a **"select executor impl" arm** — existing harnesses
compare `workers`/`cells` counts or drive a single backend, never two executor implementations.
This oracle adds that seam (a test-only switch selecting flat vs full for the same plan).

Optionally, a structural assertion that the flat arm constructs no `VersionedChannelStore`/
`TraceContext` (e.g. a `#[cfg(test)]` allocation counter or a code-path assertion), to prevent
silent regression of the bypass.

## 6. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Record drift between flat and full path | Reuse the same `GraphSink`/`transport::measure` seam; oracle asserts RequestRecord multiset parity. |
| Rust node-kind taxonomy differs from Python's disqualifying set | Enumerate `is_flat_graph` against the real `graph::model` node kinds (§7 open item), not a blind port of the Python tuple. |
| Cancellation/observability regressions on the flat arm | Flat arm registers with the same `active` registry and abort latch; e2e Ctrl-C/cancel tests must cover a flat graph. |
| Scope creep into pooling / TaskGroup semantics | Hard boundary: Increment A touches only the 1-node straight-line path; all `TaskGroup`/spawn/pooling work is Increment C. |
| `IssuanceAuthority` name collision | Use a distinct name for any runner credit authority; leave cellular `IssuanceAuthority` untouched. |

## 7. Open items to resolve during planning

1. **Rust node-kind set** for `is_flat_graph` — map the Python disqualifying kinds
   (Spawn/Subgraph/Loop/Barrier/Tool/ToolCall/ToolResult/Await) and non-disqualifying
   replay-class kinds onto the actual `graph::model` enums/structs.
2. **Where the flat flag is computed and cached** — on `GraphTracePlan`, the placement, or the
   workload factory — so `execute_trace` does not re-scan per trace.
3. **Admission path for the flat actor** — confirm the exact `node_policy.admit` / prefill-slot
   call the full path uses for the single LLM node, and reuse it verbatim.
4. **Online vs local placement** — both `LocalGraphTraceExecutionBackend` and
   `GraphWorkerBackend` need the branch; confirm the online path's `EngineGraphSink` reuse is
   compatible with skipping the executor.
5. **Is the terminal channel write skippable?** — enumerate every graph artifact that reads
   channel state for a terminal node (`graph_per_node.csv`, channel snapshots, `outputs.json`,
   SATF export) and confirm none depends on the 1-node output channel; if any does, the flat
   actor performs the single write into a minimal slot rather than a full `VersionedChannelStore`.

## 8. Sequencing

- **A (this doc):** `FlatGraphActor` + `is_flat_graph` + parity oracle. Scheduled untouched.
- **B:** route flat scheduled through `FlatGraphActor`; parity vs current `TurnDispatcher`.
- **C:** `VirtualTraceRunner` executor-reuse + context pooling for multi-node traces (the
  `TaskGroup`/spawn-mapping-heavy increment).

## 9. One-line summary

Port the v2 `FlatGraphActor` flat-graph fast path (REQ 9) to Rust as a straight-line 1-node
trace executor that reuses the existing `GraphSink` dispatch/measure seam while allocating no
scheduler/channel-store/context — proven RequestRecord-identical to the current path by an
oracle modeled on `workers_characterization.rs` — so the trace substrate stops paying the
per-trace tax on degenerate traces and scheduled can later fold in (Increment B) without a
regression.
