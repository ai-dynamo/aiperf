<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Flat-graph fast path

## Purpose

Define the flat-graph fast path for graph traces containing one LLM node.
`aiperf_runtime::graph::flat` provides eligibility, cancellation, and
`FlatGraphActor`; local and worker-backed production graph placement route
eligible plans through the actor without allocating the scheduler, channel
store, trace context, or readiness/fan-in machinery used by general graph
programs. Product-level byte-parity against the general executor is proven
through the real `aiperf` binary.

## Built

### Eligibility

`is_flat_graph(&GraphRecord)` returns true only when the graph contains exactly
one `LlmNode` and that node has no channel-requirement inputs. Zero-node,
multi-node, and fan-in graphs fail closed. `GraphRecord` has one executable node
type; behavior such as spawn, fork, subgraph, loop, barrier, and tool execution is
lowered into multiple nodes and edges before eligibility is evaluated.

Unit tests cover the eligible single-node shape and reject zero nodes, two nodes,
and one node with a channel input.

### Actor and cancellation

`FlatGraphActor` is built over the same `GraphSink`, `PromptMaterializer`,
`NodeDispatchPolicy`, and `NodeFailurePolicy` seams used by general graph
execution. `run`:

1. resolves the single node and materializes it with an empty channel-input map;
2. admits through `NodeDispatchPolicy`;
3. forwards node id, materialized messages, maximum tokens, and dispatch options
   through `GraphSink::dispatch_with_options`;
4. reports the first token and terminal status through the admission permit;
5. classifies failed, cancelled, and sink-error outcomes through the configured
   node-failure policy; and
6. completes without a channel write or successor scheduling.

The actor's structure contains no `Scheduler`, producer-count table, channel
specification map, node-index map, `VersionedChannelStore`, or `TraceContext`.
`FlatAbort` supplies a worker-local `Rc<Cell<bool>>` plus `Notify` latch.
Admission and dispatch both select against the latch, so a pre-tripped or
in-flight cancellation can stop the operation without constructing a
`TraceContext`.

Unit tests prove one eligible trace produces exactly one dispatch, forwards
`max_tokens`, materializes one message, and emits no dispatch when cancellation
is already tripped.

`flatgraph_disabled` reads `AIPERF_DISABLE_FLATGRAPH` once and caches the result.
The kill switch forces the general executor for external parity runs without
changing the authored workload.

### General graph substrate

Graph and non-graph inputs use separate execution paths. `NativeDatasetPlan`
selects `PreparedLinear`, `StaticAccuracy`, or `Graph`; general graph execution
uses `TraceExecutor`, `GraphSink`, and `VersionedChannelStore`, while scheduled
workloads use `TurnDispatcher`. The general graph substrate supplies dependency
readiness, fan-in, channel versioning, and successor scheduling for graphs that
are not eligible for the flat actor.

### Production routing

`LocalGraphTraceExecutionBackend::execute_trace` selects `FlatGraphActor` when
the plan is eligible, the environment kill switch is off, and the validation-only
`force_full` flag is false. Every other plan uses `TraceExecutor`. The backend
tracks `FlatAbort` latches beside active `TraceContext`s, so `cancel_inflight`
cancels either execution arm through the same placement API.

`GraphWorkerBackend` constructs an `EngineGraphSink`, applies the configured node
admission and failure policies to `LocalGraphTraceExecutionBackend`, registers
the local backend in its active-execution map, and calls
`LocalGraphTraceExecutionBackend::execute_trace`. Eligible worker-backed product
traces therefore use the flat actor while retaining the worker's endpoint
runtime, observer, metrics, cache-bust marker, record finalization, and
cancellation behavior.

Routing tests prove that an eligible plan selects the flat arm, `force_full`
selects `TraceExecutor`, and cancellation during blocked admission returns the
same `TraceError::Cancelled` classification without constructing the general
executor.

### Built invariants

- The flat path emits the same `RequestRecord`, metric observations, errors, and
  terminal classification as general graph execution for the same eligible
  trace.
- It reuses the shared dispatch and measurement seams rather than implementing a
  reduced record or transport path.
- It is clock-agnostic and introduces no direct wall-clock or Tokio timer access.
- Graph outputs remain byte-identical when the eligibility gate is disabled or a
  plan is ineligible.
- Cancellation, Ctrl-C behavior, observer delivery, and worker-local metrics
  remain reachable through the same backend-owned mechanisms.
- Parity applies to the request-record and artifact contracts. The implementation
  does not require a distinct minimal span stream or change the established span
  schema.

### Validation

An end-to-end test drives the same seeded single-LLM-node `dag_jsonl` program
through the real `aiperf profile` binary twice — once through `FlatGraphActor`
(default) and once through `TraceExecutor` (`AIPERF_DISABLE_FLATGRAPH=1`) — and
asserts an identical deterministic per-record projection (input and output
sequence length, reasoning tokens, output-length mismatch, conversation, turn,
correlation id, cancellation flag, and error) and record count. Routing and
cancellation are covered by unit tests (see Production routing). A `debug`-level
`graph flat fast path` marker confirms the flat branch fires once per eligible
trace in the external process, so the same records are produced by the flat actor
and reproduced exactly by the general executor.

The terminal channel write is omitted because no production artifact reads a
terminal node's output channel; only test-scoped helpers snapshot channel state.

## Future requirements

### Remaining validation

- An in-crate oracle that structurally proves the flat arm constructs no
  `VersionedChannelStore` or `TraceContext`, complementing the external
  byte-parity test.
- Coverage that eligible `weka_trace` and `dynamo_trace` inputs (not only
  `dag_jsonl`) reach the flat actor through normal Config-v2 execution, and that
  sketch-metrics mode and cellular classification as `Graph` hold on the flat arm.

### Hardening and later work

- Cache eligibility when a graph plan or placement is prepared instead of
  rescanning the graph for each trace, while retaining a defensive eligibility
  check at the actor boundary.
- **Do not pursue caching the per-trace `TraceExecutor` construction for a
  multi-node speedup** — it was prototyped (extract a graph-derived
  `GraphCompilation`, cache by fingerprint, reuse across traces) and measured on
  the real binary as a confirmed no-op: the cache hit perfectly (599:1) yet cache-on
  was marginally *slower* (5-node graph +4%, ~100-node +1%). The construction is
  negligible next to per-request dispatch/serde/metrics, and the fingerprint costs
  as much as it saves. The flat-actor win comes from bypassing the executor
  *runtime* (firing gates, channel await/versioning, snapshot), which multi-node
  traces inherently need. If a multi-node lever is ever wanted, target the two
  per-trace graph deep-clones (source template clone + `EngineGraphSink` node clone)
  by sharing `Rc<GraphRecord>` end to end — but even those are small versus dispatch.
- Keep scheduled-workload routing, scheduled/graph executor unification,
  multi-node executor reuse, and multi-node context pooling outside the
  single-node fast path until they have independent contracts and parity
  coverage.
- Any runner-credit abstraction must use a name distinct from cellular
  `IssuanceAuthority`, which owns record ordinals.

### Risks and mitigations

- **Eligibility drift:** keep the predicate exhaustive for the concrete
  `graph::model` representation and fail closed as executable node forms expand.
- **Record or metric drift:** reuse `GraphSink::dispatch_with_options` and the
  shared reduction/measurement path; compare raw per-record output.
- **Cancellation or observability loss:** register both execution arms with the
  same active-run and abort mechanisms.
- **Hidden channel consumers:** inventory every graph artifact and add a minimal
  output slot if any artifact requires terminal channel state.
- **Scope expansion:** keep multi-node pooling, scheduled-workload routing, and
  executor unification outside this contract.

## Source anchors

- `rust/runtime/src/graph/flat.rs` (`is_flat_graph`, `FlatAbort`,
  `FlatGraphActor`, and focused unit tests); `rust/runtime/src/graph/mod.rs`
  exports the module.
- `rust/runtime/src/graph/model.rs` (`GraphRecord`, `GraphTracePlan`, `LlmNode`),
  `materialize.rs` (`PromptMaterializer`), `policy.rs`
  (`NodeDispatchPolicy`), and `sink.rs` (`GraphSink`).
- Production routing:
  `rust/runtime/src/graph/execution.rs` (`LocalGraphTraceExecutionBackend`),
  `executor.rs` (`TraceExecutor`), `context.rs` (`TraceContext`), and
  `channel_store.rs` (`VersionedChannelStore`);
  `rust/runtime/src/engine/graph_execution.rs` (`GraphWorkerBackend`,
  `EngineGraphSink`); and `rust/runtime/src/engine/execute.rs`
  (`NativeDatasetPlan` routing).
- `rust/e2e/tests/test_flatgraph_parity.rs` and the
  `tests/fixtures/dag/single_node.dag.jsonl` fixture prove flat-vs-executor
  byte-parity through the real `aiperf` binary;
  `rust/runtime/src/engine/workers_characterization.rs` and
  `rust/runtime/tests/graph_transport_graph.rs` supply the deterministic graph and
  per-record validation patterns.
