<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Conditional graph lowering

## Purpose

Define how model-independent conditional branching and recorded non-LLM content
enter the flat Graph-IR **without adding a runtime node kind, edge kind, or
reactive machinery**. The Graph-IR runtime executes exactly one node kind
(`LlmNode`) and one edge kind (`StaticEdge`); every richer authored shape — a
routing branch, a recorded tool result, a fan-out/fan-in join — is resolved and
collapsed into that substrate at **lowering**, before any `GraphTracePlan`
exists. This spec states the contract that lets an authored conditional graph
reach the runtime, and draws the line between what is lowerable and what is not.

The governing doctrine is a single rule:

> Branch resolution is permitted when it is a **pure function of pre-execution
> data** — a recorded trace, a pinned `selected_branches` map, or a static-seed
> weighted sample. Branch resolution that consumes a **live output** — a value
> or completion produced by a node in the current run — is forbidden: it would
> require dispatch-time edge resolution and in-flight cancellation, the reactive
> machinery the flat core intentionally omits.

Pinned, recorded, and weighted branching are all model-independent and resolve
eagerly. Branch-on-live-output does not, and is out of scope for this spec.

## Built

The eager-conditional pass composes over machinery that already exists; nothing
below is new.

### Flat substrate

`GraphRecord` carries only `nodes: BTreeMap<String, LlmNode>` and
`edges: Vec<StaticEdge>` (`graph/model.rs`). `LlmNode` writes exactly one channel
(`output`), reads channel state only through `PromptItem::Splice` items, and
carries the generation cap, streaming flag, AND-fan-in `inputs`, and node-level
`min_start_delay_us`. `StaticEdge` carries four independent delay anchors
(completion, absolute floor, predecessor-start, predecessor-first-token). There
is no conditional edge, multi-output node, typed channel beyond `text`/`messages`,
or reducer beyond `overwrite`/`add_messages`. The IR-model layer silently ignores
unknown node/edge fields; the strict decode boundary is the engine input adapter
(`engine/graph_input.rs`, `deny_unknown_fields` envelopes, unknown-`format` hard
error).

### Per-trace resolved-graph seam

`ParsedGraph { graph, graphs: BTreeMap<String, GraphRecord>, traces }` and
`resolve_trace_graph` select, per trace, either the shared `graph` or a named
entry in `graphs` via `TraceRecord.graph_ref` (`graph/model.rs`). Emitting one
fully-resolved `GraphRecord` per trace, keyed by trace id with
`graph_ref = Some(trace.id)`, is the established norm: the `dag_jsonl` catalog
compiler (`graph/lowering.rs`, `lower_catalog`), the WEKA compiler
(`graph/recorded/weka/mod.rs`), and the Dynamo compiler
(`graph/recorded/dynamo/mod.rs`) all do exactly this. The recorded compilers also
prove that agent structure — subagents, tools, async-launched children — is
**flattened into flat nodes and edges at lowering** (`flatten_entries` +
`async_ancestors`), not represented at runtime.

### Channel seed semantics

`TraceRecord.initial_state` seeds a channel at `write_seq 0` under the synthetic
writer `__init__` (`graph/channel_store.rs`). A seed is a reducer **base**: it is
fully visible to a downstream prompt splice (`reduce_channel_at_seq`), but it is
**not** an arrival (`capture` filters `write_seq != 0`) and it does not decrement
producer accounting. `count: "all"` resolves against the static producer count of
the **emitted** graph (`producers_per_channel`), so producers that a lowering
pass removes are never counted.

### Firing-delay composition

`compute_firing_gate_us` (`graph/executor.rs`) folds every incoming edge anchor
and the node-level `min_start_delay_us` with a running `max`. A recorded
per-node latency therefore has a well-defined home on a rerouted successor edge.

### Eager-conditional graph adapter

`rust/runtime/src/graph/conditional/` is a strict native compiler that ingests an
authored conditional graph and emits one flat `GraphRecord` per trace. It is
selected by dataset `format: conditional_graph` and is registered through the
same path as the other graph compilers
(`ConditionalGraphRunnerGraphInputAdapter` in `engine/graph_input.rs`;
`conditional_graph` also recognized in `engine/cellular_kind.rs`'s
graph-classification match alongside `dag_jsonl`/`weka_trace`/`dynamo_trace`).

**Authored model.** A `graph:` block declares state channels, nodes (including
non-dispatching *replay* nodes that carry recorded `outputs`), and edges
(including `branches: {<key>: <target>|[<target>...]}` conditional edges with an
optional static `branch_weights`). A `traces:` block declares, per trace,
`initial_state`, a pinned `selected_branches` map, optional per-trace branch
distributions, `replay_outputs` keyed by node id, arrival time, and token hints.
The adapter performs its **own** strict decode (`deny_unknown_fields`); the
IR-model layer will not reject a malformed authored body.

**Per-trace lowering algorithm.**

1. **Resolve** each conditional edge's branch key with fixed precedence:
   `selected_branches[source]` → per-trace distribution → `branch_weights`
   sampled by a deterministic RNG seeded on `(workload_seed, trace.id, source)`.
   The key is a pure function of pre-execution data.
2. **Prune** to the taken subgraph: walk from `START` over static edges and taken
   branch targets, and drop every node and edge not reached. Untaken-branch
   producers vanish, so `count: "all"` fan-ins close on exactly the taken
   producers with no runtime "was this producer going to fire?" bookkeeping.
3. **Fold** recorded non-LLM content into the flat substrate:
   - a replay node's `outputs` are pre-seeded into `TraceRecord.initial_state`
     from `replay_outputs[node_id]`, and the node is dropped; a downstream splice
     resolves the seed with no re-encode;
   - a multi-output replay node pre-seeds N channel keys (the single-`output`
     `LlmNode` limit is a runtime-dispatch limit, never engaged at build time);
   - the node's recorded latency (`duration_ms` / `wait`) collapses onto the
     rerouted successor edge's `delay_after_predecessor_us`;
   - typed `image`/`json` channels collapse to `text`/`messages` plus segment
     bytes — the flat core does no structured field-extraction from channel
     state, so there is no runtime type to preserve.
4. **Emit** a flat `GraphRecord` into `parsed.graphs[trace.id]`, set
   `graph_ref = Some(trace.id)`, and run `graph::validate::validate`.

After pruning, each trace retains exactly one live user-visible terminal path, so
recorded dual-terminal graphs need no first-writer-wins accounting.

**Forbidden (not lowerable; require the omitted reactive primitive).** These all
reduce to *cancel-in-flight-on-live-completion* and are explicitly out of scope:

- branching on a **live** produced channel value;
- a loop whose break condition reads a **live** output (a recorded/known
  iteration count instead unrolls to a counted loop and is lowerable);
- a barrier with `policy: "any"` that races siblings and cancels the losers on
  **live** completion (a recorded winner instead prunes the losers and is
  lowerable).

Reopening any of these is a separate spec plus adversarial review, not an
extension of this one.

### Registration footprint

The format touches: `rust/runtime/src/graph/conditional/` (`mod.rs`, `model.rs`,
`resolve.rs`, `fold.rs`); `ConditionalGraphRunnerGraphInputAdapter` in
`engine/graph_input.rs` with a `deny_unknown_fields` envelope; the builtin
resolver array in `engine/graph_input.rs`; and the graph-classification match in
`engine/cellular_kind.rs`. No runtime executor, reducer, or channel-store change
was required — the pass emits only `LlmNode`/`StaticEdge`, which the existing
executor already runs, and eligible single-node resolved traces still reach the
flat fast path (see [flatgraph-fast-path.md](flatgraph-fast-path.md)).

### Validation

`rust/e2e-tests/tests/test_conditional_graph.rs` drives an authored conditional graph
(`e2e/tests/fixtures/conditional/conditional_shopping.yaml`) with pinned
`selected_branches` through the real `aiperf` binary against a deterministic
`aiperf-mock-server`, asserting the per-record projection for each taken path
(the branch fan-out, folded replay content, folded edge delays, and the single
terminal).

## Future requirements

### Reserved data-model extension: pre-committed arrivals

A seed is a base, never a counted arrival. If a corpus ever requires a recorded
non-LLM output to occupy an **arrival slot** in a downstream `count` / `count:
"all"` gate (rather than serve as a splice value or reducer base), the minimal
addition is an `initial_arrivals` field on `TraceRecord` that pre-commits
`write_seq > 0` log entries under a synthetic writer at store construction —
incrementing `arrival_count` and decrementing `producers_remaining`. This
preserves the single-executable-kind IR: no dispatch, no reactive machinery, no
replay node. It is reserved, not built; no current target requires it.

## Source anchors

- Flat substrate and per-trace seam: `rust/runtime/src/graph/model.rs`
  (`GraphRecord`, `LlmNode`, `StaticEdge`, `ParsedGraph`, `resolve_trace_graph`,
  `TraceRecord.graph_ref`/`initial_state`).
- Eager-conditional adapter: `rust/runtime/src/graph/conditional/` (`mod.rs`,
  `model.rs`, `resolve.rs`, `fold.rs`) — authored-model decode, branch
  resolution/pruning, and the replay fold.
- `rust/e2e-tests/tests/test_conditional_graph.rs`,
  `rust/e2e-tests/tests/fixtures/conditional/` — e2e coverage.
- Other eager per-trace compilers over the same seam: `rust/runtime/src/graph/lowering.rs`
  (`lower_catalog`), `rust/runtime/src/graph/recorded/weka/mod.rs`,
  `rust/runtime/src/graph/recorded/dynamo/mod.rs`, and the shared
  `rust/runtime/src/graph/recorded/trie/` lowerer.
- Seed, arrival, and producer accounting: `rust/runtime/src/graph/channel_store.rs`;
  static producer counts `rust/runtime/src/graph/channels.rs`.
- Firing-delay composition: `rust/runtime/src/graph/executor.rs`
  (`compute_firing_gate_us`).
- Adapter registration and strict decode: `rust/runtime/src/engine/graph_input.rs`,
  `rust/runtime/src/engine/cellular_kind.rs`, `rust/runtime/src/engine/dataset_input.rs`.
- Related records: [graph-runtime.md](graph-runtime.md),
  [flatgraph-fast-path.md](flatgraph-fast-path.md), [dataset.md](dataset.md).
