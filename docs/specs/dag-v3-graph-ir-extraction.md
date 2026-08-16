<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
SPDX-License-Identifier: Apache-2.0
-->

# DAG-v3 Graph-IR extraction record

## Purpose

This record captures the architectural review of the historical `ajc/dag-v3`
implementation at `60a072aaee`. It defines which graph-runtime ideas remain
suitable for AIPerf's native Rust system, which must be adapted, and which
historical implementation couplings must not be copied unchanged.

The branch is evidence and a test corpus, not an implementation dependency.
Its graph-related delta from the Rust-port fork point contains approximately
46,000 lines across 142 files. This is not an instruction to reintroduce the
Python executor, but neither is later scope reduction evidence that its rich
semantic vocabulary was unsound. The branch is especially relevant to
Pinterest-style production assistant workflows, Conflux captures, and
observability-derived graphs in addition to AgentX workloads.

This record complements [graph-runtime.md](graph-runtime.md),
[recorded-agent-replay-rust-port.md](recorded-agent-replay-rust-port.md), and
[agentic-eval-platform.md](agentic-eval-platform.md).

## Built

The Rust Graph-IR has retained DAG-v3's highest-value execution foundation:

- per-trace append-only channel logs, captured read versions, deterministic
  reduction order, producer accounting, and unsatisfiable-wait errors;
- immutable graph scheduling data shared across trace instances;
- trace-local execution and cancellation through the injected runtime handle;
- static-edge timing gates, including completion-, start-, and
  first-token-anchored timing; and
- structural validation for topology, state channels, reachability, and
  fireability/deadlock cycles.

These are native Rust behavior, not a compatibility layer over the Python
executor. Existing LLM-only workloads remain on the narrow `LlmNode` substrate
and preserve the flat-graph fast path where eligible.

## Extraction decisions

### Adopt: versioned channels and firing semantics

Keep the existing `VersionedChannelStore` as the canonical state model. A node
reads a captured channel version, never a live mutable view. Initial state is a
reducer seed rather than a producer arrival. Writes are deterministically
ordered by `(write_seq, writer_node_id)`, and a wait whose remaining producers
cannot meet its declared count fails as orphaned rather than hanging.

The DAG-v3 test corpus remains useful for overwrite conflicts, all-producer
fan-in, failed producers, streams, and concurrent writes. Port individual
behavior tests only when the corresponding native Rust feature is supported;
do not port Python internals.

### Adopt: stable execution identity

The future agent path shall distinguish immutable task/template identity, benchmark
run identity, trace-instance identity, trajectory-document identity, and
agent-invocation identity. Delegated execution uses one canonical deterministic
child-invocation key constructor. A template key must never be used where an
instance key is required. Child input, replay evidence, and outputs are scoped
to the child invocation; a parent observes them only through explicit artifact
or channel mappings.

### Adapt: layered import validation

DAG-v3's separation of parsing, normalization, derivation, and rule-based
validation is worth retaining, but its large universal rule catalog is not. The
`agent_recording` adapter shall use a strict-import profile:

```text
raw recording artifact
  -> strict source DTO
  -> normalized recording + provenance/loss report
  -> GraphTraceProgram
  -> runtime, semantic, and capability validation
```

Unsupported source constructs shall be rejected or recorded in a machine-readable
degradation report. They must not silently become a replay node, a default
graph, or an omitted event. General workload decoding may tolerate unknown
fields for format compatibility; evaluation-recording imports may not.

Validation findings shall carry a stable rule id, severity, source location,
plain-language message, and, where possible, a concrete remediation. Maintain
separate layers for graph structure/fireability, recording completeness and
tool correlation, sandbox capability compatibility, and controller provenance
or resume eligibility.

### Retain semantics; adapt execution: barriers, delegation, and joins

DAG-v3 correctly exposed difficult lifecycle cases. `Spawn`, `Await`,
`Barrier`, `Subgraph`, bounded `Loop`, tool-call/result, replay, and marker
nodes remain valid *semantic* graph vocabulary. For example, a Pinterest-style
router/safety graph makes race, branch, and terminal semantics application
visible rather than a private driver detail.

The initial `GraphNode::{Llm, Tool}` executable graph remains an appropriate
delivery scope. It is not the ceiling on representation: a Rust-native
semantic graph is lowered through explicit capabilities into executable LLM,
tool, replay, lease, barrier, and controller operations. See
[semantic-agent-graph.md](semantic-agent-graph.md).

When these features are added:

- an `any`, quorum, or timeout race cannot first await every input channel;
- persist the closure reason before cancelling losing work;
- distinguish expected race cancellation from agent or infrastructure failure;
- consume loser errors and retain them as evidence;
- make an await timeout bound the await, not an independently owned child;
- cancel parent-coupled children during parent cleanup; and
- isolate concurrently implementing children in overlays/clones. They return
  immutable candidate patches or artifacts and never concurrently mutate the
  canonical task worktree.

### Adapt: trace evidence schema

DAG-v3's span fields are useful even though its Python `SpanBuilder` is not.
The provider-neutral event/trajectory schema and terminal supplement shall represent:

- trace, invocation, parent-event, node, and request correlation ids;
- scheduling/readiness order and edge expected/actual/drift timing;
- dispatch, first-token, terminal, cancellation, and error timestamps;
- model/token/cache usage and tool resource measurements;
- budget before/after, policy decisions, and sandbox identity; and
- immutable artifact references, terminal status, closure reason, and
  evaluator evidence.

Workers and cells emit bounded serializable facts. The controller owns final
artifact writes, ordering, manifest generation, and result folding. Do not
restore mutable untyped span dictionaries, worker-owned final JSONL files, or
wall-clock-only timestamping.

### Adapt: replay and snapshot boundaries

Recorded assistant output and live response output are distinct values with
different authority. Lowering makes the selected source explicit, prevents live
responses from contaminating a recorded successor prompt, and preserves the
source artifact digest that justified the choice. This is a regression-test
requirement for `GraphTraceProgram` and its response store, not a reason to
restore generic `ReplayNode` execution.

## Do not copy unchanged

Do not copy these *implementation couplings*. They are distinct from rejecting
the corresponding semantic graph facts:

- Python's universal tagged node executor and its `asyncio`/wall-clock
  coupling. Native Rust lowering owns the equivalent behavior;
- generic loops or unbounded retries; retained loop semantics require explicit budgets
  and termination conditions;
- unrestricted phase-scoped detached children that outlive a trace without
  bounded accounting;
- tool dispatch through an LLM endpoint/credit path; `ToolSandbox` and
  `ToolDispatcher` remain separate capability-gated seams;
- eager support for a large ecosystem-specific adapter matrix;
- silently degrading unknown source semantics into executable defaults;
- Python `asyncio`/wall-clock execution in place of the injected `Clock` and
  placement model; or
- DAG-v3's multiprocessing/records transport in place of the native cellular
  protocol and controller-owned supplements.

## Future requirements

1. Add a Rust-native semantic graph/template/trace-instance DTO and a strict
   `agent_recording` source DTO with normalized provenance/loss
   report before adding foreign recording adapters.
2. Define and test a canonical child-invocation key plus distinct template,
   trace, trajectory, and invocation identity types.
3. Add typed event/supplement DTOs with the evidence fields above, preserving
   controller-only final artifact ownership.
4. Port selected DAG-v3 behavioral tests for channel failure, response-source
   isolation, barrier closure/cancellation, child lifetime, and timeout
   semantics as each native feature is introduced.
5. Add layered validator APIs and diagnostic DTOs. The existing structural
   graph validator remains the runtime layer rather than being replaced.
6. Keep terminal, controller-owned checkpoints authoritative. Do not claim
   mid-trace or distributed resume until an independently verified lifecycle
   contract exists.

## Source anchors

- `ajc/dag-v3:src/aiperf/orchestrator/graph/channel_store.py` — historical
  channel-store semantics and edge-case test source.
- `ajc/dag-v3:src/aiperf/orchestrator/graph/dispatch/{barrier,spawn,await_node}.py`
  — historical race, child-lifetime, and join behavior to preserve as tests.
- `ajc/dag-v3:src/aiperf/dataset/loader/graph/{validator.py,spawn_keys.py}` —
  layered-validation and identity lessons.
- `ajc/dag-v3:src/aiperf/orchestrator/graph/span_builder.py` — fields to
  preserve in the typed evidence schema, not an implementation template.
- `rust/runtime/src/graph/{channel_store,executor,scheduler,validate,model}.rs`
  — built native graph foundation.
- `docs/specs/recorded-agent-replay-rust-port.md` — planned agent-replay
  seams and compatibility revision notes.
