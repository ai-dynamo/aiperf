<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
SPDX-License-Identifier: Apache-2.0
-->

# Semantic agent graph and source-fidelity contract

## Purpose

This record defines the native Rust representation of an agent or production
assistant workflow. It is intentionally broader than the initial executable
Graph-IR. A graph must preserve meaningful source topology and evidence even
when a selected execution profile cannot yet execute every semantic operation.

The design derives from DAG-v3's production-assistant, Conflux, SATF,
LangSmith, Dynamo, and coding-agent work. DAG-v3 is historical semantic and
test evidence only: no Python process, module, ABI, plugin runtime, or
`asyncio` behavior is part of the resulting product.

## Built

The native runtime already provides LLM graph execution with state channels,
versioned capture/reduction, producer accounting, static edges, timing gates,
trace-local placement, an injected `Clock`, and structural validation. These
remain the execution foundation. The existing `LlmNode` path and flat-graph
fast path retain their behavior.

## Semantic model

Every accepted source is normalized into Rust-owned, versioned serde DTOs:

```rust
struct SemanticGraph {
    version: SchemaVersion,
    template: GraphTemplate,
    traces: Vec<TraceInstance>,
    provenance: GraphProvenance,
}

enum SemanticNode {
    Llm(LlmSpec),
    ToolCall(ToolCallSpec),
    ToolResult(ToolResultSpec),
    Replay(ReplaySpec),
    Spawn(SpawnSpec),
    Await(AwaitSpec),
    Barrier(BarrierSpec),
    Subgraph(SubgraphSpec),
    Loop(BoundedLoopSpec),
    Marker(MarkerSpec),
}
```

`GraphTemplate` holds typed channels, reducers, nodes, static and conditional
edges, named bodies, and endpoint-independent semantics. `TraceInstance` holds
initial state, selected branches, arrival/timing facts, recorded outputs,
per-trace overrides, and source references. Template identity and trace or
invocation identity are different types.

This vocabulary supports a Pinterest-style router/safety/tool graph, a
LangChain run tree, a tool-using coding agent, or a captured serving workflow.
It is not a claim that every source has complete evidence for every field.

### Scope and stabilization order

`SemanticGraph` is a destination architecture, not an instruction to freeze a
universal ontology now. Its variants remain provisional until a source fixture
and conformance test establish their authority, normalization, and lowering
behavior. The first implementation work is the evidence/provenance contract
and two concrete source-to-lowering slices; only then may a failing fixture
justify adding or stabilizing another semantic variant. This prevents a
lowest-common-denominator enum or opaque extension bags from encoding guesses
about heterogeneous sources.

### Semantic invariants

- A channel declares its type, reducer, streaming behavior, and producer
  requirements. Initial state seeds a reducer but never counts as an arrival.
- A join is explicit. Multiple static predecessor edges schedule work but do
  not imply AND fan-in; requirements or a `BarrierSpec` express it.
- A conditional edge preserves the selected branch and, when authored,
  distribution metadata. Unchosen branches stay semantically distinct.
- `BarrierSpec` records `all`, `any`, or quorum policy, timeout, closure
  reason, winner, and cancelled participants.
- `SpawnSpec`/`AwaitSpec` preserve invocation hierarchy, state inheritance,
  output mapping, and cancellation ownership. A child worktree policy is part
  of its invocation lease.
- Tool intent and tool observation are distinct, correlated by a stable call
  id, and retain argument/result artifact references separately.
- A loop is structurally bounded by a maximum iteration count and explicit
  break/aggregation contract. Unbounded graph cycles are invalid.

## Native lowering and capability contract

Semantic representation is not direct execution. A Rust-native lowerer turns
one `SemanticGraph` plus an `ExecutionProfile` into an owned executable
program and a report:

```rust
trait GraphLowerer {
    fn lower(
        &self,
        graph: &SemanticGraph,
        profile: ExecutionProfile,
    ) -> Result<(ExecutableProgram, LoweringReport), LoweringError>;
}
```

The initial executable profile may contain only `Llm` and sandboxed `Tool`
operations. Other semantic operations lower to deterministic replay, an
`InvocationLease`, controller-owned barrier coordination, bounded driver
composition, or an explicit refusal. No source meaning silently disappears.

The initial lowered node type is deliberately named `ExecutableGraphNode`, not
`SemanticNode` or an unqualified canonical `GraphNode`:

```rust
enum ExecutableGraphNode { Llm(LlmNode), Tool(ToolNode) }
```

`GraphTraceProgram` carries this executable program. The broader
`SemanticGraph` remains the canonical source-preserving model and must lower
through the report above before it enters this narrow runtime.

The initial `ToolNode` is a recorded-command execution shape, not a stable
provider-neutral live-tool wire contract. A stable live-tool representation
must add a call id, structured request and result references, artifact
references, channel requirements, and typed completion, error, and cancellation
state. Until then, no adapter may erase tool-call/result correlation merely to
fit this narrow executable node.

`LoweringReport` records the chosen mechanism, required capability, and
per-node outcome. The runtime uses native `Clock`, channel-store, tool,
artifact, verifier, and invocation-lease traits; it does not emulate Python
dispatch or task ownership.

### Transform closure

Every warmup, snapshot, retry, graph-variant, or import rewrite runs a native
closure validator before execution. It rejects dangling channel producers,
handles, invocation references, graph bodies, terminal paths, and joins whose
requirements no longer match surviving producers. This is a correctness gate,
not best-effort linting.

A rewrite returns either a transformed graph plus `TransformationReport` or a
typed refusal; returning the unchanged graph is not a successful substitute for
an unsupported transform. The same rule applies to named graph/body resolution:
a missing reference is a typed validation failure, never a warning followed by
top-level graph execution. Closure validation runs after import, lowering, and
every rewrite, immediately before execution, and includes tool input/call
correlation as well as channel, invocation, and body references.

## Source fidelity and provenance

Fidelity is a vector of facts, never one adapter-wide promise. For each
request, parameter, content item, token/cache count, timing observation,
tool-call/result link, topology edge, child lifecycle, terminal path, and
response/output, record one evidence grade:

```text
ObservedWire | ObservedSource | DerivedDeterministically
| InferredHeuristic { algorithm, inputs, confidence }
| Synthesized { method, seed } | UserOverride | Missing
```

The import outcome is computed from this vector and the execution policy:

```text
faithful | lossless_normalized | lossy_normalized | synthetic | unsupported
```

An `ImportReport` contains source bytes/digest, source schema/version, source
location map, companion artifact authority, every transformation, before/after
reference, reason, evidence grade, and strict-mode disposition. Raw source and
normalized graph have separate immutable identities.

Strict evaluator-grade import rejects missing request content/parameters,
unresolved content references, missing tool correlation, or heuristic child
topology. Permissive performance mode may execute a declared degraded or
synthetic graph, but it may not upgrade the claim to faithful replay.

## Adapter policy

### Source profiles

**Conflux** is the strongest P0 serving-replay input because its proxy export
can observe provider-native request bodies, normalized responses, model and
parameter metadata, request ids, timing, and agent threads. It is not
automatically faithful: endpoint base URL may be absent, tool duration may be
inferred from inter-call gaps, parent/subagent attachment may be heuristic, and
some lowering modes collapse a child to a terminal tool result. A colocated
authoritative request-DAG artifact takes precedence over an attachment
heuristic; the report names that authority.

**SATF** is the strongest portable contract when its pinned schema and content
store resolve. It can preserve content references, tool arguments/results,
request parameters, timing, and explicit semantic nodes. Hidden reasoning,
some response identities/order, unresolved content, and synthesized
performance-only calls remain limitations. Strict mode rejects unresolved
declared references instead of allowing a quiet synthetic fallback.

**Dynamo** is valuable for workflow identity, session affinity, KV/cache shape,
topology, and timing experiments, but is ordinarily text-free. Prompt content
is synthesized and request parameters, finish/error state, multimodal detail,
and exact placement may be absent. It is a performance/cache-stress source,
not evaluator-grade behavioral replay without companion evidence.

**LangSmith/LangChain** is a production topology/observability source. A run
tree may have partial records, and a derived union graph can turn parallel
children into conditional alternatives or substitute placeholder prompts. It
is P2 for graph-shape discovery, regression mining, and declared synthetic
load—not request-faithful replay. Strict import rejects skipped/incomplete runs
instead of silently accepting them.

**Codex and Claude Code session logs** may preserve messages and tool payloads,
but are not wire captures. Their historical shared lowering aggregated tool
batches and made a union assistant/tool chain; Claude parent-tool ids existed
in source sidecars while the lowered result made children sibling traces. Keep
parsers and raw evidence, but use one invocation topology per source capture
for strict mode rather than accepting that aggregation as lossless.

- **P0:** native recording and SATF/Conflux import. SATF is the portable
  contract; Conflux is the strongest wire-capture source. Both still emit a
  per-fact report.
- **P1:** Dynamo import for topology, timing, and cache-pressure experiments.
  Its text-free/synthesized content path is not evaluator-grade replay.
- **P2:** LangSmith/LangChain and CLI session imports for topology discovery,
  regression mining, and explicitly synthetic serving loads. A unioned run
  tree or assistant/tool chain is not automatically an original trajectory.

Unknown extensions are rejected in strict mode. A permissive import may retain
an opaque extension only when its report names the extension and every lost
execution capability.

## Future requirements

1. Implement `EvidenceGrade`, source-location maps, raw-source and normalized
   immutable identities, `ImportReport`, `LoweringReport`, and strict versus
   permissive policy before broad semantic DTO support.
2. Add a pure Rust lowerer/capability registry and a fallible closure validator;
   missing references and unsupported rewrites must be explicit outcomes.
3. Prove native SATF and Conflux importer/lowerer slices with golden
   conformance fixtures for request bytes, tool correlation, topology, timing,
   source authority, and refusal behavior.
4. Introduce and stabilize semantic variants only where those fixtures require
   them; add per-invocation event artifacts for tool correlation, barrier
   closure, cancellation, and lowering facts.
5. Add Dynamo and observability/CLI adapters only with their declared fidelity
   limitations and strict-mode refusal behavior.

## Source anchors

- `ajc/dag-v3:tests/fixtures/graph/pinterest.yaml` — production assistant
  routing, safety, tool chain, and alternate terminal paths.
- `ajc/dag-v3:docs/whitepapers/dataflow-execution-model.md` — readiness,
  channels, explicit joins, and race cancellation.
- `ajc/dag-v3:src/aiperf/dataset/loader/graph/models.py` — semantic vocabulary.
- `ajc/dag-v3:docs/whitepapers/{graph-conflux,satf-graph-replay}.md` — source
  evidence and portable replay contracts.
- `ajc/dag-v3:src/aiperf/dataset/loader/graph/adapters/{conflux,langsmith,dynamo_trace}.py`
  — source-specific fidelity limitations.
