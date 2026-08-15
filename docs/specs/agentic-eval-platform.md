<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Agentic SWE evaluation platform

## Purpose

This record defines how AIPerf grows from a high-fidelity agent-shaped inference
load generator into an evaluation substrate for software-engineering agents.
It does **not** define a new benchmark or make AIPerf the authority for a
single public leaderboard. AIPerf is the reproducible execution, evidence,
comparison, and task-integrity layer underneath a portfolio of task suites.

The design has three independent concerns:

1. **Task quality** — whether a task, environment, and verifier fairly measure
   the stated engineering objective.
2. **Agent quality** — whether an agent solves the task safely and correctly.
3. **System performance** — the latency, cost, resource consumption, cache
   behavior, and throughput of the complete agent graph.

No aggregate result may silently conflate these concerns. A task failure caused
by infrastructure drift, an invalid verifier, or denied sandbox capability is
not an agent failure. Likewise, a correct task result does not make an
unboundedly expensive or unsafe graph a production-quality agent.

The primary source design is the planned recorded-agent replay port. This
record supplies the product-level experiment and evidence model around its
`GraphTraceProgram`, heterogeneous nodes, tool sandbox, and controller-owned
artifact seams.

## Built

AIPerf already supplies the substrate this design composes over:

- `GraphTracePlan`, graph state channels, static edges, reducers, prompt
  materialization, graph gates, and trace-local placement;
- deterministic virtual and wall-clock graph execution;
- worker-local request measurement, cellular execution, controller-owned
  folding, immutable segment stores, and artifact exporters;
- AgentX's byte-exact legacy replay path, which remains a parity reference
  until Graph-IR supersedes it; and
- a planned recorded-agent replay design that introduces `GraphTraceProgram`,
  `GraphNode::{Llm, Tool}`, trace environment recipes, tool dispatch, replay
  metrics policy, and durable trace supplements.

The built Graph-IR is an LLM trajectory executor. It does not yet execute a
live SWE-agent tool loop, grade a task, or own a benchmark registry. This is a
deliberate boundary: existing `dag_jsonl`, WEKA, and Dynamo inputs must retain
their LLM-only behavior and their eligible flat-graph fast path.

## Product model

The durable identity hierarchy is:

```text
EvalSuite@version
  -> DatasetManifest@version
    -> TaskSpec@digest
      -> TrialSpec
        -> GraphTraceProgram
          -> Attempt
            -> Span/Event DAG + immutable artifacts
              -> Verifier results, scores, and human review
```

### Task and dataset identity

```rust
struct TaskSpec {
    id: TaskId,
    digest: Blake3Digest,
    instruction: ArtifactRef,
    environment: SandboxRecipe,
    verifier: VerifierSpec,
    agent_contract: AgentContract,
    resource_budget: ResourceBudget,
    policy: PolicySnapshot,
    provenance: TaskProvenance,
}

struct DatasetManifest {
    id: DatasetId,
    version: DatasetVersion,
    tasks: Vec<TaskRef>,
    selection_policy: SelectionPolicy,
}
```

A `TaskRef` contains the task id and immutable digest. A manifest is a
reproducible selection, not a directory scan or an online registry lookup. A
local/private manifest is fully valid. Registries may distribute manifests but
are never required for offline execution.

`TaskProvenance` records source repository and base revision, task acquisition
time and contamination cutoff, language/OS/build traits, reference-patch
availability, and known task-health findings. The reference patch is evidence
for task QA; it is not the sole grading oracle.

### Trial and experiment identity

```rust
struct TrialSpec {
    task: TaskRef,
    graph: GraphVariantRef,
    model: ModelSpec,
    seed: u64,
    budget: TrialBudget,
    policy: PolicySnapshot,
    environment_digest: Blake3Digest,
}

struct ExperimentSpec {
    trials: TrialMatrix,
    repetitions: NonZeroU32,
    aggregation: AggregationPolicy,
    comparison: Option<PairedComparisonSpec>,
}
```

The resolved `TrialSpec` is immutable. It pins model/provider parameters,
prompts and tool schemas, graph version, dataset/task versions, image and
verifier digests, policy/network grants, seed, source revision, and runtime
environment. Results are never overwritten; a rerun is a new attempt attached
to the same trial identity.

`PairedComparisonSpec` makes the graph itself experimentally testable. It holds
task, model, seed, policy, image, and budget fixed while varying exactly one
declared factor such as planner strategy, critic, branch count, context policy,
tool interface, verifier feedback, or retry budget. Its report includes paired
success delta and confidence interval, latency/cost delta, critical-path change,
tool/token distributions, and task-level failure movement.

## Graph and execution model

`GraphTraceProgram` is the unit placed, resumed, warmed, and measured. It
extends the existing trace plan with optional trace-local warmup, a profiling
graph, environment recipe, driver, and replay metadata. Environment setup and
teardown are outside profiling time, with separately reported durations.

Nodes become a tagged union:

```rust
enum GraphNode {
    Llm(LlmNode),
    Tool(ToolNode),
    Spawn(SpawnNode),
    Join(JoinNode),
    Gate(GateNode),
    Checkpoint(CheckpointNode),
    Evaluate(EvaluateNode),
}
```

`LlmNode` continues to use the shared endpoint/transport/observer path.
`ToolNode` writes a structured observation to normal graph channels but consumes
no inference request credit. `EvaluateNode` invokes a declared verifier and is
not allowed to mutate the agent's canonical worktree.

The scheduler never decides whether a live agent has another turn. A
`TraceProgramDriver` owns recorded replay versus live progression;
`AgentTurnCoordinator` is the future live-agent decision seam; and an
`AgentToolCallDecoder` plus `AgentObservationFormatter` bridge model-specific
tool calls to provider-neutral `ToolNode` inputs and outputs.

Trace placement remains local: parent and descendants stay on one worker-local
current-thread runtime. Do not add `Arc<Mutex<_>>` coordination to the request,
token, or node hot path. Cells return bounded serializable supplements; the
controller performs final artifact writes and associative result folding.

### Branch isolation and merge

A spawned agent may not concurrently mutate the canonical task workspace.
Every implementation branch receives its own copy-on-write overlay or cloned
workspace. A branch returns an immutable candidate patch/artifact reference.
Only an explicit selector or merge node can apply a candidate to a subsequent
canonical workspace snapshot. This is both an execution-correctness and
security boundary.

## Sandbox and verifier contracts

An image name is insufficient task identity. `SandboxRecipe` includes image
digest, base revision, staged artifacts, mounts, working directory, interpreter
and setup commands, command policy, resource limits, network mode, allowed
hosts, secret policy, and cleanup behavior.

Capability preflight is fail-closed. The executor must reject a trial before
environment spend when its provider cannot meet persistent-workspace,
read-only-base, overlay, network, descendant-termination, staging, isolation,
or resource guarantees required by the task or agent contract.

The verifier runs in a separate sandbox or a separately restored workspace
snapshot. It receives the candidate patch and permitted evidence, not the
agent's ambient credentials or mutable control channel. Verifiers should prefer
functional tests, properties/metamorphic checks, and negative controls over
gold-patch equality so alternate correct implementations can pass.

Tool-executing programs require wall-clock online execution. They are rejected
for `SimClock`, dry-run, virtual transport, and open-loop replay unless a
deterministic tool simulator is explicitly declared; otherwise tool durations
would be fabricated or double-counted.

## Evidence, replay, and review

Every attempt emits an append-only, versioned event log. It is more than a
prompt transcript:

```text
Run -> TaskSample -> Attempt -> Span
```

Span types are `agent`, `llm`, `tool`, `sandbox`, `artifact`, `evaluator`, and
`security`. Each event records stable ids, parent id, node/turn/tool-call id,
attempt number, timestamp, input and output snapshot digests, model request and
response references, tool request/result references, sandbox state digest,
budget before/after, artifact references, and typed terminal/error state.

The schema supports native and imported trajectories. Import adapters may ingest
SWE-agent, OpenHands/ATIF, or other JSONL records, but imported evidence retains
its source schema/version and does not pretend to be natively observed.

Checkpoint after every graph-node terminal state and tool boundary. Resume uses
stable sample ids and preserves completed samples and immutable artifacts. It
records retry count, retry reason, and retry budget separately so retry-induced
distribution shifts cannot appear as a free quality improvement.

Scores are independent, versioned records:

```rust
struct Score {
    subject: SubjectRef, // run, sample, span, or artifact
    evaluator: EvaluatorRef,
    value: ScoreValue,
    verdict: Verdict,
    rationale: ArtifactRef,
    evidence: Vec<ArtifactRef>,
}
```

Human review is a structured queue over a trace/span and its evaluator evidence.
Labels include verdict, failure taxonomy, severity, confidence, and repairability.
LLM judges may supply quality signals but are never security hard gates and must
be calibrated against reviewed examples.

## Evaluation and task health

Each task result contains separate dimensions:

1. **Replay fidelity**: bytes, tool order, lifecycle, and environment behavior
   match the declared recording or contract.
2. **System performance**: model and tool wall time, queue/setup time, CPU,
   memory, network, tokens, cost, cache behavior, end-to-end latency, and
   critical-path metrics.
3. **Task quality**: deterministic verifier result, artifact/patch evidence,
   policy/security verdict, and optional calibrated judge/human scores.

Promotion policies are declarative portfolios of gates. Typical hard gates are
functional verifier pass, sandbox/policy pass, artifact integrity pass, and
bounded cost/latency. Aggregate quality is reported with uncertainty and cannot
mask a failed hard gate.

Task health is itself a graph and a versioned result:

```text
build -> oracle/reference solve -> negative control -> repeated known-agent runs
      -> drift classifier -> quarantine or version bump -> publication
```

`TaskVerdict` is `valid`, `conditionally_valid`, or `broken`, with reasons,
confidence, image/dependency/verifier digests, expected resource envelope, and
evidence. Infrastructure-invalid, task-invalid, and agent-failed trials remain
distinct. Invalid tasks are excluded from aggregate capability scores.

Security task families are maintained separately from frozen regressions and
collect system evidence: changed-file manifests, command output, canaries,
protected-file hashes, host probes, tool/MCP poisoning, repository prompt
injection, terminal-output injection, exfiltration attempts, and verifier
sabotage attempts.

## Task factory and suite policy

Task acquisition creates candidates, not trusted benchmarks:

```text
issue/PR ingestion -> base-commit reconstruction -> setup synthesis
  -> failing-test reproduction -> task/verifier authoring -> independent review
  -> oracle + candidate-agent ensemble -> human acceptance -> versioned publish
```

The factory stores provenance, source cutoff, leakage risk, repository setup
evidence, patch complexity, and independent reviewer outcomes. AIPerf should
maintain multiple suite classes: a small stable nightly sentinel set, broader
periodic suites, fresh public-issue suites, original functional tasks, and
private production regressions. No single public benchmark is a general
capability score.

## Future requirements

Implement in this dependency order:

1. Add immutable `TaskSpec`, `DatasetManifest`, `TrialSpec`, and resolved
   artifact/provenance DTOs without changing current graph inputs.
2. Add `GraphTraceProgram` and the provider-neutral event/artifact schema.
3. Add `GraphNode::{Llm, Tool}` and the tool-dispatch/sandbox capability seams.
4. Add controller-owned checkpoints, attempt/retry records, and safe resume.
5. Implement the recorded-response driver first; add the live-agent coordinator
   only after tool and environment lifecycle are proven.
6. Add separate verifier execution and the three-layer score model.
7. Add task-health validation and quarantine before publishing benchmark
   results or accepting generated tasks.
8. Add paired graph-variant experiments and comparison reports.
9. Add task-factory, functional-verifier, and production-trace-to-regression
   workflows once the evidence model is stable.

## Source anchors

- `docs/specs/recorded-agent-replay-rust-port.md` — planned Graph-IR extension
  for recorded agent replay and live tools.
- `docs/specs/graph-runtime.md` — built graph runtime and dataflow constraints.
- `docs/specs/execution-model.md` — worker-local execution and measurement
  boundaries.
- `docs/specs/cellular.md` — controller/cell ownership and folded results.
- `docs/specs/accuracy.md` — evaluator injection and grading separation.
- `rust/runtime/src/graph/` — current graph model, executor, policies, and
  trace placement.
- `rust/runtime/src/agentx/` and `rust/runtime/src/agentic_replay.rs` — legacy
  parity and recorded agent-shaped traffic path.
