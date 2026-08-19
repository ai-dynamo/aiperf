<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Native Harbor agentic benchmarking

## Purpose

This record defines the NativeGraph Harbor profile: a reproducible, scored,
end-to-end agent benchmark in which AIPerf's native Rust runtime owns the live
agent graph and dispatches every model request. Harbor-compatible tasks provide
the immutable task, environment, verifier, evidence, and score contract.

The primary benchmark unit is a complete **episode**:

```text
immutable trial -> native graph execution -> Rust-dispatched model calls
                -> supervised tool/environment adapters -> frozen evidence
                -> independent verifier -> append-only score
```

Graph-node or edge latency is explanatory telemetry of a scored episode, not a
substitute for its outcome. Reinforcement-learning rollout evaluation is an
episode profile over the same contract. Policy training is secondary.

## Built

The current native Harbor implementation supplies immutable local and
pinned-Git task acquisition, standard and JSON package import, external and
installed agent commands, Docker/strict-Compose environments, declared-artifact
verifier isolation, rewards, lifecycle records, regrade, and a limited ordered
multi-step layout. Schema-1.1 `NativeGraph` runs one standard-task episode
through the same matrix and verifier path. A sealed rollout selects its exact
environment protocol, runtime, stepper, and action encoder before provisioning.
Rust drives the bounded live policy/environment loop through the selected model
binding and a task-minted, secret-free Docker adapter sidecar. The sidecar uses
the task image with a private no-network workspace mount; it cannot write the
verifier workspace. Each accepted transition uploads one bounded, declared-path
workspace-patch archive, which Rust validates and atomically applies before
committing descriptor rollout evidence. The sidecar is reaped before artifact
collection, so post-terminal sidecar writes cannot affect independent
verification or scoring. The task and agent Docker networks remain `no-network`,
and Docker proves `NoAdapterEgress` for the sealed plan.
One `externally_driven` standard task can instead select the built-in
`terminal_v1` compatibility factory. The CLI prepares that factory against the
exact resolved package and trial before Docker, supervises its manifest driver
to one terminal response, then retains only a compatibility digest beside the
ordinary Harbor verifier result. This lower-fidelity path neither resolves a
model runtime nor exposes model credentials, an HTTP client, generic Docker
authority, or the raw terminal payload.

The runtime also supplies native Graph-IR scheduling, state channels, reducers,
worker-local metrics, trace-local placement, cellular folding, and injected
clocks. Its current `eval::semantic` implementation is a narrow fallible
`Llm`/`Tool`/`Barrier` lowering scaffold; it is not yet the richer
source-normalizing semantic graph described in `semantic-agent-graph.md`.

Recorded mini-SWE-agent replay is a separate built product: Rust reissues
predetermined recorded model requests and may execute recorded tool commands.
It is not a live agent loop and does not grade a Harbor task. NativeGraph closes
that gap without changing the meaning of recorded replay.

## Product profiles

### NativeGraph, the primary profile

NativeGraph is the full-fidelity profile. Rust owns the executable topology,
model bindings, model credentials, model dispatch, request/response timestamps,
tool invocation boundary, cancellation, branch/loop bounds, and episode record.
External code may implement a tool, environment, heuristic, or user-authored
policy, but it cannot directly invoke a benchmark model endpoint in this
profile. A policy can return a validated model-call intent; Rust selects the
pinned `ModelBinding`, dispatches it through AIPerf's native endpoint and
transport seams, and returns the result through the protocol.

Exact-profile preflight removes model credentials from every adapter process and
requires an enforceable network policy that prevents adapters from reaching any
benchmark model endpoint directly. The policy may deny adapter egress or route
declared egress through an enforcing AIPerf proxy, but an ordinary public-network
environment is insufficient. A provider that cannot enforce this boundary
refuses NativeGraph before provisioning; it does not downgrade silently.

All graph topology is declared in the immutable program. Rust owns static and
conditional edges, joins, bounded loops, retries, branch cancellation,
invocation hierarchy, workspace ownership, and merge policy. An adapter cannot
silently add an unbounded loop or direct network model call.

### Supervised external episode, the compatibility profile

`externally_driven` is the terminal-only compatibility profile for one standard
task. Single-task preflight validates the immutable `terminal_v1` factory,
driver selection, exact resolved trial, and lifecycle command provenance before
Docker; it rejects `--agent-command` and `--model-runtime`. The sealed factory
can only request one terminal response from its authorized driver session. Rust
continues to own task identity, process lifecycle, environment, budgets,
declared artifacts, verifier, score, and outer episode timing.

The result is classified as `externally_driven`, never exact NativeGraph. With
no capture proxy, its fidelity is `Missing`. Its verifier-authored reward and
score remain ordinary Harbor authority, while the compatibility lifecycle is
exported as the typed identity of its sole `Compatibility` lifecycle event, not
as generic verifier or frozen-attempt evidence. Raw driver terminal bytes are
bounded, digested, discarded at the private protocol boundary, and absent from
product output. The legacy `refuse` factory remains registered as an explicit
unavailable selector.

A planned optional AIPerf-native capture proxy and capture executor will
improve observability of declared HTTP(S) model or tool calls in this profile.
Proxy evidence will be observation, not control: TLS-pinned, in-process,
non-HTTP, or bypassed traffic will lower the fidelity classification rather
than being silently described as fully observed.

## Trait and seam composition

NativeGraph extends AIPerf's existing trait-and-seam architecture. It does not
introduce a second graph executor, trace-program type, driver registry, or
endpoint runtime. A normalized NativeGraph source lowers into the existing
`GraphTraceProgram`; the selected `TraceProgramDriverFactory` creates a live
driver, and the selected `AgentTurnCoordinatorFactory` advances model-dependent
turns through the existing invocation-lease, tool-dispatcher, graph-channel,
reducer, `GraphSink`, endpoint, transport, and observer seams. The Harbor
coordinator owns the outer scored lifecycle but does not reschedule individual
graph nodes.

The composition remains narrow and capability-selected; no monolithic executor,
major-slice enum dispatcher, global mutable state, or parallel registry universe
is added. Built-in components may be statically linked Rust implementations
registered at application bootstrap, but coordination depends on the same
traits used by injected tests and custom distributions.

| Responsibility | Composition contract |
|---|---|
| Time and deadlines | injected `Clock`; live adapters require `RealClock` |
| Graph profile and lowering | a NativeGraph source adapter plus `GraphLowerer`; output is an existing `GraphTraceProgram` with a live driver spec |
| Live progression | existing `TraceProgramDriverFactory` and `AgentTurnCoordinatorFactory`, extended by capability-bearing live implementations |
| Model execution | `NativeGraphModelBindingResolver` constructs the existing worker-local graph endpoint runtime and `GraphSink` path |
| State and content | existing typed channels/reducers plus `SegmentStore` and Rust-owned artifact references |
| Adapter protocol/runtime | `AdapterProtocolFactory` and `AdapterRuntimeFactory`, with role and wire-version capabilities |
| Tools and RL environments | existing `ToolDispatcherFactory` plus a narrow `EnvironmentStepperFactory` for reset/transition operations |
| Policy decisions | a live `AgentTurnCoordinatorFactory` implementation; `ExternalEpisodeDriverFactory` remains compatibility-only |
| Verification and rewards | `EpisodeEvaluatorFactory` composed over the existing verifier, `RewardDocument`, score, and regrade seams |
| Suite execution | `SuiteSchedulerFactory` and associative `EpisodeAggregator` over resource leases and cellular supplements |
| Fidelity observation | `FidelityObserverFactory`; the proxy is one compatibility observer, not an execution authority |
| Provider cleanup | `ProviderRecovery` returns a typed recovery outcome independently of healthy episode execution |
| Metrics and export | existing worker-local observation/metrics and registered exporter seams; the controller owns final folds and writes |

Registrations are frozen before workers start. Unknown names, duplicate
registrations, incompatible capabilities, or a profile that lacks one required
seam fail before environment spend. Extension traits do not make user code
trusted: external adapters remain capability-limited supervised processes, and
only Rust-owned validation, measurement, evidence, verifier, and scoring facts
are authoritative.

NativeGraph additions extend the existing `AIPerfRegistry`,
`ExecutionFactories`, and trace-driver composition points. There is no
`NativeGraphEpisodeExecutor`, NativeGraph-only graph scheduler, or aggregate
NativeGraph registry containing replacements for already-built graph, driver,
endpoint, tool, artifact, clock, or observer infrastructure.

## Immutable package, suite, and CLI contract

NativeGraph introduces a future strict standard-task schema revision. A task
declares a graph program and profile without weakening the existing schema:

```toml
schema_version = "1.1"

[native_graph]
profile = "native_graph"
program = "agent_graph.json"
model_bindings = "models.toml"
adapter_manifest = "adapters.toml"
```

`agent_graph.json` is a versioned, deny-unknown-fields Rust DTO. It declares
typed channels/reducers, nodes, static/conditional edges, branch selectors,
bounded loops, retries, invocation/workspace policy, allowed adapters, and
terminal outputs. `models.toml` pins allowed model/endpoint parameters; secrets
are injected only by the Rust-owned environment policy and never serialized in
the graph artifact. Adapter executable/configuration paths are explicitly
listed, never discovered from a directory scan.

The resolved model contract is complete enough to construct today's native
endpoint path without an eval-specific transport implementation:

```rust
struct ModelBindingSpec {
    id: ModelBindingId,
    endpoint_profile_id: String,
    endpoint_factory_id: EndpointId,
    transport_factory_id: String,
    model: String,
    urls: Vec<String>,
    streaming: bool,
    tokenizer: TokenizerBindingSpec,
    authentication: Vec<HeaderSecretRef>,
    generation: GenerationDefaults,
    max_connect_retries: u32,
    request_timeout_ms: NonZeroU64,
    capture: ModelCapturePolicy,
}

struct HeaderSecretRef {
    header: String,
    secret: ModelSecretId,
}
```

`TokenizerBindingSpec` selects the registered local tokenizer or a pinned
server-tokenizer URL and policy. `GenerationDefaults` contains only validated,
finite endpoint parameters. `ModelCapturePolicy` selects bounded raw-exchange
and redaction behavior. URLs, endpoint and transport factory ids, tokenizer,
retry, timeout, generation, and capture policy are package identity. Secret ids
are identity; secret values are not.

The `native_graph` profile requires one strict
`--model-runtime <model-runtime.toml>`. That file maps logical `ModelSecretId`
values to host environment-variable names; it cannot override any pinned model,
URL, endpoint, transport, tokenizer, retry, timeout, or capture field. The CLI
parses the package and runtime file, resolves all factory ids against the frozen
`AIPerfRegistry`/`ExecutionFactories`, resolves secret values through the
Rust-owned `SecretProvider`, builds `ValidatedEndpointProfileV2` plus the
selected `NativeTransportExecution` and input-token counter, and prepares the
existing worker-local graph endpoint runtime under `RealClock`. Resolution and
credential stripping complete before any task environment or adapter process is
provisioned. Adapters receive only binding ids and correlated model results.

The `externally_driven` profile has no Rust-owned model binding or secret
mapping. Its lifecycle record uses the distinct `externally_driven` contract,
must exactly match the immutable manifest driver argv, and cannot use
`--agent-command` or `--model-runtime`. A single-task invocation resolves one
matrix trial, prepares the exact `terminal_v1` capability, and runs the sealed
Docker compatibility executor. Unknown, mismatched, and `refuse` selectors fail
before Docker. Authored `--suite` execution remains an explicit refusal because
the suite input has no external lifecycle-provenance contract.

The current import contract uses the same schema revision and names its sole
declared driver and exact registered compatibility-factory selector explicitly:

```toml
schema_version = "1.1"

[native_graph]
profile = "externally_driven"
driver = "episode-driver"
external_driver_factory_id = "terminal_v1"
adapter_manifest = "adapters.toml"
```

Schema 1.1 NativeGraph profiles are mutually exclusive with the existing
ordered `[[steps]]` layout in the first release. Import fails before
provisioning when both are authored. A NativeGraph program is one episode with
its own bounded graph composition; existing schema-1.0 single- and multi-step
execution keeps its current behavior. The ordinary schema-1.0
`agent_command` path is also retained and is not reclassified as a streaming
`externally_driven` episode.

A suite uses a strict `suite.toml` manifest with ordered task references,
immutable task digests, graph/model/policy axes, seed schedule, repetitions,
parallelism limits, and paired-comparison factors. Native CLI surfaces are:

```text
aiperf eval --task <task-directory> [--model-runtime <model-runtime.toml>]
aiperf eval --suite <suite.toml> [--model-runtime <model-runtime.toml>]
```

The currently runnable vertical slice is deliberately narrower than the full
suite contract: a standard schema-1.1 `native_graph` task may select one sealed,
bounded environment rollout, its task and agent Docker networks must both be
`no-network`, and its caller must provide native-graph lifecycle provenance.
Rust validates the declared topology, rollout selectors, model binding, prompt,
and decision limits before provisioning. It performs model calls through the
selected AIPerf endpoint/transport/tokenizer seams; the Docker session accepts
only its task-minted adapter sidecar start and produces descriptor-only
reset/transition evidence. Accepted transitions carry bounded workspace patches
that Rust atomically commits to the verifier workspace; no adapter mount reaches
that workspace directly. A task becomes one resolved, independently verified, scored matrix
trial. `--suite` uses that same path only when exactly one resolved trial
matches the supplied lifecycle request. Externally driven graphs and
multi-lifecycle suite provenance remain typed refusals in `--suite`. The
`externally_driven` single-task path independently resolves one trial without a
model runtime and runs only the exact `terminal_v1` driver capability.

The importer records both the complete source snapshot and a resolved native
graph plan. In addition to the complete source digest, the package identity
extends the existing executable-source projection: it retains the complete
environment and verifier/test bindings and adds every selected graph, model,
adapter executable/configuration, driver, and policy byte. A suite digest
includes ordered task references and resolved axes. A rerun creates a new
attempt, never overwrites evidence.

## Duplex supervised-adapter protocol

Each external adapter is a supervised child process in the task-owned
environment. The initial wire representation is versioned newline-delimited
JSON with strict Rust DTOs, a maximum frame size enforced before JSON
allocation, bounded queues, monotonic per-episode sequence numbers, and
concurrent bounded stdout/stderr drains. Stderr is diagnostic-only.

The protocol is duplex and operation-correlated:

```text
Rust -> Hello { protocol, adapter_role, capabilities }
node -> Ready { protocol, supported_capabilities, implementation_digest }
Rust -> Reset { episode, seed, immutable identities }
node -> ResetAck { effective_seed, implementation_digest }
Rust -> InvokeTool | Decide | DeliverModelResult | ResetEnvironment
node -> ToolResult | ModelIntent | Decision | EnvironmentReset | Checkpoint | Ack
Rust -> StepEnvironment { action_ref }
node -> Transition { observation_ref, reward, terminated, truncated, info_ref }
node -> PutArtifactRequest
Rust -> PutArtifactHandle | ArtifactCommitted
node -> GetArtifactRequest
Rust -> GetArtifactHandle
Rust -> Cancel | Shutdown
node -> CancelAck
```

`ModelIntent` carries a call id and validated inputs, not an endpoint credential
or raw network authority. Rust rejects an intent outside the pinned binding or
declared graph transition, then dispatches the model call itself and supplies a
correlated `DeliverModelResult`. `InvokeTool` gives a tool adapter a frozen input
snapshot and returns a typed `ToolResult`. Every response carries the operation,
episode, span, and sequence ids it satisfies.

When the compatibility runner executes an `externally_driven` episode, only its
driver is permitted to emit an `EpisodeTerminalCandidate`.
NativeGraph tool, environment, heuristic, and policy adapters terminate only
their correlated operation; Rust alone evaluates graph terminality and emits
the canonical episode terminal record. RL reset produces the initial observation
through `EnvironmentReset`. Each
`StepEnvironment`/`Transition` pair advances exactly one declared environment
step. For a package-selected encoder, the adapter must consume that action from
Rust through its one-shot bounded artifact-read grant before Rust accepts the
transition. Rust derives discounted and undiscounted return from this authoritative
transition stream and gives the frozen stream to the verifier for an independent
return check.

Artifact exchange is Rust-owned: bounded upload/download handles or read-only
mounted snapshots have declared quotas; Rust hashes and atomically freezes bytes
before publishing an artifact reference. Children cannot mint trusted digests or
hand a verifier a mutable path. Startup, reset, heartbeat, idle, operation,
cancellation-ack, and forced-reap deadlines are distinct and recorded.

## Time, execution, and workspace authority

Supervised live adapters run under `RealClock`. Rust send/receive timestamps
and native transport measurements are authoritative; adapter timestamps are
retained as source-reported evidence only. Reports distinguish queue, adapter,
model-provider, tool/environment, critical-path, and end-to-end durations.
Budget charging is explicit and separate from setup/teardown measurement.

Each parallel invocation owns an `InvocationLease`: an immutable source snapshot
and a copy-on-write or cloned mutable worktree. A graph must declare how branch
outputs merge. Parallel branches never write the canonical task worktree and a
verifier sees only frozen, causally selected output artifacts.

## Reset and parallel suites

Fresh-process adapters are universally valid. Reusable adapters are opt-in and
must acknowledge `Reset` before each episode. A reset failure kills and reaps
the worker; no episode runs afterward. Successful reuse is classified as
declared and conformance-tested, not magically proven: acceptance compares
seeded fresh-worker baselines, uses contamination canaries, and records adapter
implementation/environment identity. Workers are never reused across tasks.

Rust deterministically expands suite axes, identities, seeds, artifact paths,
stable output order, and cellular folds. It does not claim deterministic wall
clock or model output from a live shared provider. Resource/capacity leases,
endpoint interference, retry policy, invalid denominators, repetitions, and
multiple-comparison policy are explicit report inputs.

The bounded matrix scheduler is established around the first independently
scored single-episode vertical slice. Bounded dynamic controls enter through
the same `EpisodeRunner`, resource-lease, stable-order, attempt, and aggregation
contracts; they do not add alternate direct-execution paths. RL, compatibility
capture, and cellular extensions use those same contracts. A package may now
retain strict rollout environment selectors, including its action-encoder id;
Rust resolves those selectors exactly and refuses missing or incompatible
registrations before adapter provisioning. The current action-encoder seam
freezes a selected decision into a Rust-owned artifact and runs the bounded
live policy-to-environment loop. The adapter must read the selected action
through Rust's one-shot artifact grant before it can return that transition.

Controller-local cellular execution now has sealed, bounded result receipts
that bind a completed-attempt digest to the issued plan, grant, cell, task,
trial, and attempt before one ordered fold can release capacity. It remains a
local controller boundary: Velo transport, remote artifact transfer, and cell
launch integration are not yet implemented. External compatibility likewise
uses the bounded digest/counter observation profile, executes one sealed terminal
driver transaction, and emits only its typed compatibility lifecycle-event
identity. It does not capture model calls or retain raw driver data, so its
current fidelity is always `Missing`.

## Results, verifier, and RL authority

Result axes are orthogonal:

- integrity: `valid` or `invalid` task/environment/protocol evidence;
- execution: `completed`, `failed`, `cancelled`, or `truncated`;
- score: `verified` or `unavailable`; and
- comparability: `scored` or explicitly `unscored`.

A verifier may assign a zero or negative verified score to a failed agent
attempt; valid scored failures remain in success/reward denominators. Invalid
infrastructure/verifier outcomes remain separate.

The verifier receives frozen declared evidence in its own workspace. It never
reads a live mutable agent environment under the same isolation guarantee. A
task may retain a Rust-captured observation stream, but that stream is frozen
and digest-addressed before verification.

RL evaluation declares trusted environment identity, horizon, termination versus
truncation, gamma, return aggregation, authoritative reward source, and the
verifier's independent return check. Rust records each observation/action/reward
reference and derived return. Training/update loops are outside this delivery.
Only after the complete evaluation release gate proves immutable scored rollout
lineage may a separate implementation plan define train/evaluation splits,
checkpoint digests, update lineage, and the hard rule that evaluation trials
cannot mutate policy or environment state.

## Provider recovery policy

Docker cleanup after an uncertain provider-side create is an open policy while
the native provider contract is finalized. It must never be presented as an
ordinary successful benchmark completion without an exact-resource cleanup
outcome; any recovery tail is recorded separately from the healthy episode
measurement.

## Future requirements and acceptance

1. Implement strict package/suite DTOs, source projections, resolver, digests,
   and CLI diagnostics with unsupported schema refusal.
2. Implement the duplex adapter state machine, bounded artifacts, process
   supervision, and a non-Rust conformance adapter fixture.
3. Extend the built NativeGraph lowerer/executor only through declared adapter
   contracts, preserving Rust-owned model dispatch and typed refusal for
   unsupported nodes.
4. Extend the bounded scored-episode scheduler with fresh/reused adapter
   semantics and stable cellular folding through that same scheduler.
5. Implement verified episode records, proxy fidelity classifications, paired
   comparisons, and RL rollout evaluation. Defer training to a separate plan
   written only after immutable evaluation lineage passes the release gate.
6. Add real Docker end-to-end acceptance for model-call authority, protocol
   violations, model/tool cancellation, artifact mutation, branch worktree
   isolation, reset contamination, verifier isolation, seeded suite expansion,
   regrade, valid failed/zero-score denominators, and an adapter attempt to bypass
   Rust by contacting the benchmark model endpoint directly. The live-path
   fixture must run twice with deterministic model response A selecting edge/tool
   A and response B selecting edge/tool B; in both runs the selected tool or
   environment observation must change the next Rust-dispatched model request,
   and the independent verifier must produce distinct expected scores. A
   pre-authored or response-ignoring path must fail this acceptance.

## Source anchors

- `rust/runtime/src/graph/` — native graph execution, channels, reducers, and
  trace placement.
- `rust/runtime/src/eval/semantic/` — current narrow semantic lowering scaffold.
- `rust/runtime/src/eval/execution/` — Harbor task, sandbox, verifier, artifact,
  Docker, Compose, lifecycle, and regrade boundaries.
- `rust/runtime/src/engine/graph_input.rs` and
  `rust/runtime/src/graph/recorded/agent_recording/` — recorded-agent replay
  adapter, distinct from the proposed live NativeGraph executor.
- `rust/runtime/src/eval/native_graph/factories.rs` — exact `terminal_v1`
  preparation and the legacy unavailable selector.
- `rust/runtime/src/eval/native_graph/episode_runner.rs` — sealed Docker
  compatibility transaction and ordinary Harbor completion.
- `rust/cli/src/eval/native_graph.rs` — single-task NativeGraph and external
  compatibility CLI composition plus typed lifecycle-event result rendering.
- `rust/e2e-tests/tests/test_harbor_external_compatibility.rs` — real Docker
  product acceptance for Missing fidelity, verifier score authority, and raw
  terminal-payload exclusion.
