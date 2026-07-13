<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AIPerf: `aiperf-runner` as the sole native execution surface

**Date:** 2026-07-11
**Author:** Anthony Casagrande (Tech Lead) + Codex
**Status:** decided — partially built; mode reachability incomplete
**Decision:** every end-user AIPerf execution backed by the native Rust implementation enters
through a versioned `aiperf-runner` operation. There is no native `aiperf` CLI, Python inference
fallback, second Rust executable, direct Python-to-Rust library binding, or mode-specific process
surface.

**Companions:**

- `2026-07-11-python-orchestrator-rust-single-run-design.md` owns Python outer-loop orchestration
  and the fresh-process boundary.
- `2026-07-11-aiperf-runner-owned-endpoint-registry-design.md` owns endpoint identity, catalog,
  preparation, profiles, and endpoint validation.
- `2026-07-10-steppable-clock-injected-engine-design.md` owns the Dynamo offline engine/clock/parity
  behavior.
- `2026-07-09-graph-ir-rust-port-design.md` and `2026-07-10-unified-graph-runtime-design.md` own
  Graph-IR semantics and runtime behavior.
- `2026-07-10-aiperf-rust-accuracy-accumulator-design.md` owns static and stateful canonical
  evaluator/provider behavior.

This spec owns **product reachability**: the operation envelope, backend/workload selection,
capability negotiation, feature-bearing runner distributions, common preparation/execution/report
contract, and subprocess proofs that make those library implementations usable through the only
native executable.

---

## 0. Thesis

The native product is one executable with multiple statically composed backend and workload
implementations:

```text
Python aiperf command
    |
    | structural Config v2, sweeps, trials, search, presentation
    v
strict RunnerRequest
    |
    v
aiperf-runner
    |
    +-- RunnerRegistry
    |     +-- endpoint factories
    |     +-- dataset/sampler factories
    |     +-- backend factories
    |     `-- workload factories
    |
    +-- validate
    |     `-- authored request -> validated plan (+ deferred checks)
    |
    `-- execute
          +-- online HTTP  + scheduled / graph / static accuracy / agentic
          `-- Dynamo offline + scheduled / graph
```

ONLINE-real and ONLINE-mock are the same `online_http` backend. They differ only in configured
target URL; a mock HTTP server is not another runner mode. OFFLINE-mock is the feature-gated
`dynosim` backend using `SimClock` and the in-process steppable engine.

Backend and workload are orthogonal selections. A new backend does not acquire its own CLI or
duplicate every workload DTO, and a new workload does not acquire its own transport. Factories
lower the versioned request into today's real seams—`Clock`, `RequestSink<R>`, `TurnDispatcher`,
`GraphSink`, phase/runtime traits, observers, and reporters—not an aspirational parallel harness.

---

## 1. Current code truth

### 1.1 Product-reachable through runner v1

The current `crates/aiperf-runner` protocol and executor expose:

- scheduled online HTTP phases: concurrency, Poisson, Gamma, constant/request-rate,
  user-centric, and fixed schedule;
- native datasets, endpoint dialects, tokenizer configuration, phases, ramps, cancellation,
  adaptive control, metrics, and artifacts;
- static evaluator-backed accuracy through a supervised Python worker;
- GPU/server/network telemetry and live Python OTel/MLflow sidecars;
- one native-v2 report and strict terminal response.

This is the built center of the target architecture.

### 1.2 Built as Rust libraries but not runner-v1 reachable

The following remain implemented but lack an end-user request in runner protocol v1:

| Capability | Current Rust home | Missing runner surface |
|---|---|---|
| Online Graph-IR | `aiperf-graph`, `aiperf_graph::runtime::drive_real` | graph workload DTO, runner composition, subprocess proof |
| Dynamo scheduled offline | `aiperf::dynosim` behind `dynosim` | backend DTO, feature forwarding, capabilities, report/terminal projection |
| Dynamo Graph-IR offline | `aiperf::dynosim::run_graph_offline` and graph DES | backend/workload pair, feature forwarding, subprocess proof |
| Stateful agentic accuracy | `aiperf::agentic`, `agentic_gateway`, canonical Python providers | agentic workload DTO, callback/gateway lifecycle, runner subprocess canaries |

The removed native `aiperf` CLI is not an alternate route. Focused library tests prove the
algorithms but do not make them product-reachable.

### 1.3 Current protocol is not extensible enough

Runner v1 deserializes one fixed `RunRequest` and then executes it. It has no operation tag,
backend selection, Graph-IR workload, stateful agentic workload, or offline feature inventory.
`aiperf-runner` depends on the `aiperf` library but does not forward its `dynosim` /
`dynamo-full` features.

Capabilities are handwritten static arrays rather than a description of the exact composed
runner. This spec and the runner-owned endpoint companion replace that pattern with frozen
registries and deterministic descriptors.

---

## 2. Required invariants

1. **One native executable.** Every native AIPerf run uses `aiperf-runner`.
2. **Fresh process per run.** Validation may use a separate short-lived process, but execution
   retains one fresh child per variation/trial.
3. **No Python inference fallback.** Missing runner capabilities fail before execution.
4. **Orthogonal backend/workload selection.** The request describes both; compatibility is
   validated explicitly.
5. **Trait-backed registries.** Backend and workload IDs select statically linked factories, not
   central string branches or a closed enum of implementation kinds.
6. **Exact-distribution handshake.** Capability, validation, and execution processes prove the
   exact selected runner image through `distribution_id`.
7. **Authored request first.** Python projects structural/authored state without creating run
   artifacts, warming tokenizers, fetching datasets, or importing native-equivalent loaders.
8. **Rust preparation.** Rust resolves every input it already knows how to resolve, performs
   deferred semantic checks, and creates artifacts only after complete validation.
9. **One endpoint registry.** Every backend/workload pair receives prepared endpoints from the
   runner-owned registry; modes do not construct private built-in catalogs.
10. **One transport/clock timeline.** Workloads use the injected backend and clock seams. No mode
    measures or schedules outside them.
11. **One report family.** Every successful operation writes native-v2 with common run identity,
    metrics, backend/workload provenance, and mode-specific typed blocks.
12. **Fail-closed optional features.** A request for an uncompiled backend, workload, topology,
    router, offload, evaluator capability, or report feature is rejected before side effects.
13. **Canonical external providers remain external.** Supervised Python evaluator/environment
    workers remain when they own canonical benchmark semantics; they never become an inference
    executor.
14. **Delete duplicate Python behavior.** When a Rust capability is runner-reachable, the Python
    implementation of that AIPerf capability is removed rather than retained in parallel.
15. **Dynamo product separation.** The Python `aiperf dynosim` facade may continue to expose
    Dynamo-owned products and raw canonical parsers. It is not an AIPerf-runner fallback and does
    not satisfy AIPerf offline reachability.

---

## 3. Protocol-v2 operation envelope

### 3.1 Request

Every stdin operation uses one strict tagged envelope:

```rust
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunnerEnvelopeV2 {
    pub protocol_version: ProtocolV2,
    pub operation: RunnerOperationV2,
    pub expected_distribution_id: String,
    pub run: AuthoredRunSpecV2,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunnerOperationV2 {
    Validate,
    Execute,
}
```

Wire examples:

```json
{
  "protocol_version": 2,
  "operation": "validate",
  "expected_distribution_id": "blake3:...",
  "run": {}
}
```

```json
{
  "protocol_version": 2,
  "operation": "execute",
  "expected_distribution_id": "blake3:...",
  "run": {}
}
```

The implementation first decodes only enough of the envelope to select a supported version, then
strictly decodes that version's full DTO. Unknown versions, operations, and fields fail closed.

`--capabilities` remains the only command-line operation and writes one capability line. Every
run/validation input stays on stdin so secrets do not enter argv or process listings.

### 3.2 Responses and exit codes

| Operation | Success event | Failure event |
|---|---|---|
| `validate` | `run_validation` with `success=true` | `run_validation` with typed errors |
| `execute` | `run_terminal` with `success=true` and report path | `run_terminal` with typed stage/error |

Each process writes exactly one non-empty JSON line to stdout and then exits:

| Exit | Contract |
|---:|---|
| `0` | operation succeeded |
| `1` | well-formed operation failed semantically, during preparation, or during execution; typed stdout response present |
| `2` | bootstrap/protocol/stdout-contract failure |

Stderr is redacted diagnostic text only. It is never parsed for normal machine control.

### 3.3 Protocol-v1 compatibility

The runner temporarily accepts the current protocol-v1 fully resolved online/static-accuracy
request. New fields, backends, workloads, and extensions target v2 only. New Python requires v2
for runner-wide modes and never falls back to a legacy executor. V1 is removed after the announced
compatibility window and subprocess matrix is green.

---

## 4. Authored run model

```rust
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthoredRunSpecV2 {
    pub identity: RunIdentitySpec,
    pub artifact_target: PathBuf,
    pub models: ModelsSpec,
    pub endpoints: EndpointProfilesSpec,
    pub backend: NamedBackendSpec,
    pub workload: NamedWorkloadSpec,
    pub metrics: MetricsSpec,
    pub artifacts: ArtifactSpec,
    pub sidecars: SidecarSpec,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NamedBackendSpec {
    #[serde(rename = "type")]
    pub id: BackendId,
    pub config: Box<serde_json::value::RawValue>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NamedWorkloadSpec {
    #[serde(rename = "type")]
    pub id: WorkloadId,
    pub config: Box<serde_json::value::RawValue>,
}
```

The outer DTO is strict. Each selected statically linked factory strictly deserializes its own
`config` with `deny_unknown_fields`; using `RawValue` is not an untyped escape hatch. It lets a
linked implementation own its configuration without extending a core enum or accepting unknown
keys.

`AuthoredRunSpecV2` contains no Python `ResolvedConfig`. In particular:

- `artifact_target` is selected but does not exist yet;
- tokenizer identity/revision/trust policy lives inside the selected workload source and has not
  been cache-warmed;
- dataset path/public identity/filters/options are authored inputs, not Python loader results;
- endpoint profiles are raw and become worker-local prepared bindings in Rust;
- backend engine/router/topology inputs have not initialized an engine or socket;
- evaluator/provider configuration has not spawned a worker or environment.

Python remains responsible for structural YAML/Jinja/environment/sweep expansion and secrets
substitution. It explicitly projects every accepted field into this narrower ABI.

---

## 5. Backend registry and factory seam

### 5.1 Identity and descriptor

```rust
pub struct BackendDescriptor {
    pub id: &'static str,
    pub description: &'static str,
    pub clock_kind: ClockKind,
    pub supports_scheduled: bool,
    pub supports_graph: bool,
    pub semantic_responses: bool,
    pub feature_flags: &'static [&'static str],
}

pub trait RunnerBackendFactory: std::fmt::Debug + Send + Sync {
    fn descriptor(&self) -> &'static BackendDescriptor;

    fn validate(
        &self,
        authored: &serde_json::value::RawValue,
        requirements: &WorkloadRequirements,
    ) -> RunnerResult<ValidatedBackendConfig>;

    fn prepare(
        &self,
        config: ValidatedBackendConfig,
        context: &PreparationContext,
    ) -> RunnerResult<Box<dyn PreparedBackend>>;
}
```

The exact internal prepared-backend traits may remain split along today's typed
`RequestSink<HttpRequest>`, `TurnDispatcher`, and `GraphSink` boundaries. This spec does not require
an aspirational universal `Backend`/`Harness` trait. It requires the runner factory to lower into
those current seams without string matching or a second transport/clock path.

The mutable builder freezes before validation or execution. Duplicate IDs are rejected. Backend
descriptors and feature flags are enumerated deterministically into capabilities.

### 5.2 `online_http`

The default backend owns:

- `RealClock`;
- `aiperf-transport-http` hyper client configuration;
- h1/h2c/UDS/TLS, connection reuse, cancellation, SSE, and trace timing;
- URL selection and sticky routing;
- real inference endpoints or loopback mock endpoints with identical code.

ONLINE-real versus ONLINE-mock is not represented in the protocol. The configured URL determines
the target, and reports may classify loopback/mock provenance only when explicitly authored.

### 5.3 `dynosim`

The feature-gated backend owns:

- `SimClock` and the idle/quiescence DES pump;
- passive `SteppableReplay` initialization and terminal operations;
- engine/router JSON, aggregate/disaggregate topology, separate profiles, trace cutoff, routing,
  cancellation, and backend-owned capacity facts;
- parity comparison between AIPerf and Dynamo common summaries;
- optional router runtime, ZMQ events, KV offload, AIC forward pass, and profiling features.

`aiperf-runner` forwards Cargo features explicitly:

```toml
[features]
dynosim = ["aiperf/dynosim"]
dynamo-router-runtime = ["dynosim", "aiperf/dynamo-router-runtime"]
dynamo-zmq-events = ["dynosim", "aiperf/dynamo-zmq-events"]
dynamo-kvbm-offload = ["dynosim", "aiperf/dynamo-kvbm-offload"]
dynamo-aic-forward-pass = ["dynosim", "aiperf/dynamo-aic-forward-pass"]
dynamo-profile = ["dynosim", "aiperf/dynamo-profile"]
dynamo-full = [
    "dynamo-router-runtime",
    "dynamo-zmq-events",
    "dynamo-kvbm-offload",
    "dynamo-aic-forward-pass",
    "dynamo-profile",
]
```

The exact selected runner advertises only compiled features. Requesting an absent feature fails
static validation. Requested offload initialization remains fail-closed. An official offline-capable
runner distribution and its dependency/source/lock identity are part of the release matrix; a
default build that lacks Dynamo cannot claim offline product support.

---

## 6. Workload registry and factory seam

```rust
pub struct WorkloadDescriptor {
    pub id: &'static str,
    pub description: &'static str,
    pub requires_semantic_responses: bool,
    pub supports_real_clock: bool,
    pub supports_sim_clock: bool,
    pub required_backend_features: &'static [&'static str],
}

pub struct WorkloadRequirements {
    pub semantic_responses: bool,
    pub clock_kind: ClockKind,
    pub backend_features: BTreeSet<String>,
}

pub trait RunnerWorkloadFactory: std::fmt::Debug + Send + Sync {
    fn descriptor(&self) -> &'static WorkloadDescriptor;

    fn validate(
        &self,
        authored: &serde_json::value::RawValue,
        registries: &RunnerRegistry,
    ) -> RunnerResult<ValidatedWorkloadConfig>;

    fn requirements(
        &self,
        config: &ValidatedWorkloadConfig,
    ) -> WorkloadRequirements;

    fn prepare(
        &self,
        config: ValidatedWorkloadConfig,
        backend: &mut dyn PreparedBackend,
        context: &PreparationContext,
    ) -> RunnerResult<Box<dyn PreparedWorkload>>;
}
```

The public factory seam is object-safe and startup-only. Hot loops retain their typed/generic
current implementations after preparation; the runner does not introduce a per-token dynamic
registry lookup or shared lock.

### 6.1 `scheduled`

This workload generalizes runner v1. Its strict config owns:

- authored native dataset source and tokenizer policy;
- ordered warmup/profiling phases;
- concurrency, request-rate, user-centric, fixed-schedule, and one-pass sources;
- ramps, cancellation, adaptive control, stop bounds, slots, and samplers;
- endpoint profile materialization and per-turn prepared `EndpointKey`s.

It supports `online_http` and `dynosim`. The same phase/workload policy is injected with the
selected clock/dispatcher. Fixed schedule and dataset timing validation occur after dataset load
and before artifact creation or scheduling.

All canonical Dynamo trace formats enter through the unified dataset/fixed-schedule source or a
lowering into the same scheduled representation. They do not create a second public trace runner.

### 6.2 `graph`

The strict Graph-IR workload config owns:

- graph source/IR and unified dataset references;
- worker count, duration gate, firing-gate inputs, and deterministic merge order;
- graph endpoint materialization and response/metric configuration;
- optional offline trace/event-source inputs required by the Dynamo backend.

It supports `online_http` through `drive_real` and `dynosim` through
`drive_sim_with_source`. Each graph worker owns its metric accumulator, endpoint binding table, and
prepared sink; workers merge once in deterministic order.

Graph mode remains subject to its current feature coverage. Unsupported phase lists, arrival/slot
actuators, or other unbuilt Graph consumers fail validation rather than appearing as inert config.

### 6.3 `static_accuracy`

The strict config owns the canonical evaluator worker identity, benchmark/tasks, problem limits,
tokenizer policy, callback-free static load/grade protocol, scheduled inference phases, and typed
accuracy artifacts.

It requires semantic response text and therefore supports `online_http` only. The evaluator remains
a supervised pinned Python process; Rust owns every inference request, endpoint/transport operation,
timing event, metric, terminal response capture, and report join.

### 6.4 `agentic`

The strict config owns:

- provider/environment/benchmark identity and pinned worker selection;
- episode/task concurrency and overall inference concurrency;
- primary/environment/verifier call policy;
- optional authenticated inference-gateway binding;
- endpoint profiles, canonical agent/provider settings, cancellation, and artifacts.

It requires semantic model responses and supports `online_http` only. `aiperf-runner` supervises the
provider worker, starts the Rust `AgenticInferenceGateway` when requested, admits every callback
through the same Rust slot/scheduled/endpoint/transport path, and writes native-v2 reward,
provenance, episode, and per-purpose accounting blocks.

Harbor, AgentLab/BrowserGym, and MCPMark continue to own their canonical task/environment/agent/
verifier loops. Their workers never contact the target model except through the authenticated Rust
callback path.

---

## 7. Compatibility matrix

The built-in target matrix is:

| Backend | `scheduled` | `graph` | `static_accuracy` | `agentic` |
|---|:---:|:---:|:---:|:---:|
| `online_http` | yes | yes | yes | yes |
| `dynosim` | yes | yes | no | no |

The matrix is derived from workload requirements and backend descriptors at registry freeze; it is
not a handwritten runtime switch. Capabilities serialize the computed supported pairs. A linked
factory may add a new ID/pair without editing a core enum, subject to its trait and protocol schema.

Accuracy is online-only because a timing simulator cannot produce model-semantic answer text.
ONLINE-mock can execute accuracy only when the HTTP mock intentionally supplies semantically valid
fixture responses; that is still the `online_http` backend, not Dynamo offline.

---

## 8. Capabilities

`--capabilities` describes the exact frozen runner distribution:

```json
{
  "event": "runner_capabilities",
  "capabilities_schema_version": 2,
  "protocol_versions": [1, 2],
  "report_schema_version": "2.0",
  "distribution_id": "blake3:...",
  "backends": [
    {
      "id": "online_http",
      "clock": "real",
      "features": ["h1", "h2c", "uds", "tls"]
    },
    {
      "id": "dynosim",
      "clock": "sim",
      "features": ["aggregate", "disaggregate", "kv_routing"]
    }
  ],
  "workloads": [
    {"id": "scheduled"},
    {"id": "graph"},
    {"id": "static_accuracy"},
    {"id": "agentic"}
  ],
  "supported_pairs": [
    ["online_http", "scheduled"],
    ["online_http", "graph"],
    ["online_http", "static_accuracy"],
    ["online_http", "agentic"],
    ["dynosim", "scheduled"],
    ["dynosim", "graph"]
  ],
  "endpoints": [],
  "extensions": []
}
```

The actual descriptors include all typed feature inventories required for fail-closed validation:
phase features, graph features, Dynamo topologies/routers/trace formats/offload capabilities,
evaluator/provider capabilities, endpoint catalog, telemetry sources, and artifact formats.

No capability list is maintained separately from the frozen factories/registries. A feature-gated
implementation absent from the exact binary is absent from capabilities.

---

## 9. Validation, preparation, and execution

### 9.1 Static validation

The `validate` operation:

1. verifies `distribution_id`;
2. resolves backend/workload/endpoint IDs through frozen registries;
3. strictly deserializes their owned configs;
4. validates the backend/workload compatibility and compiled feature set;
5. validates every rule possible without external dataset/evaluator/server IO;
6. returns `completeness` plus typed deferred checks.

It creates no artifact directory, warms no tokenizer, downloads no dataset, starts no evaluator,
initializes no Dynamo engine, and sends no inference traffic.

### 9.2 Complete preparation

The `execute` operation repeats static validation and then:

1. loads/localizes the dataset and tokenizer through Rust registries;
2. discovers and binds every endpoint profile reference;
3. validates fixed timing, graph inputs, evaluator-authored sources, and other deferred content;
4. prepares a worker-local endpoint table and compiled templates/selectors;
5. prepares the backend clock/sink/engine without beginning workload events;
6. completes backend/workload compatibility validation using prepared facts;
7. creates the run artifact directory and materializes authorized user files;
8. starts supervised sidecars/evaluators and the runtime;
9. executes and finalizes reports/artifacts.

If preparation requires remote cache writes, those use a content-addressed shared cache and are not
run artifacts. A failed preparation never leaves a partially authoritative native-v2 report.

### 9.3 Execution

Every workload emits observations through the normal local-loop observer graph. Online uses
`RealClock`; offline uses `SimClock`; no feature reads wall time or `tokio::time` directly. Endpoint
bindings are worker-local, metrics are worker-local and merged once, and no optional sidecar can
backpressure request dispatch.

All terminal paths drain/cancel according to the owning workload/phase policy, finalize typed
backend/workload facts, and then serialize native-v2. Failure terminals identify `protocol`,
`validation`, `preparation`, `execution`, or `reporting` stage without leaking secrets.

---

## 10. Native-v2 and artifacts

Every report contains common provenance:

```json
{
  "run": {
    "distribution_id": "blake3:...",
    "backend": "online_http",
    "workload": "scheduled",
    "extensions": [],
    "endpoint_profiles": []
  }
}
```

Mode-specific typed blocks remain additive:

- online HTTP trace/network/TLS facts;
- Dynamo engine/router/topology/capacity/parity facts;
- Graph worker/IR/firing facts;
- static evaluator identity/accuracy records;
- agentic provider/environment/verifier/reward/episode and per-purpose accounting.

The common metric catalog and report identity do not fork by mode. Offline returns are rejected
unless the complete common AIPerf/Dynamo summary bytes match as required by the offline spec.
Numeric values crossing JSON remain finite or absent.

Rust writes native run artifacts. Python reads the terminal/report and performs remaining
presentation, cross-run aggregation, upload, and plotting work. Python compatibility artifacts are
derived from Rust-owned results and are not another metric authority.

---

## 11. Packaging and distribution selection

The normal Python installation selects a platform-specific `aiperf-runner` companion package.
Official distributions may differ in compiled optional backends, but every distribution is
self-describing and content-identified.

Discovery order remains:

1. explicit `--runner-bin`;
2. `AIPERF_RUNNER_BIN`;
3. matching installed companion package;
4. `PATH` for development.

The release matrix MUST contain:

- a stock online runner for every supported platform;
- an official offline-capable runner wherever the pinned Dynamo dependency is supported;
- source/lock/feature provenance and exact `distribution_id` for each;
- fresh-install capabilities and loopback subprocess tests;
- no silent substitution of an online-only binary for an offline request.

Custom statically linked extensions ship a custom runner distribution and are selected explicitly.
Protocol/capability/report compatibility—not Python package-version equality—is authoritative.

---

## 12. Migration plan

### Increment 1 — common v2 envelope and authored projection

1. Add exact distribution identity and capability schema v2.
2. Add `validate`/`execute` envelopes and typed terminal stages.
3. Add side-effect-free Python `AuthoredRunSpecV2` projection.
4. Split path selection from artifact creation and remove `run.resolved` from v2.
5. Keep protocol v1 online/static-accuracy compatibility.

### Increment 2 — factory registries and current online path

1. Add backend/workload builder/frozen registries.
2. Register `online_http`, `scheduled`, and `static_accuracy` factories.
3. Lower v1/v2 scheduled requests into the same current runner implementation.
4. Derive capabilities and supported pairs from descriptors.
5. Inject the runner-owned endpoint registry and prepared worker tables.

### Increment 3 — online Graph-IR

1. Define the strict graph workload config and authored Graph-IR source projection.
2. Prepare unified dataset/endpoint/worker-local metric state.
3. Compose `drive_real` with the normal online transport.
4. Add process-level graph reports, failures, and throughput/metric tests.

### Increment 4 — Dynamo offline

1. Forward every Dynamo feature through `aiperf-runner`.
2. Register `dynosim` only in feature-bearing builds.
3. Add strict engine/router/topology/trace/feature config owned by that factory.
4. Compose scheduled and graph workloads through the existing `SimClock`/steppable paths.
5. Restore the complete parity/fail-closed matrix as runner subprocess tests.

### Increment 5 — stateful agentic

1. Register the `agentic` workload and strict provider/gateway config.
2. Supervise canonical workers and callback gateway from the runner process.
3. Route primary/environment/verifier calls through the shared prepared online backend.
4. Restore Harbor, BrowserGym, and MCPMark real subprocess canaries.
5. Prove cancellation/infrastructure failures never become wrong answers.

### Increment 6 — deletion and v1 retirement

1. Remove Python resolver/implementation behavior now owned by Rust preparation.
2. Remove any alternate native or Python inference execution route.
3. Remove legacy CLI/canary wording once runner subprocess proofs replace it.
4. Remove protocol v1 after the compatibility matrix passes.
5. Retain library-level focused tests as algorithm tests, not product-entry tests.

---

## 13. Verification gates

### Protocol and composition

1. Capability, validate, and execute processes agree on exact `distribution_id`.
2. Unknown versions/operations/fields fail with exit `2` and one typed response where possible.
3. Backend/workload/endpoints in capabilities exactly equal the frozen registries.
4. Supported pairs are computed from descriptors/requirements and deterministically serialized.
5. A custom factory/extension appears in validation and execution without Python registration.

### Reachability matrix

1. `online_http + scheduled`: real HTTP/SSE subprocess, all phase families, adaptive controls,
   artifacts, telemetry, and native-v2.
2. `online_http + graph`: real Graph-IR transport subprocess and deterministic worker merge.
3. `online_http + static_accuracy`: real supervised static evaluator subprocess.
4. `online_http + agentic`: real Harbor, BrowserGym, and MCPMark provider subprocess canaries.
5. `dynosim + scheduled`: all applicable scheduled workloads, ramps, cancellation,
   adaptive controls, trace formats, topologies, routers, artifacts, and exact parity.
6. `dynosim + graph`: Graph-IR DES with engine/cancellation/sleeper events and exact parity.
7. Invalid accuracy/agentic plus offline combinations fail static validation.

### Side effects and failure

1. Static validation creates no artifacts, cache entries, workers, engines, or traffic.
2. Deferred validation completes before run artifact creation and scheduling.
3. Missing compiled features fail before backend initialization.
4. Digest mismatch fails before semantic validation.
5. Stderr and typed errors redact secrets and URL userinfo.
6. A failed run never emits a successful/partial authoritative report.

### Packaging

1. Fresh online installation finds and executes its packaged runner.
2. Fresh offline-capable installation advertises and executes Dynamo offline.
3. Online-only installation rejects offline without fallback.
4. Release containers execute Python -> runner -> online mock and offline in-process smoke tests.
5. Platform CI builds the native artifact and runs Cargo/process tests, not Python tests alone.

---

## 14. Rejected alternatives

### Keep a native `aiperf` CLI for missing modes

Rejected. That recreates a second schema, capability surface, and product entry point. A
library-only mode remains unavailable until projected through `aiperf-runner`.

### Let Python execute the missing mode temporarily

Rejected. Python may orchestrate or supervise canonical external libraries, but it does not become
an alternate inference scheduler, transport, metric engine, graph executor, or offline adapter.

### Treat `aiperf dynosim` as AIPerf offline reachability

Rejected. That facade exposes Dynamo-owned products and parsers. It does not execute AIPerf's
shared Rust front end or satisfy this runner contract.

### Define one executable or wire protocol per mode

Rejected. It guarantees configuration and reporting drift. Backend and workload factories compose
inside one versioned runner envelope.

### Pass raw argv into Rust mode parsers

Rejected for AIPerf runner operations. The subprocess boundary is a strict versioned DTO with
unknown-field rejection. Raw argv forwarding remains appropriate only for the separate canonical
Dynamo-owned facade where Dynamo owns the parser.

### Hardcode a central backend/workload enum and match

Rejected. Backend and workload are extension seams and use registered trait factories. The wire
uses stable IDs; each factory strictly owns its config.

### Reuse one long-lived runner across trials

Rejected. Fresh execution processes isolate allocator, connection pool, RNG, engine, evaluator,
and extension state. The Python outer loop owns iteration and convergence.

### Claim product support from library tests

Rejected. Product reachability requires a real Python-orchestrator -> `aiperf-runner` subprocess
proof for the exact backend/workload pair and report contract.

---

## 15. Completion criteria

This design is complete when:

- the only native AIPerf executable is `aiperf-runner`;
- protocol v2 accepts authored, side-effect-free `validate` and `execute` operations;
- exact runner distribution identity is verified across capability/validation/execution processes;
- backend/workload factories and capabilities are derived from one frozen runner registry;
- scheduled and Graph-IR workloads are product-reachable over online HTTP and Dynamo offline;
- static accuracy and stateful agentic workloads are product-reachable over online HTTP;
- every applicable mode uses the same endpoint registry, dataset store, clock/transport seams,
  observer/metrics engine, and native-v2 reporter;
- official feature-bearing distributions expose every supported optional backend fail-closed;
- runner subprocess tests replace removed native-CLI product proofs;
- Python contains no duplicate implementation of any capability now runner-reachable;
- protocol v1 and every fallback route have been removed after compatibility.

Until then, capabilities—not library presence or this design record—are the authority for what the
exact selected runner can execute.

## Addendum — 2026-07-12 (native KServe/gRPC protocol-v2 pairs)

The base runner now registers and subprocess-proves `online_http + scheduled`
and `online_grpc + scheduled` protocol-v2 execution. The latter is a new
real-clock backend over `aiperf-transport-grpc`; every multi-worker lane owns a
current-thread runtime, `LocalSet`, Clock, prepared endpoint table, dense gRPC
binding table, and Tonic channel set.

The KServe endpoint family is deliberately absent from protocol-v1 endpoint
compatibility. `kserve_v1_predict` names KServe's V1 HTTP dialect, not runner
protocol v1, and executes through the same authored v2 prepared-binding path.
Python accepts `grpc://` / `grpcs://` only with `online_grpc`, exact-image
capabilities advertise only the registered pair, and the v1 projector rejects
the backend. This addendum supersedes the earlier “pair execution pending” and
empty-`supported_pairs` implementation-status statements; unregistered pairs
and uncomposed sidecar/readiness lifecycles still fail closed.

## Addendum — 2026-07-12 (direct pair adapters and completed canonical matrix)

The canonical Python product path is now protocol-v2-only. It projects one
authored request, verifies the exact runner image, and preflights one registered
pair. It contains no Config-v1 resolver call, protocol-v1 request builder, or
fallback branch. Protocol v1 remains accepted by the Rust executable only as
an isolated compatibility decoder and is not selected by `aiperf profile`.

Runner composition implements the double-dispatch structure designed here:
open backend and workload factories validate their own raw configuration, an
explicit `RunnerPairFactory` owns cross-component compatibility, and
preparation returns one `PreparedRunnerOperation`. Online execution further
uses a direct `OnlineWorkloadAdapter -> PreparedOnlineHarness` transition. The
coordinator does not match on workload strings or convert v2 values through a
v1 DTO. Startup-only typed lowering into shared runtime values is the single
adapter load, not a second wire conversion.

The executable pair matrix is now:

| Runner distribution | Executable protocol-v2 pairs |
|---|---|
| Base | `online_http + scheduled`, `online_http + graph`, `online_http + static_accuracy`, `online_http + agentic`, `online_grpc + scheduled` |
| `dynosim` feature | every base pair plus `dynosim + scheduled` and `dynosim + graph` |

`supported_pairs` is still derived from registered executable adapters, so a
base image never claims the optional offline pairs. The scheduled offline
adapter loads the authored dataset once into the unified store before running
all phases on one simulator engine. The graph adapters pass authored
`dag_jsonl` directly to the registered graph-input adapter, producing
`GraphTracePlan`s and one frozen segment store without a `Dataset`,
`Conversation`, `DagMetadata`, Python resolver, or protocol-v1 intermediate.

The stateful agentic pair supervises the canonical Python evaluator over JSONL,
starts the authenticated Rust inference gateway, and routes primary and
auxiliary calls through the ordinary prepared endpoint, scheduling, transport,
metrics, and report path. Python Config-v2 subprocess coverage now complements
the existing runner process proof, so the mode is product-reachable without
restoring a native CLI or giving Python model-server coordinates.

This addendum supersedes Sections 1.1–1.3's implementation snapshot, the
migration plan's pending increments 3–5, and the completion criterion that
allowed a Python fallback until protocol-v1 retirement. Remaining unregistered
combinations and unsupported per-workload lifecycle fields continue to fail
closed; capability truth remains exact-image-specific.

## Addendum — 2026-07-12 (runner protocol-v1 fully removed)

Protocol-v1 support has been deleted from `aiperf-runner` entirely. The runner
now advertises `protocol_versions: [2]` only and rejects any non-v2 request as a
protocol-v2 failure envelope. Removed from `crates/aiperf-runner`: the v1
request `dispatch` entry, `execute_v1` and the `execute_run*` execution chain,
the `RunRequest` / `RunSpec` / `RunTerminal` / `EndpointSpec` / `DatasetSpec` /
`AccuracySpec` wire DTOs, the `load_protocol_v1` graph-input adapters, and the
`Legacy` capability/enum variants, along with the accompanying v1 tests. Python
had already dropped its v1 wire projection.

This supersedes Section 3.3 "Protocol-v1 compatibility," the "runner temporarily
accepts the current protocol-v1" language, the migration steps that kept v1
online/static-accuracy compatibility, and the two prior addenda sentences
describing an "isolated compatibility decoder": no v1 decoder, authority,
request builder, or fallback remains on the runner. (The `aiperf-endpoints`
crate keeps its own internal `EndpointType` metadata/compatibility adapters —
those are unrelated to the removed runner wire protocol.)

## Addendum — 2026-07-12 (transport vocabulary and telemetry-watch pair)

The runner execution axis has been renamed from backend to transport in the
strict protocol-v2 surface. Capabilities now emit `transports`, authored requests
carry `run.transport`, and registered IDs are `http`, `grpc`,
`dynosim_offline`, and `dynosim_online`. The older `online_http`,
`online_grpc`, and single `dynosim` backend names in this spec are superseded.

The base runner's executable protocol-v2 matrix now also includes the
operational history plane:

| Runner distribution | Executable protocol-v2 pairs |
|---|---|
| Base | `http + scheduled`, `http + graph`, `http + static_accuracy`, `http + agentic`, `http + telemetry_watch`, `grpc + scheduled` |
| `dynosim` feature | every base pair plus `dynosim_offline + scheduled`, `dynosim_offline + graph`, `dynosim_online + scheduled`, and `dynosim_online + graph` |

Exact deployment-owned evaluator roots may still conditionally add
`http + evaluation`. Capability truth remains exact-image-specific, and
`protocol_versions` remains `[2]`.

## Addendum — 2026-07-13 (BenchmarkRun wire + runner catalog)

The strict request envelope, `supported_pairs` / backend-workload capabilities
shape, and distribution-id pinning described in this spec are superseded by
`specs/2026-07-13-benchmarkrun-wire-and-runner-catalog-design.md` (decided / not
yet implemented).

Authoritative replacement: BenchmarkRun-shaped `run` (including `resolved`),
plugins.yaml-like JSON catalog discovery from the linked binary, no
`workload`/`backend` wire dialect, no `expected_distribution_id`. The
performance-only cut keeps scheduled/graph execution selected from Config
shape and `transport.type` (`http` / `grpc` / `dynosim_offline` /
`dynosim_online`). Agentic / static-accuracy / evaluation / telemetry-watch
pairs leave the product wire as Config sheds them. Sole-runner ownership and
fail-closed unknown combinations remain.
