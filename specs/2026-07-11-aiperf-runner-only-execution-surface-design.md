<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AIPerf: `aiperf --execute` as the sole native execution surface

**Date:** 2026-07-11
**Author:** Anthony Casagrande (Tech Lead) + Codex
**Status:** built — sole native executable, protocol-v2-only, canonical transport/workload
matrix composed from frozen registries. The strict request envelope and discovery contract
described historically here are **superseded by**
`2026-07-13-benchmarkrun-wire-and-runner-catalog-design.md`, which owns the current
BenchmarkRun-shaped wire and the plugins.yaml-shaped runner catalog.
**Decision:** every end-user AIPerf execution backed by the native Rust implementation enters
through the `aiperf` binary's versioned execution operation (`aiperf --execute`, the internal hidden
mode the entry point re-execs). There is no separate second Rust executable, Python inference
fallback, direct Python-to-Rust library binding, or mode-specific process surface.

**Companions:**

- `2026-07-13-benchmarkrun-wire-and-runner-catalog-design.md` owns the authoritative request wire
  (BenchmarkRun-shaped `run`) and capability discovery (plugins.yaml-shaped JSON catalog). It
  supersedes the historical envelope/`supported_pairs`/`distribution_id`-pin contract sketched in
  §3, §4, and §8 below.
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

This spec owns **product reachability**: the operation model, transport/workload selection,
capability discovery, feature-bearing runner distributions, common preparation/execution/report
contract, and subprocess proofs that make those library implementations usable through the only
native executable.

---

## 0. Thesis

The native product is one executable with multiple statically composed transport and workload
implementations:

```text
Python aiperf command
    |
    | structural Config v2, sweeps, trials, search, presentation
    v
BenchmarkRun-shaped run (protocol v2)
    |
    v
aiperf --execute   (the aiperf binary's execution engine)
    |
    +-- RunnerRegistry
    |     +-- endpoint factories
    |     +-- dataset/sampler factories
    |     +-- transport factories
    |     +-- workload factories  (own their prepare/lowering)
    |     `-- descriptor compatibility predicate  (no pair objects)
    |
    +-- validate
    |     `-- authored run -> validated plan (+ deferred checks)
    |
    `-- execute
          +-- http / grpc  + scheduled / graph
          `-- dynosim_offline / dynosim_online + scheduled / graph
```

ONLINE-real and ONLINE-mock are the same `http` transport. They differ only in configured target
URL; a mock HTTP server is not another runner mode. OFFLINE-mock is the feature-gated
`dynosim_offline` transport using `SimClock` and the in-process steppable engine;
`dynosim_online` is the wall-clock apples-to-apples variant.

Transport and workload are orthogonal selections. A new transport does not acquire its own CLI or
duplicate every workload DTO, and a new workload does not acquire its own transport. Factories
lower the versioned run into today's real seams—`Clock`, `RequestSink<R>`, `TurnDispatcher`,
`GraphSink`, phase/runtime traits, observers, and reporters—not an aspirational parallel harness.
Transport × workload compatibility is a **descriptor predicate**
(`validate_descriptor_compatibility`: `semantic_responses`, `required_transport_features`,
`clock_kinds`), not a reified pair object — the coordinator resolves each axis by id and composes
them inline, so it still never matches on transport or workload strings.

> **History reversal (2026-07-14).** This section originally specified an explicit `RunnerPairFactory`
> ("open double dispatch") that reified each transport×workload cell. That mechanism is **struck**: a
> per-cell object is the O(transport × workload) anti-pattern the orthogonal-axes design exists to
> avoid. The decided design keeps only transport and workload factories joined by the compatibility
> predicate; the workload factory owns `prepare`/`validate_run` and receives the validated transport.
> The "coordinator never string-matches" invariant is preserved (map lookups by id + the predicate).
> The prose below has been rewritten to the target; the `RunnerPairFactory` type still exists in code
> until it is deleted in Stage 1 of `2026-07-14-unified-execution-substrate-design.md` §2.3 (which owns
> the rationale and staged plan). Where §1 says "current code truth," read the pair removal as decided-
> but-pending that stage.

---

## 1. Current code truth

`aiperf --execute` (the `aiperf` binary's hidden execution mode; crate `aiperf-cli`, execution layer
`aiperf_runtime::engine` under `rust/runtime/src/engine/`) is the sole strict Rust
execution surface on the product path. It
speaks **protocol v2 only**: it reads one stdin request, composes the stock application, and
`run_v2` rejects any non-v2 or malformed request as a v2 failure envelope. `--capabilities` is the
only command-line operation and writes the plugins.yaml-shaped catalog (§8).

### 1.1 Product-reachable, built matrix

The frozen registry (`registry.rs::BuiltinRunnerRegistryFactory`) registers the `http` and `grpc`
transports and the `scheduled` and `graph` workloads; the executable matrix is the descriptor-compatible
cross-product (no pair objects). The base build's executable protocol-v2 combinations:

| Runner distribution | Executable protocol-v2 pairs |
|---|---|
| Base | `http + scheduled`, `http + graph`, `grpc + scheduled` |
| `dynosim` feature | every base pair plus `dynosim_offline + {scheduled, graph}` and `dynosim_online + {scheduled, graph}` |

`http` is a `RealClock` hyper transport (`aiperf_runtime::transport_http`) with h1/h2c/UDS/TLS/SSE,
connection reuse, and post-send cancellation. `grpc` is a `RealClock` Tonic transport
(`aiperf_runtime::transport_grpc`) where every multi-worker lane owns a current-thread runtime, `LocalSet`,
Clock, prepared endpoint table, and dense gRPC binding table. `dynosim_offline` uses `SimClock` and
the idle/quiescence DES pump; `dynosim_online` uses `RealClock` for apples-to-apples comparison with
Dynamo's live driver. All four feed the same observer/metrics/report path.

The `scheduled` workload owns concurrency, Poisson, Gamma, constant/request-rate, user-centric, and
fixed-schedule phases plus ramps, cancellation, adaptive control, stop bounds, slots, and samplers.
The `graph` workload composes `drive_real` (online) or `drive_sim_with_source` (offline) with
per-worker metric accumulators merged once in deterministic order. Both consume the runner-owned
endpoint registry and worker-local prepared bindings.

### 1.2 Performance-only product cut

The canonical Python product path is protocol-v2-only. It projects one authored BenchmarkRun,
verifies the exact runner image, preflights one registered pair against the linked catalog, and
executes it. It contains no Config-v1 resolver call, protocol-v1 request builder, or fallback
branch.

The current product wire is the **performance path**: scheduled and graph execution selected from
Config shape and `transport.type`. The stateful `agentic`, static-accuracy, provider-neutral
`evaluation`, and telemetry-watch pairs have left the product wire as Config sheds them; the
`http + static_accuracy` adapter remains in-tree behind a distribution gate
(`online_execution::register_http_static_accuracy_pair*`) but is not registered by the base image.
Any workload the linked registry does not compose fails closed.

### 1.3 Composition structure

Runner composition is two orthogonal registries joined by a predicate: transport and workload
factories validate their own raw configuration; `validate_descriptor_compatibility` gates the
combination; the **workload factory owns** `prepare`/`validate_run` (receiving the validated transport
and its id) and returns one `PreparedRunnerOperation`. Online execution uses a direct
`OnlineWorkloadAdapter -> PreparedOnlineHarness` transition. The coordinator does not match on
transport or workload strings or convert v2 values through a v1 DTO. (The `RunnerPairFactory`
pair-object mechanism is struck by design — deleted in Stage 1 per §0's note.) Startup-only typed lowering into shared
runtime values is the single adapter load, not a second wire conversion. `RunnerApplication` freezes
the linked registry at bootstrap; duplicate IDs are rejected and capabilities are derived from the
frozen factories, never a handwritten static array.

---

## 2. Required invariants

1. **One native executable.** Every native AIPerf run uses `aiperf-cli`.
2. **Fresh process per run.** Validation may use a separate short-lived process, but execution
   retains one fresh child per variation/trial.
3. **No Python inference fallback.** Missing runner capabilities fail before execution.
4. **Orthogonal transport/workload selection.** The run describes both; compatibility is validated
   by the descriptor predicate composing the two registries (no pair object).
5. **Trait-backed registries.** Transport and workload IDs select statically linked factories, not
   central string branches or a closed enum of implementation kinds.
6. **Exact-image handshake.** Capability and execution processes identify the exact selected runner
   image; Python preflights the linked catalog before executing.
7. **Authored run first.** Python projects structural/authored state without creating run
   artifacts, warming tokenizers, fetching datasets, or importing native-equivalent loaders.
8. **Rust preparation.** Rust resolves every input it already knows how to resolve, performs
   deferred semantic checks, and creates artifacts only after complete validation.
9. **One endpoint registry.** Every transport/workload pair receives prepared endpoints from the
   runner-owned registry; modes do not construct private built-in catalogs.
10. **One transport/clock timeline.** Workloads use the injected transport and clock seams. No mode
    measures or schedules outside them.
11. **One report family.** Every successful operation writes native-v2 with common run identity,
    metrics, transport/workload provenance, and mode-specific typed blocks.
12. **Fail-closed optional features.** A request for an uncompiled transport, workload, topology,
    router, offload, or report feature is rejected before side effects.
13. **Canonical external providers remain external.** Supervised Python evaluator/environment
    workers remain when they own canonical benchmark semantics; they never become an inference
    executor.
14. **No duplicate Python behavior.** When a Rust capability is runner-reachable, the Python
    implementation of that AIPerf capability is removed rather than retained in parallel.
15. **Dynamo product separation.** The Python `python -m dynamo.*` facades continue to expose
    Dynamo-owned products and raw canonical parsers. They are not an AIPerf-runner fallback and do
    not satisfy AIPerf offline reachability; Dynamo replay is authored through `aiperf profile` with
    `transport.type: dynosim_offline|dynosim_online`.

---

## 3. Protocol-v2 operation model

The runner advertises `protocol_versions: [2]` only. Protocol v1 has been **fully removed**: the v1
`dispatch` entry, `execute_v1`/`execute_run*` chain, the `RunRequest`/`RunSpec`/`RunTerminal`/
`EndpointSpec`/`DatasetSpec`/`AccuracySpec` wire DTOs, the `load_protocol_v1` graph-input adapters,
and the `Legacy` capability/enum variants are gone. No v1 decoder, authority, request builder, or
fallback remains on the runner. (The `aiperf_runtime::endpoints` module keeps its own internal
`EndpointType` metadata/compatibility adapters — unrelated to the removed runner wire protocol.)

### 3.1 Request envelope

Every stdin operation uses one strict tagged envelope (`protocol_v2.rs::RunnerEnvelopeV2`):

```rust
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunnerEnvelopeV2 {
    pub protocol_version: u32,          // must equal RUNNER_PROTOCOL_V2
    pub operation: RunnerOperationV2,   // Validate | Execute
    pub run: BenchmarkRunWireV2,        // BenchmarkRun-shaped run
}
```

The exact shape of `run` (BenchmarkRun-shaped, including its Python-resolved `resolved` bindings) is
defined authoritatively by `2026-07-13-benchmarkrun-wire-and-runner-catalog-design.md`. There is no
`expected_distribution_id` on the wire and no `workload`/`backend` wire dialect; the runner derives
scheduled-versus-graph from Config shape and binds the transport from `cfg.transport.type`. Unknown
protocol versions, operations, and fields fail closed.

`--capabilities` remains the only command-line operation and writes one catalog line. Every
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

Stderr is redacted diagnostic text only (`redaction.rs`). It is never parsed for normal machine
control.

---

## 4. Authored run model

Python projects one side-effect-free BenchmarkRun-shaped run and never creates run artifacts, warms
tokenizers, fetches datasets, or imports native-equivalent loaders. The exact fields are owned by
the BenchmarkRun wire spec; the runner-relevant invariants are:

- the artifact directory is selected but does not exist yet;
- tokenizer identity/revision/trust policy lives inside the authored Config and has not been
  cache-warmed;
- dataset path/public identity/filters/options are authored inputs, not Python loader results;
- endpoint profiles are raw and become worker-local prepared bindings in Rust;
- transport engine/router/topology inputs have not initialized an engine or socket.

Each selected statically linked factory strictly deserializes its own portion of the Config with
`deny_unknown_fields`; a linked implementation owns its configuration without extending a core enum
or accepting unknown keys. Python remains responsible for structural YAML/Jinja/environment/sweep
expansion and secrets substitution, and explicitly projects every accepted field into this narrower
ABI.

---

## 5. Transport registry and factory seam

### 5.1 Identity and descriptor

Transport factories (`registry.rs::RunnerTransportFactory`) expose a deterministic descriptor
(`id`, `description`, `clock` family, statically compiled `features`) and `validate` / `prepare`
methods. The exact internal prepared-transport traits remain split along today's typed
`RequestSink<HttpRequest>`, gRPC binding, `TurnDispatcher`, and `GraphSink` boundaries; this spec
does not require an aspirational universal `Transport`/`Harness` trait. It requires the runner
factory to lower into those current seams without string matching or a second transport/clock path.

The mutable builder freezes before validation or execution. Duplicate IDs are rejected. Transport
descriptors and feature flags are enumerated deterministically into the catalog.

### 5.2 `http` and `grpc`

`http` owns `RealClock`, `aiperf_runtime::transport_http` hyper client configuration, h1/h2c/UDS/TLS,
connection reuse, cancellation, SSE, trace timing, URL selection, and sticky routing. Real inference
endpoints and loopback mock endpoints use identical code; ONLINE-real versus ONLINE-mock is not
represented in the protocol — the configured URL determines the target. Python accepts `grpc://` /
`grpcs://` URLs only with the `grpc` transport.

### 5.3 `dynosim_offline` and `dynosim_online`

The feature-gated transports own `SteppableReplay` initialization and terminal operations,
engine/router JSON, aggregate/disaggregate topology, separate profiles, trace cutoff, routing,
cancellation, backend-owned capacity facts, and parity comparison between AIPerf and Dynamo common
summaries. `dynosim_offline` runs on `SimClock` and the idle/quiescence DES pump; `dynosim_online`
runs on `RealClock` for apples-to-apples comparison with Dynamo's live driver. Optional router
runtime, ZMQ events, KV offload, AIC forward pass, and profiling ride the same seam.

`aiperf-cli` forwards Cargo features explicitly:

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
static validation; requested offload initialization remains fail-closed. A default build that lacks
Dynamo cannot claim offline product support.

---

## 6. Workload registry and factory seam

Workload factories (`registry.rs::RunnerWorkloadFactory`) expose a descriptor (`id`, `description`,
supported clock kinds, `required_transport_features`), `validate`, `requirements`, and `prepare`.
The public factory seam is object-safe and startup-only. Hot loops retain their typed/generic
current implementations after preparation; the runner does not introduce a per-token dynamic
registry lookup or shared lock.

### 6.1 `scheduled`

The strict config generalizes the former runner v1 request. It owns the authored native dataset
source and tokenizer policy; ordered warmup/profiling phases; concurrency, request-rate,
user-centric, fixed-schedule, and one-pass sources; ramps, cancellation, adaptive control, stop
bounds, slots, and samplers; and endpoint profile materialization into per-turn prepared
`EndpointKey`s.

It supports `http`, `grpc`, `dynosim_offline`, and `dynosim_online`. The same phase/workload policy
is injected with the selected clock/dispatcher. Fixed schedule and dataset timing validation occur
after dataset load and before artifact creation or scheduling. The scheduled offline adapter loads
the authored dataset once into the unified store before running all phases on one simulator engine.

All canonical Dynamo trace formats enter through the unified dataset/fixed-schedule source or a
lowering into the same scheduled representation. They do not create a second public trace runner.

### 6.2 `graph`

The strict Graph-IR workload config owns the graph source/IR and unified dataset references; worker
count, duration gate, firing-gate inputs, and deterministic merge order; graph endpoint
materialization and response/metric configuration; and optional offline trace/event-source inputs
required by the Dynamo transport.

It supports `http` through `drive_real` and the dynosim transports through `drive_sim_with_source`.
Each graph worker owns its metric accumulator, endpoint binding table, and prepared sink; workers
merge once in deterministic order. The graph adapters pass authored `dag_jsonl` directly to the
registered graph-input adapter, producing `GraphTracePlan`s and one frozen segment store without a
`Dataset`, `Conversation`, `DagMetadata`, Python resolver, or protocol-v1 intermediate. Unsupported
phase lists, arrival/slot actuators, or other unbuilt Graph consumers fail validation rather than
appearing as inert config.

### 6.3 Sheddable canonical workloads

The `static_accuracy`, stateful `agentic`, provider-neutral `evaluation`, and telemetry-watch
workloads are **not** part of the current performance-only product wire. Their design intent is
preserved here and their canonical-provider contract is owned by the accuracy-accumulator companion
spec:

- **`static_accuracy`** requires semantic response text (`http` only) and drives inference through
  the ordinary scheduled path while a supervised pinned Python evaluator owns load/grade. Its
  in-tree adapter (`online_execution::register_http_static_accuracy_pair*`) remains gated and is not
  registered by the base image.
- **`agentic`** requires semantic responses (`http` only), supervises Harbor / AgentLab+BrowserGym /
  MCPMark canonical workers, and routes primary/environment/verifier calls through the authenticated
  Rust inference gateway and the ordinary prepared endpoint/scheduling/transport/metrics path. Its
  workers never contact the target model except through that callback.
- **`evaluation`** requires deployment-owned attested evaluator roots; without them the
  `http + evaluation` pair is absent.
- **telemetry-watch** (the former operational-history pair) has been removed from the runner
  entirely.

Re-adding any of these to the product wire registers its workload factory (admitted against transports
by the compat predicate) and restores its subprocess proof; until then, capability truth for a given
image excludes them.

---

## 7. Compatibility matrix

The built-in target matrix is:

| Transport | `scheduled` | `graph` |
|---|:---:|:---:|
| `http` | yes | yes |
| `grpc` | yes | no |
| `dynosim_offline` | yes | yes |
| `dynosim_online` | yes | yes |

The matrix is derived from workload requirements and transport descriptors at registry freeze; it is
not a handwritten runtime switch. The catalog serializes only the linked, executable transports and
workloads. A linked factory may add a new ID/pair without editing a core enum, subject to its trait
and Config schema. The `dynosim_*` transports appear only in a feature-bearing build; `grpc`
currently pairs with `scheduled` only.

---

## 8. Capability discovery

`--capabilities` describes the exact frozen runner distribution as a plugins.yaml-shaped JSON
catalog (`protocol.rs::RunnerCatalog`, emitted by `RunnerApplication::catalog()`):

```json
{
  "schema_version": "…",
  "endpoint": { "…": { "description": "…", "metadata": { } } },
  "transport": {
    "http":            { "description": "…", "metadata": { "transport_type": "http", "clock": "real", "features": ["h1", "h2c", "uds", "tls"], "url_schemes": ["http", "https"] } },
    "grpc":            { "description": "…", "metadata": { "transport_type": "grpc", "clock": "real" } },
    "dynosim_offline": { "description": "…", "metadata": { "transport_type": "dynosim_offline", "clock": "sim" } },
    "dynosim_online":  { "description": "…", "metadata": { "transport_type": "dynosim_online", "clock": "real" } }
  },
  "custom_dataset_loader": { },
  "public_dataset_loader": { },
  "dataset_sampler": { }
}
```

The authoritative catalog schema is owned by
`2026-07-13-benchmarkrun-wire-and-runner-catalog-design.md`. This is a plugins.yaml-shaped inventory
of the exact linked binary: endpoint dialects, transports, dataset loaders, and samplers — **not** a
`supported_pairs` / transport-workload matrix and **not** a distribution-id pin. A feature-gated
implementation absent from the exact binary is absent from the catalog. Python performs pair
preflight against this catalog before executing; an absent capability fails closed without
conversion to any legacy path.

---

## 9. Validation, preparation, and execution

### 9.1 Static validation

The `validate` operation:

1. resolves transport/workload/endpoint IDs through frozen registries;
2. strictly deserializes their owned configs;
3. validates transport/workload compatibility through the descriptor predicate and the compiled feature set;
4. validates every rule possible without external dataset/evaluator/server IO;
5. returns `completeness` plus typed deferred checks.

It creates no artifact directory, warms no tokenizer, downloads no dataset, initializes no Dynamo
engine, and sends no inference traffic.

### 9.2 Complete preparation

The `execute` operation repeats static validation and then:

1. loads/localizes the dataset and tokenizer through Rust registries;
2. discovers and binds every endpoint profile reference;
3. validates fixed timing, graph inputs, and other deferred content;
4. prepares a worker-local endpoint table and compiled templates/selectors;
5. prepares the transport clock/sink/engine without beginning workload events;
6. completes transport/workload compatibility validation using prepared facts;
7. creates the run artifact directory and materializes authorized user files;
8. starts supervised sidecars and the runtime;
9. executes and finalizes reports/artifacts.

If preparation requires remote cache writes, those use a content-addressed shared cache and are not
run artifacts. A failed preparation never leaves a partially authoritative native-v2 report.

### 9.3 Execution

Every workload emits observations through the normal local-loop observer graph. Online uses
`RealClock`; offline uses `SimClock`; no feature reads wall time or `tokio::time` directly. Endpoint
bindings are worker-local, metrics are worker-local and merged once, and no optional sidecar can
backpressure request dispatch.

All terminal paths drain/cancel according to the owning workload/phase policy, finalize typed
transport/workload facts, and then serialize native-v2. Failure terminals identify `protocol`,
`validation`, `preparation`, `execution`, or `reporting` stage without leaking secrets.

---

## 10. Native-v2 and artifacts

Every report contains common provenance (run identity, transport, workload, endpoint profiles).
Mode-specific typed blocks remain additive:

- online HTTP trace/network/TLS facts;
- Dynamo engine/router/topology/capacity/parity facts;
- Graph worker/IR/firing facts.

The common metric catalog and report identity do not fork by mode. Offline returns are rejected
unless the complete common AIPerf/Dynamo summary bytes match as required by the offline spec.
Numeric values crossing JSON remain finite or absent.

Rust writes native run artifacts. Python reads the terminal/report and performs remaining
presentation, cross-run aggregation, upload, and plotting work. Python compatibility artifacts are
derived from Rust-owned results and are not another metric authority.

---

## 11. Packaging and distribution selection

There is exactly **one** `aiperf` wheel. maturin compiles the `pyext` pyo3 module into
`aiperf._native` and packages the `src/aiperf` frontend; `tools/wheel_repack.py` (run by
`make wheel`) then repacks the native `aiperf` binary directly into the wheel's scripts directory
(`aiperf-<ver>.data/scripts/aiperf`). Because the wheel carries a native binary it is platform +
CPython-ABI specific. There is no separate platform-specific companion package.

The execution child is the **same `aiperf` binary re-execing itself** as `aiperf --execute`; no
external binary is discovered and there is no discovery-order search. The only override is
`AIPERF_EXEC_BIN`, which points the execution child at a differently-compiled build (for example a
`dynosim`/custom-features binary). Official builds may differ in compiled optional transports, but
every build is self-describing through its in-process capabilities catalog.

The release matrix MUST contain:

- a stock online `aiperf` wheel for every supported platform;
- an official offline-capable build (`dynosim`/`full` features) wherever the pinned Dynamo
  dependency is supported, selectable at run time via `AIPERF_EXEC_BIN`;
- source/lock/feature provenance for each;
- fresh-install catalog and loopback subprocess tests;
- no silent substitution of an online-only binary for an offline request.

Custom statically linked extensions ship a custom `aiperf` build and are selected explicitly via
`AIPERF_EXEC_BIN`. Protocol/catalog/report compatibility — not Python package-version equality —
is authoritative.

---

## 12. Verification gates

### Protocol and composition

1. Capability and execute processes agree on the exact linked runner image.
2. Unknown versions/operations/fields fail with exit `2` and one typed response where possible.
3. Transports/workloads/endpoints in the catalog exactly equal the frozen registries.
4. Supported pairs are computed from descriptors/requirements and deterministically serialized.
5. A custom factory/extension appears in validation and execution without Python registration.

### Reachability matrix

1. `http + scheduled`: real HTTP/SSE subprocess, all phase families, adaptive controls, artifacts,
   telemetry, and native-v2.
2. `http + graph`: real Graph-IR transport subprocess and deterministic worker merge.
3. `grpc + scheduled`: real Tonic subprocess with per-lane runtime/LocalSet and native-v2.
4. `dynosim_offline + {scheduled, graph}`: all applicable scheduled workloads and Graph-IR DES with
   ramps, cancellation, adaptive controls, trace formats, topologies, routers, artifacts, and exact
   parity.
5. `dynosim_online + {scheduled, graph}`: wall-clock apples-to-apples parity with Dynamo's live
   driver.
6. Unregistered transport/workload combinations fail static validation.

### Side effects and failure

1. Static validation creates no artifacts, cache entries, workers, engines, or traffic.
2. Deferred validation completes before run artifact creation and scheduling.
3. Missing compiled features fail before transport initialization.
4. Stderr and typed errors redact secrets and URL userinfo.
5. A failed run never emits a successful/partial authoritative report.

### Packaging

1. Fresh online installation finds and executes its packaged runner.
2. Fresh offline-capable installation advertises and executes Dynamo offline.
3. Online-only installation rejects offline without fallback.
4. Release containers execute Python -> runner -> online mock and offline in-process smoke tests.
5. Platform CI builds the native artifact and runs Cargo/process tests, not Python tests alone.

---

## 13. Rejected alternatives

### Keep a native `aiperf` CLI for missing modes

Rejected. That recreates a second schema, capability surface, and product entry point. A
library-only mode remains unavailable until projected through `aiperf-cli`.

### Let Python execute the missing mode temporarily

Rejected. Python may orchestrate or supervise canonical external libraries, but it does not become
an alternate inference scheduler, transport, metric engine, graph executor, or offline adapter.

### Treat a `python -m dynamo.*` facade as AIPerf offline reachability

Rejected. Those facades expose Dynamo-owned products and parsers. They do not execute AIPerf's
shared Rust front end or satisfy this runner contract. AIPerf offline is authored through
`aiperf profile` with `transport.type: dynosim_offline|dynosim_online`.

### Define one executable or wire protocol per mode

Rejected. It guarantees configuration and reporting drift. Transport and workload factories compose
inside one versioned BenchmarkRun run.

### Pass raw argv into Rust mode parsers

Rejected for AIPerf runner operations. The subprocess boundary is a strict versioned DTO with
unknown-field rejection. Raw argv forwarding remains appropriate only for the separate canonical
Dynamo-owned facades where Dynamo owns the parser.

### Hardcode a central transport/workload enum and match

Rejected. Transport and workload are extension seams and use registered trait factories. The wire
uses stable IDs; each factory strictly owns its config.

### Reuse one long-lived runner across trials

Rejected. Fresh execution processes isolate allocator, connection pool, RNG, engine, and extension
state. The Python outer loop owns iteration and convergence.

### Claim product support from library tests

Rejected. Product reachability requires a real Python-orchestrator -> `aiperf-cli` subprocess
proof for the exact transport/workload pair and report contract.

---

## 14. Completion criteria

This design is complete when:

- the only native AIPerf executable is `aiperf-cli`;
- protocol v2 accepts authored, side-effect-free `validate` and `execute` operations, and protocol
  v1 has been removed;
- the exact linked runner image is verified across capability and execution processes;
- transport/workload factories and the catalog are derived from one frozen runner registry;
- scheduled and Graph-IR workloads are product-reachable over `http`/`grpc` and Dynamo offline;
- every applicable mode uses the same endpoint registry, dataset store, clock/transport seams,
  observer/metrics engine, and native-v2 reporter;
- official feature-bearing distributions expose every supported optional transport fail-closed;
- runner subprocess tests replace any removed native-CLI product proofs;
- Python contains no duplicate implementation of any capability now runner-reachable.

The exact runner catalog — not library presence or this design record — is the authority for what
the selected runner can execute.

---

## Addendum — 2026-07-14

Superseded by `2026-07-14-unified-execution-substrate-design.md` (Stage 1, built): the
**`RunnerPairFactory` mechanism** this spec referenced — and the
**`validate_descriptor_compatibility` predicate together with the `supported_pairs`
inventory** it derived — are **removed from the tree**. This addendum revises the
composition *mechanism* only; the completion criteria in §14 and the product-reachability
matrix (scheduled/graph over http/grpc, dynosim behind the feature) are unchanged.

What changed, and why:

- **No pair object, no compatibility predicate, no pair inventory.** There is no
  `RunnerPairFactory`, no `pairs: BTreeMap<(transport_id, workload_id), …>` map, no
  `register_pair`, no `validate_descriptor_compatibility`, and no `supported_pairs`
  catalog field. `git grep` for any of these in `rust/runtime/src/engine` returns nothing.
- **Two independent registries, no gate.** The runner now exposes a transport registry
  and a workload registry as **orthogonal axes with no admission gate between them**:
  every workload runs over every transport. `prepare` / `validate_run` moved onto the
  workload factory (`RunnerWorkloadFactory`), which resolves the transport's
  dispatcher/placement from `RunnerExecutionFactories` keyed by `transport_id` and is
  otherwise transport-blind. Selection is map lookups by id — never a `match` on
  transport/workload strings — so this spec's Invariant (no runtime string switch) is
  **preserved**; only the reified-cell mechanism is gone.
- **Genuinely transport-specific limits surface at point-of-use, not as admission.**
  The earlier design admitted the transport × workload cross-product up-front via the
  descriptor predicate. With the flatten, there is no up-front cross-product admission at
  all; any constraint a transport genuinely cannot satisfy (e.g. a token-native gRPC body
  for an endpoint that does not stream) surfaces where it is exercised, not as a
  registry-time compatibility rejection.
- **`grpc + graph` falls out for free.** The visible symptom this unlocks — a `dag_jsonl`
  graph dataset dispatching over Tonic — needs no hand-added cell; it is proven by
  `rust/runtime/tests/test_graph_grpc.rs`.

The body above is retained as the historical record of the pair-factory design; where it
and this addendum conflict, the addendum is authoritative.
