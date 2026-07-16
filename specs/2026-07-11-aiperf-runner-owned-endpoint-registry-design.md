<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AIPerf: runner-owned endpoint registry, validation, and execution

**Date:** 2026-07-11
**Author:** Anthony Casagrande (Tech Lead) + Codex
**Status:** foundation built (open registry, KServe factories, prepared online/gRPC execution,
protocol-v2-only runner); remaining workload/lifecycle adapter convergence is open.
**Decision:** the exact selected `aiperf-cli` binary is the sole authority for endpoint
identity, metadata, semantic validation, normalization, request/response behavior, and execution.
Python remains a structural Config-v2 and orchestration front end while that front end exists; it
does not retain an endpoint implementation, endpoint manifest, endpoint enum, metadata fallback,
or endpoint-specific validation engine.

Endpoint identity, preparation, validation, catalog publication, and endpoint use by runner
operations are owned here. Runner-wide reachability for scheduled online, Graph-IR, Dynamo offline,
static accuracy, and stateful agentic execution is deliberately **not** designed in this record; it
belongs to the normative companion `2026-07-11-aiperf-runner-only-execution-surface-design.md`.
Product deletion is complete only when that companion makes every built Rust mode runner-addressable
with the same composed endpoint registry.

The `aiperf-endpoints`, `aiperf-dataset`, and `aiperf-extensions` crates named throughout this record
are now modules of the `aiperf-runtime` runtime library (`aiperf_runtime::endpoints`,
`aiperf_runtime::dataset`, `aiperf_runtime::extensions`); the `aiperf` product binary is the
`aiperf-cli` crate (`rust/cli`), and the v2 execution layer lives in `aiperf_runtime::runner_protocol`.
Code truth lives in `rust/runtime/src/endpoints*`, `rust/runtime/src/extensions*`, and
`rust/runtime/src/runner_protocol/*`.

**Companions:**

- `2026-07-11-aiperf-rust-endpoints-design.md` remains authoritative for the faithful request
  formatting, response parsing, replay, and input-accounting behavior of each dialect.
- `2026-07-11-aiperf-rust-compile-time-extension-registry-design.md` remains authoritative for
  explicit, transactional, statically linked extension composition.
- `2026-07-11-python-orchestrator-rust-single-run-design.md` remains authoritative for the process
  boundary: Python plans runs and a fresh `aiperf-cli` process executes one run.

This spec supersedes only the older records' endpoint **identity, registry ownership, capability
publication, and configuration-validation ownership**. It does not relax any endpoint parser,
transport, timing, metric, or compile-time-extension behavior.

---

## 0. Executive decision

There will be one endpoint system, not Python and Rust endpoint systems kept in parity:

```text
CLI / YAML
    |
    | Python: structural types, interpolation, sweeps, trials, paths
    v
explicit versioned RunRequest
    |
    v
the selected aiperf binary (aiperf --execute)
    |
    +-- one frozen Rust registry
    |     +-- built-in endpoints
    |     `-- explicitly linked Cargo extensions
    |
    +-- capabilities  --> registry-derived catalog
    +-- validate      --> Rust semantic validation and normalization
    `-- run           --> execution of the same prepared state
```

Python may deserialize the runner catalog for protocol negotiation, availability diagnostics,
help, and presentation. Python MUST NOT interpret that catalog to reproduce Rust's endpoint
normalization or validation rules. Authoritative validation is an operation of the runner and is
backed by the same Rust preparation path used before execution.

The permanent endpoint-creation workflow is therefore Rust-only:

1. implement the `Endpoint` trait;
2. declare the endpoint descriptor beside the implementation;
3. register it in the built-in catalog or one statically linked `AiperfExtension`;
4. add Rust unit and runner end-to-end tests.

No Python class, `plugins.yaml` entry, generated enum, Pydantic validator, or Python transport
change is permitted as part of adding a native endpoint.

---

## 1. Current code truth and the failure it demonstrates

### 1.1 Python rejects endpoints before consulting the runner

Today Python constructs `EndpointType` from Python plugin registrations at import time:

- `src/aiperf/plugin/enums.py:68-70` creates `EndpointType` from the endpoint plugin category;
- `src/aiperf/config/endpoint.py:139-149` types Config-v2 `endpoint.type` as that enum;
- `src/aiperf/config/flags/cli_config.py:153-164` types `--endpoint-type` as the same enum;
- `src/aiperf/cli_commands/profile.py:53-64` validates Config v2 and builds the benchmark plan;
- only later, `src/aiperf/orchestrator/rust_executor.py:47-50` discovers the runner and loads its
  capabilities.

This ordering is already observably wrong. The Rust runner advertises `messages` in
`rust/runtime/src/runner_protocol/protocol.rs:49-68`, while the Python endpoint block in
`src/aiperf/plugin/plugins.yaml:184-421` and its checked-in Config-v2 schema do not contain
`messages`. A valid endpoint compiled into the selected runner is consequently rejected before
the selected runner can see it.

The Python manifest cannot be repurposed as a clean metadata-only catalog: its plugin schema and
registry require a Python class path, and entries without one are not registered. Adding a fake
class would preserve the very dual system this design removes.

### 1.2 Python currently owns endpoint semantics that Rust also implements

Endpoint metadata currently drives Python behavior in at least these paths:

| Python path | Endpoint-dependent behavior | Final owner |
|---|---|---|
| `config/endpoint.py:359-407` | disable unsupported streaming | Rust endpoint preparation |
| `config/endpoint.py:500-527` | derive/reject multipart encoding | Rust endpoint preparation |
| `config/flags/_converter_dataset.py:618-655` | text/ranking defaults and CLI compatibility | Rust dataset + endpoint preparation |
| `common/tokenizer_validator.py:226-320` | decide whether to resolve/load a tokenizer | Rust dataset + endpoint preparation |
| `config/resolution/resolvers.py:127-163` | endpoint-derived artifact naming | opaque endpoint ID or Rust report provenance |
| `common/readiness_probe.py:45-63,189-218` | hardcoded endpoint paths and payloads | Rust endpoint formatter/header/transport path |

The corresponding streaming, form-data, template, URL, and endpoint-metadata validation already
exists in `rust/runtime/src/endpoints/config.rs:100-182`. Feeding Rust metadata into the existing
Python validators would remove one data-copy but leave two executable rule engines. That is not
the target architecture.

### 1.3 Rust itself has multiple endpoint authorities

The native implementation also needs consolidation:

1. `rust/runtime/src/endpoints/metadata.rs:8-48` defines a closed `EndpointType` enum.
2. The same file stores a separate metadata table selected by that enum.
3. `rust/runtime/src/dataset/request.rs:97-226` manually registers endpoint names and keeps a
   second enum-indexed map.
4. `rust/runtime/src/runner_protocol/protocol.rs:49-68` hardcodes the capability list independently.
5. `rust/runtime/src/runner_protocol/execute.rs:351` constructs `AiperfRegistry::builtin()` inside
   execution rather than accepting the runner's composed registry.

As a result, the current compile-time endpoint extension proof is not a proof of a new dialect. It
registers `ChatEndpoint` under another name, but the identity remains `EndpointType::Chat`. A name
whose dialect is not already in the closed enum is rejected during Serde deserialization before
registry lookup, and the production runner never applies endpoint extensions to the registry it
uses for capabilities or execution.

### 1.4 Per-turn selection can produce an invalid adapter/config pair

`rust/runtime/src/multiturn.rs:1017-1038` resolves an authored per-turn endpoint, clones the
default configuration, overwrites its enum identity, and clamps streaming. It does not fully
revalidate that configuration for the selected dialect. A form-data mismatch can therefore
survive until the dispatch invariant in `rust/runtime/src/http/endpoint_dispatch.rs:297-327`.

The final model MUST make the selected adapter and its validated effective configuration
inseparable and MUST bind every distinct endpoint/profile before request scheduling.

---

## 2. Required invariants

1. **Exact-binary authority.** Capabilities, validation, and execution use one registry composed
   by the exact selected `aiperf-cli` binary.
2. **Open endpoint identity.** The process boundary accepts a validated string identifier; a core
   enum cannot bound statically linked extensions.
3. **One registration record.** An endpoint's canonical ID, aliases, descriptor, and implementation
   enter the registry together. Capability publication is derived from that record.
4. **One validation implementation.** Runner validation and normal execution call the same Rust
   semantic preparation code.
5. **Validated binding.** Runtime code receives an endpoint adapter and an effective configuration
   that were validated together; it never mutates an endpoint kind enum.
6. **No request-path discovery.** Names and aliases are resolved before scheduling. Turns carry a
   dense binding/profile key, not a string lookup or fresh adapter construction.
7. **No global mutable registry.** The builder is mutable only during process composition; the
   registry is frozen before runtimes and workers start.
8. **No dynamic Rust ABI.** Extensions remain ordinary statically linked Cargo dependencies applied
   explicitly and transactionally.
9. **No Python fallback.** A missing, incompatible, or older runner is an actionable error. Python
   does not fall back to `plugins.yaml` or a checked-in endpoint table.
10. **Delete migrated Python behavior.** Once Rust owns a behavior, the corresponding Python
    implementation and its tests are removed rather than retained for parity.
11. **Preserve all execution modes.** Endpoint formatting, validation, and lifecycle selection stay
    above the shared `{transport, clock}` seams and therefore apply to online-real, online-mock,
    and feature-gated offline-mock paths.
12. **Preserve endpoint behavior.** This registry redesign MUST NOT change the source-grounded
    formatting/parsing scars pinned by the endpoint behavior spec and fixtures.

---

## 3. Rust endpoint identity and descriptor

### 3.1 `EndpointId` replaces `EndpointType`

```rust
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct EndpointId(Box<str>);
```

`EndpointId` has a hand-written constructor and `Deserialize` implementation. It validates stable
identifier syntax, not registry membership. The initial canonical grammar is:

```text
[a-z][a-z0-9_]*
```

Canonical IDs are never silently case-folded or punctuation-rewritten. Compatibility spellings
are explicit aliases, so reports and capability digests have one stable identity. Existing wire
names remain unchanged. `chat_completions` becomes an explicit alias of canonical `chat`.

An unknown but syntactically valid ID MUST deserialize successfully and reach registry resolution,
where the runner can return an error containing the exact compiled catalog. Serde MUST NOT emit a
closed-enum "unknown variant" error.

### 3.2 The descriptor lives beside the adapter

```rust
#[derive(Debug, Serialize)]
pub struct EndpointDescriptor {
    pub id: &'static str,
    pub aliases: &'static [&'static str],
    pub description: &'static str,
    pub endpoint_path: Option<&'static str>,
    pub streaming_path: Option<&'static str>,
    pub supports_streaming: bool,
    pub produces_tokens: bool,
    pub tokenizes_input: bool,
    pub requires_form_data: bool,
    pub requires_polling: bool,
    pub requires_inline_media: bool,
    pub input_modalities: &'static [Modality],
    pub output_modalities: &'static [Modality],
    pub metrics_title: &'static str,
    pub service_kind: &'static str,
}
```

The actual implementation may retain individual modality booleans internally if that avoids an
unrelated report migration, but the public capability DTO presents stable lists. The descriptor
contains declarative facts only. Conditional or vendor-specific validation remains executable
trait behavior, not a metadata mini-language.

Each endpoint implementation returns its own descriptor:

```rust
pub trait Endpoint: std::fmt::Debug + Send + Sync {
    fn descriptor(&self) -> &'static EndpointDescriptor;

    fn validate_config(
        &self,
        config: &mut EffectiveEndpointConfig,
    ) -> EndpointResult<()> {
        Ok(())
    }

    fn format_payload(&self, request: &RequestInfo) -> EndpointResult<Value>;
    fn format_headers(&self, config: &EffectiveEndpointConfig)
        -> BTreeMap<String, String>;
    fn parse_response(&self, response: &ServerResponse)
        -> EndpointResult<Option<ParsedResponse>>;

    // Existing extraction, replay, and assistant-turn hooks remain.
}
```

Common descriptor-driven validation runs before `Endpoint::validate_config`. Endpoint-specific
rules belong to the relevant implementation: for example, `TemplateEndpoint` owns the requirement
for a template body, and chat compatibility parsing belongs to `ChatEndpoint`. Central code MUST
NOT match a closed endpoint enum or branch on endpoint ID strings.

### 3.3 Remove identity from endpoint configuration

Endpoint identity is selection, not configuration. Replace the current configuration field
`endpoint_type: EndpointType` with separate authored and effective types:

```rust
pub struct RawEndpointConfig {
    pub urls: Vec<String>,
    pub path: Option<String>,
    pub streaming: bool,
    pub request_content_type: Option<RequestContentType>,
    pub template: Option<String>,
    pub response_field: Option<String>,
    pub timeout_seconds: f64,
    pub headers: BTreeMap<String, String>,
    pub api_key: Option<String>,
    pub extra: Option<Map<String, Value>>,
    // Remaining transport/body policy fields.
}

pub struct EffectiveEndpointConfig {
    // Private validated/normalized representation of the same policy.
}
```

Only the registry/binding preparation path may create `EffectiveEndpointConfig`.

### 3.4 Adapter/config binding: worker-local, allocation-free on dispatch

Binding is expressed through two explicit seams — an object-safe factory registered at composition
time and a worker-local prepared value — plus a dense copyable key:

```rust
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct EndpointKey(u32);

pub trait EndpointFactory: std::fmt::Debug + Send + Sync {
    fn descriptor(&self) -> &'static EndpointDescriptor;

    fn prepare(
        &self,
        config: EffectiveEndpointConfig,
    ) -> EndpointResult<Box<dyn PreparedEndpoint>>;
}

pub trait PreparedEndpoint: std::fmt::Debug {
    fn descriptor(&self) -> &'static EndpointDescriptor;
    fn config(&self) -> &EffectiveEndpointConfig;

    // Existing format/header/parse/extraction/replay behavior is exposed here
    // or through object-safe leaf traits owned by the endpoint behavior spec.
}
```

The registry stores immutable `Arc<dyn EndpointFactory>` values only during startup/composition.
Before worker execution, each worker calls the selected factories and owns a dense
`Vec<Box<dyn PreparedEndpoint>>` (`PreparedEndpointTable`) in worker-local state. A turn carries only
a copyable `EndpointKey`. Dispatch performs one checked indexed borrow and no `Arc` clone, mutex
operation, string lookup, endpoint construction, or endpoint-config mutation; `PreparedRequest`
contains no endpoint configuration.

This preparation seam is not deferred to a hypothetical future dialect — it is the shipped model.
Raw/template preparation resolves template source safely, constructs the Minijinja
environment/template, and compiles the JMESPath response selector once per effective worker/profile
binding. Stateless endpoints use a blanket factory whose preparation only stores their validated
configuration. `EffectiveEndpointConfig` and prepared-binding fields are private outside their
owning module; only validation/preparation constructors can create them, so callers cannot
construct an adapter and mismatched configuration with public struct literals.

---

## 4. Endpoint registry ownership and composition

### 4.1 Move the registry into `aiperf-endpoints`

`BuiltinEndpointResolver` currently lives in `aiperf-dataset` and imports every concrete adapter.
That reverses ownership: datasets consume endpoint resolution but do not own the endpoint catalog.

Move the lookup trait and concrete builder/frozen registry into `aiperf-endpoints`:

```rust
pub trait EndpointResolver: Send + Sync {
    fn resolve(&self, id: &EndpointId) -> EndpointResult<Arc<dyn Endpoint>>;
}

pub struct EndpointRegistryBuilder { /* startup-only maps */ }
pub struct EndpointRegistry { /* frozen deterministic catalog */ }
```

The builder registers one implementation without repeating its name:

```rust
let mut endpoints = EndpointRegistryBuilder::new();
endpoints.register(ChatEndpoint)?;
endpoints.register(ResponsesEndpoint)?;
endpoints.register(MessagesEndpoint)?;
```

Registration validates the descriptor ID and aliases atomically. Duplicate canonical IDs, aliases,
and collisions between a canonical ID and another entry's alias are errors. The frozen registry
stores canonical entries in deterministic lexical order for capabilities and digest generation.

The registry has no default endpoint. The authored run/profile selects its default explicitly.

### 4.2 `AiperfRegistry` remains the aggregate composition seam

`aiperf-extensions` continues to own the aggregate registry and transactional extension
application. Its endpoint category now wraps `EndpointRegistryBuilder` rather than the dataset-owned
resolver. An extension remains ordinary Rust:

```rust
impl AiperfExtension for AcmeExtension {
    fn name(&self) -> &str { "acme-endpoints" }

    fn register(&self, registry: &mut AiperfRegistryBuilder)
        -> Result<(), ExtensionError>
    {
        registry.endpoints_mut().register(AcmeEndpoint)?;
        Ok(())
    }
}
```

The extension test MUST implement a genuinely new endpoint descriptor and behavior. Registering a
new alias for `ChatEndpoint` is not sufficient proof.

### 4.3 The runner is the production composition root

```rust
fn build_linked_registry() -> Result<AiperfRegistry> {
    let mut registry = AiperfRegistryBuilder::builtin()?;

    #[cfg(feature = "acme-endpoint")]
    registry.register_extension(&acme_endpoint::AcmeExtension)?;

    registry.freeze()
}
```

The extension list is explicit, feature-gated, ordered, searchable, and fixed at compile time. No
environment variable, manifest, Python package, linker inventory, or shared library can inject
code into an already-built runner.

`aiperf-cli` builds this registry once at process startup. `--capabilities`, validation, and
normal execution all borrow the same frozen value. `execute_run` is changed to accept an injected
registry or becomes a method on a runner application object. It MUST NOT call
`AiperfRegistry::builtin()` internally.

---

## 5. One preparation and validation path

### 5.1 Runner application object and authored-request order

```rust
pub struct Runner {
    registry: AiperfRegistry,
}

impl Runner {
    pub fn capabilities(&self) -> RunnerCapabilities;
    pub fn validate(&self, run: AuthoredRunSpecV2) -> RunValidation;
    pub fn execute(&self, run: AuthoredRunSpecV2) -> Result<RunTerminal>;
}
```

Both `validate` and `execute` invoke the same pure semantic preparation function:

```rust
fn validate_run(
    run: &AuthoredRunSpecV2,
    registry: &AiperfRegistry,
) -> Result<ValidatedRun>;
```

The permanent request order places all Python side effects after the runner accepts static
validation:

```text
Python structural Config-v2 validation and outer-loop expansion
    |
    v
side-effect-free AuthoredRunSpecV2 projection
    |
    v
aiperf static validation
    |
    v
Rust dataset/tokenizer/backend preparation and deferred validation
    |
    v
artifact creation, supervised workers, scheduling, and dispatch
```

`AuthoredRunSpecV2` is built only from authored/structurally resolved configuration. It MUST NOT
read a fully resolved `BenchmarkRun.resolved`, require a pre-created artifact directory, require a
cache-localized tokenizer, or require Python to import a dataset loader. It carries the selected
artifact target path without creating it, authored tokenizer identity/options rather than a
Python-warmed cache result, authored local/inline/public dataset identity and options rather than
Python loader metadata, the endpoint/profile IDs and raw endpoint policy, and the backend/workload
request owned by the runner-only execution-surface companion. No Python resolver may perform a side
effect before the selected runner accepts static validation.

Execution extends `ValidatedRun` into IO-bound `PreparedRun` state by loading the selected dataset,
tokenizer, and optional sidecars. No endpoint semantic rule is reimplemented in the IO phase. For
local or inline datasets, static validation also scans authored endpoint/profile references.
References only discoverable through a remote dataset are bound immediately after load and before
scheduling or HTTP dispatch. Dataset cache population needed to read remote input is preparation
state, not a run artifact.

Pure static validation MUST occur before:

- artifact directory creation;
- tokenizer download/cache warming;
- public dataset network access;
- readiness probes;
- telemetry or Python worker startup;
- Tokio scheduling or inference traffic.

### 5.2 Binding algorithm

For every endpoint profile, the registry performs:

```text
resolve authored ID or alias
    |
    v
select canonical adapter + descriptor
    |
    v
generic structural/config validation
    |
    v
descriptor-driven normalization
    |
    v
Endpoint::validate_config
    |
    v
EndpointBinding { canonical ID, adapter, effective config }
```

Generic validation includes URL syntax, timeout ranges, path syntax, wait-for-model coherence, and
request-content-type compatibility. Descriptor-driven normalization preserves established user
semantics:

- omitted multipart encoding is derived when the endpoint requires form data;
- an explicit conflicting content type is rejected;
- unsupported streaming is normalized off with a structured notice during the compatibility
  window;
- phase validation then sees the effective streaming value, so prefill concurrency is rejected if
  streaming became unavailable;
- template inference/requirements are resolved before execution without mutating an endpoint enum.

The normalized effective configuration and notices are recorded in native report provenance.

### 5.3 Structured validation result and completeness

Validation is a strict JSON operation of `aiperf-cli`, using the same one-line stdout discipline
as capabilities and terminal results:

```json
{
  "event": "run_validation",
  "protocol_version": 2,
  "success": false,
  "completeness": "static",
  "deferred_checks": [
    {
      "code": "dataset_endpoint_profiles",
      "path": "run.dataset",
      "reason": "endpoint profile references require loading the remote dataset"
    }
  ],
  "errors": [
    {
      "code": "unknown_endpoint",
      "path": "run.endpoint.profiles.primary.dialect",
      "message": "endpoint \"acme_chat\" is not compiled into this runner",
      "available": ["chat", "messages", "responses"],
      "hint": "select a runner containing the required compile-time extension"
    }
  ],
  "notices": []
}
```

Diagnostics use stable codes and field paths; human text may improve without becoming a protocol
key. Secrets, URL userinfo, API keys, headers, and payload bodies are redacted. Validation failure
never creates artifacts or sends traffic.

Static validation cannot inspect endpoint/profile references stored only inside rows of a remote
dataset, so every result reports its scope:

```rust
#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ValidationCompleteness {
    Static,
    Complete,
}

#[derive(Serialize)]
pub struct DeferredCheck {
    pub code: &'static str,
    pub path: String,
    pub reason: String,
}
```

`success=true, completeness=static` means every check possible without external dataset IO passed;
it is not a claim that remote contents are valid or available. `success=true, completeness=complete`
means every endpoint/profile reference in the prepared dataset has also been bound successfully.
Normal execution performs:

```text
static validation
    -> remote/local dataset load into preparation state
    -> bind every deferred endpoint/profile reference
    -> complete semantic validation
    -> create run artifacts
    -> start workers and scheduling
```

Any deferred failure is a preparation failure and occurs before run artifact creation, evaluator or
telemetry worker startup, scheduling, or inference traffic. A separate public networkful `prepare`
operation is not required by protocol v2; an implementation may add one only through a later
versioned protocol extension.

### 5.4 Strict protocol-v2 operation envelope

The runner-wide companion owns the complete operation protocol. Endpoint validation uses this strict
subset:

```json
{
  "protocol_version": 2,
  "operation": "validate",
  "expected_distribution_id": "blake3:...",
  "run": {}
}
```

Normal execution changes only the operation:

```json
{
  "protocol_version": 2,
  "operation": "execute",
  "expected_distribution_id": "blake3:...",
  "run": {}
}
```

The Rust envelope is a strict tagged sum type. The runner first deserializes only the protocol
version, selects that version's strict envelope, and then dispatches the operation. `validate` emits
exactly one `run_validation` JSON line. `execute` emits exactly one `run_terminal` JSON line.

Process exit codes are stable:

| Code | Meaning |
|---:|---|
| `0` | requested operation succeeded |
| `1` | well-formed request reached a semantic, preparation, or execution failure; stdout contains the typed failure response |
| `2` | malformed/unsupported protocol, invalid stdout contract, or runner bootstrap failure |

An `expected_distribution_id` mismatch is a typed protocol failure, happens before request
validation, and exits `2`. Stderr is diagnostic only and MUST be redacted; machine consumers use the
typed stdout response. `--capabilities` remains a side-effect-free bootstrap query rather than an
stdin operation; it uses the same executable and independently composes an equivalent frozen
registry. Capability, validation, and execution normally run in separate processes and do not share
one in-memory registry value; the invariant is deterministic composition from the same verified
executable.

---

## 6. Per-turn endpoints become prebound profiles

The current dataset field `turn.endpoint` ambiguously names a dialect while inheriting the run's
URLs, path, credentials, content type, and headers. That is sufficient only when every dialect in a
conversation targets the same service configuration.

The runner's protocol-v1 support has been deleted: `aiperf-cli` advertises `protocol_versions:
[2]` only and rejects any non-v2 request as a protocol-v2 failure envelope. The former v1 `dispatch`
entry, `execute_v1`/`execute_run*` chain, the `RunRequest`/`RunSpec`/`RunTerminal`/`EndpointSpec`/
`DatasetSpec`/`AccuracySpec` wire DTOs, the `load_protocol_v1` graph-input adapters, and the
`Legacy` capability/enum variants are gone. No v1 authority remains on the runner; only unregistered
workload/lifecycle combinations continue to fail closed. (The `aiperf_runtime::endpoints` module retains its
own internal `EndpointType` metadata/compatibility adapters, independent of the removed runner wire
protocol.)

### 6.1 Protocol-v2 endpoint profiles

The wire model separates a profile name from a dialect ID:

```json
{
  "endpoint": {
    "default_profile": "primary",
    "profiles": {
      "primary": {
        "dialect": "chat",
        "urls": ["https://openai.example"],
        "api_key": "..."
      },
      "claude": {
        "dialect": "messages",
        "urls": ["https://anthropic.example"],
        "api_key": "..."
      }
    }
  }
}
```

`turn.endpoint` then refers to a profile, not a dialect. The loader resolves each authored profile
name to a dense `EndpointKey`; the hot path carries that key and an `EndpointBinding`. Different
profiles may use different URLs, credentials, headers, paths, and endpoint-specific options.

---

## 7. Registry-derived capabilities and protocol evolution

### 7.1 Capability catalog

`RunnerCapabilities::current()` ceases to be a `const` containing handwritten endpoint names.
Capabilities are built from the frozen registry:

```json
{
  "event": "runner_capabilities",
  "capabilities_schema_version": 2,
  "protocol_versions": [2],
  "report_schema_version": "2.0",
  "runner_version": "0.12.0",
  "distribution_id": "blake3:...",
  "extensions": [
    {"name": "acme-endpoints", "version": "1.3.0"}
  ],
  "endpoint_types": ["chat", "messages", "responses"],
  "endpoints": [
    {
      "id": "messages",
      "aliases": [],
      "description": "Anthropic Messages API",
      "endpoint_path": "/v1/messages",
      "streaming_path": null,
      "supports_streaming": true,
      "tokenizes_input": true,
      "produces_tokens": true,
      "input_modalities": ["text", "image"],
      "output_modalities": ["tokens"],
      "request_encoding": "json",
      "lifecycle": "direct",
      "service_kind": "llm",
      "metrics_title": "LLM Metrics"
    }
  ]
}
```

`endpoint_types` is retained as a derived convenience field but is generated from the canonical
`endpoints` entries. Aliases appear only on their canonical entry. Catalog ordering and JSON
serialization are deterministic.

`distribution_id` is an executable-content identity, computed as:

```text
BLAKE3(
  "aiperf-runner-distribution-v1\0"
  || bytes_of_the_executing_image
)
```

`bytes_of_the_executing_image` means the complete executable image from which the current process
was launched, read through the platform's current-executable handle/path with replacement-safe
semantics where the OS exposes them. If a platform cannot obtain the executing image reliably, the
runner fails capability/bootstrap validation rather than substituting package version, Git revision,
or catalog metadata. It identifies the exact runner distribution, not merely the package version. An
endpoint catalog may additionally expose a `catalog_id` for canonicalization, but that identifier
does not replace `distribution_id` and is not used for exact-executable TOCTOU protection.

### 7.2 Handshake and time-of-check/time-of-use protection

Python resolves one `RunnerInstallation`, hashes the selected image using the same versioned domain,
loads its capabilities once, and injects the same object through validation and execution. Protocol
v2 requests carry the expected `distribution_id`. Every validation/execute process recomputes its own
value before comparing `expected_distribution_id`; a run process whose distribution differs from the
capability process fails before artifacts or traffic.

This detects the executable being replaced between capability negotiation and execution by any
behavior-distinct executable — even when package version, descriptors, and extension names are
unchanged — and makes custom statically linked distributions reportable without requiring their
package version to equal the stock Python orchestration package version.

### 7.3 Protocol posture

The runner is protocol-v2-only. Capability schema v2 carries full descriptors, extension identities,
and the executable-content digest; the derived `endpoint_types` field remains for convenience.
Protocol v2 carries the required distribution digest, structured validation with completeness
reporting, and named endpoint profiles. The runner parses a minimal protocol envelope, selects the
version-specific strict DTO, dispatches the operation, and echoes the selected protocol version in
every validation and terminal response. There is no v1 wire path to negotiate against.

New Python MUST fail clearly when paired with a catalog-less or non-v2 runner. It MUST NOT fall back
to Python endpoint metadata.

---

## 8. Python becomes endpoint-neutral

### 8.1 Structural Config-v2 validation only

Change both Python endpoint-type annotations from the plugin-generated enum to a normalized,
non-empty string:

```python
endpoint_type: str = "chat"
```

The portable checked-in JSON schema likewise declares `endpoint.type` as a string. It cannot list
the extensions compiled into the selected `aiperf --execute` binary (overridable via
`AIPERF_EXEC_BIN`).

Python continues to validate structure needed for orchestration: field shapes, interpolation,
sweep syntax, trial/search configuration, and path projection. It does not:

- decide whether an endpoint supports streaming;
- derive or reject its request body encoding;
- decide whether its input/output is tokenized;
- select text/ranking/media synthetic defaults from endpoint metadata;
- decide whether a tokenizer is required;
- construct an endpoint readiness payload;
- derive endpoint behavior from `service_kind` or metric-title metadata.

Direct `AIPerfConfig.model_validate()` means structural validation. A command/API promising an
executable configuration MUST additionally call the selected runner's validation operation.

### 8.2 The execution handle is explicit and injected

Capability negotiation and the execution handle are a reusable value rather than a hidden module
global:

```python
@dataclass(frozen=True)
class RunnerInstallation:
    binary: Path
    capabilities: RunnerCapabilities
    distribution_id: str

    def validate(self, request: dict[str, Any]) -> RunValidation: ...
    def execute(self, request: dict[str, Any]) -> RunTerminal: ...
```

It is resolved once per CLI invocation and passed explicitly through runner-aware preflight and the
executor. It is not a module-global singleton and Pydantic constructors do not secretly spawn a
subprocess.

The execution `binary` is the **same `aiperf` binary re-execing itself** as `aiperf --execute`; no
external binary is discovered and there is no discovery-order search. The only override is
`AIPERF_EXEC_BIN`, which points the execution child at a differently-compiled build (for example a
`dynosim`/custom-features binary).

`aiperf --help`, schema generation, plotting, and other non-execution commands continue to work
without composing the execution engine.

### 8.3 Plan validation

For a fixed grid/zip/scenario plan, every distinct projected run is validated before the first run
executes. Adaptive proposals are validated when created and before entering the executor. The same
`RunnerInstallation` is used for all cells unless the user explicitly starts a different CLI
invocation.

Runner notices are presented by Python but never recomputed there. The effective endpoint profile
and normalizations are reported by Rust and used for provenance/result presentation.

### 8.4 Help and schemas

- Portable CLI help exposes `--endpoint-type TEXT`.
- A runner-backed `aiperf runner endpoints` command lists the exact selected catalog.
- The portable checked-in Config-v2 schema accepts any syntactically valid endpoint ID.
- An optional exact-schema command may decorate a schema with endpoint IDs/descriptions from one
  selected runner for IDE completion. That generated artifact embeds the distribution digest and is
  never a runtime authority.
- Official documentation may be generated from the stock release runner in CI, clearly labeled as
  the stock catalog.

### 8.5 Deletion ledger

The endpoint migration is incomplete until these Python ownership points are deleted or made
endpoint-agnostic:

1. endpoint entries in `src/aiperf/plugin/plugins.yaml`;
2. endpoint enum generation in `src/aiperf/plugin/enums.py`;
3. Python endpoint classes and response parsers;
4. endpoint-specific Pydantic validators;
5. endpoint metadata consumers in dataset/tokenizer conversion;
6. hardcoded readiness payload/path tables;
7. Python endpoint-aware aiohttp execution;
8. endpoint-aware Python record/post-processing behavior already owned by native reports;
9. tests that exist only to pin the deleted Python execution path.

A temporary parity test may protect a short migration interval, but it MUST be deleted with the
last Python metadata consumer. A permanent Python/Rust parity suite would institutionalize the dual
system this spec forbids.

---

## 9. Packaging the only executor

There is exactly **one** `aiperf` wheel that carries the native execution engine. maturin compiles
the `pyext` pyo3 module into `aiperf._native` and packages the `src/aiperf` frontend;
`tools/wheel_repack.py` (run by `make wheel`) then repacks the native `aiperf` binary directly into
the wheel's scripts directory (`aiperf-<ver>.data/scripts/aiperf`). Because the wheel carries a
native binary it is platform + CPython-ABI specific — there is no separate companion package and no
pure-Python-only orchestration wheel depending on one. The execution child is the same binary
re-execing itself as `aiperf --execute`; a custom compile-time extension build is selected
explicitly via `AIPERF_EXEC_BIN`.

Release gates include:

- one platform + ABI-specific `aiperf` wheel for every supported OS/architecture;
- a fresh-environment smoke test that runs the repacked binary without `AIPERF_EXEC_BIN`;
- capabilities-catalog protocol/digest verification;
- a frontend -> `aiperf --execute` -> loopback mock benchmark in the release container;
- native Cargo tests in the primary CI path, not only inherited Python tests;
- exact binary identity and linked extension provenance in native-v2 output.

Package version equality is not the compatibility authority because custom feature-bearing builds
are valid. Protocol versions, capability schema, report schema, and the distribution digest are.

---

## 10. Implementation sequence

### Increment 1 — open Rust identity and one registry

1. Add `EndpointId`, adapter-owned descriptors, and explicit aliases.
2. Move endpoint registry/resolver ownership into `aiperf-endpoints`.
3. Replace the enum-indexed map and `resolve_type` with canonical ID resolution.
4. Add deterministic registry enumeration and freeze semantics.
5. Generate legacy `endpoint_types` and full endpoint descriptors from that registry.
6. Replace central endpoint enum branches with trait behavior or descriptor-driven generic policy.
7. Keep existing endpoint behavior fixtures byte/semantic exact.

### Increment 2 — runner composition and validation

1. Build built-ins and linked extensions once in the runner composition root.
2. Inject that frozen registry into capabilities and execution.
3. Add `RawEndpointConfig`, `EffectiveEndpointConfig`, and `EndpointBinding`.
4. Remove endpoint identity from `EndpointConfig`.
5. Implement shared `validate_run`/preparation and structured diagnostics.
6. Validate after effective streaming normalization and before side effects.
7. Prove a genuinely new extension ID in capabilities, validation, formatting, parsing, and a
   loopback run.

### Increment 3 — endpoint-neutral Python

1. Change Config-v2 and CLI endpoint types to strings.
2. Introduce and inject `RunnerInstallation`.
3. Remove Python streaming/content-type endpoint normalization.
4. Move synthetic defaults, tokenizer necessity, readiness probing, and endpoint-dependent
   cross-field validation to Rust.
5. Validate fixed plans before execution and adaptive proposals before admission.
6. Make the portable schema endpoint-open and add exact runner-backed discovery/help.

### Increment 4 — protocol v2 and endpoint profiles

1. Add capability/distribution digests and extension provenance.
2. Add version-dispatched request envelopes and structured validation terminals.
3. Add named endpoint profiles and dense prebound `EndpointKey`s.
4. Carry the expected distribution digest in every v2 request.
5. Delete the v1 adapter; the runner is protocol-v2-only.

### Increment 5 — packaging and deletion

1. Ship the single repacked `aiperf` wheel and container artifact.
2. Delete the Python endpoint plugin category, enum, classes, validators, transport, and migrated
   consumers.
3. Delete temporary parity machinery.
4. Protocol v1 and its legacy capability fields are removed; the runner advertises `[2]` only.

Each increment is independently testable. New endpoints MUST NOT be added to the Python manifest
during the migration; the first increments must make a Rust-only endpoint selectable before
further dialects land.

---

## 11. Verification gates

### Registry and extension invariants

1. Canonical capability endpoint IDs exactly equal frozen registry entries.
2. Every alias resolves to its canonical entry; aliases are not advertised as separate endpoints.
3. Duplicate IDs/aliases fail transactionally with actionable diagnostics.
4. Catalog ordering and distribution digest are deterministic across processes.
5. A separate extension crate introduces a new endpoint ID not present in core.
6. That extension appears in capabilities and executes through the same runner registry.

### Validation invariants

1. Unknown syntactically valid IDs reach registry diagnostics rather than Serde enum errors.
2. Unsupported streaming, multipart derivation/conflicts, template requirements, URL validation,
   tokenizer necessity, and endpoint/phase compatibility are pinned in Rust.
3. Invalid endpoint configuration fails before artifacts, workers, readiness probes, or traffic.
4. Per-turn/profile bindings are validated before scheduling and are never mutated at dispatch.
5. Validation and execution share the same normalization golden tests.
6. Structured errors contain stable codes, paths, available choices, and redacted hints.

### Protocol and Python invariants

1. New Python + new runner/v2 succeeds; a non-v2 request is rejected as a v2 failure envelope.
2. New Python + catalog-less or non-v2 runner fails with an upgrade instruction and no fallback.
4. A custom endpoint can be selected, validated, and run without any Python registration/change.
5. Fixed sweep variations and adaptive proposals are validated against the same installation.
6. Digest mismatch fails before artifacts or inference traffic.
7. `aiperf --help` works without a runner; execution commands fail clearly when it is absent.
8. A deletion guard proves config/orchestration modules do not import Python endpoint metadata.

### Packaging invariants

1. A fresh stock installation finds its packaged runner.
2. The release container executes a minimal real subprocess benchmark.
3. Every supported platform wheel runs `--capabilities` and a loopback smoke test.
4. Native report provenance records runner path, version, distribution digest, and extensions.

---

## 12. Rejected alternatives

### Keep Python and Rust metadata in parity

Rejected. It leaves two authorities, makes every endpoint a cross-language change, and has already
drifted for `messages`. A parity test detects duplication; it does not remove it.

### Generate a permanent Python endpoint manifest or enum from Rust

Rejected as a runtime authority. A checked-in/generated artifact describes the stock build that
produced it, not an arbitrary `AIPERF_EXEC_BIN` build with different linked extensions. Such
artifacts are acceptable only for documentation or IDE assistance and must carry the source digest.

### Query Rust metadata and keep Python's endpoint validators

Rejected as the end state. This centralizes metadata values but retains two implementations of
streaming, encoding, tokenizer, and cross-field semantics. Python calls the runner validation
operation instead.

### Generate Pydantic/Cyclopts endpoint enums at import time

Rejected. It couples imports, help, schema generation, and non-execution commands to spawning an
external binary; it is incompatible with programmatic model construction and still does not remove
duplicate semantic validators.

### Keep the closed Rust `EndpointType` enum and add variants for extensions

Rejected. An out-of-tree statically linked extension cannot add a variant to a core enum. The wire
would reject a new dialect before the registry extension seam could resolve it.

### Create a fresh endpoint object per request

Rejected. Current adapters are stateless and immutable. Registry adapter instances and validated
run/profile bindings are shared; request-local state remains request-local. A future stateful
construction need uses an injected factory seam, not per-request registry discovery.

### Runtime manifest discovery or Rust dynamic libraries

Rejected. This repository deliberately has no `plugins.yaml`, global discovery, or unstable Rust
dynamic-library ABI. Extensions are compiled dependencies applied explicitly by the runner
distribution.

### Let Python choose another executor when the runner is missing

Rejected. There is no dual execution system. Missing runner installation is a configuration/setup
error, not a request to fall back to legacy Python inference.

---

## 13. Completion criteria

This design is complete only when all of the following are true:

- `aiperf-cli` constructs one frozen, extension-aware registry used by capabilities, validation,
  and execution;
- endpoint identity is open and registry-addressed rather than a closed enum;
- endpoint descriptors and behavior are co-located and registered once;
- the runner owns every endpoint-dependent normalization and semantic check;
- default and per-turn/profile adapters are prebound with validated effective configurations;
- a new compiled endpoint requires no Python source or generated-artifact change;
- the normal Python package/container reliably supplies or selects the exact runner;
- Python contains no endpoint manifest, enum, endpoint implementation, semantic validator,
  transport fallback, or metadata fallback for behavior already native;
- temporary compatibility and parity machinery has been removed;
- all three execution modes continue to consume the same endpoint, transport, clock, and metric
  paths.

### Built state

The following are built:

- `aiperf_runtime::endpoints` (formerly `aiperf-endpoints`) owns validated open `EndpointId` syntax,
  adapter-local descriptors, explicit aliases, deterministic `BTreeMap`-backed startup registration,
  and a frozen registry. Endpoint identity is separate from `RawEndpointConfig`, and the registry
  alone constructs the private validated `EffectiveEndpointConfig`.
- Object-safe `EndpointFactory` and worker-local `PreparedEndpoint` values bind behavior and policy
  before dispatch; `PreparedRequest` carries no endpoint configuration, and `EndpointKey` plus
  `PreparedEndpointTable` provide the dense lookup seam. Raw/template factories compile the Minijinja
  template and JMESPath selector once per prepared worker/profile binding. Readiness is an
  object-safe dialect-owned policy with an explicit unsupported result.
- The frozen registry owns nine open KServe factories (three OpenAI-compatible HTTP routes, KServe V1
  Predict, five KServe V2 OIP dialects) plus `vllm_generate`; each owns its descriptor and prepared
  behavior without a closed `EndpointType` variant.
- `aiperf_runtime::extensions` transactionally registers a genuinely new test dialect whose ID and behavior
  have no `EndpointType` variant; duplicate IDs and aliases fail atomically and catalogs remain
  sorted.
- Selected online execution convergence is built: runner-v2 HTTP scheduled and native gRPC scheduled
  adapters prepare one dense endpoint table per worker; the `aiperf_runtime::transport_grpc` registry
  prepares a matching dense `GrpcEndpointBinding` table for the five gRPC-capable KServe V2 IDs (see
  `2026-07-12-aiperf-native-grpc-kserve-v2-design.md`).
- The runner is protocol-v2-only, advertising `protocol_versions: [2]`.

Remaining open work: unregistered workload/lifecycle combinations still fail closed, and full
runner-wide reachability convergence for every built Rust mode is owned by the runner-only
execution-surface companion. Until every criterion above holds, the code is in migration and the
code—not this spec—remains the source of truth for which pieces are actually built.
