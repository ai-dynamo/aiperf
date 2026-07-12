<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AIPerf: runner-owned endpoint registry, validation, and execution

**Date:** 2026-07-11
**Author:** Anthony Casagrande (Tech Lead) + Codex
**Status:** decided — not yet built
**Decision:** the exact selected `aiperf-runner` binary is the sole authority for endpoint
identity, metadata, semantic validation, normalization, request/response behavior, and execution.
Python remains a structural Config-v2 and orchestration front end while that front end exists; it
does not retain an endpoint implementation, endpoint manifest, endpoint enum, metadata fallback,
or endpoint-specific validation engine.

**Companions:**

- `2026-07-11-aiperf-rust-endpoints-design.md` remains authoritative for the faithful request
  formatting, response parsing, replay, and input-accounting behavior of each dialect.
- `2026-07-11-aiperf-rust-compile-time-extension-registry-design.md` remains authoritative for
  explicit, transactional, statically linked extension composition.
- `2026-07-11-python-orchestrator-rust-single-run-design.md` remains authoritative for the process
  boundary: Python plans runs and a fresh `aiperf-runner` process executes one run.

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
the selected aiperf-runner binary
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
`crates/aiperf-runner/src/protocol.rs:49-68`, while the Python endpoint block in
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
exists in `crates/aiperf-endpoints/src/config.rs:100-182`. Feeding Rust metadata into the existing
Python validators would remove one data-copy but leave two executable rule engines. That is not
the target architecture.

### 1.3 Rust itself has multiple endpoint authorities

The native implementation also needs consolidation:

1. `crates/aiperf-endpoints/src/metadata.rs:8-48` defines a closed `EndpointType` enum.
2. The same file stores a separate metadata table selected by that enum.
3. `crates/aiperf-dataset/src/request.rs:97-226` manually registers endpoint names and keeps a
   second enum-indexed map.
4. `crates/aiperf-runner/src/protocol.rs:49-68` hardcodes the capability list independently.
5. `crates/aiperf-runner/src/execute.rs:351` constructs `AiperfRegistry::builtin()` inside
   execution rather than accepting the runner's composed registry.

As a result, the current compile-time endpoint extension proof is not a proof of a new dialect. It
registers `ChatEndpoint` under another name, but the identity remains `EndpointType::Chat`. A name
whose dialect is not already in the closed enum is rejected during Serde deserialization before
registry lookup, and the production runner never applies endpoint extensions to the registry it
uses for capabilities or execution.

### 1.4 Per-turn selection can produce an invalid adapter/config pair

`crates/aiperf/src/multiturn.rs:1017-1038` resolves an authored per-turn endpoint, clones the
default configuration, overwrites its enum identity, and clamps streaming. It does not fully
revalidate that configuration for the selected dialect. A form-data mismatch can therefore
survive until the dispatch invariant in `crates/aiperf/src/http/endpoint_dispatch.rs:297-327`.

The final model MUST make the selected adapter and its validated effective configuration
inseparable and MUST bind every distinct endpoint/profile before request scheduling.

---

## 2. Required invariants

1. **Exact-binary authority.** Capabilities, validation, and execution use one registry composed
   by the exact selected `aiperf-runner` binary.
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

### 3.4 Adapter/config binding

```rust
#[derive(Clone)]
pub struct EndpointBinding {
    pub canonical_id: EndpointId,
    pub endpoint: Arc<dyn Endpoint>,
    pub config: Arc<EffectiveEndpointConfig>,
}
```

Current adapters are stateless zero-sized implementations. One adapter instance is registered and
shared; a new adapter is not constructed per request. A binding is created once per distinct
run/profile configuration. If a future dialect requires compiled run-scoped state, the registry
MUST expose an object-safe `EndpointFactory` seam with a blanket stateless factory rather than
placing mutable state in the global catalog or adding endpoint-ID branches.

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

`aiperf-runner` builds this registry once at process startup. `--capabilities`, validation, and
normal execution all borrow the same frozen value. `execute_run` is changed to accept an injected
registry or becomes a method on a runner application object. It MUST NOT call
`AiperfRegistry::builtin()` internally.

---

## 5. One preparation and validation path

### 5.1 Runner application object

```rust
pub struct Runner {
    registry: AiperfRegistry,
}

impl Runner {
    pub fn capabilities(&self) -> RunnerCapabilities;
    pub fn validate(&self, request: RunRequest) -> RunValidation;
    pub fn execute(&self, request: RunRequest) -> Result<RunTerminal>;
}
```

Both `validate` and `execute` invoke the same pure semantic preparation function:

```rust
fn validate_run(
    request: RunRequest,
    registry: &AiperfRegistry,
) -> Result<ValidatedRun>;
```

Execution extends `ValidatedRun` into IO-bound `PreparedRun` state by loading the selected dataset,
tokenizer, and optional sidecars. No endpoint semantic rule is reimplemented in the IO phase. For
local or inline datasets, validation may also scan authored endpoint/profile references. References
only discoverable through a remote dataset are bound immediately after load and before scheduling
or HTTP dispatch.

Pure validation MUST occur before:

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

### 5.3 Structured validation result

Validation is a strict JSON operation of `aiperf-runner`, using the same one-line stdout discipline
as capabilities and terminal results:

```json
{
  "event": "run_validation",
  "protocol_version": 2,
  "success": false,
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

---

## 6. Per-turn endpoints become prebound profiles

The current dataset field `turn.endpoint` ambiguously names a dialect while inheriting the run's
URLs, path, credentials, content type, and headers. That is sufficient only when every dialect in a
conversation targets the same service configuration.

### 6.1 Protocol-v1 compatibility

The v1 adapter retains current meaning during migration:

- the run's `endpoint.type` is the default dialect ID;
- each unique authored `turn.endpoint` is resolved as a dialect ID;
- the base raw config is cloned and independently validated for each unique dialect;
- every resulting binding is cached before scheduling;
- invalid combinations fail before the affected turn can dispatch.

### 6.2 Protocol-v2 endpoint profiles

The final wire model separates a profile name from a dialect ID:

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
  "protocol_versions": [1, 2],
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

`endpoint_types` is retained temporarily for old orchestration clients but is derived from the
canonical `endpoints` entries. Aliases appear only on their canonical entry. Catalog ordering and
JSON serialization are deterministic.

`distribution_id` is a BLAKE3 digest over the canonical capability contract, linked extension
identities/versions, and other execution-relevant build identity. It identifies the exact runner
distribution, not merely the package version.

### 7.2 Handshake and time-of-check/time-of-use protection

Python resolves one `RunnerInstallation`, loads its capabilities once, and injects the same object
through validation and execution. Protocol v2 requests carry the expected `distribution_id`. A run
process whose distribution differs from the capability process fails before artifacts or traffic.

This handles the executable being replaced between capability negotiation and execution and makes
custom statically linked distributions reportable without requiring their package version to equal
the stock Python orchestration package version.

### 7.3 Compatibility sequence

1. Add capability schema v2, full descriptors, extension identities, and a digest additively.
2. Keep legacy `endpoint_types`, derived from the registry.
3. Change Rust's v1 internal endpoint type from the closed enum to `EndpointId`; the JSON field
   remains the same string and therefore does not require a wire break.
4. Add protocol v2 for the required distribution digest, structured validation, and named endpoint
   profiles.
5. Parse a minimal protocol envelope before selecting a version-specific strict DTO. The current
   runner deserializes directly into v1 and cannot negotiate this safely.
6. Echo the selected protocol version in every validation and terminal response.
7. Remove v1 and legacy `endpoint_types` after the compatibility window.

New Python MUST fail clearly when paired with a catalog-less old runner. It MUST NOT fall back to
Python endpoint metadata.

---

## 8. Python becomes endpoint-neutral

### 8.1 Structural Config-v2 validation only

Change both Python endpoint-type annotations from the plugin-generated enum to a normalized,
non-empty string:

```python
endpoint_type: str = "chat"
```

The portable checked-in JSON schema likewise declares `endpoint.type` as a string. It cannot list
the extensions compiled into an arbitrary `AIPERF_RUNNER_BIN`.

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

### 8.2 `RunnerInstallation` is explicit and injected

Extract runner discovery and capability negotiation from `RustSubprocessExecutor` into a reusable
value:

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

Discovery precedence is:

1. explicit `--runner-bin` API/CLI selection;
2. `AIPERF_RUNNER_BIN`;
3. the matching installed stock runner package;
4. `PATH`, primarily for development.

`aiperf --help`, schema generation, plotting, and other non-execution commands continue to work
without a runner. `profile` and exact executable-config validation fail early with installation
guidance if no compatible runner exists.

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

The runner-only architecture is not shippable until the normal installation contains a compatible
runner. The current project packaging is Python-only, runner discovery relies on an external
binary, and current container/release paths do not establish a stock native artifact.

The stock release uses a platform-specific companion wheel containing `aiperf-runner`; the main
orchestration wheel may remain pure Python and depends on the matching platform package. A custom
compile-time extension distribution ships its own runner and selects it explicitly.

Release gates include:

- platform wheels for every supported OS/architecture;
- a fresh-environment smoke test that finds the packaged runner without `AIPERF_RUNNER_BIN`;
- `--capabilities` protocol/digest verification;
- a Python -> runner -> loopback mock benchmark in the release container;
- native Cargo tests in the primary CI path, not only inherited Python tests;
- exact runner identity and linked extension provenance in native-v2 output.

Package version equality is not the compatibility authority because custom runner distributions
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
5. Retain the v1 adapter for the compatibility window.

### Increment 5 — packaging and deletion

1. Ship the stock runner companion package and container artifact.
2. Delete the Python endpoint plugin category, enum, classes, validators, transport, and migrated
   consumers.
3. Delete temporary parity machinery.
4. Remove protocol v1 and legacy capability fields after the announced compatibility window.

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

1. Old Python + new runner/v1 succeeds during the compatibility window.
2. New Python + new runner/v2 succeeds.
3. New Python + catalog-less old runner fails with an upgrade instruction and no fallback.
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
produced it, not an arbitrary selected `AIPERF_RUNNER_BIN` with different linked extensions. Such
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

- `aiperf-runner` constructs one frozen, extension-aware registry used by capabilities, validation,
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

Until these criteria hold, the code is in migration and the code—not this spec—remains the source
of truth for which pieces are actually built.

## Addendum — 2026-07-11 (adversarial adjudication and binding corrections)

An independent adversarial review, a neutral reproduction, a rebuttal, and a final adjudication
upheld this spec's ownership decision and required the corrections below. This addendum is
authoritative wherever the original body is ambiguous or conflicts with it.

Runner-wide reachability for scheduled online, Graph-IR, Dynamo offline, static accuracy, and
stateful agentic execution is deliberately **not** designed here. It belongs to the normative
companion `2026-07-11-aiperf-runner-only-execution-surface-design.md`. This endpoint record owns
only endpoint identity, preparation, validation, catalog publication, and endpoint use by those
runner operations. Product deletion is complete only when that companion makes every built Rust
mode runner-addressable with the same composed endpoint registry.

### A. Authored request before side effects

The protocol-v1 order in `2026-07-11-python-orchestrator-rust-single-run-design.md`—Python resolver
side effects followed by projection—is superseded for protocol v2. The permanent order is:

```text
Python structural Config-v2 validation and outer-loop expansion
    |
    v
side-effect-free AuthoredRunRequest projection
    |
    v
aiperf-runner static validation
    |
    v
Rust dataset/tokenizer/backend preparation and deferred validation
    |
    v
artifact creation, supervised workers, scheduling, and dispatch
```

`AuthoredRunRequest` is built only from authored/structurally resolved configuration. It MUST NOT
read `BenchmarkRun.resolved`, require a pre-created artifact directory, require a cache-localized
tokenizer, or require Python to import a dataset loader. The request carries:

- the selected artifact target path without creating it;
- authored tokenizer identity/options rather than a Python-warmed cache result;
- authored local/inline/public dataset identity and options rather than Python loader metadata;
- the endpoint/profile IDs and raw endpoint policy;
- the backend/workload request owned by the runner-only execution-surface companion.

Rust consumes those authored dataset and tokenizer inputs directly while constructing
`PreparedRun`. Python path selection that remains necessary for orchestration is split from
directory creation and user-file materialization. No Python resolver may perform a side effect
before the selected runner accepts static validation.

Protocol v1 may retain the old fully resolved projection only during its compatibility window. It
is not the model for new fields or operations.

### B. Validation completeness is explicit

Static validation cannot inspect endpoint/profile references stored only inside rows of a remote
dataset. `RunValidation` therefore reports its scope:

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

Every validation result contains:

```json
{
  "completeness": "static",
  "deferred_checks": [
    {
      "code": "dataset_endpoint_profiles",
      "path": "run.dataset",
      "reason": "endpoint profile references require loading the remote dataset"
    }
  ]
}
```

`success=true, completeness=static` means every check possible without external dataset IO passed;
it is not a claim that remote contents are valid or available. `success=true,
completeness=complete` means every endpoint/profile reference in the prepared dataset has also
been bound successfully.

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
telemetry worker startup, scheduling, or inference traffic. Dataset cache population needed to read
remote input is preparation state, not a run artifact. A separate public networkful `prepare`
operation is not required by protocol v2; an implementation may add one only through a later
versioned protocol extension.

### C. Strict protocol-v2 operation envelope

The runner-wide companion owns the complete operation protocol. Endpoint validation uses this
strict subset:

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
validation, and exits `2`. Stderr is diagnostic only and MUST be redacted; machine consumers use
the typed stdout response.

`--capabilities` remains a side-effect-free bootstrap query rather than an stdin operation. It uses
the same executable and independently composes an equivalent frozen registry. Capability,
validation, and execution normally run in separate processes; they do not share one in-memory
registry value. The invariant is deterministic composition from the same verified executable.

### D. `distribution_id` is an executable-content identity

The original phrase “other execution-relevant build identity” is replaced by this exact contract:

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
or catalog metadata.

The Python `RunnerInstallation` hashes the selected image using the same versioned domain and
requires the capability process to report that value. Every validation/execute process recomputes
its own value before comparing `expected_distribution_id`. This detects replacement by any
behavior-distinct executable even when package version, descriptors, and extension names are
unchanged.

Endpoint catalog canonicalization may additionally have a `catalog_id`, but that identifier does
not replace `distribution_id` and is not used for exact-executable TOCTOU protection. Tests build
two behavior-distinct runner images and prove their distribution IDs differ.

### E. Prepared bindings are worker-local and allocation-free on dispatch

The illustrative `EndpointBinding` in section 3.4 is superseded by two explicit seams:

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
`Vec<Box<dyn PreparedEndpoint>>` in worker-local state. A turn carries only copyable
`EndpointKey`. Dispatch performs one checked indexed borrow and no `Arc` clone, mutex operation,
string lookup, endpoint construction, or endpoint-config mutation.

The initial implementation MUST include this preparation seam; it is not deferred to a hypothetical
future dialect. Raw/template preparation resolves template source safely, constructs the Minijinja
environment/template, and compiles the JMESPath response selector once per effective worker/profile
binding. Stateless endpoints use a blanket factory whose preparation only stores their validated
configuration.

`EffectiveEndpointConfig` and prepared-binding fields are private outside their owning module.
Only validation/preparation constructors can create them, so callers cannot construct an adapter
and mismatched configuration with public struct literals.

### F. Adjudicated scope and completion gate

The following concerns were explicitly overruled and require no design reversal:

- moving registry ownership into `aiperf-endpoints` does not create a dependency cycle;
- open string IDs are necessary for genuinely new statically linked endpoint dialects;
- linked extensions can provide ordinary static descriptors;
- the current per-turn adapter/config mismatch is real and requires prebinding.

The core runner-owned endpoint architecture is accepted. It becomes implementation-ready only with
this addendum and the separate runner-only execution-surface companion. No backend/workload mode DTO
is added to this endpoint spec.

## Addendum — 2026-07-11 (open-registry foundation implementation)

The first in-tree implementation slice is now built, without claiming the full completion criteria
in section 13:

- `aiperf-endpoints` owns validated open `EndpointId` syntax, adapter-local descriptors, explicit
  aliases, deterministic `BTreeMap`-backed startup registration, and a frozen registry;
- endpoint identity is separate from `RawEndpointConfig`, while the registry alone constructs the
  private validated `EffectiveEndpointConfig`;
- object-safe `EndpointFactory` and worker-local `PreparedEndpoint` values bind behavior and policy
  before dispatch; `PreparedRequest` contains no endpoint configuration, and `EndpointKey` plus
  `PreparedEndpointTable` provide the dense lookup seam;
- raw/template factories compile the Minijinja template and JMESPath selector once per prepared
  worker/profile binding;
- readiness is an object-safe dialect-owned policy with an explicit unsupported result, so callers
  cannot substitute a chat probe;
- `aiperf-extensions` transactionally registers a test dialect whose ID and behavior have no
  `EndpointType` variant; duplicate IDs and aliases fail atomically and catalogs remain sorted.

Protocol-v1 `EndpointType`, `EndpointMetadata`, and the dataset-owned compatibility resolver remain
temporarily present, and the production runner still publishes a handwritten endpoint list and
constructs built-ins internally. Consequently, exact-binary registry authority, registry-derived
capabilities, shared validate/execute preparation, and full runner injection are still pending.
This addendum records implementation progress only; it does not narrow the original completion
criteria or authorize a second permanent endpoint authority.
