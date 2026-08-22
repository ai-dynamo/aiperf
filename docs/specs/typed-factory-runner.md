<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Typed-factory runner

## Purpose

State the target for the execution boundary once
[config-model-unification.md](config-model-unification.md) lands: **the runtime
consumes the typed `BenchmarkRun` directly, and a component is selected by an
open, normalized string id (`RegistryId`) whose config is a typed struct for
built-ins and an opaque `RawValue` (decoded by the plugin's own factory) only for
the runtime-loaded-plugin tail** — instead of the `AuthoredRunSpecV2` projection
that today re-serializes every section to `Value` and re-decodes it per factory.
This is the completion of the config-model-unification arc and the step to parity
with the Python reference, where the same typed `BenchmarkConfig` rides inside
`BenchmarkRun` to the child and every service reads `cfg.endpoint` / `cfg.transport`
typed. It types the config payload (the whole prize) while keeping the discriminant
a plain string — which is what Python's `ExtensibleStrEnum` already is — and
confines the opaque config seam to the one place extensibility requires it (the
plugin tail), rather than applying `RawValue` uniformly to built-ins and
first-party fields that have no extensibility reason to be opaque.

This record is forward-looking. It describes work that is **not built**.

## Built

Today the child does not consume `BenchmarkConfig` directly. `coordinator.rs`
takes `envelope.run.into_authored()` and runs the rest of the engine against
`AuthoredRunSpecV2` — a *second* run model produced by projecting the typed
`BenchmarkConfig`:

- **`BenchmarkRunWireV2.cfg`** is the typed `BenchmarkConfig`, but
  `into_authored` re-serializes each section to `serde_json::Value`
  (`to_value(&cfg.runtime)`, `to_value(&cfg.phases)`, …), reshapes it, derives
  `workload_id` from the typed `workload_kind(&cfg)` classifier, folds
  content-server into sidecars, and hand-builds `json!` blobs for three strict
  per-workload DTOs (`ScheduledWorkloadConfigV2`, `GraphWorkloadConfigV2`,
  `StaticAccuracyWorkloadConfigV2`). The graph DTO carries **five** fields the
  scheduled DTO does not (`protocol_v2.rs:404-422`): `weka_semantics`,
  `ignore_trace_delays`, `recorded_agent_default`, `planned_replay_traces`, and a
  conditionally-attached `system_idle_gap_cap_seconds` (emitted only when
  `weka_semantics` is `legacy` or `agentx`). Only the first two and the cap are
  plain copies of typed `BenchmarkConfig` fields; `recorded_agent_default` is a
  *derivation* (`cfg.scenario.as_deref() == Some("recorded-agent-default")`) and
  `planned_replay_traces` comes from `BenchmarkRunWireV2`, not from
  `BenchmarkConfig` at all.
- **Components are open `RawValue`.** `NamedRunnerComponentSpecV2 { id:
  ComponentId, config: Box<RawValue> }` carries an opaque payload "strictly
  decoded by the selected factory." `TransportFactory::validate(&self, authored:
  &RawValue, …)` and `WorkloadFactory::validate(&self, authored: &RawValue)`
  each parse their own slice into a `Box<dyn ValidatedTransportConfig>` /
  `Box<dyn ValidatedWorkloadConfig>`. The core never depends on a component's
  config type; dispatch is a registry lookup by `ComponentId`.

The `RawValue` boundary buys **compile decoupling** — the runtime crate does not
name each transport/workload config type — at the cost of end-to-end type safety
and a whole projection module. The registry is nonetheless **frozen at
bootstrap** ("one frozen implementation universe for a fresh runner child"): the
set of transports/workloads/exporters is fixed at compile time, so the openness
the `RawValue` seam provides is never exercised by true runtime plugins. The
Python reference runs the identical product with typed `#[serde(tag = "type")]`
unions (`phases`, `datasets`) and flat typed `endpoint`/`transport`, **zero**
projection and **zero** `RawValue` — the existence proof that the projection is a
Rust implementation choice, not a requirement of a typed run model.

## Future requirements

> Feasibility audited (2026-07-26) in two waves of adversarial code review — first
> the claim decomposition (transport union, workload collapse, `into_authored`
> deletion, seam typing, registry reduction), then the extensible-enum encoding
> (an **empirical** serde compile-test, normalized lookup, two-path dispatch, wire
> parity). No hard blocker survived; the material corrections each wave forced are
> folded into the steps below — most importantly that the enum needs a **manual
> `Deserialize`** (the derive is broken) and that the port is of the extensible-enum
> *pattern*, not Python's *values*.

### 1. Typed component unions on `BenchmarkConfig`

Every component config the runner selects becomes a typed field on the one model,
following the shape `phases`/`datasets` already have:

- `transport` — the **config payload** becomes typed per built-in
  (`HttpTransportConfig`, `GrpcTransportConfig`, `DryRunConfig`, and under their
  features the `dynosim_offline`/`dynosim_online` configs). The **discriminant
  stays an open, normalized `RegistryId` string**, decoded by a plain derive and
  dispatched by `match id.as_str()` with a plugin-tail default arm — see §3,
  which is authoritative on this point. An earlier draft of this record made
  `transport` a closed `#[serde(tag = "type")]` enum; the 2026-07-26 empirical
  serde test retired that encoding, and no closed transport enum is proposed
  here. Feature gating applies to the built-in `match` arms, not to enum
  variants.
- Workload kind stays **emergent**, computed by the existing
  `config::model::workload_kind(&BenchmarkConfig)` from dataset + phase shape —
  never a wire field. `workload_kind()` is already a total, IO-free function over
  typed fields returning `{Scheduled, Graph}`, so an exhaustive 2-arm match is a
  sound substitute for the `ComponentId` lookup. The two `*WorkloadConfigV2` DTOs
  differ by **five** graph-only fields (see [Built](#built)). Three
  (`weka_semantics`, `ignore_trace_delays`, `system_idle_gap_cap_seconds`) are
  typed fields on `BenchmarkConfig` and collapse into typed-optional fields
  consulted only on the graph arm. The other two do **not** collapse for free and
  are the real work of this step: `recorded_agent_default` is a derivation over
  `cfg.scenario` that must be recomputed (or the scenario string carried and the
  comparison moved into the graph arm), and `planned_replay_traces` is
  controller-injected run data with no `BenchmarkConfig` home at all — see §5
  step 4. Caveat: a third `static_accuracy`
  workload shape exists (`StaticAccuracyWorkloadConfigV2`, adds `accuracy`, drops
  `failure_policy`) but is registered only in tests and unreachable via
  `workload_kind` today; if the accuracy path is later promoted to a distinct
  workload id (the existing `TODO(step-1)`), the exhaustive match grows a
  `StaticAccuracy` arm and a third typed-optional `accuracy` field.
- The untyped seams inside the model are typed out in the same pass:
  `dataset.synthesis: Option<serde_json::Value>` becomes
  `Option<TraceSynthesisSpec>` (that struct already exists as the decode target in
  `engine/dataset_input.rs`, is `deny_unknown_fields`, and every key the two
  producers write is already one of its fields — the change is mechanical, but the
  struct must gain `Serialize`/`Default` and move to / be re-exported from
  `config::model::dataset` so the resolver can name it without an engine
  dependency); `weka_semantics: Option<String>` becomes an enum that **must carry
  `#[serde(alias)]` + input normalization** to preserve the consumer's existing
  leniency (case-insensitive, trimmed, and the `graph-ir`/`graphir`/`graph_ir`
  spelling trio) — a naive strict enum would be a behavior regression; and
  `failure_policy: Option<serde_json::Value>` becomes its typed `OnFailure` shape.
  The scenario resolver step (`apply_scenario_*`, faithful to Python's
  `ScenarioResolver`) then mutates **typed fields** instead of assembling a
  `serde_json::Map` of string keys.

### 2. `BenchmarkRun` is the sole runner vocabulary

`EnvelopeV2 { run: BenchmarkRun, … }` is decoded with plain
`serde_json::from_slice`; `deny_unknown_fields` on the typed model is the wire
strictness. `BenchmarkRunWireV2`, `AuthoredRunSpecV2`, `into_authored`, and
`NamedRunnerComponentSpecV2` are **deleted**. The child composition root
(`coordinator.rs`) drives the engine against `&BenchmarkConfig` and its typed
`resolved` facts, exactly as the Python services read `cfg`. `decode_execute_wire`
reduces to decoding one `EnvelopeV2`; the dual authoring/bare-run accept path is
subsumed because authoring `Inputs` already resolve to a `BenchmarkRun` before
the boundary (config-model-unification step 4).

### 3. Typed factory seam (string discriminant + typed config)

The refactor separates two axes that the current design and an earlier draft of
this record conflated:

- **The discriminant** (the component `type`/id) — **stays an open, normalized
  string**, not an enum. It is `RegistryId` (`extensions/registry_id.rs`): a
  `#[serde(transparent)]` newtype whose custom `Deserialize` already normalizes
  `trim().to_ascii_lowercase().replace('-', "_")` (Python `_normalize_name`) and
  validates non-empty, with `Borrow<str>` for map lookup. A discriminant is
  inherently open (plugins register ids at bootstrap) and can never be an
  exhaustive enum, so a string id is the honest representation — it is exactly what
  Python's `ExtensibleStrEnum` *is* (a `str` subclass with a convenience
  known-values layer). This drops the manual-`Deserialize`/`untagged`-tail
  machinery entirely (see below).
- **The config payload** — **becomes a typed struct**, always. This is the whole
  prize (parse-don't-validate, decode-time errors, no `into_authored`
  `Value`-shuffling) and it is cheap: a built-in config is one
  `#[derive(Deserialize)] #[serde(deny_unknown_fields)]` struct.

A component on the wire is `{ "type": <RegistryId>, "config": { … } }` — the
config nested under its own key, which is the shape
`NamedRunnerComponentSpecV2 { id, config }` already emits today (an earlier draft
of this record described the built-in fields as flat; they are not). Decoding is
a plain derive plus a match on the id — **no hand-written enum `Deserialize`**:

```rust
struct ComponentSpec { #[serde(rename = "type")] id: RegistryId, config: Box<RawValue> }

fn decode_transport(spec: ComponentSpec, reg: &Registry) -> Result<TransportBinding> {
    match spec.id.as_str() {
        "http"    => HttpConfig::try_from_raw(&spec.config)?.into(),   // typed, deny_unknown_fields, try_from invariants
        "grpc"    => GrpcConfig::try_from_raw(&spec.config)?.into(),
        "dry_run" => DryRunConfig::try_from_raw(&spec.config)?.into(),
        _         => reg.get(&spec.id)?.validate(&spec.config)?.into(), // plugin tail: frozen-registry dyn factory
    }
}
```

Why a string id and not an extensible enum: an **empirical serde test
(2026-07-26)** showed the enum encoding (`#[serde(tag="type")]` + a derived
`#[serde(untagged)]` `RawValue` tail) *compiles but misbehaves* — unknown tags
fail (`RawValue` can't decode from serde's buffered `Content`), the untagged arm
matches structurally on literal `id`/`config` keys, and a built-in failing
`deny_unknown_fields` silently mis-routes into the tail. Making the enum work
requires a fragile hand-written `Deserialize` (buffer → peek `type` → strip →
dispatch). The `RegistryId`-string discriminant needs none of that: the id
deserializes with a plain derive, and the `match` above is ordinary code. The
built-in arms still produce typed configs, so downstream drivers get
`HttpConfig`/`GrpcConfig` typed; the in-memory container for that heterogeneous
result (a small internal `enum`, or `Box<dyn Validated…>`) is a downstream
ergonomics choice the **wire never sees**.

- **Built-in configs** are typed structs that fold their per-variant semantic
  invariants (dry_run finiteness, dynosim gating, HTTP's control-plane gate) into
  the parse via `#[serde(try_from = "…")]` — parse-don't-validate, no separate
  post-decode `validate()`.
- **The plugin tail** keeps `Box<RawValue>` decoded by the plugin's own `dyn`
  factory. Because the registry **freezes at bootstrap**, the id → factory lookup
  is total and immutable for the run, resolved once at prepare. This is the
  load-bearing use of the `RawValue`/`dyn` boundary; it is not applied to
  built-ins.
- **Closed core knobs stay enums.** `DispatchMode` and `HopRouting` are *not*
  plugin categories — there is no tail — so exhaustiveness and typo-catch are pure
  wins. They remain `#[derive(Deserialize)]` enums, but routed through the same
  `normalize_ident` seam (fixing a latent bug: they use `rename_all` today and do
  **not** normalize, so `global_hop`/`GLOBAL-HOP` currently fail). Rule of thumb:
  **closed set → enum; open/plugin set → `RegistryId`.**

`WorkloadRequirements`, `RunContext`, and the transport-swap principle are
preserved — the match keys on the id string instead of an enum variant, nothing
else changes.

This is the faithful Rust shape of Python's plugin architecture. In origin/main
every plugin category (`transport`, `endpoint`, `dataset_sampler`, …) declares its
discriminant as an **`ExtensibleStrEnum`** — a `str` with base members plus a
runtime `_extensions` dict, normalized via `_normalize_name`; the per-variant
config is a `PluginEntry.metadata: dict[str, Any]` validated on demand against the
category's `metadata_class` Pydantic model (`get_typed_metadata`). `RegistryId` is
the `str` discriminant; the typed built-in struct / plugin-`validate(&RawValue)`
is the `metadata` + `metadata_class` decode. Two rules follow:

- **Normalization is one shared seam.** `RegistryId`'s custom `Deserialize` already
  is `_normalize_name`; a shared `normalize_ident` + a small `macro_rules!` applies
  the same fold to the closed knob enums and to clap's `value_parser`, so the CLI
  and the wire accept byte-identical vocabulary. The fold is **lower + `-`→`_`
  only** — Python does not strip separators, so `graphir` is a hardcoded alias,
  not normalization.
- **Do not port the lazy-import layer.** Python's `PluginEntry.load()` /
  `class_path` / `importlib` / AST-`validate` is Python resolving *implementation
  code by string at runtime*. Rust built-ins are compiled in; real Rust plugins
  load through a `.so`/wasm host loader. Port the *type model* (`ExtensibleStrEnum`
  → `RegistryId` + typed configs), not the loader.

Per-category vocabulary is *not* a 1:1 wire port of Python's values (a parity
audit found endpoints match exactly, but Rust's `transport` set adds
`grpc`/`dynosim_*`/`dry_run` with no Python `TransportType` plugin, phases use
`PhaseKind` not `TimingMode`, and the dataset trace path is `file` not `custom`).
That is fine — the *mechanism* is the port, the *values* are Rust-native where the
runtimes differ.

### 4. Registry role after the change

`AIPerfRegistry`/`AIPerfExtension` shrinks from "transactional registry of
config-decoding `dyn` factories" to "descriptor catalog + executor provider": it
still backs `--capabilities` output and any genuinely `dyn` executor seams, but
it no longer owns component-config decode or `ComponentId` → factory lookup for
selection. [extension-registry.md](extension-registry.md) is updated in the same
change to describe the reduced surface; the frozen-at-bootstrap guarantee is
unchanged (a built-in id resolves in the `match`'s typed arm; an unknown id falls
to the plugin tail and fails closed at registry lookup, as today).
Diagnostic-parity caveat: the helpful "available: …" list comes from registry
lookup, which the plugin tail still performs — but the built-in arms bypass it,
so a typo'd built-in id must still produce that same list from the `match`'s
default arm rather than a bare decode error. Only **transports
and workloads** carry the `RawValue`-per-factory config-decode role; endpoints,
samplers, exporters, and actuators are already typed or name-keyed and are
untouched. The `--capabilities` catalog is unaffected — `Catalog::from_registry`
reads only `&'static` descriptors, never `validate()`.

### 5. Migration

One transport family at a time, byte-exact against the mock server at each step:

1. Introduce typed built-in configs decoded per id (`match id.as_str()`)
   alongside the existing `RawValue` path; `into_authored` populates them from the
   typed `cfg` (no behavior change). The discriminant is `RegistryId`; no enum for
   the tag.
2. Move `native_execution` selection to the id `match` for built-ins. Two
   obligations the audit surfaced: (a) the registry resolves not just config but
   the transport's **`NativeTransportExecution` binding** (`resolve_native_execution`
   → `transport_factory(id).native_execution(...)`), so the built-in `match` arms
   must supply those bindings directly or every workload's prepare/`validate_run`
   breaks with "transport not registered"; (b) `--capabilities` builds its catalog
   from registered factory **descriptors**, so built-ins must still contribute
   their `TransportDescriptor`/`WorkloadDescriptor`; (c) a **third** consumer
   outside the profile path resolves transports by id —
   `CurrentNativeGraphModelBindingResolver::resolve`
   (`rust/runtime/src/eval/native_graph/model_runtime.rs:94`, with further lookups
   at `:309` and `:389`) calls
   `registry.transport_factory(binding.transport_factory_id())` for its
   `UnknownTransport` rejection and reads `transport.descriptor().url_schemes` for
   `validate_transport_urls`. That is the `aiperf eval --model-runtime`
   native-graph path, which the http + grpc **profile** e2e suites do not
   exercise; it needs its own verification, or built-ins must keep contributing a
   registry entry for descriptor lookup even after selection leaves the registry.
   Keep the `id → factory` lookup for the plugin tail; only built-ins leave it.
   Verify with the http + grpc e2e suites **and** the native-graph eval path.
3. Repeat for the workload seam; collapse `ScheduledWorkloadConfigV2` /
   `GraphWorkloadConfigV2` into typed-optional fields — all **five** graph-only
   fields, including the `recorded_agent_default` derivation and the
   `weka_semantics`-conditional `system_idle_gap_cap_seconds` attachment, whose
   conditionality is behavioral (`online_execution.rs:1251` branches on
   `workload.recorded_agent_default`). Verify graph + scheduled e2e.
4. Delete `AuthoredRunSpecV2`, `into_authored`, `NamedRunnerComponentSpecV2`,
   `BenchmarkRunWireV2`; point `coordinator.rs` at `BenchmarkConfig`. The
   protocol-v2 request/response module is reduced to the `EnvelopeV2` outer shape
   and the diagnostic/result types. **Corrected (contest round 2):** `into_authored`
   injects data that is **not** in `BenchmarkConfig`. `planned_replay_traces`
   lives on `BenchmarkRunWireV2` (`protocol_v2.rs:325`), is written by the
   controller at `cellular_controller.rs:1978`, copied into
   `GraphWorkloadConfigV2` (`registry.rs:904`) by `into_authored`
   (`protocol_v2.rs:413-415`), and consumed at `entrypoints.rs:415` as
   `expected_replay_traces`. It appears **nowhere** in `runtime/src/config/` or
   `cli/src/`. Deleting `BenchmarkRunWireV2` therefore requires first giving this
   field a home on `BenchmarkRun` — a run-level fact alongside `cfg`, not a
   `BenchmarkConfig` field, since it is controller-derived rather than authored —
   or cellular graph replay loses its trace expectation. Every *other* field is a
   copy or a pure derivation (`workload_kind`, `parse_dispatch_mode`,
   `worker_count` from `available_parallelism`).
   The `validate_run` seam is also **not one body**: "consumes only `models.items`
   and `sidecars.live_streaming`" holds for `online_execution.rs:105` alone.
   `ws_execution.rs:151` additionally consumes `run.artifacts.trace` and five
   sidecar fields (rejecting trace artifacts and all sidecars), and
   `dry_run.rs:539` consumes `run.dispatch` and `run.workload.id.as_str()`
   (rejecting sharded dispatch and graph workloads under virtual workers). Each
   body repoints on its own terms.
   Two further hazards remain: (a) today's projection performs **lossy endpoint/model
   transforms** (rename `timeout` → `timeout_seconds`, drop `url_strategy`, retain
   only `name`/`weight`) that the migration must reproduce or push into the
   endpoint/model factories; (b) **`resource_presence`** is *not* a naive
   `cfg.field.is_some()` map — `into_authored` hardcodes a present/absent
   classification that drives `validate_resource_requirements`' Required/Optional/
   **Forbidden** matrix, so the repoint must reconstruct that exact classification
   explicitly or the Forbidden checks change behavior. A naive "just pass `cfg`"
   would regress both.

Each step keeps the stdin protocol wire-compatible; only the internal projection
is removed. **The boundary to hold fixed is not `EnvelopeV2`.** `EnvelopeV2`
(`protocol_v2.rs:118`) is never deserialized from stdin — it is constructed
in-process at `cli/src/execute_mode.rs:477` after decode, and its own doc comment
says it is "reconstructed around the bare `BenchmarkRunWireV2` stdin payload".
The actual stdin contract is what `decode_execute_wire` (`protocol_v2.rs:191`)
accepts: an `AuthoringWireV2` (`{"authoring": <Inputs>, sweep_id, variation,
trial}`, `protocol_v2.rs:155`) or a bare `BenchmarkRunWireV2`, discriminated by
presence of the `authoring` key (`resolved_run_bytes`, `protocol_v2.rs:207`). The
authoring arm is the one AIPerf itself writes; the bare-resolved-run arm is
retained for external harnesses. Step 4's deletion of `BenchmarkRunWireV2`
therefore *does* touch the stdin contract — the bare arm must either be re-typed
to `BenchmarkRun` or dropped with a stated compatibility break. §2's
"`EnvelopeV2 { run: BenchmarkRun, … }` is decoded with plain
`serde_json::from_slice`" describes a boundary that does not exist today.

## Non-goals and trade-offs

This is a correctness/type-safety refactor, deliberately made with eyes open:

- **We reintroduce a match on component id.** The current code prides itself on
  "never matching on a transport kind" via `dyn`. The typed design matches on the
  `RegistryId` string exactly once, at the selection boundary, with a plugin-tail
  fallthrough — not an exhaustive enum match (the open set forbids that), just an
  ordinary `match id.as_str()` whose default arm defers to the frozen registry.
- **The runtime crate gains a compile dependency on every built-in component
  config type.** That is the cost of typing built-in configs. It is acceptable
  because the built-in set is frozen at compile time; plugin configs stay opaque
  (`RawValue`), so the coupling does not extend to the open tail.
- **Not touched:** the `dispatch` seam (`Dispatchable`/`RequestSink`/
  `RequestObserver`), the `Clock` seam, the phase/scheduling runtime, metrics,
  exporters' output logic, and the outer stdio `EnvelopeV2` contract. This change
  is about *config decode and component selection*, not the hot path.
- **Dynamic plugins are a planned future, and this design accommodates them.**
  AIPerf may grow runtime-loaded transports/workloads (WASM / subprocess /
  `abi_stable`). Those cannot be typed configs (their type is unknown at the
  core's compile time), so the plugin tail's `RawValue` config (keyed by
  `RegistryId`, decoded by the plugin's own `dyn` factory against the
  bootstrap-frozen registry) is retained precisely for them. The refactor does
  **not** close the set; it types built-in configs and confines the opaque seam to
  the open tail. This supersedes an earlier framing of this record that scoped
  dynamic plugins out — they are in scope, via the string id + `RawValue` tail.
- **"Zero `RawValue`" is a direction, not a literal end state.** Three residual
  `RawValue` uses are legitimate and stay: the plugin tail (the load-bearing one —
  a runtime-loaded plugin's config is opaque to the host by definition); the `dynosim` transport variant's nested Dynamo engine/router args
  (opaque pass-throughs to Dynamo's own parser); and the dataset payload inside
  the workload arm (adapter input; dataset selection is name/structural-probe, not
  a component-config union). This change types the built-in component majority and
  the first-party inner seams; it does not chase `RawValue` out of the places it
  belongs.

## Source anchors

- `rust/runtime/src/engine/protocol_v2.rs` — `BenchmarkRunWireV2`,
  `AuthoredRunSpecV2`, `into_authored`, `NamedRunnerComponentSpecV2`,
  `ScheduledWorkloadConfigV2`/`GraphWorkloadConfigV2` (the projection to delete);
  `EnvelopeV2` (the outer shape to keep).
- `rust/runtime/src/engine/coordinator.rs` — `envelope.run.into_authored()`, the
  child composition root to repoint at `BenchmarkConfig`.
- `rust/runtime/src/engine/registry.rs` — `TransportFactory`/`WorkloadFactory`,
  `ValidatedTransportConfig`/`ValidatedWorkloadConfig`, `WorkloadRequirements`,
  `native_execution` (the factory seam to make typed).
- `rust/runtime/src/extensions/mod.rs` — `AIPerfRegistry` capability accessors
  (the registry surface that shrinks).
- `rust/runtime/src/config/model/` and `rust/runtime/src/config/resolve.rs` — the
  typed `BenchmarkConfig`/`BenchmarkRun` and resolver step that gains the typed
  `Transport` union and typed `synthesis`/`weka_semantics`.
- `docs/specs/config-model-unification.md`, `docs/specs/runner-protocol.md`,
  `docs/specs/extension-registry.md` — the records this one completes and amends.
- Python reference (origin/main): `src/aiperf/plugin/extensible_enums.py`
  (`ExtensibleStrEnum` — the discriminant blueprint: base members + runtime
  `_extensions`, `_normalize_name` lookup), `src/aiperf/plugin/categories.yaml`
  (per-category `protocol`/`enum`/`metadata_class`), `src/aiperf/plugin/types.py`
  (`PluginEntry` — lazy `class_path` load + `metadata: dict` validated via
  `get_typed_metadata`; the layer NOT ported).
</parameter>
</invoke>
