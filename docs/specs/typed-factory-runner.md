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

**Status (contest O8, proven).** This record is *partly* built, not
forward-looking. Implementation lives on `ajc/typed-factory-runner-v2`:
`b7619602fb` already ships §5 step 3's DTO collapse as the unified
`WorkloadConfigV2` (`engine/registry.rs:876`, `:883`), and step 1's typed
`cfg.transport` consumer is `transport_component`
(`engine/protocol_v2.rs`), pinned against the projection it replaces by
`transport_component_matches_inline_projection`. Steps 2 and 4 are not built.
Read every "will" below as "will, where not already noted as landed" — an
implementer who treats the whole record as unstarted will redo step 3.

**Scope premise: the Rust tree is greenfield.** There is no released native
protocol-v2 stdin contract and no external consumer of it to preserve. The
`--execute` boundary is written and read by the same binary, and the
"bare resolved-run" arm exists because the code grew that way, not because a
published contract requires it. Nothing in this record is constrained by wire
compatibility, and no step needs a compatibility shim. Where a difference between
today's shapes matters, it matters because a *behavior* would be lost — strictness,
a validation check, a controller-derived field — and the migration must port that
behavior forward on its own merits. "Byte-exact against the mock server" below is
therefore an assertion about **observed run output**, never about stdin bytes.

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

- `transport` — **this field already exists and is already typed.**
  `BenchmarkConfig.transport: Option<Transport>` (`config/model/config.rs:75`)
  where `Transport` (`config/model/transport.rs:18`) is a **closed**
  `#[serde(tag = "type", rename_all = "snake_case")]` enum with variants `Http`,
  `Grpc`, `DynosimOffline(DynosimConfig)`, `DynosimOnline(DynosimConfig)`,
  `DryRun(DryRunConfig)`, `Websocket(WebSocketTransportConfig)`. Nothing needs to
  be introduced here. Two consequences the earlier drafts of this record missed:

  - **`Http` and `Grpc` are unit variants** — they carry no config payload at
    all. "Introduce typed built-in configs" is therefore *vacuous* for the two
    transports §5 names as its verification vehicle; the only transports with a
    payload to type are dynosim, dry_run, and websocket, and those are already
    typed too.
  - **The projection is pure loss.** `into_authored` (`protocol_v2.rs:369-371`)
    takes this typed enum, calls `serde_json::to_value(&cfg.transport)`, and
    feeds the result to `component_from_inline` to manufacture a
    `NamedRunnerComponentSpecV2 { id, config: Box<RawValue> }` that a factory then
    re-decodes. Typed → `Value` → `RawValue` → typed. Deleting the projection
    means reading `cfg.transport` and matching it; there is no decode to design.

  This forces a fork §5 must choose before step 1, because `Transport` is
  **closed** and therefore cannot express a plugin transport:

  - **(A) Keep `Transport` closed.** An exhaustive `match` on the enum is the
    honest dispatch, no `RegistryId` string and no plugin tail are involved for
    this field, and §3's apparatus does not apply to it. Cost: plugin transports
    remain inexpressible in an authored config — which is already true today.
  - **(B) Open `Transport` to a plugin tail.** Then it becomes something like
    `TransportSpec { id: RegistryId, config: TransportConfig }` with a
    `Plugin(Box<RawValue>)` arm. This is a **change to the authored config wire**
    and to `Inputs`/`resolve` in `aiperf-cli`, not a removal of an internal
    projection, and it is materially larger than anything §5 scopes.

  **This record selects (A).** Opening the transport set is a separate change
  with its own authoring-surface and Config-v2 schema consequences; it is not a
  prerequisite for deleting `AuthoredRunSpecV2`, and bundling it would make an
  already-delicate migration unshippable. §3's open-`RegistryId` design remains
  correct for the *runner component seam* it was written about — a different
  layer, and the one being deleted — but it is **not** the design of
  `BenchmarkConfig.transport`.
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

`decode_execute_wire` (`protocol_v2.rs:191`) remains the stdin contract and keeps
**both** arms: the `AuthoringWireV2` arm AIPerf itself writes and the bare
resolved-run arm retained for external harnesses. What changes is the *type* of
the bare arm. `EnvelopeV2` stays exactly what it is today — a struct constructed
in-process at `cli/src/execute_mode.rs:477` *after* decode, never deserialized
from stdin. **There is no `serde_json::from_slice::<EnvelopeV2>` in this design**;
an earlier draft of this record required one, and it described a boundary that
does not exist (corrected, contest round 2, and restated normatively here in
round 4 rather than left as an appended footnote).

`AuthoredRunSpecV2`, `into_authored`, and `NamedRunnerComponentSpecV2` are
**deleted**. The child composition root (`coordinator.rs`) drives the engine
against **`&BenchmarkRun`**, reading authored sections through `run.cfg` and
run-level facts off `run` itself, exactly as the Python services read `cfg` out
of the run they were handed.

**Corrected (contest O4, proven).** Earlier drafts of this record said
"`&BenchmarkConfig`" here and again in §5 step 4, while the port list two
paragraphs below places `benchmark_id`, `artifact_dir`, `planned_replay_traces`,
`trial`, `variation`, `resolved`, and `variables` on `BenchmarkRun`. Those are
run-level facts with no `BenchmarkConfig` home at all (`config/model/run.rs:17-48`
vs `config/model/config.rs`), so the two statements could not both hold: an engine
handed `&BenchmarkConfig` cannot see `validate()`'s `benchmark_id`/`artifact_dir`
checks, cannot reach `planned_replay_traces`, and cannot resolve the export stem
derivation's inputs. **`BenchmarkRun` is the runner vocabulary throughout this
record** — that is what §2's title says and what the port list requires.
`BenchmarkConfig` remains the *authored* vocabulary reached as `run.cfg`, which is
what §1's typed component unions live on. Any remaining "point the engine at
`BenchmarkConfig`" phrasing is an error against this paragraph.

The bare arm is re-typed from `BenchmarkRunWireV2` to `BenchmarkRun` and
`BenchmarkRunWireV2` is **deleted**. Because the tree is greenfield (see Purpose),
this is not a compatibility question — but it is not a free rename either. The two
types differ on every axis that matters, and each difference is a behavior that
would silently vanish if the swap were done naively (verified against the tree,
contest round 4):

| | `BenchmarkRunWireV2` (`protocol_v2.rs:293-329`) | `BenchmarkRun` (`config/model/run.rs:17-48`) |
|---|---|---|
| strictness | `#[serde(deny_unknown_fields)]` | none — unknown top-level fields accepted |
| `resolved` | `serde_json::Value` (open) | typed `Resolved` |
| `variation` | `Option<VariationSpec>` | `Option<serde_json::Value>` |
| `trial` | `usize` | `u32` |
| `variables` | `BTreeMap<String, Value>` | `serde_json::Map<String, Value>` |
| `planned_replay_traces` | `BTreeSet<PlannedReplayTraceInstance>` | **absent** (see §5 step 4) |
| outer validation | `validate_outer()` (`protocol_v2.rs:332-350`): non-empty `benchmark_id`, non-empty `artifact_dir`, **non-empty** `datasets` (see O1 — the message says "exactly one" but the `ensure!` only tests `!datasets.is_empty()`) | **none** |
| export stem derivation | `into_authored` (`protocol_v2.rs:462-475`) rewrites `export.genai_perf.stem` and `export.timeslice.stem` from `artifacts.records_path` | **none** — `cfg.export` carries the undelivered authored value (see O2) |

The table is therefore a **port list**, not a compatibility obligation. Each row
is decided on merit:

- `deny_unknown_fields` — **port it.** `BenchmarkRun` gains
  `#[serde(deny_unknown_fields)]`. Strict decode is the stated goal of this whole
  arc (§2's opening claim that "`deny_unknown_fields` on the typed model is the
  wire strictness" is only true once the attribute is actually there); dropping it
  would make the migration a strictness *regression* in the name of typing.
- `validate_outer()` — **port it, but read it before porting it.** Two of its
  three checks are what they look like: non-empty `benchmark_id`, non-empty
  `artifact_dir`. The third is not. **Corrected (contest O1, proven):** its
  message reads `"run.cfg.datasets must contain exactly one dataset"`, but the
  `ensure!` it guards tests only
  `self.cfg.datasets.as_ref().is_some_and(|datasets| !datasets.is_empty())`
  (`protocol_v2.rs:342-348`). A two-dataset run passes, and `into_authored` then
  silently discards the tail — `cfg.datasets.and_then(|d| d.into_iter().next())`
  at `protocol_v2.rs:359-362`. So an earlier draft of this record asked
  `BenchmarkRun::validate()` to enforce a cardinality the code never enforced,
  which would have been a **behavior change smuggled in under the word "port"**:
  configs that run today would start failing to decode.

  Both readings are defensible and the choice must be explicit, not incidental:

  - **Port the check verbatim** (`!datasets.is_empty()`) and fix only the
    misleading message. Zero behavior change; silent truncation survives.
  - **Enforce real cardinality** (`datasets.len() == 1`). The tree is greenfield,
    silently dropping an authored dataset is a bug not a feature, and the existing
    message already documents the intended contract.

  This record selects **enforce real cardinality**, on the greenfield premise —
  but it is a behavior change and is recorded as one. It carries two obligations:
  the step-4 test list gains a case asserting a two-dataset config is *rejected*
  with a message naming the count, and the `ensure!` message stops lying. Neither
  the current check nor `into_authored`'s `.next()` may be deleted before that
  test exists, or the truncation moves from silent-and-tested-nowhere to
  silent-and-unreachable.
- `planned_replay_traces` — **port it**, as a run-level field on `BenchmarkRun`
  alongside `cfg`, not a `BenchmarkConfig` field, because it is controller-derived
  rather than authored (see §5 step 4 and O1).
- `resolved: Value` → `Resolved`, `variation: VariationSpec` → `Value`,
  `trial: usize` → `u32`, `variables: BTreeMap` → `serde_json::Map` — **accept
  `BenchmarkRun`'s shapes.** These are the typed model winning, which is the
  point. The one to check rather than assume is `variation`: `BenchmarkRun` holds
  it as an open `Value` where the wire DTO had a typed `VariationSpec`, so
  re-typing `BenchmarkRun::variation` to `VariationSpec` is the correct direction
  and belongs in this change rather than being inherited as-is.

- **Export stem derivation — port it. Added by contest O2 (proven).** The port
  list above was drawn from `BenchmarkRunWireV2`'s *fields* and so missed a
  behavior that lives only in `into_authored`'s **body**: at
  `protocol_v2.rs:462-475` it reads `artifacts_spec.records_path`, strips a
  `.jsonl` suffix from the file name, and — when the stem is non-empty —
  overwrites `export_cfg.genai_perf.stem` with it and
  `export_cfg.timeslice.stem` with `format!("{stem}_aiperf")`. This is the live
  implementation of `--profile-export-prefix` / `artifacts.prefix` for the
  summary and timeslice outputs; `cfg.export` itself never carries the derived
  value. Pointing execution at typed `cfg.export` without reproducing this
  cross-field transform does not fail — it silently reverts the summary and
  timeslice files to the default `profile_export_aiperf.{json,csv}` and
  `profile_export_aiperf_timeslices.*` names while per-record output stays under
  the authored custom stem, i.e. one run emitting two different prefixes. It must
  become an inherent derivation on the typed model (alongside
  `BenchmarkRun::validate()`, applied at the same boundary), and step 4's test
  list gains a case asserting that a run with a custom prefix names the
  per-record, summary, **and** timeslice artifacts from the same stem. This entry
  also generalizes: the remaining step-4 audit must walk `into_authored`'s body
  for other field-to-field transforms, not just its struct definition.

What step 4 deletes is both the *projection* (`into_authored`,
`AuthoredRunSpecV2`, `NamedRunnerComponentSpecV2`) and the *wire DTO*
(`BenchmarkRunWireV2`), with the four ported behaviors above landing on
`BenchmarkRun` first.

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

**Scope of this section.** Everything below describes the *runner component
seam* — `NamedRunnerComponentSpecV2 { id, config }`, the structure §5 step 4
deletes — and any future genuinely-open component category. It is **not** the
design of `BenchmarkConfig.transport`, which is already the closed typed
`Transport` enum (see §1) and which this record does not open. Where the two
appeared to conflict in earlier drafts, §1 governs the config model and this
section governs the seam.

A seam component on the wire is `{ "type": <RegistryId>, "config": { … } }` — the
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
   typed `cfg` (no behavior change). **Corrected (round 3):** for transports there
   is nothing to introduce — `cfg.transport` is already the closed typed
   `Transport` enum, and `Http`/`Grpc` are unit variants with no payload. Step 1
   for the transport family is therefore *not* "add typed configs" but "add a
   consumer that reads `cfg.transport` directly", run alongside the existing
   projection, and assert the two produce identical bindings. Dispatch is an
   exhaustive `match` on `Transport`, not `match id.as_str()`; the `RegistryId`
   string and the plugin tail belong to the seam in §3, not to this field.
2. Move `native_execution` selection to the exhaustive `Transport` match for
   built-ins. Three obligations the audits surfaced: (a) the registry resolves not
   just config but
   the transport's **`NativeTransportExecution` binding** (`resolve_native_execution`
   → `transport_factory(id).native_execution(...)`), so the match arms
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
   Keep the `id → factory` lookup for the §3 seam; only `cfg.transport` selection
   leaves it. Verify with the http + grpc e2e suites **and** the native-graph eval
   path. Note that http and grpc alone do **not** exercise a payload-bearing
   transport arm, so dry_run (`rust/dry-run-tests`) must be in the same gate.
   **(d) Added while implementing step 2:** two doc comments on the very seam
   this step narrows assert the opposite property, and both become false the
   moment the match lands. `registry.rs:215` calls `native_execution` "the seam
   that makes a transport *swappable*: the workload asks the registered
   transport for its execution binding … never matching on a transport kind",
   and `registry.rs:236` states "There is no `match` on a closed transport enum:
   adding a native transport means registering a factory that returns its own
   binding, and nothing in the workloads changes." Step 2 must rewrite both.
   That rewrite is not a concession, because the openness they describe is
   already unreachable from authored config: `Transport`
   (`config/model/transport.rs:16-18`) is `#[serde(tag = "type")]` over six
   variants, so an out-of-tree `AIPerfExtension` may still call the public
   `register_transport` (`registry.rs:530`) with a new id, but no Config v2
   document can select it — `cfg.transport` fails to decode on an unknown tag.
   Registry openness is therefore real only for the *id-addressed* consumers,
   which are exactly obligations (b) and (c), and both already require the
   migration to keep those entries. Verified one-to-one correspondence today:
   every in-tree registration (`registry.rs:735`, `:770`, `:788`,
   `dry_run.rs:582`, `offline_execution.rs:851-852`) registers exactly one of
   the six `Transport` variants, so step 2 opens no selection gap. What it does
   introduce is a standing duty to keep the match arms and the registry in
   correspondence, and the compiler enforces only the match side — a factory
   registered under an id with no arm fails at run time, not at build time.

   **Corrected (contest O3, proven): the correspondence is not one-to-one in a
   lean build.** `Transport` is a plain `#[serde(tag = "type")]` enum with **no**
   `#[cfg]` on any variant (`config/model/transport.rs:16-49`), so all six always
   deserialize. Their factories do not always register: the registrations are
   feature-gated throughout `registry.rs` (`:758`, `:776`, `:795`, `:838`,
   `:1484`, `:1494`, `:1566`, `:1598`, `:2142`, `:2190`, `:2212`) — `Grpc` behind
   `grpc`, `Websocket` behind its feature, `DynosimOffline`/`DynosimOnline` behind
   `dynosim`. The verified correspondence quoted above holds only for a build with
   every feature on. A lean build therefore has variants that decode and select an
   id nothing registered, and today's failure for that case is a registry lookup
   miss at run time.

   This is not a reason to reject step 2 — it is a reason step 2 improves on the
   status quo, provided it is written to. The exhaustive match must carry
   **explicit feature-gated rejection arms**, not a silent fallthrough: each of
   the four gated variants gets a `#[cfg(not(feature = "…"))]` arm returning a
   named error ("transport `grpc` selected but this binary was built without the
   `grpc` feature"), so the diagnostic names the build rather than the registry.
   Step 2's gate gains a lean-build compile check —
   `cargo check -p aiperf-cli --no-default-features` plus the feature-bearing
   builds already in CLAUDE.md — because a match written against the full feature
   set is exactly the code that fails to compile, or silently falls through, when
   features are subtracted. Obligation (d)'s "one-to-one correspondence" claim is
   restated as: one-to-one **per compiled feature set**, with the uncompiled
   remainder converted from a run-time lookup miss into a build-named refusal.
3. Repeat for the workload seam; collapse `ScheduledWorkloadConfigV2` /
   `GraphWorkloadConfigV2` into typed-optional fields — all **five** graph-only
   fields, including the `recorded_agent_default` derivation and the
   `weka_semantics`-conditional `system_idle_gap_cap_seconds` attachment. All
   four graph-only fields that survive the DTO are consumed inside `lower_graph`
   (`online_execution.rs:1215`), and dropping any is a **silent** behavior loss,
   not a decode error: `workload.recorded_agent_default` gates
   `validate_canonical_recorded_agent_bundle(&prepared.bundle)` (canonical-bundle
   validation simply stops running if the flag is lost);
   `workload.system_idle_gap_cap_seconds` and `workload.ignore_trace_delays` flow
   into `NativeGraphDatasetPlan`; `workload.planned_replay_traces` is assigned to
   `plan.planned_replay_traces` after `build_common_plan`. Note also that
   `lower_graph` takes `&AuthoredRunSpecV2` and reads `run.endpoints.identities()`
   and `run.identity.random_seed`, and passes `run` on to `build_common_plan` —
   further step-4 repoint surface. Verify graph + scheduled e2e.
4. Delete `AuthoredRunSpecV2`, `into_authored`, `NamedRunnerComponentSpecV2`, and
   `BenchmarkRunWireV2`; point `coordinator.rs` at **`BenchmarkRun`** (per O4 —
   *not* `BenchmarkConfig`, which cannot carry the run-level facts this same step
   ports). The tree is
   greenfield, so the bare stdin arm is simply re-typed to `BenchmarkRun` — but
   only *after* §2's port list lands on `BenchmarkRun`:
   `#[serde(deny_unknown_fields)]`, an inherent `validate()` carrying
   `validate_outer`'s three checks (with the dataset check tightened to real
   cardinality per O1, and its message corrected), the export-stem derivation
   from `artifacts.records_path` per O2, `planned_replay_traces` as a run-level
   field, and `variation: Option<VariationSpec>`. Deleting the DTO before those land is
   the silent-regression path. The
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
   The `validate_run` seam is also **not one body — and not even one trait**
   (corrected again, contest round 4; the round-2 inventory was itself the same
   single-call-site generalization it criticized). There are **two** traits:
   `NativeTransportExecution::validate_run(&self, run, context)` (transport-level,
   2-arg) and `WorkloadFactory::validate_run(&self, run, context, transport,
   workload, transport_id)` (workload-level, 5-arg, `registry.rs:293`). The
   verified inventory is seven-plus bodies:
   - **Transport-level.** `online_execution.rs:105` (http) is the *only* body the
     original "consumes only `models.items` and `sidecars.live_streaming`"
     inventory described. `grpc_execution.rs:73` → `validate_grpc_run` (`:85`)
     consumes `context.default_endpoint_profile()`, `context.endpoint_profiles()`,
     and `profile.config.urls`, and rejects **all** sidecars.
     `ws_execution.rs:151` consumes `run.artifacts.trace` and five sidecar fields
     (rejecting trace artifacts and all sidecars). `dry_run.rs:539` consumes
     `run.dispatch` and `run.workload.id.as_str()` (rejecting sharded dispatch and
     graph workloads under virtual workers).
   - **Workload-level.** `online_execution.rs:228` (scheduled) delegates to the
     transport binding, or falls through the `dynosim_or_unsupported!` macro
     (`online_execution.rs:135-151`) to
     `offline_execution::dynosim_scheduled_validate_run` (`:887`, which requires
     `workload.worker_count == 1`), then runs `validate_authored_tokenizer`.
     `online_execution.rs:324` (graph) is the body that actually consumes
     `run.sidecars.live_streaming`, at `:337` — round 2 mis-attributed that field
     to the transport level. `online_execution.rs:447` (static accuracy) requires
     `transport_id == "http"` and consumes `run.models.items.len()`.
   Each body repoints on its own terms. A repoint audited against
   `online_execution.rs:105` alone silently changes the gRPC, WebSocket, dry-run,
   dynosim, graph, and static-accuracy rejection surfaces.
   Two further hazards remain.

   **(a) Endpoint/model transforms — mostly no-ops, with one live exception**
   (corrected, contest round 4; an earlier draft called all three "lossy
   transforms the migration must reproduce", which overstated two of them).
   `endpoint_profile` (`protocol_v2.rs:552-563`) renames `timeout` →
   `timeout_seconds` and removes `url_strategy`; `models_from_config` (`:507-521`)
   retains only `name`/`weight`. Against the typed model these are no-ops:
   `Endpoint` (`config/model/endpoint.rs:117`) already stores `timeout_seconds`
   (`:137`) and has neither a `timeout` nor a `url_strategy` field — `url_strategy`
   exists only in the authoring layer (`cli/src/flags.rs:368`,
   `cli/src/yaml.rs:898`, consumed and validated at `yaml.rs:1673`) and never
   reaches `BenchmarkConfig` — and `ModelItem` (`config/model/models.rs:22`) has
   only `name` and `weight`. For the **default** profile
   (`serde_json::to_value(&cfg.endpoint)`, `protocol_v2.rs:446`) and for models,
   the migration may simply drop these transforms.
   They are **not** no-ops for the override profiles. `cfg.endpoint_profiles`
   (`config/model/config.rs:118`) is an open
   `serde_json::Map<String, serde_json::Value>`, not a typed section, and
   `into_authored` feeds it through the same `endpoint_profile` at
   `protocol_v2.rs:448-451`/`:547`. An authored override may therefore still carry
   `timeout` and `url_strategy` keys, and the rename/removal is live for it. The
   migration must keep the transform on that open map — or type `endpoint_profiles`
   as a `BTreeMap<String, Endpoint>`, which is a larger change than this record
   scopes.

   **(b) `resource_presence` is not a naive `cfg.field.is_some()` map.** The exact
   classification `into_authored` hardcodes (`protocol_v2.rs:496-501`) is
   `models: true`, `endpoints: true`, `metrics: true`, `artifacts: true` —
   *unconditionally*, regardless of whether the corresponding `Option` on
   `BenchmarkConfig` is `None` — plus `sidecars: sidecars_present`, where
   `sidecars_present` (`protocol_v2.rs:432-443`) is
   `serde_json::to_value(&cfg.sidecars).as_object().is_some_and(|o| !o.is_empty())`.
   That is emptiness of the *serialized object*, a categorically different
   predicate from `Option::is_some`: every field of `Sidecars`
   (`config/model/telemetry.rs:206-221`) carries
   `skip_serializing_if = "Option::is_none"`, so `Some(Sidecars::default())`
   serializes to `{}` and classifies as **absent**, while `cfg.sidecars.is_some()`
   would call it present. That flip changes the Required/Optional/**Forbidden**
   matrix in `validate_resource_requirements` — precisely the regression this
   paragraph exists to prevent, and precisely what a naive "just pass `cfg`"
   produces. Note also that the *other* construction path,
   `AuthoredRunSpecV2::deserialize` (`protocol_v2.rs:661-667`), already uses the
   naive `wire.resources.X.is_some()` form: the two paths disagree today, so the
   migration must pick one deliberately rather than inherit whichever it happens
   to touch first.

   **Selected (contest O5, proven): preserve `into_authored`'s algorithm**
   (`protocol_v2.rs:496-501` plus `sidecars_present` at `:432-443`) verbatim as an
   inherent derivation on the typed model. "Pick one deliberately" was not itself a
   decision, and the two candidates are not equivalent — the `Deserialize` impl's
   naive `wire.resources.X.is_some()` form (`:661-667`) differs on **five** of five
   entries, not just `sidecars`: it reports `models`/`endpoints`/`metrics`/
   `artifacts` as *absent* whenever the corresponding `Option` is `None`, where
   `into_authored` reports them present unconditionally. `into_authored` is the path
   every AIPerf-authored run actually takes (`decode_execute_wire`'s authoring arm),
   so it is the observed production behavior and the only one with e2e coverage;
   adopting the `Deserialize` form would move four resources from Optional to
   Forbidden/Required transitions in `validate_resource_requirements` for every run
   that omits a section. The `Deserialize` impl disappears with
   `AuthoredRunSpecV2` in this same step, so the divergence is closed by deletion
   rather than reconciled.

**Verification gates (executable; added contest round 4, which is when this
section first had any).** Every step runs, from `rust/` with the project venv
active:

```bash
cargo fmt --check && cargo clippy --all-targets
cargo test -p aiperf-runtime && cargo test -p aiperf-runtime --features engine
```

`cargo test -p aiperf-runtime` alone runs **zero** engine tests — the `engine`
feature gates the entire projection this record changes — so both invocations are
mandatory at every step, not just the last. Beyond that shared floor:

- **Step 1** (a `cfg.transport` consumer running alongside the projection): a
  temporary differential assertion that both paths produce identical bindings,
  plus `cargo test -p aiperf-e2e-tests --test test_default_behavior --test
  test_chat_endpoint --test test_completions_endpoint --test
  test_kserve_grpc_endpoint`.
- **Step 2** (selection moves to the exhaustive `Transport` match): the step-1
  gate, plus `cargo test -p aiperf-dry-run-tests --test dry_run --test
  virtual_workers` — the only payload-bearing built-in transport arm any suite
  exercises, since `Http` and `Grpc` are unit variants — plus `cargo test -p
  aiperf-e2e-tests --test test_websocket` for the `Websocket(WebSocketTransportConfig)`
  arm, plus `cargo test -p aiperf-e2e-tests --test test_harbor_native_graph_rollout`
  for obligation (c)'s `aiperf eval --model-runtime` path, which the profile
  suites do not touch.
- **Step 3** (workload seam; the five graph-only fields): the step-2 gate, plus
  `cargo test -p aiperf-e2e-tests --test test_conditional_graph --test
  test_flatgraph_parity --test test_ignore_trace_delays --test
  test_recorded_agent_replay --test test_dag_full_topology`.
  `test_ignore_trace_delays` is the named guard for `ignore_trace_delays`, but
  `recorded_agent_default` and `system_idle_gap_cap_seconds` have **no** dedicated
  e2e target today — and both are silent losses, not decode errors. Step 3 must
  therefore *add* two before it may be called green: a graph run asserting
  `validate_canonical_recorded_agent_bundle` still rejects a non-canonical bundle,
  and a `weka_semantics: legacy` run asserting the idle cap still applies.
- **Step 4** (delete the projection; repoint `coordinator.rs`) — this step
  previously carried **no** gate at all, which is exactly how it could have
  shipped while dropping controller-authored `planned_replay_traces`. Its gate is
  the full step-3 gate, plus the cellular suites that exercise that field and the
  bare-run stdin arm: `cargo test -p aiperf-e2e-tests --test test_cellular --test
  test_graph_cellular --test test_grpc_cellular --test
  test_recorded_agent_cellular --test test_cellular_dataset_shipping`. A step-4
  change that has not run `test_graph_cellular` and `test_recorded_agent_cellular`
  is not verified, regardless of what else is green.

  **Corrected (contest O6, proven): that gate exercises the repoint, not the port
  list.** Every suite named above is an end-to-end profile run; none of them
  asserts `deny_unknown_fields`, the cardinality tightening selected under O1, the
  export-stem derivation from O2, `variation: Option<VariationSpec>`, or the
  `resource_presence` algorithm selected under O5. A step-4 change could land all
  five wrong and still be green, which is the same shape of hole this section was
  written to close. Worse, the closest thing the tree has to such a test does not
  run: `protocol_v2.rs:1304` declares `#[cfg(any())] mod tests` — an
  always-false cfg that compiles the entire module out — and among its bodies is
  `outer_contract_rejects_unknown_fields`, i.e. the one existing unknown-field
  assertion has been silently disabled since `904cc07e2a`. Step 4 must therefore
  add **named unit tests on `BenchmarkRun` itself**, in addition to the e2e gate:

  - an unknown top-level field on a bare run payload is rejected (replacing the
    disabled `outer_contract_rejects_unknown_fields`, on the typed model);
  - `BenchmarkRun::validate()` rejects empty `benchmark_id`, empty `artifact_dir`,
    and a two-dataset `cfg.datasets`, the last with a message naming the count
    (O1's obligation);
  - a run whose `artifacts.records_path` carries a custom stem names the
    per-record, summary, and timeslice artifacts from that same stem (O2);
  - `variation` round-trips as a typed `VariationSpec`, not an open `Value`;
  - `resource_presence` for a config with `models`/`endpoints`/`metrics`/
    `artifacts` absent and `sidecars: Some(Sidecars::default())` matches
    `into_authored`'s classification — four `true` and `sidecars: false` (O5).

  Re-enabling or deleting `#[cfg(any())] mod tests` is part of step 4, not
  optional cleanup: leaving a disabled module next to the code it was written to
  guard is how the gap recurs.

Each step keeps the stdin accept path *structurally* intact — both arms of
`decode_execute_wire` survive; the bare arm changes type. No step owes anything to
an external consumer (see Purpose: greenfield), so the property each step must
hold is behavioral, not byte-level: strictness, validation, and controller-derived
state carried forward per §2's port list, and identical observed run output
against the mock server. **The boundary to hold fixed is not `EnvelopeV2`.** `EnvelopeV2`
(`protocol_v2.rs:118`) is never deserialized from stdin — it is constructed
in-process at `cli/src/execute_mode.rs:477` after decode, and its own doc comment
says it is "reconstructed around the bare `BenchmarkRunWireV2` stdin payload".
The actual stdin contract is what `decode_execute_wire` (`protocol_v2.rs:191`)
accepts: an `AuthoringWireV2` (`{"authoring": <Inputs>, sweep_id, variation,
trial}`, `protocol_v2.rs:155`) or a bare `BenchmarkRunWireV2`, discriminated by
presence of the `authoring` key (`resolved_run_bytes`, `protocol_v2.rs:207`). The
authoring arm is the one AIPerf itself writes; the bare-resolved-run arm is
retained for external harnesses. Deleting `BenchmarkRunWireV2` therefore changes the
bare arm's type. On a greenfield tree that is an ordinary refactor, not a
compatibility break — the obligation it creates is §2's port list, not a shim. §2 no longer claims
otherwise; the `serde_json::from_slice::<EnvelopeV2>` requirement that stood in
earlier drafts has been removed from §2 rather than merely footnoted here.

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
  exporters' output logic, and the stdio accept path (`decode_execute_wire`'s two
  arms — `EnvelopeV2` is an in-process struct built after decode, not a wire
  shape; see §2). This change
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

  **Corrected (contest O7, proven): as specified, the migration does not actually
  leave a selectable tail, and this must be fixed in the design rather than
  asserted away.** The claim above needs an authored object carrying
  `{ RegistryId, RawValue }` to survive. It does not: §1 selects transports from
  the closed `BenchmarkConfig.transport` enum and workloads from an exhaustive
  match, §3 identifies `NamedRunnerComponentSpecV2` as the `{ RegistryId,
  RawValue }` seam, and §5 step 4 deletes it. What remains after step 4 is a
  registry that still *accepts* `register_transport` with any id (`registry.rs:530`)
  and still resolves `transport_factory(id)` — but with no authored surface that
  can name one, because `cfg.transport` is `#[serde(tag = "type")]` over six
  variants and fails to decode on an unknown tag (the same reachability fact
  recorded as step 2's obligation (d)).

  So the honest current state is: **registry openness survives for id-addressed
  *internal* consumers, and authored selection of a runtime-loaded plugin does
  not exist either before or after this migration.** The non-goal is therefore
  restated as a *requirement on the eventual plugin work*, not a property this
  design delivers: whenever dynamic plugins land, `BenchmarkConfig.transport`
  (and the workload selector) must grow an explicit open tail variant — e.g. a
  `#[serde(other)]`-style `Plugin { id: RegistryId, config: Box<RawValue> }` arm —
  and that arm is the thing that reconstitutes the deleted seam. Until such an
  arm exists, no config can select a plugin, and this record must not be read as
  evidence that one can. Adding that arm is out of scope here; recording that it
  is *required*, and that its absence is a gap rather than a design property, is
  in scope.
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
  `BenchmarkRunWireV2` (also deleted; its `deny_unknown_fields`, `validate_outer`,
  `planned_replay_traces`, and `VariationSpec` typing port onto `BenchmarkRun`
  first — §2);
  `EnvelopeV2` (an in-process struct constructed after decode — not a wire shape).
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
