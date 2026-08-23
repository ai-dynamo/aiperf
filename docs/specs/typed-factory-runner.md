<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Typed-factory runner

## Purpose

State the target for the execution boundary once
[config-model-unification.md](config-model-unification.md) lands: **the runtime
consumes the typed `BenchmarkRun` directly, and a component is selected from the
typed config model itself** — an exhaustive `match` on the closed
`BenchmarkConfig.transport` enum, and an exhaustive match on the two-arm
`workload_kind(&cfg)` classifier — instead of the `AuthoredRunSpecV2` projection
that today re-serializes every section to `serde_json::Value`, manufactures a
`{ id, config: RawValue }` component spec, and re-decodes it per factory.

This is the completion of the config-model-unification arc and the step to parity
with the Python reference, where the same typed `BenchmarkConfig` rides inside
`BenchmarkRun` to the child and every service reads `cfg.endpoint` /
`cfg.transport` typed.

What the change delivers is every **typed** section of the config carried typed
end to end: no typed→`Value`→`RawValue`→typed round-trip, no per-factory
re-decode, decode-time errors, and the opaque `RawValue` seam confined to the two
residual uses listed under [Non-goals](#non-goals-and-trade-offs).

It is not a claim that no `Value` survives anywhere. `cfg.endpoint_profiles` is
declared on `BenchmarkConfig` as an open `serde_json::Map<String, Value>`
(`config/model/config.rs:118`), so it is untyped *at its source*, and this record
deliberately keeps the `endpoint_profile` transform over it (§2). Typing that map
is a separate change. The same applies to `cfg.failure_policy`'s current
`Option<Value>` and to the residual `RawValue` uses. The round-trip this record
eliminates is the one it created itself — re-serializing sections that were
already typed.

**Scope premise: the Rust tree is greenfield.** There is no released native
protocol-v2 stdin contract and no external consumer of it to preserve. The
`--execute` boundary is written and read by the same binary, and the
"bare resolved-run" arm exists because the code grew that way, not because a
published contract requires it. Nothing in this record is constrained by wire
compatibility, and no step needs a compatibility shim. Where a difference between
today's shapes matters, it matters because a *behavior* would be lost —
strictness, a validation check, a controller-derived field — and the migration
must port that behavior forward on its own merits. "Byte-exact against the mock
server" below is therefore an assertion about **observed run output**, never
about stdin bytes.

## Status

This record is partly built. Implementation lives on
`ajc/typed-factory-runner-v2`:

- **Step 3's DTO collapse is landed** (`b7619602fb`): the per-workload DTOs are
  the unified `WorkloadConfigV2` (`engine/registry.rs:855`), with
  `ScheduledWorkloadConfigV2` and `GraphWorkloadConfigV2` retained as type
  aliases (`:902`, `:904`).
- **Step 1's typed `cfg.transport` consumer is landed** as
  `transport_component` (`engine/protocol_v2.rs`), pinned against the projection
  it replaces by `transport_component_matches_inline_projection`. What landed is
  a typed *producer* whose output is still a `NamedRunnerComponentSpecV2 { id,
  config }`; `AuthoredRunSpecV2` retains only that projected field
  (`protocol_v2.rs:586`). The pin is at **component** level, not binding level —
  which step 1 establishes is the correct and sufficient assertion, but the
  status must not be read as "typed `Transport` now reaches the selection
  boundary". It does not, and will not until step 2 adds the `transport_typed`
  carrier.
- **`3f77a3adac`** fixed the `system_idle_gap_cap_seconds` projection guard (see
  [Built](#built)).
- **Steps 2 and 4 are not built.**

Read every "will" below as "will, where not already noted as landed".

## Built

Today the child does not consume `BenchmarkConfig` directly. `coordinator.rs`
takes `envelope.run.into_authored()` and runs the rest of the engine against
`AuthoredRunSpecV2` — a *second* run model produced by projecting the typed
`BenchmarkConfig`:

- **`BenchmarkRunWireV2.cfg`** is the typed `BenchmarkConfig`, but
  `into_authored` re-serializes each section to `serde_json::Value`
  (`to_value(&cfg.runtime)`, `to_value(&cfg.phases)`, …), reshapes it, derives
  `workload_id` from the typed `workload_kind(&cfg)` classifier, folds
  content-server into sidecars, and hand-builds `json!` blobs for the strict
  per-workload DTOs. The graph DTO carries **five** fields the scheduled DTO does
  not (`protocol_v2.rs:404-422`): `weka_semantics`, `ignore_trace_delays`,
  `recorded_agent_default`, `planned_replay_traces`, and
  `system_idle_gap_cap_seconds`. Only the first two and the cap are plain copies
  of typed `BenchmarkConfig` fields; `recorded_agent_default` is a *derivation*
  (`cfg.scenario.as_deref() == Some("recorded-agent-default")`) and
  `planned_replay_traces` comes from `BenchmarkRunWireV2`, not from
  `BenchmarkConfig` at all.

  The cap is attached for **every** `WorkloadKind::Graph` run, unconditionally:
  `3f77a3adac` ("fix(engine): project the system idle-gap cap under graph-ir
  too") replaced an earlier
  `matches!(cfg.weka_semantics.as_deref(), Some("legacy") | Some("agentx"))`
  guard with a plain `if let Some(cap) = cfg.system_idle_gap_cap_seconds`. That
  guard made the flag a silent no-op under graph-ir even though `resolve.rs`
  validates it there, its rejection message names graph-ir as supported, and
  `lower_graph` reads it into `NativeGraphDatasetPlan`. The migration carries the
  fixed behavior forward; reintroducing the predicate re-creates the bug.
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

## Target design

> Feasibility audited (2026-07-26) in two waves of adversarial code review —
> first the claim decomposition (transport union, workload collapse,
> `into_authored` deletion, seam typing, registry reduction), then the
> extensible-enum encoding (an **empirical** serde compile-test, normalized
> lookup, two-path dispatch, wire parity). No hard blocker survived.

### 1. Selection comes from the typed config model

`BenchmarkConfig.transport` (`config/model/config.rs:75`) is already
`Option<Transport>`, and `Transport` (`config/model/transport.rs:16-49`) is
already a closed `#[serde(tag = "type", rename_all = "snake_case")]` enum over
six variants: `Http`, `Grpc`, `DynosimOffline(DynosimConfig)`,
`DynosimOnline(DynosimConfig)`, `DryRun(DryRunConfig)`, and
`Websocket(WebSocketTransportConfig)`. `Http` and `Grpc` are **unit variants
carrying no payload at all**. There is nothing to introduce here; the union is
landed.

**The record selects fork (A): keep `Transport` closed.** Selection becomes an
exhaustive `match` on the enum, and workload selection an exhaustive match on
`workload_kind(&cfg)` → `{ Scheduled, Graph }`. Exhaustiveness is the point: the
compiler enforces arm coverage, and an unknown authored `type` is a serde decode
failure naming the expected variants rather than a registry lookup miss. The
alternative (fork B — an open string discriminant with a plugin tail) is
recorded in [§4](#4-open-id-plugin-seam-future-work) as the design the eventual
plugin work inherits; it is not what this migration builds.

`StaticAccuracyWorkloadConfigV2` remains outside `workload_kind`'s two arms — it
is selected by the static-accuracy path, not by the classifier — and the
exhaustive workload match must not be read as claiming otherwise.

Outstanding typing on the authored side, which this arc also completes:

- `dataset.synthesis` → `Option<TraceSynthesisSpec>`.
- `weka_semantics` → a closed enum. `#[serde(alias)]` alone cannot express
  today's acceptance set, because serde aliases are exact byte matches on the
  wire string. The live decoder is `weka_wants_legacy`
  (`online_execution.rs:287-294`), which folds
  `semantics.map(|s| s.trim().to_ascii_lowercase())` and *then* matches: `None`,
  `""`, `"graph-ir"`, `"graphir"`, `"graph_ir"` → graph-ir; `"legacy"`,
  `"agentx"` → legacy; anything else is
  `"unknown weka semantics {other:?}; expected 'legacy' or 'graph-ir'"`. Three
  behaviors a plain aliased enum drops: the **trim** (so `" legacy "` is accepted
  today), the **case fold** (so `"Legacy"` is accepted today), and the
  **empty string** mapping to graph-ir rather than to a decode error. The typed
  field therefore needs a hand-written `Deserialize`/`FromStr` reproducing that
  fold, and the error message moves with it verbatim. `None` and `""` collapsing
  to the same variant means the field is `Option<WekaSemantics>` with absent and
  blank both meaning `GraphIr`.
- `failure_policy` → typed `OnFailure`.

### 2. `BenchmarkRun` is the sole runner vocabulary

`decode_execute_wire` (`protocol_v2.rs:191`) remains the stdin contract and keeps
**both** arms: the `AuthoringWireV2` arm AIPerf itself writes
(`{"authoring": <Inputs>, sweep_id, variation, trial}`, `protocol_v2.rs:155`)
and the bare resolved-run arm retained for external harnesses, discriminated by
presence of the `authoring` key (`resolved_run_bytes`, `protocol_v2.rs:207`).
What changes is the *type* of the bare arm.

`EnvelopeV2` (`protocol_v2.rs:118`) is **not** a wire shape and is not the
boundary to hold fixed. It is a struct constructed in-process at
`cli/src/execute_mode.rs:477` *after* decode — its own doc comment says it is
"reconstructed around the bare `BenchmarkRunWireV2` stdin payload". There is no
`serde_json::from_slice::<EnvelopeV2>` anywhere in this design.

`AuthoredRunSpecV2`, `into_authored`, and `NamedRunnerComponentSpecV2` are
**deleted**. The child composition root (`coordinator.rs`) drives the engine
against **`&BenchmarkRun`**, reading authored sections through `run.cfg` and
run-level facts off `run` itself, exactly as the Python services read `cfg` out
of the run they were handed. `BenchmarkRun` is the runner vocabulary throughout
this record; `BenchmarkConfig` is the *authored* vocabulary reached as `run.cfg`.
This distinction is load-bearing: `benchmark_id`, `artifact_dir`,
`planned_replay_traces`, `trial`, `variation`, `resolved`, and `variables` are
run-level facts with no `BenchmarkConfig` home at all (`config/model/run.rs:17-48`
vs `config/model/config.rs`). An engine handed `&BenchmarkConfig` could not see
`validate()`'s `benchmark_id`/`artifact_dir` checks, could not reach
`planned_replay_traces`, and could not resolve the export-stem derivation's
inputs.

The bare arm is re-typed from `BenchmarkRunWireV2` to `BenchmarkRun`, and
`BenchmarkRunWireV2` is **deleted**. Because the tree is greenfield this is not a
compatibility question — but it is not a free rename either. The two types differ
on every axis that matters, and each difference is a behavior that would silently
vanish if the swap were done naively:

| | `BenchmarkRunWireV2` (`protocol_v2.rs:293-329`) | `BenchmarkRun` (`config/model/run.rs:17-48`) |
|---|---|---|
| strictness | `#[serde(deny_unknown_fields)]` | none — unknown top-level fields accepted |
| `resolved` | `serde_json::Value` (open) | typed `Resolved` |
| `variation` | `Option<VariationSpec>` | `Option<serde_json::Value>` |
| `trial` | `usize` | `u32` |
| `variables` | `BTreeMap<String, Value>` | `serde_json::Map<String, Value>` |
| `planned_replay_traces` | `BTreeSet<PlannedReplayTraceInstance>` | **absent** |
| outer validation | `validate_outer()` (`protocol_v2.rs:332-350`) | **none** |
| export stem derivation | `into_authored` (`protocol_v2.rs:464-475`) rewrites `export.genai_perf.stem` and `export.timeslice.stem` from `artifacts.records_path` | **none** — `cfg.export` carries the undelivered authored value |

That table is a **port list**, not a compatibility obligation. Each row is
decided on merit below. Several ported behaviors live only in `into_authored`'s
*body* or in the *shape* it hands the factory, so they do not appear in the table
at all; they are listed with the rest.

#### Port list

**`deny_unknown_fields` — port it.** `BenchmarkRun` gains
`#[serde(deny_unknown_fields)]`. Strict decode is the stated goal of this whole
arc; dropping it would make the migration a strictness *regression* in the name
of typing.

**`validate_outer()` — port it, with the dataset check tightened.** Two of its
three checks are what they look like: non-empty `benchmark_id`, non-empty
`artifact_dir`. The third is not. Its message reads
`"run.cfg.datasets must contain exactly one dataset"`, but the `ensure!` it
guards tests only
`self.cfg.datasets.as_ref().is_some_and(|datasets| !datasets.is_empty())`
(`protocol_v2.rs:342-348`). A two-dataset run passes, and `into_authored` then
silently discards the tail — `cfg.datasets.and_then(|d| d.into_iter().next())`
(`protocol_v2.rs:359-362`).

This record selects **enforce real cardinality** (`datasets.len() == 1`) on the
greenfield premise: silently dropping an authored dataset is a bug, not a
feature, and the existing message already documents the intended contract. This
is a behavior change and is recorded as one — configs that run today start
failing to decode. It carries two obligations: the step-4 test list gains a case
asserting a two-dataset config is *rejected* with a message naming the count, and
the `ensure!` message stops lying. Neither the current check nor
`into_authored`'s `.next()` may be deleted before that test exists, or the
truncation moves from silent-and-tested-nowhere to silent-and-unreachable.

**`planned_replay_traces` — port it as a run-level field on `BenchmarkRun`**
alongside `cfg`, not as a `BenchmarkConfig` field, because it is
controller-derived rather than authored. It lives on `BenchmarkRunWireV2`
(`protocol_v2.rs:325`), is written by the controller at
`cellular_controller.rs:1978`, copied into the graph workload DTO by
`into_authored` (`protocol_v2.rs:413-415`), and consumed at `entrypoints.rs:415`
as `expected_replay_traces`. It appears **nowhere** in `runtime/src/config/` or
`cli/src/`. Deleting `BenchmarkRunWireV2` without giving it a home loses cellular
graph replay's trace expectation.

**`variation: Value` → `Option<VariationSpec>` — re-type `BenchmarkRun`, after
relocating the type.** `BenchmarkRun` holds `variation` as an open `Value`
(`config/model/run.rs:45-46`) where the wire DTO had a typed `VariationSpec`;
here the wire DTO is the stricter one, so the re-type is the correct direction
and belongs in this change rather than being inherited as-is.

It cannot be done by referring to the existing type. `VariationSpec` is declared
at `engine/protocol.rs:279`, inside `pub mod engine`, which is
`#[cfg(feature = "engine")]` (`runtime/src/lib.rs:40-41`). `config` is **not**
feature-gated (`lib.rs:48`), so `BenchmarkRun` compiles in every build; pointing
an unconditional field at a feature-gated type fails to compile whenever `engine`
is off — which is the default, and is the plain `cargo test -p aiperf-runtime`
invocation the gates run at every step. The step must therefore **move**
`VariationSpec` into `config/model/` (with `engine::protocol` re-exporting it so
existing `protocol::VariationSpec` paths keep working), not `use` it across the
boundary. Its `deny_unknown_fields` and its `BTreeMap<String, Value> values`
field move unchanged. Duplicating the struct is rejected: two `deny_unknown_fields`
definitions of one wire shape is the drift this arc exists to remove.

**`trial: usize` → `u32` and `variables: BTreeMap` → `serde_json::Map` — accept
`BenchmarkRun`'s shapes.** These are the typed model winning, which is the point.

The `variables` ordering question is settled here rather than deferred, because
the two types genuinely differ: `serde_json` is built with `preserve_order` in
this workspace (`runtime/Cargo.toml:110`), so `serde_json::Map` iterates in
**insertion** order while `BTreeMap` iterates **sorted**. That difference is
nevertheless inert, because `variables` has no consumer: it is decoded onto
`BenchmarkRunWireV2` (`protocol_v2.rs:326-328`) and read nowhere —
`into_authored` never touches it and `AuthoredRunSpecV2` has no `variables`
field, so nothing downstream can observe either order. Duplicate keys are
last-wins in both types. The swap is therefore free, and no step-4 test is owed
for it.

**`resolved: Value` → `Resolved` — accept it; the narrowing is real but much
smaller than the module doc implies.** `Resolved`
(`config/model/resolved.rs:13-46`) has sixteen fields, of which **fourteen are
`Option<...>`**; only `artifact_dir_created: bool` and
`gpu_telemetry_mode: String` are not. A derived `Deserialize` defaults a missing
`Option` field to `None` on its own, so the absence of per-field
`#[serde(default)]` does *not* make those fourteen required, and the module doc's
"Every field is present in the wire object, including nulls" describes what the
producer emits, not what the decoder demands. `Resolved` also carries no
`deny_unknown_fields`, so extra keys are accepted.

The actual narrowing against `resolved: Value` has two parts. **Non-object
payloads:** `resolved: Value` accepts *any* JSON value — `3`, `"x"`, `[]`,
`true`, and explicit `null` — because `Value` is total over JSON;
`resolved: Resolved` rejects every one of them, and `#[serde(default)]` covers
only an **absent** key, so an explicit `"resolved": null` is a decode error where
today it is `Value::Null`. **Object payloads:** an object omitting
`artifact_dir_created`, one omitting `gpu_telemetry_mode`, or one whose value for
any field has the wrong JSON type. Every other object that decodes as `Value`
today still decodes. That only bites the
bare-resolved-run arm kept for external harnesses; AIPerf's own authoring arm
re-projects a full `Resolved`. It is the right direction, the greenfield premise
carries it, and it needs no dedicated step-4 test — the two mandatory keys are
enforced by the compiler-generated decoder, and no e2e run omits them.

**Export stem derivation — port it.** This behavior lives only in
`into_authored`'s body: at `protocol_v2.rs:464-475` it reads
`artifacts_spec.records_path`, strips a `.jsonl` suffix from the file name, and —
when the stem is non-empty — overwrites `export_cfg.genai_perf.stem` with it and
`export_cfg.timeslice.stem` with `format!("{stem}_aiperf")`. This is the live
implementation of `--profile-export-prefix` / `artifacts.prefix` for the summary
and timeslice outputs; `cfg.export` itself never carries the derived value.
Pointing execution at typed `cfg.export` without reproducing this cross-field
transform does not fail — it silently reverts the summary and timeslice files to
the default `profile_export_aiperf.{json,csv}` and
`profile_export_aiperf_timeslices.*` names while per-record output stays under
the authored custom stem, i.e. one run emitting two different prefixes. It must
become an inherent derivation on the typed model, applied at the same boundary as
`BenchmarkRun::validate()`.

**Two more all-or-nothing decode fallbacks — `artifacts` and `export`. Neither
was listed, and the artifacts one is reachable.** The metrics decision above is
not the only `.unwrap_or_default()` in `into_authored`; there are three, and only
metrics had a decision recorded:

- **`artifacts` (`protocol_v2.rs:454-458`).** `serde_json::from_value::<ArtifactSpecV2>(to_value(&cfg.artifacts)?).unwrap_or_default()`
  discards the **entire** artifacts section on any decode error. That error path
  is reachable from a valid `BenchmarkConfig`, because the two types disagree on
  one field: `UserFile.format` is a plain `String`
  (`config/model/artifacts.rs:10-15`) while `UserFileSpecV2.format` is the closed
  `UserFileFormatV2 { Json, Yaml, Text }` (`protocol_v2.rs:1083-1104`). So a run
  carrying `cfg.artifacts.user_files[0].format = "bogus"` type-checks, decodes as
  `BenchmarkConfig`, is rejected by `ArtifactSpecV2`, and loses `records_path`,
  `raw_path`, `inputs_path`, `trace`, and the whole dry-run analysis family to
  `ArtifactSpecV2::default()` — every field of which is `#[serde(default)]`
  (`protocol_v2.rs:943-983`), so the failure produces no error and no output
  rather than a diagnostic. **Decision: do not port the fallback.** The same
  greenfield rationale applied to metrics and to dataset truncation applies with
  more force here, because the discarded section decides whether artifacts are
  written at all. `validate()` surfaces
  `artifacts.user_files[i].format must be one of json, yaml, text`, and step 4
  adds a named test for it. Typing `UserFile.format` as the enum outright is the
  better end state and is compatible with this; it is not required by this
  record, which only forbids the silent discard.
- **`export` (`protocol_v2.rs:459-463`).** The same shape, and it deserves its
  own note because it is not a bridge between one type and a stricter view of
  itself: `cfg.export` is `Option<Export>` (`config/model/config.rs:99`,
  `config/model/export.rs:265-285`) and the target is a **separately maintained**
  `crate::export::ExportConfig` (`export/mod.rs:279-301`). They are not the same
  struct and do not have the same fields — `ExportConfig` carries `timeslice`,
  `accuracy_csv`, and a `genai_perf.stem` that typed `GenaiPerf` has no field for
  (which is why the stem derivation above exists). `ExportConfig` is
  `#[serde(default, deny_unknown_fields)]`, so the bridge works today only
  because every key `Export` serializes happens to be a known `ExportConfig` key;
  it decodes successfully on the current tree, which is why exports work at all
  and why the stem overwrite is observable. But the failure mode is silent and
  total: any key added to `Export` and not to `ExportConfig` disables **every**
  exporter at once, with no error. The migration should read typed `cfg.export`
  directly and convert field-by-field, or — if the re-decode is kept for now —
  replace `.unwrap_or_default()` with a surfaced error. This is a second
  typed → `Value` → typed hop that §2's headline claim does not mention, and it
  survives the projection's deletion unless it is repointed.

**Three mandatory-section rejections — port them verbatim, messages included.**
`BenchmarkConfig` holds all of these sections as `Option`
(`config/model/config.rs:66`, `:69`, `:75`, `:81`), so nothing on the typed model
reproduces any of them:

- **`cfg.models` absent is a hard error.** On the implementation branch the
  lowering is
  `cfg.models.map(ModelsSpec::from).ok_or_else(|| anyhow!("run.cfg.models must be an object"))?`;
  on `ajc/rust` the same rejection is `models_from_config`'s
  `as_object().ok_or_else(|| anyhow!("run.cfg.models must be an object"))?`
  (`protocol_v2.rs:507-511`), reached with `Value::Null` when the section is
  `None`.
- **The default `cfg.endpoint` absent is a hard error**, by the same mechanism:
  `serde_json::to_value(&cfg.endpoint)` yields `Null` and `endpoint_profile`
  (`protocol_v2.rs:552-556`) rejects it with `"run.cfg.endpoint must be an
  object"`.
- **`cfg.transport` absent is a hard error.** `transport_component` routes `None`
  to `component_from_inline(Value::Null, "run.cfg.transport")` precisely to
  preserve the prior *transport must be an object* failure when unset.

These three are the only thing making models/endpoint/transport mandatory;
without them a run omitting a transport reaches component selection with `None`
and fails somewhere less legible, or not at all. They move into
`BenchmarkRun::validate()`.

**`BenchmarkRun::validate()` is not a boundary the cellular controller crosses,
and this record does not claim otherwise.** The controller path is selected in
`cli/src/execute_mode.rs:113-129`, *before* `decode_execute_wire`: it works on a
raw `serde_json::Value` obtained from `resolved_envelope_from_input`, mutates it
in `build_cell_envelope`, and ships it. It never constructs `BenchmarkRunWireV2`,
so `validate_outer()` (`coordinator.rs:138`) and `into_authored`
(`coordinator.rs:159`) do not run there **today** either. Moving the checks onto
`BenchmarkRun::validate()` therefore neither adds nor removes controller-side
coverage; the invariant that holds before and after is that **each cell
revalidates its own envelope** on the ordinary `run_v2` path. Two obligations
follow: the migration must not assume `validate()` has run when the controller
mutates an envelope, and it must not "simplify" the controller onto typed
`BenchmarkRun` as a side effect of this change — that is a separate step, and
doing it here would put a validation boundary in front of the mutation for the
first time and change which runs fail and where.

**The controller has its own metrics decoder, and the metrics decision below does
not reach it unless the migration repoints it.** `cellular_metrics_config`
(`cellular_controller.rs:2845-2856`) reads `/run/cfg/metrics` straight out of the
raw envelope `Value` and repeats the same swallow —
`.map(|value| serde_json::from_value(value).unwrap_or_default())` into
`engine::protocol::MetricsSpec` — with no `BenchmarkRunWireV2`, no
`into_authored`, and no `validate()` anywhere upstream of it. It runs on the
controller's startup path (`cellular_controller.rs:862`, deriving the
`MetricsConfig` the merge folds with) and again from
`cellular_will_use_exact_fold` (`:2225-2231`, deciding exact-vs-sketch storage).
So a cellular run whose `metrics.slos` carries a string value would, after the
change below, still have the controller silently build a **default** merge and
storage policy while each cell independently rejects the run — a split where the
controller's fold policy and the cells' verdict disagree, and where step 4 can
pass every `BenchmarkRun::validate()` test listed in this record without
touching it.

This is a required port, not an observation: `cellular_metrics_config` must go
through the same strict typed conversion `validate()` uses, so the controller
fails on the same input the cells fail on, and step 4 adds a controller test for
a **non-numeric SLO value**. The existing controller test at
`cellular_controller.rs:4532-4552` covers an unknown SLO *name*, which is a
different rejection and passes either way. If instead the controller keeps a
lenient fallback, that has to be written down as a deliberate asymmetry with a
reason — this record does not choose that, because a merge policy derived from
config the run is about to reject has no defensible meaning.

**The metrics fallback — do *not* port as-is; surface the error.** Today an
invalid metric SLO silently defaults the whole metrics section. On
`ajc/typed-factory-runner-v2` that is
`cfg.metrics.and_then(|m| MetricsSpec::try_from(m).ok()).unwrap_or_default()`,
where `TryFrom` (`engine/protocol.rs:209-231`, which exists on that branch only)
fails on the *first* non-numeric `slos` entry, so `.ok()` discards
`slice_duration_seconds`, every valid SLO, `sketch`, and `steady_state` along
with it. On `ajc/rust` the swallow is **wider**:
`serde_json::from_value(metrics).unwrap_or_default()` (`protocol_v2.rs:490`) over
a `deny_unknown_fields` `MetricsSpec` (`engine/protocol.rs:164-179`) defaults on
**any** decode failure — an unknown key, a mistyped `slice_duration_seconds`, a
non-bool `sketch`. Silently dropping an authored metrics section because one SLO
is a string is the same class of defect as the dataset truncation above, and the
greenfield premise resolves it the same way: `validate()` **surfaces** the error
(`metrics.slos["x"] must be a number`) instead of defaulting. This is a behavior
change and is recorded as one.

**Two rejections enforced by the projection's *decode*, not its body.** A
statement-level walk of `into_authored` misses these; they follow from the shape
it hands the factory:

- **`cfg.runtime.workers = Some(0)` is rejected**, by
  `ensure!(worker_count > 0 && worker_count <= usize::MAX as u64, "run.cfg.runtime.workers must be a positive usize")`
  (`protocol_v2.rs:381-385`). `Runtime.workers` is `Option<u32>`
  (`config/model/runtime.rs:18`) with no positive-value validation anywhere
  upstream — `grep -rn "workers > 0"` over `rust/` finds no other config-path
  check. The same lines carry a live *default* the typed path also owes: absent
  `workers` resolves to `default_worker_count()` (machine parallelism), **not**
  `1`.
- **`cfg.phases` absent is rejected.** `serde_json::to_value(&cfg.phases)`
  (`protocol_v2.rs:394-395`) emits JSON `null` for `None` — `skip_serializing_if`
  on the `BenchmarkConfig` field does not apply to a direct `to_value` of the
  field — and the factory then decodes into the mandatory
  `WorkloadConfigV2.phases: Vec<PhaseSpec>` (`registry.rs:853`; on `ajc/rust`,
  `ScheduledWorkloadConfigV2` at `:845-859`), which fails. `BenchmarkConfig`
  holds `phases: Option<Vec<Phase>>` (`config/model/config.rs:96`), so once the
  DTO is deleted there is no decode to fail.

Neither rejection survives automatically, and both have downstream backstops that
a careless implementation would silently rely on: `validate_common_workload`
(`registry.rs:1650-1666`) asserts `worker_count > 0` ("workload worker_count must
be positive") and `!phases.is_empty()` ("workload phases cannot be empty"),
`build_common_plan` re-asserts both (`execute/plan.rs:509-510`), and
`build_native` re-asserts workers (`turn_execution.rs:296`). So the accurate
statement is not "zero-worker execution becomes reachable" — it is that **the
failure moves from a decode error at the projection boundary to an `ensure!`
several layers in, with a different message, and only if the migration keeps
routing the typed values through `validate_common_workload`.** That routing is a
step-3/step-4 obligation, not a given: `validate_common_workload` takes
`(worker_count: usize, dataset, tokenizer, phases: &[PhaseSpec])` — values the
decoded DTO supplies today.

Decisions:

- **`workers`.** `BenchmarkRun::validate()` carries the check verbatim, message
  included. The `default_worker_count()` fallback for absent `workers` moves with
  it.
- **`phases`.** Absent and empty collapse to the same condition on the typed
  model (`cfg.phases.as_deref().unwrap_or_default()`), and this record keeps them
  collapsed: `validate()` rejects both with one message naming the section rather
  than reproducing the DTO's absent-vs-empty split (today absent fails at decode
  and empty fails later in `validate_common_workload`). Merging them changes
  message and failure point only — both inputs are rejected before and after.
- Neither decision permits deleting the `validate_common_workload` /
  `build_common_plan` assertions; they stay as defense in depth for the cellular
  and eval paths that construct plans without going through
  `BenchmarkRun::validate()`.

**Three tokenizer rejections that live in the DTO's decode.** `into_authored`
maps `cfg.tokenizer: None` to `{}` (`protocol_v2.rs:388-392`), and every workload
then calls `validate_authored_tokenizer(&workload.tokenizer)`
(`online_execution.rs:248`, `:344`, `:467`, `offline_execution.rs:1450`), which
runs `AuthoredTokenizerV2::decode` (`online_execution.rs:960-985`). That decode
enforces three things the typed model does not:

- `name` is a **mandatory** field on `AuthoredTokenizerV2`
  (`online_execution.rs:594-607`), so an absent `cfg.tokenizer` section — which
  projects to `{}` — fails to decode outright.
- `!config.name.trim().is_empty() && config.name.trim() == config.name`, i.e. a
  blank or whitespace-padded name is rejected.
- `!config.revision.trim().is_empty()`.

A fourth behavior rides the same decode and is not a rejection:
`trust_remote_code = true` emits a `tracing::warn!`
(`online_execution.rs:975-983`) explaining that the native `tokenizers` library
never executes repository Python, so the flag is inert and the tokenizer loads as
if it were false. It exists so that harness command lines which pass the flag
unconditionally still run *and* still tell the operator the flag did nothing.
Deleting the decode deletes the only place that warning is emitted — silently,
since nothing fails. Port it with the three rejections. Note the flag is not
inert everywhere: `lower` still forwards it as
`resolver.resolve(&self.name, &self.revision, self.trust_remote_code)`
(`online_execution.rs:1059`), so the warning's claim is scoped to repository-code
execution, not to the resolver, and the ported text must keep that scope.

`Tokenizer` (`config/model/tokenizer.rs:15-38`) is a plain derived struct with a
mandatory `name` but **no** semantic validation, and `BenchmarkConfig.tokenizer`
is `Option<Tokenizer>`, so all three rejections vanish with the decode. Both
`ensure!` messages appear nowhere else in the tree. **Port all three into
`BenchmarkRun::validate()`, messages included**, in the same batch as the
models/endpoint/transport rejections; the absent-section case joins them as a
fourth mandatory section.

**`parse_dispatch_mode`'s cellular-aware default — port the derivation, but it
is already inert at execution time.** `parse_dispatch_mode`
(`protocol_v2.rs:260-272`) is not `runtime.dispatch.unwrap_or_default()`. An
explicit `runtime.dispatch` wins, but when it is absent the default branches on
`runtime.cells`: `cells > 1` resolves `DispatchMode::Sharded`, and `cells <= 1`
resolves `DispatchMode::default()` (`Global`). The rationale in the function's
own doc comment is that a cellular run has already forfeited single-process
byte-exact determinism, so `Global`'s shared cross-thread admission gate inside a
cell buys parity that is already gone and costs pure overhead.

**That branch does not, however, reach any process that executes requests.** A
run with `cells > 1` is promoted to the cellular controller in
`cli/src/execute_mode.rs:113-129`, and the controller is not an issuer; the cells
are. Every cell envelope is built by the single call site at
`cellular_controller.rs:1149-1150`, and `build_cell_envelope` overwrites the
field — `runtime.insert("cells", 1)` (`cellular_controller.rs:1994`) — before the
envelope is serialized to the cell, so a cell process always parses `cells == 1`
and always lands on `Global`. The one consumer of the resolved value,
`dispatch_mode: run.dispatch` (`online_execution.rs:1758`), therefore never sees
`Sharded` from this derivation; only an explicitly authored
`runtime.dispatch: sharded` produces it. Nothing in the controller reads
`dispatch` at all.

So a typed reader writing `cfg.runtime.dispatch.unwrap_or_default()` changes **no
executing run's dispatch mode**. What it changes is the controller-side
projection and the CLI/runner parity assertion that compares the two resolution
paths (`cli/src/profile.rs:1669-1675`). Port the derivation anyway — dropping it
would silently make that parity assertion vacuous and would erase the recorded
intent for cells that are launched some other way — but port it as a documented
projection detail, not as a performance guard. The claim that omitting it is a
large cellular performance regression is **wrong**, and this record previously
made it.

There is a live gap underneath: the intended cellular default is unreachable, so
the c4-144 measurement it cites does not describe what cells actually run today.
Whether cells *should* default to `Sharded` is a real question, and it is
deliberately **out of scope here** — changing it is a runtime behavior change,
not a typing migration. It is recorded so the migration does not launder it into
a "preserved behavior" it never had. The three exact unit tests for the
derivation (`runtime_dispatch_defaults_to_global_when_absent`,
`runtime_dispatch_defaults_to_sharded_for_cellular`,
`runtime_explicit_dispatch_wins_over_cellular_default`) are **live**: they sit in
`mod dispatch_mode_tests`, declared `#[cfg(test)]` at `protocol_v2.rs:1410-1411`,
not in the disabled module above it. They compile and run under
`cargo test -p aiperf-runtime --features engine`, and they move with the
derivation. What they pin is the projection, not cell behavior — which is why
they stayed green while the `Sharded` arm became unreachable at execution
time.

**`parse_hop_routing` — a repoint, listed for completeness.**
`parse_hop_routing` (`protocol_v2.rs:283-290`) is the sibling of the above and is
the easy case: `Option<HopRouting>` in, `Option<HopRouting>` out, absent stays
`None`, an unrecognized string is a hard error. `Runtime.hop_routing` is already
`Option<HopRouting>` (`config/model/runtime.rs:36`), so the typed decode
reproduces every one of those behaviors, including the rejection. The only work
is repointing its single consumer, `hop_routing: run.hop_routing`
(`online_execution.rs:1759`), at `cfg.runtime.hop_routing`, and accepting that
the error text loses the `run.cfg.runtime.hop_routing:` path prefix the manual
`map_err` adds. Nothing is at risk; it appears here because its sibling does and
an omission would read as an oversight.

**`resource_presence` — preserve `into_authored`'s algorithm verbatim.** It is
not a naive `cfg.field.is_some()` map. The exact classification
`into_authored` hardcodes (`protocol_v2.rs:496-501`) is `models: true`,
`endpoints: true`, `metrics: true`, `artifacts: true` — *unconditionally*,
regardless of whether the corresponding `Option` on `BenchmarkConfig` is `None` —
plus `sidecars: sidecars_present`, where `sidecars_present`
(`protocol_v2.rs:432-443`) is
`serde_json::to_value(&cfg.sidecars).as_object().is_some_and(|o| !o.is_empty())`.
That is emptiness of the *serialized object*, a categorically different predicate
from `Option::is_some`: every field of `Sidecars`
(`config/model/telemetry.rs:216-230`) carries
`skip_serializing_if = "Option::is_none"`, so `Some(Sidecars::default())`
serializes to `{}` and classifies as **absent**, while `cfg.sidecars.is_some()`
would call it present. That flip changes the Required/Optional/**Forbidden**
matrix in `validate_resource_requirements`.

The *other* construction path, `AuthoredRunSpecV2::deserialize`
(`protocol_v2.rs:661-667`), already uses the naive `wire.resources.X.is_some()`
form, so the two paths disagree today. They differ on **five** of five entries,
not just `sidecars`: the naive form reports
`models`/`endpoints`/`metrics`/`artifacts` as *absent* whenever the corresponding
`Option` is `None`. `into_authored` is the path every AIPerf-authored run
actually takes (`decode_execute_wire`'s authoring arm), so it is the observed
production behavior and the only one with e2e coverage; adopting the
`Deserialize` form would move four resources from Optional to
Forbidden/Required transitions for every run that omits a section. The
`Deserialize` impl disappears with `AuthoredRunSpecV2` in step 4, so the
divergence is closed by deletion rather than reconciled.

**The sidecar adapter boundary — a `{ id, RawValue }` seam this migration does
*not* delete, and the §3 claim must be scoped to say so.** `into_authored`
serializes typed `cfg.sidecars` to a `Value` and re-decodes it into
`SidecarSpecV2` (`protocol_v2.rs:430-443`), whose five fields are each
`Option<Box<RawValue>>` (`protocol_v2.rs:1105-1124`). `authored_inputs()`
(`:1132-1143`) then pairs each present body with a fixed string id and hands
`AuthoredSidecarInput { id: &str, config: &RawValue }`
(`sidecar_input.rs:273-279`) to a resolver over an **open** adapter registry,
where `SidecarInputAdapter::validate(&RawValue)` performs "the sole full strict
decode" (`sidecar_input.rs:311-318`). That is structurally the same
id-plus-opaque-body shape §1 removes for transports, it is reached from a typed
`Sidecars` (`config/model/telemetry.rs:214-230`), and **nothing in this record
touches it**. Three consequences, all of which the record must state rather than
imply:

- The §3/§4 statement that step 4 deletes the `{ RegistryId, RawValue }` seam is
  true of `NamedRunnerComponentSpecV2` only. The sidecar seam is open by
  design — adapters are registered, not enumerated — and survives.
- Feeding those adapters after the projection dies still requires
  typed `Sidecars` → `RawValue`. That is a round-trip, and it is the one place
  the Purpose section's claim would be false if left unqualified. Either the
  migration re-points `SidecarInputAdapter::validate` at typed values (a
  five-adapter change this record does not scope), or it re-serializes at the
  adapter boundary and says so. This record selects **re-serialize and say so**:
  the seam's openness is the reason it exists, and closing it is a separate
  decision from deleting the projection.
- `SidecarSpecV2` carries a fifth field, `live_streaming`, that typed `Sidecars`
  does not have at all. Since `into_authored` builds the DTO by re-decoding the
  serialized typed value, `live_streaming` is already unreachable from Config v2
  and always `None`. Deleting the DTO makes that structural instead of
  incidental. It is not a loss, but a reader comparing the two types will find
  the field and should not have to re-derive this.

One thing here is *not* owed. `SidecarSpecV2::validate_outer`
(`protocol_v2.rs:1145-1159`) checks each present body parses as JSON and is an
object, with per-field messages — but its only caller is
`AuthoredRunSpecV2::validate_outer` (`protocol_v2.rs:722-751`), and that function
has no production caller at all: the sole reachable `validate_outer` in
`coordinator.rs:138` is `EnvelopeV2`'s, which delegates to
`BenchmarkRunWireV2::validate_outer` (`protocol_v2.rs:138`, `:332-350`). A
repo-wide grep finds `AuthoredRunSpecV2::validate_outer` called only from
`runtime/tests/recorded_agent_protocol.rs:63`/`:90`. So the sidecar,
models, metrics, artifacts, and endpoint `validate_outer` bodies gated behind it
are dead in production today, and porting them would *add* validation rather than
preserve it. They are listed here as deliberately dropped, with the note that a
`RawValue` body that is valid JSON but not an object will now fail inside the
adapter's own strict decode instead.

**Endpoint and model transforms — mostly no-ops, with one live exception.**
`endpoint_profile` (`protocol_v2.rs:552-563`) renames `timeout` →
`timeout_seconds` and removes `url_strategy`; `models_from_config` (`:507-521`)
retains only `name`/`weight`. Against the typed model these are no-ops:
`Endpoint` (`config/model/endpoint.rs:117`) already stores `timeout_seconds`
(`:138`) and has neither a `timeout` nor a `url_strategy` field — `url_strategy`
exists only in the authoring layer (`cli/src/flags.rs:368`, `cli/src/yaml.rs:915`,
consumed and validated at `yaml.rs:1690`) and never reaches `BenchmarkConfig` —
and `ModelItem` (`config/model/models.rs:22`) has only `name` and `weight`. For
models, the migration may simply drop the transform.

**It may not drop the default-profile transform, because that transform supplies
the profile's identity.** `endpoint_profile` does not only rename and remove; its
first statement is
`profile.insert("id".to_owned(), Value::String(id.to_owned()))`
(`protocol_v2.rs:557`), and `endpoint_profiles` calls it as
`endpoint_profile("default", default)` (`:545`). Typed `Endpoint`
(`config/model/endpoint.rs:115-174`) has **no `id` field at all** — the literal
string `"default"` exists nowhere in the typed model and is manufactured here.
Downstream the field is required, not decorative: `endpoint_profile_identity`
(`:917-935`) fails with `id must be a string` when it is absent and additionally
requires non-empty and untrimmed-free, and `RunContext::default_endpoint_profile`
(`registry.rs:1256-1259`) resolves profiles by the literal name `"default"`. A
migration that drops this transform produces profiles with no identity, and every
`default_endpoint_profile()` lookup fails.

The same statement is what names the **override** profiles: the loop at `:546-548`
passes each `cfg.endpoint_profiles` map key as `id`, so the key becomes the
profile's `profile_id`. That is the map key being promoted into the value, which
no typed re-read reproduces — and note it *overwrites*, so an override body that
already carried its own `"id"` key has it replaced by the map key today.

Two obligations, neither optional: derive the default profile's `id` as the
literal `"default"`, and derive each override profile's `id` from its map key,
overwriting any authored `id`. The sort ported below orders profiles; it does not
identify them, and the two are separate ports.

They are **not** no-ops for the override profiles. `cfg.endpoint_profiles`
(`config/model/config.rs:118`) is an open `serde_json::Map<String, Value>`, not a
typed section, and `into_authored` feeds it through the same `endpoint_profile`
at `protocol_v2.rs:448-451`/`:547`. An authored override may therefore still
carry `timeout` and `url_strategy` keys, and the rename/removal is live for it.
The migration must keep the transform on that open map — or type
`endpoint_profiles` as a `BTreeMap<String, Endpoint>`, which is a larger change
than this record scopes.

**Override-profile ordering is part of that transform — port it.** Before
building the profile list, `into_authored` does
`cfg.endpoint_profiles.into_iter().collect::<BTreeMap<_, _>>()`
(`protocol_v2.rs:448-451`), and `endpoint_profiles` (`:540-550`) then pushes
`"default"` first and the rest **in that sorted order**. Because `serde_json` is
built with `preserve_order` here (`runtime/Cargo.toml:110`), iterating
`cfg.endpoint_profiles` directly yields *authored* order instead, so the
collection into a `BTreeMap` is a real normalization and not an incidental type
choice. It is observable: profile position is the profile's identity downstream —
`ValidatedEndpointProfileV2`s are indexed by `enumerate()` position
(`registry.rs:1124`) and resolved back by index (`registry.rs:1244`), and the
iteration accessor is documented as "authored order" (`registry.rs:1262`). A
typed consumer that drops the sort reorders every multi-profile run's profile
indices. Sort the override keys at the same boundary, `"default"` still first.

### 3. Registry role after the change

`AIPerfRegistry`/`AIPerfExtension` shrinks from "transactional registry of
config-decoding `dyn` factories" to "descriptor catalog + executor provider": it
still backs `--capabilities` output and any genuinely `dyn` executor seams, but
it no longer owns component-config decode or `ComponentId` → factory lookup for
*selection*. [extension-registry.md](extension-registry.md) is updated in the
same change to describe the reduced surface; the frozen-at-bootstrap guarantee is
unchanged.

There is no plugin tail to fall through to. After the migration, transport
selection is an exhaustive `match` on a closed enum reached from a
`#[serde(tag = "type")]` field: an unknown id is a **decode** failure on
`cfg.transport` (serde's own "unknown variant `xyz`, expected one of …", which
already enumerates the variants), never a registry miss, and there is no default
arm to route. The diagnostic obligation that survives is step 2's: a variant
whose factory is not compiled in must produce a build-named refusal rather than a
registry lookup miss.

What remains registry-addressed is exactly step 2's obligations (b) and (c) —
descriptor contribution for `--capabilities`, and
`CurrentNativeGraphModelBindingResolver::resolve`'s `transport_factory(id)`
lookup. Only **transports and workloads** carry the `RawValue`-per-factory
config-decode role; endpoints, samplers, exporters, and actuators are already
typed or name-keyed and are untouched. The `--capabilities` catalog is
unaffected — `Catalog::from_registry` reads only `&'static` descriptors, never
`validate()`.

### 4. Open-id plugin seam (future work)

**Non-normative.** Nothing in this section is a task in the
[Migration](#migration); no step implements it, and an implementer building the
migration should read §1–§3 and the migration only. It is retained because when
the dynamic-plugin arm lands — the `Plugin { id: RegistryId, config: Box<RawValue> }`
variant recorded under [Non-goals](#non-goals-and-trade-offs) as *required and
absent* — this is the worked design for it, including an empirical serde finding
that rules out the obvious encoding.

The design separates two axes:

- **The discriminant** (the component `type`/id) **stays an open, normalized
  string**, not an enum. It is `RegistryId` (`extensions/registry_id.rs`): a
  `#[serde(transparent)]` newtype whose custom `Deserialize` normalizes
  `trim().to_ascii_lowercase().replace('-', "_")` (Python `_normalize_name`) and
  validates non-empty, with `Borrow<str>` for map lookup. A discriminant is
  inherently open — plugins register ids at bootstrap — so a string id is the
  honest representation, and it is exactly what Python's `ExtensibleStrEnum` *is*
  (a `str` subclass with a convenience known-values layer).
- **The config payload becomes a typed struct** for built-ins. That is the whole
  prize (parse-don't-validate, decode-time errors, no `Value`-shuffling) and it
  is cheap: one `#[derive(Deserialize)] #[serde(deny_unknown_fields)]` struct
  each.

A seam component on the wire is `{ "type": <RegistryId>, "config": { … } }` — the
config nested under its own key, which is the shape
`NamedRunnerComponentSpecV2 { id, config }` emits today. Decoding is a plain
derive plus a match on the id — **no hand-written enum `Deserialize`**:

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
(2026-07-26)** showed the enum encoding (`#[serde(tag="type")]` plus a derived
`#[serde(untagged)]` `RawValue` tail) *compiles but misbehaves* — unknown tags
fail (`RawValue` cannot decode from serde's buffered `Content`), the untagged arm
matches structurally on literal `id`/`config` keys, and a built-in failing
`deny_unknown_fields` silently mis-routes into the tail. Making the enum work
requires a fragile hand-written `Deserialize` (buffer → peek `type` → strip →
dispatch). The `RegistryId`-string discriminant needs none of that.

Further properties of that design:

- **Built-in configs** fold their per-variant semantic invariants (dry_run
  finiteness, dynosim gating, HTTP's control-plane gate) into the parse via
  `#[serde(try_from = "…")]` — no separate post-decode `validate()`.
- **The plugin tail** keeps `Box<RawValue>` decoded by the plugin's own `dyn`
  factory. Because the registry freezes at bootstrap, the id → factory lookup is
  total and immutable for the run, resolved once at prepare. This is the
  load-bearing use of the `RawValue`/`dyn` boundary.
- **Closed core knobs stay enums.** `DispatchMode` and `HopRouting` are not
  plugin categories — there is no tail — so exhaustiveness and typo-catch are
  pure wins. They remain derived enums but route through a shared
  `normalize_ident` seam, fixing a latent bug: they use `rename_all` today and do
  **not** normalize, so `global_hop`/`GLOBAL-HOP` currently fail. Rule of thumb:
  **closed set → enum; open/plugin set → `RegistryId`.**
- `WorkloadRequirements`, `RunContext`, and the transport-swap principle are
  preserved — the match keys on the id string instead of an enum variant.
- **Do not port Python's lazy-import layer.** `PluginEntry.load()` / `class_path`
  / `importlib` / AST-`validate` is Python resolving *implementation code by
  string at runtime*. Rust built-ins are compiled in; real Rust plugins load
  through a `.so`/wasm host loader. Port the *type model*, not the loader.
- Per-category vocabulary is *not* a 1:1 wire port of Python's values. A parity
  audit found endpoints match exactly, but Rust's `transport` set adds
  `grpc`/`dynosim_*`/`dry_run` with no Python `TransportType` plugin, phases use
  `PhaseKind` not `TimingMode`, and the dataset trace path is `file` not
  `custom`. The *mechanism* is the port; the *values* are Rust-native where the
  runtimes differ.

## Migration

One transport family at a time, byte-exact against the mock server at each step.

### Step 1 — a typed `cfg.transport` consumer alongside the projection

For transports there is nothing to introduce: `cfg.transport` is already the
closed typed `Transport` enum, and `Http`/`Grpc` are unit variants with no
payload. Step 1 is therefore "add a consumer that reads `cfg.transport`
directly", run alongside the existing projection, and assert the two produce
identical **components**.

Component equality is the correct assertion, and the only one available.
A `NativeTransportExecution` *binding* cannot be differenced here, because step 1
leaves no `Transport` at the selection boundary (see step 2's prerequisite).
Component equality is nonetheless sufficient: `resolve_native_execution` is

```rust
// online_execution.rs:118-128
let factory = context.product_registry().transport_factory(transport_id)
    .ok_or_else(|| anyhow::anyhow!("transport {transport_id:?} is not registered"))?;
factory.native_execution(transport, context)
```

and nothing else — the binding is a total function of the `{ id, config }` pair
plus a `RunContext` neither path varies. Pinning both halves byte-exact pins the
binding transitively. That is what the landed
`transport_component_matches_inline_projection` asserts (`typed.id == inline.id`
and `typed.config.get() == inline.config.get()` over `all_variants()`, all six),
and it is the differential gate step 1 owes.

**That transitivity is exactly as wide as step 1 and no wider.** It holds because
both step-1 paths end in the same `factory.native_execution(transport, context)`
call, so equal inputs give an equal binding. Step 2 removes that call: obligation
(a) has the match arms supply bindings **directly**, so the shared terminal
function the argument depends on is gone, and a component-equality result carries
nothing about whether a hand-written arm reproduces what the registry lookup
returned. Step 2 therefore owes its own differential — for each variant, the arm
and `registry.transport_factory(id).native_execution(...)` must be shown to
produce the same binding, asserted while both are still reachable — and this
record does not claim otherwise. An earlier formulation here said no
binding-level differential was owed at any step; that generalized a step-1
property past the step that makes it true.

Dispatch is an exhaustive `match` on `Transport`, not `match id.as_str()`; the
`RegistryId` string and the plugin tail belong to §4, not to this field.

### Step 2 — move `native_execution` selection to the exhaustive `Transport` match

**Prerequisite: step 2 needs a typed `Transport` value at the selection boundary,
and after step 1 there is not one.** Step 1's typed producer
(`transport_component(cfg.transport.as_ref())`) still emits a
`NamedRunnerComponentSpecV2 { id, config }`: it serializes the variant, removes
the `"type"` tag, and hands the remainder on as `RawValue`. `AuthoredRunSpecV2`
keeps only `pub transport: NamedRunnerComponentSpecV2` (`protocol_v2.rs:586`),
and the selection site `resolve_native_execution` (`online_execution.rs:118`) has
signature `(&RunContext, &dyn ValidatedTransportConfig, transport_id: &str)`,
called from the five workload bodies that are themselves handed
`&AuthoredRunSpecV2`.

The intermediate this record selects: **`AuthoredRunSpecV2` gains a typed
`transport_typed: Transport` field**, populated by `into_authored` straight from
`cfg.transport`, carried *alongside* the existing projected
`NamedRunnerComponentSpecV2`. The match arms read `run.transport_typed`; the
component spec stays for exactly the id-addressed consumers named in obligations
(b) and (c) below. Both fields die together in step 4 when the struct does, so
the duplication is bounded to steps 2–3 and costs one `Transport` clone per run.
The alternative — hoisting step 4's `BenchmarkRun` repoint ahead of step 2 — is
rejected: it merges the two largest steps and forfeits the differential
assertion step 1 exists to provide.

Obligations:

**(a) Bindings, not just configs.** The registry resolves the transport's
`NativeTransportExecution` binding, so the match arms must supply those bindings
directly or every workload's prepare/`validate_run` breaks with "transport not
registered".

**(b) Descriptors for `--capabilities`.** The catalog is built from registered
factory descriptors, so built-ins must still contribute their
`TransportDescriptor`/`WorkloadDescriptor`.

**(c) A third consumer outside the profile path.**
`CurrentNativeGraphModelBindingResolver::resolve`
(`rust/runtime/src/eval/native_graph/model_runtime.rs:94`, with further lookups
at `:309` and `:389`) calls
`registry.transport_factory(binding.transport_factory_id())` for its
`UnknownTransport` rejection and reads `transport.descriptor().url_schemes` for
`validate_transport_urls`. That is the `aiperf eval --model-runtime`
native-graph path, which the http + grpc **profile** e2e suites do not exercise;
it needs its own verification. Keep the `id → factory` lookup; only
`cfg.transport` selection leaves it.

**(d) Two doc comments assert the opposite property and must be rewritten.**
`registry.rs:215` calls `native_execution` "the seam that makes a transport
*swappable*: the workload asks the registered transport for its execution
binding … never matching on a transport kind", and `registry.rs:236` states
"There is no `match` on a closed transport enum: adding a native transport means
registering a factory that returns its own binding, and nothing in the workloads
changes." Both become false when the match lands.

That rewrite is not a concession, because the openness they describe is already
unreachable from authored config: `Transport` is `#[serde(tag = "type")]` over
six variants, so an out-of-tree `AIPerfExtension` may still call the public
`register_transport` (`registry.rs:530`) with a new id, but no Config v2 document
can select it. Registry openness is real only for the *id-addressed* consumers,
which are exactly obligations (b) and (c).

**(e) The variant↔registration correspondence is per feature set.** Every
in-tree registration (`registry.rs:735`, `:770`, `:788`, `dry_run.rs:582`,
`offline_execution.rs:851-852`) registers exactly one of the six `Transport`
variants, so step 2 opens no selection gap **in a build with every feature on**.
But `Transport` carries **no** `#[cfg]` on any variant
(`config/model/transport.rs:16-49`), so all six always deserialize, while the
factory registrations are feature-gated throughout `registry.rs` (`:758`, `:776`,
`:795`, `:838`, `:1484`, `:1494`, `:1566`, `:1598`, `:2142`, `:2190`, `:2212`) —
`Grpc` behind `grpc`, `Websocket` behind its feature,
`DynosimOffline`/`DynosimOnline` behind `dynosim`. A lean build therefore has
variants that decode and select an id nothing registered, and today's failure for
that case is a registry lookup miss at run time.

This is a reason step 2 *improves* on the status quo, provided it is written to.
The exhaustive match must carry **explicit feature-gated rejection arms**, not a
silent fallthrough: each gated variant gets a `#[cfg(not(feature = "…"))]` arm
returning a named error ("transport `grpc` selected but this binary was built
without the `grpc` feature"), so the diagnostic names the build rather than the
registry. Step 2's gate gains a lean-build compile check, because a match written
against the full feature set is exactly the code that fails to compile, or
silently falls through, when features are subtracted. The standing duty is to
keep the match arms and the registrations in correspondence; the compiler
enforces only the match side.

### Step 3 — collapse the workload DTOs

Repeat for the workload seam: collapse `ScheduledWorkloadConfigV2` /
`GraphWorkloadConfigV2` into typed-optional fields, carrying all **five**
graph-only fields — including the `recorded_agent_default` derivation and the
unconditional-within-graph `system_idle_gap_cap_seconds` attachment (see
[Built](#built); the `weka_semantics` predicate must **not** come back).

**All five have live readers, and `lower_graph` is not the only one.** Four are
read inside `lower_graph` (`online_execution.rs:1215-1282`):
`recorded_agent_default` (`:1251`) gates
`validate_canonical_recorded_agent_bundle(&prepared.bundle)`, so canonical-bundle
validation simply stops running if the flag is lost; `ignore_trace_delays`
(`:1262`) and `system_idle_gap_cap_seconds` (`:1263`) flow into
`NativeGraphDatasetPlan`; `planned_replay_traces` (`:1280`) is assigned to
`plan.planned_replay_traces` after `build_common_plan`. Three of the five also
have readers *outside* it:

- `system_idle_gap_cap_seconds` is read again by `lower_legacy_agentic`
  (`online_execution.rs:1632`), where it becomes
  `PhaseSpec::AgenticReplay { system_idle_gap_cap_seconds, .. }` — the AgentX
  legacy arm.
- `ignore_trace_delays` is read again by `prepare_dynosim_graph`
  (`offline_execution.rs:1543`) into `PreparedDynosimGraphOperation` — the
  dynosim arm, which never enters `lower_graph` at all.
- `weka_semantics` is consumed by **no** lowering function. It is read only in
  the graph workload's `validate_run`, as
  `weka_wants_legacy(workload.weka_semantics.as_deref())`
  (`online_execution.rs:347`, `:381`) — which is what selects
  `lower_legacy_agentic` over `lower_graph` in the first place.

Dropping any of the five is a **silent** behavior loss, not a decode error. Note
also that `lower_graph` takes `&AuthoredRunSpecV2` and reads
`run.endpoints.identities()` and `run.identity.random_seed`, and passes `run` on
to `build_common_plan` — further step-4 repoint surface.

### Step 4 — delete the projection and repoint `coordinator.rs`

Delete `AuthoredRunSpecV2`, `into_authored`, `NamedRunnerComponentSpecV2`, and
`BenchmarkRunWireV2`; point `coordinator.rs` at **`BenchmarkRun`**. The bare
stdin arm is simply re-typed — but only *after* §2's port list lands on
`BenchmarkRun`. Deleting the DTO before those land is the silent-regression path.
The protocol-v2 request/response module is reduced to the `EnvelopeV2` outer
shape and the diagnostic/result types.

**The `validate_run` seam is not one body, and not even one trait.** There are
**two** traits: `NativeTransportExecution::validate_run(&self, run, context)`
(transport-level, 2-arg) and `WorkloadFactory::validate_run(&self, run, context,
transport, workload, transport_id)` (workload-level, 5-arg, `registry.rs:293`).
The verified inventory is seven-plus bodies:

- **Transport-level.** `online_execution.rs:105` (http) consumes only
  `models.items` and `sidecars.live_streaming`. `grpc_execution.rs:73` →
  `validate_grpc_run` (`:85`) consumes `context.default_endpoint_profile()`,
  `context.endpoint_profiles()`, and `profile.config.urls`, and rejects **all**
  sidecars. `ws_execution.rs:151` consumes `run.artifacts.trace` and five sidecar
  fields (rejecting trace artifacts and all sidecars). `dry_run.rs:539` consumes
  `run.dispatch` and `run.workload.id.as_str()` (rejecting sharded dispatch and
  graph workloads under virtual workers).
- **Workload-level.** `online_execution.rs:228` (scheduled) delegates to the
  transport binding, or falls through the `dynosim_or_unsupported!` macro
  (`online_execution.rs:135-151`) to
  `offline_execution::dynosim_scheduled_validate_run` (`:887`, which requires
  `workload.worker_count == 1`), then runs `validate_authored_tokenizer`.
  `online_execution.rs:324` (graph) is the body that consumes
  `run.sidecars.live_streaming`, at `:337`. `online_execution.rs:447` (static
  accuracy) requires `transport_id == "http"` and consumes
  `run.models.items.len()`.

Each body repoints on its own terms. A repoint audited against
`online_execution.rs:105` alone silently changes the gRPC, WebSocket, dry-run,
dynosim, graph, and static-accuracy rejection surfaces.

Of `into_authored`'s own content, only `workload_kind` and `worker_count` from
`available_parallelism` are plain copies or derivations of the typed fields. The
rest is not: `planned_replay_traces` has no `BenchmarkConfig` home;
`parse_dispatch_mode` branches on `runtime.cells` rather than reading the field;
the endpoint-profile transform injects an `id` no typed field carries; and the
mandatory-section rejections, the tokenizer rejections, the three
`.unwrap_or_default()` fallbacks (`metrics`, `artifacts`, `export`), the
export-stem derivation, `resource_presence`, and the `workers`/`phases`
decode-shape rejections are validation or transform branches.

**Treat §2's port list as a floor, not a closed enumeration.** An earlier form of
this paragraph asserted that all nontrivial branches were enumerated there. That
assertion was false when written — the `artifacts` and `export` fallbacks above,
and the `id` injection in §2's endpoint entry, were all missing — and it is the
kind of claim that stops the next reader from re-walking the function. Three
separate audits of `into_authored` each missed a behavior of the same class:
something that lives in the function's body or in the shape it hands the factory
rather than in a field of any struct. The migration owes a fresh statement-level
walk of `into_authored` at the start of step 4, checking every `?`, every
`unwrap_or_default`, every `insert`/`remove` on a `Map`, and every value
constructed rather than copied — not a re-read of this list.

## Verification gates

Every step runs, from `rust/` with the project venv active:

```bash
cargo fmt --check && cargo clippy --all-targets
cargo test -p aiperf-runtime && cargo test -p aiperf-runtime --features engine
```

`cargo test -p aiperf-runtime` alone runs **zero** engine tests — the `engine`
feature gates the entire projection this record changes — so both invocations are
mandatory at every step, not just the last. Beyond that shared floor:

**Step 1.** The component-equality differential assertion described above, plus
`cargo test -p aiperf-e2e-tests --test test_default_behavior --test
test_chat_endpoint --test test_completions_endpoint --test
test_kserve_grpc_endpoint`.

**Step 2.** The step-1 gate, plus the arm↔registry binding differential named in
step 2's prerequisite, plus `cargo test -p aiperf-dry-run-tests --test dry_run
--test virtual_workers` and `cargo test -p aiperf-e2e-tests --test test_websocket`
— the two payload-bearing arms any suite exercises today, `Http` and `Grpc` being
unit variants — plus `cargo test -p aiperf-e2e-tests --test
test_harbor_native_graph_rollout` for obligation (c)'s `aiperf eval
--model-runtime` path. Plus a lean-build compile check (`cargo check -p
aiperf-cli --no-default-features`) alongside the feature-bearing builds in
`CLAUDE.md`, for obligation (e).

**Neither `DynosimOffline` nor `DynosimOnline` is executed by any of that**, and
they are the two arms most exposed to a match rewrite: both carry a
`DynosimConfig` payload with nested `RawValue` engine/router args, so unlike the
unit variants they have a config half that a hand-written arm can get wrong. No
e2e or dry-run suite runs either — `grep -rl dynosim` over `e2e-tests/` and
`dry-run-tests/` matches one file, `global_dispatch_real_clock.rs`, and only in
a doc comment. Step 2's gate is therefore incomplete until it adds a
`--features dynosim` run that executes both arms. Until such a suite exists, the
minimum is the binding differential above run over `all_variants()` under
`--features dynosim`, which at least covers the config half without a live
Dynamo target; a socket-free `dynosim_offline` profile run is the real gate and
this record names it as owed work rather than pretending the arms are covered.

**Step 3.** The step-2 gate, plus `cargo test -p aiperf-e2e-tests --test
test_conditional_graph --test test_flatgraph_parity --test
test_ignore_trace_delays --test test_recorded_agent_replay --test
test_dag_full_topology --test test_system_idle_gap_cap`.

`test_ignore_trace_delays` is the named guard for `ignore_trace_delays`.
`recorded_agent_default` and `system_idle_gap_cap_seconds` had no dedicated e2e
target, and both are silent losses rather than decode errors, so two were written:
`recorded_agent_default_scenario_rejects_non_canonical_bundle`
(`e2e-tests/tests/test_recorded_agent_replay.rs`) and
`e2e-tests/tests/test_system_idle_gap_cap.rs`. Reaching either check required
fixing the path to it, which found three live defects:

- `--weka-semantics` was dropped under `--config`. `yaml.rs` built
  `Inputs::weka_semantics` as `None` and never overlaid the flag, so a
  config-authored `scenario: recorded-agent-default` (timing mode
  `agentic_replay`) always resolved to `legacy`, and legacy lowering requires a
  `hugging_face` dataset source — making the scenario unreachable from a config
  file over a file dataset.
- `apply_scenario_synthesis` materialized a synthesis block for *every* scenario
  under graph-ir. `RecordedAgentDatasetInput` is `deny_unknown_fields` with no
  `synthesis` field, so `recorded-agent-default` — whose four synthesis fields
  are all `None` — failed to decode outright.
- The idle cap's guard had to be written on the **graph-ir** arm, since `legacy`
  lowering needs a HuggingFace download and cannot run offline against the mock
  server. That exposed the projection guard `3f77a3adac` fixed.

**Step 4.** The full step-3 gate, plus the cellular suites that exercise
`planned_replay_traces` and the bare-run stdin arm: `cargo test -p
aiperf-e2e-tests --test test_cellular --test test_graph_cellular --test
test_grpc_cellular --test test_recorded_agent_cellular --test
test_cellular_dataset_shipping`. A step-4 change that has not run
`test_graph_cellular` and `test_recorded_agent_cellular` is not verified,
regardless of what else is green.

**That e2e gate exercises the repoint, not the port list.** Every suite named
above is an end-to-end profile run; none asserts `deny_unknown_fields`, the
dataset cardinality tightening, the export-stem derivation, `variation:
Option<VariationSpec>`, or the `resource_presence` algorithm. A step-4 change
could land all five wrong and still be green. Worse, the closest thing the tree
has to such a test does not run: `protocol_v2.rs:1284` declares
`#[cfg(any())] mod tests` — an always-false cfg that compiles that module
(`:1284-1408`) out — and among its bodies is
`outer_contract_rejects_unknown_fields`, i.e. the one existing unknown-field
assertion has been silently disabled since `904cc07e2a`. The scope of that
disablement is exactly that module: the `#[cfg(test)] mod dispatch_mode_tests`
that follows at `:1410-1411` is live, so the dispatch and hop-routing derivations
*are* covered and only the outer-contract bodies are dark. Step 4 therefore adds **named unit tests on `BenchmarkRun`
itself**, in addition to the e2e gate:

- an unknown top-level field on a bare run payload is rejected (replacing the
  disabled `outer_contract_rejects_unknown_fields`, on the typed model);
- `BenchmarkRun::validate()` rejects empty `benchmark_id`, empty `artifact_dir`,
  and a two-dataset `cfg.datasets`, the last with a message naming the count;
- a run whose `artifacts.records_path` carries a custom stem names the
  per-record, summary, and timeslice artifacts from that same stem;
- `variation` round-trips as a typed `VariationSpec`, not an open `Value`;
- a `resolved` object missing `artifact_dir_created` is rejected while one
  missing any of the fourteen `Option` fields still decodes (the exact shape of
  the `Resolved` narrowing), and an omitted `resolved` section still defaults;
- `resource_presence` for a config with `models`/`endpoints`/`metrics`/
  `artifacts` absent and `sidecars: Some(Sidecars::default())` matches
  `into_authored`'s classification — four `true` and `sidecars: false`;
- a run with `cfg.models` absent, one with the default `cfg.endpoint` absent, and
  one with `cfg.transport` absent are each rejected with the message
  `into_authored` used, and a run whose `metrics.slos` carries a non-numeric
  threshold is *rejected* naming the offending key rather than silently falling
  back to `MetricsSpec::default()`;
- a run with `cfg.tokenizer` absent is rejected, as are one with a
  whitespace-padded `tokenizer.name` and one with a blank `tokenizer.revision`,
  each carrying `AuthoredTokenizerV2::decode`'s message;
- `runtime.dispatch` absent with `runtime.cells > 1` resolves `Sharded`, absent
  with `cells <= 1` resolves `Global`, and an explicit value wins over both —
  ported from the live `mod dispatch_mode_tests` (`protocol_v2.rs:1410-1411`).
  They assert the projection, not cell behavior: `build_cell_envelope` rewrites
  `cells` to `1`, so no cell reaches the `Sharded` arm;
- a config authoring two override `endpoint_profiles` in reverse-sorted key order
  produces profiles in sorted order after `"default"`, and each override still
  has `timeout` renamed to `timeout_seconds` and `url_strategy` dropped;
- `weka_semantics` accepts `" Legacy "` and `"AgentX"` as legacy, `""` and
  absent as graph-ir, and rejects an unknown value with
  `weka_wants_legacy`'s message;
- a run with `cfg.runtime.workers: Some(0)` is rejected with
  `"run.cfg.runtime.workers must be a positive usize"`; a run with `cfg.runtime`
  absent resolves `worker_count` to `default_worker_count()` rather than `1`; and
  a run with `cfg.phases` absent and a run with `cfg.phases: Some(vec![])` are
  both rejected by `validate()` with the same message.

Re-enabling or deleting the `#[cfg(any())] mod tests` at `protocol_v2.rs:1284`
is part of step 4, not optional cleanup: leaving a disabled module next to the
code it was written to guard is how the gap recurs. This applies to that module
alone — `mod dispatch_mode_tests` at `:1410` needs porting, not re-enabling.

Each step keeps the stdin accept path *structurally* intact — both arms of
`decode_execute_wire` survive; the bare arm changes type. No step owes anything
to an external consumer, so the property each step must hold is behavioral, not
byte-level: strictness, validation, and controller-derived state carried forward
per §2's port list, and identical observed run output against the mock server.

## Non-goals and trade-offs

This is a correctness/type-safety refactor, deliberately made with eyes open:

- **We reintroduce a match on the transport kind — an exhaustive one.** The
  current code prides itself on "never matching on a transport kind" via `dyn`
  (the two doc comments step 2 rewrites). The typed design matches on the closed
  `Transport` enum exactly once, at the selection boundary, and on
  `workload_kind`'s two arms for workloads. Exhaustiveness is the point and the
  compiler enforces it; the cost is the standing duty from step 2's obligation
  (e) to keep the arms and the feature-gated registrations in correspondence.
- **The runtime crate gains a compile dependency on every built-in component
  config type.** That is the cost of typing built-in configs, and it is
  acceptable because the built-in set is frozen at compile time. The dependency
  is bounded by the closed variant set: it grows only when
  `Transport`/`workload_kind` grows. A future plugin arm would reintroduce an
  untyped tail rather than extend this coupling.
- **Not touched:** the `dispatch` seam
  (`Dispatchable`/`RequestSink`/`RequestObserver`), the `Clock` seam, the
  phase/scheduling runtime, metrics, exporters' output logic, and the stdio
  accept path. This change is about *config decode and component selection*, not
  the hot path.
- **Dynamic plugins are future work, and this migration does not deliver a
  selectable tail.** AIPerf may grow runtime-loaded transports/workloads (WASM /
  subprocess / `abi_stable`). Those cannot be typed configs — their type is
  unknown at the core's compile time — so §4's string id plus `RawValue` config
  is the shape they will need.

  As specified, though, **the migration leaves no authored surface that can name
  a plugin.** That claim needs an authored object carrying
  `{ RegistryId, RawValue }` to survive, and there is not one: §1 selects
  transports from the closed `BenchmarkConfig.transport` enum and workloads from
  an exhaustive match, §4 identifies `NamedRunnerComponentSpecV2` as the
  `{ RegistryId, RawValue }` seam, and step 4 deletes it. What remains is a
  registry that still *accepts* `register_transport` with any id
  (`registry.rs:530`) and still resolves `transport_factory(id)` — but with no
  config that can select one, because `cfg.transport` fails to decode on an
  unknown tag.

  So the honest state is: **registry openness survives for id-addressed
  *internal* consumers, and authored selection of a runtime-loaded plugin exists
  neither before nor after this migration.** This is recorded as a *requirement
  on the eventual plugin work*, not a property this design delivers: whenever
  dynamic plugins land, `BenchmarkConfig.transport` (and the workload selector)
  must grow an explicit open tail variant — a
  `Plugin { id: RegistryId, config: Box<RawValue> }` arm — and that arm is what
  reconstitutes the deleted seam. Until it exists, no config can select a plugin,
  and this record must not be read as evidence that one can. Adding the arm is
  out of scope here; recording that it is *required*, and that its absence is a
  gap rather than a design property, is in scope.
- **"Zero `RawValue`" is a direction, not a literal end state.** Three uses
  survive this migration. Two are inner adapter inputs: the `dynosim` transport
  variants' nested Dynamo engine/router args (opaque pass-throughs to Dynamo's
  own parser), and the dataset payload inside the workload arm (dataset
  selection is name/structural-probe, not a component-config union). The third
  is the **sidecar adapter boundary**, and it is a different thing: a live
  `{ open id, RawValue }` seam, not an opaque leaf — see §2's port entry. A
  runtime-loaded plugin's config would be a fourth, when such a plugin can be
  selected. This change types the built-in component majority and the
  first-party inner seams; it does not chase `RawValue` out of the places it
  belongs, and it does not remove the sidecar seam.

## Source anchors

- `rust/runtime/src/engine/protocol_v2.rs` — `BenchmarkRunWireV2`,
  `AuthoredRunSpecV2`, `into_authored`, `NamedRunnerComponentSpecV2` (the
  projection to delete). `BenchmarkRunWireV2`'s `deny_unknown_fields`,
  `validate_outer`, `planned_replay_traces`, and `VariationSpec` typing port onto
  `BenchmarkRun` first (§2). `EnvelopeV2` is an in-process struct constructed
  after decode, not a wire shape.
- `rust/runtime/src/engine/coordinator.rs` — `envelope.run.into_authored()`, the
  child composition root to repoint at **`BenchmarkRun`**, with authored sections
  reached as `run.cfg`.
- `rust/runtime/src/engine/registry.rs` — `TransportFactory`/`WorkloadFactory`,
  `ValidatedTransportConfig`/`ValidatedWorkloadConfig`, `WorkloadRequirements`,
  `native_execution` (the factory seam to make typed), the unified
  `WorkloadConfigV2` (`:855`) and its aliases (`:902`, `:904`),
  `validate_common_workload` (`:1650-1666`).
- `rust/runtime/src/engine/online_execution.rs` — `resolve_native_execution`
  (`:118-128`), `lower_graph` (`:1215-1282`), `lower_legacy_agentic`
  (`:1364-1664`), and the workload-level `validate_run` bodies.
- `rust/runtime/src/extensions/mod.rs` — `AIPerfRegistry` capability accessors
  (the registry surface that shrinks); `extensions/registry_id.rs` —
  `RegistryId`.
- `rust/runtime/src/config/model/` and `rust/runtime/src/config/resolve.rs` — the
  typed `BenchmarkConfig`/`BenchmarkRun`, the landed `Transport` union
  (`model/transport.rs:16-49`), `Resolved` (`model/resolved.rs:13-68`), and the
  resolver step that gains typed `synthesis`/`weka_semantics`/`failure_policy`.
- `docs/specs/config-model-unification.md`, `docs/specs/runner-protocol.md`,
  `docs/specs/extension-registry.md` — the records this one completes and amends.
- Python reference (origin/main): `src/aiperf/plugin/extensible_enums.py`
  (`ExtensibleStrEnum` — the discriminant blueprint: base members plus runtime
  `_extensions`, `_normalize_name` lookup), `src/aiperf/plugin/categories.yaml`
  (per-category `protocol`/`enum`/`metadata_class`), `src/aiperf/plugin/types.py`
  (`PluginEntry` — lazy `class_path` load plus `metadata: dict` validated via
  `get_typed_metadata`; the layer NOT ported).
