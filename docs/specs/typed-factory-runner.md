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

What the change delivers is the config payload typed end to end: no `Value`
round-trip, no per-factory re-decode, decode-time errors, and the opaque
`RawValue` seam confined to the two residual uses listed under
[Non-goals](#non-goals-and-trade-offs).

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
- `weka_semantics` → a closed enum. It needs `#[serde(alias)]` plus
  normalization to preserve today's leniency across `graph-ir`, `graphir`, and
  `graph_ir`; the fold is **lower + `-`→`_` only** (Python does not strip
  separators, so `graphir` is a hardcoded alias, not normalization).
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

**`variation: Value` → `Option<VariationSpec>` — re-type `BenchmarkRun`.**
`BenchmarkRun` holds `variation` as an open `Value` where the wire DTO had a
typed `VariationSpec`; here the wire DTO is the stricter one, so the re-type is
the correct direction and belongs in this change rather than being inherited
as-is.

**`trial: usize` → `u32` and `variables: BTreeMap` → `serde_json::Map` — accept
`BenchmarkRun`'s shapes.** These are the typed model winning, which is the point.
`variables`' ordering and duplicate-key behavior under `serde_json::Map` is a
detail step 4 should confirm rather than assume.

**`resolved: Value` → `Resolved` — accept it, but record it as a tightening.**
`Resolved` (`config/model/resolved.rs:13-46`) carries **no** per-field
`#[serde(default)]`; its module doc states the contract outright: "Every field is
present in the wire object, including nulls". Its `impl Default` (`:48-68`) only
serves `BenchmarkRun::resolved`'s own `#[serde(default)]`, i.e. omitting the
whole section. So the re-type accepts a strictly narrower set of payloads than
`resolved: Value` did: a `resolved` object present but missing *any* one of its
sixteen keys — including the `Option` ones, which without `serde(default)` are
required-but-nullable — decodes today and is rejected after. That only bites the
bare-resolved-run arm kept for external harnesses; AIPerf's own authoring arm
re-projects a full `Resolved`. It is the right direction and the greenfield
premise carries it, but it is a behavior change, and step 4 should confirm the
narrowing with an actual decode test rather than by reading field attributes.

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

`Tokenizer` (`config/model/tokenizer.rs:15-38`) is a plain derived struct with a
mandatory `name` but **no** semantic validation, and `BenchmarkConfig.tokenizer`
is `Option<Tokenizer>`, so all three rejections vanish with the decode. Both
`ensure!` messages appear nowhere else in the tree. **Port all three into
`BenchmarkRun::validate()`, messages included**, in the same batch as the
models/endpoint/transport rejections; the absent-section case joins them as a
fourth mandatory section.

**`parse_dispatch_mode`'s cellular-aware default — port the derivation, not
`unwrap_or_default()`.** `parse_dispatch_mode` (`protocol_v2.rs:260-272`) is not
`runtime.dispatch.unwrap_or_default()`. An explicit `runtime.dispatch` wins, but
when it is absent the default branches on `runtime.cells`: `cells > 1` resolves
`DispatchMode::Sharded`, and `cells <= 1` resolves `DispatchMode::default()`
(`Global`). The rationale is recorded in the function's own doc comment: a
cellular run has already forfeited single-process byte-exact determinism, so
`Global`'s shared cross-thread admission gate inside a cell buys parity that is
already gone and costs pure overhead — measured ~7-8x slower than `Sharded` in
cellular mode on a c4-144.

A typed reader that writes `cfg.runtime.dispatch.unwrap_or_default()` therefore
silently switches **every cellular run without an explicit dispatch** from
`Sharded` to `Global`, which is a large performance regression rather than a
failure, and no gate catches it: the three exact unit tests for this derivation
(`runtime_dispatch_defaults_to_global_when_absent`,
`runtime_dispatch_defaults_to_sharded_for_cellular`,
`runtime_explicit_dispatch_wins_over_cellular_default`) sit inside the
`#[cfg(any())] mod tests` at `protocol_v2.rs:1284` and never compile, and none of
the five cellular e2e suites authors `runtime.dispatch` at all. The derivation
moves onto the typed model verbatim, and those three tests move with it — this is
the second reason re-enabling that module is a step-4 obligation rather than
cleanup.

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

**Endpoint and model transforms — mostly no-ops, with one live exception.**
`endpoint_profile` (`protocol_v2.rs:552-563`) renames `timeout` →
`timeout_seconds` and removes `url_strategy`; `models_from_config` (`:507-521`)
retains only `name`/`weight`. Against the typed model these are no-ops:
`Endpoint` (`config/model/endpoint.rs:117`) already stores `timeout_seconds`
(`:138`) and has neither a `timeout` nor a `url_strategy` field — `url_strategy`
exists only in the authoring layer (`cli/src/flags.rs:368`, `cli/src/yaml.rs:915`,
consumed and validated at `yaml.rs:1690`) and never reaches `BenchmarkConfig` —
and `ModelItem` (`config/model/models.rs:22`) has only `name` and `weight`. For
the **default** profile (`serde_json::to_value(&cfg.endpoint)`,
`protocol_v2.rs:446`) and for models, the migration may simply drop these
transforms.

They are **not** no-ops for the override profiles. `cfg.endpoint_profiles`
(`config/model/config.rs:118`) is an open `serde_json::Map<String, Value>`, not a
typed section, and `into_authored` feeds it through the same `endpoint_profile`
at `protocol_v2.rs:448-451`/`:547`. An authored override may therefore still
carry `timeout` and `url_strategy` keys, and the rename/removal is live for it.
The migration must keep the transform on that open map — or type
`endpoint_profiles` as a `BTreeMap<String, Endpoint>`, which is a larger change
than this record scopes.

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
and it is the differential gate step 1 owes. **No binding-level differential is
owed at any step.**

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
and the mandatory-section rejections, the tokenizer rejections, the metrics
fallback, the export-stem derivation, `resource_presence`, and the
`workers`/`phases` decode-shape rejections are validation or transform branches.
All are enumerated in §2's port list.

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

**Step 2.** The step-1 gate, plus `cargo test -p aiperf-dry-run-tests --test
dry_run --test virtual_workers` — the only payload-bearing built-in transport arm
any suite exercises, since `Http` and `Grpc` are unit variants — plus `cargo test
-p aiperf-e2e-tests --test test_websocket` for the
`Websocket(WebSocketTransportConfig)` arm, plus `cargo test -p aiperf-e2e-tests
--test test_harbor_native_graph_rollout` for obligation (c)'s
`aiperf eval --model-runtime` path. Plus a lean-build compile check
(`cargo check -p aiperf-cli --no-default-features`) alongside the feature-bearing
builds in `CLAUDE.md`, for obligation (e).

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
`#[cfg(any())] mod tests` — an always-false cfg that compiles the entire module
out — and among its bodies is `outer_contract_rejects_unknown_fields`, i.e. the
one existing unknown-field assertion has been silently disabled since
`904cc07e2a`. Step 4 therefore adds **named unit tests on `BenchmarkRun`
itself**, in addition to the e2e gate:

- an unknown top-level field on a bare run payload is rejected (replacing the
  disabled `outer_contract_rejects_unknown_fields`, on the typed model);
- `BenchmarkRun::validate()` rejects empty `benchmark_id`, empty `artifact_dir`,
  and a two-dataset `cfg.datasets`, the last with a message naming the count;
- a run whose `artifacts.records_path` carries a custom stem names the
  per-record, summary, and timeslice artifacts from that same stem;
- `variation` round-trips as a typed `VariationSpec`, not an open `Value`;
- a `resolved` object present but missing one key is rejected (the `Resolved`
  narrowing), and an omitted `resolved` section still defaults;
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
  the three tests currently stranded in `#[cfg(any())] mod tests`;
- a run with `cfg.runtime.workers: Some(0)` is rejected with
  `"run.cfg.runtime.workers must be a positive usize"`; a run with `cfg.runtime`
  absent resolves `worker_count` to `default_worker_count()` rather than `1`; and
  a run with `cfg.phases` absent and a run with `cfg.phases: Some(vec![])` are
  both rejected by `validate()` with the same message.

Re-enabling or deleting `#[cfg(any())] mod tests` is part of step 4, not optional
cleanup: leaving a disabled module next to the code it was written to guard is
how the gap recurs.

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
- **"Zero `RawValue`" is a direction, not a literal end state.** Two uses survive
  this migration: the `dynosim` transport variants' nested Dynamo engine/router
  args (opaque pass-throughs to Dynamo's own parser), and the dataset payload
  inside the workload arm (adapter input; dataset selection is
  name/structural-probe, not a component-config union). A runtime-loaded
  plugin's config would be a third — when such a plugin can be selected. This
  change types the built-in component majority and the first-party inner seams;
  it does not chase `RawValue` out of the places it belongs.

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
