<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Config model unification

## Purpose

This record states the target for the Config-v2 front end: **one strongly-typed
`BenchmarkConfig`/`BenchmarkRun` model, shared by the CLI producer and the runtime
consumer, serialized to itself across the process boundary** — mirroring the
origin/main Python Pydantic `AIPerfConfig` → `build_benchmark_plan` → `BenchmarkRun`
pipeline. The goal is to remove the untyped projection seam where a producer can
emit a config field the strict consumer rejects only at execute time, and to
delete the duplicate, partially-typed reimplementations that bracket the model
today. [runner-protocol.md](runner-protocol.md) records the boundary as built;
this record states how it is meant to converge.

## Built

All four migration steps below have shipped and are merged into `ajc/rust-dag-v3`.
The front end is now typed end to end on the input side; the residue is on the
output side, where the typed config is re-projected into the factory-owned
authored seam.

**Producer (CLI) — two authoring constructors, one resolution.** `aiperf profile`
normalizes both input schemas — `ProfileFlags` (a `clap` struct, via
`cli/src/load.rs`) and the YAML serde schema `ConfigFile`/`*Section` (via
`cli/src/yaml.rs`) — into the flat `Inputs` bag, which now lives in
`runtime/src/config/resolve.rs`. The CLI no longer lowers: `pub fn
resolve(Inputs) -> BenchmarkRun` is the single semantic resolution, owned by the
runtime, and `cli/src/load.rs` and `cli/src/model/mod.rs` are re-export shims
over `aiperf_runtime::config`. The two mappers still enumerate the config
surface twice — that duplication is inherent to having two authoring surfaces
and is mirrored in Python (`convert_cli_to_aiperf` + `load_config`) — but they
converge one step earlier than before, so drift can no longer produce two
different resolutions.

**Wire — authoring crosses the boundary, the runtime resolves.** The `--execute`
child receives `{"authoring": <Inputs>}` (`AuthoringWireV2`,
`protocol_v2.rs:155`), decoded by `decode_execute_wire` (`protocol_v2.rs:191`),
which resolves it in the runtime. Every profile path ships authoring: single
run, flag sweep, YAML sweep, the four adaptive-search loops, recipe sweeps, and
cellular cells. The resolved-wire sender was deleted; `decode_execute_wire`
retains a bare-`BenchmarkRunWireV2` acceptance path for payloads that arrive
already resolved. Sweeps and searches override the swept axis on a clone of the
authoring `Inputs` (`search::apply_override_inputs`, `profile.rs:1308`), so a
swept run and a hand-authored equivalent are byte-identical by construction.
`BenchmarkRunWireV2.cfg` is the typed `BenchmarkConfig` under
`deny_unknown_fields`; the loose `BenchmarkConfigWireV2` (a `Value` per section)
is deleted.

**The remaining seam is `into_authored`, and it still hand-builds one blob.**
`BenchmarkRunWireV2::into_authored` (`protocol_v2.rs:353`) adapts the canonical
config nesting to the linked-factory seam. It classifies via the typed
`workload_kind(&cfg)` rather than by sniffing the dataset type, which is the
intended target shape. But it re-serializes each typed section back to
`serde_json::Value` and assembles the workload config as a `serde_json::json!`
blob, and the per-workload strict DTOs (`ScheduledWorkloadConfigV2`,
`GraphWorkloadConfigV2`, `StaticAccuracyWorkloadConfigV2`, each
`deny_unknown_fields`) still decode that blob from `Box<RawValue>` in
`online_execution.rs`. Consequently the `weka_semantics` hazard is contained by
an explicit `if workload_kind == WorkloadKind::Graph` guard
(`protocol_v2.rs:404`) rather than made structurally impossible: the graph-only
fields (`weka_semantics`, `ignore_trace_delays`, `recorded_agent_default`,
`planned_replay_traces`, `system_idle_gap_cap_seconds`) are attached only inside
that branch, because emitting any of them on the scheduled DTO — even as `null`
— fails its strict decode. Misplacement is now a typed-source mistake in one
function instead of an untyped-blob mistake anywhere, but it is still caught at
execute time, not at authoring time.

**Validation is ported but not wired.** `runtime/src/config/validate.rs` holds
the raise-only cross-field invariants ported from the Python
`@model_validator(mode="after")` methods, entered through `pub fn validate(cfg:
&BenchmarkConfig)`. It has **no production callers** — the only references are
its own `mod tests`. `aiperf config validate` (`cli/src/config/mod.rs:138`) still
runs `yaml::resolve` alone, a YAML schema check. The runtime's
`OperationV2::Validate` remains implemented and unspawned. So the offline
cross-field pass exists and is tested, but nothing in a real run or a real
`config validate` invocation executes it.

## Future requirements

The convergence target and its migration. All four steps shipped on
`ajc/config-model-unification` and are merged into `ajc/rust-dag-v3`; the two
places where the built result stops short of the target are called out per step
below and carried as the remaining work in this record.

**Target shape.** An `aiperf_runtime::config` module (an always-compiled module
in the runtime crate — `aiperf-cli` consumes it through its existing
`aiperf-runtime` dependency, so no separate crate is introduced; a leaf crate was
tried first and reverted because it bought no isolation cli didn't already have
and forced a `DispatchMode` dependency cycle) owns:

- `AiperfConfig` (envelope: `schema_version`, `benchmark`, `sweep`, `multi_run`,
  `variables`, `random_seed`).
- `BenchmarkConfig` — one flat model: `models`, `endpoint`, `endpoint_profiles`,
  `tokenizer`, `runtime`, `metrics`, `artifacts`, `export`, telemetry sidecars,
  `accuracy`, `failure_policy`, `scenario`, plus `datasets: Vec<Dataset>` and
  `phases: Vec<Phase>` as `#[serde(tag = "type")]` internally-tagged enums.
  Workload-specific fields (e.g. `weka_semantics`) are typed fields on the one
  model, consulted where relevant — impossible to misplace because there is one
  model.
- `BenchmarkRun { benchmark_id, sweep_id, cfg: BenchmarkConfig, variation, trial,
  artifact_dir, random_seed }` and `BenchmarkPlan { configs: Vec<BenchmarkConfig>,
  variations, variation_seeds, trials, … }`.
- `fn resolve(flags, yaml) -> AiperfConfig` (flag overlay → one validate),
  `fn build_benchmark_plan(AiperfConfig) -> BenchmarkPlan`, and
  `fn workload_kind(&BenchmarkConfig) -> WorkloadKind` (computed, not a type).

The wire is typed with `deny_unknown_fields`; the child deserializes the same
model the producer emits. `BenchmarkConfigWireV2` is removed and the executor
classifies through `workload_kind()` rather than by sniffing the dataset type —
both built. The target also called for removing `into_authored` and the
per-workload `*WorkloadConfigV2` DTOs so the executor matches directly on typed
`phases`/`datasets`/`transport`; that half is **not** built. Both survive, and
their removal was deliberately spun out into
[typed-factory-runner.md](typed-factory-runner.md), which owns the completion of
this arc: the runtime consuming `BenchmarkConfig` directly and selecting a
component by `RegistryId` whose config is a typed struct for built-ins and an
opaque `RawValue` only for the runtime-plugin tail. That record is
forward-looking and states it is not built. Until it lands, the typed config is
re-projected into `Value` on the way out and graph-only fields need the
hand-written workload-kind guard. "Unknown component id fails closed" is preserved by a serde
enum that errors on an unknown tag.

**Migration (each step ships green):**

1. **Unify the wire type.** *(Built in part.)* `BenchmarkConfig`/`BenchmarkRun`
   moved into the `aiperf_runtime::config` module and the runtime deserializes
   that type directly; `BenchmarkConfigWireV2` and the dataset-type sniff are
   deleted, replaced by typed access + `workload_kind()`, and wire round-trip
   tests guard it. The `json!` blob, the graph-guard, and the per-workload DTOs
   were **not** deleted here; that work moved to
   [typed-factory-runner.md](typed-factory-runner.md). The
   `weka_semantics` bug class is therefore narrowed, not eliminated: the field
   is a typed field on the one model and can only be misplaced inside
   `into_authored`'s single workload-kind branch (`protocol_v2.rs:404`), but a
   misplacement still fails at execute-time strict decode rather than at
   authoring time.
2. **Collapse the producer front-end.** *(Built via the no-lower architecture.)*
   The CLI no longer lowers: `Inputs`, `build`, `phase_validate`, and `redact` moved
   into `aiperf_runtime::config::resolve` (`pub fn resolve(Inputs) -> BenchmarkRun`).
   The `--execute` wire carries the authoring `Inputs` (`{"authoring": <Inputs>}`);
   `decode_execute_wire` resolves it in the runtime before `into_authored`. Every
   profile path — single run, flag sweep, YAML sweep, the four adaptive search
   loops, recipe sweeps — ships authoring `Inputs`; the resolved-wire sender and
   branch were deleted. This mirrors Python (runner resolves; the two authoring
   constructors, flags→`Inputs` and YAML→`Inputs`, remain, as Python keeps
   `convert_cli_to_aiperf` + `load_config`). Behavioral A/B parity vs base was
   proven byte-identical for single, sweep, recipe, and adaptive runs against the
   mock server (only non-deterministic timing/ids differ); it also fixed a latent
   bug where the resolved-cfg patch left the `input_config` echo stale.
3. **One sweep plan.** *(Built.)* All sweep/search override mechanisms are unified
   onto the authoring layer: `search::apply_override_inputs` sets the swept axis on
   the authoring `Inputs` (correct pre-resolution — resolution then expands scalars
   like `isl:128` to a Distribution), and every sweep/search path ships that
   authoring `Inputs` for the runtime to resolve, retiring the three prior
   `Value`/`ProfileFlags`/resolved-cfg mutation mechanisms. The typed
   `build_benchmark_plan` grid/zip/magic-list seam remains available. Adaptive
   search stays a runtime ask-tell loop.
4. **Validators as offline passes.** *(Built: nine raise-only invariants ported to
   `config::validate` — the seven cross-field checks plus `cache_bust_compatibility`
   (full) and `agentic_cache_warmup` (no-scenario branch; the scenario branch needs a
   runtime scenario-registry `timing_mode` lookup with no config-time representation,
   as in Python, so it defers). The authored `timing_mode`/`cache_bust` fields were
   added to the typed model, skip-serialized so the wire stays byte-identical.
   **The wiring half is not built.** `config::validate::validate` has no
   production callers — its only references are its own `mod tests`.
   `aiperf config validate` (`cli/src/config/mod.rs:138`) still runs
   `yaml::resolve` alone, and `OperationV2::Validate` is still never spawned, so
   no real run and no real `config validate` invocation executes the ported
   invariants.)* Port the raise-only cross-field invariants
   (phase↔dataset compatibility, prefill⇒streaming, cache-bust, agentic-warmup) as
   validate-time functions; keep the mutating ones (tokenizer/seed defaults) as
   resolution passes; wire `aiperf config validate` to run them offline.

The `scenario` resolver (lookup-and-stamp) and the adaptive-search Bayesian loop
are mirrored, not redesigned. Risk concentrates in step 1's wire compatibility
(round-trip tests) and preserving unknown-id-fails-closed under typed enums.

## Source anchors

- Authoring (CLI): `rust/cli/src/flags.rs` (`ProfileFlags`), `rust/cli/src/yaml.rs`
  (YAML → `Inputs`), `rust/cli/src/load.rs` (flags → `Inputs`; re-exports the moved
  `Inputs`/`resolve`), `rust/cli/src/model/mod.rs` (re-export shim over the moved
  typed model), `rust/cli/src/profile.rs` (`AuthoringWire`, per-path child drive),
  `rust/cli/src/sweep/`, `rust/cli/src/search.rs` (`apply_override_inputs`).
- Typed model + resolution (runtime): `rust/runtime/src/config/model/` (canonical
  typed model, `workload_kind.rs`), `rust/runtime/src/config/resolve.rs` (`Inputs`,
  `pub fn resolve(Inputs) -> BenchmarkRun`), `rust/runtime/src/config/phase_validate.rs`,
  `rust/runtime/src/config/validate.rs` (offline cross-field invariants; currently
  uncalled), `rust/runtime/src/config/redact.rs`.
- Wire + consumer: `rust/cli/src/execute_mode.rs` (child-side decode entry),
  `rust/runtime/src/engine/protocol_v2.rs` (`AuthoringWireV2`,
  `decode_execute_wire`, `BenchmarkRunWireV2`, `into_authored`),
  `rust/runtime/src/engine/registry.rs` (`ScheduledWorkloadConfigV2`,
  `GraphWorkloadConfigV2`, `StaticAccuracyWorkloadConfigV2`, `strict_decode`),
  `rust/runtime/src/engine/online_execution.rs` (factory `validate`/`prepare`),
  `rust/runtime/src/engine/coordinator.rs` (`Validate`/`Execute` split),
  `rust/runtime/src/engine/cell_launcher.rs` (cellular authoring payload).
- CLI validate command: `rust/cli/src/config/mod.rs` (`aiperf config validate`).
- Python blueprint: `src/aiperf/config/config.py` (`AIPerfConfig`,
  `BenchmarkConfig`), `src/aiperf/config/resolution/plan.py` (`BenchmarkRun`,
  `BenchmarkPlan`), `src/aiperf/config/loader/plan.py` (`build_benchmark_plan`),
  `src/aiperf/config/sweep/expand.py` (`expand_sweep`),
  `src/aiperf/config/flags/resolver.py` (`resolve_config`).
