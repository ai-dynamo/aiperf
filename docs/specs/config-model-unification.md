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

The current front end is a typed model wrapped in untyped plumbing on both sides.

**Producer (CLI) — three schemas for one config.** `aiperf profile` resolves a run
through: two independent input schemas (`ProfileFlags`, a `clap` struct; and a
separate YAML serde schema `ConfigFile`/`*Section`), both hand-mapped into a flat
`Inputs` bag (~180 fields), then imperatively assembled by `load::build` into the
canonical typed `cli::model::BenchmarkConfig`/`BenchmarkRun`. The two mappers
(flags→`Inputs`, YAML→`Inputs`) enumerate the same config surface twice and drift
(e.g. the baseten guard covers only the flags path). Defaults are hardcoded in
`load.rs` rather than carried on the model; flag overlay (`apply_cli_overrides`)
is a partial hand patch, not a general merge; sweeps use three unrelated
override mechanisms (mutate `ProfileFlags`, mutate a pre-typed `Value`, mutate a
post-built `cfg` `Value`) with no unified plan object.

**Wire — the typing is discarded and rebuilt fragmented.** The CLI serializes the
typed `BenchmarkRun` (`serde_json::to_vec`) to the `--execute` child's stdin. The
runtime re-parses it into a deliberately loose `BenchmarkConfigWireV2` (nearly
every section `serde_json::Value`), then `into_authored` reshapes those Values,
decides `workload_id` = `graph` vs `scheduled` from the dataset type, and hands a
hand-built `json!` blob to per-workload strict DTOs (`ScheduledWorkloadConfigV2`,
`GraphWorkloadConfigV2`, `StaticAccuracyWorkloadConfigV2`), each
`deny_unknown_fields`. Because the producer builds an untyped blob and the
consumers are strict — and the two divergent workload DTOs differ only in a
graph-only field — a field emitted for the wrong workload (the `weka_semantics`
leak) fails only at execute-time `prepare_with_context`, never at authoring.

**Validation is split and one half is dead.** `aiperf config validate` runs only
`yaml::resolve` (a YAML schema check); the runtime's `OperationV2::Validate` — which
already strict-decodes every component config offline, no I/O — is implemented but
never spawned. So the strict decode that would catch producer/consumer drift runs
only under a real `--execute`.

**The Python blueprint the Rust side diverged from.** origin/main models this as
**one flat `BenchmarkConfig`** (Pydantic) inside an `AIPerfConfig` envelope; both
flags and YAML feed it through a transient dict merge that ends at a single
`model_validate`. Polymorphism lives in exactly two list fields — `phases` and
`datasets` — each a `Discriminator("type")` union; `endpoint`/`transport`/sidecars
are flat. Workload "kind" (graph/scheduled/accuracy) is **not a type**: it is
emergent from dataset+phase compatibility, enforced by validators. `build_benchmark_plan`
expands sweeps into `list[BenchmarkConfig]`, then wraps each `(variation, trial)`
into a `BenchmarkRun` that embeds one `BenchmarkConfig` plus run identity. The same
model crosses the parent→child boundary unchanged.

## Future requirements

The convergence target and its migration. Steps 1 and 4 are built (branch
`ajc/config-model-unification`); steps 2 and 3 are partially built with the
structural remainder blocked as noted.

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

The wire is `serde(BenchmarkRun)` with `deny_unknown_fields` and a `protocol_version`
tripwire; the child deserializes the same `BenchmarkRun`. `BenchmarkConfigWireV2`,
`into_authored`, and the per-workload `*WorkloadConfigV2` DTOs are removed; the
executor matches on typed `phases`/`datasets`/`transport` and calls `workload_kind()`.
"Unknown component id fails closed" is preserved by a serde enum that errors on an
unknown tag.

**Migration (each step ships green):**

1. **Unify the wire type.** *(Built.)* Move `BenchmarkConfig`/`BenchmarkRun` into the
   `aiperf_runtime::config` module; the runtime deserializes that type directly. Delete the reshape, the `json!` blob,
   the graph-guard, and the fragmented DTOs; replace with typed access +
   `workload_kind()`. Guard with wire round-trip tests (CLI serialize ==
   runtime deserialize, byte-identical). This step alone eliminates the
   `weka_semantics` bug class.
2. **Collapse the producer front-end.** *(Resolved to its correct end-state, not a
   one-deserialize collapse.)* Investigation (five deep passes + the Python
   reference) established the Rust producer is **already** the same shape as
   Python's: two authoring-surface constructors (CLI flags, YAML) → one typed
   intermediate → one lowering — mirroring Python's `convert_cli_to_aiperf` +
   `load_config` → one `model_validate`. Two input surfaces genuinely need two
   constructors (Python keeps two), and the flag path performs CLI-only
   byte-affecting derivations the YAML path does not (fixed-schedule
   `request_count` from `count_schedule_entries(file)`, sentinel URLs, warmup
   synthesis, a post-`into_inputs` `kv_block_size` mutation), so a single
   flag∪YAML deep-merge cannot be byte-reconstructed. The achievable
   improvements landed: `ProfileFlags` bools → `Option<bool>` (explicit-set
   signal) and the flags↔YAML drift unification. A true single authoring model
   would require moving resolution to the runtime (Python-style, runner-resolves),
   which changes the wire representation — a separate phase, not this one.
3. **One sweep plan.** *(Partial: typed `build_benchmark_plan` grid/zip/magic-list
   seam built and tested. Retiring the live grid/recipe override mechanisms is
   blocked by the same producer property — the grid path mutates `ProfileFlags` so
   `count_schedule_entries`/count-pooling run in `load::resolve`, which the typed
   post-resolution path cannot reproduce byte-exact; `sweep_parity` — which drives
   `load::resolve` directly — is the proof.)* Replace the three override mechanisms
   with `build_benchmark_plan` over the typed model (dotted-path apply, alpha-sorted
   keys, `base+N` seed derivation). Adaptive search stays a runtime ask-tell loop.
4. **Validators as offline passes.** *(Built: nine raise-only invariants ported to
   `config::validate` — the seven cross-field checks plus `cache_bust_compatibility`
   (full) and `agentic_cache_warmup` (no-scenario branch; the scenario branch needs a
   runtime scenario-registry `timing_mode` lookup with no config-time representation,
   as in Python, so it defers). The authored `timing_mode`/`cache_bust` fields were
   added to the typed model, skip-serialized so the wire stays byte-identical.)*
   Port the raise-only
   cross-field invariants
   (phase↔dataset compatibility, prefill⇒streaming, cache-bust, agentic-warmup) as
   validate-time functions; keep the mutating ones (tokenizer/seed defaults) as
   resolution passes; wire `aiperf config validate` to run them offline.

The `scenario` resolver (lookup-and-stamp) and the adaptive-search Bayesian loop
are mirrored, not redesigned. Risk concentrates in step 1's wire compatibility
(round-trip tests) and preserving unknown-id-fails-closed under typed enums.

## Source anchors

- Producer: `rust/cli/src/flags.rs`, `rust/cli/src/yaml.rs`, `rust/cli/src/load.rs`
  (`Inputs`, `resolve`, `build`), `rust/cli/src/model/` (the canonical typed model),
  `rust/cli/src/profile.rs`, `rust/cli/src/sweep/`, `rust/cli/src/search.rs`.
- Wire + consumer: `rust/cli/src/execute.rs`,
  `rust/runtime/src/engine/protocol_v2.rs` (`BenchmarkRunWireV2`,
  `BenchmarkConfigWireV2`, `into_authored`), `rust/runtime/src/engine/registry.rs`
  (`ScheduledWorkloadConfigV2`, `GraphWorkloadConfigV2`, `strict_decode`),
  `rust/runtime/src/engine/online_execution.rs` (factory `validate`/`prepare`),
  `rust/runtime/src/engine/coordinator.rs` (`Validate`/`Execute` split).
- Python blueprint: `src/aiperf/config/config.py` (`AIPerfConfig`,
  `BenchmarkConfig`), `src/aiperf/config/resolution/plan.py` (`BenchmarkRun`,
  `BenchmarkPlan`), `src/aiperf/config/loader/plan.py` (`build_benchmark_plan`),
  `src/aiperf/config/sweep/expand.py` (`expand_sweep`),
  `src/aiperf/config/flags/resolver.py` (`resolve_config`).
