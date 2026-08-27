<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Plugin ABI Boundary Gap-Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Shrink the host/plugin ABI closure from 193 types to ~118 and instrument the universe-rebuild rate, so the recompile-boundary matrix in the plugin design is true of this codebase instead of aspirational.

**Architecture:** Six independent refactors of `rust/runtime`, each cutting one measured leak edge that drags host-private implementation into the plugin ABI closure. Every task is behavior-preserving; the observable deliverable is a smaller measured ABI closure and a lower measured universe-bump rate. Task 1 builds the measurement first so every later task proves its own effect.

**Tech Stack:** Rust 2024 / resolver 3, `cargo`, `rustdoc --output-format json`, `criterion` (`rust/runtime/benches/chat_dispatch_bench.rs`), `git`.

**Spec:**
- `docs/superpowers/specs/2026-08-26-native-rust-runtime-plugins-design.md` (normative)
- `docs/superpowers/specs/2026-08-27-native-rust-plugin-recompile-boundaries.md` (recompile matrix)

**Relationship to the existing plan:** This plan does **not** replace
`docs/superpowers/plans/2026-08-26-native-rust-runtime-plugins-implementation.md`
(40 tasks, execution tracker ACTIVE). It closes four gaps that plan does not
cover, verified by grep against it:

| Gap | Mentions in the 40-task plan | Status |
|---|---:|---|
| `ExecutionSinkBuilder::Sink` associated-type erasure / `ThreadPerCoreExecutor<B>` | 0 | **gap** |
| `WorkerMaterializer` concrete-struct-at-boundary leak | 0 | **gap** |
| `MetricTag` closed 60-variant enum in the ABI | 0 | **gap** |
| ABI-closure / universe-bump churn measurement and gate | 0 | **gap** |
| `NativeReport` reachable whole from `Exporter` | 1 (unrelated, `otel_per_record`) | **gap** |
| `EndpointType` closed 19-variant enum | 3 (compatibility pinning only, never opened) | **partial gap** |
| Closed `ExportConfig` aggregate at the boundary | 3 (Task 6 forbids it explicitly) | already covered — **not in this plan** |
| `RunContext` forbidden at the boundary | 8 | already covered — **not in this plan** |

**Sequencing:** Tasks 1–6 here land **before** Task 4 ("Extract boundary-owned
core values and host service traits") of the 40-task plan. Task 4 moves
definitions into `aiperf-core`; every type this plan evicts first is a type Task
4 then never has to move. Running these after Task 4 means moving the same code
twice.

## Global Constraints

- Every task is **behavior-preserving**. No wire format, artifact byte, metric
  value, or report field changes. Report snapshot tests must stay byte-identical.
- Copyright header on every new source file is the two SPDX lines only.
- `//!` module docs and `///` docs on every new public item.
- No `unwrap()`/`expect()` in production code on the hot path
  (`transport/`, `dispatch/`, `clock/`, `timing/`, `scheduler.rs`,
  `scheduled.rs`, `request_rate.rs`, `phase_runtime.rs`, `graph/`, `engine/`,
  `metrics_core/`). If unavoidable, comment why it cannot fail.
- `tracing` only, never `log`. Structured fields, not formatted strings.
- No new `Arc<Mutex<_>>` on per-request or per-token paths.
- All measurement and scheduling time routes through `Clock`.
- Environment for every command: `source .venv/bin/activate` from the repo root,
  then `cd rust`.
- `aiperf-runtime`'s `engine` module is behind the `engine` Cargo feature.
  `cargo test -p aiperf-runtime` alone runs **zero** engine tests. Every task
  that touches `engine/` must run both:
  `cargo test -p aiperf-runtime` and `cargo test -p aiperf-runtime --features engine`.
- Commit **before** compiling. If a build or test fails afterward, fix it in a
  **new** follow-up commit. Never `--amend`, never rebase, never reset.
- Commit whole files (`git add <file> …`). Never hunk-level staging.
- Base branch for diffs and PRs is `origin/main`; there is no local `main`.

## Baseline Numbers (measured 2026-08-27 at `110e00321a`)

These are the numbers Task 1 must reproduce before any other task runs. If the
tool disagrees with these, the tool is wrong — fix the tool, not the baseline.

| Measure | Value |
|---|---:|
| ABI closure, `RunContext` narrowed | 193 types / 52 files |
| ABI type-definition lines | 2,083 |
| Total lines in those files | 32,461 |
| Implementation lines co-resident with ABI types | 30,378 (94%) |
| Universe-bump rate, file-granular, 120 merge units | 19 / 54 code units (35%) |
| Universe-bump rate, type-granular | 13 / 54 code units (24%) |
| Target after Tasks 2–6 | ~118 types, 7–8 / 54 (13–15%) |

## File Structure

**New files**

- `rust/xtask/Cargo.toml`, `rust/xtask/src/main.rs` — cargo-xtask entry point.
- `rust/xtask/src/abi_closure.rs` — computes the ABI-facing type closure from
  rustdoc JSON, given a seed set. One responsibility: closure computation.
- `rust/xtask/src/abi_churn.rs` — replays git history against a closure snapshot
  and reports the universe-bump rate. One responsibility: history replay.
- `rust/xtask/abi-seeds.toml` — the boundary seed set and the blocked-edge list,
  checked in and reviewed.
- `rust/xtask/abi-baseline.json` — committed snapshot the gate compares against.
- `rust/runtime/src/metrics_core/report_view.rs` — the narrow `ReportView`
  accessor trait exporters consume instead of `&NativeReport`.
- `rust/runtime/src/multiturn/materializer.rs` — `CreditMaterializer` trait,
  extracted from `multiturn.rs`.
- `rust/runtime/src/endpoints/type_id.rs` — interned `EndpointTypeId`.
- `rust/runtime/src/metrics_core/tag_id.rs` — interned `MetricTagId`.

**Modified files**

- `rust/runtime/src/engine/turn_execution.rs` — Tasks 2 and 5.
- `rust/runtime/src/multiturn.rs` — Task 2.
- `rust/runtime/src/export/mod.rs` — Task 3.
- `rust/runtime/src/metrics_core/report.rs` — Task 3.
- `rust/runtime/src/endpoints/metadata.rs`, `endpoints/registry.rs` — Task 4.
- `rust/runtime/src/metrics_core/catalog.rs`, `metrics_core/ingest.rs` — Task 4.
- `rust/runtime/src/body_plan.rs`, `metrics_core/accumulator.rs`,
  `scheduled.rs` — Task 6.
- `.github/workflows/` — Task 1 and Task 6 gates.

---

### Task 1: Build the ABI-closure and universe-churn measurement gate

Nothing else in this plan can be verified without this. It also gives the
40-task plan the evidence its recompile matrix currently asserts without proof.

**Files:**
- Create: `rust/xtask/Cargo.toml`
- Create: `rust/xtask/src/main.rs`
- Create: `rust/xtask/src/abi_closure.rs`
- Create: `rust/xtask/src/abi_churn.rs`
- Create: `rust/xtask/abi-seeds.toml`
- Create: `rust/xtask/abi-baseline.json`
- Create: `rust/xtask/tests/closure_baseline.rs`
- Modify: `rust/Cargo.toml` (add `"xtask"` to `members`)
- Create: `.github/workflows/rust-abi-closure.yml`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `cargo xtask abi-closure --seeds rust/xtask/abi-seeds.toml --json` printing
    `{"types": <usize>, "files": <usize>, "type_lines": <usize>,
    "file_lines": <usize>, "entries": [{"name": String, "file": String,
    "start": usize, "end": usize}]}`.
  - `cargo xtask abi-churn --since <rev> --merges <n>` printing
    `{"code_units": <usize>, "universe": <usize>, "host_only": <usize>,
    "one_plugin": <usize>}`.
  - `cargo xtask abi-gate` exiting non-zero when the measured closure exceeds
    `abi-baseline.json`.
  - Later tasks call `cargo xtask abi-closure` to prove their own reduction.

- [ ] **Step 1: Write the failing baseline test**

Create `rust/xtask/tests/closure_baseline.rs`:

```rust
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pins the measured ABI closure so a boundary regression fails CI.

use aiperf_xtask::abi_closure::{compute, Seeds};

#[test]
fn closure_matches_committed_baseline() {
    let seeds = Seeds::load("abi-seeds.toml").expect("seed file");
    let measured = compute(&seeds).expect("closure");
    let baseline = std::fs::read_to_string("abi-baseline.json").expect("baseline");
    let baseline: serde_json::Value = serde_json::from_str(&baseline).expect("json");

    assert_eq!(
        measured.types.len(),
        baseline["types"].as_u64().expect("types") as usize,
        "ABI closure size changed; if intentional, regenerate abi-baseline.json"
    );
}

#[test]
fn closure_excludes_run_context() {
    let seeds = Seeds::load("abi-seeds.toml").expect("seed file");
    let measured = compute(&seeds).expect("closure");
    assert!(
        !measured.types.contains_key("RunContext"),
        "RunContext is forbidden at the plugin boundary (design.md)"
    );
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && cd rust && cargo test -p aiperf-xtask --test closure_baseline`

Expected: FAIL — `error[E0432]: unresolved import` / no package named `aiperf-xtask`.

- [ ] **Step 3: Write the seed file**

Create `rust/xtask/abi-seeds.toml`. These are the 18 boundary entry points and
the edges the design forbids:

```toml
# Boundary seed set: every type reachable from these in a field or signature
# position is in the host ABI universe.
seeds = [
  "ExecutionBackendConfig", "PreparedTurn", "TurnToSend", "MeasuredContext",
  "DispatchResult", "InferenceDimensions", "PreparedEndpointTable",
  "WorkerMaterializer", "RequestObserver", "TurnResponseObserver",
  "RequestExecutor", "Clock", "WorkerSink", "ExecutionSinkBuilder",
  "Exporter", "EndpointFactory", "AIPerfExtension", "Request",
]

# Edges the normative design forbids at the boundary. Present here so the tool
# measures the design's target state, not today's accidental reachability.
# design.md: "Today's RunContext, closed Transport, concrete HTTP sink config,
# complete AIPerfRegistry, CLI config, and orchestration engine objects are
# forbidden at the boundary."
blocked = [
  "RunContext", "AIPerfRegistry", "ExecutionFactories",
  "TransportFactory", "WorkloadFactory",
]
```

- [ ] **Step 4: Implement closure computation**

Create `rust/xtask/src/abi_closure.rs`. Drive rustdoc JSON rather than parsing
Rust source — the design already commits CI to rustdoc JSON for the ownership
table (`design.md`), so this shares that mechanism and avoids a second,
divergent notion of "what is public".

```rust
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Computes the host/plugin ABI-facing type closure.
//!
//! A type is ABI-facing when it is reachable from a boundary seed through a
//! field, variant, argument, or return position. Doc-comment mentions and
//! method bodies do not create reachability: the host never instantiates or
//! interprets a type it merely reads about.

use anyhow::{Context, Result};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

/// One ABI-facing type and where it is defined.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Entry {
    /// Type name as written in source.
    pub name: String,
    /// Repository-relative path of the defining file.
    pub file: String,
    /// First line of the definition, 1-based.
    pub start: usize,
    /// Last line of the definition, 1-based inclusive.
    pub end: usize,
}

/// The seed set and blocked edges that define the boundary.
#[derive(Debug, serde::Deserialize)]
pub struct Seeds {
    /// Boundary entry points.
    pub seeds: Vec<String>,
    /// Edges the design forbids; traversal stops at these.
    pub blocked: Vec<String>,
}

impl Seeds {
    /// Read a seed file relative to the xtask crate root.
    pub fn load(path: &str) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("reading seed file {path}"))?;
        toml::from_str(&raw).with_context(|| format!("parsing seed file {path}"))
    }
}

/// The measured closure.
#[derive(Debug, serde::Serialize)]
pub struct Closure {
    /// Reachable types, keyed by name.
    pub types: BTreeMap<String, Entry>,
    /// Distinct files contributing at least one reachable type.
    pub files: BTreeSet<String>,
    /// Total lines occupied by reachable type definitions.
    pub type_lines: usize,
    /// Total lines in the contributing files.
    pub file_lines: usize,
}

/// Compute the closure by breadth-first traversal from the seeds.
pub fn compute(seeds: &Seeds) -> Result<Closure> {
    let index = rustdoc::index_for("aiperf-runtime")
        .context("building rustdoc JSON index for aiperf-runtime")?;
    let blocked: BTreeSet<&str> = seeds.blocked.iter().map(String::as_str).collect();

    let mut types: BTreeMap<String, Entry> = BTreeMap::new();
    let mut queue: VecDeque<String> = seeds.seeds.iter().cloned().collect();

    while let Some(name) = queue.pop_front() {
        if types.contains_key(&name) || blocked.contains(name.as_str()) {
            continue;
        }
        let Some(def) = index.definition(&name) else {
            continue;
        };
        types.insert(name.clone(), def.entry.clone());
        // Only structural positions propagate. `def.referenced_types()` returns
        // field, variant, argument, and return types -- never body-local types
        // and never types named only in documentation.
        for next in def.referenced_types() {
            if !types.contains_key(&next) {
                queue.push_back(next);
            }
        }
    }

    let files: BTreeSet<String> = types.values().map(|e| e.file.clone()).collect();
    let type_lines = types.values().map(|e| e.end - e.start + 1).sum();
    let file_lines = files
        .iter()
        .map(|f| std::fs::read_to_string(f).map(|s| s.lines().count()).unwrap_or(0))
        .sum();

    Ok(Closure { types, files, type_lines, file_lines })
}
```

Implement the supporting `rustdoc` module in the same crate: it shells out to
`cargo +nightly rustdoc -p aiperf-runtime --lib -- -Z unstable-options
--output-format json`, deserializes `target/doc/aiperf_runtime.json`, and maps
each item id to its `span` (file, `begin.0`, `end.0`) and its structural type
references. Record the exact nightly toolchain in `rust-toolchain.toml` so the
gate is reproducible — rustdoc JSON has no format stability guarantee.

- [ ] **Step 5: Run to verify the baseline reproduces**

Run:
```bash
source .venv/bin/activate && cd rust
cargo run -p aiperf-xtask -- abi-closure --seeds xtask/abi-seeds.toml --json | tee /tmp/abi.json
```

Expected: `types` is `193` and `files` is `52`, matching the Baseline Numbers
table. If it does not, the traversal rules are wrong — most likely it is
following doc mentions (too many) or missing types behind `Box`/`Rc`/`Arc`/
`Result`/`Option`/futures (too few). Both count as reachable per
`design.md`: "including behind `Box`, `Rc`, `Arc`, `Result`, a future, a private
field, or another container."

- [ ] **Step 6: Commit the tool and the baseline**

```bash
cd /home/anthony/nvidia/projects/aiperf/ajc/rust
cp /tmp/abi.json rust/xtask/abi-baseline.json
git add rust/xtask rust/Cargo.toml
git commit -F - <<'EOF'
feat(xtask): measure the host/plugin ABI closure

Computes the ABI-facing type closure from rustdoc JSON and pins it in
abi-baseline.json so a boundary regression fails CI. Seeds and forbidden
edges are checked in and reviewed rather than inferred.

Measured at this commit: 193 types across 52 files.
EOF
```

- [ ] **Step 7: Implement the churn replay**

Create `rust/xtask/src/abi_churn.rs`. For each merge unit in `--merges n`
first-parent history, diff `H^1..H`, and classify:

- `UNIVERSE` — a `-U0` hunk overlaps an ABI type's line span **as computed at
  that commit** (recompute spans via `git show <rev>:<path>`; line numbers drift
  and using HEAD spans against historical diffs silently misclassifies).
- `one_plugin` — every changed `rust/` path is under a single plugin-candidate
  root.
- `host_only` — otherwise.

Report both the file-granular and type-granular rate. The gap between them is
the implementation-co-residency cost and is the number Task 6 drives down.

- [ ] **Step 8: Verify the churn baseline reproduces**

Run:
```bash
source .venv/bin/activate && cd rust
cargo run -p aiperf-xtask -- abi-churn --merges 120
```

Expected: `code_units: 54`, file-granular `universe: 19` (35%), type-granular
`universe: 13` (24%), `one_plugin: 1`.

- [ ] **Step 9: Wire the CI gate and commit**

Create `.github/workflows/rust-abi-closure.yml` running
`cargo run -p aiperf-xtask -- abi-gate` on pull requests touching `rust/`.
The gate fails when the measured closure exceeds the committed baseline.
Regenerating the baseline is allowed but shows up as a reviewed diff — which is
the point: growing the ABI universe should be a visible decision.

```bash
git add rust/xtask/src/abi_churn.rs .github/workflows/rust-abi-closure.yml
git commit -F - <<'EOF'
feat(xtask): gate ABI-closure growth and report universe-bump rate

abi-churn replays first-parent merge history against per-commit type spans
and reports how many units would rebuild every plugin. Baseline at this
commit: 35% file-granular, 24% type-granular, 1 single-plugin unit of 54.
EOF
```

---

### Task 2: Erase `WorkerMaterializer` to a trait object

**Why:** `ExecutionSinkBuilder::build_credit_materializer` returns the concrete
`WorkerMaterializer` struct, which holds `WorkerMaterializationRecipe`, which
holds `TextTokenizer` and the whole `Dataset`. Measured leak path:

```
TextTokenizer <- WorkerMaterializationRecipe <- WorkerMaterializer <- ExecutionSinkBuilder
Dataset <- ConversationSession <- RequestMaterializer <- WorkerMaterializationRecipe <- ...
```

This single edge puts the entire `dataset` module in the plugin ABI. Evicting it
removes 13 types directly and takes `TextTokenizer`, `Tokenizer`,
`InputTokenCounter`, `Dataset`, `ConversationSession`, and `DatasetError` with
it. It also explains two of the 13 real universe-wide merges measured
(`e839b7dbe` random range ratio, `e62422c14` ShareGPT batch encoding) — both
changed only tokenizer internals.

**Files:**
- Create: `rust/runtime/src/multiturn/materializer.rs`
- Modify: `rust/runtime/src/multiturn.rs:2026` (`WorkerMaterializer`)
- Modify: `rust/runtime/src/engine/turn_execution.rs:267` and `:326`
  (`build_credit_materializer` on `HttpSinkBuilder` and `GrpcSinkBuilder`)
- Modify: `rust/runtime/src/engine/turn_execution.rs:168`
  (`CreditMaterializerFactory`)
- Test: `rust/runtime/src/engine/turn_execution.rs` (in-module `mod tests`)

**Interfaces:**
- Consumes: `cargo xtask abi-closure` from Task 1.
- Produces: `pub trait CreditMaterializer` with the single method
  `fn materialize(&self, identity: CreditIdentity) -> Result<PreparedTurn>`.
  `ExecutionSinkBuilder::build_credit_materializer` returns
  `Result<Option<Box<dyn CreditMaterializer>>>`. `CreditMaterializerFactory::build_worker`
  returns `Result<Box<dyn CreditMaterializer>>`. Task 5 relies on this
  signature.

- [ ] **Step 1: Write the failing test**

Add to the `mod tests` block in `rust/runtime/src/engine/turn_execution.rs`:

```rust
#[test]
fn credit_materializer_is_object_safe_and_dataset_free() {
    // A plugin-side materializer must be constructible without any dataset,
    // tokenizer, or runtime type -- that is the whole point of erasing it.
    struct FixedMaterializer(PreparedTurn);

    impl CreditMaterializer for FixedMaterializer {
        fn materialize(&self, _identity: CreditIdentity) -> Result<PreparedTurn> {
            Ok(self.0.clone())
        }
    }

    let turn = test_util::prepared_turn("m", "hello");
    let erased: Box<dyn CreditMaterializer> = Box::new(FixedMaterializer(turn.clone()));
    let got = erased
        .materialize(CreditIdentity::for_test(7))
        .expect("materialize");
    assert_eq!(got.model, turn.model);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
source .venv/bin/activate && cd rust
cargo test -p aiperf-runtime --features engine credit_materializer_is_object_safe
```

Expected: FAIL — `cannot find trait CreditMaterializer in this scope`.

- [ ] **Step 3: Extract the trait**

Create `rust/runtime/src/multiturn/materializer.rs`:

```rust
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Worker-local rebuild of identity-only routed credits.
//!
//! `--dispatch global-push` routes a credit carrying only identity and has the
//! receiving worker build the body. The concrete implementation owns a dataset
//! and a tokenizer; neither may reach the plugin boundary, so the boundary sees
//! only this trait.

use crate::multiturn::CreditIdentity;
use crate::transport::core::dispatch::PreparedTurn;
use anyhow::Result;

/// Rebuilds one routed credit into a dispatchable turn.
///
/// Object-safe on purpose: the host owns the dataset and tokenizer behind the
/// implementation, and a transport plugin only ever calls through this trait.
pub trait CreditMaterializer {
    /// Rebuild the turn identified by `identity` from resident run state.
    fn materialize(&self, identity: CreditIdentity) -> Result<PreparedTurn>;
}

impl CreditMaterializer for super::WorkerMaterializer {
    fn materialize(&self, identity: CreditIdentity) -> Result<PreparedTurn> {
        super::WorkerMaterializer::materialize(self, identity)
    }
}
```

Then change the two trait signatures in
`rust/runtime/src/engine/turn_execution.rs`:

```rust
pub trait CreditMaterializerFactory: Send + Sync {
    /// Build one worker's materializer over `table`, its own dense-key table.
    fn build_worker(&self, table: PreparedEndpointTable) -> Result<Box<dyn CreditMaterializer>>;
}
```

```rust
    /// Build this worker's materializer for identity-only credits, if the run
    /// routes them. `None` keeps issuer-side materialization.
    fn build_credit_materializer(&self) -> Result<Option<Box<dyn CreditMaterializer>>> {
        Ok(None)
    }
```

Update the two implementations at `:267` and `:326` to wrap their existing
return value in `Box::new(...)`. Do not change materialization behavior.

Note on cost: `materialize` is called once per routed credit under
`global-push` only, not per token. One `Box` indirection there is not a
token-path cost. Task 5 covers the call that *is* per-request.

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
source .venv/bin/activate && cd rust
cargo test -p aiperf-runtime && cargo test -p aiperf-runtime --features engine
```

Expected: PASS, including the new test and every existing `global-push` test.

- [ ] **Step 5: Prove the closure shrank**

Run:
```bash
source .venv/bin/activate && cd rust
cargo run -p aiperf-xtask -- abi-closure --seeds xtask/abi-seeds.toml --json \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["types"], d["files"])'
```

Expected: `180` types (down from 193). If it still reports 193, some other edge
still reaches `WorkerMaterializer` — find it with
`cargo run -p aiperf-xtask -- abi-closure --why WorkerMaterializer`.

- [ ] **Step 6: Commit**

```bash
git add rust/runtime/src/multiturn/materializer.rs rust/runtime/src/multiturn.rs \
        rust/runtime/src/engine/turn_execution.rs rust/xtask/abi-baseline.json
git commit -F - <<'EOF'
refactor(engine): erase WorkerMaterializer at the transport boundary

build_credit_materializer returned a concrete struct holding the tokenizer and
the whole Dataset, putting the dataset module in the plugin ABI. Returning
Box<dyn CreditMaterializer> evicts 13 types including TextTokenizer, Dataset,
ConversationSession, and DatasetError.

ABI closure: 193 -> 180 types.
EOF
```

---

### Task 3: Narrow `NativeReport` behind an accessor trait

**Why:** `Exporter::export` takes `&NativeReport`
(`rust/runtime/src/export/mod.rs:310`). `NativeReport`
(`metrics_core/report.rs:1082`) transitively reaches 33 ABI types — the largest
single contributor to the closure. Measured effect of cutting this edge: **−43
types**, the biggest win in the plan.

**Files:**
- Create: `rust/runtime/src/metrics_core/report_view.rs`
- Modify: `rust/runtime/src/metrics_core/report.rs:1082`
- Modify: `rust/runtime/src/export/mod.rs:310`
- Test: `rust/runtime/src/metrics_core/report_view.rs` (in-module `mod tests`)

**Interfaces:**
- Consumes: nothing from Task 2 (independent; may run in parallel).
- Produces: `pub trait ReportView` with `fn run_summary(&self) -> &RunSummary`,
  `fn metric(&self, tag: MetricTagId) -> Option<MetricValue>`,
  `fn metric_names(&self) -> &[Arc<str>]`, and
  `fn per_record(&self) -> Option<&[RecordRow]>`. `Exporter::export` takes
  `&dyn ReportView`. Task 4 replaces `MetricTag` with `MetricTagId` in this
  signature.

- [ ] **Step 1: Write the failing test**

Add to `rust/runtime/src/metrics_core/report_view.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exporters_read_reports_through_the_narrow_view() {
        // An exporter plugin must be writable against ReportView alone, with no
        // access to NativeReport's internal structure.
        fn summarize(view: &dyn ReportView) -> usize {
            view.metric_names().len()
        }

        let report = crate::metrics_core::report::test_util::two_metric_report();
        assert_eq!(summarize(&report), 2);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
source .venv/bin/activate && cd rust
cargo test -p aiperf-runtime exporters_read_reports_through_the_narrow_view
```

Expected: FAIL — `cannot find trait ReportView in this scope`.

- [ ] **Step 3: Define the view and implement it for `NativeReport`**

Create `rust/runtime/src/metrics_core/report_view.rs`:

```rust
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Narrow read-only projection of a finalized report for exporters.
//!
//! `NativeReport` reaches 33 further types. Handing exporters the whole struct
//! would put all of them in the host ABI universe, so every change to any of
//! them would rebuild every plugin. Exporters get accessors instead.

use crate::metrics_core::catalog::MetricTag;
use crate::metrics_core::report::{MetricValue, RecordRow, RunSummary};
use std::sync::Arc;

/// Read-only accessors an exporter needs from a finalized report.
pub trait ReportView {
    /// Run-level facts: timestamps, model, request counts.
    fn run_summary(&self) -> &RunSummary;

    /// One aggregate metric value, absent when the run produced none.
    fn metric(&self, tag: MetricTag) -> Option<MetricValue>;

    /// Names of every metric present, in stable report order.
    fn metric_names(&self) -> &[Arc<str>];

    /// Per-record rows when the run retained them; `None` under sketch metrics.
    fn per_record(&self) -> Option<&[RecordRow]>;
}
```

Implement `ReportView for NativeReport` in `report.rs` by delegating to the
existing fields. Change `Exporter::export` in `export/mod.rs:310`:

```rust
    fn export(
        &self,
        report: &dyn ReportView,
        artifact_dir: &Path,
        cfg: &ExportConfig,
    ) -> anyhow::Result<()>;
```

Update each built-in exporter to take `&dyn ReportView`. Where an exporter
currently reaches a `NativeReport` field with no accessor, **add the accessor** —
do not widen the trait to return `&NativeReport`. If an exporter needs
something genuinely structural, that is a finding to record, not to route
around.

Leave `cfg: &ExportConfig` alone. Task 6 of the 40-task plan replaces it; doing
it here would collide with that task's ownership.

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
source .venv/bin/activate && cd rust
cargo test -p aiperf-runtime && cargo test -p aiperf-runtime --features engine
```

Expected: PASS. Report snapshot tests must be **byte-identical** — this task
changes who can see what, not what is written.

- [ ] **Step 5: Prove the closure shrank**

Run:
```bash
source .venv/bin/activate && cd rust
cargo run -p aiperf-xtask -- abi-closure --seeds xtask/abi-seeds.toml --json \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["types"], d["files"])'
```

Expected: `137` types when run after Task 2, or `150` when run standalone.

- [ ] **Step 6: Commit**

```bash
git add rust/runtime/src/metrics_core/report_view.rs \
        rust/runtime/src/metrics_core/report.rs rust/runtime/src/export \
        rust/xtask/abi-baseline.json
git commit -F - <<'EOF'
refactor(export): give exporters a narrow report view

Exporter::export took &NativeReport, which reaches 33 further types and put
all of them in the host ABI universe. A ReportView accessor trait cuts that
edge without changing a byte of any emitted artifact.

ABI closure: -43 types, the largest single reduction in the boundary.
EOF
```

---

### Task 4: Open `EndpointType` and `MetricTag` to interned registry IDs

**Why:** `EndpointType` (`endpoints/metadata.rs:76`) is a closed 19-variant enum
and `MetricTag` (`metrics_core/catalog.rs:21`) a closed 60-variant enum. Adding
an endpoint or a metric — precisely what plugins exist to do — is a variant
addition, which is a universe-wide rebuild. This is the sharpest contradiction
in the current shape: the extension points are closed.

Measured: `6a2c6c5fb` ("audio transcription port") was universe-wide solely
because it added an `EndpointType` variant.

**Files:**
- Create: `rust/runtime/src/endpoints/type_id.rs`
- Create: `rust/runtime/src/metrics_core/tag_id.rs`
- Modify: `rust/runtime/src/endpoints/metadata.rs:76`
- Modify: `rust/runtime/src/endpoints/registry.rs`
- Modify: `rust/runtime/src/metrics_core/catalog.rs:21`
- Modify: `rust/runtime/src/metrics_core/ingest.rs:217`
- Modify: `rust/runtime/src/metrics_core/report_view.rs` (from Task 3)

**Interfaces:**
- Consumes: `ReportView` from Task 3.
- Produces: `EndpointTypeId(u32)` and `MetricTagId(u32)`, both `Copy`, both with
  `fn as_str(&self) -> &'static str` and
  `fn resolve(name: &str) -> Option<Self>`. `RecordIngest::metric_overrides`
  becomes `Vec<(MetricTagId, MetricValue)>`. `ReportView::metric` takes
  `MetricTagId`.

- [ ] **Step 1: Write the failing test**

Add to `rust/runtime/src/endpoints/type_id.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_new_endpoint_type_registers_without_touching_the_enum() {
        // Registering an endpoint kind must not require editing a closed enum;
        // that is what makes "add an endpoint" a plugin-local change.
        let mut registry = EndpointTypeRegistry::builtin();
        let id = registry
            .register("audio_transcription")
            .expect("register a new endpoint type");
        assert_eq!(id.as_str(), "audio_transcription");
        assert_eq!(EndpointTypeId::resolve_in(&registry, "audio_transcription"), Some(id));
    }

    #[test]
    fn duplicate_registration_is_rejected() {
        let mut registry = EndpointTypeRegistry::builtin();
        registry.register("chat").expect_err("chat is already built in");
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
source .venv/bin/activate && cd rust
cargo test -p aiperf-runtime a_new_endpoint_type_registers_without_touching_the_enum
```

Expected: FAIL — `cannot find type EndpointTypeRegistry in this scope`.

- [ ] **Step 3: Implement the interned IDs**

Create `rust/runtime/src/endpoints/type_id.rs`:

```rust
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Open endpoint-kind identity.
//!
//! A closed enum makes "add an endpoint" a universe-wide rebuild, which defeats
//! the point of endpoint plugins. Kinds are interned to a dense `u32` at
//! registration, so comparison and table indexing stay as cheap as the enum
//! discriminant they replace.

/// Dense index of one registered endpoint kind.
///
/// `Copy` and `u32`-sized: this is used as a dense table key on the prepared
/// endpoint path and must not become a string comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EndpointTypeId(u32);
```

Give the registry a `builtin()` constructor that interns today's 19 names **in
their current declaration order**, so existing dense indices are unchanged.
Apply the same shape in `metrics_core/tag_id.rs` for `MetricTagId` over today's
60 tags.

Critical compatibility constraint: `EndpointType` is serialized. The 40-task
plan already requires "transparent `EndpointType` source spelling and
serialization compatibility". Serialize `EndpointTypeId` through `as_str()` so
every existing config, artifact, and golden file round-trips unchanged. Add a
round-trip test over all 19 built-in names asserting the exact prior strings.

`RecordIngest::metric_overrides` at `ingest.rs:217` is a per-record hot-path
field. `MetricTagId` is `Copy` and `u32`, same as the enum discriminant, so this
is not a hot-path regression — but Task 5's benchmark will confirm rather than
assume.

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
source .venv/bin/activate && cd rust
cargo test -p aiperf-runtime && cargo test -p aiperf-runtime --features engine
```

Expected: PASS. Every serialization golden must be byte-identical.

- [ ] **Step 5: Commit**

```bash
git add rust/runtime/src/endpoints/type_id.rs rust/runtime/src/metrics_core/tag_id.rs \
        rust/runtime/src/endpoints rust/runtime/src/metrics_core rust/xtask/abi-baseline.json
git commit -F - <<'EOF'
refactor(endpoints,metrics): intern endpoint kinds and metric tags

EndpointType (19 variants) and MetricTag (60 variants) were closed enums in
the plugin ABI, so adding an endpoint or a metric -- the two things plugins
exist to do -- rebuilt every plugin. Both become dense interned u32 ids with
identical serialized spelling and identical table-index cost.
EOF
```

---

### Task 5: De-generify the thread-per-core executor

**This is the risky one. It touches the request hot path and it is the task most
likely to be rejected on measurement. Do it last among the refactors and do not
proceed past Step 5 without the benchmark.**

**Why:** `ExecutionSinkBuilder` carries an associated type
(`engine/turn_execution.rs:257`), and the executor is generic over it end to end:
`dimension_sink: B::Sink` (`:931`), `impl<B: ExecutionSinkBuilder>
ThreadPerCoreExecutor<B>` (`:1189`, `:1545`, `:1830`, `:1919`). A `cdylib`
plugin cannot supply an associated type across a library boundary, so `B` must
be erased. Nothing in the 40-task plan addresses this (0 mentions), and it is
load-bearing: without it there are no transport plugins at all.

**Countervailing constraint:** `design.md` Goals — "Add no abstraction or
dispatch layer to request- or token-processing paths and introduce no
statistically significant benchmark regression." This task can violate that.
The benchmark is the gate, not a formality.

Partial mitigation already present: `WorkerSink` is `#[async_trait(?Send)]`, so
`dispatch_measured` already returns a boxed future. The added cost is the vtable
call, not a new allocation.

**Files:**
- Modify: `rust/runtime/src/engine/turn_execution.rs:255-267` (`ExecutionSinkBuilder`)
- Modify: `rust/runtime/src/engine/turn_execution.rs:931` (`dimension_sink`)
- Modify: `rust/runtime/src/engine/turn_execution.rs:1189,1545,1830,1919`
- Modify: `rust/runtime/src/engine/ws_execution.rs:291`
- Modify: `rust/runtime/src/engine/grpc_turn_execution.rs:104`
- Test: `rust/runtime/benches/chat_dispatch_bench.rs`

**Interfaces:**
- Consumes: `Box<dyn CreditMaterializer>` from Task 2.
- Produces: `ExecutionSinkBuilder::build_sink` returns
  `Result<Box<dyn WorkerSinkExec>>` where
  `pub trait WorkerSinkExec: WorkerSink + RequestExecutor {}` with a blanket
  impl. `ThreadPerCoreExecutor` loses its type parameter.

- [ ] **Step 1: Record the pre-change benchmark**

Run:
```bash
source .venv/bin/activate && cd rust
cargo bench -p aiperf-runtime --bench chat_dispatch_bench -- --save-baseline pre-erase
```

Expected: a saved criterion baseline named `pre-erase`. Record the mean and the
confidence interval in the commit message of Step 6. **Do not skip this** — after
the refactor there is no way to reconstruct it without reverting.

- [ ] **Step 2: Write the failing test**

Add to the `mod tests` block in `rust/runtime/src/engine/turn_execution.rs`:

```rust
#[test]
fn a_transport_builds_a_sink_without_naming_its_concrete_type() {
    // A cdylib plugin cannot supply an associated type across the library
    // boundary, so the executor must accept an erased sink.
    fn build_erased(builder: &dyn ExecutionSinkBuilder) -> Box<dyn WorkerSinkExec> {
        let clock: Rc<dyn Clock> = Rc::new(RealClock::new());
        builder.build_sink(clock, 0).expect("build erased sink")
    }

    let builder = TestSinkBuilder::default();
    let sink = build_erased(&builder);
    assert!(sink.supports_response_streaming());
}
```

Note `&dyn ExecutionSinkBuilder` in the signature: that is itself the assertion.
The trait is not object-safe today because of the associated type, so this test
cannot compile until the erasure lands.

- [ ] **Step 3: Run test to verify it fails**

Run:
```bash
source .venv/bin/activate && cd rust
cargo test -p aiperf-runtime --features engine a_transport_builds_a_sink_without_naming
```

Expected: FAIL — `the trait ExecutionSinkBuilder cannot be made into an object`
… `because it contains the generic associated type Sink`.

- [ ] **Step 4: Erase the associated type**

In `engine/turn_execution.rs`:

```rust
/// A worker-local sink that both measures and executes.
///
/// Blanket-implemented, so a transport implements `WorkerSink` and
/// `RequestExecutor` and gets this for free. It exists only to give the
/// executor one object-safe supertrait to store.
pub trait WorkerSinkExec: WorkerSink + RequestExecutor {}

impl<T: WorkerSink + RequestExecutor> WorkerSinkExec for T {}

/// Constructs a `!Send` transport sink inside each worker reactor.
pub trait ExecutionSinkBuilder: Send + Sync + 'static {
    /// Short worker-thread name infix (e.g. `"http"`, `"grpc"`).
    fn label(&self) -> &'static str;

    /// Build one worker-local sink on `clock` for `worker_id`.
    fn build_sink(&self, clock: Rc<dyn Clock>, worker_id: usize)
        -> Result<Box<dyn WorkerSinkExec>>;

    /// Build this worker's materializer for identity-only credits, if the run
    /// routes them. `None` keeps issuer-side materialization.
    fn build_credit_materializer(&self) -> Result<Option<Box<dyn CreditMaterializer>>> {
        Ok(None)
    }
}
```

Then drop `<B>` from `ThreadPerCoreExecutor` at `:1189`, `:1545`, `:1830`, and
`:1919`, and change `dimension_sink: B::Sink` at `:931` to
`dimension_sink: Box<dyn WorkerSinkExec>`. Update the three implementations
(`HttpSinkBuilder` `:307`, `GrpcSinkBuilder` `grpc_turn_execution.rs:104`,
`WebSocketSinkBuilder` `ws_execution.rs:291`) to box their return.

Keep `#[async_trait(?Send)]` on `WorkerSink`. Do **not** take this opportunity
to change anything else about the executor — a mixed diff here is unreviewable
and unbisectable.

- [ ] **Step 5: Run tests to verify they pass**

Run:
```bash
source .venv/bin/activate && cd rust
cargo test -p aiperf-runtime && cargo test -p aiperf-runtime --features engine
```

Expected: PASS across HTTP, gRPC, WebSocket, dry-run, and every dispatch mode
(`sharded`, `global`, `global-hop`, `global-push`).

- [ ] **Step 6: Run the benchmark gate**

Run:
```bash
source .venv/bin/activate && cd rust
cargo bench -p aiperf-runtime --bench chat_dispatch_bench -- --baseline pre-erase
```

Expected: criterion reports **no statistically significant regression**.

**If it regresses:** stop. Do not commit a hot-path regression to satisfy a
packaging goal. Record the measured delta and escalate the design question:
either the transport boundary moves up a level (the executor stays
monomorphized and only the *sink construction* crosses the boundary), or the
performance goal in `design.md` is amended with the measured number. Both are
legitimate; silently absorbing the regression is not.

- [ ] **Step 7: Commit**

```bash
git add rust/runtime/src/engine/turn_execution.rs \
        rust/runtime/src/engine/ws_execution.rs \
        rust/runtime/src/engine/grpc_turn_execution.rs
git commit -F - <<'EOF'
refactor(engine): erase the transport sink associated type

ExecutionSinkBuilder carried an associated Sink type and ThreadPerCoreExecutor
was generic over it, so the trait was not object-safe and no cdylib transport
could implement it. Sinks are now Box<dyn WorkerSinkExec>.

chat_dispatch_bench vs pre-erase baseline: <paste criterion summary>
EOF
```

---

### Task 6: Split the type/impl seam and gate implementation in the ABI crates

**Why:** `host_abi_universe_id` takes `abi_facing_compiled_crate_artifact ->
digest` (`design.md:488`). That is the digest of the compiled `.rlib`, which
changes for **any** change to the crate — including a private function body. So
the universe-bump rate is driven by the ABI crates' whole contents, not just
their type definitions. Today, 94% of the lines in the 52 would-be ABI files are
implementation:

```
impl / total   (ABI)  commits/400   file
3941 /  4054   ( 113)      54       engine/turn_execution.rs
3678 /  3776   (  98)      57       multiturn.rs
2875 /  2916   (  41)      21       metrics_core/accumulator.rs
2221 /  2298   (  77)      30       body_plan.rs
1638 /  1683   (  45)      19       scheduled.rs
```

`design.md` assigns "endpoint body-planning and response-reduction helpers" to
`aiperf-core`. As written that puts all 2,221 implementation lines of
`body_plan.rs` in the universe, so its 30-commits-per-400 churn becomes 30
full-fleet rebuilds. The helpers belong in `aiperf-endpoint-sdk`
(plugin-private, selective rebuild); only the values belong in core.

**Trap to respect:** `recompile-boundaries.md:160` — a category SDK containing
*any* boundary type voids the selective-rebuild claim. The split must be clean
in both directions. `BodyPlan`, `RequestBody`, `FieldProgram`, `FieldValue`, and
`LiteralValue` go to core; the planning logic goes to the endpoint SDK; neither
file keeps a foot in both camps.

**Files:**
- Modify: `rust/runtime/src/body_plan.rs` → split into
  `body_plan/model.rs` (types) and `body_plan/plan.rs` (logic)
- Modify: `rust/runtime/src/multiturn.rs` → extract types to `multiturn/model.rs`
- Modify: `rust/runtime/src/metrics_core/accumulator.rs` → extract types to
  `metrics_core/accumulator_model.rs`
- Modify: `rust/runtime/src/scheduled.rs` → extract `TurnResponseObserver` and
  the observation values to `scheduled/observe.rs`
- Create: `rust/xtask/src/abi_impl_budget.rs`
- Modify: `.github/workflows/rust-abi-closure.yml`

**Interfaces:**
- Consumes: the closure tool from Task 1 and every eviction from Tasks 2–5.
- Produces: `cargo xtask abi-impl-budget` printing
  `{"type_lines": <usize>, "impl_lines": <usize>, "ratio": <f64>}` and failing
  when implementation lines in ABI-contributing files exceed the committed
  budget.

- [ ] **Step 1: Write the failing budget test**

Add `rust/xtask/tests/impl_budget.rs`:

```rust
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The universe id hashes the compiled ABI crate artifact, so implementation
//! co-resident with boundary types is implementation that rebuilds every plugin.

use aiperf_xtask::abi_impl_budget::measure;

#[test]
fn abi_files_are_mostly_type_definitions() {
    let m = measure().expect("measure");
    assert!(
        m.ratio < 0.50,
        "ABI-contributing files are {:.0}% implementation ({} impl lines); \
         boundary types must not share a file with logic that churns",
        m.ratio * 100.0,
        m.impl_lines
    );
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
source .venv/bin/activate && cd rust
cargo test -p aiperf-xtask --test impl_budget
```

Expected: FAIL — ratio is 0.94, far above the 0.50 threshold.

- [ ] **Step 3: Split the four hot files**

For each of `body_plan.rs`, `multiturn.rs`, `metrics_core/accumulator.rs`, and
`scheduled.rs`: move the ABI type definitions into a sibling `*_model.rs` (or
`model.rs` submodule), leave every `impl` block, free function, and private
helper in the original file, and re-export from the original path so no import
site changes. This is a pure move — no signature, no behavior, no field order
changes.

Work one file per commit. A four-file move in one commit is not reviewable and
not bisectable if a report snapshot shifts.

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
source .venv/bin/activate && cd rust
cargo test -p aiperf-runtime && cargo test -p aiperf-runtime --features engine
```

Expected: PASS, byte-identical report snapshots.

- [ ] **Step 5: Verify the budget and the churn rate together**

Run:
```bash
source .venv/bin/activate && cd rust
cargo test -p aiperf-xtask --test impl_budget
cargo run -p aiperf-xtask -- abi-churn --merges 120
```

Expected: budget test PASS, and `abi-churn` reporting file-granular and
type-granular universe rates that have **converged** — the whole point of the
split is that the two numbers stop differing. Target: 7–8 of 54 code units
(13–15%), down from 19 (35%).

- [ ] **Step 6: Commit and wire the gate**

```bash
git add rust/runtime/src rust/xtask .github/workflows/rust-abi-closure.yml
git commit -F - <<'EOF'
refactor(runtime): separate boundary types from implementation

host_abi_universe_id hashes the compiled ABI crate artifact, not its type
definitions, so implementation co-resident with boundary types rebuilds every
plugin on every change. The four hottest ABI files were 94% implementation.

Adds an xtask budget gate so the ratio cannot silently regress.
Universe-bump rate: 35% file-granular -> <paste measured> after the split.
EOF
```

---

## Self-Review

**Spec coverage.** This plan deliberately covers only the gaps enumerated in the
"Relationship to the existing plan" table. `ExportConfig` (40-task plan Task 6)
and `RunContext` (Tasks 4/5) are covered there and intentionally absent here.
Every other measured leak edge from the 2026-08-27 audit has a task:
`WorkerMaterializer` → Task 2, `NativeReport` → Task 3, `EndpointType` and
`MetricTag` → Task 4, `ExecutionSinkBuilder` → Task 5, implementation
co-residency → Task 6, measurement → Task 1.

**Known gap, deliberately not closed.** Leak 5 from the audit — `config::model`
types reaching the boundary via `HopRouting <- ExecutionBackendConfig` and
`Distribution <- Rankings <- ResponseData <- ParsedResponse <-
TurnResponseObserver` — has no task here. Cutting it is worth a further ~5 types
and one universe-wide merge unit, but it overlaps 40-task-plan Task 18
("Normalize open transport and exporter Config v2 forms"), and splitting
ownership of the config surface across two plans invites exactly the kind of
conflict the tracker's single-owner rule exists to prevent. Raise it as an
amendment to Task 18 rather than implementing it here.

**Type consistency.** `CreditMaterializer` (Task 2) is consumed by name in Task
5's `ExecutionSinkBuilder`. `ReportView` (Task 3) is amended by Task 4's
`MetricTagId`. `WorkerSinkExec` appears only in Task 5. No task references a
type no task defines.

**Sequencing.** Task 1 first — measurement before change. Tasks 2, 3, and 4 are
mutually independent and may run in parallel worktrees. Task 5 last among the
refactors because it is the one that can be rejected on benchmark evidence.
Task 6 last overall, because its budget only makes sense once the closure has
stopped moving.

**Risk register.**

| Risk | Task | Mitigation |
|---|---|---|
| Hot-path regression from sink erasure | 5 | `chat_dispatch_bench` baseline captured *before*; explicit stop-and-escalate |
| Serialization drift from interning | 4 | Round-trip test over all 19 + 60 built-in names against exact prior strings |
| rustdoc JSON format instability | 1 | Pin the nightly in `rust-toolchain.toml`; the gate is reproducible or it is not a gate |
| Report snapshot drift from the view | 3 | Byte-identical snapshot assertion in Step 4 |
| Four-file move obscuring a real change | 6 | One file per commit, explicit in Step 3 |
