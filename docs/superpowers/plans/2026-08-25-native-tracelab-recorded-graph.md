# Native TraceLab Recorded-Graph Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Load TraceLab plain/gzip JSONL through the native recorded-graph path with faithful timing, cache-block reuse, subagent recovery, config routing, and executable integration coverage.

**Architecture:** Convert acquired TraceLab rows into the existing strict WEKA-shaped in-memory format, then reuse the native WEKA Graph-IR compiler. Register `tracelab` as a graph format and preserve the original format identity on the final bundle.

**Tech Stack:** Rust 2024, serde_json, chrono, flate2, native Graph-IR/WEKA compiler, clap/config v2, cargo tests with sccache.

**Spec:** `docs/specs/2026-08-25-native-tracelab-recorded-graph.md`

## Global Constraints

- Work only in `/mnt/4tb/ajc/port-044-082a51827e` based on `f423b618da`.
- Every production behavior begins with a failing Rust test and observed failure.
- Use `RUSTC_WRAPPER=sccache` and `CARGO_TARGET_DIR=/mnt/4tb/aiperf-target-port044`.
- The final merge has exact upstream `082a51827eb9f755aacb93122cafe0383cf99f6e` as second parent and retains the completed native first-parent tree.
- Do not add a parallel replay engine or import upstream Python implementation files.

---

### Task 1: TraceLab conversion and source acquisition

**Files:**
- Create: `rust/runtime/src/graph/recorded/tracelab.rs`
- Modify: `rust/runtime/src/graph/recorded/mod.rs`

**Interfaces:**
- Produces: `pub async fn compile_tracelab_trace_input(RecordedTraceInputConfig, &dyn TextTokenizer) -> Result<GraphInputBundle, RecordedTraceError>`.
- Consumes: existing `compile_weka_trace_input` through an inline array of strict WEKA documents.

- [x] **Step 1: Write failing converter/source tests** covering timing/order/hash compaction, Claude/Codex joins and fallback, plain/gzip/error behavior, and safe identifiers.
- [x] **Step 2: Run the focused runtime test filter** and verify failure because the module/API does not exist.
- [x] **Step 3: Implement the minimum typed conversion and one-read source acquisition** using stable session order, shared trace-local hash minters, deterministic containment matching, and explicit option validation.
- [x] **Step 4: Run the focused test filter** and verify all TraceLab converter/source tests pass.
- [x] **Step 5: Commit** conversion, tests, finding, design and plan.

### Task 2: Graph registration and config projection

**Files:**
- Modify: `rust/runtime/src/config/model/workload_kind.rs`
- Modify: `rust/runtime/src/engine/graph_input.rs`
- Modify: `rust/runtime/src/engine/cellular_kind.rs`
- Modify: `rust/runtime/src/engine/cellular_controller.rs`
- Modify: `rust/runtime/src/engine/cellular_cell.rs`
- Modify: `rust/runtime/src/config/resolve.rs`
- Modify: `rust/cli/src/load.rs`

**Interfaces:**
- Produces: a built-in `tracelab` graph adapter and CLI/config projection with `options.block_size`.
- Consumes: `compile_tracelab_trace_input` from Task 1.

- [x] **Step 1: Write failing adapter, workload-classification and CLI projection tests** for `--input-file`, `--custom-dataset-type tracelab`, and `--isl-block-size 128`.
- [x] **Step 2: Run the focused engine/CLI tests** and verify unknown-format or scheduled-workload failure.
- [x] **Step 3: Register the adapter and update every native graph-format boundary** while keeping one authoritative inventory where available.
- [x] **Step 4: Run the focused engine/CLI tests** and verify graph routing plus exact option projection.
- [x] **Step 5: Commit** registration/config behavior and tests.

### Task 3: Real compiler and native-binary integration

**Files:**
- Create: `rust/runtime/tests/tracelab_recorded_graph.rs`
- Modify: the narrow existing Rust E2E test module selected after inventory.

**Interfaces:**
- Consumes: public compiler/adapter and built `aiperf` binary.
- Produces: external integration evidence over actual fixture files and runtime output.

- [x] **Step 1: Write a failing real-file integration test** that loads plain and gzip corpora, checks equivalent Graph-IR, subagent topology, root caps and block-size effects.
- [x] **Step 2: Run it and verify the expected missing-registration/compiler failure.**
- [x] **Step 3: Add the minimum native-binary dry-run integration** using the original TraceLab fixture and JSON artifact assertions.
- [x] **Step 4: Run both integration paths** and verify successful native loading/execution.
- [x] **Step 5: Commit** integration coverage.

### Task 4: Review, verification, ancestry and closure

**Files:**
- Create: `.superpowers/sdd/2026-08-25-native-tracelab-recorded-graph/graham-review.md`
- Create: `.superpowers/sdd/2026-08-25-native-tracelab-recorded-graph/graham-rereview.md`
- Modify: `artifacts/archives/origin-main-findings/commit-044-082a51827e.md`
- Modify: `docs/porting-origin-main-campaign.md`

**Interfaces:**
- Produces: reviewed implementation commit, exact two-parent ancestry merge and evidence-backed closure.

- [x] **Step 1: Run formatting, focused tests, runtime engine tests, CLI tests, Clippy, and native-binary integration with fresh output.**
- [x] **Step 2: Perform a full Graham review over every changed hunk, record all findings, fix them with regression tests, and rerun focused verification.**
- [x] **Step 3: Perform Graham re-review and record approval only if no finding remains.**
- [ ] **Step 4: Create the two-parent `ours`-tree merge with exact upstream as second parent; verify parent order, tree equality and absence of upstream Python import.**
- [x] **Step 5: Update tracker/finding closure evidence, rerun final checks, and commit the closure note.**
