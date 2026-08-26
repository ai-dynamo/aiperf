# Native search-space shape inference Implementation Plan

> **Author:** Sol
>
> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve a truthful native boundary for upstream `d8d49e8c2a` until native generic adaptive search exists.

**Architecture:** Do not attach semantics to the inert `ProfileFlags::search_space` field. The Rust recipe planner remains a closed, typed axis system; its absence of a generic dimension parser is the reason the Python fix has no safe partial port.

**Tech Stack:** Rust 2024, clap, native recipe/search modules, Markdown campaign records, cargo test.

**Spec:** `docs/specs/2026-08-26-native-search-space-shape-inference.md`

## Global Constraints

- Do not implement shape inference until a native generic, repeatable `--search-space` parser and ask/tell planner are in scope.
- Do not make a normal profile run change shape merely because it contains an otherwise unexecuted search-space string.
- A future port must use TDD: each production behavior begins with an observed failing Rust test, and it must cover all requirements in the linked spec.
- Use `RUSTC_WRAPPER=sccache`, `SCCACHE_DIR=/mnt/4tb/sccache-port062`, and `CARGO_TARGET_DIR=/mnt/4tb/aiperf-target-port062` for Rust verification.

---

### Task 1: Close the non-applicable upstream repair

**Files:**
- Modify: `docs/porting-origin-main-campaign.md`
- Create: `artifacts/archives/origin-main-findings/commit-062-d8d49e8c2a.md`
- Create: `docs/specs/2026-08-26-native-search-space-shape-inference.md`
- Create: `docs/superpowers/plans/2026-08-26-native-search-space-shape-inference.md`

**Interfaces:**
- Consumes: upstream Python commit `d8d49e8c2adc76072625c0789ac1029967b639a5` and the current native `ProfileFlags`, `search`, `profile`, and `load` modules.
- Produces: an evidence-backed campaign disposition and a future feature contract; it produces no Rust public API or runtime behavior.

- [x] **Step 1: Inspect the exact upstream implementation and tests**

Read the upstream converter and Optuna changes with their unit tests. Record that the changed behavior requires both parsed arbitrary dimensions and a generic trial overlay.

- [x] **Step 2: Inspect the native execution seam**

Read `rust/cli/src/flags.rs`, `rust/cli/src/profile.rs`, `rust/cli/src/search.rs`, `rust/cli/src/bayes.rs`, and `rust/cli/src/load.rs`. Establish that recipes use only `AxisKind` values for concurrency, ISL, and OSL and that `search_space` has no consumer.

- [x] **Step 3: Record the no-op decision**

Write the finding, spec, plan, and campaign row. State the future prerequisites precisely and do not add a test that merely duplicates an absent Python API.

- [x] **Step 4: Verify and commit closure evidence**

Run the focused native search tests and formatting check with the dedicated target/cache. Review the documentation-only diff with the Graham rubric, add the receipt, then commit the closure records.
