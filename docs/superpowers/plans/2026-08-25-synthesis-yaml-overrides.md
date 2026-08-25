# Synthesis YAML Overrides Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore native CLI-over-YAML precedence for `--synthesis-*` flags without changing unrelated YAML resolution behavior.

**Architecture:** Extend the YAML override path in `rust/cli/src/yaml.rs` so it merges explicit synthesis CLI fields into `Inputs.synthesis` instead of dropping them. Keep `load.rs` as the CLI-only synthesis authoring path and let existing runtime validation continue to police `baseten_trace` constraints.

**Tech Stack:** Rust, clap-derived `ProfileFlags`, serde_json, cargo test

**Spec:** `docs/specs/2026-08-25-native-synthesis-yaml-overrides.md`

## Global Constraints

- Use TDD: every production change follows a recorded failing Rust test.
- Do not modify the shared checkout; work only in the isolated `/mnt/4tb` worktree.
- Preserve exact upstream ancestry by merging `6480e5467f` with a non-fast-forward merge commit.
- Use `/usr/bin/sccache` and a Cargo target under `/mnt/4tb` for Rust verification.

---

### Task 1: Capture scope and exact upstream ancestry

**Files:**
- Create: `docs/specs/2026-08-25-native-synthesis-yaml-overrides.md`
- Create: `docs/superpowers/plans/2026-08-25-synthesis-yaml-overrides.md`

- [ ] Commit the spec and plan.
- [ ] Merge upstream commit `6480e5467f` exactly with `--no-ff`.

### Task 2: Prove the native YAML precedence bug

**Files:**
- Modify: `rust/cli/src/yaml.rs`

- [ ] Add a focused failing unit test showing `--synthesis-max-osl` does not currently override a YAML `dataset.synthesis.maxOsl`.
- [ ] Run the targeted test and record the RED failure.

### Task 3: Implement synthesis overlay semantics

**Files:**
- Modify: `rust/cli/src/yaml.rs`

- [ ] Add the minimal synthesis-overlay helper(s) needed to merge explicit CLI synthesis fields into `Inputs.synthesis`.
- [ ] Keep YAML-authored synthesis fields intact when the CLI did not set them.
- [ ] Preserve existing baseten/runtime validation behavior.

### Task 4: Port the applicable upstream unit coverage and close the record

