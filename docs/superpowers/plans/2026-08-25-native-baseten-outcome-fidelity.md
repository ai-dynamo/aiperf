# Native Baseten Recorded-Outcome Fidelity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve Baseten E2E duration, TTFT, and cached-token reference ground truth through the native loader and composer without affecting dispatch.

**Architecture:** Add one optional dispatch-neutral outcome record to `Turn`; carry the values through the existing private `BasetenRow` JSON bridge and attach the record during composition. Exercise real Parquet loading through both module-local and public-registry tests.

**Tech Stack:** Rust 2024, serde, parquet/Arrow, Tokio tests, `/usr/bin/sccache`.

**Spec:** `docs/specs/2026-08-25-native-baseten-outcome-fidelity.md`

## Global Constraints

- Base every commit on campaign HEAD `106019c5a18910f9b60fe5d1e4fa05fd8f31deba`.
- Do not import tracker #39's pending native or Python implementation.
- The ancestry merge's exact second parent is `215be05b6a534fb19b84bf83f711db2d20f5bea1`; do not cherry-pick it.
- Recorded outcomes never alter request bodies, scheduling, token accounting, or replay speedup behavior.
- Use `RUSTC_WRAPPER=/usr/bin/sccache` and `CARGO_TARGET_DIR=/mnt/4tb/aiperf-origin-port-040-target` for Rust verification.

---

### Task 1: Restore the mandated base's testability

**Files:**
- Modify: `rust/runtime/src/endpoints/mod.rs`

**Interfaces:**
- Consumes: the existing crate-private `implementation::capture_endpoint_policy` re-export.
- Produces: exactly one crate-private re-export and no endpoint behavior change.

- [x] Retain the baseline compiler output showing duplicate-import error `E0252`.
- [x] Remove only the duplicate re-export introduced by the campaign merge.
- [x] Re-run the focused Baseten suite and record the clean baseline count before adding #40 tests.
- [x] Commit the prerequisite independently from the semantic port.

### Task 2: Prove the missing native behavior

**Files:**
- Modify: `rust/runtime/src/dataset/loader/baseten.rs`
- Create: `rust/runtime/tests/baseten_outcome_fidelity.rs`

**Interfaces:**
- Consumes: real Baseten Parquet rows and `LoaderRegistry::with_builtin_formats`.
- Produces: failing unit assertions for default, omit-KV, closed-loop, and missing-value behavior plus a failing public-registry integration assertion.

- [x] Extend the private Parquet fixture with literal outcome values and add focused replay coverage.
- [x] Run the focused unit test and observe a compile failure because `Turn::recorded_outcome` does not exist.
- [x] Add the omit-KV, closed-loop, and missing-value assertions without duplicating equivalent cases.
- [x] Add a public-registry integration fixture with hand-derived literal outcome assertions.
- [x] Run the integration test and observe the same missing-interface failure.

### Task 3: Carry recorded outcomes without dispatch effects

**Files:**
- Modify: `rust/runtime/src/dataset/model.rs`
- Modify: `rust/runtime/src/dataset/loader/baseten.rs`

**Interfaces:**
- Produces: public `RecordedOutcome { duration_e2e_ms, duration_ttft_ms, cached_tokens_reference }` and `Turn::recorded_outcome: Option<RecordedOutcome>`.
- Consumes: optional Baseten source values through `parse_row`, `row_to_value`, and `row_from_value`.

- [x] Add the smallest serde-compatible `RecordedOutcome` model and defaulted `Turn` field.
- [x] Parse and round-trip `duration_ttft_ms` and `cached_tokens_reference` beside the existing E2E duration.
- [x] Attach an outcome only when at least one value exists; leave request construction unchanged.
- [x] Run the focused unit and integration tests and observe all cases pass.
- [x] Run direct rustfmt on both production files and the integration test.
- [x] Commit the semantic port and tests together.

### Task 4: Record exact ancestry and complete two Graham passes

**Files:**
- Modify: `docs/porting-origin-main-campaign.md`
- Modify: `artifacts/archives/origin-main-findings/commit-040-215be05b6a.md`
- Create: `.superpowers/sdd/2026-08-25-native-baseten-outcome-fidelity/graham-review.md`
- Create: `.superpowers/sdd/2026-08-25-native-baseten-outcome-fidelity/graham-rereview.md`

**Interfaces:**
- Consumes: completed native first-parent tree and exact upstream commit `215be05b6a534fb19b84bf83f711db2d20f5bea1`.
- Produces: a two-parent ours-tree merge, concrete verification receipt, and zero unresolved Critical or Important Graham findings.

- [x] Review every changed hunk for unnecessary allocation/cloning, serde compatibility, dispatch leakage, naming, comments, and test duplication.
- [x] Fix every validated review finding and record the first-pass findings.
- [x] Re-review the corrected range from the exact campaign base and record the final verdict.
- [x] Create the two-parent ours-tree merge and prove its second parent and tree identity.
- [x] Run focused unit and integration tests, runtime library tests with `parquet`, Clippy, formatting, docs checks, and exact-range whitespace checks.
- [x] Update the campaign ledger and finding with exact commit ids, test counts, ancestry proof, and Graham verdict.
