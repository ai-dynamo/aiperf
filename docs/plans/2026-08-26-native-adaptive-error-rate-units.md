# Native adaptive error-rate units Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make native adaptive `error_rate` SLAs use the exported percentage-point unit and completed-request denominator.

**Architecture:** Keep rate-unit semantics centralized in `DefaultSlaEvaluator`, so CLI, YAML, and protocol-v2 adaptive configurations share validation and values. The controller continues to decide when a zero-success window is evaluable; the E2E checks the emitted controller artifact through the product binary.

**Tech Stack:** Rust 2024, native adaptive runtime, Cargo tests, Rust mock server.

**Spec:** `docs/specs/2026-08-26-native-adaptive-error-rate-units.md`

## Global Constraints

- `error_rate` and `request_error_rate` are percentage points in `[0, 100]` over successes plus errors.
- Cancellations remain excluded from the error-rate denominator; success and cancellation rates remain fractional.
- Preserve Clock-owned controller timing and the schema-v2 artifact contract.
- Use `RUSTC_WRAPPER=/usr/bin/sccache` and a target directory under `/mnt/4tb` for Cargo builds.

---

### Task 1: Specify native evaluator units and validation

**Files:**
- Modify: `rust/runtime/src/adaptive_core/sla.rs`

**Interfaces:**
- Consumes: `WindowStats::{successful_requests, errors, cancelled}`.
- Produces: `DefaultSlaEvaluator::values` percentage-point `error_rate` aliases and `validate_filters` bounds/warning behavior.

- [ ] **Step 1: Write the failing tests**

```rust
let stats = WindowStats { successful_requests: vec![sample(...), sample(...)], errors: 1, cancelled: 1, .. };
assert_eq!(value("request_error_rate", &stats), 100.0 / 3.0);
assert!(!passes(le(1.0), 100.0 / 3.0));
assert!(validate(le(-1.0)).is_err());
assert!(validate(le(101.0)).is_err());
```

- [ ] **Step 2: Run the focused test to verify it fails**

Run: `RUSTC_WRAPPER=/usr/bin/sccache CARGO_TARGET_DIR=/mnt/4tb/aiperf-target-port048 cargo test -p aiperf-runtime --lib adaptive_core::sla::tests::error_rate_matches_exported_percentage_unit_and_denominator`

Expected: FAIL because the evaluator returns `0.25`.

- [ ] **Step 3: Implement the minimal evaluator change**

```rust
let completed = stats.completed() + stats.errors;
let value = (completed > 0).then_some(100.0 * stats.errors as f64 / completed as f64).unwrap_or(0.0);
```

Validate the `[0, 100]` domain for only the two error-rate aliases and warn once during setup for `0 < threshold < 1`.

- [ ] **Step 4: Run the focused evaluator suite**

Run: `RUSTC_WRAPPER=/usr/bin/sccache CARGO_TARGET_DIR=/mnt/4tb/aiperf-target-port048 cargo test -p aiperf-runtime --lib adaptive_core::sla::tests`

Expected: PASS, including unchanged success/cancellation denominator tests.

- [ ] **Step 5: Commit**

```bash
git add rust/runtime/src/adaptive_core/sla.rs
git commit -m "fix(adaptive): align error-rate SLA units"
```

### Task 2: Prove product artifact behavior

**Files:**
- Modify: `rust/e2e-tests/tests/test_adaptive_scale.rs`

**Interfaces:**
- Consumes: the Config-v2 `adaptive_scale.sla` projection and deterministic Rust mock error injection.
- Produces: native-binary artifact evidence that `request_error_rate:avg:le:1` reports percentage points and fails above one percent.

- [ ] **Step 1: Write the failing E2E**

```rust
let mut cfg = MockServerConfig::default();
cfg.error_rate = 100.0;
// Run an adaptive error-rate-only phase with request_error_rate.avg.le: 1.
// Assert its adaptive_window SLA value is 100.0 and sla_passed is false.
```

- [ ] **Step 2: Run it to verify the pre-port behavior fails**

Run: `RUSTC_WRAPPER=/usr/bin/sccache CARGO_TARGET_DIR=/mnt/4tb/aiperf-target-port048 cargo test -p aiperf-e2e-tests --test test_adaptive_scale adaptive_error_rate -- --nocapture`

Expected: FAIL because a fractional value of `1.0` incorrectly passes `le: 1`.

- [ ] **Step 3: Keep the test product-facing**

Use the existing `AIPerfHarness`, temporary Config-v2 YAML, and emitted JSONL; do not add production test hooks.

- [ ] **Step 4: Run the adaptive E2E suite**

Run: `RUSTC_WRAPPER=/usr/bin/sccache CARGO_TARGET_DIR=/mnt/4tb/aiperf-target-port048 cargo test -p aiperf-e2e-tests --test test_adaptive_scale -- --nocapture`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add rust/e2e-tests/tests/test_adaptive_scale.rs
git commit -m "test(e2e): cover adaptive error-rate units"
```

### Task 3: Record closure and review

**Files:**
- Modify: `docs/porting-origin-main-campaign.md`, `docs/specs/README.md`, `docs/origin-main-findings/commit-048-260d00f5e9.md`

- [ ] **Step 1: Run formatting, targeted runtime, and E2E verification**

Run: `cargo fmt --check && RUSTC_WRAPPER=/usr/bin/sccache CARGO_TARGET_DIR=/mnt/4tb/aiperf-target-port048 cargo test -p aiperf-runtime --lib adaptive_core && RUSTC_WRAPPER=/usr/bin/sccache CARGO_TARGET_DIR=/mnt/4tb/aiperf-target-port048 cargo test -p aiperf-e2e-tests --test test_adaptive_scale -- --nocapture`

Expected: PASS.

- [ ] **Step 2: Perform two Graham review passes**

Review the final diff for hot-path allocations, error handling, logging, comments, test behavior, and unrelated surface.

- [ ] **Step 3: Commit closure record**

```bash
git add docs
git commit -m "docs(port): close adaptive error-rate units"
```
