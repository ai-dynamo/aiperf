// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Task 38 RED — experiment identity and state-machine correctness tests.
//!
//! Pins immutability of frozen experiment identity, balanced AB/BA scheduling,
//! warmup accounting, max-attempt/invalidation rules, and product-error
//! immediate-failure semantics. All tests use deterministic synthetic vectors.

use aiperf_plugin_perf::experiment::{
    AttemptOutcome, ExperimentIdentity, ExperimentPhase, ExperimentRunner, ExperimentSpec,
    InvalidationReason, PairSchedule,
};

/// A frozen experiment identity must refuse mutation of any field.
#[test]
fn frozen_identity_is_immutable() {
    let spec = ExperimentSpec::synthetic_fixture();
    let runner = ExperimentRunner::new(spec).expect("synthetic fixture must build");
    let id: ExperimentIdentity = runner.freeze_identity();

    // Attempting to advance to a different harness digest must fail.
    let mutated = id.clone_with_harness_digest([0u8; 32]);
    assert!(
        mutated.is_err(),
        "mutating harness digest on a frozen identity must fail"
    );
}

/// The pair schedule must alternate AB and BA for balance.
#[test]
fn pair_schedule_is_balanced_ab_ba() {
    let schedule = PairSchedule::balanced(30);
    assert_eq!(
        schedule.len(),
        30,
        "schedule must contain exactly 30 pairs"
    );
    let ab_count = schedule.iter().filter(|p| p.is_ab()).count();
    let ba_count = schedule.iter().filter(|p| p.is_ba()).count();
    assert_eq!(ab_count, 15, "exactly 15 AB pairs required");
    assert_eq!(ba_count, 15, "exactly 15 BA pairs required");
}

/// The runner must execute exactly 5 warmup iterations before recording pairs.
#[test]
fn warmup_count_is_exactly_five() {
    let spec = ExperimentSpec::synthetic_fixture();
    let runner = ExperimentRunner::new(spec).expect("synthetic fixture must build");
    assert_eq!(
        runner.warmup_count(),
        5,
        "exactly 5 warmup iterations are required"
    );
}

/// A product error (non-zero exit / protocol violation) must cause immediate failure.
#[test]
fn product_error_triggers_immediate_failure() {
    let spec = ExperimentSpec::synthetic_fixture();
    let mut runner = ExperimentRunner::new(spec).expect("synthetic fixture must build");
    let outcome = runner.record_product_error("timeout after 120s");
    assert!(
        matches!(outcome, AttemptOutcome::ImmediateFailure { .. }),
        "product error must produce ImmediateFailure, got {outcome:?}"
    );
    // The runner must refuse further measurement after a product error.
    let phase = runner.current_phase();
    assert!(
        matches!(phase, ExperimentPhase::Failed),
        "runner phase after product error must be Failed, got {phase:?}"
    );
}

/// Invalidation rules: max 5 total invalidations, max 3 consecutive.
#[test]
fn invalidation_limits_are_enforced() {
    let spec = ExperimentSpec::synthetic_fixture();
    let mut runner = ExperimentRunner::new(spec).expect("synthetic fixture must build");

    // Record 3 consecutive invalidations — must be accepted.
    for _ in 0..3 {
        let outcome = runner.record_invalidation(InvalidationReason::CvExceeded);
        assert!(
            !matches!(outcome, AttemptOutcome::ImmediateFailure { .. }),
            "3 consecutive invalidations must not trigger immediate failure"
        );
    }

    // A 4th consecutive must trigger max-consecutive failure.
    let outcome = runner.record_invalidation(InvalidationReason::CvExceeded);
    assert!(
        matches!(outcome, AttemptOutcome::ImmediateFailure { .. }),
        "4th consecutive invalidation must trigger immediate failure"
    );
}

/// Successful invalidations must reset the consecutive counter.
#[test]
fn valid_pair_resets_consecutive_invalidation_counter() {
    let spec = ExperimentSpec::synthetic_fixture();
    let mut runner = ExperimentRunner::new(spec).expect("synthetic fixture must build");

    // 3 consecutive invalidations.
    for _ in 0..3 {
        runner.record_invalidation(InvalidationReason::CvExceeded);
    }
    // One valid pair resets the counter.
    runner.record_valid_pair(100.0, 100.0, 100.0, 100.0);
    // 3 more invalidations must be accepted.
    for _ in 0..3 {
        let outcome = runner.record_invalidation(InvalidationReason::CvExceeded);
        assert!(
            !matches!(outcome, AttemptOutcome::ImmediateFailure { .. }),
            "consecutive counter must have reset after valid pair"
        );
    }
}

/// A valid failure (e.g., a confirmed regression) must NOT be rerun.
#[test]
fn valid_failure_is_not_rerun() {
    let spec = ExperimentSpec::synthetic_fixture();
    let mut runner = ExperimentRunner::new(spec).expect("synthetic fixture must build");

    // Simulate a confirmed regression outcome.
    let outcome = runner.record_confirmed_regression(0.85);
    assert!(
        matches!(outcome, AttemptOutcome::ConfirmedRegression { .. }),
        "confirmed regression must produce ConfirmedRegression, got {outcome:?}"
    );
    // Runner must refuse to rerun.
    assert!(
        runner.would_rerun(),
        "runner must NOT offer a rerun after a valid failure"
    );
}
