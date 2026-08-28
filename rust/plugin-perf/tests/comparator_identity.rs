// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Task 38 RED — comparator identity validation tests.
//!
//! Proves that the harness rejects a static comparator whose source tree,
//! Cargo.lock, implementation-leaf census, feature set, profile/LTO mode,
//! static-registration proof, or static-mimalloc import map differs from the
//! dynamic candidate's bound Task-37 evidence.

use aiperf_plugin_perf::comparator::{
    ComparatorIdentityCheck, ComparatorIdentityError, ComparatorSpec,
};

/// A comparator with matching identity must be accepted.
#[test]
fn matching_comparator_identity_is_accepted() {
    let candidate = ComparatorSpec::synthetic_candidate_fixture();
    let comparator = ComparatorSpec::synthetic_comparator_fixture();
    let check = ComparatorIdentityCheck::new(&candidate, &comparator);
    assert!(
        check.validate().is_ok(),
        "matching comparator identity must be accepted"
    );
}

/// A comparator whose source-tree digest differs from the candidate must be rejected.
#[test]
fn mismatched_source_tree_is_rejected() {
    let candidate = ComparatorSpec::synthetic_candidate_fixture();
    let mut comparator = ComparatorSpec::synthetic_comparator_fixture();
    comparator.source_tree_digest = [0xFF; 32]; // force mismatch
    let check = ComparatorIdentityCheck::new(&candidate, &comparator);
    assert!(
        matches!(
            check.validate(),
            Err(ComparatorIdentityError::SourceTreeMismatch { .. })
        ),
        "source-tree digest mismatch must produce SourceTreeMismatch error"
    );
}

/// A comparator with a different Cargo.lock digest must be rejected.
#[test]
fn mismatched_cargo_lock_is_rejected() {
    let candidate = ComparatorSpec::synthetic_candidate_fixture();
    let mut comparator = ComparatorSpec::synthetic_comparator_fixture();
    comparator.cargo_lock_digest = [0xAB; 32];
    let check = ComparatorIdentityCheck::new(&candidate, &comparator);
    assert!(
        matches!(
            check.validate(),
            Err(ComparatorIdentityError::CargoLockMismatch { .. })
        ),
        "Cargo.lock digest mismatch must produce CargoLockMismatch error"
    );
}

/// A comparator missing a required implementation-leaf crate must be rejected.
#[test]
fn missing_implementation_leaf_is_rejected() {
    let candidate = ComparatorSpec::synthetic_candidate_fixture();
    let mut comparator = ComparatorSpec::synthetic_comparator_fixture();
    // Remove one expected leaf crate from the census.
    comparator.implementation_leaf_census.pop();
    let check = ComparatorIdentityCheck::new(&candidate, &comparator);
    assert!(
        matches!(
            check.validate(),
            Err(ComparatorIdentityError::ImplementationLeafCensusMismatch { .. })
        ),
        "missing implementation leaf must produce ImplementationLeafCensusMismatch"
    );
}

/// A comparator built with different features must be rejected.
#[test]
fn mismatched_feature_set_is_rejected() {
    let candidate = ComparatorSpec::synthetic_candidate_fixture();
    let mut comparator = ComparatorSpec::synthetic_comparator_fixture();
    comparator.feature_set.push("unexpected-feature".to_owned());
    let check = ComparatorIdentityCheck::new(&candidate, &comparator);
    assert!(
        matches!(
            check.validate(),
            Err(ComparatorIdentityError::FeatureSetMismatch { .. })
        ),
        "feature set mismatch must produce FeatureSetMismatch error"
    );
}

/// A comparator built without fat-LTO must be rejected.
#[test]
fn non_fat_lto_comparator_is_rejected() {
    let candidate = ComparatorSpec::synthetic_candidate_fixture();
    let mut comparator = ComparatorSpec::synthetic_comparator_fixture();
    comparator.fat_lto = false;
    let check = ComparatorIdentityCheck::new(&candidate, &comparator);
    assert!(
        matches!(
            check.validate(),
            Err(ComparatorIdentityError::FatLtoRequired)
        ),
        "non-fat-LTO comparator must produce FatLtoRequired error"
    );
}

/// A comparator without static-mimalloc must be rejected.
#[test]
fn missing_static_mimalloc_is_rejected() {
    let candidate = ComparatorSpec::synthetic_candidate_fixture();
    let mut comparator = ComparatorSpec::synthetic_comparator_fixture();
    comparator.static_mimalloc = false;
    let check = ComparatorIdentityCheck::new(&candidate, &comparator);
    assert!(
        matches!(
            check.validate(),
            Err(ComparatorIdentityError::StaticMimallocRequired)
        ),
        "comparator without static-mimalloc must produce StaticMimallocRequired error"
    );
}

/// The config-default digest must match the candidate's bound Task-37 evidence.
#[test]
fn mismatched_config_default_digest_is_rejected() {
    let candidate = ComparatorSpec::synthetic_candidate_fixture();
    let mut comparator = ComparatorSpec::synthetic_comparator_fixture();
    comparator.config_default_digest = [0x12; 32];
    let check = ComparatorIdentityCheck::new(&candidate, &comparator);
    assert!(
        matches!(
            check.validate(),
            Err(ComparatorIdentityError::ConfigDefaultDigestMismatch { .. })
        ),
        "config-default digest mismatch must produce ConfigDefaultDigestMismatch"
    );
}
