// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Comparator census: the statically-linked comparison build must carry the
//! same component census the dynamic default distribution publishes.
//!
//! The comparator exists so a parity run has a baseline that is identical in
//! everything but linkage. If its census drifts from the default distribution,
//! any measured difference is a census difference, not a linkage difference,
//! and the parity number is meaningless.

use aiperf_plugin_static_comparator::static_inventory::{
    CensusError, DEFAULT_DISTRIBUTION_CENSUS, StaticComparatorRegistry,
    default_distribution_registry,
};

#[test]
fn static_census_matches_dynamic_default() {
    let registry = default_distribution_registry().expect("the default registry builds");
    registry
        .assert_census(DEFAULT_DISTRIBUTION_CENSUS)
        .expect("the static census matches the dynamic default census");

    let ids: Vec<&str> = DEFAULT_DISTRIBUTION_CENSUS
        .iter()
        .map(|(id, _)| *id)
        .collect();
    assert!(ids.contains(&"nvidia/transport-http"));
    assert!(ids.contains(&"nvidia/transport-grpc"));
    assert!(ids.contains(&"nvidia/transport-dry-run"));
    assert!(ids.contains(&"nvidia/endpoints"));
    assert!(ids.contains(&"nvidia/export-basic"));
    assert!(
        !ids.contains(&"nvidia/transport-dynosim"),
        "dynosim is feature-gated and is not in the default distribution"
    );
}

#[test]
fn a_census_that_drifts_is_refused() {
    let mut registry = StaticComparatorRegistry::new();
    registry
        .register("nvidia/export-basic", "0.13.0")
        .expect("first registration");
    let duplicate = registry
        .register("nvidia/export-basic", "0.13.0")
        .expect_err("a duplicate component id is refused");
    assert!(matches!(duplicate, CensusError::Duplicate(_)));

    let missing = registry
        .assert_census(&[
            ("nvidia/export-basic", "0.13.0"),
            ("nvidia/transport-http", "0.13.0"),
        ])
        .expect_err("a short census is refused");
    assert!(matches!(missing, CensusError::Mismatch { .. }));

    let version_drift = registry
        .assert_census(&[("nvidia/export-basic", "9.9.9")])
        .expect_err("a version drift is refused");
    assert!(matches!(version_drift, CensusError::Mismatch { .. }));
}
