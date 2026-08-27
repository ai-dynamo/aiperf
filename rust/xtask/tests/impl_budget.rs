// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The universe id hashes the compiled ABI crate artifact, so implementation
//! co-resident with boundary types is implementation that rebuilds every plugin.

use std::sync::OnceLock;

use aiperf_xtask::abi_impl_budget::{
    MAX_ABI_FILES, MAX_ABI_TYPES, MAX_GLOBAL_IMPL_LINES, Measurement, ensure_within_budget,
    measure,
};

fn measurement() -> Measurement {
    static MEASUREMENT: OnceLock<Measurement> = OnceLock::new();
    MEASUREMENT
        .get_or_init(|| measure().expect("measure"))
        .clone()
}

#[test]
fn authorized_splits_fit_the_scoped_budget() {
    let m = measurement();
    ensure_within_budget(&m).expect("ABI implementation budget");
}

#[test]
fn global_implementation_regression_is_rejected() {
    let mut m = measurement();
    m.impl_lines = MAX_GLOBAL_IMPL_LINES + 1;
    assert!(ensure_within_budget(&m).is_err());
}

#[test]
fn mixed_boundary_model_is_rejected() {
    let mut m = measurement();
    m.boundary_files[0].impl_lines = m.boundary_files[0].type_lines;
    m.boundary_files[0].ratio = 0.50;
    assert!(ensure_within_budget(&m).is_err());
}

#[test]
fn closure_growth_is_rejected() {
    let mut type_growth = measurement();
    type_growth.abi_types = MAX_ABI_TYPES + 1;
    assert!(ensure_within_budget(&type_growth).is_err());

    let mut file_growth = measurement();
    file_growth.abi_files = MAX_ABI_FILES + 1;
    assert!(ensure_within_budget(&file_growth).is_err());
}
