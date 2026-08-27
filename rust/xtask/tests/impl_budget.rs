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
