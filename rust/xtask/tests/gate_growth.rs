// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Prevents count-preserving ABI-universe substitutions from bypassing the gate.

use aiperf_xtask::abi_closure::{Baseline, Entry, ensure_no_growth};

#[test]
fn gate_rejects_a_new_type_when_the_total_count_is_unchanged() {
    let baseline = baseline_with("ExistingBoundary");
    let measured = baseline_with("NewPrivateImplementation");

    let error = ensure_no_growth(&measured, &baseline).expect_err("new ABI type must fail");
    assert!(
        error.to_string().contains("NewPrivateImplementation"),
        "gate error must name the new ABI type: {error}"
    );
}

fn baseline_with(name: &str) -> Baseline {
    Baseline {
        types: 1,
        files: 1,
        type_lines: 3,
        file_lines: 10,
        entries: vec![Entry {
            name: name.to_owned(),
            file: "runtime/src/boundary.rs".to_owned(),
            start: 2,
            end: 4,
        }],
    }
}
