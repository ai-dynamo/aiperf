// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pins the measured ABI closure so a boundary regression fails CI.

use aiperf_xtask::abi_closure::{Seeds, compute};

#[test]
fn closure_matches_committed_baseline() {
    let seeds = Seeds::load("abi-seeds.toml").expect("seed file");
    let measured = compute(&seeds).expect("closure");
    let baseline = std::fs::read_to_string("abi-baseline.json").expect("baseline");
    let baseline: serde_json::Value = serde_json::from_str(&baseline).expect("json");

    assert_eq!(
        measured.types.len(),
        baseline["types"].as_u64().expect("types") as usize,
        "ABI closure size changed; if intentional, regenerate abi-baseline.json"
    );
}

#[test]
fn closure_excludes_run_context() {
    let seeds = Seeds::load("abi-seeds.toml").expect("seed file");
    let measured = compute(&seeds).expect("closure");
    assert!(
        !measured.types.contains_key("RunContext"),
        "RunContext is forbidden at the plugin boundary (design.md)"
    );
}
