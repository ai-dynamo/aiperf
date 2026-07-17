// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! `aiperf validate mooncake-trace` command contract coverage.

use std::process::Command;

fn fixture(name: &str) -> String {
    format!(
        "{}/../../tools/parity/validate/{name}",
        env!("CARGO_MANIFEST_DIR")
    )
}

#[test]
fn validate_pass_is_byte_exact() {
    let out = Command::new(env!("CARGO_BIN_EXE_aiperf"))
        .args([
            "validate",
            "mooncake-trace",
            "--input",
            &fixture("valid.jsonl"),
        ])
        .output()
        .expect("run");
    assert!(out.status.success());
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "Validation passed: 5 rows are Mooncake-compatible.\n"
    );
}

#[test]
fn validate_fail_flags_all_rows() {
    let out = Command::new(env!("CARGO_BIN_EXE_aiperf"))
        .args([
            "validate",
            "mooncake-trace",
            "--input",
            &fixture("invalid.jsonl"),
        ])
        .output()
        .expect("run");
    assert_eq!(out.status.code(), Some(1));
    let s = String::from_utf8_lossy(&out.stdout);
    assert!(
        s.starts_with("Validation failed with 4 error(s):"),
        "got: {s}"
    );
}
