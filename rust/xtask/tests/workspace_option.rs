// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Exercises source-workspace selection through the real xtask binary.

use std::process::Command;

#[test]
fn closure_validates_the_selected_workspace() {
    let missing_workspace = std::env::temp_dir().join("aiperf-abi-workspace-without-manifest");
    std::fs::create_dir_all(&missing_workspace).expect("test workspace");
    let output = Command::new(env!("CARGO_BIN_EXE_aiperf-xtask"))
        .args([
            "abi-closure",
            "--workspace",
            missing_workspace.to_str().expect("UTF-8 temporary path"),
        ])
        .output()
        .expect("run xtask");

    assert!(!output.status.success(), "invalid workspace must fail");
    let stderr = String::from_utf8(output.stderr).expect("UTF-8 stderr");
    assert!(
        stderr.contains("has no Cargo.toml"),
        "workspace validation error missing from stderr: {stderr}"
    );
}
