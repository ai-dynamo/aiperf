// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![cfg(not(feature = "dynamo-offline"))]

use std::process::Command;

#[test]
fn default_build_hides_offline_cli_options() {
    let output = Command::new(env!("CARGO_BIN_EXE_aiperf"))
        .args(["--offline", "--requests", "1"])
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("unexpected argument '--offline'"),
        "stderr: {stderr}"
    );
}
