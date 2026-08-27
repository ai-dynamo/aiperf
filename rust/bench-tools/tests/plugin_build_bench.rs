// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Subprocess contract for the plugin build measurement command.

const COMMIT: &str = "0123456789abcdef0123456789abcdef01234567";
const DIGEST: &str = "blake3:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

#[test]
fn identity_option_is_validated_before_the_child_executes() {
    let directory = tempfile::tempdir().expect("temporary fixture directory");
    let artifact = directory.path().join("artifact");
    std::fs::write(&artifact, b"artifact bytes").expect("artifact fixture is written");
    let sentinel = directory.path().join("child-ran");
    let command = env!("CARGO_BIN_EXE_plugin_build_bench");

    let invalid = std::process::Command::new(command)
        .args([
            "--scenario",
            "build-default",
            "--pair-id",
            "pair-00",
            "--variant",
            "static",
            "--target-dir",
            directory.path().to_str().expect("temporary path is UTF-8"),
            "--artifact",
            artifact.to_str().expect("temporary path is UTF-8"),
            "--commit",
            COMMIT,
            "--experiment-identity-digest",
            DIGEST,
            "--unexpected",
            "value",
            "--",
            "/usr/bin/touch",
            sentinel.to_str().expect("temporary path is UTF-8"),
        ])
        .output()
        .expect("invalid build invocation executes");
    assert!(!invalid.status.success());
    assert!(
        !sentinel.exists(),
        "invalid options must reject before child execution"
    );

    let valid = std::process::Command::new(command)
        .args([
            "--scenario",
            "build-default",
            "--pair-id",
            "pair-00",
            "--variant",
            "static",
            "--target-dir",
            directory.path().to_str().expect("temporary path is UTF-8"),
            "--artifact",
            artifact.to_str().expect("temporary path is UTF-8"),
            "--commit",
            COMMIT,
            "--experiment-identity-digest",
            DIGEST,
            "--",
            "/usr/bin/true",
        ])
        .output()
        .expect("valid build invocation executes");
    assert!(
        valid.status.success(),
        "{}",
        String::from_utf8_lossy(&valid.stderr)
    );
    let output: serde_json::Value =
        serde_json::from_slice(&valid.stdout).expect("measurement is canonical JSON");
    assert_eq!(output["sample"]["experiment_identity_digest"], DIGEST);
}
