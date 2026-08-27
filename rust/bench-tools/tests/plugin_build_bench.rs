// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Refusal contract for the superseded standalone build measurement command.

const COMMIT: &str = "0123456789abcdef0123456789abcdef01234567";
const DIGEST: &str = "blake3:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

#[test]
fn standalone_build_measurement_is_non_authoritative_and_never_executes_child() {
    let directory = tempfile::tempdir().expect("temporary fixture directory");
    let artifact = directory.path().join("artifact");
    std::fs::write(&artifact, b"artifact bytes").expect("artifact fixture is written");
    let sentinel = directory.path().join("child-ran");
    let command = env!("CARGO_BIN_EXE_plugin_build_bench");

    let output = std::process::Command::new(command)
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
            "/usr/bin/touch",
            sentinel.to_str().expect("temporary path is UTF-8"),
        ])
        .output()
        .expect("standalone refusal executes");
    assert!(!output.status.success());
    assert!(
        !sentinel.exists(),
        "standalone measurements must reject before child execution"
    );
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("same-process paired build controller"),
        "unexpected refusal: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}
