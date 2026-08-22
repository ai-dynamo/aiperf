// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Public `aiperf kube` contract.
//!
//! The hermetic tests drive the real binary without a cluster. The kind tests
//! are `#[ignore]`d so ordinary `cargo test` stays hermetic; CI provisions a
//! cluster and runs them with
//! `cargo test -p aiperf-e2e-tests --test kube_cli_contract -- --ignored`.

mod common;

use std::process::{Command, Stdio};

use common::exec_binary;

/// Every command the native surface owns. Nothing here reaches Python.
const NATIVE_COMMANDS: &[&str] = &[
    "init",
    "validate",
    "profile",
    "sweep",
    "generate",
    "attach",
    "list",
    "logs",
    "results",
    "show",
    "debug",
    "watch",
    "preflight",
    "dashboard",
    "index",
];

fn kube(args: &[&str]) -> (i32, String, String) {
    let output = Command::new(exec_binary())
        .arg("kube")
        .args(args)
        .env("HF_HUB_OFFLINE", "1")
        .stdin(Stdio::null())
        .output()
        .expect("run aiperf kube");
    (
        output.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

#[test]
fn help_lists_every_native_command() {
    let (code, stdout, stderr) = kube(&["--help"]);
    assert_eq!(code, 0, "kube --help failed: {stderr}");
    for command in NATIVE_COMMANDS {
        assert!(
            stdout.contains(command),
            "kube help omits {command}: {stdout}"
        );
    }
}

#[test]
fn no_command_delegates_to_python() {
    let (_, stdout, stderr) = kube(&["--help"]);
    let combined = format!("{stdout}{stderr}").to_lowercase();
    for marker in ["python -m aiperf", "aiperf.entrypoint", "aiperf.cli"] {
        assert!(
            !combined.contains(marker),
            "native kube help references the Python distribution: {marker}"
        );
    }
}

#[test]
fn unknown_commands_fail_closed() {
    let (code, _, stderr) = kube(&["teleport"]);
    assert_ne!(code, 0, "an unknown kube command must fail");
    assert!(
        stderr.contains("unknown native Kubernetes command"),
        "unexpected failure text: {stderr}"
    );
}

#[test]
fn envelope_commands_require_an_envelope() {
    let (code, _, stderr) = kube(&["validate"]);
    assert_ne!(code, 0, "validate without an envelope must fail");
    assert!(!stderr.is_empty(), "failures must explain themselves");
}

#[test]
fn commands_without_a_shipped_backend_refuse_before_cluster_access() {
    for command in ["sweep", "index", "dashboard"] {
        let (code, _, stderr) = kube(&[command]);
        assert_ne!(code, 0, "{command} must refuse without a shipped backend");
        assert!(
            stderr.contains("unavailable"),
            "{command} produced an ambiguous refusal: {stderr}"
        );
    }
}

// requires: the workflow-provisioned kind target and KUBECONFIG
#[test]
#[ignore]
fn kind_native_cli_reaches_the_workflow_provisioned_cluster() {
    let (code, _, stderr) = kube(&["preflight"]);
    assert_eq!(
        code, 0,
        "preflight failed against the live cluster: {stderr}"
    );
    let (code, _, stderr) = kube(&["list", "--namespace", "aiperf-system"]);
    assert_eq!(code, 0, "list failed against the live cluster: {stderr}");
}
