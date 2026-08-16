// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Black-box Docker execution coverage for standard Harbor benchmark tasks.

mod common;

use std::{fs, process::Command};

use common::exec_binary;

const IMAGE_DIGEST: &str =
    "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

#[test]
#[ignore = "requires a Docker daemon and pulls alpine:3.20"]
fn standard_task_execution_preserves_image_workdir_without_a_cli_override() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = temporary.path().join("image-workdir");
    fs::create_dir_all(task_root.join("environment")).unwrap();
    fs::create_dir_all(task_root.join("tests")).unwrap();
    fs::write(
        task_root.join("task.toml"),
        "schema_version = \"1.0\"\n[task]\nname = \"example/image-workdir\"\n",
    )
    .unwrap();
    fs::write(task_root.join("instruction.md"), "Verify image defaults.\n").unwrap();
    fs::write(
        task_root.join("environment/Dockerfile"),
        "FROM alpine:3.20\nRUN mkdir -p /work /logs/verifier && printf image > /work/preloaded.txt\nWORKDIR /image-work\n",
    )
    .unwrap();
    fs::write(
        task_root.join("tests/test.sh"),
        "test \"$(cat /image-work/pwd.txt)\" = /image-work\ntest \"$(cat /work/preloaded.txt)\" = image\nprintf '{\"reward\":1.0}' > /logs/verifier/reward.json\n",
    )
    .unwrap();

    let output = Command::new(exec_binary())
        .args([
            "eval",
            "--task",
            task_root.to_string_lossy().as_ref(),
            "--image",
            IMAGE_DIGEST,
            "--agent-command",
            "pwd > pwd.txt",
        ])
        .output()
        .expect("start native aiperf eval");

    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let summary: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(summary["task"], "example/image-workdir");
    assert_eq!(summary["reward"]["reward"], 1.0);
    assert!(summary.get("attempt").is_none());
    assert!(summary.get("lineage").is_none());
}

#[test]
#[ignore = "requires a Docker daemon and pulls alpine:3.20"]
fn standard_task_exposes_only_declared_host_secrets_to_each_active_phase() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = temporary.path().join("declared-secrets");
    fs::create_dir_all(task_root.join("environment")).unwrap();
    fs::create_dir_all(task_root.join("tests")).unwrap();
    fs::write(
        task_root.join("task.toml"),
        "schema_version = \"1.0\"\n[task]\nname = \"example/declared-secrets\"\n[environment]\nworkdir = \"/work\"\n[environment.env]\nBASE_TOKEN = \"${BENCHMARK_BASE_TOKEN}\"\n[agent.env]\nAGENT_TOKEN = \"${BENCHMARK_AGENT_TOKEN}\"\n[verifier.env]\nVERIFIER_TOKEN = \"${BENCHMARK_VERIFIER_TOKEN}\"\n",
    )
    .unwrap();
    fs::write(task_root.join("instruction.md"), "Use scoped secrets.\n").unwrap();
    fs::write(
        task_root.join("environment/Dockerfile"),
        "FROM alpine:3.20\nRUN mkdir -p /work /logs/verifier\n",
    )
    .unwrap();
    fs::write(
        task_root.join("tests/test.sh"),
        "test \"$BASE_TOKEN\" = base-token\ntest \"$VERIFIER_TOKEN\" = verifier-token\ntest -z \"${AGENT_TOKEN+x}\"\ntest -z \"${BENCHMARK_UNDECLARED_TOKEN+x}\"\ntest -f /work/agent-ran\nprintf '{\"reward\":1.0}' > /logs/verifier/reward.json\n",
    )
    .unwrap();

    let output = Command::new(exec_binary())
        .env("BENCHMARK_BASE_TOKEN", "base-token")
        .env("BENCHMARK_AGENT_TOKEN", "agent-token")
        .env("BENCHMARK_VERIFIER_TOKEN", "verifier-token")
        .env("BENCHMARK_UNDECLARED_TOKEN", "must-not-reach-docker")
        .args([
            "eval",
            "--task",
            task_root.to_string_lossy().as_ref(),
            "--image",
            IMAGE_DIGEST,
            "--agent-command",
            "test \"$BASE_TOKEN\" = base-token && test \"$AGENT_TOKEN\" = agent-token && test -z \"${VERIFIER_TOKEN+x}\" && test -z \"${BENCHMARK_UNDECLARED_TOKEN+x}\" && touch agent-ran",
        ])
        .output()
        .expect("start native aiperf eval");

    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    let stderr = String::from_utf8(output.stderr).unwrap();
    for value in [
        "base-token",
        "agent-token",
        "verifier-token",
        "must-not-reach-docker",
    ] {
        assert!(!stdout.contains(value), "summary leaked {value}");
        assert!(!stderr.contains(value), "diagnostic leaked {value}");
    }
}

#[test]
#[ignore = "requires a Docker daemon and pulls alpine:3.20"]
fn missing_declared_secret_stops_before_the_agent_without_leaking_host_values() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = temporary.path().join("missing-secret");
    fs::create_dir_all(task_root.join("environment")).unwrap();
    fs::create_dir_all(task_root.join("tests")).unwrap();
    fs::write(
        task_root.join("task.toml"),
        "schema_version = \"1.0\"\n[task]\nname = \"example/missing-secret\"\n[agent.env]\nREQUIRED_TOKEN = \"${BENCHMARK_MISSING_TOKEN}\"\n",
    )
    .unwrap();
    fs::write(task_root.join("instruction.md"), "Require a host secret.\n").unwrap();
    fs::write(
        task_root.join("environment/Dockerfile"),
        "FROM alpine:3.20\nRUN mkdir -p /logs/verifier\n",
    )
    .unwrap();
    fs::write(
        task_root.join("tests/test.sh"),
        "printf '{\"reward\":1.0}' > /logs/verifier/reward.json\n",
    )
    .unwrap();

    let output = Command::new(exec_binary())
        .env_remove("BENCHMARK_MISSING_TOKEN")
        .env("BENCHMARK_UNRELATED_SECRET", "host-value-must-not-leak")
        .args([
            "eval",
            "--task",
            task_root.to_string_lossy().as_ref(),
            "--image",
            IMAGE_DIGEST,
            "--agent-command",
            "touch agent-ran",
        ])
        .output()
        .expect("start native aiperf eval");

    assert!(
        !output.status.success(),
        "missing secret must stop the agent"
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(
        stderr.contains("BENCHMARK_MISSING_TOKEN"),
        "stderr: {stderr}"
    );
    assert!(!stdout.contains("host-value-must-not-leak"));
    assert!(!stderr.contains("host-value-must-not-leak"));
    assert!(
        stdout.trim().is_empty(),
        "failed runs must not emit a summary"
    );
}
