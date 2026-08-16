// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Black-box Docker execution coverage for standard Harbor benchmark tasks.

mod common;

use std::{
    collections::BTreeSet,
    fs,
    io::{Read, Write},
    net::TcpListener,
    process::{Command, Output, Stdio},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, TryRecvError},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use aiperf_runtime::eval::{
    DockerProcessSandbox, HarborImporter, HarborSandboxRecipe, HarborSource, NativeSourceAcquirer,
};
use common::exec_binary;

const IMAGE_DIGEST: &str =
    "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

static DOCKER_E2E_LOCK: Mutex<()> = Mutex::new(());

#[test]
#[ignore = "requires a Docker daemon and pulls alpine:3.20"]
fn imported_compose_multi_step_snapshot_survives_origin_mutation_and_removal() {
    use std::os::unix::fs::PermissionsExt;

    let _docker_lock = DOCKER_E2E_LOCK.lock().unwrap();
    let temporary = tempfile::tempdir().unwrap();
    let task_root = temporary.path().join("owned-source-snapshot");
    for directory in [
        "environment/context-empty",
        "tests/empty",
        "steps/one",
        "steps/two/tests/empty",
    ] {
        fs::create_dir_all(task_root.join(directory)).unwrap();
    }
    fs::write(
        task_root.join("task.toml"),
        r#"schema_version = "1.0"
multi_step_reward_strategy = "mean"
[task]
name = "example/owned-source-snapshot"
[environment]
workdir = "/work"
[[steps]]
name = "one"
[[steps]]
name = "two"
[steps.verifier]
environment_mode = "separate"
"#,
    )
    .unwrap();
    fs::write(task_root.join("instruction.md"), "Unused instruction.\n").unwrap();
    fs::write(
        task_root.join("environment/Dockerfile"),
        r#"FROM alpine:3.20
COPY context.txt /snapshot/context.txt
COPY context-empty /snapshot/context-empty
COPY helper.sh /snapshot/helper.sh
RUN test "$(cat /snapshot/context.txt)" = original-context && test -d /snapshot/context-empty && test -x /snapshot/helper.sh && mkdir -p /work /logs/verifier && chmod 0777 /logs/verifier
"#,
    )
    .unwrap();
    fs::write(
        task_root.join("environment/docker-compose.yaml"),
        "services:\n  api:\n    image: alpine:3.20\n    command: [\"sleep\", \"infinity\"]\n",
    )
    .unwrap();
    fs::write(
        task_root.join("environment/context.txt"),
        "original-context\n",
    )
    .unwrap();
    fs::write(
        task_root.join("environment/helper.sh"),
        "#!/bin/sh\nexit 0\n",
    )
    .unwrap();
    fs::write(
        task_root.join("tests/test.sh"),
        "test \"$(cat /snapshot/context.txt)\" = original-context\ntest -d /snapshot/context-empty\ntest -x /snapshot/helper.sh\ntest -d /tests/empty\ntest -x /tests/helper.sh\ntest \"$(cat /tests/helper.sh)\" = original-root-helper\nprintf '{\"reward\":1.0}' > /logs/verifier/reward.json\n",
    )
    .unwrap();
    fs::write(task_root.join("tests/helper.sh"), "original-root-helper\n").unwrap();
    fs::write(
        task_root.join("steps/one/instruction.md"),
        "Run step one.\n",
    )
    .unwrap();
    fs::write(
        task_root.join("steps/two/instruction.md"),
        "Run step two.\n",
    )
    .unwrap();
    fs::write(
        task_root.join("steps/two/tests/test.sh"),
        "test \"$(cat /snapshot/context.txt)\" = original-context\ntest -d /snapshot/context-empty\ntest -x /snapshot/helper.sh\ntest -d /tests/empty\ntest -x /tests/helper.sh\ntest \"$(cat /tests/helper.sh)\" = original-step-helper\nprintf '{\"reward\":1.0}' > /logs/verifier/reward.json\n",
    )
    .unwrap();
    fs::write(
        task_root.join("steps/two/tests/helper.sh"),
        "original-step-helper\n",
    )
    .unwrap();
    for helper in [
        "environment/helper.sh",
        "tests/helper.sh",
        "steps/two/tests/helper.sh",
    ] {
        fs::set_permissions(task_root.join(helper), fs::Permissions::from_mode(0o755)).unwrap();
    }

    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    fs::write(task_root.join("environment/context.txt"), "mutated\n").unwrap();
    fs::write(task_root.join("tests/helper.sh"), "mutated\n").unwrap();
    fs::remove_dir_all(&task_root).unwrap();

    let recipe = HarborSandboxRecipe::for_standard_task(IMAGE_DIGEST, None).unwrap();
    let result = DockerProcessSandbox::new()
        .execute_multi_step(&recipe, &imported.package, &["true".to_owned()])
        .unwrap();

    assert_eq!(result.steps.len(), 2);
    assert_eq!(result.steps[0].reward.metrics["reward"], 1.0);
    assert_eq!(result.steps[1].reward.metrics["reward"], 1.0);
}

#[test]
#[ignore = "requires a Docker daemon and pulls alpine:3.20"]
fn standard_task_execution_preserves_image_workdir_without_a_cli_override() {
    let _docker_lock = DOCKER_E2E_LOCK.lock().unwrap();
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
    assert_summary(&summary, "example/image-workdir");
}

#[test]
#[ignore = "requires a Docker daemon and pulls alpine:3.20"]
fn standard_task_exposes_only_declared_host_secrets_to_each_active_phase() {
    let _docker_lock = DOCKER_E2E_LOCK.lock().unwrap();
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
    let _docker_lock = DOCKER_E2E_LOCK.lock().unwrap();
    let recorder = RequestRecorder::new();
    let temporary = tempfile::tempdir().unwrap();
    let task_root = temporary.path().join("missing-secret");
    fs::create_dir_all(task_root.join("environment")).unwrap();
    fs::create_dir_all(task_root.join("tests")).unwrap();
    fs::write(
        task_root.join("task.toml"),
        "schema_version = \"1.0\"\n[task]\nname = \"example/missing-secret\"\n[agent.env]\nREQUIRED_TOKEN = \"${BENCHMARK_MISSING_TOKEN}\"\n[verifier]\nenvironment_mode = \"separate\"\n",
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
        &format!(
            "wget -qO /dev/null {}/verifier\nprintf '{{\"reward\":1.0}}' > /logs/verifier/reward.json\n",
            recorder.url()
        ),
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
            &format!("wget -qO /dev/null {}/agent", recorder.url()),
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
    recorder.assert_no_request("missing secret must prevent both agent and verifier startup");
}

#[test]
#[ignore = "requires a Docker daemon and pulls alpine:3.20"]
fn standard_task_enforces_phase_network_user_environment_artifact_and_verifier_isolation() {
    let _docker_lock = DOCKER_E2E_LOCK.lock().unwrap();
    let recorder = RequestRecorder::new();
    let temporary = tempfile::tempdir().unwrap();
    let task_root = standard_task(
        &temporary,
        "network-user-env-artifacts",
        "artifacts = [{ source = \"/workspace/output\", destination = \"published\", exclude = [\"*.tmp\"] }]\n[environment]\nworkdir = \"/workspace\"\nuser = \"bench\"\nnetwork = \"no-network\"\n[environment.env]\nSCOPE = \"baseline\"\n[agent]\nuser = \"bench\"\nnetwork = \"no-network\"\n[agent.env]\nSCOPE = \"agent\"\n[verifier]\nenvironment_mode = \"separate\"\nuser = \"root\"\nnetwork = \"public\"\n[verifier.env]\nSCOPE = \"verifier\"\n[verifier.environment]\nnetwork = \"public\"\n",
        "FROM alpine:3.20\nRUN adduser -D bench && mkdir -p /workspace /logs/verifier && chown -R bench:bench /workspace && chmod 0777 /logs/verifier\nWORKDIR /image-default\n",
        &format!(
            "test \"$SCOPE\" = verifier\ntest \"$(id -un)\" = root\nwget -qO /dev/null {}/verifier\ntest \"$(cat /workspace/published/result.txt)\" = exact-bytes\ntest ! -e /workspace/published/drop.tmp\ntest \"$(cat /workspace/published/agent-scope.txt)\" = agent\ntest \"$(cat /workspace/published/agent-user.txt)\" = bench\ntest \"$(cat /workspace/published/agent-pwd.txt)\" = /workspace\ntest ! -e /workspace/agent-only\ntest -f /tests/test.sh\nprintf '{{\"reward\":1.0}}' > /logs/verifier/reward.json\n",
            recorder.url()
        ),
    );
    let output = run_eval(
        &task_root,
        &format!(
            "test ! -e /tests/test.sh && ! wget -qO /dev/null {}/agent && test \"$SCOPE\" = agent && test \"$(id -un)\" = bench && test \"$(pwd)\" = /workspace && mkdir -p output && printf exact-bytes > output/result.txt && printf excluded > output/drop.tmp && printf \"$SCOPE\" > output/agent-scope.txt && id -un > output/agent-user.txt && pwd > output/agent-pwd.txt && printf private > agent-only",
            recorder.url()
        ),
    );

    assert_success(&output);
    assert_eq!(
        recorder.next_request("the public verifier phase must reach the test server"),
        "/verifier"
    );
    recorder.assert_no_request("the no-network agent phase must not reach the test server");
    let summary: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_summary(&summary, "example/network-user-env-artifacts");
    let artifacts = summary["artifacts"].as_array().unwrap();
    assert_eq!(artifacts.len(), 4);
    let paths = artifacts
        .iter()
        .map(|artifact| {
            let pair = artifact.as_array().expect("artifact must be a pair");
            assert_eq!(pair.len(), 2);
            assert!(pair[0].as_str().is_some());
            let digest = pair[1].as_str().expect("artifact digest must be a string");
            assert!(
                digest.starts_with("blake3:")
                    && digest.len() == "blake3:".len() + 64
                    && digest["blake3:".len()..]
                        .chars()
                        .all(|character| character.is_ascii_hexdigit()),
                "unexpected artifact digest {digest}"
            );
            pair[0].as_str().unwrap().to_owned()
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(
        paths,
        BTreeSet::from([
            "published/agent-pwd.txt".to_owned(),
            "published/agent-scope.txt".to_owned(),
            "published/agent-user.txt".to_owned(),
            "published/result.txt".to_owned(),
        ])
    );
    let result_digest = artifacts
        .iter()
        .find_map(|artifact| {
            let pair = artifact.as_array()?;
            (pair.first()?.as_str()? == "published/result.txt")
                .then(|| pair.get(1)?.as_str())
                .flatten()
        })
        .expect("result artifact digest");
    assert_eq!(
        result_digest,
        format!("blake3:{}", blake3::hash(b"exact-bytes").to_hex())
    );
}

#[test]
#[ignore = "requires a Docker daemon and pulls alpine:3.20"]
fn standard_task_default_public_agent_connects_to_the_controlled_host() {
    let _docker_lock = DOCKER_E2E_LOCK.lock().unwrap();
    let recorder = RequestRecorder::new();
    let temporary = tempfile::tempdir().unwrap();
    let task_root = standard_task(
        &temporary,
        "public-agent",
        "[environment]\nworkdir = \"/work\"\nnetwork = \"public\"\n",
        "FROM alpine:3.20\nRUN mkdir -p /work /logs/verifier\n",
        "test -f /work/agent-ran\nprintf '{\"reward\":1.0}' > /logs/verifier/reward.json\n",
    );
    let output = run_eval(
        &task_root,
        &format!(
            "wget -qO /dev/null {}/agent && touch agent-ran",
            recorder.url()
        ),
    );

    assert_success(&output);
    assert_eq!(
        recorder.next_request("the default public agent must reach the controlled host"),
        "/agent"
    );
}

#[test]
#[ignore = "requires a Docker daemon and pulls alpine:3.20"]
fn standard_task_retries_readiness_before_starting_the_agent() {
    let _docker_lock = DOCKER_E2E_LOCK.lock().unwrap();
    let temporary = tempfile::tempdir().unwrap();
    let task_root = standard_task(
        &temporary,
        "readiness-retry",
        "[environment]\nworkdir = \"/work\"\n[environment.healthcheck]\ncommand = [\"/bin/sh\", \"-c\", \"if test -f /tmp/ready; then exit 0; fi; touch /tmp/ready; exit 1\"]\nstart_interval_sec = 0.01\ninterval_sec = 0.01\ntimeout_sec = 1\nretries = 2\n",
        "FROM alpine:3.20\nRUN mkdir -p /work /logs/verifier\n",
        "test -f /work/agent-ran\ntest -f /tmp/ready\nprintf '{\"reward\":1.0}' > /logs/verifier/reward.json\n",
    );
    let output = run_eval(&task_root, "test -f /tmp/ready && touch agent-ran");

    assert_success(&output);
    let summary: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_summary(&summary, "example/readiness-retry");
}

#[test]
#[ignore = "requires a Docker daemon and pulls alpine:3.20"]
fn exhausted_readiness_prevents_agent_and_verifier_network_sentinels() {
    let _docker_lock = DOCKER_E2E_LOCK.lock().unwrap();
    let recorder = RequestRecorder::new();
    let temporary = tempfile::tempdir().unwrap();
    let task_root = standard_task(
        &temporary,
        "readiness-exhaustion",
        "[environment.healthcheck]\ncommand = [\"false\"]\ntimeout_sec = 1\nretries = 1\n",
        "FROM alpine:3.20\nRUN mkdir -p /logs/verifier\n",
        &format!("wget -qO /dev/null {}/verifier\n", recorder.url()),
    );
    let output = run_eval(
        &task_root,
        &format!("wget -qO /dev/null {}/agent", recorder.url()),
    );

    assert_failure_without_summary(&output, "unhealthy");
    recorder.assert_no_request("readiness exhaustion must prevent agent and verifier startup");
}

#[test]
#[ignore = "requires a Docker daemon and pulls alpine:3.20"]
fn agent_timeout_cleans_up_the_docker_task_container_before_verifier_startup() {
    let _docker_lock = DOCKER_E2E_LOCK.lock().unwrap();
    let recorder = RequestRecorder::new();
    let temporary = tempfile::tempdir().unwrap();
    let task_root = standard_task(
        &temporary,
        "agent-timeout",
        "[agent]\ntimeout_sec = 0.2\n[verifier]\ntimeout_sec = 2\n",
        "FROM alpine:3.20\nRUN mkdir -p /logs/verifier\n",
        &format!("wget -qO /dev/null {}/verifier\n", recorder.url()),
    );
    let before = task_container_names();
    let output = run_eval(&task_root, "sleep 300 & sleep 2");

    assert_failure_without_summary(&output, "timed out");
    assert_eq!(
        task_container_names(),
        before,
        "timeout leaked a task container"
    );
    recorder.assert_no_request("agent timeout must prevent verifier startup");
}

#[test]
#[ignore = "requires a Docker daemon and pulls alpine:3.20"]
fn separate_verifier_timeout_removes_every_task_container_before_returning() {
    let _docker_lock = DOCKER_E2E_LOCK.lock().unwrap();
    let temporary = tempfile::tempdir().unwrap();
    let task_root = standard_task(
        &temporary,
        "separate-verifier-timeout",
        "[agent]\ntimeout_sec = 2\n[verifier]\ntimeout_sec = 0.2\nenvironment_mode = \"separate\"\n",
        "FROM alpine:3.20\nRUN mkdir -p /logs/verifier\n",
        "sleep 300 & sleep 2\nprintf '{\"reward\":1.0}' > /logs/verifier/reward.json\n",
    );
    let before = task_container_names();
    let output = run_eval(&task_root, "true");

    assert_failure_without_summary(&output, "timed out");
    assert_eq!(
        task_container_names(),
        before,
        "separate verifier timeout leaked a task container"
    );
}

#[test]
#[ignore = "requires a Docker daemon and pulls alpine:3.20"]
fn multi_step_mean_execution_preserves_boundaries_and_reports_every_step() {
    let _docker_lock = DOCKER_E2E_LOCK.lock().unwrap();
    let temporary = tempfile::tempdir().unwrap();
    let task_root = multi_step_boundary_task(&temporary, "mean");
    let output = run_eval(
        &task_root,
        r#"case "$AIPERF_EVAL_INSTRUCTION" in
  *Prepare*)
    test "$BASE_SCOPE" = root &&
    test "$AGENT_SCOPE" = inherited &&
    test "$(id -un)" = bench &&
    test "$(pwd)" = /workspace &&
    test ! -e /tests/test.sh &&
    printf persistent > shared.txt &&
    printf first-snapshot > result.txt &&
    printf agent-private > private.txt
    ;;
  *Finish*)
    test "$BASE_SCOPE" = root &&
    test "$AGENT_SCOPE" = overridden &&
    test "$(id -un)" = root &&
    test "$(pwd)" = /workspace &&
    test "$(cat shared.txt)" = persistent &&
    test ! -e /tests/test.sh &&
    test ! -e /tests/stale-from-prepare &&
    sleep 1.2 &&
    printf second-snapshot > result.txt
    ;;
  *) exit 91 ;;
esac"#,
    );

    assert_success(&output);
    let summary: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(summary["task"], "example/multi-step-mean");
    assert_eq!(
        summary["reward"],
        serde_json::json!({
            "only_first": 0.5,
            "only_second": 0.25,
            "quality": 0.5,
        })
    );
    assert_eq!(
        summary["steps"],
        serde_json::json!([
            {
                "name": "prepare",
                "artifacts": [["result.txt", artifact_digest(b"first-snapshot")]],
                "reward": {"only_first": 1.0, "quality": 0.25},
            },
            {
                "name": "finish",
                "artifacts": [["result.txt", artifact_digest(b"second-snapshot")]],
                "reward": {"only_second": 0.5, "quality": 0.75},
            },
        ])
    );
    assert_eq!(
        summary["artifacts"],
        serde_json::json!([["result.txt", artifact_digest(b"second-snapshot")]])
    );
}

#[test]
#[ignore = "requires a Docker daemon and pulls alpine:3.20"]
fn multi_step_final_execution_reports_the_final_reward_unchanged() {
    let _docker_lock = DOCKER_E2E_LOCK.lock().unwrap();
    let temporary = tempfile::tempdir().unwrap();
    let task_root = multi_step_reward_task(&temporary);
    let output = run_eval(
        &task_root,
        r#"case "$AIPERF_EVAL_INSTRUCTION" in
  *First*) printf first > current-step ;;
  *Second*) test "$(cat current-step)" = first && printf second > current-step ;;
  *) exit 91 ;;
esac"#,
    );

    assert_success(&output);
    let summary: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(summary["task"], "example/multi-step-final");
    assert_eq!(
        summary["reward"],
        serde_json::json!({"final": 1.0, "score": 0.9})
    );
    assert_eq!(
        summary["steps"][0]["reward"],
        serde_json::json!({"score": 0.2})
    );
    assert_eq!(
        summary["steps"][1]["reward"],
        serde_json::json!({"final": 1.0, "score": 0.9})
    );
}

#[test]
#[ignore = "requires a Docker daemon and pulls alpine:3.20"]
fn first_multi_step_verifier_failure_stops_successors_and_cleans_every_container() {
    let _docker_lock = DOCKER_E2E_LOCK.lock().unwrap();
    let recorder = RequestRecorder::new();
    let temporary = tempfile::tempdir().unwrap();
    let task_root = multi_step_failure_task(&temporary, recorder.url());
    let before = task_container_names();
    let output = run_eval(
        &task_root,
        &format!(
            r#"case "$AIPERF_EVAL_INSTRUCTION" in
  *First*) wget -qO /dev/null {}/agent-one ;;
  *Second*) wget -qO /dev/null {}/agent-two ;;
  *) exit 91 ;;
esac"#,
            recorder.url(),
            recorder.url(),
        ),
    );

    assert_failure_without_summary(&output, "planned Docker phase");
    assert_eq!(
        task_container_names(),
        before,
        "failed multi-step execution leaked a task container"
    );
    assert_eq!(
        recorder.next_request("the first agent must run"),
        "/agent-one"
    );
    assert_eq!(
        recorder.next_request("the first verifier must run"),
        "/verifier-one"
    );
    recorder.assert_no_request("a failed verifier must prevent every successor phase");
}

fn multi_step_boundary_task(
    temporary: &tempfile::TempDir,
    reward_strategy: &str,
) -> std::path::PathBuf {
    let task_root = temporary
        .path()
        .join(format!("multi-step-{reward_strategy}"));
    fs::create_dir_all(task_root.join("environment")).unwrap();
    fs::create_dir_all(task_root.join("tests")).unwrap();
    fs::create_dir_all(task_root.join("steps/prepare")).unwrap();
    fs::create_dir_all(task_root.join("steps/finish/tests")).unwrap();
    fs::write(
        task_root.join("task.toml"),
        format!(
            r#"schema_version = "1.0"
multi_step_reward_strategy = "{reward_strategy}"
artifacts = ["/workspace/result.txt"]

[task]
name = "example/multi-step-{reward_strategy}"

[environment]
workdir = "/workspace"
user = "bench"
network = "public"

[environment.env]
BASE_SCOPE = "root"

[agent]
timeout_sec = 1

[agent.env]
AGENT_SCOPE = "inherited"

[verifier]
timeout_sec = 1
user = "root"

[verifier.env]
VERIFIER_SCOPE = "inherited"

[[steps]]
name = "prepare"

[[steps]]
name = "finish"

[steps.agent]
timeout_sec = 3
user = "root"

[steps.agent.env]
AGENT_SCOPE = "overridden"

[steps.verifier]
timeout_sec = 3
environment_mode = "separate"
user = "bench"

[steps.verifier.env]
VERIFIER_SCOPE = "overridden"

[steps.verifier.environment]
workdir = "/verify"
"#,
        ),
    )
    .unwrap();
    fs::write(
        task_root.join("instruction.md"),
        "Unused root instruction.\n",
    )
    .unwrap();
    fs::write(
        task_root.join("environment/Dockerfile"),
        "FROM alpine:3.20\nRUN adduser -D bench && mkdir -p /workspace /verify /logs/verifier && chown -R bench:bench /workspace /verify && chmod 0777 /logs/verifier\nWORKDIR /image-default\n",
    )
    .unwrap();
    fs::write(
        task_root.join("tests/test.sh"),
        "test \"$BASE_SCOPE\" = root\ntest \"$VERIFIER_SCOPE\" = inherited\ntest \"$(id -un)\" = root\ntest \"$(pwd)\" = /workspace\ntest \"$(cat result.txt)\" = first-snapshot\ntest -f /tests/root-only\ntest ! -e /tests/finish-only\ntouch /tests/stale-from-prepare\nprintf '{\"quality\":0.25,\"only_first\":1.0}' > /logs/verifier/reward.json\n",
    )
    .unwrap();
    fs::write(task_root.join("tests/root-only"), "prepare\n").unwrap();
    fs::write(
        task_root.join("steps/prepare/instruction.md"),
        "Prepare the persistent workspace.\n",
    )
    .unwrap();
    fs::write(
        task_root.join("steps/finish/instruction.md"),
        "Finish the persistent workspace.\n",
    )
    .unwrap();
    fs::write(
        task_root.join("steps/finish/tests/test.sh"),
        "test \"$BASE_SCOPE\" = root\ntest \"$VERIFIER_SCOPE\" = overridden\ntest \"$(id -un)\" = bench\ntest \"$(pwd)\" = /verify\ntest -f /tests/finish-only\ntest ! -e /tests/root-only\ntest ! -e /tests/stale-from-prepare\ntest ! -e /workspace/private.txt\ntest \"$(cat result.txt)\" = second-snapshot\nsleep 1.2\nprintf '{\"quality\":0.75,\"only_second\":0.5}' > /logs/verifier/reward.json\n",
    )
    .unwrap();
    fs::write(task_root.join("steps/finish/tests/finish-only"), "finish\n").unwrap();
    task_root
}

fn multi_step_reward_task(temporary: &tempfile::TempDir) -> std::path::PathBuf {
    let task_root = temporary.path().join("multi-step-final");
    fs::create_dir_all(task_root.join("environment")).unwrap();
    fs::create_dir_all(task_root.join("tests")).unwrap();
    fs::create_dir_all(task_root.join("steps/first")).unwrap();
    fs::create_dir_all(task_root.join("steps/second")).unwrap();
    fs::write(
        task_root.join("task.toml"),
        r#"schema_version = "1.0"
multi_step_reward_strategy = "final"
[task]
name = "example/multi-step-final"
[environment]
workdir = "/work"
[[steps]]
name = "first"
[[steps]]
name = "second"
"#,
    )
    .unwrap();
    fs::write(
        task_root.join("instruction.md"),
        "Unused root instruction.\n",
    )
    .unwrap();
    fs::write(
        task_root.join("environment/Dockerfile"),
        "FROM alpine:3.20\nRUN mkdir -p /work /logs/verifier\n",
    )
    .unwrap();
    fs::write(
        task_root.join("tests/test.sh"),
        "case \"$(cat current-step)\" in\n  first) printf '{\"score\":0.2}' > /logs/verifier/reward.json ;;\n  second) printf '{\"score\":0.9,\"final\":1.0}' > /logs/verifier/reward.json ;;\n  *) exit 92 ;;\nesac\n",
    )
    .unwrap();
    fs::write(
        task_root.join("steps/first/instruction.md"),
        "First reward step.\n",
    )
    .unwrap();
    fs::write(
        task_root.join("steps/second/instruction.md"),
        "Second reward step.\n",
    )
    .unwrap();
    task_root
}

fn multi_step_failure_task(
    temporary: &tempfile::TempDir,
    recorder_url: &str,
) -> std::path::PathBuf {
    let task_root = temporary.path().join("multi-step-failure");
    fs::create_dir_all(task_root.join("environment")).unwrap();
    fs::create_dir_all(task_root.join("tests")).unwrap();
    fs::create_dir_all(task_root.join("steps/first")).unwrap();
    fs::create_dir_all(task_root.join("steps/second")).unwrap();
    fs::write(
        task_root.join("task.toml"),
        r#"schema_version = "1.0"
multi_step_reward_strategy = "mean"
[task]
name = "example/multi-step-failure"
[environment]
network = "public"
[verifier]
environment_mode = "separate"
[[steps]]
name = "first"
[[steps]]
name = "second"
"#,
    )
    .unwrap();
    fs::write(
        task_root.join("instruction.md"),
        "Unused root instruction.\n",
    )
    .unwrap();
    fs::write(
        task_root.join("environment/Dockerfile"),
        "FROM alpine:3.20\nRUN mkdir -p /logs/verifier\n",
    )
    .unwrap();
    fs::write(
        task_root.join("tests/test.sh"),
        format!("wget -qO /dev/null {recorder_url}/verifier-one\nexit 17\n"),
    )
    .unwrap();
    fs::write(
        task_root.join("steps/first/instruction.md"),
        "First failing step.\n",
    )
    .unwrap();
    fs::write(
        task_root.join("steps/second/instruction.md"),
        "Second forbidden step.\n",
    )
    .unwrap();
    task_root
}

fn artifact_digest(bytes: &[u8]) -> String {
    format!("blake3:{}", blake3::hash(bytes).to_hex())
}

fn standard_task(
    temporary: &tempfile::TempDir,
    name: &str,
    manifest_suffix: &str,
    dockerfile: &str,
    verifier: &str,
) -> std::path::PathBuf {
    let task_root = temporary.path().join(name);
    fs::create_dir_all(task_root.join("environment")).unwrap();
    fs::create_dir_all(task_root.join("tests")).unwrap();
    fs::write(
        task_root.join("task.toml"),
        format!("schema_version = \"1.0\"\n{manifest_suffix}\n[task]\nname = \"example/{name}\"\n"),
    )
    .unwrap();
    fs::write(
        task_root.join("instruction.md"),
        "Run the benchmark task.\n",
    )
    .unwrap();
    fs::write(task_root.join("environment/Dockerfile"), dockerfile).unwrap();
    fs::write(task_root.join("tests/test.sh"), verifier).unwrap();
    task_root
}

fn run_eval(task_root: &std::path::Path, agent_command: &str) -> Output {
    Command::new(exec_binary())
        .args([
            "eval",
            "--task",
            task_root.to_string_lossy().as_ref(),
            "--image",
            IMAGE_DIGEST,
            "--agent-command",
            agent_command,
        ])
        .output()
        .expect("start native aiperf eval")
}

fn start_eval(task_root: &std::path::Path, agent_command: &str) -> std::process::Child {
    Command::new(exec_binary())
        .args([
            "eval",
            "--task",
            task_root.to_string_lossy().as_ref(),
            "--image",
            IMAGE_DIGEST,
            "--agent-command",
            agent_command,
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("start native aiperf eval")
}

fn compose_standard_task(
    temporary: &tempfile::TempDir,
    name: &str,
    manifest_suffix: &str,
    verifier: &str,
) -> std::path::PathBuf {
    let task_root = standard_task(
        temporary,
        name,
        manifest_suffix,
        "FROM alpine:3.20\nRUN mkdir -p /work /logs/verifier\n",
        verifier,
    );
    fs::write(
        task_root.join("environment/docker-compose.yaml"),
        "services:\n  api:\n    image: alpine:3.20\n    command: [\"sleep\", \"infinity\"]\n",
    )
    .unwrap();
    task_root
}

#[test]
#[ignore = "requires a Docker daemon and pulls alpine:3.20"]
fn compose_sidecar_evaluation_keeps_verifier_isolated_and_cleans_up() {
    let _docker_lock = DOCKER_E2E_LOCK.lock().unwrap();
    let before = compose_resource_ids();
    let temporary = tempfile::tempdir().unwrap();
    let task_root = temporary.path().join("compose-sidecar");
    fs::create_dir_all(task_root.join("environment")).unwrap();
    fs::create_dir_all(task_root.join("tests")).unwrap();
    fs::write(
        task_root.join("task.toml"),
        r#"schema_version = "1.0"
artifacts = [{ source = "/data", destination = "evidence", service = "api" }]
[task]
name = "example/compose-sidecar"
[environment]
workdir = "/work"
user = "bench"
[environment.healthcheck]
command = ["/bin/sh", "-c", "test -d /work && touch /tmp/health-ready"]
retries = 1
[verifier]
environment_mode = "separate"
[[verifier.collect]]
service = "api"
command = ["/bin/sh", "-c", "mkdir -p /data && printf hooked > /data/evidence.txt"]
timeout_sec = 10
"#,
    )
    .unwrap();
    fs::write(
        task_root.join("instruction.md"),
        "Write an agent-only file.\n",
    )
    .unwrap();
    fs::write(
        task_root.join("environment/Dockerfile"),
        "FROM alpine:3.20\nRUN adduser -D -u 1000 bench && mkdir -p /work /logs/verifier && chmod 0777 /logs/verifier\n",
    )
    .unwrap();
    fs::write(
        task_root.join("environment/docker-compose.yaml"),
        "services:\n  api:\n    image: alpine:3.20\n    command: [\"sleep\", \"infinity\"]\n",
    )
    .unwrap();
    fs::write(
        task_root.join("tests/test.sh"),
        "test -r /work/evidence.txt\ntest \"$(cat /work/evidence.txt)\" = hooked\ntest ! -e /work/agent-only.txt\nprintf '{\"reward\":1.0}' > /logs/verifier/reward.json\n",
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
            "/bin/sh -c 'test -f /tmp/health-ready && printf agent > /work/agent-only.txt'",
        ])
        .output()
        .expect("start native aiperf Compose eval");
    assert_success(&output);
    let summary: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_summary(&summary, "example/compose-sidecar");
    assert_eq!(summary["artifacts"][0][0], "evidence/evidence.txt");
    assert_eq!(compose_resource_ids(), before);
}

#[test]
#[ignore = "requires a Docker daemon and pulls alpine:3.20"]
fn compose_sidecar_readiness_dns_and_final_evidence_preserve_verifier_isolation() {
    let _docker_lock = DOCKER_E2E_LOCK.lock().unwrap();
    let before = compose_resource_ids();
    let temporary = tempfile::tempdir().unwrap();
    let task_root = temporary.path().join("compose-boundaries");
    fs::create_dir_all(task_root.join("environment")).unwrap();
    fs::create_dir_all(task_root.join("tests")).unwrap();
    fs::write(
        task_root.join("task.toml"),
        r#"schema_version = "1.0"
artifacts = [{ source = "/data/evidence", destination = "sidecar-evidence", service = "api" }]
[task]
name = "example/compose-boundaries"
[environment]
workdir = "/work"
user = "bench"
[environment.healthcheck]
command = ["/bin/sh", "-c", "test -w /work && touch /work/health-ready"]
retries = 20
start_interval_sec = 0.1
interval_sec = 0.1
timeout_sec = 2
[agent.env]
AGENT_SECRET = "${COMPOSE_AGENT_SECRET}"
[verifier]
environment_mode = "separate"
[[verifier.collect]]
service = "api"
command = ["/bin/sh", "-c", "mkdir -p /data/evidence && printf collected > /data/evidence/result.txt"]
timeout_sec = 10
"#,
    )
    .unwrap();
    fs::write(
        task_root.join("instruction.md"),
        "Prove the Compose service is ready before running the agent.\n",
    )
    .unwrap();
    fs::write(
        task_root.join("environment/Dockerfile"),
        "FROM alpine:3.20\nRUN adduser -D -u 1000 bench && mkdir -p /work /logs/verifier && chmod 0777 /logs/verifier\n",
    )
    .unwrap();
    fs::write(
        task_root.join("environment/docker-compose.yaml"),
        "services:\n  main:\n    depends_on: [api]\n  api:\n    image: alpine:3.20\n    command: [\"sleep\", \"infinity\"]\n",
    )
    .unwrap();
    fs::write(
        task_root.join("tests/test.sh"),
        "test \"$(cat /work/sidecar-evidence/result.txt)\" = collected\ntest ! -e /work/agent-only.txt\ntest ! -e /data/evidence/result.txt\ntest -z \"${AGENT_SECRET+x}\"\n! nslookup api >/dev/null\nprintf '{\"reward\":1.0}' > /logs/verifier/reward.json\n",
    )
    .unwrap();

    let output = Command::new(exec_binary())
        .env("COMPOSE_AGENT_SECRET", "agent-secret-value")
        .args([
            "eval",
            "--task",
            task_root.to_string_lossy().as_ref(),
            "--image",
            IMAGE_DIGEST,
            "--agent-command",
            "/bin/sh -c 'test \"$AGENT_SECRET\" = agent-secret-value && test -f /work/health-ready && nslookup api >/dev/null && printf private > /work/agent-only.txt'",
        ])
        .output()
        .expect("start native aiperf Compose eval");

    assert_success(&output);
    let summary: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_summary(&summary, "example/compose-boundaries");
    assert_eq!(
        summary["artifacts"],
        serde_json::json!([[
            "sidecar-evidence/result.txt",
            artifact_digest(b"collected"),
        ]])
    );
    assert!(!String::from_utf8_lossy(&output.stdout).contains("agent-secret-value"));
    assert!(!String::from_utf8_lossy(&output.stderr).contains("agent-secret-value"));
    assert_eq!(compose_resource_ids(), before);
}

#[test]
#[ignore = "requires a Docker daemon and pulls alpine:3.20"]
fn compose_unhealthy_start_prevents_agent_and_verifier_and_cleans_the_project() {
    let _docker_lock = DOCKER_E2E_LOCK.lock().unwrap();
    let recorder = RequestRecorder::new();
    let temporary = tempfile::tempdir().unwrap();
    let task_root = compose_standard_task(
        &temporary,
        "compose-unhealthy",
        "[environment]\nnetwork = \"public\"\n[environment.healthcheck]\ncommand = [\"false\"]\ntimeout_sec = 1\nretries = 1\n[verifier]\nenvironment_mode = \"separate\"\n",
        &format!("wget -qO /dev/null {}/verifier\n", recorder.url()),
    );
    let before = compose_resource_ids();
    let output = run_eval(
        &task_root,
        &format!("wget -qO /dev/null {}/agent", recorder.url()),
    );

    assert_failure_without_summary(&output, "unhealthy");
    recorder.assert_no_request("unhealthy Compose startup must prevent every phase");
    assert_eq!(
        compose_resource_ids(),
        before,
        "unhealthy Compose startup leaked resources"
    );
}

#[test]
#[ignore = "requires a Docker daemon and pulls alpine:3.20"]
fn compose_terminal_evidence_failures_prevent_the_separate_verifier_and_clean_the_project() {
    let _docker_lock = DOCKER_E2E_LOCK.lock().unwrap();
    let selected = std::env::var("AIPERF_E2E_EVIDENCE_CASE").ok();
    for (name, manifest_suffix, diagnostic) in [
        (
            "compose-hook-nonzero",
            "[verifier]\nenvironment_mode = \"separate\"\n[[verifier.collect]]\nservice = \"api\"\ncommand = [\"/bin/sh\", \"-c\", \"exit 31\"]\ntimeout_sec = 10\n",
            "planned Docker phase",
        ),
        (
            "compose-hook-timeout",
            "[verifier]\nenvironment_mode = \"separate\"\n[[verifier.collect]]\nservice = \"api\"\ncommand = [\"/bin/sh\", \"-c\", \"sleep 300\"]\ntimeout_sec = 0.2\n",
            "timed out",
        ),
        (
            "compose-archive-failure",
            "artifacts = [{ source = \"/missing\", destination = \"evidence\", service = \"api\" }]\n[verifier]\nenvironment_mode = \"separate\"\n",
            "archive",
        ),
    ] {
        if selected.as_deref().is_some_and(|selected| selected != name) {
            continue;
        }
        let recorder = RequestRecorder::new();
        let temporary = tempfile::tempdir().unwrap();
        let task_root = compose_standard_task(
            &temporary,
            name,
            &format!("{manifest_suffix}\n[environment]\nnetwork = \"public\"\n"),
            &format!("wget -qO /dev/null {}/verifier\n", recorder.url()),
        );
        let before_runs = compose_run_labels();
        let started = Instant::now();
        let mut child = start_eval(
            &task_root,
            &format!("wget -qO /dev/null {}/agent && sleep 2", recorder.url()),
        );
        assert_eq!(
            next_request_from_child(
                &recorder,
                &mut child,
                "agent must run before terminal evidence collection",
                Duration::from_secs(60)
            ),
            "/agent"
        );
        let run = asserted_new_compose_run(&before_runs);
        assert!(
            !compose_resource_ids_for_run(&run).is_empty(),
            "the active Compose run must own resources before evidence collection"
        );
        let output = child
            .wait_with_output()
            .expect("wait for native aiperf eval");

        assert_failure_without_summary(&output, diagnostic);
        if name.ends_with("timeout") {
            assert!(
                started.elapsed() < Duration::from_secs(30),
                "{name} exceeded the end-to-end timeout budget: {:?}",
                started.elapsed()
            );
        }
        assert_eq!(
            compose_resource_ids_for_run(&run),
            BTreeSet::new(),
            "{name} leaked a resource from its exact Compose run"
        );
    }
}

#[test]
#[ignore = "requires a Docker daemon and pulls alpine:3.20"]
fn compose_agent_and_separate_verifier_failures_cleanup_the_project() {
    let _docker_lock = DOCKER_E2E_LOCK.lock().unwrap();
    let temporary = tempfile::tempdir().unwrap();
    let recorder = RequestRecorder::new();
    let agent_timeout = compose_standard_task(
        &temporary,
        "compose-agent-timeout",
        "[environment]\nnetwork = \"public\"\n[agent]\ntimeout_sec = 0.2\n[verifier]\ntimeout_sec = 2\nenvironment_mode = \"separate\"\n",
        "exit 91\n",
    );
    let before_runs = compose_run_labels();
    let started = Instant::now();
    let mut child = start_eval(
        &agent_timeout,
        &format!("wget -qO /dev/null {}/agent && sleep 300", recorder.url()),
    );
    assert_eq!(
            next_request_from_child(
                &recorder,
                &mut child,
                "timed agent phase must start before its deadline",
                Duration::from_secs(60)
            ),
        "/agent"
    );
    let run = asserted_new_compose_run(&before_runs);
    let output = child
        .wait_with_output()
        .expect("wait for native aiperf eval");
    assert_failure_without_summary(&output, "timed out");
    assert!(
        started.elapsed() < Duration::from_secs(30),
        "agent timeout exceeded the end-to-end timeout budget: {:?}",
        started.elapsed()
    );
    assert_eq!(compose_resource_ids_for_run(&run), BTreeSet::new());

    let verifier_failure = compose_standard_task(
        &temporary,
        "compose-verifier-failure",
        "[environment]\nnetwork = \"public\"\n[verifier]\nenvironment_mode = \"separate\"\n",
        "exit 37\n",
    );
    let before_runs = compose_run_labels();
    let child = start_eval(
        &verifier_failure,
        &format!("wget -qO /dev/null {}/agent", recorder.url()),
    );
    assert_eq!(
        recorder.next_request_with_timeout(
            "agent must complete before separate verifier failure",
            Duration::from_secs(60)
        ),
        "/agent"
    );
    let run = asserted_new_compose_run(&before_runs);
    let output = child
        .wait_with_output()
        .expect("wait for native aiperf eval");
    assert_failure_without_summary(&output, "planned Docker phase");
    assert_eq!(
        compose_resource_ids_for_run(&run),
        BTreeSet::new(),
        "separate verifier failure leaked its exact Compose run resources"
    );
}

#[test]
#[ignore = "requires a Docker daemon and pulls alpine:3.20"]
fn compose_terminal_evidence_tears_down_the_project_before_the_separate_verifier_starts() {
    let _docker_lock = DOCKER_E2E_LOCK.lock().unwrap();
    let recorder = RequestRecorder::new();
    let temporary = tempfile::tempdir().unwrap();
    let task_root = compose_standard_task(
        &temporary,
        "compose-teardown-before-verifier",
        "artifacts = [{ source = \"/data\", destination = \"evidence\", service = \"api\" }]\n[environment]\nnetwork = \"public\"\n[verifier]\nenvironment_mode = \"separate\"\n[[verifier.collect]]\nservice = \"api\"\ncommand = [\"/bin/sh\", \"-c\", \"mkdir -p /data && printf collected > /data/result\"]\ntimeout_sec = 10\n",
        &format!(
            "test \"$(cat /work/evidence/result)\" = collected\nwget -qO /dev/null {}/verifier\nprintf '{{\"reward\":1.0}}' > /logs/verifier/reward.json\n",
            recorder.url()
        ),
    );
    let before_runs = compose_run_labels();
    let child = start_eval(
        &task_root,
        &format!("wget -qO /dev/null {}/agent && sleep 2", recorder.url()),
    );
    assert_eq!(
        recorder.next_request_with_timeout(
            "agent must start before terminal sidecar evidence",
            Duration::from_secs(60)
        ),
        "/agent"
    );
    let run = asserted_new_compose_run(&before_runs);
    assert_eq!(
        recorder.next_request_with_timeout(
            "separate verifier must run after collecting sidecar evidence",
            Duration::from_secs(60)
        ),
        "/verifier"
    );
    assert_eq!(
        compose_resource_ids_for_run(&run),
        BTreeSet::new(),
        "the Compose project survived into the separate verifier phase"
    );
    let output = child
        .wait_with_output()
        .expect("wait for native aiperf eval");
    assert_success(&output);
}

#[test]
#[ignore = "requires a Docker daemon and pulls alpine:3.20"]
fn compose_multi_step_execution_retains_the_main_project_workspace_until_the_final_step() {
    let _docker_lock = DOCKER_E2E_LOCK.lock().unwrap();
    let temporary = tempfile::tempdir().unwrap();
    let task_root = compose_standard_task(
        &temporary,
        "compose-multi-step-persistence",
        "multi_step_reward_strategy = \"final\"\nartifacts = [\"/work/result.txt\"]\n[environment]\nworkdir = \"/work\"\n[[steps]]\nname = \"one\"\n[[steps]]\nname = \"two\"\n[steps.verifier]\nenvironment_mode = \"separate\"\n",
        "if test -f /work/state; then\n  test \"$(cat /work/state)\" = one && test \"$(cat /work/result.txt)\" = first && printf '{\"score\":0.25}' > /logs/verifier/reward.json\nelse\n  test \"$(cat /work/result.txt)\" = second && printf '{\"score\":1.0}' > /logs/verifier/reward.json\nfi\n",
    );
    fs::create_dir_all(task_root.join("steps/one")).unwrap();
    fs::create_dir_all(task_root.join("steps/two")).unwrap();
    fs::write(
        task_root.join("steps/one/instruction.md"),
        "Compose step one.\n",
    )
    .unwrap();
    fs::write(
        task_root.join("steps/two/instruction.md"),
        "Compose step two.\n",
    )
    .unwrap();

    let before = compose_resource_ids();
    let output = run_eval(
        &task_root,
        r#"case "$AIPERF_EVAL_INSTRUCTION" in
  *one*) printf one > state && printf first > result.txt ;;
  *two*) test "$(cat state)" = one && nslookup api >/dev/null && printf two > state && printf second > result.txt ;;
  *) exit 42 ;;
esac"#,
    );

    assert_success(&output);
    let summary: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(summary["task"], "example/compose-multi-step-persistence");
    assert_eq!(summary["reward"], serde_json::json!({"score": 1.0}));
    assert_eq!(
        summary["steps"],
        serde_json::json!([
            {
                "name": "one",
                "artifacts": [["result.txt", artifact_digest(b"first")]],
                "reward": {"score": 0.25},
            },
            {
                "name": "two",
                "artifacts": [["result.txt", artifact_digest(b"second")]],
                "reward": {"score": 1.0},
            },
        ])
    );
    assert_eq!(
        compose_resource_ids(),
        before,
        "multi-step Compose project leaked resources"
    );
}

fn assert_success(output: &Output) {
    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn assert_failure_without_summary(output: &Output, diagnostic: &str) {
    assert!(
        !output.status.success(),
        "evaluation unexpectedly succeeded"
    );
    assert!(
        String::from_utf8_lossy(&output.stderr).contains(diagnostic),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        String::from_utf8_lossy(&output.stdout).trim().is_empty(),
        "failed evaluation emitted a JSON summary: {}",
        String::from_utf8_lossy(&output.stdout)
    );
}

fn assert_summary(summary: &serde_json::Value, task: &str) {
    let object = summary.as_object().expect("summary must be a JSON object");
    assert_eq!(
        object.keys().map(String::as_str).collect::<Vec<_>>(),
        ["task", "artifacts", "reward"]
    );
    assert_eq!(summary["task"], task);
    assert_eq!(summary["reward"], serde_json::json!({ "reward": 1.0 }));
    assert!(summary["artifacts"].is_array());
}

fn task_container_names() -> BTreeSet<String> {
    let output = Command::new("docker")
        .args(["container", "ls", "--all", "--format", "{{.Names}}"])
        .output()
        .expect("inspect Docker containers");
    assert!(
        output.status.success(),
        "docker container listing failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .filter(|name| name.starts_with("aiperf-eval-"))
        .map(str::to_owned)
        .collect()
}

fn compose_resource_ids() -> BTreeSet<String> {
    [
        ("container", vec!["container", "ls", "--all", "--quiet"]),
        ("network", vec!["network", "ls", "--quiet"]),
        ("volume", vec!["volume", "ls", "--quiet"]),
    ]
    .into_iter()
    .flat_map(|(kind, arguments)| {
        let output = Command::new("docker")
            .args(arguments)
            .args(["--filter", "label=com.docker.compose.project"])
            .output()
            .expect("inspect Compose resources");
        assert!(
            output.status.success(),
            "docker {kind} listing failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8(output.stdout)
            .unwrap()
            .lines()
            .map(move |id| format!("{kind}:{id}"))
            .collect::<Vec<_>>()
    })
    .collect()
}

fn compose_run_labels() -> BTreeSet<String> {
    let output = Command::new("docker")
        .args([
            "container",
            "ls",
            "--all",
            "--quiet",
            "--filter",
            "label=aiperf.run",
        ])
        .output()
        .expect("list Compose run containers");
    assert!(
        output.status.success(),
        "Docker Compose run container listing failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .filter_map(|container| {
            let output = Command::new("docker")
                .args([
                    "container",
                    "inspect",
                    "--format",
                    "{{index .Config.Labels \"aiperf.run\"}}",
                    container,
                ])
                .output()
                .expect("inspect Compose run label");
            assert!(
                output.status.success(),
                "Docker Compose run label inspection failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            let label = String::from_utf8(output.stdout).unwrap();
            let label = label.trim();
            (!label.is_empty()).then(|| label.to_owned())
        })
        .collect()
}

fn compose_resource_ids_for_run(run: &str) -> BTreeSet<String> {
    [
        ("container", vec!["container", "ls", "--all", "--quiet"]),
        ("network", vec!["network", "ls", "--quiet"]),
        ("volume", vec!["volume", "ls", "--quiet"]),
    ]
    .into_iter()
    .flat_map(|(kind, arguments)| {
        let output = Command::new("docker")
            .args(arguments)
            .args(["--filter", &format!("label=aiperf.run={run}")])
            .output()
            .expect("inspect Compose run resources");
        assert!(
            output.status.success(),
            "docker {kind} run resource listing failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8(output.stdout)
            .unwrap()
            .lines()
            .map(move |id| format!("{kind}:{id}"))
            .collect::<Vec<_>>()
    })
    .collect()
}

fn asserted_new_compose_run(before: &BTreeSet<String>) -> String {
    let runs = compose_run_labels();
    let new = runs.difference(before).collect::<Vec<_>>();
    assert_eq!(
        new.len(),
        1,
        "exactly one run label must appear after the controlled agent request; active runs: {runs:?}"
    );
    (*new[0]).clone()
}

struct RequestRecorder {
    url: String,
    requests: Receiver<String>,
    shutdown: Arc<AtomicBool>,
    server: Option<JoinHandle<()>>,
}

fn next_request_from_child(
    recorder: &RequestRecorder,
    child: &mut std::process::Child,
    context: &str,
    timeout: Duration,
) -> String {
    let deadline = Instant::now() + timeout;
    loop {
        match recorder.requests.try_recv() {
            Ok(request) => return request,
            Err(TryRecvError::Disconnected) => panic!("{context}: recorder disconnected"),
            Err(TryRecvError::Empty) => {}
        }
        if let Some(status) = child.try_wait().expect("poll native aiperf eval") {
            let mut stdout = Vec::new();
            let mut stderr = Vec::new();
            child
                .stdout
                .take()
                .expect("native aiperf eval stdout")
                .read_to_end(&mut stdout)
                .expect("read native aiperf eval stdout");
            child
                .stderr
                .take()
                .expect("native aiperf eval stderr")
                .read_to_end(&mut stderr)
                .expect("read native aiperf eval stderr");
            panic!(
                "{context}: native aiperf eval exited early with {status}; stdout: {}; stderr: {}",
                String::from_utf8_lossy(&stdout),
                String::from_utf8_lossy(&stderr)
            );
        }
        assert!(Instant::now() < deadline, "{context}: timed out waiting for request");
        thread::sleep(Duration::from_millis(10));
    }
}

impl RequestRecorder {
    fn new() -> Self {
        let listener = TcpListener::bind(("0.0.0.0", 0)).expect("bind request recorder");
        listener.set_nonblocking(true).unwrap();
        let port = listener.local_addr().unwrap().port();
        let gateway = public_network_gateway();
        let (sender, requests) = mpsc::channel();
        let shutdown = Arc::new(AtomicBool::new(false));
        let is_shutdown = shutdown.clone();
        let server = thread::spawn(move || {
            while !is_shutdown.load(Ordering::Relaxed) {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        let mut request = [0; 1024];
                        let count = stream.read(&mut request).unwrap_or(0);
                        let path = String::from_utf8_lossy(&request[..count])
                            .lines()
                            .next()
                            .and_then(|line| line.split_whitespace().nth(1))
                            .unwrap_or("/")
                            .to_owned();
                        let _ = sender.send(path);
                        let _ = stream.write_all(
                            b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nOK",
                        );
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(10));
                    }
                    Err(_) => break,
                }
            }
        });
        Self {
            url: format!("http://{gateway}:{port}"),
            requests,
            shutdown,
            server: Some(server),
        }
    }

    fn url(&self) -> &str {
        &self.url
    }

    fn next_request(&self, context: &str) -> String {
        self.next_request_with_timeout(context, Duration::from_secs(3))
    }

    fn next_request_with_timeout(&self, context: &str, timeout: Duration) -> String {
        self.requests
            .recv_timeout(timeout)
            .unwrap_or_else(|error| panic!("{context}: {error}"))
    }

    fn assert_no_request(&self, context: &str) {
        assert!(
            self.requests
                .recv_timeout(Duration::from_millis(300))
                .is_err(),
            "{context}"
        );
    }
}

impl Drop for RequestRecorder {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
        if let Some(server) = self.server.take() {
            server.join().unwrap();
        }
    }
}

fn public_network_gateway() -> String {
    let network = "aiperf-eval-public";
    let inspect = Command::new("docker")
        .args([
            "network",
            "inspect",
            network,
            "--format",
            "{{(index .IPAM.Config 0).Gateway}}",
        ])
        .output()
        .expect("inspect public Docker network");
    if !inspect.status.success() {
        let created = Command::new("docker")
            .args(["network", "create", network])
            .status()
            .expect("create public Docker network");
        assert!(created.success(), "create public Docker network");
    }
    let gateway = Command::new("docker")
        .args([
            "network",
            "inspect",
            network,
            "--format",
            "{{(index .IPAM.Config 0).Gateway}}",
        ])
        .output()
        .expect("inspect public Docker network gateway");
    assert!(gateway.status.success());
    String::from_utf8(gateway.stdout).unwrap().trim().to_owned()
}
