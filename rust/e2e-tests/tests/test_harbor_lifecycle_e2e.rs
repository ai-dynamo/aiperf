// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Docker-backed lifecycle acceptance coverage for native Harbor evaluation.

mod common;

use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
    sync::Mutex,
};

use aiperf_runtime::eval::ScoreVersion;
use common::exec_binary;

const IMAGE_DIGEST: &str =
    "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const POLICY_DIGEST: &str =
    "blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

static DOCKER_E2E_LOCK: Mutex<()> = Mutex::new(());

fn docker_e2e_lock() -> std::sync::MutexGuard<'static, ()> {
    DOCKER_E2E_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn git<const N: usize>(repository: &Path, arguments: [&str; N]) {
    let status = Command::new("git")
        .arg("-c")
        .arg("commit.gpgsign=false")
        .arg("-C")
        .arg(repository)
        .args(arguments)
        .status()
        .expect("run git");
    assert!(status.success(), "git arguments: {arguments:?}");
}

fn git_stdout<const N: usize>(repository: &Path, arguments: [&str; N]) -> String {
    let output = Command::new("git")
        .arg("-C")
        .arg(repository)
        .args(arguments)
        .output()
        .expect("run git");
    assert!(output.status.success(), "git arguments: {arguments:?}");
    String::from_utf8(output.stdout)
        .expect("Git emits UTF-8 revisions")
        .trim()
        .to_owned()
}

fn lifecycle_request(command: &str) -> String {
    format!(
        r#"{{"version":1,"agent_variant":"native:lifecycle-e2e","model":{{"provider":"native","model":"test"}},"seed":17,"policy":"{POLICY_DIGEST}","runtime":"native:e2e","attempt":"lifecycle-e2e-attempt","budget":{{"execution_seconds":10.0,"verifier_seconds":10.0}},"agent_contract":"external","command":["/bin/sh","-c",{command:?}],"initial_score":{{"metric":"reward","rationale":"{POLICY_DIGEST}"}},"regrade":{{"metric":"quality","rationale":"{POLICY_DIGEST}"}}}}"#
    )
}

fn write_task(repository: &Path) -> PathBuf {
    let task = repository.join("task");
    fs::create_dir_all(task.join("environment")).expect("create task environment");
    fs::create_dir_all(task.join("tests")).expect("create task tests");
    fs::write(
        task.join("task.toml"),
        r#"schema_version = "1.0"
artifacts = ["/work/result.txt"]
[task]
name = "example/lifecycle-e2e"
[environment]
workdir = "/work"
[agent]
timeout_sec = 10
[agent.env]
AIPERF_HARBOR_P0_AGENT_CREDENTIAL = "agent-only-credential"
[verifier]
environment_mode = "separate"
timeout_sec = 10
"#,
    )
    .expect("write task manifest");
    fs::write(
        task.join("instruction.md"),
        "Produce the declared artifact.\n",
    )
    .expect("write instruction");
    fs::write(
        task.join("environment/Dockerfile"),
        "FROM alpine:3.20\nRUN mkdir -p /work /logs/verifier && chmod 0777 /logs/verifier\n",
    )
    .expect("write Dockerfile");
    fs::write(
        task.join("tests/test.sh"),
        "test \"$(cat /work/result.txt)\" = lifecycle-artifact\ntest ! -e /work/private.txt\ntest -z \"${AIPERF_HARBOR_P0_AGENT_CREDENTIAL:-}\"\nprintf '{\"reward\":1.0,\"quality\":0.75}' > /logs/verifier/reward.json\nprintf '0.01' > /logs/verifier/reward.txt\n",
    )
    .expect("write verifier");
    task
}

fn run_lifecycle_eval(
    repository: &Path,
    revision: &str,
    lifecycle: &Path,
    output: &Path,
    command: &str,
    harbor_process_spy_directory: &Path,
) -> Output {
    let path = format!(
        "{}:{}",
        harbor_process_spy_directory.display(),
        std::env::var("PATH").unwrap_or_default()
    );
    Command::new(exec_binary())
        .args([
            "eval",
            "--git-repository",
            repository.to_string_lossy().as_ref(),
            "--git-revision",
            revision,
            "--git-path",
            "task",
            "--image",
            IMAGE_DIGEST,
            "--agent-command",
            command,
            "--lifecycle-request",
            lifecycle.to_string_lossy().as_ref(),
            "--lifecycle-output",
            output.to_string_lossy().as_ref(),
        ])
        .env("PATH", path)
        .output()
        .expect("start native lifecycle evaluation")
}

fn write_harbor_process_spy(temporary: &Path) -> (PathBuf, PathBuf) {
    let directory = temporary.join("harbor-process-spy");
    fs::create_dir(&directory).expect("create Harbor process spy directory");
    let invoked = temporary.join("harbor-process-invoked");
    let program = directory.join("harbor");
    fs::write(
        &program,
        format!(
            "#!/bin/sh\nprintf invoked > {}\nexit 97\n",
            invoked.display()
        ),
    )
    .expect("write Harbor process spy");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;

        let mut permissions = fs::metadata(&program)
            .expect("read Harbor process spy permissions")
            .permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&program, permissions).expect("make Harbor process spy executable");
    }
    (directory, invoked)
}

#[test]
#[ignore = "requires a Docker daemon and pulls alpine:3.20"]
fn pinned_docker_lifecycle_withholds_agent_credential_and_workspace_and_never_invokes_harbor() {
    let _docker_lock = docker_e2e_lock();
    let temporary = tempfile::tempdir().expect("create temporary repository");
    let repository = temporary.path().join("harbor-tasks");
    fs::create_dir(&repository).expect("create repository");
    git(&repository, ["init"]);
    git(
        &repository,
        ["config", "user.email", "eval@example.invalid"],
    );
    git(&repository, ["config", "user.name", "AIPerf Eval"]);
    let task = write_task(&repository);
    git(&repository, ["add", "task"]);
    git(&repository, ["commit", "-m", "pinned lifecycle task"]);
    let revision = git_stdout(&repository, ["rev-parse", "HEAD"]);

    // The pinned source must come from the selected object, not the caller's
    // mutable working tree after the revision was recorded.
    fs::write(
        task.join("task.toml"),
        "schema_version = \"1.0\"\n[task]\nname = \"mutated/not-executable\"\n",
    )
    .expect("mutate source after pinning");
    fs::remove_file(task.join("tests/test.sh")).expect("remove mutable verifier source");

    let command = "test \"$AIPERF_HARBOR_P0_AGENT_CREDENTIAL\" = agent-only-credential && printf lifecycle-artifact > result.txt && printf private > private.txt";
    let request = temporary.path().join("lifecycle-request.json");
    let record = temporary.path().join("lifecycle-record.json");
    let (harbor_process_spy_directory, harbor_process_invoked) =
        write_harbor_process_spy(temporary.path());
    fs::write(&request, lifecycle_request(command)).expect("write lifecycle request");
    let output = run_lifecycle_eval(
        &repository,
        &revision,
        &request,
        &record,
        command,
        &harbor_process_spy_directory,
    );
    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let summary: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("lifecycle summary is JSON");
    let persisted: serde_json::Value = serde_json::from_slice(
        &fs::read(&record).expect("lifecycle record is persisted atomically"),
    )
    .expect("persisted lifecycle record is JSON");
    assert_eq!(summary["task"], "example/lifecycle-e2e");
    assert_eq!(summary["reward"]["reward"], 1.0);
    assert_eq!(summary["reward"]["quality"], 0.75);
    assert_eq!(summary["lifecycle"], persisted);
    assert_eq!(
        persisted["source"],
        serde_json::json!({
            "kind": "pinned_git",
            "repository": repository,
            "revision": revision,
            "package_path": "task/task.toml",
        })
    );
    assert_eq!(persisted["trial"]["seed"], 17);
    assert_eq!(persisted["initial_score"]["value"], 1.0);
    assert_eq!(persisted["regraded_score"]["value"], 0.75);
    let initial_score: ScoreVersion = serde_json::from_value(persisted["initial_score"].clone())
        .expect("persisted initial score remains strict and valid");
    assert_eq!(
        persisted["regraded_score"]["predecessor"],
        initial_score.identity_digest().as_str()
    );
    assert_eq!(
        persisted["verifier_result"]["evidence"],
        serde_json::json!([format!(
            "blake3:{}",
            blake3::hash(b"lifecycle-artifact").to_hex()
        )])
    );
    assert_eq!(
        summary["artifacts"],
        serde_json::json!([[
            "result.txt",
            format!("blake3:{}", blake3::hash(b"lifecycle-artifact").to_hex())
        ]])
    );
    assert!(
        !harbor_process_invoked.exists(),
        "native evaluation must not invoke a Harbor executable"
    );
}
