// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

use std::fs;
use std::process::Command;
use std::sync::Mutex;
use std::time::Duration;

static DOCKER_TIMEOUT_TEST_LOCK: Mutex<()> = Mutex::new(());

#[test]
fn native_eval_command_runs_a_local_harbor_package() {
    let temporary = tempfile::tempdir().unwrap();
    let package_path = temporary.path().join("task.json");
    fs::write(
        &package_path,
        br#"{"id":"repair-1","instruction":"Fix","environment":"blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","verifier":"blake3:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","agent_command":["sh","-c","printf patch > \"$AIPERF_EVAL_ROOT/results/patch.diff\""],"verifier_command":["sh","-c","test -f results/patch.diff && printf '{\"reward\":1.0}' > reward.json"],"declared_artifacts":["/results/patch.diff"]}"#,
    )
    .unwrap();

    let exit = aiperf_cli::dispatch::run(&[
        "eval".to_owned(),
        "--task".to_owned(),
        package_path.to_string_lossy().into_owned(),
        "--image".to_owned(),
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_owned(),
    ])
    .unwrap();

    assert_eq!(exit, 0);
}

#[test]
fn native_eval_command_runs_a_pinned_git_harbor_package() {
    let temporary = tempfile::tempdir().unwrap();
    let repository = temporary.path().join("tasks");
    fs::create_dir(&repository).unwrap();
    run_git(&repository, ["init"]);
    run_git(
        &repository,
        ["config", "user.email", "eval@example.invalid"],
    );
    run_git(&repository, ["config", "user.name", "Native Eval"]);
    fs::write(
        repository.join("task.json"),
        br#"{"id":"repair-1","instruction":"Fix","environment":"blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","verifier":"blake3:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","agent_command":["sh","-c","printf patch > \"$AIPERF_EVAL_ROOT/results/patch.diff\""],"verifier_command":["sh","-c","test -f results/patch.diff && printf '{\"reward\":1.0}' > reward.json"],"declared_artifacts":["/results/patch.diff"]}"#,
    )
    .unwrap();
    run_git(&repository, ["add", "task.json"]);
    run_git(&repository, ["commit", "-m", "pinned task"]);
    let revision = git_output(&repository, ["rev-parse", "HEAD"]);

    let exit = aiperf_cli::dispatch::run(&[
        "eval".to_owned(),
        "--git-repository".to_owned(),
        repository.to_string_lossy().into_owned(),
        "--git-revision".to_owned(),
        revision,
        "--git-path".to_owned(),
        "task.json".to_owned(),
        "--image".to_owned(),
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_owned(),
    ])
    .unwrap();

    assert_eq!(exit, 0);
}

#[test]
#[ignore = "requires a Docker daemon and the local openclaw sandbox image"]
fn native_eval_command_runs_a_pinned_standard_task_directory_in_docker() {
    let temporary = tempfile::tempdir().unwrap();
    let repository = temporary.path().join("tasks");
    fs::create_dir(&repository).unwrap();
    run_git(&repository, ["init"]);
    run_git(
        &repository,
        ["config", "user.email", "eval@example.invalid"],
    );
    run_git(&repository, ["config", "user.name", "Native Eval"]);
    fs::create_dir_all(repository.join("task/environment")).unwrap();
    fs::create_dir_all(repository.join("task/tests")).unwrap();
    fs::write(
        repository.join("task/task.toml"),
        "schema_version = \"1.0\"\n[task]\nname = \"example/pinned-standard\"\n",
    )
    .unwrap();
    fs::write(repository.join("task/instruction.md"), "Write a result.\n").unwrap();
    fs::write(
        repository.join("task/environment/Dockerfile"),
        "FROM openclaw-sandbox:bookworm-slim\n",
    )
    .unwrap();
    fs::write(
        repository.join("task/tests/test.sh"),
        "test -f /work/result.txt\nmkdir -p /logs/verifier\nprintf 1 > /logs/verifier/reward.txt\n",
    )
    .unwrap();
    run_git(&repository, ["add", "."]);
    run_git(&repository, ["commit", "-m", "standard task"]);
    let revision = git_output(&repository, ["rev-parse", "HEAD"]);

    let exit = aiperf_cli::dispatch::run(&[
        "eval".to_owned(),
        "--git-repository".to_owned(),
        repository.to_string_lossy().into_owned(),
        "--git-revision".to_owned(),
        revision,
        "--git-path".to_owned(),
        "task/task.toml".to_owned(),
        "--agent-command".to_owned(),
        "printf result > result.txt".to_owned(),
    ])
    .unwrap();
    assert_eq!(exit, 0);
}

#[test]
fn native_eval_command_runs_a_standard_task_directory() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = temporary.path().join("repair-1");
    fs::create_dir_all(task_root.join("environment")).unwrap();
    fs::create_dir_all(task_root.join("tests")).unwrap();
    fs::write(
        task_root.join("task.toml"),
        "schema_version = \"1.0\"\n[task]\nname = \"example/repair-1\"\n",
    )
    .unwrap();
    fs::write(task_root.join("instruction.md"), "Write the result.\n").unwrap();
    fs::write(task_root.join("environment/Dockerfile"), "FROM scratch\n").unwrap();
    fs::write(
        task_root.join("tests/test.sh"),
        "test -f result.txt && printf 1 > reward.txt\n",
    )
    .unwrap();

    let exit = aiperf_cli::dispatch::run(&[
        "eval".to_owned(),
        "--task".to_owned(),
        task_root.to_string_lossy().into_owned(),
        "--image".to_owned(),
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_owned(),
        "--agent-command".to_owned(),
        "printf result > result.txt".to_owned(),
        "--sandbox".to_owned(),
        "local".to_owned(),
    ])
    .unwrap();

    assert_eq!(exit, 0);
}

#[test]
fn native_eval_rejects_standard_task_verifier_mode_override() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = temporary.path().join("mode-conflict");
    fs::create_dir_all(task_root.join("environment")).unwrap();
    fs::create_dir_all(task_root.join("tests")).unwrap();
    fs::write(
        task_root.join("task.toml"),
        "schema_version = \"1.0\"\n[task]\nname = \"example/mode-conflict\"\n[verifier]\nenvironment_mode = \"separate\"\n",
    )
    .unwrap();
    fs::write(task_root.join("instruction.md"), "Write a result.\n").unwrap();
    fs::write(task_root.join("environment/Dockerfile"), "FROM scratch\n").unwrap();
    fs::write(task_root.join("tests/test.sh"), "exit 0\n").unwrap();

    let error = aiperf_cli::dispatch::run(&[
        "eval".to_owned(),
        "--task".to_owned(),
        task_root.to_string_lossy().into_owned(),
        "--verifier-mode".to_owned(),
        "shared".to_owned(),
        "--agent-command".to_owned(),
        "true".to_owned(),
    ])
    .expect_err("standard task mode must not be silently overridden");

    assert!(
        error
            .to_string()
            .contains("--verifier-mode conflicts with the standard task")
    );
}

#[test]
#[ignore = "requires a Docker daemon and the local openclaw sandbox image"]
fn native_eval_command_runs_a_standard_task_directory_in_docker() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = temporary.path().join("docker-repair-1");
    fs::create_dir_all(task_root.join("environment")).unwrap();
    fs::create_dir_all(task_root.join("tests")).unwrap();
    fs::write(
        task_root.join("task.toml"),
        "schema_version = \"1.0\"\nartifacts = [\"/work/result.txt\"]\n[task]\nname = \"example/docker-repair-1\"\n[verifier]\nenvironment_mode = \"separate\"\n",
    )
    .unwrap();
    fs::write(task_root.join("instruction.md"), "Write the result.\n").unwrap();
    fs::write(
        task_root.join("environment/Dockerfile"),
        "FROM openclaw-sandbox:bookworm-slim\n",
    )
    .unwrap();
    fs::write(
        task_root.join("tests/test.sh"),
        "test -f /work/result.txt\ntest ! -e /work/agent-secret\nmkdir -p /logs/verifier\nprintf '{\"reward\":1.0}' > /logs/verifier/reward.json\n",
    )
    .unwrap();

    let exit = aiperf_cli::dispatch::run(&[
        "eval".to_owned(),
        "--task".to_owned(),
        task_root.to_string_lossy().into_owned(),
        "--image".to_owned(),
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_owned(),
        "--agent-command".to_owned(),
        "test ! -e /tests/test.sh && printf secret > agent-secret && printf result > result.txt"
            .to_owned(),
    ])
    .unwrap();

    assert_eq!(exit, 0);
}

#[test]
#[ignore = "requires a Docker daemon and the local openclaw sandbox image"]
fn docker_timeout_removes_agent_container_after_descendant_command() {
    let _docker_test_lock = DOCKER_TIMEOUT_TEST_LOCK.lock().unwrap();
    let task_root = docker_timeout_task(
        "agent",
        "mkdir -p /logs/verifier\nprintf 1 > /logs/verifier/reward.txt\n",
        "shared",
        0.2,
        2.0,
    );

    let error = aiperf_cli::dispatch::run(&docker_eval_arguments(
        task_root.path(),
        "sleep 300 & sleep 2",
    ))
    .expect_err("an agent command exceeding its configured timeout must fail");

    let execution_error = error.downcast_ref::<aiperf_runtime::eval::EvalExecutionError>();
    assert!(
        matches!(
            execution_error,
            Some(aiperf_runtime::eval::EvalExecutionError::Timeout {
                phase: aiperf_runtime::eval::EvalExecutionPhase::Agent,
                timeout,
            }) if *timeout == Duration::from_millis(200)
        ),
        "unexpected agent timeout result: {execution_error:?}"
    );
    assert_task_containers_absent();
}

#[test]
#[ignore = "requires a Docker daemon and the local openclaw sandbox image"]
fn docker_timeout_removes_separate_verifier_container_after_descendant_command() {
    let _docker_test_lock = DOCKER_TIMEOUT_TEST_LOCK.lock().unwrap();
    let task_root = docker_timeout_task(
        "verifier",
        "sleep 300 & sleep 2\nmkdir -p /logs/verifier\nprintf 1 > /logs/verifier/reward.txt\n",
        "separate",
        2.0,
        0.2,
    );

    let error = aiperf_cli::dispatch::run(&docker_eval_arguments(task_root.path(), "true"))
        .expect_err("a verifier command exceeding its configured timeout must fail");

    let execution_error = error
        .downcast_ref::<aiperf_runtime::eval::EvalExecutionError>()
        .expect("Docker evaluation errors must preserve their typed execution cause");
    assert!(
        matches!(
            execution_error,
            aiperf_runtime::eval::EvalExecutionError::Timeout {
                phase: aiperf_runtime::eval::EvalExecutionPhase::Verifier,
                timeout,
            } if *timeout == Duration::from_millis(200)
        ),
        "unexpected verifier timeout result: {execution_error:?}"
    );
    assert_task_containers_absent();
}

fn docker_timeout_task(
    name: &str,
    verifier_script: &str,
    verifier_mode: &str,
    agent_timeout: f64,
    verifier_timeout: f64,
) -> tempfile::TempDir {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = temporary.path();
    fs::create_dir_all(task_root.join("environment")).unwrap();
    fs::create_dir_all(task_root.join("tests")).unwrap();
    fs::write(
        task_root.join("task.toml"),
        format!(
            "schema_version = \"1.0\"\n[task]\nname = \"example/docker-timeout-{name}\"\n[agent]\ntimeout_sec = {agent_timeout}\n[verifier]\ntimeout_sec = {verifier_timeout}\nenvironment_mode = \"{verifier_mode}\"\n"
        ),
    )
    .unwrap();
    fs::write(task_root.join("instruction.md"), "Complete the task.\n").unwrap();
    fs::write(
        task_root.join("environment/Dockerfile"),
        "FROM openclaw-sandbox:bookworm-slim\n",
    )
    .unwrap();
    fs::write(task_root.join("tests/test.sh"), verifier_script).unwrap();
    temporary
}

fn docker_eval_arguments(task_root: &std::path::Path, agent_command: &str) -> Vec<String> {
    vec![
        "eval".to_owned(),
        "--task".to_owned(),
        task_root.to_string_lossy().into_owned(),
        "--image".to_owned(),
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_owned(),
        "--agent-command".to_owned(),
        agent_command.to_owned(),
    ]
}

fn assert_task_containers_absent() {
    let prefix = format!("aiperf-eval-{}-", std::process::id());
    let output = Command::new("docker")
        .args(["container", "ls", "--all", "--format", "{{.Names}}"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "unable to inspect Docker containers: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let names = String::from_utf8_lossy(&output.stdout);
    let remaining = names
        .lines()
        .filter(|name| name.starts_with(&prefix))
        .collect::<Vec<_>>();
    assert!(
        remaining.is_empty(),
        "task containers remained after the evaluation API returned: {remaining:?}"
    );
}

fn run_git<const N: usize>(repository: &std::path::Path, arguments: [&str; N]) {
    let status = Command::new("git")
        .arg("-c")
        .arg("commit.gpgsign=false")
        .arg("-C")
        .arg(repository)
        .args(arguments)
        .status()
        .unwrap();
    assert!(status.success());
}

fn git_output<const N: usize>(repository: &std::path::Path, arguments: [&str; N]) -> String {
    let output = Command::new("git")
        .arg("-C")
        .arg(repository)
        .args(arguments)
        .output()
        .unwrap();
    assert!(output.status.success());
    String::from_utf8(output.stdout).unwrap().trim().to_owned()
}
