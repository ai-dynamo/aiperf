// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

use std::fs;
use std::process::Command;

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
