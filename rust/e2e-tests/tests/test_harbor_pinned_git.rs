// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

use std::fs;
use std::process::Command;

use aiperf_runtime::eval::{ArtifactDigest, HarborImporter, HarborSource, NativeSourceAcquirer};

fn run_git<const N: usize>(repository: &std::path::Path, arguments: [&str; N]) {
    let status = Command::new("git")
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

#[test]
fn pinned_git_source_reproduces_imported_task_and_source_identity() {
    let temporary = tempfile::tempdir().unwrap();
    let repository = temporary.path().join("tasks");
    fs::create_dir(&repository).unwrap();
    run_git(&repository, ["init"]);
    run_git(
        &repository,
        ["config", "user.email", "eval@example.invalid"],
    );
    run_git(&repository, ["config", "user.name", "Native Eval"]);
    let package = br#"{"id":"repair-1","instruction":"Fix","environment":"blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","verifier":"blake3:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}"#;
    fs::write(repository.join("task.json"), package).unwrap();
    run_git(&repository, ["add", "task.json"]);
    run_git(&repository, ["commit", "-m", "pinned task"]);
    let revision = git_output(&repository, ["rev-parse", "HEAD"]);
    fs::write(
        repository.join("task.json"),
        br#"{"not":"the pinned source"}"#,
    )
    .unwrap();
    run_git(&repository, ["add", "task.json"]);
    run_git(&repository, ["commit", "-m", "later task"]);

    let source =
        HarborSource::pinned_git(repository.to_string_lossy(), revision, "task.json").unwrap();
    let importer = HarborImporter::new(&NativeSourceAcquirer);
    let first = importer.import(&source).unwrap();
    let second = importer.import(&source).unwrap();

    assert_eq!(first.task, second.task);
    assert_eq!(first.report, second.report);
    assert_eq!(
        first.report.source_digest,
        ArtifactDigest::from_bytes(package)
    );
}
