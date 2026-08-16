// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::fs;
use std::process::Command;

use aiperf_runtime::eval::{
    ArtifactDigest, HarborImporter, HarborSource, ImportDisposition, NativeSourceAcquirer,
    SourceAcquirer,
};

#[derive(Default)]
struct MemoryAcquirer {
    packages: BTreeMap<String, Vec<u8>>,
}

impl SourceAcquirer for MemoryAcquirer {
    fn acquire(
        &self,
        source: &HarborSource,
    ) -> Result<Vec<u8>, aiperf_runtime::eval::HarborImportError> {
        self.packages
            .get(source.location())
            .cloned()
            .ok_or_else(|| {
                aiperf_runtime::eval::HarborImportError::Unavailable(source.location().to_owned())
            })
    }
}

#[test]
fn retains_executable_task_material() {
    let bytes = supported_package();
    let source = HarborSource::local("fixtures/repair-1").unwrap();
    let mut acquirer = MemoryAcquirer::default();
    acquirer
        .packages
        .insert(source.location().to_owned(), bytes.clone());

    let imported = HarborImporter::new(&acquirer).import(&source).unwrap();

    assert_eq!(
        imported.report.source_digest,
        ArtifactDigest::from_bytes(&bytes)
    );
    assert_eq!(
        imported.report.disposition,
        ImportDisposition::LosslessNormalized
    );
    assert_eq!(imported.task.id.as_str(), "repair-1");
    assert_eq!(imported.package.id, "repair-1");
    assert_eq!(imported.package.instruction, "Fix the failing test");
    assert_eq!(
        imported.package.environment,
        "blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    );
    assert_eq!(
        imported.package.verifier,
        "blake3:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    );
    assert_eq!(
        imported.package.agent_command,
        vec!["sh", "-c", "printf patch > out/patch.diff"]
    );
    assert_eq!(
        imported.package.verifier_command,
        vec!["sh", "-c", "printf '{\"reward\":1.0}' > reward.json"]
    );
    assert_eq!(
        imported.package.declared_artifacts,
        vec!["/results/patch.diff"]
    );
    assert_eq!(
        imported.package.source_digest(),
        imported.report.source_digest
    );
}

#[test]
fn rejects_malformed_executable_task_material() {
    let malformed_packages = [
        package_with_mutation(|package| package["agent_command"] = serde_json::json!([])),
        package_with_mutation(|package| {
            package["verifier_command"] = serde_json::json!(["sh", " "])
        }),
        package_with_mutation(|package| {
            package["declared_artifacts"] = serde_json::json!(["results/patch.diff"])
        }),
        package_with_mutation(|package| package["declared_artifacts"] = serde_json::json!(["/"])),
        package_with_mutation(|package| {
            package["declared_artifacts"] = serde_json::json!(["/results/../secret"])
        }),
        package_with_mutation(|package| {
            package["declared_artifacts"] =
                serde_json::json!(["/results/patch.diff", "//results/patch.diff"])
        }),
        package_with_mutation(|package| package["unknown_package_field"] = serde_json::json!(true)),
        package_with_mutation(|package| {
            package.as_object_mut().unwrap().remove("agent_command");
        }),
    ];

    for bytes in malformed_packages {
        let source = HarborSource::local("fixtures/malformed").unwrap();
        let mut acquirer = MemoryAcquirer::default();
        acquirer
            .packages
            .insert(source.location().to_owned(), bytes);

        assert!(HarborImporter::new(&acquirer).import(&source).is_err());
    }
}

fn supported_package() -> Vec<u8> {
    br#"{
        "id":"repair-1",
        "instruction":"Fix the failing test",
        "environment":"blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "verifier":"blake3:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        "agent_command":["sh","-c","printf patch > out/patch.diff"],
        "verifier_command":["sh","-c","printf '{\"reward\":1.0}' > reward.json"],
        "declared_artifacts":["/results/patch.diff"]
    }"#
    .to_vec()
}

fn package_with_mutation(mutate: impl FnOnce(&mut serde_json::Value)) -> Vec<u8> {
    let mut package = serde_json::from_slice(&supported_package()).unwrap();
    mutate(&mut package);
    serde_json::to_vec(&package).unwrap()
}

#[test]
fn unsupported_semantics_return_report_before_provisioning() {
    let source = HarborSource::local("fixtures/unsupported").unwrap();
    let mut acquirer = MemoryAcquirer::default();
    acquirer.packages.insert(
        source.location().to_owned(),
        br#"{"id":"repair-1","unsupported_semantics":"sidecar"}"#.to_vec(),
    );

    let refusal = HarborImporter::new(&acquirer).import(&source).unwrap_err();

    assert_eq!(refusal.disposition(), Some(ImportDisposition::Unsupported));
}

#[test]
fn native_acquirer_reads_local_and_pinned_git_package_bytes() {
    let temporary = tempfile::tempdir().unwrap();
    let package = br#"{"id":"repair-1","instruction":"Fix","environment":"blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","verifier":"blake3:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}"#;
    let local_path = temporary.path().join("local-task.json");
    fs::write(&local_path, package).unwrap();
    let acquirer = NativeSourceAcquirer;

    assert_eq!(
        acquirer
            .acquire(&HarborSource::local(local_path.to_string_lossy()).unwrap())
            .unwrap(),
        package
    );

    let repository = temporary.path().join("task-repository");
    fs::create_dir(&repository).unwrap();
    run_git(&repository, ["init"]);
    run_git(
        &repository,
        ["config", "user.email", "eval@example.invalid"],
    );
    run_git(&repository, ["config", "user.name", "Native Eval"]);
    let package_path = repository.join("task.json");
    fs::write(&package_path, package).unwrap();
    run_git(&repository, ["add", "task.json"]);
    run_git(&repository, ["commit", "-m", "task A"]);
    let revision = git_output(&repository, ["rev-parse", "HEAD"]);
    fs::write(&package_path, br#"{"different":"head"}"#).unwrap();
    run_git(&repository, ["add", "task.json"]);
    run_git(&repository, ["commit", "-m", "task B"]);

    assert_eq!(
        acquirer
            .acquire(
                &HarborSource::pinned_git(repository.to_string_lossy(), revision, "task.json",)
                    .unwrap(),
            )
            .unwrap(),
        package
    );
}

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
