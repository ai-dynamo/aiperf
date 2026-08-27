// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Same-process authority for paired plugin build measurements.

use std::path::{Path, PathBuf};

use aiperf_bench_tools::build_pair::{
    BuildDurationUseV1, BuildLtoV1, BuildPairMemberV1, BuildPairPlanV1,
    build_pair_authority_blake3_v1, run_paired_build_v1,
};
use aiperf_bench_tools::plugin_stats::Variant;

const COMMIT: &str = "0123456789abcdef0123456789abcdef01234567";
const IDENTITY: &str = "blake3:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

fn digest(bytes: &[u8]) -> String {
    format!("blake3:{}", blake3::hash(bytes))
}

fn write_executable(path: &Path, bytes: &[u8]) {
    use std::os::unix::fs::PermissionsExt as _;

    std::fs::write(path, bytes).expect("fixture executable is written");
    let mut permissions = std::fs::metadata(path)
        .expect("fixture executable metadata is available")
        .permissions();
    permissions.set_mode(0o755);
    std::fs::set_permissions(path, permissions).expect("fixture executable is executable");
}

struct Fixture {
    _directory: tempfile::TempDir,
    cargo: PathBuf,
    cargo_digest: String,
    rustc: PathBuf,
    rustc_digest: String,
    rustc_verbose_version: Vec<u8>,
    sysroot: PathBuf,
    sysroot_identity: String,
    static_source: PathBuf,
    dynamic_source: PathBuf,
    static_target: PathBuf,
    dynamic_target: PathBuf,
    static_identity: String,
    dynamic_identity: String,
    static_lock: String,
    dynamic_lock: String,
}

impl Fixture {
    fn new() -> Self {
        let directory = tempfile::tempdir().expect("temporary fixture directory");
        let cargo = directory.path().join("cargo");
        let script = b"#!/bin/sh\nset -eu\nprintf '%s\\n' \"$@\" > \"$CARGO_TARGET_DIR/argv\"\nprintf 'incremental=%s\\nlto=%s\\n' \"$CARGO_INCREMENTAL\" \"$CARGO_PROFILE_RELEASE_LTO\" > \"$CARGO_TARGET_DIR/environment\"\ncp implementation.rs \"$CARGO_TARGET_DIR/implementation-seen\"\ncase \"$CARGO_TARGET_DIR\" in\n  *static-target) cp static-artifact-source \"$CARGO_TARGET_DIR/artifact.bin\" ;;\n  *dynamic-target) cp dynamic-artifact-source \"$CARGO_TARGET_DIR/artifact.bin\" ;;\n  *) exit 65 ;;\nesac\n";
        write_executable(&cargo, script);
        let sysroot = directory.path().join("sysroot");
        std::fs::create_dir_all(&sysroot).expect("sysroot exists");
        let sysroot_identity_bytes = b"frozen sysroot identity\n";
        std::fs::write(sysroot.join("sysroot.identity"), sysroot_identity_bytes)
            .expect("sysroot identity is written");
        let rustc = directory.path().join("rustc");
        let rustc_verbose_version =
            b"rustc 1.91.0-nightly\nbinary: rustc\ncommit-hash: feedface\n".to_vec();
        let rustc_script = format!(
            "#!/bin/sh\nset -eu\ncase \"$1\" in\n  -vV) printf '%b' 'rustc 1.91.0-nightly\\nbinary: rustc\\ncommit-hash: feedface\\n' ;;\n  --print) [ \"$2\" = sysroot ]; printf '%s\\n' '{}' ;;\n  *) exit 64 ;;\nesac\n",
            sysroot.display()
        );
        write_executable(&rustc, rustc_script.as_bytes());

        let static_source = directory.path().join("static-source");
        let dynamic_source = directory.path().join("dynamic-source");
        std::fs::create_dir_all(&static_source).expect("static source exists");
        std::fs::create_dir_all(&dynamic_source).expect("dynamic source exists");
        let static_identity_bytes = b"shared complete source identity\n";
        let dynamic_identity_bytes = b"shared complete source identity\n";
        let static_lock_bytes = b"shared lock\n";
        let dynamic_lock_bytes = b"shared lock\n";
        std::fs::write(static_source.join("source.identity"), static_identity_bytes)
            .expect("static source identity is written");
        std::fs::write(
            dynamic_source.join("source.identity"),
            dynamic_identity_bytes,
        )
        .expect("dynamic source identity is written");
        std::fs::write(static_source.join("Cargo.lock"), static_lock_bytes)
            .expect("static lock is written");
        std::fs::write(dynamic_source.join("Cargo.lock"), dynamic_lock_bytes)
            .expect("dynamic lock is written");
        for source in [&static_source, &dynamic_source] {
            std::fs::write(source.join("static-artifact-source"), b"static artifact")
                .expect("static artifact input is written");
            std::fs::write(source.join("dynamic-artifact-source"), b"dynamic artifact")
                .expect("dynamic artifact input is written");
            std::fs::write(source.join("implementation.rs"), b"shared implementation\n")
                .expect("implementation input is written");
            std::fs::create_dir_all(source.join("retained-empty-directory"))
                .expect("empty source directory is retained");
        }

        let static_target = directory.path().join("static-target");
        let dynamic_target = directory.path().join("dynamic-target");
        Self {
            cargo_digest: digest(script),
            cargo,
            rustc_digest: digest(rustc_script.as_bytes()),
            rustc,
            rustc_verbose_version,
            sysroot,
            sysroot_identity: digest(sysroot_identity_bytes),
            static_source,
            dynamic_source,
            static_target,
            dynamic_target,
            static_identity: digest(static_identity_bytes),
            dynamic_identity: digest(dynamic_identity_bytes),
            static_lock: digest(static_lock_bytes),
            dynamic_lock: digest(dynamic_lock_bytes),
            _directory: directory,
        }
    }

    fn member(
        &self,
        variant: Variant,
        source_root: &Path,
        source_identity: &str,
        cargo_lock: &str,
        target_root: &Path,
    ) -> BuildPairMemberV1 {
        BuildPairMemberV1 {
            variant,
            source_root: source_root.to_path_buf(),
            source_identity_receipt: PathBuf::from("source.identity"),
            source_identity_blake3: source_identity.to_owned(),
            cargo_lock_blake3: cargo_lock.to_owned(),
            target_root: target_root.to_path_buf(),
            artifact_relative_path: PathBuf::from("artifact.bin"),
            expected_artifact_blake3: match variant {
                Variant::Static => digest(b"static artifact"),
                Variant::Dynamic => digest(b"dynamic artifact"),
            },
        }
    }

    fn plan(&self) -> BuildPairPlanV1 {
        BuildPairPlanV1 {
            scenario: "build-default".to_owned(),
            pair_id: "pair-00".to_owned(),
            source_commit: COMMIT.to_owned(),
            experiment_identity_blake3: IDENTITY.to_owned(),
            cargo_executable: self.cargo.clone(),
            cargo_executable_blake3: self.cargo_digest.clone(),
            rustc_executable: self.rustc.clone(),
            rustc_executable_blake3: self.rustc_digest.clone(),
            rustc_verbose_version: self.rustc_verbose_version.clone(),
            sysroot_root: self.sysroot.clone(),
            sysroot_identity_receipt: PathBuf::from("sysroot.identity"),
            sysroot_identity_blake3: self.sysroot_identity.clone(),
            baseline_arguments: vec![
                "build".to_owned(),
                "--locked".to_owned(),
                "--package".to_owned(),
                "fixture".to_owned(),
            ],
            profile: "release".to_owned(),
            features: vec!["alpha".to_owned(), "beta".to_owned()],
            lto: BuildLtoV1::Thin,
            static_member: self.member(
                Variant::Static,
                &self.static_source,
                &self.static_identity,
                &self.static_lock,
                &self.static_target,
            ),
            dynamic_member: self.member(
                Variant::Dynamic,
                &self.dynamic_source,
                &self.dynamic_identity,
                &self.dynamic_lock,
                &self.dynamic_target,
            ),
        }
    }
}

#[test]
fn one_controller_executes_both_members_with_one_frozen_command_and_environment() {
    let fixture = Fixture::new();
    let report = run_paired_build_v1(&fixture.plan()).expect("paired build validates and executes");

    assert_eq!(
        report.command,
        [
            fixture.cargo.to_str().expect("cargo fixture path is UTF-8"),
            "build",
            "--locked",
            "--package",
            "fixture",
            "--profile",
            "release",
            "--features",
            "alpha,beta",
        ]
    );
    assert_eq!(report.members[0].variant, Variant::Static);
    assert_eq!(report.members[1].variant, Variant::Dynamic);
    assert_eq!(
        report.members[0].artifact_blake3,
        digest(b"static artifact")
    );
    assert_eq!(
        report.members[1].artifact_blake3,
        digest(b"dynamic artifact")
    );
    assert_eq!(
        report.members[0].source_identity_blake3,
        report.members[1].source_identity_blake3
    );
    assert_ne!(
        report.members[0].source_identity_blake3,
        fixture.static_identity
    );
    assert_eq!(
        report.members[0].source_tree_receipt_bytes,
        report.members[1].source_tree_receipt_bytes
    );
    assert_eq!(
        digest(&report.members[0].source_tree_receipt_bytes),
        report.members[0].source_identity_blake3
    );
    assert_eq!(report.cargo_incremental, "1");
    assert_eq!(report.lto, BuildLtoV1::Thin);
    assert_eq!(
        report.inherited_build_environment.get("RUSTC_WRAPPER"),
        Some(&std::env::var("RUSTC_WRAPPER").ok())
    );

    let static_argv = std::fs::read(fixture.static_target.join("argv"))
        .expect("static invocation argv is retained");
    let dynamic_argv = std::fs::read(fixture.dynamic_target.join("argv"))
        .expect("dynamic invocation argv is retained");
    assert_eq!(static_argv, dynamic_argv);
    let static_environment = std::fs::read(fixture.static_target.join("environment"))
        .expect("static invocation environment is retained");
    let dynamic_environment = std::fs::read(fixture.dynamic_target.join("environment"))
        .expect("dynamic invocation environment is retained");
    assert_eq!(static_environment, dynamic_environment);
    for member in &report.members {
        assert_eq!(
            digest(&member.build_receipt_bytes),
            member.build_receipt_blake3
        );
        let receipt: serde_json::Value =
            serde_json::from_slice(&member.build_receipt_bytes).expect("receipt is canonical JSON");
        assert_eq!(receipt["rustc_executable_blake3"], fixture.rustc_digest);
        assert_eq!(receipt["sysroot_identity_blake3"], fixture.sysroot_identity);
        assert_eq!(receipt["cargo_incremental"], "1");
        assert!(receipt.get("active_duration_ns").is_none());
        let mut canonical = Vec::new();
        serde_json_canonicalizer::to_writer(&receipt, &mut canonical)
            .expect("receipt canonicalizes");
        assert_eq!(member.build_receipt_bytes, canonical);

        assert_eq!(
            report.build_duration_use,
            BuildDurationUseV1::DescriptiveNonGating
        );
        assert_eq!(
            digest(&member.build_observation_receipt_bytes),
            member.build_observation_receipt_blake3
        );
        let observation: serde_json::Value =
            serde_json::from_slice(&member.build_observation_receipt_bytes)
                .expect("observation receipt is canonical JSON");
        assert_eq!(
            observation["active_duration_ns"],
            serde_json::json!(member.active_duration_ns)
        );
        assert_eq!(observation["duration_use"], "descriptive_non_gating");
        assert_eq!(
            observation["build_authority_receipt_blake3"],
            member.build_receipt_blake3
        );
    }
    assert_eq!(digest(&report.pair_record_bytes), report.pair_record_blake3);
    let pair_record: serde_json::Value =
        serde_json::from_slice(&report.pair_record_bytes).expect("pair record is canonical JSON");
    assert_eq!(
        pair_record["member_build_receipt_blake3"],
        serde_json::json!([
            report.members[0].build_receipt_blake3,
            report.members[1].build_receipt_blake3
        ])
    );
    assert_eq!(
        pair_record["build_order"],
        serde_json::json!(["static", "dynamic"])
    );
    assert_eq!(pair_record["build_duration_use"], "descriptive_non_gating");
    assert!(
        pair_record
            .get("member_build_observation_receipt_blake3")
            .is_none()
    );
}

#[test]
fn duration_only_observations_leave_build_authority_identity_unchanged() {
    let fixture = Fixture::new();
    let report = run_paired_build_v1(&fixture.plan()).expect("paired build completes");
    let original_authority =
        build_pair_authority_blake3_v1(&report).expect("build authority validates");
    let mut changed_observations = report.clone();

    for (member, duration) in changed_observations
        .members
        .iter_mut()
        .zip([11_u128, 99_u128])
    {
        member.active_duration_ns = duration;
        let mut receipt: serde_json::Value =
            serde_json::from_slice(&member.build_observation_receipt_bytes)
                .expect("observation receipt parses");
        receipt["active_duration_ns"] = serde_json::json!(duration);
        member.build_observation_receipt_bytes =
            serde_json_canonicalizer::to_vec(&receipt).expect("observation canonicalizes");
        member.build_observation_receipt_blake3 = digest(&member.build_observation_receipt_bytes);
    }

    assert_ne!(
        report.members[0].build_observation_receipt_blake3,
        changed_observations.members[0].build_observation_receipt_blake3
    );
    assert_eq!(
        report.members[0].build_receipt_blake3,
        changed_observations.members[0].build_receipt_blake3
    );
    assert_eq!(
        original_authority,
        build_pair_authority_blake3_v1(&changed_observations)
            .expect("duration-only mutation preserves immutable authority")
    );
}

#[test]
fn command_and_environment_indirection_are_rejected_before_either_member_runs() {
    for baseline_arguments in [
        vec![
            "env".to_owned(),
            "RUSTC_WRAPPER=/tmp/evil".to_owned(),
            "cargo".to_owned(),
            "build".to_owned(),
            "--locked".to_owned(),
        ],
        vec![
            "build".to_owned(),
            "--locked".to_owned(),
            "--config".to_owned(),
            "build.rustc-wrapper=/tmp/evil".to_owned(),
        ],
        vec![
            "build".to_owned(),
            "--locked".to_owned(),
            "--features=other".to_owned(),
        ],
    ] {
        let fixture = Fixture::new();
        let mut plan = fixture.plan();
        plan.baseline_arguments = baseline_arguments;

        let error = run_paired_build_v1(&plan).expect_err("indirection must fail closed");
        assert!(
            error.to_string().contains("baseline command"),
            "unexpected refusal: {error}"
        );
        assert!(!fixture.static_target.exists());
        assert!(!fixture.dynamic_target.exists());
    }
}

#[test]
fn source_identity_and_explicit_target_roots_are_validated_before_execution() {
    let fixture = Fixture::new();
    let mut forged_identity = fixture.plan();
    forged_identity.static_member.source_identity_blake3 = IDENTITY.to_owned();
    assert!(run_paired_build_v1(&forged_identity).is_err());
    assert!(!fixture.static_target.exists());
    assert!(!fixture.dynamic_target.exists());

    let fixture = Fixture::new();
    let mut overlapping_targets = fixture.plan();
    overlapping_targets.dynamic_member.target_root = fixture.static_target.join("nested");
    assert!(run_paired_build_v1(&overlapping_targets).is_err());
    assert!(!fixture.static_target.exists());
    assert!(!fixture.dynamic_target.exists());

    let fixture = Fixture::new();
    std::os::unix::fs::symlink(&fixture.static_source, &fixture.static_target)
        .expect("target symlink is created");
    let error = run_paired_build_v1(&fixture.plan()).expect_err("target indirection must fail");
    assert!(error.to_string().contains("target_root"));
    assert!(!fixture.dynamic_target.exists());
}

#[test]
fn both_members_require_one_source_and_lock_authority_before_execution() {
    let fixture = Fixture::new();
    let mut mismatched_source = fixture.plan();
    let other_source_identity = b"other complete source identity\n";
    std::fs::write(
        fixture.dynamic_source.join("source.identity"),
        other_source_identity,
    )
    .expect("different dynamic source identity is written");
    mismatched_source.dynamic_member.source_identity_blake3 = digest(other_source_identity);
    let error = run_paired_build_v1(&mismatched_source)
        .expect_err("different complete source identities must fail closed");
    assert!(error.to_string().contains("source identity"));
    assert!(!fixture.static_target.exists());
    assert!(!fixture.dynamic_target.exists());

    let fixture = Fixture::new();
    let mut mismatched_lock = fixture.plan();
    let other_lock = b"other lock\n";
    std::fs::write(fixture.dynamic_source.join("Cargo.lock"), other_lock)
        .expect("different dynamic lock is written");
    mismatched_lock.dynamic_member.cargo_lock_blake3 = digest(other_lock);
    let error = run_paired_build_v1(&mismatched_lock)
        .expect_err("different Cargo.lock identities must fail closed");
    assert!(error.to_string().contains("Cargo.lock"));
    assert!(!fixture.static_target.exists());
    assert!(!fixture.dynamic_target.exists());
}

#[test]
fn complete_source_tree_must_match_before_either_build_runs() {
    let fixture = Fixture::new();
    std::fs::write(
        fixture.dynamic_source.join("implementation.rs"),
        b"different build-reachable implementation\n",
    )
    .expect("dynamic implementation is changed outside the receipt and lock");

    let error = run_paired_build_v1(&fixture.plan())
        .expect_err("different complete source trees must fail closed");
    assert!(
        error.to_string().contains("complete source tree"),
        "unexpected refusal: {error}"
    );
    assert!(!fixture.static_target.exists());
    assert!(!fixture.dynamic_target.exists());
}

#[test]
fn source_entry_mode_and_empty_directories_are_part_of_tree_identity() {
    use std::os::unix::fs::PermissionsExt as _;

    let fixture = Fixture::new();
    let implementation = fixture.dynamic_source.join("implementation.rs");
    let mut permissions = std::fs::metadata(&implementation)
        .expect("implementation metadata is available")
        .permissions();
    permissions.set_mode(0o755);
    std::fs::set_permissions(&implementation, permissions)
        .expect("dynamic implementation mode changes");
    let error = run_paired_build_v1(&fixture.plan())
        .expect_err("different canonical entry modes must fail closed");
    assert!(error.to_string().contains("complete source tree"));
    assert!(!fixture.static_target.exists());
    assert!(!fixture.dynamic_target.exists());

    let fixture = Fixture::new();
    std::fs::create_dir(fixture.dynamic_source.join("additional-empty-directory"))
        .expect("dynamic-only empty directory is created");
    let error = run_paired_build_v1(&fixture.plan())
        .expect_err("different empty directory census must fail closed");
    assert!(error.to_string().contains("complete source tree"));
    assert!(!fixture.static_target.exists());
    assert!(!fixture.dynamic_target.exists());
}

#[test]
fn source_links_are_refused_before_either_build_runs() {
    use std::os::unix::fs::symlink;

    let fixture = Fixture::new();
    symlink(
        fixture.dynamic_source.join("implementation.rs"),
        fixture.dynamic_source.join("implementation-link.rs"),
    )
    .expect("source symlink is created");

    let error = run_paired_build_v1(&fixture.plan()).expect_err("source links must fail closed");
    assert!(
        error.to_string().contains("source tree") || error.to_string().contains("link"),
        "unexpected refusal: {error}"
    );
    assert!(!fixture.static_target.exists());
    assert!(!fixture.dynamic_target.exists());
}

#[test]
fn compiler_and_sysroot_identity_mismatches_are_rejected_before_execution() {
    let fixture = Fixture::new();
    let mut wrong_version = fixture.plan();
    wrong_version.rustc_verbose_version = b"rustc forged\n".to_vec();
    let error = run_paired_build_v1(&wrong_version).expect_err("rustc identity must match");
    assert!(error.to_string().contains("rustc -vV"));
    assert!(!fixture.static_target.exists());
    assert!(!fixture.dynamic_target.exists());

    let fixture = Fixture::new();
    let mut wrong_sysroot = fixture.plan();
    wrong_sysroot.sysroot_identity_blake3 = IDENTITY.to_owned();
    let error = run_paired_build_v1(&wrong_sysroot).expect_err("sysroot identity must match");
    assert!(error.to_string().contains("sysroot identity"));
    assert!(!fixture.static_target.exists());
    assert!(!fixture.dynamic_target.exists());
}

#[test]
fn compiler_identity_is_revalidated_between_pair_members() {
    let fixture = Fixture::new();
    let mut plan = fixture.plan();
    let mutating_cargo = format!(
        "#!/bin/sh\nset -eu\nprintf '%s\\n' \"$@\" > \"$CARGO_TARGET_DIR/argv\"\nprintf 'incremental=%s\\nlto=%s\\n' \"$CARGO_INCREMENTAL\" \"$CARGO_PROFILE_RELEASE_LTO\" > \"$CARGO_TARGET_DIR/environment\"\ncase \"$CARGO_TARGET_DIR\" in\n  *static-target) cp static-artifact-source \"$CARGO_TARGET_DIR/artifact.bin\" ;;\n  *dynamic-target) cp dynamic-artifact-source \"$CARGO_TARGET_DIR/artifact.bin\" ;;\n  *) exit 65 ;;\nesac\nprintf '# mutation\\n' >> '{}'\n",
        fixture.rustc.display()
    );
    write_executable(&fixture.cargo, mutating_cargo.as_bytes());
    plan.cargo_executable_blake3 = digest(mutating_cargo.as_bytes());

    let error = run_paired_build_v1(&plan).expect_err("toolchain mutation must fail the pair");
    assert!(
        error
            .to_string()
            .contains("rustc_executable digest mismatch")
    );
    assert!(fixture.static_target.exists());
    assert!(!fixture.dynamic_target.exists());
}

#[test]
fn prebound_artifact_identity_rejects_build_substitution_before_second_member() {
    let fixture = Fixture::new();
    let mut plan = fixture.plan();
    plan.static_member.expected_artifact_blake3 = IDENTITY.to_owned();

    let error = run_paired_build_v1(&plan).expect_err("substituted build must fail closed");
    assert!(error.to_string().contains("artifact digest mismatch"));
    assert!(fixture.static_target.exists());
    assert!(!fixture.dynamic_target.exists());
}
