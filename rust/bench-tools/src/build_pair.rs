// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Authoritative same-process paired build measurement.

use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use serde::Serialize;

use crate::plugin_stats::Variant;

const RECORDED_BUILD_ENVIRONMENT: &[&str] = &[
    "CARGO_BUILD_JOBS",
    "CC",
    "CXX",
    "PATH",
    "RUSTC",
    "RUSTC_WRAPPER",
    "RUSTC_WORKSPACE_WRAPPER",
    "RUSTFLAGS",
    "SCCACHE_CACHE_SIZE",
    "SCCACHE_DIR",
    "SCCACHE_IDLE_TIMEOUT",
];

/// Failure returned while validating or running an authoritative build pair.
#[derive(Debug)]
pub struct BuildPairError(String);

impl BuildPairError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for BuildPairError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for BuildPairError {}

/// Cargo link-time optimization setting frozen across both builds.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BuildLtoV1 {
    /// Disable link-time optimization.
    Off,
    /// Use thin link-time optimization.
    Thin,
    /// Use fat link-time optimization.
    Fat,
}

impl BuildLtoV1 {
    fn cargo_value(self) -> &'static str {
        match self {
            Self::Off => "false",
            Self::Thin => "thin",
            Self::Fat => "fat",
        }
    }
}

/// One member of an authoritative paired build.
#[derive(Clone, Debug)]
pub struct BuildPairMemberV1 {
    /// Required member role.
    pub variant: Variant,
    /// Canonical source root used as Cargo's working directory.
    pub source_root: PathBuf,
    /// Relative path to the frozen source identity receipt.
    pub source_identity_receipt: PathBuf,
    /// Expected BLAKE3 digest of the source identity receipt bytes.
    pub source_identity_blake3: String,
    /// Expected BLAKE3 digest of the source root Cargo.lock bytes.
    pub cargo_lock_blake3: String,
    /// Explicit, isolated Cargo target root.
    pub target_root: PathBuf,
    /// Artifact path relative to the target root.
    pub artifact_relative_path: PathBuf,
    /// Artifact digest pre-bound by the experiment identity.
    pub expected_artifact_blake3: String,
}

/// Fully frozen input for one same-process paired build.
#[derive(Clone, Debug)]
pub struct BuildPairPlanV1 {
    /// Scenario identifier shared by both members.
    pub scenario: String,
    /// Pair identifier shared by both members.
    pub pair_id: String,
    /// Exact source commit shared by both members.
    pub source_commit: String,
    /// Digest of the experiment identity shared by both members.
    pub experiment_identity_blake3: String,
    /// Direct absolute path to Cargo; command indirection is forbidden.
    pub cargo_executable: PathBuf,
    /// Expected digest of the Cargo executable bytes.
    pub cargo_executable_blake3: String,
    /// Direct canonical rustc executable selected for both builds.
    pub rustc_executable: PathBuf,
    /// Expected digest of the rustc executable bytes.
    pub rustc_executable_blake3: String,
    /// Exact expected standard output from rustc -vV.
    pub rustc_verbose_version: Vec<u8>,
    /// Canonical sysroot reported by the selected rustc.
    pub sysroot_root: PathBuf,
    /// Relative path to the frozen sysroot identity receipt.
    pub sysroot_identity_receipt: PathBuf,
    /// Expected digest of the sysroot identity receipt bytes.
    pub sysroot_identity_blake3: String,
    /// Frozen Cargo arguments before controller-owned parity arguments.
    pub baseline_arguments: Vec<String>,
    /// Cargo profile frozen across both builds.
    pub profile: String,
    /// Sorted unique Cargo features frozen across both builds.
    pub features: Vec<String>,
    /// Link-time optimization setting frozen across both builds.
    pub lto: BuildLtoV1,
    /// Static member, executed first.
    pub static_member: BuildPairMemberV1,
    /// Dynamic member, executed second in the same process.
    pub dynamic_member: BuildPairMemberV1,
}

/// Evidence for one member produced by the paired controller.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct BuildPairMemberReportV1 {
    /// Member role.
    pub variant: Variant,
    /// Revalidated source identity digest.
    pub source_identity_blake3: String,
    /// Revalidated Cargo.lock digest.
    pub cargo_lock_blake3: String,
    /// Explicit target root used for the member.
    pub target_root: String,
    /// Artifact path relative to the target root.
    pub artifact_relative_path: String,
    /// Digest of the produced artifact bytes.
    pub artifact_blake3: String,
    /// Monotonic active build duration.
    pub active_duration_ns: u128,
    /// Canonical JCS build receipt bytes.
    pub build_receipt_bytes: Vec<u8>,
    /// Digest of the canonical build receipt bytes.
    pub build_receipt_blake3: String,
}

/// Authoritative evidence emitted for the complete paired build transaction.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct BuildPairReportV1 {
    /// Schema identifier.
    pub schema_version: String,
    /// Scenario identifier.
    pub scenario: String,
    /// Pair identifier.
    pub pair_id: String,
    /// Shared source commit.
    pub source_commit: String,
    /// Shared experiment identity digest.
    pub experiment_identity_blake3: String,
    /// Digest of the direct Cargo executable.
    pub cargo_executable_blake3: String,
    /// Digest of the direct rustc executable.
    pub rustc_executable_blake3: String,
    /// Exact standard output captured from rustc -vV.
    pub rustc_verbose_version: Vec<u8>,
    /// Digest of the exact rustc -vV output bytes.
    pub rustc_verbose_version_blake3: String,
    /// Canonical sysroot path reported by rustc.
    pub sysroot_root: String,
    /// Digest of the frozen sysroot identity receipt.
    pub sysroot_identity_blake3: String,
    /// Literal Cargo command arguments used for both members.
    pub command: Vec<String>,
    /// Shared Cargo profile.
    pub profile: String,
    /// Shared sorted features.
    pub features: Vec<String>,
    /// Shared LTO setting.
    pub lto: BuildLtoV1,
    /// Incremental compilation value enforced for both members.
    pub cargo_incremental: String,
    /// Digest of the complete inherited environment frozen for both children.
    pub inherited_environment_blake3: String,
    /// Existing compiler, wrapper, cache, and related settings as observed.
    pub inherited_build_environment: BTreeMap<String, Option<String>>,
    /// Static then dynamic member evidence.
    pub members: [BuildPairMemberReportV1; 2],
    /// Canonical pair record binding both member receipts and shared authority.
    pub pair_record_bytes: Vec<u8>,
    /// Digest of the canonical pair record bytes.
    pub pair_record_blake3: String,
}

#[derive(Serialize)]
struct CanonicalBuildReceiptV1<'a> {
    schema_version: &'static str,
    scenario: &'a str,
    pair_id: &'a str,
    variant: Variant,
    source_commit: &'a str,
    experiment_identity_blake3: &'a str,
    source_identity_blake3: &'a str,
    cargo_lock_blake3: &'a str,
    cargo_executable_blake3: &'a str,
    rustc_executable_blake3: &'a str,
    rustc_verbose_version: &'a [u8],
    rustc_verbose_version_blake3: &'a str,
    sysroot_root: &'a str,
    sysroot_identity_blake3: &'a str,
    inherited_environment_blake3: &'a str,
    command: &'a [String],
    profile: &'a str,
    features: &'a [String],
    lto: BuildLtoV1,
    cargo_incremental: &'static str,
    target_root: &'a str,
    artifact_relative_path: &'a str,
    artifact_blake3: &'a str,
    active_duration_ns: u128,
}

#[derive(Serialize)]
struct CanonicalPairRecordV1<'a> {
    schema_version: &'static str,
    scenario: &'a str,
    pair_id: &'a str,
    source_commit: &'a str,
    experiment_identity_blake3: &'a str,
    cargo_executable_blake3: &'a str,
    rustc_executable_blake3: &'a str,
    rustc_verbose_version_blake3: &'a str,
    sysroot_identity_blake3: &'a str,
    inherited_environment_blake3: &'a str,
    command: &'a [String],
    profile: &'a str,
    features: &'a [String],
    lto: BuildLtoV1,
    cargo_incremental: &'static str,
    member_source_identity_blake3: [&'a str; 2],
    member_cargo_lock_blake3: [&'a str; 2],
    member_target_roots: [&'a str; 2],
    member_build_receipt_blake3: [&'a str; 2],
}

/// Validate and run both builds under one frozen same-process authority.
pub fn run_paired_build_v1(plan: &BuildPairPlanV1) -> Result<BuildPairReportV1, BuildPairError> {
    validate_plan(plan)?;
    let inherited_environment = capture_environment()?;
    validate_toolchain(plan, &inherited_environment)?;
    validate_member_identity(&plan.static_member)?;
    validate_member_identity(&plan.dynamic_member)?;
    let inherited_environment_blake3 = canonical_digest(&inherited_environment)?;
    let inherited_build_environment = RECORDED_BUILD_ENVIRONMENT
        .iter()
        .map(|name| {
            (
                (*name).to_owned(),
                inherited_environment.get(*name).cloned(),
            )
        })
        .collect();
    let arguments = generated_arguments(plan);
    let mut command = vec![path_text(&plan.cargo_executable, "cargo_executable")?];
    command.extend(arguments.iter().cloned());
    let lto_environment_name = profile_lto_environment_name(&plan.profile);
    let rustc_verbose_version_blake3 = digest(&plan.rustc_verbose_version);
    let sysroot_root = path_text(&plan.sysroot_root, "sysroot_root")?;

    let static_report = run_member(
        plan,
        &plan.static_member,
        &arguments,
        &inherited_environment,
        &lto_environment_name,
        &command,
        &inherited_environment_blake3,
        &rustc_verbose_version_blake3,
        &sysroot_root,
    )?;
    validate_cargo_executable(plan)?;
    validate_toolchain(plan, &inherited_environment)?;
    validate_member_identity(&plan.static_member)?;
    validate_member_identity(&plan.dynamic_member)?;
    let dynamic_report = run_member(
        plan,
        &plan.dynamic_member,
        &arguments,
        &inherited_environment,
        &lto_environment_name,
        &command,
        &inherited_environment_blake3,
        &rustc_verbose_version_blake3,
        &sysroot_root,
    )?;
    validate_cargo_executable(plan)?;
    validate_toolchain(plan, &inherited_environment)?;
    validate_member_identity(&plan.static_member)?;
    validate_member_identity(&plan.dynamic_member)?;

    let pair_record_bytes = canonical_bytes(&CanonicalPairRecordV1 {
        schema_version: "plugin_build_pair_record/v1",
        scenario: &plan.scenario,
        pair_id: &plan.pair_id,
        source_commit: &plan.source_commit,
        experiment_identity_blake3: &plan.experiment_identity_blake3,
        cargo_executable_blake3: &plan.cargo_executable_blake3,
        rustc_executable_blake3: &plan.rustc_executable_blake3,
        rustc_verbose_version_blake3: &rustc_verbose_version_blake3,
        sysroot_identity_blake3: &plan.sysroot_identity_blake3,
        inherited_environment_blake3: &inherited_environment_blake3,
        command: &command,
        profile: &plan.profile,
        features: &plan.features,
        lto: plan.lto,
        cargo_incremental: "1",
        member_source_identity_blake3: [
            &plan.static_member.source_identity_blake3,
            &plan.dynamic_member.source_identity_blake3,
        ],
        member_cargo_lock_blake3: [
            &plan.static_member.cargo_lock_blake3,
            &plan.dynamic_member.cargo_lock_blake3,
        ],
        member_target_roots: [&static_report.target_root, &dynamic_report.target_root],
        member_build_receipt_blake3: [
            &static_report.build_receipt_blake3,
            &dynamic_report.build_receipt_blake3,
        ],
    })?;
    let pair_record_blake3 = digest(&pair_record_bytes);

    Ok(BuildPairReportV1 {
        schema_version: "plugin_build_pair/v1".to_owned(),
        scenario: plan.scenario.clone(),
        pair_id: plan.pair_id.clone(),
        source_commit: plan.source_commit.clone(),
        experiment_identity_blake3: plan.experiment_identity_blake3.clone(),
        cargo_executable_blake3: plan.cargo_executable_blake3.clone(),
        rustc_executable_blake3: plan.rustc_executable_blake3.clone(),
        rustc_verbose_version: plan.rustc_verbose_version.clone(),
        rustc_verbose_version_blake3,
        sysroot_root,
        sysroot_identity_blake3: plan.sysroot_identity_blake3.clone(),
        command,
        profile: plan.profile.clone(),
        features: plan.features.clone(),
        lto: plan.lto,
        cargo_incremental: "1".to_owned(),
        inherited_environment_blake3,
        inherited_build_environment,
        members: [static_report, dynamic_report],
        pair_record_bytes,
        pair_record_blake3,
    })
}

fn validate_plan(plan: &BuildPairPlanV1) -> Result<(), BuildPairError> {
    validate_identifier("scenario", &plan.scenario)?;
    validate_identifier("pair_id", &plan.pair_id)?;
    if !is_lower_hex(&plan.source_commit, 40) {
        return Err(BuildPairError::new(
            "source_commit must be 40 lowercase hexadecimal characters",
        ));
    }
    validate_digest(
        "experiment_identity_blake3",
        &plan.experiment_identity_blake3,
    )?;
    validate_digest("cargo_executable_blake3", &plan.cargo_executable_blake3)?;
    validate_digest("rustc_executable_blake3", &plan.rustc_executable_blake3)?;
    validate_digest("sysroot_identity_blake3", &plan.sysroot_identity_blake3)?;
    if plan.rustc_verbose_version.is_empty() {
        return Err(BuildPairError::new("rustc -vV bytes must be nonempty"));
    }
    validate_relative_path(&plan.sysroot_identity_receipt, "sysroot_identity_receipt")?;
    canonical_directory(&plan.sysroot_root, "sysroot_root")?;
    validate_profile(&plan.profile)?;
    validate_features(&plan.features)?;
    validate_baseline_arguments(&plan.baseline_arguments)?;
    validate_cargo_executable(plan)?;
    validate_rustc_executable(plan)?;
    validate_member(&plan.static_member, Variant::Static)?;
    validate_member(&plan.dynamic_member, Variant::Dynamic)?;

    let static_source = canonical_directory(&plan.static_member.source_root, "static source_root")?;
    let dynamic_source =
        canonical_directory(&plan.dynamic_member.source_root, "dynamic source_root")?;
    let static_target =
        validated_target_root(&plan.static_member.target_root, "static target_root")?;
    let dynamic_target =
        validated_target_root(&plan.dynamic_member.target_root, "dynamic target_root")?;
    ensure_disjoint(&static_source, &dynamic_source, "source roots")?;
    ensure_disjoint(&static_target, &dynamic_target, "target roots")?;
    for (target, target_name) in [
        (&static_target, "static target_root"),
        (&dynamic_target, "dynamic target_root"),
    ] {
        ensure_disjoint(target, &static_source, target_name)?;
        ensure_disjoint(target, &dynamic_source, target_name)?;
    }
    Ok(())
}

fn validate_rustc_executable(plan: &BuildPairPlanV1) -> Result<(), BuildPairError> {
    let rustc = normalized_absolute(&plan.rustc_executable, "rustc_executable")?;
    if rustc.file_name().and_then(|name| name.to_str()) != Some("rustc") {
        return Err(BuildPairError::new(
            "rustc_executable must directly name rustc",
        ));
    }
    let canonical = fs::canonicalize(&rustc).map_err(|error| {
        BuildPairError::new(format!("cannot resolve rustc_executable: {error}"))
    })?;
    if canonical != rustc {
        return Err(BuildPairError::new(
            "rustc_executable must be its canonical absolute path",
        ));
    }
    let bytes = fs::read(&rustc)
        .map_err(|error| BuildPairError::new(format!("cannot read rustc_executable: {error}")))?;
    if digest(&bytes) != plan.rustc_executable_blake3 {
        return Err(BuildPairError::new("rustc_executable digest mismatch"));
    }
    Ok(())
}

fn validate_toolchain(
    plan: &BuildPairPlanV1,
    inherited_environment: &BTreeMap<String, String>,
) -> Result<(), BuildPairError> {
    validate_rustc_executable(plan)?;
    let verbose = run_identity_command(
        &plan.rustc_executable,
        &["-vV"],
        inherited_environment,
        "rustc -vV",
    )?;
    if verbose != plan.rustc_verbose_version {
        return Err(BuildPairError::new("rustc -vV identity mismatch"));
    }

    let canonical_sysroot = canonical_directory(&plan.sysroot_root, "sysroot_root")?;
    let reported_sysroot = run_identity_command(
        &plan.rustc_executable,
        &["--print", "sysroot"],
        inherited_environment,
        "rustc --print sysroot",
    )?;
    let mut expected_sysroot = path_text(&canonical_sysroot, "sysroot_root")?.into_bytes();
    expected_sysroot.push(b'\n');
    if reported_sysroot != expected_sysroot {
        return Err(BuildPairError::new(
            "rustc reported sysroot identity mismatch",
        ));
    }
    let receipt_path = canonical_sysroot.join(&plan.sysroot_identity_receipt);
    let receipt = fs::read(&receipt_path).map_err(|error| {
        BuildPairError::new(format!(
            "cannot read sysroot identity receipt {}: {error}",
            receipt_path.display()
        ))
    })?;
    if digest(&receipt) != plan.sysroot_identity_blake3 {
        return Err(BuildPairError::new("sysroot identity digest mismatch"));
    }
    Ok(())
}

fn run_identity_command(
    executable: &Path,
    arguments: &[&str],
    inherited_environment: &BTreeMap<String, String>,
    label: &str,
) -> Result<Vec<u8>, BuildPairError> {
    let mut command = Command::new(executable);
    command.args(arguments).env_clear();
    for (name, value) in inherited_environment {
        command.env(name, value);
    }
    let output = command
        .output()
        .map_err(|error| BuildPairError::new(format!("cannot execute {label}: {error}")))?;
    if !output.status.success() {
        return Err(BuildPairError::new(format!(
            "{label} exited with {}",
            output.status
        )));
    }
    Ok(output.stdout)
}

fn validate_cargo_executable(plan: &BuildPairPlanV1) -> Result<(), BuildPairError> {
    let cargo = normalized_absolute(&plan.cargo_executable, "cargo_executable")?;
    if cargo.file_name().and_then(|name| name.to_str()) != Some("cargo") {
        return Err(BuildPairError::new(
            "cargo_executable must directly name cargo",
        ));
    }
    let canonical = fs::canonicalize(&cargo).map_err(|error| {
        BuildPairError::new(format!("cannot resolve cargo_executable: {error}"))
    })?;
    if canonical != cargo {
        return Err(BuildPairError::new(
            "cargo_executable must be its canonical absolute path",
        ));
    }
    let bytes = fs::read(&cargo)
        .map_err(|error| BuildPairError::new(format!("cannot read cargo_executable: {error}")))?;
    if digest(&bytes) != plan.cargo_executable_blake3 {
        return Err(BuildPairError::new("cargo_executable digest mismatch"));
    }
    Ok(())
}

fn validate_member(member: &BuildPairMemberV1, required: Variant) -> Result<(), BuildPairError> {
    if member.variant != required {
        return Err(BuildPairError::new(format!(
            "paired build member role must be {required:?}"
        )));
    }
    validate_digest("source_identity_blake3", &member.source_identity_blake3)?;
    validate_digest("cargo_lock_blake3", &member.cargo_lock_blake3)?;
    validate_digest("expected_artifact_blake3", &member.expected_artifact_blake3)?;
    validate_relative_path(&member.source_identity_receipt, "source_identity_receipt")?;
    validate_relative_path(&member.artifact_relative_path, "artifact_relative_path")?;
    canonical_directory(&member.source_root, "source_root")?;
    validated_target_root(&member.target_root, "target_root")?;
    Ok(())
}

fn validate_member_identity(member: &BuildPairMemberV1) -> Result<(), BuildPairError> {
    let identity_path = member.source_root.join(&member.source_identity_receipt);
    let identity_bytes = fs::read(&identity_path).map_err(|error| {
        BuildPairError::new(format!(
            "cannot read source identity {}: {error}",
            identity_path.display()
        ))
    })?;
    if digest(&identity_bytes) != member.source_identity_blake3 {
        return Err(BuildPairError::new(format!(
            "{:?} source identity digest mismatch",
            member.variant
        )));
    }
    let lock_path = member.source_root.join("Cargo.lock");
    let lock_bytes = fs::read(&lock_path).map_err(|error| {
        BuildPairError::new(format!("cannot read {}: {error}", lock_path.display()))
    })?;
    if digest(&lock_bytes) != member.cargo_lock_blake3 {
        return Err(BuildPairError::new(format!(
            "{:?} Cargo.lock digest mismatch",
            member.variant
        )));
    }
    Ok(())
}

fn run_member(
    plan: &BuildPairPlanV1,
    member: &BuildPairMemberV1,
    arguments: &[String],
    inherited_environment: &BTreeMap<String, String>,
    lto_environment_name: &str,
    report_command: &[String],
    inherited_environment_blake3: &str,
    rustc_verbose_version_blake3: &str,
    sysroot_root: &str,
) -> Result<BuildPairMemberReportV1, BuildPairError> {
    validated_target_root(&member.target_root, "target_root")?;
    fs::create_dir_all(&member.target_root).map_err(|error| {
        BuildPairError::new(format!(
            "cannot create target root {}: {error}",
            member.target_root.display()
        ))
    })?;
    canonical_directory(&member.target_root, "target_root")?;
    let mut command = Command::new(&plan.cargo_executable);
    command
        .args(arguments)
        .current_dir(&member.source_root)
        .env_clear();
    for (name, value) in inherited_environment {
        command.env(name, value);
    }
    command
        .env("CARGO_INCREMENTAL", "1")
        .env("CARGO_TARGET_DIR", &member.target_root)
        .env("RUSTC", &plan.rustc_executable)
        .env(lto_environment_name, plan.lto.cargo_value());

    let start = Instant::now();
    let status = command.status().map_err(|error| {
        BuildPairError::new(format!(
            "cannot execute {:?} build: {error}",
            member.variant
        ))
    })?;
    let elapsed = start.elapsed().as_nanos();
    if !status.success() {
        return Err(BuildPairError::new(format!(
            "{:?} build exited with {status}",
            member.variant
        )));
    }
    if elapsed == 0 {
        return Err(BuildPairError::new(format!(
            "{:?} build duration must be positive",
            member.variant
        )));
    }
    let artifact_path = member.target_root.join(&member.artifact_relative_path);
    canonical_directory(&member.target_root, "target_root")?;
    let artifact = fs::read(&artifact_path).map_err(|error| {
        BuildPairError::new(format!(
            "cannot read {:?} artifact {}: {error}",
            member.variant,
            artifact_path.display()
        ))
    })?;

    let target_root = path_text(&member.target_root, "target_root")?;
    let artifact_relative_path =
        path_text(&member.artifact_relative_path, "artifact_relative_path")?;
    let artifact_blake3 = digest(&artifact);
    if artifact_blake3 != member.expected_artifact_blake3 {
        return Err(BuildPairError::new(format!(
            "{:?} artifact digest mismatch",
            member.variant
        )));
    }
    let build_receipt_bytes = canonical_bytes(&CanonicalBuildReceiptV1 {
        schema_version: "plugin_build_receipt/v1",
        scenario: &plan.scenario,
        pair_id: &plan.pair_id,
        variant: member.variant,
        source_commit: &plan.source_commit,
        experiment_identity_blake3: &plan.experiment_identity_blake3,
        source_identity_blake3: &member.source_identity_blake3,
        cargo_lock_blake3: &member.cargo_lock_blake3,
        cargo_executable_blake3: &plan.cargo_executable_blake3,
        rustc_executable_blake3: &plan.rustc_executable_blake3,
        rustc_verbose_version: &plan.rustc_verbose_version,
        rustc_verbose_version_blake3,
        sysroot_root,
        sysroot_identity_blake3: &plan.sysroot_identity_blake3,
        inherited_environment_blake3,
        command: report_command,
        profile: &plan.profile,
        features: &plan.features,
        lto: plan.lto,
        cargo_incremental: "1",
        target_root: &target_root,
        artifact_relative_path: &artifact_relative_path,
        artifact_blake3: &artifact_blake3,
        active_duration_ns: elapsed,
    })?;
    let build_receipt_blake3 = digest(&build_receipt_bytes);

    Ok(BuildPairMemberReportV1 {
        variant: member.variant,
        source_identity_blake3: member.source_identity_blake3.clone(),
        cargo_lock_blake3: member.cargo_lock_blake3.clone(),
        target_root,
        artifact_relative_path,
        artifact_blake3,
        active_duration_ns: elapsed,
        build_receipt_bytes,
        build_receipt_blake3,
    })
}

fn generated_arguments(plan: &BuildPairPlanV1) -> Vec<String> {
    let mut arguments = plan.baseline_arguments.clone();
    arguments.push("--profile".to_owned());
    arguments.push(plan.profile.clone());
    if !plan.features.is_empty() {
        arguments.push("--features".to_owned());
        arguments.push(plan.features.join(","));
    }
    arguments
}

fn capture_environment() -> Result<BTreeMap<String, String>, BuildPairError> {
    std::env::vars_os()
        .map(|(name, value)| {
            let name = name.into_string().map_err(|_| {
                BuildPairError::new("inherited environment contains a non-UTF-8 name")
            })?;
            let value = value.into_string().map_err(|_| {
                BuildPairError::new(format!(
                    "inherited environment value for {name} is not UTF-8"
                ))
            })?;
            Ok((name, value))
        })
        .collect()
}

fn canonical_digest<T: Serialize>(value: &T) -> Result<String, BuildPairError> {
    Ok(digest(&canonical_bytes(value)?))
}

fn canonical_bytes<T: Serialize>(value: &T) -> Result<Vec<u8>, BuildPairError> {
    let mut bytes = Vec::new();
    serde_json_canonicalizer::to_writer(value, &mut bytes).map_err(|error| {
        BuildPairError::new(format!(
            "cannot canonicalize build authority input: {error}"
        ))
    })?;
    Ok(bytes)
}

fn validate_baseline_arguments(arguments: &[String]) -> Result<(), BuildPairError> {
    if arguments.first().map(String::as_str) != Some("build") {
        return Err(BuildPairError::new(
            "baseline command must be the direct Cargo build subcommand",
        ));
    }
    if arguments
        .iter()
        .filter(|argument| argument.as_str() == "--locked")
        .count()
        != 1
    {
        return Err(BuildPairError::new(
            "baseline command must contain --locked exactly once",
        ));
    }
    let forbidden = [
        "--all-features",
        "--config",
        "--features",
        "--manifest-path",
        "--no-default-features",
        "--profile",
        "--release",
        "--target-dir",
        "-F",
    ];
    for argument in arguments {
        if argument.is_empty() || argument.contains('\0') {
            return Err(BuildPairError::new(
                "baseline command contains an invalid argument",
            ));
        }
        if argument != "--locked"
            && (argument == "env"
                || (!argument.starts_with('-') && argument.contains('='))
                || forbidden.iter().any(|option| {
                    argument == option || argument.starts_with(&format!("{option}="))
                }))
        {
            return Err(BuildPairError::new(format!(
                "baseline command may not override controller authority: {argument}"
            )));
        }
    }
    Ok(())
}

fn validate_features(features: &[String]) -> Result<(), BuildPairError> {
    let mut previous: Option<&str> = None;
    for feature in features {
        if feature.is_empty()
            || feature.contains(',')
            || feature.contains('\0')
            || feature.chars().any(char::is_whitespace)
        {
            return Err(BuildPairError::new(
                "feature names must be nonempty Cargo feature tokens",
            ));
        }
        if previous.is_some_and(|value| value >= feature.as_str()) {
            return Err(BuildPairError::new("features must be sorted and unique"));
        }
        previous = Some(feature);
    }
    Ok(())
}

fn validate_profile(profile: &str) -> Result<(), BuildPairError> {
    if profile.is_empty()
        || profile.contains('\0')
        || !profile
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || matches!(character, '-' | '_'))
    {
        return Err(BuildPairError::new(
            "profile must be a nonempty Cargo profile token",
        ));
    }
    Ok(())
}

fn profile_lto_environment_name(profile: &str) -> String {
    let normalized = profile
        .chars()
        .map(|character| match character {
            '-' => '_',
            other => other.to_ascii_uppercase(),
        })
        .collect::<String>();
    format!("CARGO_PROFILE_{normalized}_LTO")
}

fn validate_identifier(name: &str, value: &str) -> Result<(), BuildPairError> {
    if value.is_empty() || value.contains('\0') {
        return Err(BuildPairError::new(format!(
            "{name} must be a nonempty string without NUL"
        )));
    }
    Ok(())
}

fn validate_digest(name: &str, value: &str) -> Result<(), BuildPairError> {
    if !value
        .strip_prefix("blake3:")
        .is_some_and(|hex| is_lower_hex(hex, 64))
    {
        return Err(BuildPairError::new(format!(
            "{name} must be a 64-character lowercase BLAKE3 digest"
        )));
    }
    Ok(())
}

fn is_lower_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn digest(bytes: &[u8]) -> String {
    format!("blake3:{}", blake3::hash(bytes))
}

fn validate_relative_path(path: &Path, name: &str) -> Result<(), BuildPairError> {
    if path.as_os_str().is_empty()
        || path.is_absolute()
        || path.components().any(|component| {
            !matches!(component, Component::Normal(_)) || component.as_os_str().to_str().is_none()
        })
    {
        return Err(BuildPairError::new(format!(
            "{name} must be a normalized UTF-8 relative path"
        )));
    }
    Ok(())
}

fn canonical_directory(path: &Path, name: &str) -> Result<PathBuf, BuildPairError> {
    let normalized = normalized_absolute(path, name)?;
    let canonical = fs::canonicalize(&normalized)
        .map_err(|error| BuildPairError::new(format!("cannot resolve {name}: {error}")))?;
    if canonical != normalized || !canonical.is_dir() {
        return Err(BuildPairError::new(format!(
            "{name} must be an existing canonical directory"
        )));
    }
    Ok(canonical)
}

fn normalized_absolute(path: &Path, name: &str) -> Result<PathBuf, BuildPairError> {
    if !path.is_absolute()
        || path.as_os_str().is_empty()
        || path
            .components()
            .any(|component| matches!(component, Component::CurDir | Component::ParentDir))
        || path.to_str().is_none()
    {
        return Err(BuildPairError::new(format!(
            "{name} must be a normalized UTF-8 absolute path"
        )));
    }
    Ok(path.to_path_buf())
}

fn validated_target_root(path: &Path, name: &str) -> Result<PathBuf, BuildPairError> {
    let normalized = normalized_absolute(path, name)?;
    match fs::symlink_metadata(&normalized) {
        Ok(metadata) => {
            if metadata.file_type().is_symlink() || !metadata.is_dir() {
                return Err(BuildPairError::new(format!(
                    "{name} must directly name a directory"
                )));
            }
            let canonical = fs::canonicalize(&normalized)
                .map_err(|error| BuildPairError::new(format!("cannot resolve {name}: {error}")))?;
            if canonical != normalized {
                return Err(BuildPairError::new(format!(
                    "{name} must be a canonical path without indirection"
                )));
            }
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            let parent = normalized.parent().ok_or_else(|| {
                BuildPairError::new(format!("{name} must have an existing canonical parent"))
            })?;
            let canonical_parent = fs::canonicalize(parent).map_err(|parent_error| {
                BuildPairError::new(format!("cannot resolve {name} parent: {parent_error}"))
            })?;
            if canonical_parent != parent || !canonical_parent.is_dir() {
                return Err(BuildPairError::new(format!(
                    "{name} parent must be a canonical directory"
                )));
            }
        }
        Err(error) => {
            return Err(BuildPairError::new(format!(
                "cannot inspect {name}: {error}"
            )));
        }
    }
    Ok(normalized)
}

fn ensure_disjoint(first: &Path, second: &Path, description: &str) -> Result<(), BuildPairError> {
    if paths_overlap(first, second) {
        return Err(BuildPairError::new(format!(
            "paired build {description} must not overlap"
        )));
    }
    Ok(())
}

fn paths_overlap(first: &Path, second: &Path) -> bool {
    first.starts_with(second) || second.starts_with(first)
}

fn path_text(path: &Path, name: &str) -> Result<String, BuildPairError> {
    path.to_str()
        .map(ToOwned::to_owned)
        .ok_or_else(|| BuildPairError::new(format!("{name} is not UTF-8")))
}
