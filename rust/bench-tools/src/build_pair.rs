// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Authoritative same-process paired build measurement.

use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use serde::{Deserialize, Serialize};

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

/// Normative use of controller-observed paired build durations.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BuildDurationUseV1 {
    /// Retain durations as post-build evidence without comparing or gating on them.
    DescriptiveNonGating,
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
    /// Digest of the canonical complete source-tree census.
    pub source_identity_blake3: String,
    /// Canonical JCS census covering the complete source tree.
    pub source_tree_receipt_bytes: Vec<u8>,
    /// Revalidated Cargo.lock digest.
    pub cargo_lock_blake3: String,
    /// Explicit target root used for the member.
    pub target_root: String,
    /// Artifact path relative to the target root.
    pub artifact_relative_path: String,
    /// Digest of the produced artifact bytes.
    pub artifact_blake3: String,
    /// Monotonic active build duration retained as descriptive post-build evidence.
    pub active_duration_ns: u128,
    /// Canonical JCS immutable build-authority receipt bytes.
    pub build_receipt_bytes: Vec<u8>,
    /// Digest of the immutable build-authority receipt bytes.
    pub build_receipt_blake3: String,
    /// Canonical JCS post-build observation receipt bytes.
    pub build_observation_receipt_bytes: Vec<u8>,
    /// Digest of the post-build observation receipt bytes.
    pub build_observation_receipt_blake3: String,
}

/// Authoritative evidence emitted for the complete paired build transaction.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct BuildPairReportV1 {
    #[serde(skip)]
    pub(crate) authority: BuildPairAuthorityV1,
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
    /// Frozen execution order of the paired builds.
    pub build_order: [Variant; 2],
    /// Normative non-gating use of observed build durations.
    pub build_duration_use: BuildDurationUseV1,
    /// Incremental compilation value enforced for both members.
    pub cargo_incremental: String,
    /// Digest of the complete inherited environment frozen for both children.
    pub inherited_environment_blake3: String,
    /// Existing compiler, wrapper, cache, and related settings as observed.
    pub inherited_build_environment: BTreeMap<String, Option<String>>,
    /// Static then dynamic member evidence.
    pub members: [BuildPairMemberReportV1; 2],
    /// Canonical pair record binding both immutable member receipts and shared authority.
    pub pair_record_bytes: Vec<u8>,
    /// Digest of the canonical pair record bytes.
    pub pair_record_blake3: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct BuildPairAuthorityV1;

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
}

#[derive(Serialize)]
struct CanonicalBuildObservationReceiptV1<'a> {
    schema_version: &'static str,
    scenario: &'a str,
    pair_id: &'a str,
    variant: Variant,
    experiment_identity_blake3: &'a str,
    build_authority_receipt_blake3: &'a str,
    duration_use: BuildDurationUseV1,
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
    build_order: [Variant; 2],
    build_duration_use: BuildDurationUseV1,
    member_source_identity_blake3: [&'a str; 2],
    member_cargo_lock_blake3: [&'a str; 2],
    member_target_roots: [&'a str; 2],
    member_build_receipt_blake3: [&'a str; 2],
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct SourceTreeReceiptV1 {
    schema_version: String,
    exclusions: Vec<String>,
    entries: Vec<SourceTreeEntryV1>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct SourceTreeEntryV1 {
    path: String,
    kind: SourceTreeEntryKindV1,
    canonical_mode: u32,
    length: u64,
    blake3: Option<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum SourceTreeEntryKindV1 {
    Directory,
    File,
}

struct SourceTreeAuthorityV1 {
    receipt_bytes: Vec<u8>,
    identity_blake3: String,
    cargo_lock_blake3: String,
}

/// Validate and run both builds under one frozen same-process authority.
pub fn run_paired_build_v1(plan: &BuildPairPlanV1) -> Result<BuildPairReportV1, BuildPairError> {
    validate_plan(plan)?;
    let inherited_environment = capture_environment()?;
    validate_toolchain(plan, &inherited_environment)?;
    let source_authority = acquire_pair_source_authority(plan)?;
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
    let run_context = BuildMemberRunContext {
        plan,
        arguments: &arguments,
        inherited_environment: &inherited_environment,
        lto_environment_name: &lto_environment_name,
        report_command: &command,
        inherited_environment_blake3: &inherited_environment_blake3,
        rustc_verbose_version_blake3: &rustc_verbose_version_blake3,
        sysroot_root: &sysroot_root,
    };

    let static_report = run_member(&run_context, &plan.static_member, &source_authority[0])?;
    validate_cargo_executable(plan)?;
    validate_toolchain(plan, &inherited_environment)?;
    revalidate_pair_source_authority(plan, &source_authority)?;
    let dynamic_report = run_member(&run_context, &plan.dynamic_member, &source_authority[1])?;
    validate_cargo_executable(plan)?;
    validate_toolchain(plan, &inherited_environment)?;
    revalidate_pair_source_authority(plan, &source_authority)?;

    let mut report = BuildPairReportV1 {
        authority: BuildPairAuthorityV1,
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
        build_order: [Variant::Static, Variant::Dynamic],
        build_duration_use: BuildDurationUseV1::DescriptiveNonGating,
        cargo_incremental: "1".to_owned(),
        inherited_environment_blake3,
        inherited_build_environment,
        members: [static_report, dynamic_report],
        pair_record_bytes: Vec::new(),
        pair_record_blake3: String::new(),
    };
    report.pair_record_bytes = canonical_pair_record_bytes(&report)?;
    report.pair_record_blake3 = digest(&report.pair_record_bytes);
    Ok(report)
}

/// Recompute the immutable paired-build authority identity.
///
/// Post-build observations, including active durations and their receipt digests,
/// are deliberately outside this preimage.
pub fn build_pair_authority_blake3_v1(
    report: &BuildPairReportV1,
) -> Result<String, BuildPairError> {
    if report.members[0].source_tree_receipt_bytes != report.members[1].source_tree_receipt_bytes
        || report.members[0].source_identity_blake3 != report.members[1].source_identity_blake3
    {
        return Err(BuildPairError::new(
            "reported paired build complete source tree authority mismatch",
        ));
    }
    for member in &report.members {
        validate_source_tree_receipt(
            &member.source_tree_receipt_bytes,
            &member.source_identity_blake3,
        )?;
    }
    let expected_pair_record = canonical_pair_record_bytes(report)?;
    if expected_pair_record != report.pair_record_bytes
        || digest(&report.pair_record_bytes) != report.pair_record_blake3
    {
        return Err(BuildPairError::new(
            "reported paired build authority mismatch",
        ));
    }
    Ok(digest(&expected_pair_record))
}

pub(crate) fn validate_authoritative_build_report_v1(
    report: &BuildPairReportV1,
) -> Result<[PathBuf; 2], BuildPairError> {
    let _authority = &report.authority;
    if report.schema_version != "plugin_build_pair/v1"
        || report.cargo_incremental != "1"
        || report.members[0].variant != Variant::Static
        || report.members[1].variant != Variant::Dynamic
        || report.build_order != [Variant::Static, Variant::Dynamic]
        || report.build_duration_use != BuildDurationUseV1::DescriptiveNonGating
    {
        return Err(BuildPairError::new(
            "paired build report has a non-authoritative shape",
        ));
    }
    if report.members[0].source_tree_receipt_bytes != report.members[1].source_tree_receipt_bytes
        || report.members[0].source_identity_blake3 != report.members[1].source_identity_blake3
        || report.members[0].cargo_lock_blake3 != report.members[1].cargo_lock_blake3
    {
        return Err(BuildPairError::new(
            "reported paired build complete source tree authority mismatch",
        ));
    }
    let mut artifact_paths = Vec::with_capacity(report.members.len());
    for member in &report.members {
        validate_source_tree_receipt(
            &member.source_tree_receipt_bytes,
            &member.source_identity_blake3,
        )?;
        let target_root = PathBuf::from(&member.target_root);
        canonical_directory(&target_root, "reported target_root")?;
        let artifact_relative_path = PathBuf::from(&member.artifact_relative_path);
        validate_relative_path(&artifact_relative_path, "reported artifact_relative_path")?;
        let artifact_path = target_root.join(&artifact_relative_path);
        let artifact = fs::read(&artifact_path).map_err(|error| {
            BuildPairError::new(format!(
                "cannot read reported build artifact {}: {error}",
                artifact_path.display()
            ))
        })?;
        if digest(&artifact) != member.artifact_blake3 {
            return Err(BuildPairError::new(
                "reported build artifact digest mismatch",
            ));
        }
        let expected_receipt = canonical_bytes(&CanonicalBuildReceiptV1 {
            schema_version: "plugin_build_authority_receipt/v1",
            scenario: &report.scenario,
            pair_id: &report.pair_id,
            variant: member.variant,
            source_commit: &report.source_commit,
            experiment_identity_blake3: &report.experiment_identity_blake3,
            source_identity_blake3: &member.source_identity_blake3,
            cargo_lock_blake3: &member.cargo_lock_blake3,
            cargo_executable_blake3: &report.cargo_executable_blake3,
            rustc_executable_blake3: &report.rustc_executable_blake3,
            rustc_verbose_version: &report.rustc_verbose_version,
            rustc_verbose_version_blake3: &report.rustc_verbose_version_blake3,
            sysroot_root: &report.sysroot_root,
            sysroot_identity_blake3: &report.sysroot_identity_blake3,
            inherited_environment_blake3: &report.inherited_environment_blake3,
            command: &report.command,
            profile: &report.profile,
            features: &report.features,
            lto: report.lto,
            cargo_incremental: "1",
            target_root: &member.target_root,
            artifact_relative_path: &member.artifact_relative_path,
            artifact_blake3: &member.artifact_blake3,
        })?;
        if expected_receipt != member.build_receipt_bytes
            || digest(&member.build_receipt_bytes) != member.build_receipt_blake3
        {
            return Err(BuildPairError::new(
                "reported member build receipt mismatch",
            ));
        }
        let expected_observation = canonical_bytes(&CanonicalBuildObservationReceiptV1 {
            schema_version: "plugin_build_observation_receipt/v1",
            scenario: &report.scenario,
            pair_id: &report.pair_id,
            variant: member.variant,
            experiment_identity_blake3: &report.experiment_identity_blake3,
            build_authority_receipt_blake3: &member.build_receipt_blake3,
            duration_use: report.build_duration_use,
            active_duration_ns: member.active_duration_ns,
        })?;
        if member.active_duration_ns == 0
            || expected_observation != member.build_observation_receipt_bytes
            || digest(&member.build_observation_receipt_bytes)
                != member.build_observation_receipt_blake3
        {
            return Err(BuildPairError::new(
                "reported member build observation receipt mismatch",
            ));
        }
        artifact_paths.push(artifact_path);
    }
    build_pair_authority_blake3_v1(report)?;
    artifact_paths.try_into().map_err(|_| {
        BuildPairError::new("paired build report does not contain exactly two artifacts")
    })
}

fn canonical_pair_record_bytes(report: &BuildPairReportV1) -> Result<Vec<u8>, BuildPairError> {
    canonical_bytes(&CanonicalPairRecordV1 {
        schema_version: "plugin_build_pair_authority/v1",
        scenario: &report.scenario,
        pair_id: &report.pair_id,
        source_commit: &report.source_commit,
        experiment_identity_blake3: &report.experiment_identity_blake3,
        cargo_executable_blake3: &report.cargo_executable_blake3,
        rustc_executable_blake3: &report.rustc_executable_blake3,
        rustc_verbose_version_blake3: &report.rustc_verbose_version_blake3,
        sysroot_identity_blake3: &report.sysroot_identity_blake3,
        inherited_environment_blake3: &report.inherited_environment_blake3,
        command: &report.command,
        profile: &report.profile,
        features: &report.features,
        lto: report.lto,
        cargo_incremental: "1",
        build_order: report.build_order,
        build_duration_use: report.build_duration_use,
        member_source_identity_blake3: [
            &report.members[0].source_identity_blake3,
            &report.members[1].source_identity_blake3,
        ],
        member_cargo_lock_blake3: [
            &report.members[0].cargo_lock_blake3,
            &report.members[1].cargo_lock_blake3,
        ],
        member_target_roots: [
            &report.members[0].target_root,
            &report.members[1].target_root,
        ],
        member_build_receipt_blake3: [
            &report.members[0].build_receipt_blake3,
            &report.members[1].build_receipt_blake3,
        ],
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
    if plan.static_member.source_identity_blake3 != plan.dynamic_member.source_identity_blake3 {
        return Err(BuildPairError::new(
            "paired build members must share one complete source identity",
        ));
    }
    if plan.static_member.cargo_lock_blake3 != plan.dynamic_member.cargo_lock_blake3 {
        return Err(BuildPairError::new(
            "paired build members must share one Cargo.lock identity",
        ));
    }

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

fn acquire_pair_source_authority(
    plan: &BuildPairPlanV1,
) -> Result<[SourceTreeAuthorityV1; 2], BuildPairError> {
    let static_authority = acquire_source_tree_authority(&plan.static_member)?;
    let dynamic_authority = acquire_source_tree_authority(&plan.dynamic_member)?;
    if static_authority.receipt_bytes != dynamic_authority.receipt_bytes
        || static_authority.identity_blake3 != dynamic_authority.identity_blake3
    {
        return Err(BuildPairError::new(
            "paired build members must have an identical complete source tree",
        ));
    }
    if static_authority.cargo_lock_blake3 != dynamic_authority.cargo_lock_blake3 {
        return Err(BuildPairError::new(
            "paired build members must share one Cargo.lock identity",
        ));
    }
    Ok([static_authority, dynamic_authority])
}

fn revalidate_pair_source_authority(
    plan: &BuildPairPlanV1,
    expected: &[SourceTreeAuthorityV1; 2],
) -> Result<(), BuildPairError> {
    let observed = acquire_pair_source_authority(plan)?;
    for (observed, expected) in observed.iter().zip(expected) {
        if observed.receipt_bytes != expected.receipt_bytes
            || observed.identity_blake3 != expected.identity_blake3
            || observed.cargo_lock_blake3 != expected.cargo_lock_blake3
        {
            return Err(BuildPairError::new(
                "complete source tree changed during the paired build",
            ));
        }
    }
    Ok(())
}

fn acquire_source_tree_authority(
    member: &BuildPairMemberV1,
) -> Result<SourceTreeAuthorityV1, BuildPairError> {
    validate_member_identity(member)?;
    let mut entries = Vec::new();
    collect_source_tree_entries(&member.source_root, &member.source_root, &mut entries)?;
    let receipt = SourceTreeReceiptV1 {
        schema_version: "plugin_complete_source_tree/v1".to_owned(),
        exclusions: Vec::new(),
        entries,
    };
    let receipt_bytes = canonical_bytes(&receipt)?;
    let identity_blake3 = digest(&receipt_bytes);
    validate_source_tree_receipt(&receipt_bytes, &identity_blake3)?;
    Ok(SourceTreeAuthorityV1 {
        receipt_bytes,
        identity_blake3,
        cargo_lock_blake3: member.cargo_lock_blake3.clone(),
    })
}

fn collect_source_tree_entries(
    root: &Path,
    directory: &Path,
    entries: &mut Vec<SourceTreeEntryV1>,
) -> Result<(), BuildPairError> {
    let mut children = fs::read_dir(directory)
        .map_err(|error| {
            BuildPairError::new(format!(
                "cannot read complete source tree directory {}: {error}",
                directory.display()
            ))
        })?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| BuildPairError::new(format!("cannot enumerate source tree: {error}")))?;
    children.sort_by_key(|entry| entry.file_name());

    for child in children {
        let path = child.path();
        let relative = path.strip_prefix(root).map_err(|error| {
            BuildPairError::new(format!("cannot derive source tree relative path: {error}"))
        })?;
        validate_relative_path(relative, "source tree entry")?;
        let relative = path_text(relative, "source tree entry")?;
        let metadata = fs::symlink_metadata(&path).map_err(|error| {
            BuildPairError::new(format!(
                "cannot inspect source tree entry {}: {error}",
                path.display()
            ))
        })?;
        let file_type = metadata.file_type();
        if file_type.is_symlink() {
            return Err(BuildPairError::new(format!(
                "source tree links are not allowed: {}",
                path.display()
            )));
        }
        if file_type.is_dir() {
            entries.push(SourceTreeEntryV1 {
                path: relative,
                kind: SourceTreeEntryKindV1::Directory,
                canonical_mode: 0o755,
                length: 0,
                blake3: None,
            });
            collect_source_tree_entries(root, &path, entries)?;
        } else if file_type.is_file() {
            let bytes = fs::read(&path).map_err(|error| {
                BuildPairError::new(format!(
                    "cannot read source tree file {}: {error}",
                    path.display()
                ))
            })?;
            entries.push(SourceTreeEntryV1 {
                path: relative,
                kind: SourceTreeEntryKindV1::File,
                canonical_mode: canonical_file_mode(&metadata),
                length: u64::try_from(bytes.len())
                    .map_err(|_| BuildPairError::new("source tree file length does not fit u64"))?,
                blake3: Some(digest(&bytes)),
            });
        } else {
            return Err(BuildPairError::new(format!(
                "unsupported source tree entry kind: {}",
                path.display()
            )));
        }
    }
    Ok(())
}

#[cfg(unix)]
fn canonical_file_mode(metadata: &fs::Metadata) -> u32 {
    use std::os::unix::fs::PermissionsExt as _;

    if metadata.permissions().mode() & 0o111 == 0 {
        0o644
    } else {
        0o755
    }
}

#[cfg(not(unix))]
fn canonical_file_mode(_metadata: &fs::Metadata) -> u32 {
    0o644
}

fn validate_source_tree_receipt(
    receipt_bytes: &[u8],
    expected_identity_blake3: &str,
) -> Result<(), BuildPairError> {
    let receipt: SourceTreeReceiptV1 = serde_json::from_slice(receipt_bytes).map_err(|error| {
        BuildPairError::new(format!(
            "cannot parse complete source tree receipt: {error}"
        ))
    })?;
    if receipt.schema_version != "plugin_complete_source_tree/v1"
        || !receipt.exclusions.is_empty()
        || canonical_bytes(&receipt)? != receipt_bytes
        || digest(receipt_bytes) != expected_identity_blake3
    {
        return Err(BuildPairError::new(
            "complete source tree receipt identity mismatch",
        ));
    }
    let mut previous: Option<&str> = None;
    for entry in &receipt.entries {
        validate_relative_path(Path::new(&entry.path), "source tree receipt entry")?;
        if previous.is_some_and(|path| path >= entry.path.as_str()) {
            return Err(BuildPairError::new(
                "complete source tree receipt entries must be sorted and unique",
            ));
        }
        previous = Some(&entry.path);
        match entry.kind {
            SourceTreeEntryKindV1::Directory
                if entry.canonical_mode == 0o755 && entry.length == 0 && entry.blake3.is_none() => {
            }
            SourceTreeEntryKindV1::File
                if matches!(entry.canonical_mode, 0o644 | 0o755)
                    && entry.blake3.as_deref().is_some_and(|value| {
                        validate_digest("source tree file blake3", value).is_ok()
                    }) => {}
            _ => {
                return Err(BuildPairError::new(
                    "complete source tree receipt entry is invalid",
                ));
            }
        }
    }
    Ok(())
}

struct BuildMemberRunContext<'a> {
    plan: &'a BuildPairPlanV1,
    arguments: &'a [String],
    inherited_environment: &'a BTreeMap<String, String>,
    lto_environment_name: &'a str,
    report_command: &'a [String],
    inherited_environment_blake3: &'a str,
    rustc_verbose_version_blake3: &'a str,
    sysroot_root: &'a str,
}

fn run_member(
    context: &BuildMemberRunContext<'_>,
    member: &BuildPairMemberV1,
    source_authority: &SourceTreeAuthorityV1,
) -> Result<BuildPairMemberReportV1, BuildPairError> {
    let plan = context.plan;
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
        .args(context.arguments)
        .current_dir(&member.source_root)
        .env_clear();
    for (name, value) in context.inherited_environment {
        command.env(name, value);
    }
    command
        .env("CARGO_INCREMENTAL", "1")
        .env("CARGO_TARGET_DIR", &member.target_root)
        .env("RUSTC", &plan.rustc_executable)
        .env(context.lto_environment_name, plan.lto.cargo_value());

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
        schema_version: "plugin_build_authority_receipt/v1",
        scenario: &plan.scenario,
        pair_id: &plan.pair_id,
        variant: member.variant,
        source_commit: &plan.source_commit,
        experiment_identity_blake3: &plan.experiment_identity_blake3,
        source_identity_blake3: &source_authority.identity_blake3,
        cargo_lock_blake3: &source_authority.cargo_lock_blake3,
        cargo_executable_blake3: &plan.cargo_executable_blake3,
        rustc_executable_blake3: &plan.rustc_executable_blake3,
        rustc_verbose_version: &plan.rustc_verbose_version,
        rustc_verbose_version_blake3: context.rustc_verbose_version_blake3,
        sysroot_root: context.sysroot_root,
        sysroot_identity_blake3: &plan.sysroot_identity_blake3,
        inherited_environment_blake3: context.inherited_environment_blake3,
        command: context.report_command,
        profile: &plan.profile,
        features: &plan.features,
        lto: plan.lto,
        cargo_incremental: "1",
        target_root: &target_root,
        artifact_relative_path: &artifact_relative_path,
        artifact_blake3: &artifact_blake3,
    })?;
    let build_receipt_blake3 = digest(&build_receipt_bytes);
    let build_observation_receipt_bytes = canonical_bytes(&CanonicalBuildObservationReceiptV1 {
        schema_version: "plugin_build_observation_receipt/v1",
        scenario: &plan.scenario,
        pair_id: &plan.pair_id,
        variant: member.variant,
        experiment_identity_blake3: &plan.experiment_identity_blake3,
        build_authority_receipt_blake3: &build_receipt_blake3,
        duration_use: BuildDurationUseV1::DescriptiveNonGating,
        active_duration_ns: elapsed,
    })?;
    let build_observation_receipt_blake3 = digest(&build_observation_receipt_bytes);

    Ok(BuildPairMemberReportV1 {
        variant: member.variant,
        source_identity_blake3: source_authority.identity_blake3.clone(),
        source_tree_receipt_bytes: source_authority.receipt_bytes.clone(),
        cargo_lock_blake3: source_authority.cargo_lock_blake3.clone(),
        target_root,
        artifact_relative_path,
        artifact_blake3,
        active_duration_ns: elapsed,
        build_receipt_bytes,
        build_receipt_blake3,
        build_observation_receipt_bytes,
        build_observation_receipt_blake3,
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
