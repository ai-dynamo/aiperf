// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Same-process authority for the complete runtime parity matrix.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::{Deserialize, Serialize};

use crate::build_pair::{BuildPairReportV1, validate_authoritative_build_report_v1};
use crate::exporter_policy::parse_exporter_observable_policy;
use crate::exporter_runner::{
    CompletedExporterMember, ExporterHarnessRunner, ExporterMemberSource, ExporterWorkload,
};
use crate::plugin_stats::{
    AuthoritativeIdentityInput, ControlledAttemptDecision, ControlledAttemptRecord,
    ControlledExporterPairRecord, ControlledMeasurementEvaluator, ExporterMember, FrozenCasePlan,
    MemberTerminalOutcome, PairAttemptDecision, PairedCase, PairedSample, RawMemberTerminalRecord,
    RawPairTerminalRecord, SimultaneousGateInput, SimultaneousGateReport, Variant,
    acquire_authoritative_identity, checked_in_case_plans, checked_in_inventory_digest,
};

const OUTPUT_SCHEMA_V1: &[u8] = b"plugin_runtime_member_output/v1;closed-jcs-line;scenario,pair_id,variant,experiment_identity_blake3,completed_budget,active_duration_nanoseconds,metrics";
const CALIBRATION_POLICY_BYTES: &[u8] =
    include_bytes!("../../benchmarks/exporter-observable-policy.json");
const TASKSET: &str = "/usr/bin/taskset";

/// Read-only controller coordinates for acquiring one exporter implementation.
#[derive(Clone, Copy, Debug)]
pub struct ExporterWorkloadRequest<'a> {
    scenario: &'a str,
    pair_id: &'a str,
    member: ExporterMember,
}

impl ExporterWorkloadRequest<'_> {
    /// Frozen exporter scenario identifier.
    pub fn scenario(&self) -> &str {
        self.scenario
    }

    /// Controller-scheduled pair identifier, including warmups.
    pub fn pair_id(&self) -> &str {
        self.pair_id
    }

    /// Static comparator or dynamic candidate requested by the controller.
    pub fn member(&self) -> ExporterMember {
        self.member
    }
}

/// Failure to acquire an exporter implementation for one controlled member.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExporterWorkloadAcquisitionError(String);

impl ExporterWorkloadAcquisitionError {
    /// Construct an acquisition failure without granting measurement authority.
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for ExporterWorkloadAcquisitionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for ExporterWorkloadAcquisitionError {}

/// Narrow implementation factory; all measurement authority remains in the controller.
pub trait ControlledExporterWorkloadFactory {
    /// Acquire only the exporter implementation for one controller-scheduled member.
    fn acquire(
        &mut self,
        request: ExporterWorkloadRequest<'_>,
    ) -> Result<Box<dyn ExporterWorkload>, ExporterWorkloadAcquisitionError>;
}

/// Failure while sealing, executing, or evaluating a controlled runtime matrix.
#[derive(Debug)]
pub struct ControlledRuntimeError(String);

impl ControlledRuntimeError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for ControlledRuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for ControlledRuntimeError {}

/// Complete result owned by one non-resettable controlled runner invocation.
#[derive(Clone, Debug, Serialize)]
pub struct ControlledRuntimeReportV1 {
    /// Digest of the one sealed experiment identity used by every child row.
    pub experiment_identity_blake3: String,
    /// Canonical bytes of the sealed experiment identity.
    pub experiment_identity_bytes: Vec<u8>,
    /// Terminal controller decision.
    pub decision: ControlledAttemptDecision,
    /// Statistical report, present only after a complete valid matrix.
    pub statistical_report: Option<SimultaneousGateReport>,
    /// Append-only attempt ledger for this invocation.
    pub attempt_history: Vec<ControlledAttemptRecord>,
    /// Paired build record bound into the experiment identity.
    pub paired_build_record_blake3: String,
    /// Checked-in exporter observable policy bound into the identity.
    pub observable_policy_blake3: String,
    /// Authenticated receiver-protocol set bound separately from policy JSON.
    pub receiver_protocol_authority_blake3: String,
    /// Closed terminal-output schema bound into the identity.
    pub output_schema_blake3: String,
    /// Complete checked-in workload matrix bound into the identity.
    pub workload_contract_blake3: String,
    /// Deterministic exporter corpus bound into the identity.
    pub corpus_blake3: String,
    /// Number of checked-in scenarios evaluated.
    pub scenario_count: usize,
    /// Number of retained scenario/pair combinations.
    pub retained_pair_count: usize,
    /// Number of controlled static/dynamic member executions, including warmups.
    pub executed_member_count: usize,
    /// Digests of exact child stdout in execution order; exporter adapters produce none.
    pub terminal_output_blake3: Vec<String>,
    /// Complete validated exporter evidence retained by the controller.
    pub exporter_pair_history: Vec<ControlledExporterPairRecord>,
    /// Canonical evidence binding identity, ledger, report, and raw outputs.
    pub runtime_evidence_bytes: Vec<u8>,
    /// Digest of the canonical runtime evidence bytes.
    pub runtime_evidence_blake3: String,
}

#[derive(Serialize)]
struct RuntimeEvidenceV1<'a> {
    schema_version: u8,
    experiment_identity_blake3: &'a str,
    decision: ControlledAttemptDecision,
    statistical_report: Option<&'a SimultaneousGateReport>,
    attempt_history: &'a [ControlledAttemptRecord],
    paired_build_record_blake3: &'a str,
    observable_policy_blake3: &'a str,
    receiver_protocol_authority_blake3: &'a str,
    output_schema_blake3: &'a str,
    workload_contract_blake3: &'a str,
    corpus_blake3: &'a str,
    scenario_count: usize,
    retained_pair_count: usize,
    executed_member_count: usize,
    terminal_output_blake3: &'a [String],
    exporter_pair_history: &'a [ControlledExporterPairRecord],
}

#[derive(Serialize)]
struct RuntimeAuthorityContractV1<'a> {
    schema_version: u8,
    paired_build_record_blake3: &'a str,
    member_build_receipt_blake3: [&'a str; 2],
    member_source_identity_blake3: [&'a str; 2],
    member_cargo_lock_blake3: [&'a str; 2],
    member_artifact_blake3: [&'a str; 2],
    observable_policy_blake3: &'a str,
    receiver_protocol_authority_blake3: &'a str,
    output_schema_blake3: &'a str,
    workload_contract_blake3: &'a str,
    corpus_blake3: &'a str,
    inventory_blake3: &'a str,
    admitted_environment_blake3: &'a str,
    taskset_blake3: &'a str,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct RuntimeMemberOutputV1 {
    active_duration_nanoseconds: u64,
    completed_budget: u64,
    experiment_identity_blake3: String,
    metrics: BTreeMap<String, f64>,
    pair_id: String,
    scenario: String,
    schema_version: u8,
    variant: Variant,
}

struct MemberExecution {
    outcome: MemberTerminalOutcome,
    samples: Vec<PairedSample>,
    stdout_blake3: Option<String>,
}

struct ExporterExecutionContext<'a> {
    runner: &'a ExporterHarnessRunner,
    policy: &'a crate::exporter_policy::ExporterObservablePolicyV1,
    artifacts: &'a [File; 2],
    build_report: &'a BuildPairReportV1,
    experiment_identity_bytes: &'a [u8],
}

struct ExporterMemberCoordinates<'a> {
    case: &'a FrozenCasePlan,
    pair_id: &'a str,
    variant: Variant,
}

struct RuntimeReportContext<'a> {
    experiment_identity_blake3: &'a str,
    experiment_identity_bytes: &'a [u8],
    build_report: &'a BuildPairReportV1,
    observable_policy_blake3: &'a str,
    receiver_protocol_authority_blake3: &'a str,
    output_schema_blake3: &'a str,
    workload_contract_blake3: &'a str,
    corpus_blake3: &'a str,
    scenario_count: usize,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct AttemptLedgerEntryPreimageV1 {
    schema_version: u8,
    experiment_identity_blake3: String,
    previous_entry_blake3: Option<String>,
    attempt: ControlledAttemptRecord,
    evidence_tree_bytes: Vec<u8>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct AttemptLedgerEntryV1 {
    schema_version: u8,
    experiment_identity_blake3: String,
    previous_entry_blake3: Option<String>,
    attempt: ControlledAttemptRecord,
    evidence_tree_bytes: Vec<u8>,
    entry_blake3: String,
}

struct AttemptLedger {
    file: File,
    experiment_identity_blake3: String,
    entries: Vec<AttemptLedgerEntryV1>,
}

impl AttemptLedger {
    fn acquire(
        path: &Path,
        experiment_identity_blake3: &str,
    ) -> Result<Self, ControlledRuntimeError> {
        #[cfg(unix)]
        use std::os::unix::fs::OpenOptionsExt as _;
        #[cfg(unix)]
        use std::os::unix::io::AsRawFd as _;

        let mut options = OpenOptions::new();
        options.create(true).read(true).append(true);
        #[cfg(unix)]
        options.mode(0o600).custom_flags(libc::O_NOFOLLOW);
        let mut file = options.open(path).map_err(|error| {
            ControlledRuntimeError::new(format!("cannot open attempt ledger: {error}"))
        })?;
        #[cfg(unix)]
        if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX) } != 0 {
            return Err(ControlledRuntimeError::new(format!(
                "cannot lock attempt ledger: {}",
                std::io::Error::last_os_error()
            )));
        }
        #[cfg(not(unix))]
        return Err(ControlledRuntimeError::new(
            "persistent attempt ledger locking is unavailable on this platform",
        ));

        file.seek(SeekFrom::Start(0)).map_err(|error| {
            ControlledRuntimeError::new(format!("cannot rewind attempt ledger: {error}"))
        })?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes).map_err(|error| {
            ControlledRuntimeError::new(format!("cannot read attempt ledger: {error}"))
        })?;
        let mut entries = Vec::new();
        let mut start = 0;
        for end in bytes
            .iter()
            .enumerate()
            .filter_map(|(index, byte)| (*byte == b'\n').then_some(index + 1))
        {
            let line = &bytes[start..end];
            if line == b"\n" {
                return Err(ControlledRuntimeError::new(
                    "attempt ledger contains an empty line",
                ));
            }
            let entry: AttemptLedgerEntryV1 = serde_json::from_slice(line).map_err(|error| {
                ControlledRuntimeError::new(format!("attempt ledger entry is invalid: {error}"))
            })?;
            let mut canonical = serde_json_canonicalizer::to_vec(&entry).map_err(|error| {
                ControlledRuntimeError::new(format!(
                    "cannot canonicalize attempt ledger entry: {error}"
                ))
            })?;
            canonical.push(b'\n');
            if canonical != line {
                return Err(ControlledRuntimeError::new(
                    "attempt ledger entry is not exact canonical JCS plus newline",
                ));
            }
            Self::validate_entry(&entry, experiment_identity_blake3, entries.last())?;
            entries.push(entry);
            start = end;
        }
        if start != bytes.len() {
            return Err(ControlledRuntimeError::new(
                "attempt ledger has an unterminated final entry",
            ));
        }
        Ok(Self {
            file,
            experiment_identity_blake3: experiment_identity_blake3.to_owned(),
            entries,
        })
    }

    fn validate_entry(
        entry: &AttemptLedgerEntryV1,
        experiment_identity_blake3: &str,
        previous: Option<&AttemptLedgerEntryV1>,
    ) -> Result<(), ControlledRuntimeError> {
        let expected_ordinal =
            u8::try_from(previous.map_or(1, |entry| usize::from(entry.attempt.ordinal) + 1))
                .map_err(|_| ControlledRuntimeError::new("attempt ledger ordinal overflow"))?;
        let expected_previous = previous.map(|entry| entry.entry_blake3.as_str());
        let preimage = AttemptLedgerEntryPreimageV1 {
            schema_version: entry.schema_version,
            experiment_identity_blake3: entry.experiment_identity_blake3.clone(),
            previous_entry_blake3: entry.previous_entry_blake3.clone(),
            attempt: entry.attempt.clone(),
            evidence_tree_bytes: entry.evidence_tree_bytes.clone(),
        };
        let expected_entry_blake3 = canonical_digest(&preimage, "attempt ledger preimage")?;
        if entry.schema_version != 1
            || entry.experiment_identity_blake3 != experiment_identity_blake3
            || entry.previous_entry_blake3.as_deref() != expected_previous
            || entry.attempt.ordinal != expected_ordinal
            || entry.attempt.evidence_tree_blake3 != digest(&entry.evidence_tree_bytes)
            || entry.entry_blake3 != expected_entry_blake3
            || previous.is_some_and(|previous| {
                previous.attempt.decision != ControlledAttemptDecision::Invalid
            })
        {
            return Err(ControlledRuntimeError::new(
                "attempt ledger hash chain or terminal authority is invalid",
            ));
        }
        Ok(())
    }

    fn history(&self) -> Vec<ControlledAttemptRecord> {
        self.entries
            .iter()
            .map(|entry| entry.attempt.clone())
            .collect()
    }

    fn next_attempt_ordinal(&self) -> Result<u8, ControlledRuntimeError> {
        if self
            .entries
            .last()
            .is_some_and(|entry| entry.attempt.decision != ControlledAttemptDecision::Invalid)
        {
            return Err(ControlledRuntimeError::new(
                "the first valid experiment attempt is authoritative",
            ));
        }
        if self.entries.len() >= 3 {
            return Err(ControlledRuntimeError::new(
                "three invalid attempts block the experiment",
            ));
        }
        u8::try_from(self.entries.len() + 1)
            .map_err(|_| ControlledRuntimeError::new("attempt ledger ordinal overflow"))
    }

    fn append_attempt(
        &mut self,
        attempt: ControlledAttemptRecord,
        evidence_tree_bytes: &[u8],
    ) -> Result<AttemptLedgerEntryV1, ControlledRuntimeError> {
        if attempt.ordinal != self.next_attempt_ordinal()? {
            return Err(ControlledRuntimeError::new(
                "attempt ledger append ordinal is not next",
            ));
        }
        if attempt.evidence_tree_blake3 != digest(evidence_tree_bytes) {
            return Err(ControlledRuntimeError::new(
                "attempt ledger evidence bytes do not match their digest",
            ));
        }
        let preimage = AttemptLedgerEntryPreimageV1 {
            schema_version: 1,
            experiment_identity_blake3: self.experiment_identity_blake3.clone(),
            previous_entry_blake3: self.entries.last().map(|entry| entry.entry_blake3.clone()),
            attempt,
            evidence_tree_bytes: evidence_tree_bytes.to_vec(),
        };
        let entry_blake3 = canonical_digest(&preimage, "attempt ledger preimage")?;
        let entry = AttemptLedgerEntryV1 {
            schema_version: preimage.schema_version,
            experiment_identity_blake3: preimage.experiment_identity_blake3,
            previous_entry_blake3: preimage.previous_entry_blake3,
            attempt: preimage.attempt,
            evidence_tree_bytes: preimage.evidence_tree_bytes,
            entry_blake3,
        };
        let mut line = serde_json_canonicalizer::to_vec(&entry).map_err(|error| {
            ControlledRuntimeError::new(format!(
                "cannot canonicalize attempt ledger append: {error}"
            ))
        })?;
        line.push(b'\n');
        self.file.write_all(&line).map_err(|error| {
            ControlledRuntimeError::new(format!("cannot append attempt ledger: {error}"))
        })?;
        self.file.sync_all().map_err(|error| {
            ControlledRuntimeError::new(format!("cannot sync attempt ledger: {error}"))
        })?;
        self.entries.push(entry.clone());
        Ok(entry)
    }
}

/// Execute both build-bound members across the complete checked-in matrix.
pub fn run_controlled_runtime_v1(
    _build_report: &BuildPairReportV1,
) -> Result<ControlledRuntimeReportV1, ControlledRuntimeError> {
    Err(ControlledRuntimeError::new(
        "authoritative runtime execution requires a persistent attempt ledger",
    ))
}

/// Execute both build-bound members under one persistent attempt ledger.
pub fn run_controlled_runtime_with_ledger_v1(
    build_report: &BuildPairReportV1,
    attempt_ledger_path: &Path,
) -> Result<ControlledRuntimeReportV1, ControlledRuntimeError> {
    run_controlled_runtime_internal(
        build_report,
        None,
        CALIBRATION_POLICY_BYTES,
        attempt_ledger_path,
    )
}

/// Refuse an exporter implementation that is unrelated to the acquired artifacts.
///
/// Exporter performance authority comes only from executing the exact artifact
/// descriptors validated by [`run_controlled_runtime_v1`]. An in-process
/// factory cannot prove that its Rust implementation was loaded from either
/// descriptor, so accepting one would let both pair members measure unrelated
/// code while retaining the paired build identities.
pub fn run_controlled_runtime_with_exporters_v1(
    _build_report: &BuildPairReportV1,
    _exporter_factory: &mut dyn ControlledExporterWorkloadFactory,
) -> Result<ControlledRuntimeReportV1, ControlledRuntimeError> {
    Err(ControlledRuntimeError::new(
        "unrelated in-process exporter workload cannot acquire the already-open artifact authority",
    ))
}

fn run_controlled_runtime_internal(
    build_report: &BuildPairReportV1,
    mut exporter_factory: Option<&mut dyn ControlledExporterWorkloadFactory>,
    policy_bytes: &[u8],
    attempt_ledger_path: &Path,
) -> Result<ControlledRuntimeReportV1, ControlledRuntimeError> {
    let artifact_paths = validate_authoritative_build_report_v1(build_report).map_err(|error| {
        ControlledRuntimeError::new(format!("invalid paired build authority: {error}"))
    })?;
    let cases =
        checked_in_case_plans().map_err(|error| ControlledRuntimeError::new(error.to_string()))?;
    let inventory_blake3 = checked_in_inventory_digest()
        .map_err(|error| ControlledRuntimeError::new(error.to_string()))?;
    let policy =
        parse_exporter_observable_policy(policy_bytes, &BTreeSet::new()).map_err(|error| {
            ControlledRuntimeError::new(format!("invalid checked-in exporter policy: {error}"))
        })?;
    let observable_policy_blake3 = policy
        .canonical_blake3()
        .map_err(|error| ControlledRuntimeError::new(error.to_string()))?;
    let receiver_protocol_authority_blake3 = policy.receiver_protocol_authority_blake3().to_owned();
    let output_schema_blake3 = digest(OUTPUT_SCHEMA_V1);
    let workload_contract_blake3 = canonical_digest(&cases, "workload contract")?;
    let exporter_runner = ExporterHarnessRunner::new(policy.clone())
        .map_err(|error| ControlledRuntimeError::new(error.to_string()))?;
    let corpus_blake3 = exporter_runner.corpus_blake3().to_owned();
    let taskset_blake3 = validate_taskset()?;
    let inherited_environment = capture_environment()?;
    let admitted_environment = inherited_environment
        .iter()
        .map(|(name, value)| (name.clone(), Some(value.clone())))
        .collect::<BTreeMap<_, _>>();
    let admitted_environment_blake3 =
        canonical_digest(&admitted_environment, "admitted environment")?;
    let authority_contract_blake3 = canonical_digest(
        &RuntimeAuthorityContractV1 {
            schema_version: 1,
            paired_build_record_blake3: &build_report.pair_record_blake3,
            member_build_receipt_blake3: [
                &build_report.members[0].build_receipt_blake3,
                &build_report.members[1].build_receipt_blake3,
            ],
            member_source_identity_blake3: [
                &build_report.members[0].source_identity_blake3,
                &build_report.members[1].source_identity_blake3,
            ],
            member_cargo_lock_blake3: [
                &build_report.members[0].cargo_lock_blake3,
                &build_report.members[1].cargo_lock_blake3,
            ],
            member_artifact_blake3: [
                &build_report.members[0].artifact_blake3,
                &build_report.members[1].artifact_blake3,
            ],
            observable_policy_blake3: &observable_policy_blake3,
            receiver_protocol_authority_blake3: &receiver_protocol_authority_blake3,
            output_schema_blake3: &output_schema_blake3,
            workload_contract_blake3: &workload_contract_blake3,
            corpus_blake3: &corpus_blake3,
            inventory_blake3: &inventory_blake3,
            admitted_environment_blake3: &admitted_environment_blake3,
            taskset_blake3: &taskset_blake3,
        },
        "runtime authority contract",
    )?;
    let rustc = String::from_utf8(build_report.rustc_verbose_version.clone())
        .map_err(|_| ControlledRuntimeError::new("paired build rustc identity is not UTF-8"))?;
    let current_executable = std::env::current_exe().map_err(|error| {
        ControlledRuntimeError::new(format!(
            "cannot resolve runtime harness executable: {error}"
        ))
    })?;
    let harness_artifact_blake3 = digest_file(&current_executable, "runtime harness")?;
    let source_tree_digest = canonical_digest(
        &[
            &build_report.members[0].source_identity_blake3,
            &build_report.members[1].source_identity_blake3,
        ],
        "paired source identity",
    )?;
    let cargo_lock_digest = canonical_digest(
        &[
            &build_report.members[0].cargo_lock_blake3,
            &build_report.members[1].cargo_lock_blake3,
        ],
        "paired Cargo.lock identity",
    )?;
    let observed = acquire_authoritative_identity(AuthoritativeIdentityInput {
        source_commit: build_report.source_commit.clone(),
        source_tree_digest,
        cargo_lock_digest,
        rustc,
        sysroot_digest: build_report.sysroot_identity_blake3.clone(),
        profile: build_report.profile.clone(),
        static_artifact_digest: build_report.members[0].artifact_blake3.clone(),
        dynamic_artifact_digest: build_report.members[1].artifact_blake3.clone(),
        harness_artifact_digest: harness_artifact_blake3,
        authority_contract_digest: authority_contract_blake3,
        environment: admitted_environment,
    })
    .map_err(|error| ControlledRuntimeError::new(error.to_string()))?;
    let experiment_identity_blake3 = observed.identity_digest().to_owned();
    let exporter_identity_preimage_bytes = observed
        .identity_digest_preimage_bytes()
        .map_err(|error| ControlledRuntimeError::new(error.to_string()))?;
    let experiment_identity_bytes = observed
        .canonical_identity_bytes()
        .map_err(|error| ControlledRuntimeError::new(error.to_string()))?;
    let mut attempt_ledger =
        AttemptLedger::acquire(attempt_ledger_path, &experiment_identity_blake3)?;
    let expected_attempt_ordinal = attempt_ledger.next_attempt_ordinal()?;
    let exporter_artifacts = if exporter_factory.is_some() {
        Some([
            File::open(&artifact_paths[0]).map_err(|error| {
                ControlledRuntimeError::new(format!(
                    "cannot acquire static exporter artifact: {error}"
                ))
            })?,
            File::open(&artifact_paths[1]).map_err(|error| {
                ControlledRuntimeError::new(format!(
                    "cannot acquire dynamic exporter artifact: {error}"
                ))
            })?,
        ])
    } else {
        None
    };
    let exporter_context = exporter_artifacts
        .as_ref()
        .map(|artifacts| ExporterExecutionContext {
            runner: &exporter_runner,
            policy: &policy,
            artifacts,
            build_report,
            experiment_identity_bytes: &exporter_identity_preimage_bytes,
        });
    let report_context = RuntimeReportContext {
        experiment_identity_blake3: &experiment_identity_blake3,
        experiment_identity_bytes: &experiment_identity_bytes,
        build_report,
        observable_policy_blake3: &observable_policy_blake3,
        receiver_protocol_authority_blake3: &receiver_protocol_authority_blake3,
        output_schema_blake3: &output_schema_blake3,
        workload_contract_blake3: &workload_contract_blake3,
        corpus_blake3: &corpus_blake3,
        scenario_count: cases.len(),
    };

    let mut evaluator = ControlledMeasurementEvaluator::resume(attempt_ledger.history())
        .map_err(|error| ControlledRuntimeError::new(error.to_string()))?;
    let attempt_ordinal = evaluator
        .begin_attempt()
        .map_err(|error| ControlledRuntimeError::new(error.to_string()))?;
    if attempt_ordinal != expected_attempt_ordinal {
        return Err(ControlledRuntimeError::new(
            "persistent ledger and evaluator attempt ordinals differ",
        ));
    }
    let schedule = evaluator.pair_schedule().to_vec();
    let mut executed_member_count = 0_usize;
    let mut terminal_output_blake3 = Vec::new();
    let mut measured_cases = Vec::with_capacity(cases.len());

    for case in &cases {
        let is_exporter_case = case
            .measured_metrics
            .iter()
            .any(|metric| metric == "exporter_nanoseconds_per_record");
        for warmup in 0..case.warmups {
            let pair_id = format!("warmup-{warmup:02}");
            for variant in [Variant::Static, Variant::Dynamic] {
                if is_exporter_case && exporter_factory.is_some() {
                    executed_member_count += 1;
                    let result = execute_exporter_member(
                        exporter_factory.as_deref_mut().ok_or_else(|| {
                            ControlledRuntimeError::new(
                                "exporter factory disappeared during controlled execution",
                            )
                        })?,
                        exporter_context.as_ref().ok_or_else(|| {
                            ControlledRuntimeError::new(
                                "exporter context is absent under adapter authority",
                            )
                        })?,
                        ExporterMemberCoordinates {
                            case,
                            pair_id: &pair_id,
                            variant,
                        },
                    );
                    if let Err(error) = result {
                        evaluator
                            .finish_authoritative_product_failure(format!(
                                "controlled exporter warmup failed: {error}"
                            ))
                            .map_err(|error| ControlledRuntimeError::new(error.to_string()))?;
                        return runtime_report(
                            &evaluator,
                            &mut attempt_ledger,
                            &report_context,
                            executed_member_count,
                            terminal_output_blake3,
                        );
                    }
                } else {
                    let execution = execute_member(
                        case,
                        &pair_id,
                        variant,
                        artifact_for(variant, &artifact_paths),
                        build_report,
                        &experiment_identity_blake3,
                        &inherited_environment,
                    )?;
                    executed_member_count += 1;
                    if let Some(stdout_blake3) = execution.stdout_blake3 {
                        terminal_output_blake3.push(stdout_blake3);
                    }
                    if execution.outcome != MemberTerminalOutcome::Completed {
                        evaluator
                            .finish_authoritative_product_failure(format!(
                                "warmup {pair_id} for {} {:?} failed: {:?}",
                                case.scenario, variant, execution.outcome
                            ))
                            .map_err(|error| ControlledRuntimeError::new(error.to_string()))?;
                        return runtime_report(
                            &evaluator,
                            &mut attempt_ledger,
                            &report_context,
                            executed_member_count,
                            terminal_output_blake3,
                        );
                    }
                }
            }
        }

        let mut samples = Vec::new();
        for scheduled in &schedule {
            let mut member_records = Vec::with_capacity(2);
            let mut pair_samples = Vec::new();
            let mut completed_exporters = Vec::with_capacity(2);
            for variant in scheduled.member_order {
                if is_exporter_case && exporter_factory.is_some() {
                    executed_member_count += 1;
                    match execute_exporter_member(
                        exporter_factory.as_deref_mut().ok_or_else(|| {
                            ControlledRuntimeError::new(
                                "exporter factory disappeared during controlled execution",
                            )
                        })?,
                        exporter_context.as_ref().ok_or_else(|| {
                            ControlledRuntimeError::new(
                                "exporter context is absent under adapter authority",
                            )
                        })?,
                        ExporterMemberCoordinates {
                            case,
                            pair_id: &scheduled.pair_id,
                            variant,
                        },
                    ) {
                        Ok(completed) => completed_exporters.push((variant, completed)),
                        Err(error) => {
                            evaluator
                                .finish_authoritative_product_failure(format!(
                                    "controlled exporter member failed: {error}"
                                ))
                                .map_err(|error| ControlledRuntimeError::new(error.to_string()))?;
                            return runtime_report(
                                &evaluator,
                                &mut attempt_ledger,
                                &report_context,
                                executed_member_count,
                                terminal_output_blake3,
                            );
                        }
                    }
                } else {
                    let execution = execute_member(
                        case,
                        &scheduled.pair_id,
                        variant,
                        artifact_for(variant, &artifact_paths),
                        build_report,
                        &experiment_identity_blake3,
                        &inherited_environment,
                    )?;
                    executed_member_count += 1;
                    if let Some(stdout_blake3) = execution.stdout_blake3 {
                        terminal_output_blake3.push(stdout_blake3);
                    }
                    member_records.push(RawMemberTerminalRecord {
                        variant,
                        outcome: execution.outcome,
                    });
                    pair_samples.extend(execution.samples);
                }
            }
            let raw_pair = RawPairTerminalRecord {
                scenario: case.scenario.clone(),
                pair_id: scheduled.pair_id.clone(),
                member_order: scheduled.member_order,
                members: member_records,
                asserted_reason: None,
                asserted_disposition: None,
            };
            let decision = if is_exporter_case
                && exporter_factory.is_some()
                && completed_exporters.len() == 2
            {
                let static_member = completed_exporters
                    .iter()
                    .find(|(variant, _)| *variant == Variant::Static)
                    .map(|(_, completed)| completed)
                    .ok_or_else(|| {
                        ControlledRuntimeError::new("static completed exporter member is absent")
                    })?;
                let dynamic_member = completed_exporters
                    .iter()
                    .find(|(variant, _)| *variant == Variant::Dynamic)
                    .map(|(_, completed)| completed)
                    .ok_or_else(|| {
                        ControlledRuntimeError::new("dynamic completed exporter member is absent")
                    })?;
                evaluator
                    .record_completed_exporter_pair(&policy, static_member, dynamic_member)
                    .map_err(|error| ControlledRuntimeError::new(error.to_string()))?
            } else {
                evaluator
                    .record_pair(raw_pair)
                    .map_err(|error| ControlledRuntimeError::new(error.to_string()))?
            };
            if decision != PairAttemptDecision::RetainPair {
                return runtime_report(
                    &evaluator,
                    &mut attempt_ledger,
                    &report_context,
                    executed_member_count,
                    terminal_output_blake3,
                );
            }
            samples.extend(pair_samples);
        }
        measured_cases.push(PairedCase {
            scenario: case.scenario.clone(),
            primary_metric: case.primary_metric.clone(),
            samples,
            invalidation_attempts: Vec::new(),
        });
    }

    evaluator
        .finish_authoritative_measurements(
            &SimultaneousGateInput {
                cases: measured_cases,
            },
            observed,
        )
        .map_err(|error| ControlledRuntimeError::new(error.to_string()))?;
    runtime_report(
        &evaluator,
        &mut attempt_ledger,
        &report_context,
        executed_member_count,
        terminal_output_blake3,
    )
}

fn execute_exporter_member(
    factory: &mut dyn ControlledExporterWorkloadFactory,
    context: &ExporterExecutionContext<'_>,
    coordinates: ExporterMemberCoordinates<'_>,
) -> Result<CompletedExporterMember, String> {
    let member = match coordinates.variant {
        Variant::Static => ExporterMember::Static,
        Variant::Dynamic => ExporterMember::Dynamic,
    };
    let mut workload = factory
        .acquire(ExporterWorkloadRequest {
            scenario: &coordinates.case.scenario,
            pair_id: coordinates.pair_id,
            member,
        })
        .map_err(|error| format!("exporter adapter acquisition failed: {error}"))?;
    if context.policy.observable_kind(&coordinates.case.scenario)
        == Some(crate::plugin_stats::ExporterObservableKind::ReceiverTranscript)
    {
        return Err(
            "paired runtime policy lacks a controller-selected receiver protocol".to_owned(),
        );
    }
    let index = match coordinates.variant {
        Variant::Static => 0,
        Variant::Dynamic => 1,
    };
    context
        .runner
        .run_member(
            ExporterMemberSource {
                experiment_identity_bytes: context.experiment_identity_bytes,
                attempt_ordinal: 0,
                scenario_id: &coordinates.case.scenario,
                pair_id: coordinates.pair_id,
                member,
                build_artifact: &context.artifacts[index],
                build_receipt_bytes: &context.build_report.members[index].build_receipt_bytes,
                receiver_protocol: None,
            },
            workload.as_mut(),
        )
        .map_err(|error| error.to_string())
}

fn execute_member(
    case: &FrozenCasePlan,
    pair_id: &str,
    variant: Variant,
    artifact_path: &Path,
    build_report: &BuildPairReportV1,
    experiment_identity_blake3: &str,
    inherited_environment: &BTreeMap<String, String>,
) -> Result<MemberExecution, ControlledRuntimeError> {
    let artifact_blake3 = match variant {
        Variant::Static => &build_report.members[0].artifact_blake3,
        Variant::Dynamic => &build_report.members[1].artifact_blake3,
    };
    if digest_file(artifact_path, "runtime member artifact")? != *artifact_blake3 {
        return Err(ControlledRuntimeError::new(
            "runtime member artifact changed after identity sealing",
        ));
    }
    if case.command.len() < 5 || case.command[0] != "taskset" || case.command[1] != "-c" {
        return Err(ControlledRuntimeError::new(
            "checked-in command is not an executable runtime template",
        ));
    }
    let mut command = Command::new(TASKSET);
    command
        .args(["-c", case.command[2].as_str()])
        .arg(artifact_path)
        .args(&case.command[4..])
        .env_clear();
    for (name, value) in inherited_environment {
        command.env(name, value);
    }
    command
        .env(
            "AIPERF_PARITY_EXPERIMENT_IDENTITY",
            experiment_identity_blake3,
        )
        .env("AIPERF_PARITY_METRICS", case.measured_metrics.join(","))
        .env("AIPERF_PARITY_PAIR_ID", pair_id)
        .env(
            "AIPERF_PARITY_REQUEST_BUDGET",
            case.request_budget.to_string(),
        )
        .env("AIPERF_PARITY_SCENARIO", &case.scenario)
        .env(
            "AIPERF_PARITY_VARIANT",
            match variant {
                Variant::Static => "static",
                Variant::Dynamic => "dynamic",
            },
        );
    let output = command.output().map_err(|error| {
        ControlledRuntimeError::new(format!("cannot execute controlled runtime member: {error}"))
    })?;
    if !output.status.success() {
        return Ok(MemberExecution {
            outcome: MemberTerminalOutcome::Crash(format!("process exited with {}", output.status)),
            samples: Vec::new(),
            stdout_blake3: Some(digest(&output.stdout)),
        });
    }
    let stdout_blake3 = digest(&output.stdout);
    let decoded = match decode_member_output(
        &output.stdout,
        case,
        pair_id,
        variant,
        experiment_identity_blake3,
    ) {
        Ok(decoded) => decoded,
        Err(error) => {
            return Ok(MemberExecution {
                outcome: MemberTerminalOutcome::MalformedOutput(error.to_string()),
                samples: Vec::new(),
                stdout_blake3: Some(stdout_blake3),
            });
        }
    };
    let samples = decoded
        .metrics
        .into_iter()
        .map(|(metric, value)| PairedSample {
            scenario: case.scenario.clone(),
            pair_id: pair_id.to_owned(),
            variant,
            unit: metric_unit(&metric).to_owned(),
            metric,
            value,
            commit: build_report.source_commit.clone(),
            artifact_digest: artifact_blake3.clone(),
            experiment_identity_digest: experiment_identity_blake3.to_owned(),
        })
        .collect();
    Ok(MemberExecution {
        outcome: MemberTerminalOutcome::Completed,
        samples,
        stdout_blake3: Some(stdout_blake3),
    })
}

fn decode_member_output(
    bytes: &[u8],
    case: &FrozenCasePlan,
    pair_id: &str,
    variant: Variant,
    experiment_identity_blake3: &str,
) -> Result<RuntimeMemberOutputV1, ControlledRuntimeError> {
    let decoded: RuntimeMemberOutputV1 = serde_json::from_slice(bytes).map_err(|error| {
        ControlledRuntimeError::new(format!("member output is not schema-1 JSON: {error}"))
    })?;
    let mut canonical = serde_json_canonicalizer::to_vec(&decoded).map_err(|error| {
        ControlledRuntimeError::new(format!("cannot canonicalize member output: {error}"))
    })?;
    canonical.push(b'\n');
    if canonical != bytes {
        return Err(ControlledRuntimeError::new(
            "member output is not one exact canonical JCS line",
        ));
    }
    let metric_names = decoded.metrics.keys().cloned().collect::<Vec<_>>();
    let minimum_duration_nanoseconds = case
        .minimum_duration_seconds
        .checked_mul(1_000_000_000)
        .ok_or_else(|| ControlledRuntimeError::new("minimum duration overflow"))?;
    if decoded.schema_version != 1
        || decoded.scenario != case.scenario
        || decoded.pair_id != pair_id
        || decoded.variant != variant
        || decoded.experiment_identity_blake3 != experiment_identity_blake3
        || decoded.completed_budget != case.request_budget
        || decoded.active_duration_nanoseconds < minimum_duration_nanoseconds
        || metric_names != case.measured_metrics
        || decoded
            .metrics
            .values()
            .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(ControlledRuntimeError::new(
            "member output differs from its sealed scenario/member contract",
        ));
    }
    Ok(decoded)
}

fn runtime_report(
    evaluator: &ControlledMeasurementEvaluator,
    attempt_ledger: &mut AttemptLedger,
    context: &RuntimeReportContext<'_>,
    executed_member_count: usize,
    terminal_output_blake3: Vec<String>,
) -> Result<ControlledRuntimeReportV1, ControlledRuntimeError> {
    let decision = evaluator
        .history()
        .last()
        .map(|attempt| attempt.decision)
        .ok_or_else(|| {
            ControlledRuntimeError::new(
                "controlled runtime report requires one terminal attempt ledger entry",
            )
        })?;
    let terminal_attempt = evaluator.history().last().cloned().ok_or_else(|| {
        ControlledRuntimeError::new(
            "controlled runtime report requires one terminal attempt ledger entry",
        )
    })?;
    let attempt_evidence_bytes = evaluator.last_attempt_evidence_bytes().ok_or_else(|| {
        ControlledRuntimeError::new("controlled runtime report lacks exact attempt evidence bytes")
    })?;
    attempt_ledger.append_attempt(terminal_attempt, attempt_evidence_bytes)?;
    let statistical_report = evaluator.last_statistical_report().cloned();
    let attempt_history = evaluator.history().to_vec();
    let exporter_pair_history = evaluator.exporter_pair_history().to_vec();
    let retained_pair_count = evaluator
        .raw_pair_history()
        .iter()
        .filter(|record| record.decision == PairAttemptDecision::RetainPair)
        .count();
    let runtime_evidence_bytes = serde_json_canonicalizer::to_vec(&RuntimeEvidenceV1 {
        schema_version: 1,
        experiment_identity_blake3: context.experiment_identity_blake3,
        decision,
        statistical_report: statistical_report.as_ref(),
        attempt_history: &attempt_history,
        paired_build_record_blake3: &context.build_report.pair_record_blake3,
        observable_policy_blake3: context.observable_policy_blake3,
        receiver_protocol_authority_blake3: context.receiver_protocol_authority_blake3,
        output_schema_blake3: context.output_schema_blake3,
        workload_contract_blake3: context.workload_contract_blake3,
        corpus_blake3: context.corpus_blake3,
        scenario_count: context.scenario_count,
        retained_pair_count,
        executed_member_count,
        terminal_output_blake3: &terminal_output_blake3,
        exporter_pair_history: &exporter_pair_history,
    })
    .map_err(|error| {
        ControlledRuntimeError::new(format!("cannot canonicalize runtime evidence: {error}"))
    })?;
    let runtime_evidence_blake3 = digest(&runtime_evidence_bytes);
    Ok(ControlledRuntimeReportV1 {
        experiment_identity_blake3: context.experiment_identity_blake3.to_owned(),
        experiment_identity_bytes: context.experiment_identity_bytes.to_vec(),
        decision,
        statistical_report,
        attempt_history,
        paired_build_record_blake3: context.build_report.pair_record_blake3.clone(),
        observable_policy_blake3: context.observable_policy_blake3.to_owned(),
        receiver_protocol_authority_blake3: context.receiver_protocol_authority_blake3.to_owned(),
        output_schema_blake3: context.output_schema_blake3.to_owned(),
        workload_contract_blake3: context.workload_contract_blake3.to_owned(),
        corpus_blake3: context.corpus_blake3.to_owned(),
        scenario_count: context.scenario_count,
        retained_pair_count,
        executed_member_count,
        terminal_output_blake3,
        exporter_pair_history,
        runtime_evidence_bytes,
        runtime_evidence_blake3,
    })
}

fn artifact_for(variant: Variant, paths: &[PathBuf; 2]) -> &Path {
    match variant {
        Variant::Static => &paths[0],
        Variant::Dynamic => &paths[1],
    }
}

fn metric_unit(metric: &str) -> &'static str {
    match metric {
        "successful_requests_per_second" => "requests_per_second",
        "output_tokens_per_second" => "tokens_per_second",
        "cpu_nanoseconds_per_successful_request" | "exporter_nanoseconds_per_record" => {
            "nanoseconds"
        }
        "allocated_bytes_per_successful_request" => "bytes",
        "allocation_count_per_successful_request" => "allocations",
        _ => "milliseconds",
    }
}

fn validate_taskset() -> Result<String, ControlledRuntimeError> {
    let path = Path::new(TASKSET);
    let canonical = fs::canonicalize(path).map_err(|error| {
        ControlledRuntimeError::new(format!("cannot resolve controlled taskset: {error}"))
    })?;
    if canonical != path {
        return Err(ControlledRuntimeError::new(
            "controlled taskset path is not canonical",
        ));
    }
    digest_file(path, "taskset")
}

fn capture_environment() -> Result<BTreeMap<String, String>, ControlledRuntimeError> {
    std::env::vars_os()
        .map(|(name, value)| {
            let name = name.into_string().map_err(|_| {
                ControlledRuntimeError::new("runtime environment has a non-UTF-8 name")
            })?;
            let value = value.into_string().map_err(|_| {
                ControlledRuntimeError::new(format!(
                    "runtime environment value for {name} is not UTF-8"
                ))
            })?;
            Ok((name, value))
        })
        .collect()
}

fn canonical_digest<T: Serialize>(
    value: &T,
    label: &str,
) -> Result<String, ControlledRuntimeError> {
    let bytes = serde_json_canonicalizer::to_vec(value).map_err(|error| {
        ControlledRuntimeError::new(format!("cannot canonicalize {label}: {error}"))
    })?;
    Ok(digest(&bytes))
}

fn digest_file(path: &Path, label: &str) -> Result<String, ControlledRuntimeError> {
    let bytes = fs::read(path).map_err(|error| {
        ControlledRuntimeError::new(format!("cannot read {label} {}: {error}", path.display()))
    })?;
    Ok(digest(&bytes))
}

fn digest(bytes: &[u8]) -> String {
    format!("blake3:{}", blake3::hash(bytes))
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use super::*;

    fn invalid_attempt(ordinal: u8, evidence: &[u8]) -> ControlledAttemptRecord {
        ControlledAttemptRecord {
            ordinal,
            decision: ControlledAttemptDecision::Invalid,
            reason: Some(format!("invalid attempt {ordinal}")),
            report_blake3: None,
            evidence_tree_blake3: digest(evidence),
        }
    }

    #[test]
    fn ledger_hash_chain_survives_reopen_and_three_invalid_attempts_block() {
        let directory = tempfile::tempdir().expect("ledger directory");
        let path = directory.path().join("attempts.jsonl");
        let identity = "blake3:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        let mut previous_entry = None;

        for ordinal in 1..=3 {
            let evidence = format!("{{\"attempt\":{ordinal}}}").into_bytes();
            let mut ledger = AttemptLedger::acquire(&path, identity).expect("ledger reopens");
            assert_eq!(
                ledger.next_attempt_ordinal().expect("attempt is allowed"),
                ordinal
            );
            let entry = ledger
                .append_attempt(invalid_attempt(ordinal, &evidence), &evidence)
                .expect("invalid attempt appends");
            assert_eq!(entry.previous_entry_blake3, previous_entry);
            previous_entry = Some(entry.entry_blake3.clone());
        }

        let ledger = AttemptLedger::acquire(&path, identity).expect("ledger validates");
        let error = ledger
            .next_attempt_ordinal()
            .expect_err("three invalid attempts block another invocation");
        assert!(error.to_string().contains("three invalid attempts block"));
        assert_eq!(ledger.history().len(), 3);
    }

    #[test]
    fn child_deadline_kills_and_reaps_the_member() {
        let mut command = Command::new("/bin/sh");
        command.args(["-c", "trap '' TERM; while :; do :; done"]);
        let started = Instant::now();

        let result = execute_bounded_child(&mut command, Duration::from_millis(50), 4096)
            .expect("controller observes a terminal result");

        assert_eq!(result.terminal_status, ChildTerminalStatus::TimedOut);
        assert!(started.elapsed() < Duration::from_secs(2));
        assert_eq!(unsafe { libc::kill(result.pid, 0) }, -1);
        assert_eq!(
            std::io::Error::last_os_error().raw_os_error(),
            Some(libc::ESRCH)
        );
    }
}
