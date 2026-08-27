// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Same-process authority for the complete runtime parity matrix.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread::{JoinHandle, sleep};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::build_pair::{BuildPairReportV1, validate_authoritative_build_report_v1};
use crate::exporter_policy::parse_exporter_observable_policy;
use crate::exporter_runner::{
    CompletedExporterMember, ExporterHarnessRunner, ExporterMemberSource, ExporterWorkload,
};
use crate::plugin_stats::{
    ArtifactBoundExporterMemberV1, AuthoritativeIdentityInput, ControlledAttemptDecision,
    ControlledAttemptRecord, ControlledExporterPairRecord, ControlledMeasurementEvaluator,
    ExporterMember, ExporterMemberRecord, ExporterMemberSummary, ExporterObservableKind,
    ExporterSampleContract, FrozenCasePlan, InfrastructureEvent, MemberTerminalOutcome,
    PairAttemptDecision, PairedCase, PairedSample, RawMemberTerminalRecord, RawPairTerminalRecord,
    SimultaneousGateInput, SimultaneousGateReport, Variant, acquire_authoritative_identity,
    checked_in_case_plans, checked_in_inventory_digest, validate_exporter_member_evidence,
    validate_exporter_member_record,
};

const OUTPUT_SCHEMA_V1: &[u8] = b"plugin_runtime_member_output/v1;closed-jcs-line;scenario,pair_id,variant,experiment_identity_blake3,completed_budget,active_duration_nanoseconds,metrics";
/// Observable policy bound by controlled runtime execution.
///
/// The controlled runtime always executes both members of a pair, so its
/// exporter evidence is `paired`. The static-calibration policy is the
/// authority for the separate task-1 calibration measurement and cannot admit
/// a dynamic member at all.
const PAIRED_POLICY_BYTES: &[u8] =
    include_bytes!("../../benchmarks/exporter-paired-runtime-policy.json");
const TASKSET: &str = "/usr/bin/taskset";
const MAX_MEMBER_OUTPUT_BYTES: usize = 1024 * 1024;
const DEADLINE_MULTIPLIER: u32 = 4;

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
    /// Exact canonical evidence-tree bytes for this terminal attempt.
    pub attempt_evidence_tree_bytes: Vec<u8>,
    /// Digest of the exact canonical attempt evidence-tree bytes.
    pub attempt_evidence_tree_blake3: String,
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
    /// Bounded stdout/stderr and terminal status for every executed child.
    pub terminal_member_evidence: Vec<TerminalMemberEvidenceV1>,
    /// Complete validated exporter evidence retained by the controller.
    pub exporter_pair_history: Vec<ControlledExporterPairRecord>,
    /// Complete raw pair and replacement history in controller order.
    pub raw_pair_history: Vec<crate::plugin_stats::ControlledPairAttemptRecord>,
    /// Pair-start context retained from an invocation a different boot instance ran.
    pub resumed_pair_context: Option<PairStartContextV1>,
    /// Hash-chain identity of this invocation's terminal ledger entry.
    pub ledger_entry_blake3: String,
    /// Complete ordered ledger history, including every retained evidence tree.
    pub retained_attempt_evidence: Vec<RetainedAttemptEvidenceV1>,
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
    attempt_evidence_tree_blake3: &'a str,
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
    terminal_member_evidence: &'a [TerminalMemberEvidenceV1],
    raw_pair_history: &'a [crate::plugin_stats::ControlledPairAttemptRecord],
    exporter_pair_history: &'a [ControlledExporterPairRecord],
    resumed_pair_context: Option<&'a PairStartContextV1>,
    ledger_entry_blake3: &'a str,
    ledger_previous_entry_blake3: Option<&'a str>,
    retained_ledger_entry_blake3: &'a [&'a str],
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

/// Controller-owned facts one artifact-bound exporter child must reproduce.
///
/// Every field is sealed before the child starts. A child cannot widen its own
/// admission: the controller compares the child's declared binding against this
/// expectation before any exporter sample is constructed.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExporterChildExpectationV1 {
    /// Digest of the immutable experiment identity.
    pub experiment_identity_blake3: String,
    /// Zero-based complete-attempt ordinal owned by the controller.
    pub attempt_ordinal: u64,
    /// Frozen inventory scenario.
    pub scenario_id: String,
    /// Controller-scheduled pair identifier.
    pub pair_id: String,
    /// Static comparator or dynamic candidate.
    pub member: ExporterMember,
    /// Digest of the deterministic input corpus.
    pub corpus_blake3: String,
    /// Frozen observable class for this scenario.
    pub observable_kind: ExporterObservableKind,
    /// Digest of the immutable observable policy.
    pub observable_policy_blake3: String,
    /// Digest of the executable artifact the controller launched.
    pub build_artifact_blake3: String,
    /// Digest of the authenticated build receipt.
    pub build_receipt_blake3: String,
    /// Minimum summed active duration the frozen budget must reach.
    pub minimum_active_duration_ns: u64,
}

/// Complete artifact-bound exporter member line written by one child.
///
/// This is the only exporter output shape the controlled runner admits. A bare
/// `exporter_nanoseconds_per_record` metric is a product failure, not a sample.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExporterMemberChildOutputV1 {
    /// Complete sealed member receipts, evidence, and canonical record.
    pub artifact_bound: ArtifactBoundExporterMemberV1,
    /// Digest of the immutable experiment identity.
    pub experiment_identity_blake3: String,
    /// Controller-scheduled pair identifier.
    pub pair_id: String,
    /// Frozen inventory scenario.
    pub scenario: String,
    /// Output schema version, exactly one.
    pub schema_version: u8,
    /// Static comparator or dynamic candidate.
    pub variant: Variant,
}

/// One exporter child admitted against its controller expectation.
#[derive(Clone, Debug, PartialEq)]
pub struct AdmittedExporterChildV1 {
    /// Complete sealed member evidence retained for the pair record.
    pub artifact_bound: ArtifactBoundExporterMemberV1,
    /// Validated repetition receipts and derived per-record duration.
    pub summary: ExporterMemberSummary,
    /// Validated canonical post-run member record.
    pub record: ExporterMemberRecord,
}

/// Validate one artifact-bound exporter child line before it becomes a sample.
///
/// The line must be exactly one canonical JCS line carrying the complete sealed
/// member: sixteen repetition receipts, the retained raw/comparison/provenance
/// evidence, the frozen record counts, and a binding that matches every
/// controller-owned coordinate in `expectation`.
pub fn validate_exporter_member_child_output_v1(
    bytes: &[u8],
    expectation: &ExporterChildExpectationV1,
) -> Result<AdmittedExporterChildV1, ControlledRuntimeError> {
    let value: serde_json::Value = serde_json::from_slice(bytes).map_err(|error| {
        ControlledRuntimeError::new(format!("exporter member output is not JSON: {error}"))
    })?;
    let mut canonical = serde_json_canonicalizer::to_vec(&value).map_err(|error| {
        ControlledRuntimeError::new(format!(
            "cannot canonicalize exporter member output: {error}"
        ))
    })?;
    canonical.push(b'\n');
    if canonical != bytes {
        return Err(ControlledRuntimeError::new(
            "exporter member output is not one exact canonical JCS line",
        ));
    }
    let decoded: ExporterMemberChildOutputV1 = serde_json::from_value(value).map_err(|error| {
        ControlledRuntimeError::new(format!(
            "exporter member output is not a complete artifact-bound schema-1 line: {error}"
        ))
    })?;

    let expected_variant = match expectation.member {
        ExporterMember::Static => Variant::Static,
        ExporterMember::Dynamic => Variant::Dynamic,
    };
    let binding = &decoded.artifact_bound.binding;
    if decoded.schema_version != 1
        || decoded.variant != expected_variant
        || decoded.scenario != expectation.scenario_id
        || decoded.pair_id != expectation.pair_id
        || decoded.experiment_identity_blake3 != expectation.experiment_identity_blake3
        || binding.experiment_identity_blake3 != expectation.experiment_identity_blake3
        || binding.attempt_ordinal != expectation.attempt_ordinal
        || binding.scenario_id != expectation.scenario_id
        || binding.pair_id != expectation.pair_id
        || binding.member != expectation.member
        || binding.corpus_blake3 != expectation.corpus_blake3
        || binding.observable_kind != expectation.observable_kind
        || binding.observable_policy_blake3 != expectation.observable_policy_blake3
        || binding.build_artifact_blake3 != expectation.build_artifact_blake3
        || binding.build_receipt_blake3 != expectation.build_receipt_blake3
    {
        return Err(ControlledRuntimeError::new(
            "exporter member output does not match its controller expectation",
        ));
    }

    let contract = ExporterSampleContract::normative();
    let summary =
        validate_exporter_member_evidence(&contract, binding, &decoded.artifact_bound.evidence)
            .map_err(|error| ControlledRuntimeError::new(error.to_string()))?;
    let record = validate_exporter_member_record(
        &contract,
        binding,
        &decoded.artifact_bound.evidence,
        &decoded.artifact_bound.record_bytes,
    )
    .map_err(|error| ControlledRuntimeError::new(error.to_string()))?;
    if summary.active_duration_nanoseconds < expectation.minimum_active_duration_ns {
        return Err(ControlledRuntimeError::new(
            "exporter member active duration is shorter than its frozen budget",
        ));
    }
    if summary.processed_records != contract.processed_records
        || summary.retained_artifact_records != contract.retained_artifact_records
        || !summary.exporter_nanoseconds_per_record.is_finite()
        || summary.exporter_nanoseconds_per_record < 0.0
    {
        return Err(ControlledRuntimeError::new(
            "exporter member counts do not satisfy the frozen sample contract",
        ));
    }
    Ok(AdmittedExporterChildV1 {
        artifact_bound: decoded.artifact_bound,
        summary,
        record,
    })
}

struct MemberExecution {
    outcome: MemberTerminalOutcome,
    samples: Vec<PairedSample>,
    terminal_evidence: TerminalMemberEvidenceV1,
    artifact_bound: Option<ArtifactBoundExporterMemberV1>,
}

/// Controller-observed process termination.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ChildTerminalStatus {
    /// Process exited with this status code.
    Exited(i32),
    /// Process terminated from this signal.
    Signaled(i32),
    /// Controller deadline expired and the process group was killed and reaped.
    TimedOut,
}

/// One bounded child output stream.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct BoundedChildOutput {
    /// Retained prefix, never larger than the controller bound.
    pub bytes: Vec<u8>,
    /// BLAKE3 digest of the retained bytes.
    pub blake3: String,
    /// Whether additional bytes were drained and discarded.
    pub was_truncated: bool,
}

/// One retained ledger entry, complete enough to reconstruct its attempt.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct RetainedAttemptEvidenceV1 {
    /// Attempt this entry retains.
    pub attempt_ordinal: u8,
    /// Terminal decision of that attempt.
    pub decision: ControlledAttemptDecision,
    /// Reason retained with that decision.
    pub reason: Option<String>,
    /// Hash-chain identity of this entry.
    pub entry_blake3: String,
    /// Hash-chain identity of the entry before it.
    pub previous_entry_blake3: Option<String>,
    /// Exact canonical evidence-tree bytes retained for that attempt.
    pub evidence_tree_bytes: Vec<u8>,
}

/// Complete bounded terminal evidence for one runtime member.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct TerminalMemberEvidenceV1 {
    /// Frozen scenario.
    pub scenario: String,
    /// Scheduled pair or warmup identifier.
    pub pair_id: String,
    /// Static comparator or dynamic candidate.
    pub variant: Variant,
    /// Controller-observed process identifier.
    pub pid: libc::pid_t,
    /// Terminal process status.
    pub terminal_status: ChildTerminalStatus,
    /// Bounded standard output.
    pub stdout: BoundedChildOutput,
    /// Bounded standard error.
    pub stderr: BoundedChildOutput,
}

#[derive(Debug)]
struct BoundedChildResult {
    pid: libc::pid_t,
    terminal_status: ChildTerminalStatus,
    stdout: BoundedChildOutput,
    stderr: BoundedChildOutput,
    infrastructure_event: Option<InfrastructureEvent>,
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
    resumed_pair_context: Option<&'a PairStartContextV1>,
}

#[derive(Serialize)]
struct AttemptLedgerEntryPreimageV1<'a> {
    schema_version: u8,
    experiment_identity_blake3: &'a str,
    previous_entry_blake3: Option<&'a str>,
    attempt: &'a ControlledAttemptRecord,
    evidence_tree_bytes: &'a [u8],
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

/// Hash-chain identity of one appended ledger entry.
struct AppendedLedgerEntry {
    entry_blake3: String,
    previous_entry_blake3: Option<String>,
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
            experiment_identity_blake3: &entry.experiment_identity_blake3,
            previous_entry_blake3: entry.previous_entry_blake3.as_deref(),
            attempt: &entry.attempt,
            evidence_tree_bytes: &entry.evidence_tree_bytes,
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

    /// Take the complete ordered history, including every retained evidence
    /// tree. The ledger is finished with its entries once this returns.
    fn take_entries(&mut self) -> Vec<AttemptLedgerEntryV1> {
        std::mem::take(&mut self.entries)
    }

    fn append_attempt(
        &mut self,
        attempt: ControlledAttemptRecord,
        evidence_tree_bytes: Vec<u8>,
    ) -> Result<AppendedLedgerEntry, ControlledRuntimeError> {
        if attempt.ordinal != self.next_attempt_ordinal()? {
            return Err(ControlledRuntimeError::new(
                "attempt ledger append ordinal is not next",
            ));
        }
        if attempt.evidence_tree_blake3 != digest(&evidence_tree_bytes) {
            return Err(ControlledRuntimeError::new(
                "attempt ledger evidence bytes do not match their digest",
            ));
        }
        let previous_entry_blake3 = self.entries.last().map(|entry| entry.entry_blake3.clone());
        let entry_blake3 = canonical_digest(
            &AttemptLedgerEntryPreimageV1 {
                schema_version: 1,
                experiment_identity_blake3: &self.experiment_identity_blake3,
                previous_entry_blake3: previous_entry_blake3.as_deref(),
                attempt: &attempt,
                evidence_tree_bytes: &evidence_tree_bytes,
            },
            "attempt ledger preimage",
        )?;
        let entry = AttemptLedgerEntryV1 {
            schema_version: 1,
            experiment_identity_blake3: self.experiment_identity_blake3.clone(),
            previous_entry_blake3,
            attempt,
            evidence_tree_bytes,
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
        let appended = AppendedLedgerEntry {
            entry_blake3: entry.entry_blake3.clone(),
            previous_entry_blake3: entry.previous_entry_blake3.clone(),
        };
        self.entries.push(entry);
        Ok(appended)
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
///
/// `mock_server_pid_path` names the pid file of the mock server the members
/// are measured against. It is required for `mock_death_unrelated_to_member`
/// to be observable: the controller can only classify a mock-server death it
/// can see, and passing `None` runs with host-reboot observation alone.
pub fn run_controlled_runtime_with_ledger_v1(
    build_report: &BuildPairReportV1,
    attempt_ledger_path: &Path,
    mock_server_pid_path: Option<&Path>,
) -> Result<ControlledRuntimeReportV1, ControlledRuntimeError> {
    run_controlled_runtime_internal(
        build_report,
        None,
        PAIRED_POLICY_BYTES,
        attempt_ledger_path,
        &HostLivenessSourceV1::new(
            PathBuf::from(HOST_BOOT_IDENTITY_PATH),
            mock_server_pid_path.map(Path::to_path_buf),
        ),
    )
}

/// Refuse an exporter implementation that is unrelated to the acquired artifacts.
///
/// Exporter performance authority comes only from executing the exact artifact
/// descriptors the ledger-bound entry points validate, which are
/// [`run_controlled_runtime_with_ledger_v1`] and
/// [`run_controlled_runtime_with_liveness_v1`]. An in-process
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

/// Execute both members under an explicit controller-owned liveness source.
///
/// The controller, not the measured children, observes host and mock-server
/// liveness. Only these observations can raise `host_reboot` or
/// `mock_death_unrelated_to_member` for a pair.
pub fn run_controlled_runtime_with_liveness_v1(
    build_report: &BuildPairReportV1,
    attempt_ledger_path: &Path,
    liveness: &HostLivenessSourceV1,
) -> Result<ControlledRuntimeReportV1, ControlledRuntimeError> {
    run_controlled_runtime_internal(
        build_report,
        None,
        PAIRED_POLICY_BYTES,
        attempt_ledger_path,
        liveness,
    )
}

/// Host file naming the current boot instance.
const HOST_BOOT_IDENTITY_PATH: &str = "/proc/sys/kernel/random/boot_id";

/// Controller-owned sources for host and mock-server liveness observations.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HostLivenessSourceV1 {
    boot_identity_path: PathBuf,
    mock_server_pid_path: Option<PathBuf>,
}

impl HostLivenessSourceV1 {
    /// Observe the real host and, optionally, a mock server's pid file.
    pub fn new(boot_identity_path: PathBuf, mock_server_pid_path: Option<PathBuf>) -> Self {
        Self {
            boot_identity_path,
            mock_server_pid_path,
        }
    }

    /// Observe the real host with no mock server under observation.
    pub fn host_default() -> Self {
        Self {
            boot_identity_path: PathBuf::from(HOST_BOOT_IDENTITY_PATH),
            mock_server_pid_path: None,
        }
    }

    /// Capture one complete liveness observation.
    fn observe(&self) -> Result<HostLivenessObservationV1, ControlledRuntimeError> {
        let boot_identity = fs::read_to_string(&self.boot_identity_path)
            .map_err(|error| {
                ControlledRuntimeError::new(format!(
                    "cannot observe host boot identity {}: {error}",
                    self.boot_identity_path.display()
                ))
            })?
            .trim()
            .to_owned();
        if boot_identity.is_empty() {
            return Err(ControlledRuntimeError::new(
                "observed host boot identity is empty",
            ));
        }
        let mock_server = match &self.mock_server_pid_path {
            Some(path) => match fs::read_to_string(path) {
                Ok(text) => {
                    let pid = text.trim().parse::<i64>().map_err(|error| {
                        ControlledRuntimeError::new(format!(
                            "observed mock server pid is not an integer: {error}"
                        ))
                    })?;
                    observe_process_identity(pid)?
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
                Err(error) => {
                    return Err(ControlledRuntimeError::new(format!(
                        "cannot observe mock server pid file: {error}"
                    )));
                }
            },
            None => None,
        };
        Ok(HostLivenessObservationV1 {
            boot_identity,
            mock_server,
        })
    }
}

impl Default for HostLivenessSourceV1 {
    fn default() -> Self {
        Self::host_default()
    }
}

/// One controller-owned liveness observation.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct HostLivenessObservationV1 {
    /// Identity of the running boot instance.
    pub boot_identity: String,
    /// Mock-server identity, absent when no live mock server is observed.
    pub mock_server: Option<ObservedProcessIdentityV1>,
}

/// Identity of one observed process, stable across pid reuse.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ObservedProcessIdentityV1 {
    /// Observed process identifier.
    pub pid: i64,
    /// Kernel start time, which distinguishes a reused pid from the original.
    pub start_ticks: u64,
}

/// Observe one process identity, or absence when it is gone or unreaped.
fn observe_process_identity(
    pid: i64,
) -> Result<Option<ObservedProcessIdentityV1>, ControlledRuntimeError> {
    let stat = match fs::read_to_string(format!("/proc/{pid}/stat")) {
        Ok(stat) => stat,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(ControlledRuntimeError::new(format!(
                "cannot observe process {pid}: {error}"
            )));
        }
    };
    // The executable name is the only parenthesized field and may itself
    // contain spaces, so every positional field is read after the last ')'.
    let tail = stat
        .rsplit_once(')')
        .map(|(_, tail)| tail)
        .ok_or_else(|| ControlledRuntimeError::new("observed process stat record is malformed"))?;
    let mut fields = tail.split_whitespace();
    let state = fields
        .next()
        .ok_or_else(|| ControlledRuntimeError::new("observed process stat has no state field"))?;
    if state == "Z" {
        // A dead-but-unreaped process still has a /proc entry; for liveness
        // observation it is gone.
        return Ok(None);
    }
    // starttime is the 22nd stat field, which is index 19 after the name.
    let start_ticks = fields
        .nth(18)
        .ok_or_else(|| {
            ControlledRuntimeError::new("observed process stat has no start-time field")
        })?
        .parse::<u64>()
        .map_err(|error| {
            ControlledRuntimeError::new(format!(
                "observed process start time is not an integer: {error}"
            ))
        })?;
    Ok(Some(ObservedProcessIdentityV1 { pid, start_ticks }))
}

/// Classify one pair from the controller's own start and end observations.
fn pair_infrastructure_event(
    start: &HostLivenessObservationV1,
    end: &HostLivenessObservationV1,
) -> Option<InfrastructureEvent> {
    if start.boot_identity != end.boot_identity {
        return Some(InfrastructureEvent::HostReboot);
    }
    match (&start.mock_server, &end.mock_server) {
        // Only a mock server that was alive when the pair started can die
        // during it; one appearing mid-pair is not a member disturbance.
        (Some(observed), end) if Some(observed) != end.as_ref() => {
            Some(InfrastructureEvent::MockServerDeathUnrelatedToMember)
        }
        _ => None,
    }
}

/// Controller-owned context persisted at every pair start.
///
/// A host reboot destroys the running controller, so the only way to diagnose
/// one after the fact is to have written the pair's start context first.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PairStartContextV1 {
    /// Schema version of this persisted record.
    pub schema_version: u8,
    /// Sealed experiment identity that owns the interrupted pair.
    pub experiment_identity_blake3: String,
    /// Attempt the interrupted pair belonged to.
    pub attempt_ordinal: u64,
    /// Frozen inventory scenario of the interrupted pair.
    pub scenario: String,
    /// Seeded pair identifier of the interrupted pair.
    pub pair_id: String,
    /// Seeded member order of the interrupted pair.
    pub member_order: [Variant; 2],
    /// Liveness observed when the interrupted pair started.
    pub observed: HostLivenessObservationV1,
}

/// Resolve the one pair-start context owned by a sealed experiment identity.
fn pair_start_context_path(state_root: &Path, experiment_identity_blake3: &str) -> PathBuf {
    let identity = experiment_identity_blake3
        .strip_prefix("blake3:")
        .unwrap_or(experiment_identity_blake3);
    state_root
        .join(CONTROLLER_STATE_DIRECTORY)
        .join(format!("{identity}.pair-start.json"))
}

/// Persist one pair-start context, replacing the previous one atomically.
fn persist_pair_start_context(
    path: &Path,
    context: &PairStartContextV1,
) -> Result<(), ControlledRuntimeError> {
    let bytes = serde_json_canonicalizer::to_vec(context).map_err(|error| {
        ControlledRuntimeError::new(format!("cannot canonicalize pair-start context: {error}"))
    })?;
    let staged = path.with_extension("staged");
    fs::write(&staged, &bytes).map_err(|error| {
        ControlledRuntimeError::new(format!("cannot stage pair-start context: {error}"))
    })?;
    fs::rename(&staged, path).map_err(|error| {
        ControlledRuntimeError::new(format!("cannot publish pair-start context: {error}"))
    })
}

/// Read a retained pair-start context that a different boot instance wrote.
fn resumed_pair_context(
    path: &Path,
    observed: &HostLivenessObservationV1,
) -> Result<Option<PairStartContextV1>, ControlledRuntimeError> {
    let bytes = match fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(ControlledRuntimeError::new(format!(
                "cannot read retained pair-start context: {error}"
            )));
        }
    };
    let context: PairStartContextV1 = serde_json::from_slice(&bytes).map_err(|error| {
        ControlledRuntimeError::new(format!("retained pair-start context is malformed: {error}"))
    })?;
    if context.observed.boot_identity == observed.boot_identity {
        return Ok(None);
    }
    Ok(Some(context))
}

/// Directory holding every controller-owned attempt ledger under one root.
const CONTROLLER_STATE_DIRECTORY: &str = ".aiperf-parity-state";

/// Resolve the one ledger location owned by a sealed experiment identity.
///
/// The caller names only a state root. Every requested output path under the
/// same root therefore resolves to the same ledger, so one sealed identity can
/// never restart at attempt 1 through a new output path.
pub fn controlled_attempt_ledger_path(
    state_root: &Path,
    experiment_identity_blake3: &str,
) -> PathBuf {
    let identity = experiment_identity_blake3
        .strip_prefix("blake3:")
        .unwrap_or(experiment_identity_blake3);
    state_root
        .join(CONTROLLER_STATE_DIRECTORY)
        .join(format!("{identity}.jsonl"))
}

/// Reduce a caller-requested output path to the state root that owns it.
///
/// The final component is always ignored, whatever its shape: requested paths
/// `state/attempts`, `state/attempts.jsonl`, and `state/run.d` all name the one
/// root `state`. Inferring the shape from `Path::extension` instead would give
/// a single sealed identity two roots — `state` and `state/attempts` — which is
/// precisely the restart the identity-owned ledger exists to prevent.
fn controller_state_root(requested: &Path) -> &Path {
    match requested.parent() {
        // A bare filename names the current directory as its root.
        Some(parent) if parent.as_os_str().is_empty() => Path::new("."),
        Some(parent) => parent,
        // A filesystem root has no further component to drop.
        None => requested,
    }
}

fn run_controlled_runtime_internal(
    build_report: &BuildPairReportV1,
    mut exporter_factory: Option<&mut dyn ControlledExporterWorkloadFactory>,
    policy_bytes: &[u8],
    attempt_ledger_path: &Path,
    liveness: &HostLivenessSourceV1,
) -> Result<ControlledRuntimeReportV1, ControlledRuntimeError> {
    let artifact_paths = validate_authoritative_build_report_v1(build_report).map_err(|error| {
        ControlledRuntimeError::new(format!("invalid paired build authority: {error}"))
    })?;
    let mut cases =
        checked_in_case_plans().map_err(|error| ControlledRuntimeError::new(error.to_string()))?;
    // Exporter scenarios are the only cases whose admission depends on the
    // sealed observable policy rather than on the runtime matrix itself. Run
    // them last, deterministically, so a refusal there cannot hide the runtime
    // evidence the rest of the sealed matrix already produced.
    cases.sort_by_key(|case| {
        u8::from(
            case.measured_metrics
                .iter()
                .any(|metric| metric == "exporter_nanoseconds_per_record"),
        )
    });
    let cases = cases;
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
    // Sealed once, immediately after the experiment identity exists: every member
    // of this run is executed against exactly these facts.
    let sealed_run = SealedRunContext {
        build_report,
        experiment_identity_blake3: &experiment_identity_blake3,
        inherited_environment: &inherited_environment,
    };
    let exporter_identity_preimage_bytes = observed
        .identity_digest_preimage_bytes()
        .map_err(|error| ControlledRuntimeError::new(error.to_string()))?;
    let experiment_identity_bytes = observed
        .canonical_identity_bytes()
        .map_err(|error| ControlledRuntimeError::new(error.to_string()))?;
    let state_root = controller_state_root(attempt_ledger_path);
    let ledger_path = controlled_attempt_ledger_path(state_root, &experiment_identity_blake3);
    if let Some(parent) = ledger_path.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            ControlledRuntimeError::new(format!(
                "cannot create controller state root {}: {error}",
                parent.display()
            ))
        })?;
    }
    let pair_context_path = pair_start_context_path(state_root, &experiment_identity_blake3);
    let resumed_pair_context = resumed_pair_context(&pair_context_path, &liveness.observe()?)?;
    let mut attempt_ledger = AttemptLedger::acquire(&ledger_path, &experiment_identity_blake3)?;
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
        resumed_pair_context: resumed_pair_context.as_ref(),
    };

    // Exporter evidence numbers attempts from zero, while controller attempt
    // ordinals start at one. A child handed the controller's ordinal would be
    // refused by the pair evidence check it is meant to satisfy.
    let exporter_attempt_ordinal = u64::from(expected_attempt_ordinal)
        .checked_sub(1)
        .ok_or_else(|| ControlledRuntimeError::new("controller attempt ordinal underflow"))?;
    let expectation_context = ExporterExpectationContext {
        experiment_identity_blake3: &experiment_identity_blake3,
        attempt_ordinal: exporter_attempt_ordinal,
        corpus_blake3: &corpus_blake3,
        observable_policy_blake3: &observable_policy_blake3,
        policy: &policy,
        build_report,
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
    let mut terminal_member_evidence = Vec::new();
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
                            evaluator,
                            &mut attempt_ledger,
                            &report_context,
                            executed_member_count,
                            terminal_output_blake3,
                            terminal_member_evidence,
                        );
                    }
                } else {
                    let warmup_expectation = if is_exporter_case {
                        Some(exporter_expectation(
                            case,
                            &pair_id,
                            variant,
                            &expectation_context,
                        )?)
                    } else {
                        None
                    };
                    let MemberExecution {
                        outcome,
                        samples: _,
                        terminal_evidence,
                        artifact_bound: _,
                    } = execute_member(
                        case,
                        &pair_id,
                        variant,
                        artifact_for(variant, &artifact_paths),
                        &sealed_run,
                        warmup_expectation.as_ref(),
                    )?;
                    executed_member_count += 1;
                    retain_member_evidence(&mut evaluator, &terminal_evidence)?;
                    terminal_output_blake3.push(terminal_evidence.stdout.blake3.clone());
                    terminal_member_evidence.push(terminal_evidence);
                    if outcome != MemberTerminalOutcome::Completed {
                        evaluator
                            .finish_authoritative_product_failure(format!(
                                "warmup {pair_id} for {} {:?} failed: {:?}",
                                case.scenario, variant, outcome
                            ))
                            .map_err(|error| ControlledRuntimeError::new(error.to_string()))?;
                        return runtime_report(
                            evaluator,
                            &mut attempt_ledger,
                            &report_context,
                            executed_member_count,
                            terminal_output_blake3,
                            terminal_member_evidence,
                        );
                    }
                }
            }
        }

        let mut samples = Vec::new();
        for scheduled in &schedule {
            loop {
                let pair_start_liveness = liveness.observe()?;
                persist_pair_start_context(
                    &pair_context_path,
                    &PairStartContextV1 {
                        schema_version: 1,
                        experiment_identity_blake3: experiment_identity_blake3.clone(),
                        attempt_ordinal: u64::from(attempt_ordinal),
                        scenario: case.scenario.clone(),
                        pair_id: scheduled.pair_id.clone(),
                        member_order: scheduled.member_order,
                        observed: pair_start_liveness.clone(),
                    },
                )?;
                let mut member_records = Vec::with_capacity(2);
                let mut pair_samples = Vec::new();
                let mut completed_exporters = Vec::with_capacity(2);
                let mut admitted_exporters = Vec::with_capacity(2);
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
                                    .map_err(|error| {
                                        ControlledRuntimeError::new(error.to_string())
                                    })?;
                                return runtime_report(
                                    evaluator,
                                    &mut attempt_ledger,
                                    &report_context,
                                    executed_member_count,
                                    terminal_output_blake3,
                                    terminal_member_evidence,
                                );
                            }
                        }
                    } else {
                        let pair_expectation = if is_exporter_case {
                            Some(exporter_expectation(
                                case,
                                &scheduled.pair_id,
                                variant,
                                &expectation_context,
                            )?)
                        } else {
                            None
                        };
                        let MemberExecution {
                            outcome,
                            samples,
                            terminal_evidence,
                            artifact_bound,
                        } = execute_member(
                            case,
                            &scheduled.pair_id,
                            variant,
                            artifact_for(variant, &artifact_paths),
                            &sealed_run,
                            pair_expectation.as_ref(),
                        )?;
                        if let Some(artifact_bound) = artifact_bound {
                            admitted_exporters.push((variant, artifact_bound));
                        }
                        executed_member_count += 1;
                        let terminal_evidence_index = terminal_member_evidence.len();
                        retain_member_evidence(&mut evaluator, &terminal_evidence)?;
                        terminal_output_blake3.push(terminal_evidence.stdout.blake3.clone());
                        terminal_member_evidence.push(terminal_evidence);
                        member_records.push(RawMemberTerminalRecord {
                            variant,
                            outcome,
                            samples,
                            terminal_evidence_index: Some(terminal_evidence_index),
                        });
                        if let Some(record) = member_records.last() {
                            pair_samples.extend_from_slice(&record.samples);
                        }
                    }
                }
                // Both members completing means any disturbance the controller
                // observed around them is unrelated to the members themselves.
                if let Some(event) =
                    pair_infrastructure_event(&pair_start_liveness, &liveness.observe()?)
                    && member_records
                        .iter()
                        .all(|record| record.outcome == MemberTerminalOutcome::Completed)
                    && let Some(first) = member_records.first_mut()
                {
                    first.outcome = MemberTerminalOutcome::Infrastructure(event);
                }
                // An exporter member that lost its pinned affinity, or one
                // measured across a host reboot or a mock-server death, is
                // still returned with its admitted artifact-bound evidence.
                // Only a pair whose members both completed may be recorded as
                // an authoritative parity sample; anything else goes to the
                // ordinary pair path, which replaces the whole pair in seeded
                // member order.
                let is_pair_completed = member_records
                    .iter()
                    .all(|record| record.outcome == MemberTerminalOutcome::Completed);
                let decision = if is_pair_completed
                    && is_exporter_case
                    && exporter_factory.is_none()
                    && admitted_exporters.len() == 2
                {
                    let mut static_member = None;
                    let mut dynamic_member = None;
                    for (variant, admitted) in admitted_exporters {
                        match variant {
                            Variant::Static => static_member = Some(admitted),
                            Variant::Dynamic => dynamic_member = Some(admitted),
                        }
                    }
                    let static_member = static_member.ok_or_else(|| {
                        ControlledRuntimeError::new("static admitted exporter member is absent")
                    })?;
                    let dynamic_member = dynamic_member.ok_or_else(|| {
                        ControlledRuntimeError::new("dynamic admitted exporter member is absent")
                    })?;
                    evaluator
                        .record_artifact_bound_exporter_pair(&policy, static_member, dynamic_member)
                        .map_err(|error| ControlledRuntimeError::new(error.to_string()))?
                } else if is_pair_completed
                    && is_exporter_case
                    && exporter_factory.is_some()
                    && completed_exporters.len() == 2
                {
                    let mut static_member = None;
                    let mut dynamic_member = None;
                    for (variant, completed) in completed_exporters {
                        match variant {
                            Variant::Static => static_member = Some(completed),
                            Variant::Dynamic => dynamic_member = Some(completed),
                        }
                    }
                    let static_member = static_member.ok_or_else(|| {
                        ControlledRuntimeError::new("static completed exporter member is absent")
                    })?;
                    let dynamic_member = dynamic_member.ok_or_else(|| {
                        ControlledRuntimeError::new("dynamic completed exporter member is absent")
                    })?;
                    evaluator
                        .record_completed_exporter_pair(&policy, static_member, dynamic_member)
                        .map_err(|error| ControlledRuntimeError::new(error.to_string()))?
                } else {
                    evaluator
                        .record_pair(RawPairTerminalRecord {
                            scenario: case.scenario.clone(),
                            pair_id: scheduled.pair_id.clone(),
                            member_order: scheduled.member_order,
                            members: member_records,
                            asserted_reason: None,
                            asserted_disposition: None,
                        })
                        .map_err(|error| ControlledRuntimeError::new(error.to_string()))?
                };
                match decision {
                    PairAttemptDecision::RetainPair => {
                        samples.extend(pair_samples);
                        break;
                    }
                    PairAttemptDecision::ReplaceWholePair { member_order, .. }
                        if member_order == scheduled.member_order => {}
                    PairAttemptDecision::ReplaceWholePair { .. } => {
                        return Err(ControlledRuntimeError::new(
                            "controller replacement changed the seeded member order",
                        ));
                    }
                    PairAttemptDecision::AttemptInvalid | PairAttemptDecision::ExperimentFailed => {
                        return runtime_report(
                            evaluator,
                            &mut attempt_ledger,
                            &report_context,
                            executed_member_count,
                            terminal_output_blake3,
                            terminal_member_evidence,
                        );
                    }
                }
            }
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
            SimultaneousGateInput {
                cases: measured_cases,
            },
            observed,
        )
        .map_err(|error| ControlledRuntimeError::new(error.to_string()))?;
    runtime_report(
        evaluator,
        &mut attempt_ledger,
        &report_context,
        executed_member_count,
        terminal_output_blake3,
        terminal_member_evidence,
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

#[cfg(test)]
fn execute_bounded_child(
    command: &mut Command,
    deadline: Duration,
    output_limit: usize,
) -> Result<BoundedChildResult, ControlledRuntimeError> {
    execute_monitored_child(command, deadline, output_limit, None)
}

fn execute_monitored_child(
    command: &mut Command,
    deadline: Duration,
    output_limit: usize,
    expected_affinity: Option<&BTreeSet<usize>>,
) -> Result<BoundedChildResult, ControlledRuntimeError> {
    if deadline.is_zero() || output_limit == 0 {
        return Err(ControlledRuntimeError::new(
            "child deadline and output bound must be positive",
        ));
    }
    command.stdout(Stdio::piped()).stderr(Stdio::piped());
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt as _;

        // The controller owns the child's process group so timeout and terminal
        // cleanup cannot leave descendants holding the bounded output pipes.
        unsafe {
            command.pre_exec(|| {
                if libc::setpgid(0, 0) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }
    }
    let child = command.spawn().map_err(|error| {
        ControlledRuntimeError::new(format!("cannot spawn controlled runtime member: {error}"))
    })?;
    // Every fallible step below can return early. The guard owns the spawned
    // group from here on, so an early return still kills the group, reaps the
    // leader, closes the pipes, and joins both reader threads.
    let mut owned = OwnedChildGroup::new(child);
    let pid = libc::pid_t::try_from(owned.child.id())
        .map_err(|_| ControlledRuntimeError::new("runtime member PID does not fit pid_t"))?;
    owned.pid = Some(pid);
    record_owned_child_pid(pid);
    let stdout =
        owned.child.stdout.take().ok_or_else(|| {
            ControlledRuntimeError::new("runtime member stdout pipe was not created")
        })?;
    let stderr =
        owned.child.stderr.take().ok_or_else(|| {
            ControlledRuntimeError::new("runtime member stderr pipe was not created")
        })?;
    owned.stdout_reader = Some(drain_bounded_output(stdout, output_limit));
    owned.stderr_reader = Some(drain_bounded_output(stderr, output_limit));
    let started = Instant::now();
    let mut has_seen_expected_affinity = false;
    let mut infrastructure_event = None;
    let terminal_status = loop {
        injected_child_fault("poll")?;
        if let Some(status) = owned.child.try_wait().map_err(|error| {
            ControlledRuntimeError::new(format!("cannot poll runtime member: {error}"))
        })? {
            break child_terminal_status(status);
        }
        if infrastructure_event.is_none()
            && let Some(expected) = expected_affinity
        {
            injected_child_fault("affinity")?;
            if let Some(observed) = process_affinity(pid)? {
                if observed == *expected {
                    has_seen_expected_affinity = true;
                } else if has_seen_expected_affinity {
                    infrastructure_event = Some(InfrastructureEvent::AffinityLoss);
                }
            }
        }
        if started.elapsed() >= deadline {
            kill_process_group(pid)?;
            owned.child.wait().map_err(|error| {
                ControlledRuntimeError::new(format!(
                    "cannot reap timed-out runtime member: {error}"
                ))
            })?;
            break ChildTerminalStatus::TimedOut;
        }
        sleep(Duration::from_millis(5));
    };
    let (stdout, stderr) = owned.release()?;
    Ok(BoundedChildResult {
        pid,
        terminal_status,
        stdout,
        stderr,
        infrastructure_event,
    })
}

/// Owns one spawned member's process group, leader, and both bounded readers.
///
/// Cleanup is identical on the terminal path and on every early return: the
/// group dies before the leader is reaped, and both reader threads are joined
/// after the pipes are closed so a descendant holding a pipe cannot block.
struct OwnedChildGroup {
    child: std::process::Child,
    pid: Option<libc::pid_t>,
    is_cleaned: bool,
    stdout_reader: Option<JoinHandle<Result<BoundedChildOutput, std::io::Error>>>,
    stderr_reader: Option<JoinHandle<Result<BoundedChildOutput, std::io::Error>>>,
}

impl OwnedChildGroup {
    fn new(child: std::process::Child) -> Self {
        Self {
            child,
            pid: None,
            is_cleaned: false,
            stdout_reader: None,
            stderr_reader: None,
        }
    }

    /// Run the complete cleanup once and return both bounded spools.
    ///
    /// Every step runs even after an earlier one fails; the first error is the
    /// one reported, so a cleanup fault cannot mask the primary failure. The
    /// terminal path releases the guard and the guard is then dropped, so this
    /// is guarded against a second run: re-signalling a reaped pid would target
    /// whatever process group the host has since assigned that number.
    fn cleanup(
        &mut self,
    ) -> (
        Option<ControlledRuntimeError>,
        Option<BoundedChildOutput>,
        Option<BoundedChildOutput>,
    ) {
        if self.is_cleaned {
            return (None, None, None);
        }
        self.is_cleaned = true;
        let mut first_error = injected_child_fault("cleanup").err();
        let mut retain = |error: ControlledRuntimeError| {
            if first_error.is_none() {
                first_error = Some(error);
            }
        };
        match self.pid {
            Some(pid) => {
                if let Err(error) = kill_process_group_if_present(pid) {
                    retain(error);
                }
            }
            None => {
                // No usable pid_t means no group to signal; the leader is all
                // this controller can still reach.
                let _ = self.child.kill();
            }
        }
        if let Err(error) = self.child.wait() {
            retain(ControlledRuntimeError::new(format!(
                "cannot reap runtime member: {error}"
            )));
        }
        // Readers own the pipes once installed; close whatever is still held on
        // the child so a pending reader always observes EOF.
        drop(self.child.stdout.take());
        drop(self.child.stderr.take());
        if let Err(error) = injected_child_fault("output") {
            retain(error);
        }
        let stdout = self
            .stdout_reader
            .take()
            .map(|reader| join_bounded_output(reader, "stdout"));
        let stderr = self
            .stderr_reader
            .take()
            .map(|reader| join_bounded_output(reader, "stderr"));
        let stdout = match stdout {
            Some(Ok(output)) => Some(output),
            Some(Err(error)) => {
                retain(error);
                None
            }
            None => None,
        };
        let stderr = match stderr {
            Some(Ok(output)) => Some(output),
            Some(Err(error)) => {
                retain(error);
                None
            }
            None => None,
        };
        (first_error, stdout, stderr)
    }

    /// Consume the guard on the terminal path and surface both spools.
    fn release(
        mut self,
    ) -> Result<(BoundedChildOutput, BoundedChildOutput), ControlledRuntimeError> {
        let (error, stdout, stderr) = self.cleanup();
        if let Some(error) = error {
            return Err(error);
        }
        match (stdout, stderr) {
            (Some(stdout), Some(stderr)) => Ok((stdout, stderr)),
            _ => Err(ControlledRuntimeError::new(
                "runtime member output readers were not installed",
            )),
        }
    }
}

impl Drop for OwnedChildGroup {
    fn drop(&mut self) {
        // An early return already carries the primary error, so cleanup faults
        // here are discarded rather than masking it. After `release` this is a
        // no-op.
        let _ = self.cleanup();
    }
}

#[cfg(test)]
thread_local! {
    static INJECTED_CHILD_FAULT: std::cell::Cell<Option<&'static str>> =
        const { std::cell::Cell::new(None) };
    static OWNED_CHILD_PID: std::cell::Cell<Option<libc::pid_t>> =
        const { std::cell::Cell::new(None) };
}

/// Publish the group leader the controller took ownership of, so a test that
/// forces an early return can still assert the group was killed and reaped.
#[cfg(test)]
fn record_owned_child_pid(pid: libc::pid_t) {
    OWNED_CHILD_PID.with(|cell| cell.set(Some(pid)));
}

#[cfg(test)]
fn last_owned_child_pid() -> Option<libc::pid_t> {
    OWNED_CHILD_PID.with(std::cell::Cell::take)
}

#[cfg(not(test))]
#[inline]
fn record_owned_child_pid(_pid: libc::pid_t) {}

/// Arm or disarm a controller-stage fault for the current test thread.
#[cfg(test)]
fn set_injected_child_fault(stage: Option<&'static str>) {
    INJECTED_CHILD_FAULT.with(|cell| cell.set(stage));
}

#[cfg(test)]
fn injected_child_fault(stage: &'static str) -> Result<(), ControlledRuntimeError> {
    INJECTED_CHILD_FAULT.with(|cell| {
        if cell.get() == Some(stage) {
            return Err(ControlledRuntimeError::new(format!(
                "injected controller {stage} failure"
            )));
        }
        Ok(())
    })
}

#[cfg(not(test))]
#[inline]
fn injected_child_fault(_stage: &'static str) -> Result<(), ControlledRuntimeError> {
    Ok(())
}

#[cfg(target_os = "linux")]
fn process_affinity(pid: libc::pid_t) -> Result<Option<BTreeSet<usize>>, ControlledRuntimeError> {
    // Linux initializes every byte read by sched_getaffinity before the set is inspected.
    let mut affinity: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    // SAFETY: `affinity` is a live cpu_set_t and its exact allocation size is supplied.
    if unsafe {
        libc::sched_getaffinity(pid, std::mem::size_of::<libc::cpu_set_t>(), &mut affinity)
    } != 0
    {
        let error = std::io::Error::last_os_error();
        if error.raw_os_error() == Some(libc::ESRCH) {
            return Ok(None);
        }
        return Err(ControlledRuntimeError::new(format!(
            "cannot inspect runtime member CPU affinity: {error}"
        )));
    }
    let mut cpus = BTreeSet::new();
    for cpu in 0..libc::CPU_SETSIZE as usize {
        // SAFETY: `cpu` is strictly below CPU_SETSIZE and `affinity` is initialized above.
        if unsafe { libc::CPU_ISSET(cpu, &affinity) } {
            cpus.insert(cpu);
        }
    }
    Ok(Some(cpus))
}

#[cfg(not(target_os = "linux"))]
fn process_affinity(_pid: libc::pid_t) -> Result<Option<BTreeSet<usize>>, ControlledRuntimeError> {
    Err(ControlledRuntimeError::new(
        "controller CPU-affinity monitoring requires Linux",
    ))
}

fn parse_cpu_list(value: &str) -> Result<BTreeSet<usize>, ControlledRuntimeError> {
    let mut cpus = BTreeSet::new();
    for component in value.split(',') {
        if component.is_empty() {
            return Err(ControlledRuntimeError::new(
                "checked-in CPU-affinity list contains an empty component",
            ));
        }
        let (first, last) = match component.split_once('-') {
            Some((first, last)) if !first.is_empty() && !last.is_empty() => (
                first.parse::<usize>().map_err(|_| {
                    ControlledRuntimeError::new("checked-in CPU-affinity start is not an integer")
                })?,
                last.parse::<usize>().map_err(|_| {
                    ControlledRuntimeError::new("checked-in CPU-affinity end is not an integer")
                })?,
            ),
            Some(_) => {
                return Err(ControlledRuntimeError::new(
                    "checked-in CPU-affinity range is malformed",
                ));
            }
            None => {
                let cpu = component.parse::<usize>().map_err(|_| {
                    ControlledRuntimeError::new("checked-in CPU-affinity CPU is not an integer")
                })?;
                (cpu, cpu)
            }
        };
        if first > last || last >= libc::CPU_SETSIZE as usize {
            return Err(ControlledRuntimeError::new(
                "checked-in CPU-affinity range is outside cpu_set_t",
            ));
        }
        cpus.extend(first..=last);
    }
    if cpus.is_empty() {
        return Err(ControlledRuntimeError::new(
            "checked-in CPU-affinity list is empty",
        ));
    }
    Ok(cpus)
}

fn drain_bounded_output<R>(
    mut reader: R,
    limit: usize,
) -> JoinHandle<Result<BoundedChildOutput, std::io::Error>>
where
    R: Read + Send + 'static,
{
    std::thread::spawn(move || {
        let mut retained = Vec::with_capacity(limit.min(64 * 1024));
        let mut buffer = [0_u8; 64 * 1024];
        let mut was_truncated = false;
        loop {
            let count = reader.read(&mut buffer)?;
            if count == 0 {
                break;
            }
            let remaining = limit.saturating_sub(retained.len());
            let retained_count = remaining.min(count);
            retained.extend_from_slice(&buffer[..retained_count]);
            was_truncated |= retained_count != count;
        }
        Ok(BoundedChildOutput {
            blake3: digest(&retained),
            bytes: retained,
            was_truncated,
        })
    })
}

fn join_bounded_output(
    reader: JoinHandle<Result<BoundedChildOutput, std::io::Error>>,
    stream: &str,
) -> Result<BoundedChildOutput, ControlledRuntimeError> {
    reader
        .join()
        .map_err(|_| {
            ControlledRuntimeError::new(format!("runtime member {stream} reader panicked"))
        })?
        .map_err(|error| {
            ControlledRuntimeError::new(format!("cannot read runtime member {stream}: {error}"))
        })
}

fn child_terminal_status(status: std::process::ExitStatus) -> ChildTerminalStatus {
    if let Some(code) = status.code() {
        return ChildTerminalStatus::Exited(code);
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt as _;

        ChildTerminalStatus::Signaled(status.signal().unwrap_or_default())
    }
    #[cfg(not(unix))]
    ChildTerminalStatus::Exited(-1)
}

fn kill_process_group(pid: libc::pid_t) -> Result<(), ControlledRuntimeError> {
    #[cfg(unix)]
    {
        if unsafe { libc::kill(-pid, libc::SIGKILL) } == 0 {
            return Ok(());
        }
        let error = std::io::Error::last_os_error();
        if error.raw_os_error() == Some(libc::ESRCH) {
            return Ok(());
        }
        Err(ControlledRuntimeError::new(format!(
            "cannot kill runtime member process group: {error}"
        )))
    }
    #[cfg(not(unix))]
    Err(ControlledRuntimeError::new(
        "runtime member process-group termination is unavailable on this platform",
    ))
}

fn kill_process_group_if_present(pid: libc::pid_t) -> Result<(), ControlledRuntimeError> {
    kill_process_group(pid)
}

/// Seal the controller-owned expectation one exporter child must reproduce.
fn exporter_expectation(
    case: &FrozenCasePlan,
    pair_id: &str,
    variant: Variant,
    context: &ExporterExpectationContext<'_>,
) -> Result<ExporterChildExpectationV1, ControlledRuntimeError> {
    let (member, index) = match variant {
        Variant::Static => (ExporterMember::Static, 0),
        Variant::Dynamic => (ExporterMember::Dynamic, 1),
    };
    let observable_kind = context
        .policy
        .observable_kind(&case.scenario)
        .ok_or_else(|| {
            ControlledRuntimeError::new("exporter scenario has no authorized observable class")
        })?;
    let minimum_active_duration_ns = case
        .minimum_duration_seconds
        .checked_mul(1_000_000_000)
        .ok_or_else(|| ControlledRuntimeError::new("exporter minimum duration overflow"))?;
    Ok(ExporterChildExpectationV1 {
        experiment_identity_blake3: context.experiment_identity_blake3.to_owned(),
        attempt_ordinal: context.attempt_ordinal,
        scenario_id: case.scenario.clone(),
        pair_id: pair_id.to_owned(),
        member,
        corpus_blake3: context.corpus_blake3.to_owned(),
        observable_kind,
        observable_policy_blake3: context.observable_policy_blake3.to_owned(),
        build_artifact_blake3: context.build_report.members[index].artifact_blake3.clone(),
        build_receipt_blake3: context.build_report.members[index]
            .build_receipt_blake3
            .clone(),
        minimum_active_duration_ns,
    })
}

/// Sealed controller facts shared by every exporter child expectation.
struct ExporterExpectationContext<'a> {
    experiment_identity_blake3: &'a str,
    attempt_ordinal: u64,
    corpus_blake3: &'a str,
    observable_policy_blake3: &'a str,
    policy: &'a crate::exporter_policy::ExporterObservablePolicyV1,
    build_report: &'a BuildPairReportV1,
}

/// Run-scoped facts that are identical for every member of one sealed run.
///
/// Grouping them keeps `execute_member` addressed by what actually varies per
/// member (case, pair, variant, artifact, expectation) instead of restating the
/// sealed run at each call site.
struct SealedRunContext<'a> {
    build_report: &'a BuildPairReportV1,
    experiment_identity_blake3: &'a str,
    inherited_environment: &'a BTreeMap<String, String>,
}

fn execute_member(
    case: &FrozenCasePlan,
    pair_id: &str,
    variant: Variant,
    artifact_path: &Path,
    run: &SealedRunContext<'_>,
    exporter_expectation: Option<&ExporterChildExpectationV1>,
) -> Result<MemberExecution, ControlledRuntimeError> {
    let build_report = run.build_report;
    let experiment_identity_blake3 = run.experiment_identity_blake3;
    let inherited_environment = run.inherited_environment;
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
    // The controller owns every value in the expectation and re-checks the
    // returned evidence against its own copy, so handing it to the child
    // grants no authority: it only lets a conforming child name the digests
    // the controller will require of it.
    if let Some(expectation) = exporter_expectation {
        let expectation_bytes = serde_json_canonicalizer::to_vec(expectation).map_err(|error| {
            ControlledRuntimeError::new(format!(
                "cannot canonicalize exporter child expectation: {error}"
            ))
        })?;
        let expectation_json = String::from_utf8(expectation_bytes).map_err(|_| {
            ControlledRuntimeError::new("canonical exporter child expectation is not UTF-8")
        })?;
        command.env("AIPERF_PARITY_EXPORTER_EXPECTATION", expectation_json);
    }
    let deadline = Duration::from_secs(case.minimum_duration_seconds)
        .checked_mul(DEADLINE_MULTIPLIER)
        .ok_or_else(|| ControlledRuntimeError::new("runtime member deadline overflow"))?;
    let expected_affinity = parse_cpu_list(&case.command[2])?;
    let result = execute_monitored_child(
        &mut command,
        deadline,
        MAX_MEMBER_OUTPUT_BYTES,
        Some(&expected_affinity),
    )?;
    let infrastructure_event = result.infrastructure_event;
    let terminal_evidence = TerminalMemberEvidenceV1 {
        scenario: case.scenario.clone(),
        pair_id: pair_id.to_owned(),
        variant,
        pid: result.pid,
        terminal_status: result.terminal_status.clone(),
        stdout: result.stdout,
        stderr: result.stderr,
    };
    if terminal_evidence.terminal_status == ChildTerminalStatus::TimedOut {
        return Ok(MemberExecution {
            outcome: MemberTerminalOutcome::Timeout(format!(
                "controller deadline expired after {} seconds",
                deadline.as_secs()
            )),
            samples: Vec::new(),
            terminal_evidence,
            artifact_bound: None,
        });
    }
    if terminal_evidence.terminal_status != ChildTerminalStatus::Exited(0) {
        return Ok(MemberExecution {
            outcome: MemberTerminalOutcome::Crash(format!(
                "process terminated with {:?}",
                terminal_evidence.terminal_status
            )),
            samples: Vec::new(),
            terminal_evidence,
            artifact_bound: None,
        });
    }
    let is_exporter_case = case
        .measured_metrics
        .iter()
        .any(|metric| metric == "exporter_nanoseconds_per_record");
    if is_exporter_case {
        // A child that reports the bare metric is a product failure: it never
        // produced the sealed evidence the exporter authority is defined over.
        if decode_member_output(
            &terminal_evidence.stdout.bytes,
            case,
            pair_id,
            variant,
            experiment_identity_blake3,
        )
        .is_ok_and(|bare| bare.metrics.contains_key("exporter_nanoseconds_per_record"))
        {
            return Ok(MemberExecution {
                outcome: MemberTerminalOutcome::MalformedOutput(
                    "bare exporter metric lacks complete artifact-bound sealed evidence".to_owned(),
                ),
                samples: Vec::new(),
                terminal_evidence,
                artifact_bound: None,
            });
        }
        let expectation = exporter_expectation.ok_or_else(|| {
            ControlledRuntimeError::new(
                "exporter member has no sealed controller expectation to admit against",
            )
        })?;
        let admitted = match validate_exporter_member_child_output_v1(
            &terminal_evidence.stdout.bytes,
            expectation,
        ) {
            Ok(admitted) => admitted,
            Err(error) => {
                return Ok(MemberExecution {
                    outcome: MemberTerminalOutcome::MalformedOutput(error.to_string()),
                    samples: Vec::new(),
                    terminal_evidence,
                    artifact_bound: None,
                });
            }
        };
        let samples = vec![PairedSample {
            scenario: case.scenario.clone(),
            pair_id: pair_id.to_owned(),
            variant,
            unit: metric_unit("exporter_nanoseconds_per_record").to_owned(),
            metric: "exporter_nanoseconds_per_record".to_owned(),
            value: admitted.summary.exporter_nanoseconds_per_record,
            commit: build_report.source_commit.clone(),
            artifact_digest: artifact_blake3.clone(),
            experiment_identity_digest: experiment_identity_blake3.to_owned(),
        }];
        return Ok(MemberExecution {
            outcome: infrastructure_event.map_or(
                MemberTerminalOutcome::Completed,
                MemberTerminalOutcome::Infrastructure,
            ),
            samples,
            terminal_evidence,
            artifact_bound: Some(admitted.artifact_bound),
        });
    }
    let decoded = match decode_member_output(
        &terminal_evidence.stdout.bytes,
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
                terminal_evidence,
                artifact_bound: None,
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
        outcome: infrastructure_event.map_or(
            MemberTerminalOutcome::Completed,
            MemberTerminalOutcome::Infrastructure,
        ),
        samples,
        terminal_evidence,
        artifact_bound: None,
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
    mut evaluator: ControlledMeasurementEvaluator,
    attempt_ledger: &mut AttemptLedger,
    context: &RuntimeReportContext<'_>,
    executed_member_count: usize,
    terminal_output_blake3: Vec<String>,
    terminal_member_evidence: Vec<TerminalMemberEvidenceV1>,
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
    let attempt_evidence_tree_bytes =
        evaluator
            .take_last_attempt_evidence_bytes()
            .ok_or_else(|| {
                ControlledRuntimeError::new(
                    "controlled runtime report lacks exact attempt evidence bytes",
                )
            })?;
    let attempt_evidence_tree_blake3 = digest(&attempt_evidence_tree_bytes);
    let appended = attempt_ledger.append_attempt(terminal_attempt, attempt_evidence_tree_bytes)?;
    // The ledger is finished with its entries here, so every retained evidence
    // tree moves out of it. The report additionally names the terminal tree on
    // its own field, and that one tree is therefore copied exactly once below.
    let retained_attempt_evidence = attempt_ledger
        .take_entries()
        .into_iter()
        .map(|entry| RetainedAttemptEvidenceV1 {
            attempt_ordinal: entry.attempt.ordinal,
            decision: entry.attempt.decision,
            reason: entry.attempt.reason,
            entry_blake3: entry.entry_blake3,
            previous_entry_blake3: entry.previous_entry_blake3,
            evidence_tree_bytes: entry.evidence_tree_bytes,
        })
        .collect::<Vec<_>>();
    let attempt_evidence_tree_bytes = retained_attempt_evidence
        .last()
        .map(|terminal| terminal.evidence_tree_bytes.clone())
        .ok_or_else(|| ControlledRuntimeError::new("attempt ledger retained no terminal entry"))?;
    let retained_pair_count = evaluator
        .raw_pair_history()
        .iter()
        .filter(|record| record.decision == PairAttemptDecision::RetainPair)
        .count();
    let runtime_evidence_bytes = serde_json_canonicalizer::to_vec(&RuntimeEvidenceV1 {
        schema_version: 1,
        experiment_identity_blake3: context.experiment_identity_blake3,
        decision,
        statistical_report: evaluator.last_statistical_report(),
        attempt_history: evaluator.history(),
        attempt_evidence_tree_blake3: &attempt_evidence_tree_blake3,
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
        terminal_member_evidence: &terminal_member_evidence,
        raw_pair_history: evaluator.raw_pair_history(),
        exporter_pair_history: evaluator.exporter_pair_history(),
        resumed_pair_context: context.resumed_pair_context,
        ledger_entry_blake3: &appended.entry_blake3,
        ledger_previous_entry_blake3: appended.previous_entry_blake3.as_deref(),
        retained_ledger_entry_blake3: &retained_attempt_evidence
            .iter()
            .map(|entry| entry.entry_blake3.as_str())
            .collect::<Vec<_>>(),
    })
    .map_err(|error| {
        ControlledRuntimeError::new(format!("cannot canonicalize runtime evidence: {error}"))
    })?;
    let runtime_evidence_blake3 = digest(&runtime_evidence_bytes);
    let parts = evaluator
        .into_report_parts()
        .map_err(|error| ControlledRuntimeError::new(error.to_string()))?;
    Ok(ControlledRuntimeReportV1 {
        experiment_identity_blake3: context.experiment_identity_blake3.to_owned(),
        experiment_identity_bytes: context.experiment_identity_bytes.to_vec(),
        decision,
        statistical_report: parts.statistical_report,
        attempt_history: parts.history,
        attempt_evidence_tree_bytes,
        attempt_evidence_tree_blake3,
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
        terminal_member_evidence,
        exporter_pair_history: parts.exporter_pair_history,
        raw_pair_history: parts.raw_pair_history,
        resumed_pair_context: context.resumed_pair_context.cloned(),
        ledger_entry_blake3: appended.entry_blake3,
        retained_attempt_evidence,
        runtime_evidence_bytes,
        runtime_evidence_blake3,
    })
}

/// Retain one member's terminal evidence inside the active attempt's tree.
fn retain_member_evidence(
    evaluator: &mut ControlledMeasurementEvaluator,
    evidence: &TerminalMemberEvidenceV1,
) -> Result<(), ControlledRuntimeError> {
    let value = serde_json::to_value(evidence).map_err(|error| {
        ControlledRuntimeError::new(format!("cannot retain member terminal evidence: {error}"))
    })?;
    evaluator
        .retain_terminal_member_evidence(value)
        .map_err(|error| ControlledRuntimeError::new(error.to_string()))
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
            ledger
                .append_attempt(invalid_attempt(ordinal, &evidence), evidence.clone())
                .expect("invalid attempt appends");
            let entry = ledger.entries.last().expect("appended entry is retained");
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

    /// Read the descendant pid the leader published before it exited.
    ///
    /// The leader writes this file and only then exits, and the controller
    /// cannot begin cleanup before the leader exits, so the marker is already
    /// complete once the controller returns. Polling for it would race the
    /// cleanup this fixture exists to observe.
    fn published_descendant_pid(marker: &Path) -> libc::pid_t {
        let text = fs::read_to_string(marker)
            .expect("the leader published its descendant pid before exiting");
        text.trim()
            .parse::<libc::pid_t>()
            .expect("the published descendant pid is a pid_t")
    }

    fn assert_reaped(pid: libc::pid_t) {
        for _ in 0..400 {
            if unsafe { libc::kill(pid, 0) } == -1
                && std::io::Error::last_os_error().raw_os_error() == Some(libc::ESRCH)
            {
                return;
            }
            sleep(Duration::from_millis(5));
        }
        panic!("process {pid} outlived its controller");
    }

    #[test]
    fn injected_stage_failures_still_kill_and_reap_the_owned_group() {
        for stage in ["poll", "affinity", "cleanup", "output"] {
            let mut command = Command::new("/bin/sh");
            command.args(["-c", "trap '' TERM; sleep 30"]);
            set_injected_child_fault(Some(stage));
            let error = execute_monitored_child(
                &mut command,
                Duration::from_millis(300),
                4096,
                Some(&BTreeSet::from([0_usize])),
            )
            .expect_err("an injected controller failure is surfaced");
            set_injected_child_fault(None);
            assert!(
                error.to_string().contains(stage),
                "stage {stage} produced {error}"
            );
            let pid = last_owned_child_pid().expect("the controller took ownership of a leader");
            assert_reaped(pid);
        }
    }

    #[test]
    fn owned_group_cleanup_runs_exactly_once() {
        let mut child = Command::new("/bin/sh")
            .args(["-c", "echo released"])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("the fixture leader spawns");
        let pid = libc::pid_t::try_from(child.id()).expect("the fixture pid fits pid_t");
        let stdout = child.stdout.take().expect("the fixture stdout pipe exists");
        let stderr = child.stderr.take().expect("the fixture stderr pipe exists");
        let mut owned = OwnedChildGroup::new(child);
        owned.pid = Some(pid);
        owned.stdout_reader = Some(drain_bounded_output(stdout, 4096));
        owned.stderr_reader = Some(drain_bounded_output(stderr, 4096));

        let (first_error, first_stdout, first_stderr) = owned.cleanup();
        assert!(first_error.is_none(), "{first_error:?}");
        assert_eq!(
            first_stdout
                .expect("the first cleanup surfaces stdout")
                .bytes,
            b"released\n"
        );
        assert!(first_stderr.is_some());

        // A second run would signal a pid the host has already recycled.
        let (second_error, second_stdout, second_stderr) = owned.cleanup();
        assert!(second_error.is_none(), "{second_error:?}");
        assert!(second_stdout.is_none());
        assert!(second_stderr.is_none());
    }

    #[test]
    fn a_descendant_holding_the_output_pipe_cannot_outlive_the_controller() {
        let directory = tempfile::tempdir().expect("descendant fixture directory");
        let marker = directory.path().join("descendant-pid");
        let mut command = Command::new("/bin/sh");
        // The leader publishes the descendant pid through an atomic rename and
        // only then exits, so the pid is observable without racing against the
        // controller cleanup this test exists to observe. `exec` keeps the
        // published pid holding the inherited output pipe.
        command.args([
            "-c",
            &format!(
                "sh -c 'exec sleep 30' & printf '%s' \"$!\" > \"{0}.tmp\"; mv \"{0}.tmp\" \"{0}\"; exit 0",
                marker.display()
            ),
        ]);
        let started = Instant::now();

        let result = execute_bounded_child(&mut command, Duration::from_secs(10), 4096)
            .expect("the controller reaches a terminal result");

        assert_eq!(result.terminal_status, ChildTerminalStatus::Exited(0));
        assert!(started.elapsed() < Duration::from_secs(5));
        assert_reaped(published_descendant_pid(&marker));
    }

    #[test]
    fn child_output_spools_are_bounded_and_retain_both_digests() {
        let mut command = Command::new("/bin/sh");
        command.args([
            "-c",
            "i=0; while [ $i -lt 10000 ]; do printf 'stdout-payload'; printf 'stderr-payload' >&2; i=$((i + 1)); done",
        ]);

        let result = execute_bounded_child(&mut command, Duration::from_secs(2), 4096)
            .expect("chatty member reaches a terminal result");

        assert!(matches!(
            result.terminal_status,
            ChildTerminalStatus::Exited(_) | ChildTerminalStatus::Signaled(_)
        ));
        assert!(result.stdout.bytes.len() <= 4096);
        assert!(result.stderr.bytes.len() <= 4096);
        assert_eq!(result.stdout.blake3, digest(&result.stdout.bytes));
        assert_eq!(result.stderr.blake3, digest(&result.stderr.bytes));
        assert!(result.stdout.was_truncated || result.stderr.was_truncated);
    }
}
