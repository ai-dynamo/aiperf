// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Harness-owned exporter workload execution and observable acquisition.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;
use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::time::Instant;

use serde::Serialize;

use crate::exporter_observable::{
    ArtifactTreeEntry, ArtifactTreeKind, ReceiverBody, ReceiverBodyEncoding,
    ReceiverTranscriptEntry, validate_artifact_tree_path, validate_receiver_metadata,
};
use crate::exporter_policy::{
    AuthenticatedReceiverProtocolV1, ExporterObservablePolicyV1, ProvenanceBindingV1,
    SelectedBackingPayloadV1, apply_exporter_observable_policy_v1,
};
use crate::plugin_stats::{
    ExporterEvidenceMode, ExporterMember, ExporterMemberBinding, ExporterMemberEvidence,
    ExporterMemberRecord, ExporterMemberSummary, ExporterObservableKind, ExporterRepetitionReceipt,
    ExporterSampleContract, RetainedExporterEvidence, validate_exporter_member_evidence,
    validate_exporter_member_record,
};

const CORPUS_RECORDS: u64 = 100_000;
const REPETITIONS: usize = 16;
const RETAINED_REPETITION: usize = 0;

/// Typed product or acquisition failure from the controlled exporter runner.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExporterHarnessError {
    message: String,
}

impl ExporterHarnessError {
    /// Construct an exporter-owned product error at the harness boundary.
    pub fn product(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    fn acquisition(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for ExporterHarnessError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for ExporterHarnessError {}

/// Non-digest source material and coordinates for one controlled member.
///
/// The runner hashes every identity-bearing byte source itself. Callers cannot
/// supply a binding, duration, receipt, evidence digest, or member record.
#[derive(Clone, Copy, Debug)]
pub struct ExporterMemberSource<'a> {
    /// Exact canonical pre-run experiment-identity bytes.
    pub experiment_identity_bytes: &'a [u8],
    /// Zero-based complete-attempt ordinal assigned by the controller.
    pub attempt_ordinal: u64,
    /// Frozen policy and inventory scenario.
    pub scenario_id: &'a str,
    /// Controller-scheduled pair identifier.
    pub pair_id: &'a str,
    /// Static comparator or dynamic candidate.
    pub member: ExporterMember,
    /// Already-acquired executable artifact whose descriptor the runner hashes.
    pub build_artifact: &'a File,
    /// Authenticated build receipt bytes whose digest the runner binds.
    pub build_receipt_bytes: &'a [u8],
    /// Controller-authenticated receiver identity; required only for receiver scenarios.
    pub receiver_protocol: Option<&'a AuthenticatedReceiverProtocolV1>,
}

/// Borrowed immutable record from the harness's fixed corpus.
#[derive(Clone, Copy, Debug)]
pub struct ExporterCorpusRecord<'a> {
    ordinal: u64,
    jsonl_bytes: &'a [u8],
}

impl ExporterCorpusRecord<'_> {
    /// Dense record ordinal in `0..100000`.
    pub fn ordinal(&self) -> u64 {
        self.ordinal
    }

    /// Exact canonical JSONL record bytes, including the trailing newline.
    pub fn jsonl_bytes(&self) -> &[u8] {
        self.jsonl_bytes
    }
}

/// Single-pass cursor over the harness-owned deterministic corpus.
#[derive(Debug)]
pub struct ExporterRecordStream<'a> {
    records: &'a [Vec<u8>],
    next: usize,
}

impl ExporterRecordStream<'_> {
    fn processed_records(&self) -> u64 {
        self.next as u64
    }
}

impl<'a> Iterator for ExporterRecordStream<'a> {
    type Item = ExporterCorpusRecord<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let ordinal = self.next;
        let bytes = self.records.get(ordinal)?;
        self.next += 1;
        Some(ExporterCorpusRecord {
            ordinal: ordinal as u64,
            jsonl_bytes: bytes,
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.records.len() - self.next;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for ExporterRecordStream<'_> {}

/// Exporter adapter invoked exactly once for each controlled repetition.
pub trait ExporterWorkload {
    /// Consume the single-pass fixed corpus and write through host capture.
    fn export(
        &mut self,
        repetition_ordinal: u64,
        records: &mut ExporterRecordStream<'_>,
        capture: &mut HostExporterCapture,
    ) -> Result<(), ExporterHarnessError>;
}

/// Acknowledgement returned only after the receiver has retained the request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReceiverAcknowledgement {
    sequence: u64,
    recorded_acceptances: usize,
}

impl ReceiverAcknowledgement {
    /// Dense sequence assigned to the accepted request.
    pub fn sequence(&self) -> u64 {
        self.sequence
    }

    /// Receiver transcript length after this request was recorded.
    pub fn recorded_acceptances(&self) -> usize {
        self.recorded_acceptances
    }
}

struct ArtifactCapture {
    root: tempfile::TempDir,
    files: BTreeMap<String, File>,
    explicit_directories: BTreeSet<String>,
}

enum CaptureStorage {
    ArtifactTree(ArtifactCapture),
    CapturedStream(BufWriter<File>),
    ReceiverTranscript {
        protocol: AuthenticatedReceiverProtocolV1,
        entries: Vec<ReceiverTranscriptEntry>,
        bodies: Vec<Vec<u8>>,
    },
}

/// Capability-limited host storage for one exporter repetition.
pub struct HostExporterCapture {
    storage: CaptureStorage,
}

impl HostExporterCapture {
    fn new(
        kind: ExporterObservableKind,
        receiver_protocol: Option<&AuthenticatedReceiverProtocolV1>,
    ) -> Result<Self, ExporterHarnessError> {
        let storage = match kind {
            ExporterObservableKind::ArtifactTree => CaptureStorage::ArtifactTree(ArtifactCapture {
                root: tempfile::tempdir().map_err(io_error("create artifact capture root"))?,
                files: BTreeMap::new(),
                explicit_directories: BTreeSet::new(),
            }),
            ExporterObservableKind::CapturedStream => CaptureStorage::CapturedStream(
                BufWriter::new(tempfile::tempfile().map_err(io_error("create stream capture"))?),
            ),
            ExporterObservableKind::ReceiverTranscript => CaptureStorage::ReceiverTranscript {
                protocol: receiver_protocol.cloned().ok_or_else(|| {
                    ExporterHarnessError::acquisition(
                        "receiver capture lacks its authenticated protocol identity",
                    )
                })?,
                entries: Vec::new(),
                bodies: Vec::new(),
            },
        };
        Ok(Self { storage })
    }

    /// Create one empty artifact directory beneath the host-owned root.
    pub fn create_artifact_directory(&mut self, path: &str) -> Result<(), ExporterHarnessError> {
        validate_artifact_tree_path(path).map_err(|error| {
            ExporterHarnessError::product(format!("invalid artifact directory path: {error}"))
        })?;
        let CaptureStorage::ArtifactTree(capture) = &mut self.storage else {
            return Err(ExporterHarnessError::product(
                "artifact-directory output used for a different observable class",
            ));
        };
        if capture.files.contains_key(path) || !capture.explicit_directories.insert(path.to_owned())
        {
            return Err(ExporterHarnessError::product(
                "artifact directory path is duplicate or conflicts with a file",
            ));
        }
        std::fs::create_dir_all(capture.root.path().join(path))
            .map_err(io_error("create artifact directory"))
    }

    /// Write one regular artifact through a harness-owned file handle.
    pub fn write_artifact(&mut self, path: &str, bytes: &[u8]) -> Result<(), ExporterHarnessError> {
        validate_artifact_tree_path(path).map_err(|error| {
            ExporterHarnessError::product(format!("invalid artifact file path: {error}"))
        })?;
        let CaptureStorage::ArtifactTree(capture) = &mut self.storage else {
            return Err(ExporterHarnessError::product(
                "artifact output used for a different observable class",
            ));
        };
        if capture.files.contains_key(path) || capture.explicit_directories.contains(path) {
            return Err(ExporterHarnessError::product(
                "artifact file path is duplicate or conflicts with a directory",
            ));
        }
        let absolute = capture.root.path().join(path);
        if let Some(parent) = absolute.parent() {
            std::fs::create_dir_all(parent).map_err(io_error("create artifact parent"))?;
        }
        let mut file = File::create(&absolute).map_err(io_error("create artifact file"))?;
        file.write_all(bytes)
            .map_err(io_error("write artifact file"))?;
        capture.files.insert(path.to_owned(), file);
        Ok(())
    }

    /// Write exact bytes to the harness-owned captured stream.
    pub fn write_stream(&mut self, bytes: &[u8]) -> Result<(), ExporterHarnessError> {
        let CaptureStorage::CapturedStream(writer) = &mut self.storage else {
            return Err(ExporterHarnessError::product(
                "stream output used for a different observable class",
            ));
        };
        writer
            .write_all(bytes)
            .map_err(io_error("write captured stream"))
    }

    /// Record a decoder-accepted receiver request before returning its ack.
    pub fn accept_receiver(
        &mut self,
        operation: &str,
        target: &str,
        mut metadata: Vec<[String; 2]>,
        body: &[u8],
    ) -> Result<ReceiverAcknowledgement, ExporterHarnessError> {
        let CaptureStorage::ReceiverTranscript {
            protocol,
            entries,
            bodies,
        } = &mut self.storage
        else {
            return Err(ExporterHarnessError::product(
                "receiver output used for a different observable class",
            ));
        };
        validate_receiver_metadata(&metadata).map_err(|error| {
            ExporterHarnessError::product(format!("invalid receiver metadata: {error}"))
        })?;
        metadata.retain(|pair| !protocol.removed_metadata_keys().contains(&pair[0]));
        let sequence = u64::try_from(entries.len()).map_err(|_| {
            ExporterHarnessError::acquisition("receiver acceptance sequence overflow")
        })?;
        let body = body.to_vec();
        let entry = ReceiverTranscriptEntry {
            sequence,
            operation: operation.to_owned(),
            target: target.to_owned(),
            metadata,
            body: ReceiverBody {
                encoding: ReceiverBodyEncoding::Bytes,
                length: body.len() as u64,
                blake3: format!("blake3:{}", blake3::hash(&body)),
            },
        };

        // The body and transcript entry commit precedes construction of the ack.
        bodies.push(body);
        entries.push(entry);
        Ok(ReceiverAcknowledgement {
            sequence,
            recorded_acceptances: entries.len(),
        })
    }

    fn flush(&mut self) -> Result<(), ExporterHarnessError> {
        match &mut self.storage {
            CaptureStorage::ArtifactTree(capture) => {
                for file in capture.files.values_mut() {
                    file.flush().map_err(io_error("flush artifact file"))?;
                    file.sync_all().map_err(io_error("sync artifact file"))?;
                }
            }
            CaptureStorage::CapturedStream(writer) => {
                writer.flush().map_err(io_error("flush captured stream"))?;
                writer
                    .get_mut()
                    .sync_all()
                    .map_err(io_error("sync captured stream"))?;
            }
            CaptureStorage::ReceiverTranscript { .. } => {}
        }
        Ok(())
    }

    fn finalize(self) -> Result<CapturedObservable, ExporterHarnessError> {
        match self.storage {
            CaptureStorage::ArtifactTree(capture) => finalize_artifact_capture(capture),
            CaptureStorage::CapturedStream(mut writer) => {
                writer
                    .get_mut()
                    .seek(SeekFrom::Start(0))
                    .map_err(io_error("rewind captured stream"))?;
                let mut bytes = Vec::new();
                writer
                    .get_mut()
                    .read_to_end(&mut bytes)
                    .map_err(io_error("read captured stream"))?;
                Ok(CapturedObservable {
                    raw_observable_bytes: bytes,
                    artifact_contents: BTreeMap::new(),
                    transcript_bodies: Vec::new(),
                })
            }
            CaptureStorage::ReceiverTranscript {
                protocol: _,
                entries,
                bodies,
            } => {
                let mut raw_observable_bytes =
                    serde_json_canonicalizer::to_vec(&entries).map_err(|error| {
                        ExporterHarnessError::acquisition(format!(
                            "cannot canonicalize receiver transcript: {error}"
                        ))
                    })?;
                raw_observable_bytes.push(b'\n');
                Ok(CapturedObservable {
                    raw_observable_bytes,
                    artifact_contents: BTreeMap::new(),
                    transcript_bodies: bodies,
                })
            }
        }
    }
}

struct CapturedObservable {
    raw_observable_bytes: Vec<u8>,
    artifact_contents: BTreeMap<String, Vec<u8>>,
    transcript_bodies: Vec<Vec<u8>>,
}

/// Complete harness-produced member input for controller retention.
#[derive(Debug)]
pub struct CompletedExporterMember {
    binding: ExporterMemberBinding,
    evidence: ExporterMemberEvidence,
    backing_payloads: Vec<SelectedBackingPayloadV1>,
    record: ExporterMemberRecord,
    record_bytes: Vec<u8>,
    summary: ExporterMemberSummary,
    receiver_protocol: Option<AuthenticatedReceiverProtocolV1>,
}

impl CompletedExporterMember {
    /// Controller-authenticated receiver protocol retained with this member.
    pub fn receiver_protocol(&self) -> Option<&str> {
        self.receiver_protocol
            .as_ref()
            .map(AuthenticatedReceiverProtocolV1::protocol)
    }

    /// Digest of the authenticated receiver-protocol authority retained with this member.
    pub fn receiver_protocol_authority_blake3(&self) -> Option<&str> {
        self.receiver_protocol
            .as_ref()
            .map(AuthenticatedReceiverProtocolV1::authority_blake3)
    }

    /// Internally constructed immutable member binding.
    pub fn binding(&self) -> &ExporterMemberBinding {
        &self.binding
    }

    /// Canonical receipts and the one complete retained repetition.
    pub fn evidence(&self) -> &ExporterMemberEvidence {
        &self.evidence
    }

    /// Exact policy-selected backing bytes for replay validation.
    pub fn backing_payloads(&self) -> &[SelectedBackingPayloadV1] {
        &self.backing_payloads
    }

    /// Internally constructed canonical post-run member record.
    pub fn record(&self) -> &ExporterMemberRecord {
        &self.record
    }

    /// RFC 8785 JCS record bytes with one trailing newline.
    pub fn record_bytes(&self) -> &[u8] {
        &self.record_bytes
    }

    /// Validated member summary used by statistical assembly.
    pub fn summary(&self) -> &ExporterMemberSummary {
        &self.summary
    }
}

/// Concrete owner of the generation-1 exporter corpus and capture lifecycle.
pub struct ExporterHarnessRunner {
    policy: ExporterObservablePolicyV1,
    corpus: Vec<Vec<u8>>,
    corpus_blake3: String,
}

impl ExporterHarnessRunner {
    /// Generate the fixed 100,000-record corpus and bind one validated policy.
    pub fn new(policy: ExporterObservablePolicyV1) -> Result<Self, ExporterHarnessError> {
        let corpus = build_fixed_corpus()?;
        let mut hasher = blake3::Hasher::new();
        for record in &corpus {
            hasher.update(record);
        }
        Ok(Self {
            policy,
            corpus,
            corpus_blake3: format!("blake3:{}", hasher.finalize()),
        })
    }

    /// Digest of the internally generated fixed corpus.
    pub fn corpus_blake3(&self) -> &str {
        &self.corpus_blake3
    }

    /// Run exactly sixteen sequential active write-and-flush passes.
    pub fn run_member<E: ExporterWorkload + ?Sized>(
        &self,
        source: ExporterMemberSource<'_>,
        exporter: &mut E,
    ) -> Result<CompletedExporterMember, ExporterHarnessError> {
        let binding = self.bind_member(source)?;
        let receiver_protocol = match binding.observable_kind {
            ExporterObservableKind::ReceiverTranscript => {
                let protocol = source.receiver_protocol.ok_or_else(|| {
                    ExporterHarnessError::product(
                        "receiver scenario requires an authenticated receiver protocol",
                    )
                })?;
                self.policy
                    .validate_receiver_protocol(protocol)
                    .map_err(policy_error)?;
                Some(protocol.clone())
            }
            _ => {
                if source.receiver_protocol.is_some() {
                    return Err(ExporterHarnessError::product(
                        "non-receiver scenario cannot bind a receiver protocol",
                    ));
                }
                None
            }
        };
        let mut receipts = Vec::with_capacity(REPETITIONS);
        let mut retained = None;
        let mut retained_backing = None;

        for repetition_ordinal in 0..REPETITIONS {
            let mut capture =
                HostExporterCapture::new(binding.observable_kind, receiver_protocol.as_ref())?;
            let mut records = ExporterRecordStream {
                records: &self.corpus,
                next: 0,
            };

            let started = Instant::now();
            exporter.export(repetition_ordinal as u64, &mut records, &mut capture)?;
            capture.flush()?;
            let active_duration_ns = u64::try_from(started.elapsed().as_nanos()).map_err(|_| {
                ExporterHarnessError::acquisition("active exporter duration does not fit u64")
            })?;

            if records.processed_records() != CORPUS_RECORDS {
                return Err(ExporterHarnessError::product(
                    "exporter repetition must process exactly 100000 harness records",
                ));
            }
            if active_duration_ns == 0 {
                return Err(ExporterHarnessError::acquisition(
                    "active exporter write-and-flush duration must be positive",
                ));
            }

            // Observable traversal, policy application, hashing, validation, and
            // evidence retention occur only after the active timer has stopped.
            let captured = capture.finalize()?;
            let backing_payloads = self
                .policy
                .select_host_backing_payloads(
                    &binding.scenario_id,
                    &captured.artifact_contents,
                    (binding.observable_kind == ExporterObservableKind::CapturedStream)
                        .then_some(captured.raw_observable_bytes.as_slice()),
                    &captured.transcript_bodies,
                )
                .map_err(policy_error)?;
            let provenance_binding = ProvenanceBindingV1 {
                experiment_identity_blake3: binding.experiment_identity_blake3.clone(),
                attempt_ordinal: binding.attempt_ordinal,
                scenario_id: binding.scenario_id.clone(),
                pair_id: binding.pair_id.clone(),
                member: binding.member,
                repetition_ordinal: repetition_ordinal as u64,
            };
            let applied = apply_exporter_observable_policy_v1(
                &self.policy,
                &provenance_binding,
                &captured.raw_observable_bytes,
                &backing_payloads,
            )
            .map_err(policy_error)?;

            receipts.push(ExporterRepetitionReceipt {
                schema_version: 1,
                experiment_identity_blake3: binding.experiment_identity_blake3.clone(),
                attempt_ordinal: binding.attempt_ordinal,
                scenario_id: binding.scenario_id.clone(),
                pair_id: binding.pair_id.clone(),
                member: binding.member,
                repetition_ordinal: repetition_ordinal as u64,
                corpus_blake3: binding.corpus_blake3.clone(),
                processed_records: CORPUS_RECORDS,
                observable_kind: binding.observable_kind,
                raw_observable_blake3: applied.raw_observable_blake3,
                comparison_observable_blake3: applied.comparison_observable_blake3,
                provenance_receipt_blake3: applied.provenance_receipt_blake3,
                active_duration_ns,
                build_artifact_blake3: binding.build_artifact_blake3.clone(),
                build_receipt_blake3: binding.build_receipt_blake3.clone(),
            });
            if repetition_ordinal == RETAINED_REPETITION {
                retained = Some(RetainedExporterEvidence {
                    repetition_ordinal,
                    raw_observable_bytes: captured.raw_observable_bytes,
                    comparison_observable_bytes: applied.comparison_bytes,
                    provenance_receipt_bytes: applied.provenance_receipt_bytes,
                });
                retained_backing = Some(backing_payloads);
            }
        }

        let mut repetition_receipt_bytes =
            serde_json_canonicalizer::to_vec(&receipts).map_err(|error| {
                ExporterHarnessError::acquisition(format!(
                    "cannot canonicalize exporter repetition receipts: {error}"
                ))
            })?;
        repetition_receipt_bytes.push(b'\n');
        let evidence = ExporterMemberEvidence {
            repetition_receipt_bytes,
            retained: retained.ok_or_else(|| {
                ExporterHarnessError::acquisition("retained exporter repetition is absent")
            })?,
        };
        let contract = ExporterSampleContract::normative();
        let summary = validate_exporter_member_evidence(&contract, &binding, &evidence)
            .map_err(stats_error)?;
        let retained_receipt = &summary.repetitions[evidence.retained.repetition_ordinal];
        let record = ExporterMemberRecord {
            schema_version: 1,
            experiment_identity_blake3: binding.experiment_identity_blake3.clone(),
            attempt_ordinal: binding.attempt_ordinal,
            scenario_id: binding.scenario_id.clone(),
            pair_id: binding.pair_id.clone(),
            member: binding.member,
            active_duration_ns: summary.active_duration_nanoseconds,
            processed_records: summary.processed_records,
            retained_artifact_records: summary.retained_artifact_records,
            comparison_observable_blake3: summary.comparison_observable_blake3.clone(),
            repetition_receipts_blake3: summary.repetition_receipts_blake3.clone(),
            retained_repetition_ordinal: evidence.retained.repetition_ordinal as u64,
            retained_raw_observable_blake3: retained_receipt.raw_observable_blake3.clone(),
            retained_comparison_observable_blake3: retained_receipt
                .comparison_observable_blake3
                .clone(),
            retained_provenance_receipt_blake3: retained_receipt.provenance_receipt_blake3.clone(),
            observable_policy_blake3: binding.observable_policy_blake3.clone(),
            build_artifact_blake3: binding.build_artifact_blake3.clone(),
            build_receipt_blake3: binding.build_receipt_blake3.clone(),
        };
        let mut record_bytes = serde_json_canonicalizer::to_vec(&record).map_err(|error| {
            ExporterHarnessError::acquisition(format!(
                "cannot canonicalize exporter member record: {error}"
            ))
        })?;
        record_bytes.push(b'\n');
        let record = validate_exporter_member_record(&contract, &binding, &evidence, &record_bytes)
            .map_err(stats_error)?;

        Ok(CompletedExporterMember {
            binding,
            evidence,
            backing_payloads: retained_backing.ok_or_else(|| {
                ExporterHarnessError::acquisition("retained exporter backing is absent")
            })?,
            record,
            record_bytes,
            summary,
            receiver_protocol,
        })
    }

    fn bind_member(
        &self,
        source: ExporterMemberSource<'_>,
    ) -> Result<ExporterMemberBinding, ExporterHarnessError> {
        if source.experiment_identity_bytes.is_empty()
            || source.build_receipt_bytes.is_empty()
            || source.scenario_id.is_empty()
            || source.pair_id.is_empty()
        {
            return Err(ExporterHarnessError::product(
                "exporter member source material and coordinates must be nonempty",
            ));
        }
        let observable_kind = self
            .policy
            .observable_kind(source.scenario_id)
            .ok_or_else(|| {
                ExporterHarnessError::product("exporter scenario is absent from policy")
            })?;
        if self.policy.evidence_mode() == ExporterEvidenceMode::StaticCalibration
            && (source.member != ExporterMember::Static
                || source.attempt_ordinal != 0
                || source.pair_id != "task1-static-calibration")
        {
            return Err(ExporterHarnessError::product(
                "static exporter calibration binding is invalid",
            ));
        }
        let build_artifact_blake3 = digest_file(source.build_artifact)?;
        Ok(ExporterMemberBinding {
            mode: self.policy.evidence_mode(),
            experiment_identity_blake3: format!(
                "blake3:{}",
                blake3::hash(source.experiment_identity_bytes)
            ),
            attempt_ordinal: source.attempt_ordinal,
            scenario_id: source.scenario_id.to_owned(),
            pair_id: source.pair_id.to_owned(),
            member: source.member,
            corpus_blake3: self.corpus_blake3.clone(),
            observable_kind,
            observable_policy_blake3: self.policy.canonical_blake3().map_err(policy_error)?,
            build_artifact_blake3,
            build_receipt_blake3: format!("blake3:{}", blake3::hash(source.build_receipt_bytes)),
        })
    }
}

#[derive(Serialize)]
struct FixedCorpusRecord<'a> {
    input_tokens: u64,
    ordinal: u64,
    output_tokens: u64,
    request_id: &'a str,
    success: bool,
}

fn build_fixed_corpus() -> Result<Vec<Vec<u8>>, ExporterHarnessError> {
    let mut corpus = Vec::with_capacity(CORPUS_RECORDS as usize);
    for ordinal in 0..CORPUS_RECORDS {
        let request_id = format!("exporter-record-{ordinal:06}");
        let record = FixedCorpusRecord {
            input_tokens: 128 + ordinal % 128,
            ordinal,
            output_tokens: 32 + ordinal % 64,
            request_id: &request_id,
            success: true,
        };
        let mut bytes = serde_json_canonicalizer::to_vec(&record).map_err(|error| {
            ExporterHarnessError::acquisition(format!(
                "cannot canonicalize fixed exporter corpus: {error}"
            ))
        })?;
        bytes.push(b'\n');
        corpus.push(bytes);
    }
    Ok(corpus)
}

fn finalize_artifact_capture(
    capture: ArtifactCapture,
) -> Result<CapturedObservable, ExporterHarnessError> {
    drop(capture.files);
    let mut entries = Vec::new();
    let mut contents = BTreeMap::new();
    collect_artifact_entries(
        capture.root.path(),
        capture.root.path(),
        &mut entries,
        &mut contents,
    )?;
    entries.sort_by(|left, right| left.path.cmp(&right.path));
    let mut raw_observable_bytes = serde_json_canonicalizer::to_vec(&entries).map_err(|error| {
        ExporterHarnessError::acquisition(format!(
            "cannot canonicalize artifact-tree manifest: {error}"
        ))
    })?;
    raw_observable_bytes.push(b'\n');
    Ok(CapturedObservable {
        raw_observable_bytes,
        artifact_contents: contents,
        transcript_bodies: Vec::new(),
    })
}

fn collect_artifact_entries(
    root: &Path,
    directory: &Path,
    entries: &mut Vec<ArtifactTreeEntry>,
    contents: &mut BTreeMap<String, Vec<u8>>,
) -> Result<(), ExporterHarnessError> {
    let mut children = std::fs::read_dir(directory)
        .map_err(io_error("read artifact directory"))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(io_error("read artifact entry"))?;
    children.sort_by_key(std::fs::DirEntry::file_name);
    if children.is_empty() && directory != root {
        let path = relative_artifact_path(root, directory)?;
        entries.push(ArtifactTreeEntry {
            blake3: format!("blake3:{}", blake3::hash(b"")),
            kind: ArtifactTreeKind::EmptyDirectory,
            length: 0,
            path,
        });
        return Ok(());
    }
    for child in children {
        let file_type = child
            .file_type()
            .map_err(io_error("inspect artifact entry"))?;
        let path = child.path();
        if file_type.is_dir() {
            collect_artifact_entries(root, &path, entries, contents)?;
        } else if file_type.is_file() {
            let relative = relative_artifact_path(root, &path)?;
            let bytes = std::fs::read(&path).map_err(io_error("read artifact file"))?;
            entries.push(ArtifactTreeEntry {
                blake3: format!("blake3:{}", blake3::hash(&bytes)),
                kind: ArtifactTreeKind::RegularFile,
                length: bytes.len() as u64,
                path: relative.clone(),
            });
            contents.insert(relative, bytes);
        } else {
            return Err(ExporterHarnessError::product(
                "artifact tree contains a non-file, non-directory entry",
            ));
        }
    }
    Ok(())
}

fn relative_artifact_path(root: &Path, path: &Path) -> Result<String, ExporterHarnessError> {
    let relative = path.strip_prefix(root).map_err(|_| {
        ExporterHarnessError::acquisition("artifact path escaped the host capture root")
    })?;
    let text = relative
        .to_str()
        .ok_or_else(|| ExporterHarnessError::product("artifact path is not valid Unicode"))?;
    validate_artifact_tree_path(text).map_err(|error| {
        ExporterHarnessError::product(format!("artifact path is not normalized: {error}"))
    })?;
    Ok(text.to_owned())
}

fn digest_file(file: &File) -> Result<String, ExporterHarnessError> {
    let mut hasher = blake3::Hasher::new();
    let mut buffer = [0_u8; 64 * 1024];
    let mut offset = 0_u64;
    loop {
        let read = read_file_at(file, &mut buffer, offset)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
        offset = offset
            .checked_add(read as u64)
            .ok_or_else(|| ExporterHarnessError::acquisition("build artifact offset overflow"))?;
    }
    Ok(format!("blake3:{}", hasher.finalize()))
}

#[cfg(unix)]
fn read_file_at(
    file: &File,
    buffer: &mut [u8],
    offset: u64,
) -> Result<usize, ExporterHarnessError> {
    use std::os::unix::fs::FileExt;

    file.read_at(buffer, offset)
        .map_err(io_error("read acquired build artifact"))
}

#[cfg(windows)]
fn read_file_at(
    file: &File,
    buffer: &mut [u8],
    offset: u64,
) -> Result<usize, ExporterHarnessError> {
    use std::os::windows::fs::FileExt;

    file.seek_read(buffer, offset)
        .map_err(io_error("read acquired build artifact"))
}

fn io_error(context: &'static str) -> impl FnOnce(std::io::Error) -> ExporterHarnessError {
    move |error| ExporterHarnessError::acquisition(format!("{context}: {error}"))
}

fn policy_error(error: impl fmt::Display) -> ExporterHarnessError {
    ExporterHarnessError::product(format!("exporter observable policy failed: {error}"))
}

fn stats_error(error: impl fmt::Display) -> ExporterHarnessError {
    ExporterHarnessError::product(format!("exporter evidence validation failed: {error}"))
}
