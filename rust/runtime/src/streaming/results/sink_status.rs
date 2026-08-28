// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Durable derived-sink finalization status and its compare-and-swap machine.
//!
//! One derived result sink (an exporter bound to one committed final checkpoint
//! generation) advances through exactly three durable states:
//!
//! ```text
//! PendingAttempt { next_ordinal: 0 }
//!   --(ordinary read/write/sync failure)--> PendingRetry { last_ordinal, counter_before }
//!   --(durable authoritative output)------> Complete { output_digest, output_length }
//! ```
//!
//! An ordinary failure never fabricates an aborted generation and never rolls
//! back execution: it retains the committed generation and the report lease and
//! re-enters the bounded retry supervisor at the next dense ordinal. Only a
//! durable authoritative output advances to `Complete`.
//!
//! This module is the durable status owner that `reliability` defers to. It is
//! the sole caller of [`DerivedExportReceiptReference::from_status_fields`],
//! [`VerifiedDerivedSinkAttemptStatus::from_status_owner`], and
//! [`DurableExportReceiptValidationContext::from_final_generation_status`], so a
//! restarted process reopens its exact pending attempt from the committed
//! generation plus the derived status store alone — no live issue ledger.
//!
//! Every transition is `async`: a compare-and-swap awaits the durable medium,
//! and a restart drops and reopens that medium. A synchronous model-only
//! transition would not observe the drop-and-reopen the state machine exists to
//! survive.

use std::{
    cell::RefCell, collections::BTreeMap, fmt::Debug, mem::size_of, num::NonZeroUsize, rc::Rc,
};

use async_trait::async_trait;
use bytes::Bytes;

use super::ResultPlaneError;
use crate::streaming::{
    budget::{BudgetLease, StreamingResourceBudget},
    checkpoint::{
        BudgetedCheckpointBytes, CheckpointGeneration, CommittedCheckpointGeneration,
        StreamRunIdentity,
    },
    failure::{OrdinaryStreamingFailure, ResultExportError, ResultExportFailureCode},
    identity::ContentDigest,
    reliability::{
        BudgetOwnedExportIssueReceipt, BudgetOwnedStreamingIssueReporter,
        DerivedExportReceiptReference, DurableExportReceiptValidationContext,
        OrdinaryStreamingIssue, PreparedExportReceiptPersistence, PreparedStreamingIssuePolicy,
        ResultSinkAttemptOutcome, StreamingIssueClass, StreamingIssueComponentId,
        StreamingIssueReporter, VerifiedDerivedSinkAttemptStatus,
        restore_durable_export_issue_receipt,
    },
};

/// Durable finalization state of one derived result sink.
///
/// `counter_before` is dense by construction: the forward path defines it as
/// `u64::from(last_ordinal)`, so any other pairing names no reachable status.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DerivedSinkStatus {
    /// No attempt has closed yet; the next attempt uses `next_ordinal`.
    PendingAttempt {
        /// Dense ordinal the next attempt must use.
        next_ordinal: u32,
    },
    /// An ordinary failure closed `last_ordinal` and retained its receipt.
    PendingRetry {
        /// Dense ordinal of the closed attempt.
        last_ordinal: u32,
        /// Predecessor matching counter, always `u64::from(last_ordinal)`.
        counter_before: u64,
    },
    /// The authoritative output is durable and the sink is terminal.
    Complete {
        /// Raw BLAKE3 digest of the durable authoritative output.
        output_digest: ContentDigest,
        /// Exact durable authoritative output length.
        output_length: u64,
    },
}

impl DerivedSinkStatus {
    /// Whether this state still owes the retry supervisor work.
    #[must_use]
    pub const fn is_pending(self) -> bool {
        !matches!(self, Self::Complete { .. })
    }
}

/// Stable machine-readable derived-sink finalization refusal.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SinkFinalizationFailureCode {
    /// The named successor is not reachable from the retained state.
    IllegalTransition,
    /// The authority names a different logical run.
    ForeignRun,
    /// The authority names a different checkpoint generation.
    ForeignGeneration,
    /// The authority names a different derived sink.
    ForeignSink,
    /// The submitted attempt ordinal or counter is not the dense successor.
    OrdinalMismatch,
    /// The dense retry ordinal cannot advance without wrapping.
    OrdinalOverflow,
    /// The status authority requires a committed final generation.
    NonFinalGeneration,
    /// No durable status exists for this generation and sink.
    MissingStatus,
    /// The durable status names a receipt the store cannot reach.
    MissingReceipt,
    /// The durable receipt, its embedded receipt, or its reference did not
    /// re-derive from the frozen policy and status authority.
    TamperedReceipt,
    /// No durable authoritative output backs the requested completion.
    MissingOutput,
    /// Exact capacity for the attempt, receipt, or parse was unavailable.
    Budget,
    /// The reliability plane could not prepare the closed attempt's receipt.
    ReceiptPreparation,
}

impl SinkFinalizationFailureCode {
    /// Return the stable lowercase code.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::IllegalTransition => "illegal_transition",
            Self::ForeignRun => "foreign_run",
            Self::ForeignGeneration => "foreign_generation",
            Self::ForeignSink => "foreign_sink",
            Self::OrdinalMismatch => "ordinal_mismatch",
            Self::OrdinalOverflow => "ordinal_overflow",
            Self::NonFinalGeneration => "non_final_generation",
            Self::MissingStatus => "missing_status",
            Self::MissingReceipt => "missing_receipt",
            Self::TamperedReceipt => "tampered_receipt",
            Self::MissingOutput => "missing_output",
            Self::Budget => "budget",
            Self::ReceiptPreparation => "receipt_preparation",
        }
    }
}

/// Stable machine-readable reason one derived result-sink attempt failed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SinkFailureReason {
    /// The authoritative report could not be built, written, or renamed.
    ReportPersistence,
    /// A configured optional export sink failed to produce its output.
    Export,
    /// The post-persistence report lifecycle commit failed.
    ReportCommit,
}

impl SinkFailureReason {
    /// Return the stable lowercase reason.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ReportPersistence => "report_persistence",
            Self::Export => "export",
            Self::ReportCommit => "report_commit",
        }
    }

    const fn export_failure_code(self) -> ResultExportFailureCode {
        match self {
            Self::ReportPersistence => ResultExportFailureCode::Io,
            Self::Export => ResultExportFailureCode::Attempt,
            Self::ReportCommit => ResultExportFailureCode::Unavailable,
        }
    }
}

/// Reported finalization state of one derived result sink.
///
/// [`DerivedSinkStatus`] is the durable compare-and-swap state; this is what the
/// report-persistence path reports back to its caller. It carries the failure
/// reason, which the durable state deliberately does not, and it distinguishes
/// exhaustion — an authored-policy disposition rather than a durable state, so
/// an exhausted sink is still retained as `PendingRetry` on the medium.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResultSinkState {
    /// The authoritative output is durable and the sink is terminal.
    Complete {
        /// Raw BLAKE3 digest of the durable authoritative output.
        output_digest: ContentDigest,
        /// Exact durable authoritative output length.
        output_length: u64,
    },
    /// An ordinary failure closed one attempt and the sink owes a retry.
    PendingRetry {
        /// Dense ordinal of the attempt that closed.
        attempt: u32,
        /// Reason the closed attempt failed.
        last_failure: SinkFailureReason,
    },
    /// The authored retry threshold is exhausted and the export is incomplete.
    ///
    /// Only the optional-export attempt lease is released; the committed
    /// generation and its checkpoint authority are retained unchanged.
    Exhausted {
        /// Total number of closed attempts.
        attempts: u32,
        /// Reason the last attempt failed.
        last_failure: SinkFailureReason,
    },
}

impl ResultSinkState {
    /// Derive the reported state of one closed ordinary failure.
    #[must_use]
    pub const fn from_closed_failure(
        attempt: u32,
        last_failure: SinkFailureReason,
        is_exhausted: bool,
    ) -> Self {
        if is_exhausted {
            Self::Exhausted {
                // The dense ordinal is zero-based, so the closed attempt count
                // is one greater. Saturation names the same exhausted state.
                attempts: attempt.saturating_add(1),
                last_failure,
            }
        } else {
            Self::PendingRetry {
                attempt,
                last_failure,
            }
        }
    }

    /// Whether the sink still owes the bounded retry supervisor work.
    #[must_use]
    pub const fn is_pending(self) -> bool {
        matches!(self, Self::PendingRetry { .. })
    }
}

/// Records one ordinary report-persistence failure as a durable pending retry.
///
/// The authority never rolls execution back: it closes the current attempt
/// against the committed final generation and leaves that generation, its
/// leased reader, and the diagnostic root exactly as it found them.
#[async_trait(?Send)]
pub trait ReportRetryAuthority: Debug {
    /// Atomically close the open attempt and return the reported sink state.
    async fn record_failure(
        &mut self,
        reason: SinkFailureReason,
    ) -> Result<ResultSinkState, ResultPlaneError>;
}

/// Domain separator for report-sink issue identity.
const REPORT_SINK_ISSUE_DOMAIN: &[u8] = b"aiperf.streaming.report-sink.issue.v1";

/// Durable [`ReportRetryAuthority`] over one committed final generation.
pub struct DurableReportRetryAuthority {
    store: DerivedSinkStatusStore,
    final_generation: CommittedCheckpointGeneration,
    sink_id: StreamingIssueComponentId,
    reporter: BudgetOwnedStreamingIssueReporter,
    export_budget: StreamingResourceBudget,
}

impl std::fmt::Debug for DurableReportRetryAuthority {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DurableReportRetryAuthority")
            .field("generation", &self.final_generation.generation())
            .field("sink_id", &self.sink_id)
            .finish_non_exhaustive()
    }
}

impl DurableReportRetryAuthority {
    /// Bind a retry authority to one status owner, generation, and sink.
    #[must_use]
    pub const fn new(
        store: DerivedSinkStatusStore,
        final_generation: CommittedCheckpointGeneration,
        sink_id: StreamingIssueComponentId,
        reporter: BudgetOwnedStreamingIssueReporter,
        export_budget: StreamingResourceBudget,
    ) -> Self {
        Self {
            store,
            final_generation,
            sink_id,
            reporter,
            export_budget,
        }
    }

    /// Borrow the durable status owner this authority commits through.
    #[must_use]
    pub const fn store(&self) -> &DerivedSinkStatusStore {
        &self.store
    }

    fn semantic_context_digest(&self, reason: SinkFailureReason) -> ContentDigest {
        let mut hasher = blake3::Hasher::new();
        hasher.update(REPORT_SINK_ISSUE_DOMAIN);
        hasher.update(self.sink_id.as_str().as_bytes());
        hasher.update(reason.as_str().as_bytes());
        ContentDigest::from_bytes(*hasher.finalize().as_bytes())
    }
}

#[async_trait(?Send)]
impl ReportRetryAuthority for DurableReportRetryAuthority {
    async fn record_failure(
        &mut self,
        reason: SinkFailureReason,
    ) -> Result<ResultSinkState, ResultPlaneError> {
        self.store
            .reconcile_initial(&self.final_generation, &self.sink_id)
            .await?;
        // The token proves the attempt was admitted against the attempt budget;
        // it is released once the closing compare-and-swap is durable.
        let token = self
            .store
            .open_attempt(&self.final_generation, &self.sink_id)
            .await?;
        let attempt_ordinal = token.ordinal();
        let run = *self.final_generation.run();
        let generation = self.final_generation.generation();
        let issue = OrdinaryStreamingIssue::export(
            run,
            self.sink_id.clone(),
            generation.clone(),
            StreamingIssueClass::Retryable,
            self.semantic_context_digest(reason),
            attempt_ordinal,
            generation.digest,
            OrdinaryStreamingFailure::Export(ResultExportError::failure(
                reason.export_failure_code(),
            )),
        )
        .map_err(|_| ResultPlaneError::SinkFinalization {
            code: SinkFinalizationFailureCode::ReceiptPreparation,
        })?;
        let prepared = self
            .reporter
            .prepare_export_attempt_failure(
                &run,
                &generation,
                &self.sink_id,
                attempt_ordinal,
                ResultSinkAttemptOutcome::Failed(issue),
                &self.export_budget,
            )
            .await
            .map_err(|_| ResultPlaneError::SinkFinalization {
                code: SinkFinalizationFailureCode::ReceiptPreparation,
            })?;
        let is_exhausted = prepared.is_exhausted();
        let persistence = prepared.into_persistence();
        self.store
            .commit_retry(&self.final_generation, &self.sink_id, &persistence)
            .await?;
        drop(token);
        Ok(ResultSinkState::from_closed_failure(
            attempt_ordinal,
            reason,
            is_exhausted,
        ))
    }
}

fn refuse<T>(code: SinkFinalizationFailureCode) -> Result<T, ResultPlaneError> {
    Err(ResultPlaneError::SinkFinalization { code })
}

/// Move-only budgeted authority to run exactly one derived-sink attempt.
///
/// The token is minted only by [`DerivedSinkStatusStore::open_attempt`], which
/// charges the caller's attempt budget first, so an unbudgeted attempt cannot
/// exist. Its fields are private and it has no public constructor, `Clone`, or
/// `Deserialize`, so a forged token is unnameable:
///
/// ```compile_fail
/// # use aiperf_runtime::streaming::results::sink_status::SinkAttemptToken;
/// # use aiperf_runtime::streaming::identity::ContentDigest;
/// let _forged = SinkAttemptToken {
///     ordinal: 0,
/// };
/// ```
#[derive(Debug)]
pub struct SinkAttemptToken {
    run: StreamRunIdentity,
    generation: CheckpointGeneration,
    sink_id: StreamingIssueComponentId,
    ordinal: u32,
    /// Retained solely to prove the attempt was admitted against a budget.
    lease: BudgetLease,
}

impl SinkAttemptToken {
    /// Return the dense ordinal this attempt must report.
    #[must_use]
    pub const fn ordinal(&self) -> u32 {
        self.ordinal
    }

    /// Borrow the derived sink this attempt belongs to.
    #[must_use]
    pub const fn sink_id(&self) -> &StreamingIssueComponentId {
        &self.sink_id
    }

    /// Borrow the committed generation this attempt is bound to.
    #[must_use]
    pub const fn generation(&self) -> &CheckpointGeneration {
        &self.generation
    }

    /// Return the exact admitted attempt charge.
    #[must_use]
    pub fn charged_bytes(&self) -> usize {
        self.lease.charged_bytes()
    }
}

/// Durable proof that one derived sink's authoritative output is persisted.
///
/// The only two minting paths are [`DurableSinkOutputWriter::write`] and
/// [`DurableSinkOutputProbe::probe`]; both read or write the durable medium.
/// The type has private fields, no public constructor, no `Clone`, and no
/// `Deserialize`, so no third path can produce one:
///
/// ```compile_fail
/// # use aiperf_runtime::streaming::results::sink_status::DurableOutputProof;
/// # fn cannot_fabricate(proof: DurableOutputProof) {
/// let _digest = proof.output_digest;
/// # }
/// ```
#[derive(Debug)]
pub struct DurableOutputProof {
    run: StreamRunIdentity,
    generation: CheckpointGeneration,
    sink_id: StreamingIssueComponentId,
    output_digest: ContentDigest,
    output_length: u64,
}

impl DurableOutputProof {
    /// Borrow the logical run whose medium retains the output.
    #[must_use]
    pub const fn run(&self) -> &StreamRunIdentity {
        &self.run
    }

    /// Borrow the committed generation the output was written under.
    #[must_use]
    pub const fn generation(&self) -> &CheckpointGeneration {
        &self.generation
    }

    /// Borrow the derived sink that produced the output.
    #[must_use]
    pub const fn sink_id(&self) -> &StreamingIssueComponentId {
        &self.sink_id
    }

    /// Return the raw BLAKE3 digest of the durable output.
    #[must_use]
    pub const fn output_digest(&self) -> ContentDigest {
        self.output_digest
    }

    /// Return the exact durable output length.
    #[must_use]
    pub const fn output_length(&self) -> u64 {
        self.output_length
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct StoredReceiptReference {
    receipt_digest: ContentDigest,
    receipt_length: u64,
    embedded_receipt_digest: ContentDigest,
    embedded_receipt_length: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct StoredOutput {
    digest: ContentDigest,
    length: u64,
}

#[derive(Clone, Debug)]
struct StatusRecord {
    status: DerivedSinkStatus,
    encoded_receipt: Option<Vec<u8>>,
    reference: Option<StoredReceiptReference>,
    output: Option<StoredOutput>,
}

type SubstrateKey = (u64, [u8; 32], String);

#[derive(Debug, Default)]
struct SubstrateInner {
    records: BTreeMap<SubstrateKey, StatusRecord>,
    /// Monotonic count of durable mutations, so a refusal that must precede
    /// store I/O is observable rather than merely asserted.
    write_count: u64,
}

/// Durable medium behind derived-sink status, independent of any store instance.
///
/// A [`DerivedSinkStatusStore`] is a view over this medium. Dropping every store
/// and reopening a fresh one over the same substrate models process replacement:
/// the status and its retained receipt survive, the in-memory owner does not.
#[derive(Clone, Debug, Default)]
pub struct DerivedStatusSubstrate {
    inner: Rc<RefCell<SubstrateInner>>,
}

fn substrate_key(
    generation: &CheckpointGeneration,
    sink_id: &StreamingIssueComponentId,
) -> SubstrateKey {
    (
        generation.epoch().get(),
        *generation.digest.as_bytes(),
        sink_id.as_str().to_owned(),
    )
}

impl DerivedStatusSubstrate {
    /// Construct an empty durable medium.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Return the monotonic count of durable mutations.
    #[must_use]
    pub fn write_count(&self) -> u64 {
        self.inner.borrow().write_count
    }

    /// Return the number of retained status records.
    #[must_use]
    pub fn record_count(&self) -> usize {
        self.inner.borrow().records.len()
    }

    /// Overwrite or remove the retained encoded receipt bytes.
    ///
    /// Fault-injection seam for tamper and unreachable-object coverage. It does
    /// not touch the retained reference, so a reopen must reject the mismatch
    /// rather than trust the document.
    pub fn overwrite_encoded_receipt(
        &self,
        generation: &CheckpointGeneration,
        sink_id: &StreamingIssueComponentId,
        encoded: Option<Vec<u8>>,
    ) {
        let mut inner = self.inner.borrow_mut();
        if let Some(record) = inner.records.get_mut(&substrate_key(generation, sink_id)) {
            record.encoded_receipt = encoded;
        }
    }

    /// Overwrite the retained embedded-receipt digest of the status reference.
    ///
    /// Fault-injection seam: a status whose reference no longer names the
    /// embedded receipt must refuse reopen.
    pub fn overwrite_embedded_receipt_digest(
        &self,
        generation: &CheckpointGeneration,
        sink_id: &StreamingIssueComponentId,
        digest: ContentDigest,
    ) {
        let mut inner = self.inner.borrow_mut();
        if let Some(record) = inner.records.get_mut(&substrate_key(generation, sink_id))
            && let Some(reference) = record.reference.as_mut()
        {
            reference.embedded_receipt_digest = digest;
        }
    }

    /// Install an exact durable status without transition checking.
    ///
    /// Fault-injection seam for boundary states — an exhausted dense ordinal
    /// space, for instance — that the forward path cannot reach in a test.
    pub fn force_status(
        &self,
        generation: &CheckpointGeneration,
        sink_id: &StreamingIssueComponentId,
        status: DerivedSinkStatus,
    ) {
        let mut inner = self.inner.borrow_mut();
        inner.write_count += 1;
        inner
            .records
            .entry(substrate_key(generation, sink_id))
            .and_modify(|record| record.status = status)
            .or_insert(StatusRecord {
                status,
                encoded_receipt: None,
                reference: None,
                output: None,
            });
    }

    fn record(
        &self,
        generation: &CheckpointGeneration,
        sink_id: &StreamingIssueComponentId,
    ) -> Option<StatusRecord> {
        self.inner
            .borrow()
            .records
            .get(&substrate_key(generation, sink_id))
            .cloned()
    }

    fn put(
        &self,
        generation: &CheckpointGeneration,
        sink_id: &StreamingIssueComponentId,
        record: StatusRecord,
    ) {
        let mut inner = self.inner.borrow_mut();
        inner.write_count += 1;
        inner
            .records
            .insert(substrate_key(generation, sink_id), record);
    }

    fn sinks_for(&self, generation: &CheckpointGeneration) -> Vec<(String, DerivedSinkStatus)> {
        let epoch = generation.epoch().get();
        let digest = *generation.digest.as_bytes();
        self.inner
            .borrow()
            .records
            .iter()
            .filter(|((record_epoch, record_digest, _), _)| {
                *record_epoch == epoch && *record_digest == digest
            })
            .map(|((_, _, sink), record)| (sink.clone(), record.status))
            .collect()
    }
}

/// Durable status owner for every derived sink of one logical run.
///
/// The store is a view: it holds no state of its own beyond the run identity,
/// the shared durable medium, and the attempt budget it admits attempts from.
#[derive(Debug)]
pub struct DerivedSinkStatusStore {
    run: StreamRunIdentity,
    substrate: DerivedStatusSubstrate,
    attempt_budget: StreamingResourceBudget,
}

impl DerivedSinkStatusStore {
    /// Open a status view over one durable medium.
    #[must_use]
    pub fn open(
        run: StreamRunIdentity,
        substrate: DerivedStatusSubstrate,
        attempt_budget: StreamingResourceBudget,
    ) -> Self {
        Self {
            run,
            substrate,
            attempt_budget,
        }
    }

    /// Borrow the durable medium this view reads and writes.
    #[must_use]
    pub const fn substrate(&self) -> &DerivedStatusSubstrate {
        &self.substrate
    }

    fn check_authority(
        &self,
        final_generation: &CommittedCheckpointGeneration,
    ) -> Result<(), ResultPlaneError> {
        if !final_generation.is_final() {
            return refuse(SinkFinalizationFailureCode::NonFinalGeneration);
        }
        if final_generation.run() != &self.run {
            return refuse(SinkFinalizationFailureCode::ForeignRun);
        }
        Ok(())
    }

    /// Find or install the initial durable status for one generation and sink.
    ///
    /// A process that crashed before its first compare-and-swap leaves no record
    /// at all. Reconciliation is keyed by the committed generation and the sink
    /// alone, so the successor process finds the same starting point without a
    /// surviving in-memory owner.
    pub async fn reconcile_initial(
        &self,
        final_generation: &CommittedCheckpointGeneration,
        sink_id: &StreamingIssueComponentId,
    ) -> Result<DerivedSinkStatus, ResultPlaneError> {
        self.check_authority(final_generation)?;
        let generation = final_generation.generation();
        if let Some(record) = self.substrate.record(&generation, sink_id) {
            return Ok(record.status);
        }
        let status = DerivedSinkStatus::PendingAttempt { next_ordinal: 0 };
        self.substrate.put(
            &generation,
            sink_id,
            StatusRecord {
                status,
                encoded_receipt: None,
                reference: None,
                output: None,
            },
        );
        Ok(status)
    }

    /// Load the retained durable status, if one exists.
    pub async fn load(
        &self,
        final_generation: &CommittedCheckpointGeneration,
        sink_id: &StreamingIssueComponentId,
    ) -> Result<Option<DerivedSinkStatus>, ResultPlaneError> {
        self.check_authority(final_generation)?;
        Ok(self
            .substrate
            .record(&final_generation.generation(), sink_id)
            .map(|record| record.status))
    }

    /// Mint the budgeted move-only attempt token at the exact status ordinal.
    ///
    /// Every refusal below precedes any durable mutation, so a refused attempt
    /// leaves the retained status and its receipt byte-identical.
    pub async fn open_attempt(
        &self,
        final_generation: &CommittedCheckpointGeneration,
        sink_id: &StreamingIssueComponentId,
    ) -> Result<SinkAttemptToken, ResultPlaneError> {
        self.check_authority(final_generation)?;
        let generation = final_generation.generation();
        let Some(record) = self.substrate.record(&generation, sink_id) else {
            return refuse(SinkFinalizationFailureCode::MissingStatus);
        };
        let ordinal = match record.status {
            DerivedSinkStatus::PendingAttempt { next_ordinal } => next_ordinal,
            DerivedSinkStatus::PendingRetry { last_ordinal, .. } => last_ordinal
                .checked_add(1)
                .ok_or(ResultPlaneError::SinkFinalization {
                    code: SinkFinalizationFailureCode::OrdinalOverflow,
                })?,
            DerivedSinkStatus::Complete { .. } => {
                return refuse(SinkFinalizationFailureCode::IllegalTransition);
            }
        };
        let lease = self
            .attempt_budget
            .try_acquire(1, size_of::<SinkAttemptToken>())
            .map_err(|_| ResultPlaneError::SinkFinalization {
                code: SinkFinalizationFailureCode::Budget,
            })?;
        Ok(SinkAttemptToken {
            run: self.run,
            generation,
            sink_id: sink_id.clone(),
            ordinal,
            lease,
        })
    }

    /// Compare-and-swap one closed ordinary failure into `PendingRetry`.
    ///
    /// The persistence handoff is borrowed, not consumed: its exact encoded and
    /// parsed leases stay live through the durable write, so the retained bytes
    /// are provably the ones the intact owner still holds.
    pub async fn commit_retry(
        &self,
        final_generation: &CommittedCheckpointGeneration,
        sink_id: &StreamingIssueComponentId,
        persistence: &PreparedExportReceiptPersistence,
    ) -> Result<DerivedSinkStatus, ResultPlaneError> {
        self.check_authority(final_generation)?;
        let generation = final_generation.generation();
        let Some(record) = self.substrate.record(&generation, sink_id) else {
            return refuse(SinkFinalizationFailureCode::MissingStatus);
        };
        let attempt_ordinal = persistence.attempt_ordinal();
        let expected_ordinal = match record.status {
            DerivedSinkStatus::PendingAttempt { next_ordinal } => next_ordinal,
            DerivedSinkStatus::PendingRetry { last_ordinal, .. } => last_ordinal
                .checked_add(1)
                .ok_or(ResultPlaneError::SinkFinalization {
                    code: SinkFinalizationFailureCode::OrdinalOverflow,
                })?,
            DerivedSinkStatus::Complete { .. } => {
                return refuse(SinkFinalizationFailureCode::IllegalTransition);
            }
        };
        if attempt_ordinal != expected_ordinal {
            return refuse(SinkFinalizationFailureCode::OrdinalMismatch);
        }
        let counter_before = u64::from(attempt_ordinal);
        if persistence.counter_before() != counter_before {
            return refuse(SinkFinalizationFailureCode::OrdinalMismatch);
        }
        let reference = persistence.receipt_reference();
        let status = DerivedSinkStatus::PendingRetry {
            last_ordinal: attempt_ordinal,
            counter_before,
        };
        self.substrate.put(
            &generation,
            sink_id,
            StatusRecord {
                status,
                encoded_receipt: Some(persistence.encoded_bytes().to_vec()),
                reference: Some(StoredReceiptReference {
                    receipt_digest: *reference.receipt_digest(),
                    receipt_length: reference.receipt_length(),
                    embedded_receipt_digest: *reference.embedded_receipt_digest(),
                    embedded_receipt_length: reference.embedded_receipt_length(),
                }),
                output: record.output,
            },
        );
        Ok(status)
    }

    /// Compare-and-swap one durable authoritative output into `Complete`.
    pub async fn commit_complete(
        &self,
        final_generation: &CommittedCheckpointGeneration,
        proof: DurableOutputProof,
    ) -> Result<DerivedSinkStatus, ResultPlaneError> {
        self.check_authority(final_generation)?;
        let generation = final_generation.generation();
        if proof.run != self.run {
            return refuse(SinkFinalizationFailureCode::ForeignRun);
        }
        if proof.generation != generation {
            return refuse(SinkFinalizationFailureCode::ForeignGeneration);
        }
        let Some(record) = self.substrate.record(&generation, &proof.sink_id) else {
            return refuse(SinkFinalizationFailureCode::MissingStatus);
        };
        if matches!(record.status, DerivedSinkStatus::Complete { .. }) {
            return refuse(SinkFinalizationFailureCode::IllegalTransition);
        }
        let Some(output) = record.output else {
            return refuse(SinkFinalizationFailureCode::MissingOutput);
        };
        if output.digest != proof.output_digest || output.length != proof.output_length {
            return refuse(SinkFinalizationFailureCode::ForeignSink);
        }
        let status = DerivedSinkStatus::Complete {
            output_digest: proof.output_digest,
            output_length: proof.output_length,
        };
        self.substrate.put(
            &generation,
            &proof.sink_id,
            StatusRecord {
                status,
                encoded_receipt: record.encoded_receipt,
                reference: record.reference,
                output: record.output,
            },
        );
        Ok(status)
    }

    /// Rebuild the verified predecessor status from the durable fields alone.
    ///
    /// This is the ledger-free reopen seam: the run and generation come from the
    /// committed authority, the ordinal and counter from the durable status, and
    /// the reference from the four persisted digest/length fields.
    pub async fn reopen_verified_status(
        &self,
        final_generation: &CommittedCheckpointGeneration,
        sink_id: &StreamingIssueComponentId,
    ) -> Result<VerifiedDerivedSinkAttemptStatus, ResultPlaneError> {
        self.check_authority(final_generation)?;
        let generation = final_generation.generation();
        let Some(record) = self.substrate.record(&generation, sink_id) else {
            return refuse(SinkFinalizationFailureCode::MissingStatus);
        };
        let DerivedSinkStatus::PendingRetry {
            last_ordinal,
            counter_before,
        } = record.status
        else {
            return refuse(SinkFinalizationFailureCode::IllegalTransition);
        };
        let Some(reference) = record.reference else {
            return refuse(SinkFinalizationFailureCode::MissingReceipt);
        };
        VerifiedDerivedSinkAttemptStatus::from_status_owner(
            final_generation,
            sink_id.clone(),
            last_ordinal,
            counter_before,
            DerivedExportReceiptReference::from_status_fields(
                reference.receipt_digest,
                reference.receipt_length,
                reference.embedded_receipt_digest,
                reference.embedded_receipt_length,
            ),
        )
        .map_err(|_| ResultPlaneError::SinkFinalization {
            code: SinkFinalizationFailureCode::OrdinalMismatch,
        })
    }

    /// Strictly reopen the retained export receipt without a live issue ledger.
    ///
    /// The durable document supplies only comparison inputs; every retained fact
    /// is recomputed through the frozen policy and the verified status.
    pub async fn reopen_receipt(
        &self,
        final_generation: &CommittedCheckpointGeneration,
        sink_id: &StreamingIssueComponentId,
        policy: &PreparedStreamingIssuePolicy,
        encoded_budget: &StreamingResourceBudget,
        parsed_budget: &StreamingResourceBudget,
    ) -> Result<BudgetOwnedExportIssueReceipt, ResultPlaneError> {
        let status = self
            .reopen_verified_status(final_generation, sink_id)
            .await?;
        let generation = final_generation.generation();
        let Some(record) = self.substrate.record(&generation, sink_id) else {
            return refuse(SinkFinalizationFailureCode::MissingStatus);
        };
        let Some(encoded) = record.encoded_receipt else {
            return refuse(SinkFinalizationFailureCode::MissingReceipt);
        };
        let lease = encoded_budget
            .acquire(1, encoded.len())
            .await
            .map_err(|_| ResultPlaneError::SinkFinalization {
                code: SinkFinalizationFailureCode::Budget,
            })?;
        let charged = BudgetedCheckpointBytes::new(Bytes::from(encoded), lease).map_err(|_| {
            ResultPlaneError::SinkFinalization {
                code: SinkFinalizationFailureCode::Budget,
            }
        })?;
        let context = DurableExportReceiptValidationContext::from_final_generation_status(
            final_generation,
            policy,
            &status,
        )
        .map_err(|_| ResultPlaneError::SinkFinalization {
            code: SinkFinalizationFailureCode::TamperedReceipt,
        })?;
        restore_durable_export_issue_receipt(charged, &context, parsed_budget)
            .await
            .map_err(|_| ResultPlaneError::SinkFinalization {
                code: SinkFinalizationFailureCode::TamperedReceipt,
            })
    }
}

/// Bounded restartable supervisor over one generation's pending derived sinks.
///
/// The supervisor holds no cursor of its own: it pages the durable medium in
/// stable sink order, so a replacement process resumes at the same page without
/// inheriting any in-memory state.
#[derive(Debug)]
pub struct DerivedSinkRetrySupervisor {
    store: DerivedSinkStatusStore,
    page_size: NonZeroUsize,
}

impl DerivedSinkRetrySupervisor {
    /// Construct a supervisor with one bounded page size.
    #[must_use]
    pub const fn new(store: DerivedSinkStatusStore, page_size: NonZeroUsize) -> Self {
        Self { store, page_size }
    }

    /// Borrow the durable status owner.
    #[must_use]
    pub const fn store(&self) -> &DerivedSinkStatusStore {
        &self.store
    }

    /// Return one bounded page of pending sinks after the given sink identity.
    pub async fn pending_page(
        &self,
        final_generation: &CommittedCheckpointGeneration,
        after: Option<&StreamingIssueComponentId>,
    ) -> Result<Vec<(StreamingIssueComponentId, DerivedSinkStatus)>, ResultPlaneError> {
        self.store.check_authority(final_generation)?;
        let generation = final_generation.generation();
        let mut page = Vec::new();
        for (sink, status) in self.store.substrate.sinks_for(&generation) {
            if after.is_some_and(|bound| sink.as_str() <= bound.as_str()) {
                continue;
            }
            if !status.is_pending() {
                continue;
            }
            let sink_id = StreamingIssueComponentId::new(sink).map_err(|_| {
                ResultPlaneError::SinkFinalization {
                    code: SinkFinalizationFailureCode::ForeignSink,
                }
            })?;
            page.push((sink_id, status));
            if page.len() == self.page_size.get() {
                break;
            }
        }
        Ok(page)
    }
}

/// Durable authoritative-output writer for one derived sink.
///
/// Together with [`DurableSinkOutputProbe`] this is one of exactly two paths
/// that mint a [`DurableOutputProof`].
#[derive(Debug)]
pub struct DurableSinkOutputWriter {
    run: StreamRunIdentity,
    generation: CheckpointGeneration,
    sink_id: StreamingIssueComponentId,
    substrate: DerivedStatusSubstrate,
}

impl DurableSinkOutputWriter {
    /// Bind a writer to one committed final generation and derived sink.
    pub fn new(
        final_generation: &CommittedCheckpointGeneration,
        sink_id: StreamingIssueComponentId,
        substrate: DerivedStatusSubstrate,
    ) -> Result<Self, ResultPlaneError> {
        if !final_generation.is_final() {
            return refuse(SinkFinalizationFailureCode::NonFinalGeneration);
        }
        Ok(Self {
            run: *final_generation.run(),
            generation: final_generation.generation(),
            sink_id,
            substrate,
        })
    }

    /// Persist the authoritative output and mint its durable proof.
    ///
    /// The move-only attempt token is consumed here, so exactly one durable
    /// output can exist per admitted attempt.
    pub async fn write(
        &self,
        token: SinkAttemptToken,
        bytes: &[u8],
    ) -> Result<DurableOutputProof, ResultPlaneError> {
        if token.run != self.run {
            return refuse(SinkFinalizationFailureCode::ForeignRun);
        }
        if token.generation != self.generation {
            return refuse(SinkFinalizationFailureCode::ForeignGeneration);
        }
        if token.sink_id != self.sink_id {
            return refuse(SinkFinalizationFailureCode::ForeignSink);
        }
        let Some(record) = self.substrate.record(&self.generation, &self.sink_id) else {
            return refuse(SinkFinalizationFailureCode::MissingStatus);
        };
        let output_length =
            u64::try_from(bytes.len()).map_err(|_| ResultPlaneError::SinkFinalization {
                code: SinkFinalizationFailureCode::Budget,
            })?;
        let output_digest = ContentDigest::from_bytes(*blake3::hash(bytes).as_bytes());
        self.substrate.put(
            &self.generation,
            &self.sink_id,
            StatusRecord {
                status: record.status,
                encoded_receipt: record.encoded_receipt,
                reference: record.reference,
                output: Some(StoredOutput {
                    digest: output_digest,
                    length: output_length,
                }),
            },
        );
        Ok(DurableOutputProof {
            run: self.run,
            generation: self.generation.clone(),
            sink_id: self.sink_id.clone(),
            output_digest,
            output_length,
        })
    }
}

/// Durable authoritative-output probe for one derived sink.
///
/// A process that crashed after its durable write but before its `Complete`
/// compare-and-swap recovers the same proof from here.
#[derive(Debug)]
pub struct DurableSinkOutputProbe {
    run: StreamRunIdentity,
    generation: CheckpointGeneration,
    sink_id: StreamingIssueComponentId,
    substrate: DerivedStatusSubstrate,
}

impl DurableSinkOutputProbe {
    /// Bind a probe to one committed final generation and derived sink.
    pub fn new(
        final_generation: &CommittedCheckpointGeneration,
        sink_id: StreamingIssueComponentId,
        substrate: DerivedStatusSubstrate,
    ) -> Result<Self, ResultPlaneError> {
        if !final_generation.is_final() {
            return refuse(SinkFinalizationFailureCode::NonFinalGeneration);
        }
        Ok(Self {
            run: *final_generation.run(),
            generation: final_generation.generation(),
            sink_id,
            substrate,
        })
    }

    /// Mint the durable proof when an authoritative output already exists.
    pub async fn probe(&self) -> Result<Option<DurableOutputProof>, ResultPlaneError> {
        let Some(record) = self.substrate.record(&self.generation, &self.sink_id) else {
            return Ok(None);
        };
        Ok(record.output.map(|output| DurableOutputProof {
            run: self.run,
            generation: self.generation.clone(),
            sink_id: self.sink_id.clone(),
            output_digest: output.digest,
            output_length: output.length,
        }))
    }
}

/// Build one committed final generation for in-crate tests.
///
/// The report-persistence ordering tests live beside the coordinator, so the
/// committed authority they exercise is minted here rather than reconstructed
/// from checkpoint internals in a second place.
#[cfg(test)]
pub(crate) fn committed_final_generation_for_test(
    run: StreamRunIdentity,
    epoch: u64,
) -> CommittedCheckpointGeneration {
    use crate::streaming::{
        checkpoint::{
            AcquisitionHorizon, AdmissionHorizon, CheckpointCut, CheckpointEpoch,
            CheckpointGenerationCandidate, CheckpointGenerationPublicationProof,
            CheckpointParticipantPlan, CheckpointTerminalReason, DecodeHorizon, DiscoveryHorizon,
            EventTimeWatermark, OrderedActionHorizon, TerminalActionHorizon,
        },
        identity::{GlobalSequence, SessionCausalFrontier},
        reliability::HandledIssueCut,
        unit::{EventTimeUtc, SourcePosition},
    };

    let event_time = EventTimeUtc::new(1).expect("valid event time");
    let cut = CheckpointCut {
        discovered: DiscoveryHorizon::new(SourcePosition::new(1)),
        acquired: AcquisitionHorizon::new(SourcePosition::new(1)),
        decoded: DecodeHorizon::new(SourcePosition::new(1)),
        ordered: OrderedActionHorizon::new(GlobalSequence::new(1)),
        admitted: AdmissionHorizon::new(GlobalSequence::new(1)),
        terminal: TerminalActionHorizon::new(GlobalSequence::new(1)),
        event_watermark: EventTimeWatermark::Hard {
            through: event_time,
        },
        causal_frontier: SessionCausalFrontier {
            through_sequence: GlobalSequence::new(1),
            event_time: Some(event_time),
            digest: ContentDigest::from_bytes([0xe1; 32]),
        },
        handled_issues: HandledIssueCut::empty(),
    };
    let plan = CheckpointParticipantPlan::new([]).expect("valid empty participant plan");
    let candidate = CheckpointGenerationCandidate::new(
        run,
        CheckpointEpoch::new(epoch),
        None,
        cut,
        &plan,
        ContentDigest::from_bytes([0xe2; 32]),
        ContentDigest::from_bytes([0xe3; 32]),
        Vec::new(),
        ContentDigest::from_bytes([0xe4; 32]),
        true,
        Some(CheckpointTerminalReason::Completed),
    )
    .expect("valid generation candidate");
    let proof = CheckpointGenerationPublicationProof::for_generation(candidate.generation());
    candidate
        .promote(
            &run,
            &plan,
            &ContentDigest::from_bytes([0xe2; 32]),
            &ContentDigest::from_bytes([0xe3; 32]),
            proof,
        )
        .expect("promote committed generation")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::streaming::{
        budget::BudgetLimits,
        identity::LogicalReplayRunId,
        reliability::{
            StreamingIssueDisposition, StreamingIssueScopeKind, StreamingIssueThresholdRule,
        },
    };

    fn component(value: &str) -> StreamingIssueComponentId {
        StreamingIssueComponentId::new(value).expect("valid component identity")
    }

    fn policy() -> PreparedStreamingIssuePolicy {
        PreparedStreamingIssuePolicy::new([StreamingIssueThresholdRule::new(
            component("export_retryable"),
            StreamingIssueScopeKind::Export,
            StreamingIssueClass::Retryable,
            None,
            3,
            StreamingIssueDisposition::ExportIncomplete,
            None,
        )
        .expect("valid retryable export rule")])
        .expect("valid export policy")
    }

    fn committed_final(run: StreamRunIdentity, epoch: u64) -> CommittedCheckpointGeneration {
        committed_final_generation_for_test(run, epoch)
    }

    fn budget(items: usize, bytes: usize) -> StreamingResourceBudget {
        StreamingResourceBudget::new(BudgetLimits {
            max_items: items,
            max_bytes: bytes,
        })
        .expect("valid test budget")
    }

    async fn prepared_failure(
        run: StreamRunIdentity,
        generation: &CheckpointGeneration,
        sink_id: &StreamingIssueComponentId,
        attempt_ordinal: u32,
        export_budget: &StreamingResourceBudget,
    ) -> PreparedExportReceiptPersistence {
        let mut reporter =
            BudgetOwnedStreamingIssueReporter::new(run, policy(), budget(64, 128 * 1024))
                .expect("budget-owned reporter");
        let issue = OrdinaryStreamingIssue::export(
            run,
            sink_id.clone(),
            generation.clone(),
            StreamingIssueClass::Retryable,
            ContentDigest::from_bytes([0xc3; 32]),
            attempt_ordinal,
            ContentDigest::from_bytes([0xc4; 32]),
            OrdinaryStreamingFailure::Export(ResultExportError::failure(
                ResultExportFailureCode::Io,
            )),
        )
        .expect("valid export issue");
        reporter
            .prepare_export_attempt_failure(
                &run,
                generation,
                sink_id,
                attempt_ordinal,
                ResultSinkAttemptOutcome::Failed(issue),
                export_budget,
            )
            .await
            .expect("prepare export failure")
            .into_persistence()
    }

    fn block_on<F: Future>(future: F) -> F::Output {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("current-thread runtime");
        tokio::task::LocalSet::new().block_on(&runtime, future)
    }

    #[test]
    fn prepared_export_failure_cannot_mix_decision_and_receipt() {
        block_on(async {
            let run = StreamRunIdentity::new(LogicalReplayRunId::from_bytes([0x71; 32]));
            let committed = committed_final(run, 4);
            let sink_id = component("native_report");
            let export_budget = budget(16, 64 * 1024);
            let persistence =
                prepared_failure(run, &committed.generation(), &sink_id, 0, &export_budget).await;

            // The decision and the receipt travel in one move-only value: the
            // persistence handoff exposes both, and there is no accessor that
            // yields one while leaving the other behind. Consuming the handoff
            // therefore cannot pair a decision with a foreign receipt.
            assert_eq!(persistence.counter_before(), 0);
            assert_eq!(persistence.attempt_ordinal(), 0);
            assert!(!persistence.is_exhausted());
            assert!(!persistence.encoded_bytes().is_empty());
            let reference = persistence.receipt_reference();
            assert_eq!(
                reference.receipt_length(),
                persistence.encoded_bytes().len() as u64
            );
            assert_eq!(
                reference.receipt_digest(),
                &ContentDigest::from_bytes(*blake3::hash(persistence.encoded_bytes()).as_bytes())
            );

            drop(persistence);
            assert_eq!(export_budget.snapshot().used_items, 0);
        });
    }

    #[test]
    fn durable_output_proof_rejects_foreign_run_generation_or_sink() {
        block_on(async {
            let run = StreamRunIdentity::new(LogicalReplayRunId::from_bytes([0x72; 32]));
            let foreign_run = StreamRunIdentity::new(LogicalReplayRunId::from_bytes([0x73; 32]));
            let committed = committed_final(run, 5);
            let foreign_generation = committed_final(run, 6);
            let foreign_run_generation = committed_final(foreign_run, 5);
            let sink_id = component("native_report");
            let substrate = DerivedStatusSubstrate::new();
            let store = DerivedSinkStatusStore::open(run, substrate.clone(), budget(8, 4096));
            store
                .reconcile_initial(&committed, &sink_id)
                .await
                .expect("install initial status");

            let token = store
                .open_attempt(&committed, &sink_id)
                .await
                .expect("admit attempt");
            let writer =
                DurableSinkOutputWriter::new(&committed, sink_id.clone(), substrate.clone())
                    .expect("bind writer");
            let proof = writer.write(token, b"report").await.expect("durable write");

            // The proof binds its own run, generation, and sink, so a store that
            // holds a different authority refuses it before any mutation.
            let foreign_store =
                DerivedSinkStatusStore::open(foreign_run, substrate.clone(), budget(8, 4096));
            let before = substrate.write_count();
            assert_eq!(
                foreign_store
                    .commit_complete(&foreign_run_generation, proof)
                    .await,
                Err(ResultPlaneError::SinkFinalization {
                    code: SinkFinalizationFailureCode::ForeignRun,
                })
            );
            assert_eq!(substrate.write_count(), before);

            let probe = DurableSinkOutputProbe::new(&committed, sink_id, substrate.clone())
                .expect("bind probe");
            let recovered = probe.probe().await.expect("probe").expect("durable output");
            assert_eq!(
                store.commit_complete(&foreign_generation, recovered).await,
                Err(ResultPlaneError::SinkFinalization {
                    code: SinkFinalizationFailureCode::ForeignGeneration,
                })
            );
        });
    }
}
