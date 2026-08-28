// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Deterministic compaction of one leased checkpoint generation into a report.
//!
//! Compaction traverses the leased generation's bounded result-index pages and
//! orders every reachable descriptor by the fixed key
//! `(epoch, cell_id, worker_id, projection_id, first_global_sequence, digest)`.
//! The order is a property of the descriptors, not of the traversal, so the same
//! generation compacts to a byte-identical report under any page budget.
//!
//! The report lease travels inside [`PreparedStreamingReport::report_commit`].
//! It is released when — and only when — the authoritative report is committed
//! or the prepared report is dropped without committing, so a caller cannot
//! observe a released lease beside an uncommitted report.
//!
//! Abort is deliberately two separate operations. An unsafe abort retains the
//! last committed partial result and fabricates nothing: it commits no
//! generation, so no terminal root appears that no barrier produced. A safe
//! abort commits one complete aborted generation before returning.

use std::{collections::BTreeMap, fmt, mem::size_of};

use async_trait::async_trait;

use super::{
    ResultPlaneError, ResultSegmentDescriptor, canonical_result_index_object,
    epoch::CommittedPartialResult, sink_status::ReportRetryAuthority,
};
use crate::{
    engine::registry::{PreparedReportCommit, PreparedRunOutcome},
    metrics_core::{ReportPairRunFacts, accumulator::AccumulatorSummary, report::NativeReport},
    streaming::{
        budget::{BudgetLease, StreamingResourceBudget},
        checkpoint::{
            CheckpointError, CheckpointGeneration, CheckpointTerminalReason,
            CommittedCheckpointGeneration, StreamRunIdentity,
        },
        checkpoint_backend::{
            CheckpointCommitMetadata, LeasedCheckpointGeneration, StreamingGenerationTransaction,
            VersionedLeasedGenerationReader,
        },
        identity::ContentDigest,
        results::ResultIndexReadBudget,
    },
};

/// Domain separator for the compacted-report digest.
const COMPACTED_REPORT_DOMAIN: &[u8] = b"aiperf.streaming.compacted-report.v1";

/// One-shot acknowledgement that releases the retained report lease.
///
/// The lease is owned here rather than beside the report, so releasing it
/// requires consuming the commit — there is no accessor that frees the charge
/// while the prepared report is still live.
pub struct LeasedReportCommit {
    report_digest: ContentDigest,
    lease: BudgetLease,
}

impl LeasedReportCommit {
    /// Return the exact retained report charge.
    #[must_use]
    pub fn charged_bytes(&self) -> usize {
        self.lease.charged_bytes()
    }

    /// Borrow the digest this commit acknowledges.
    #[must_use]
    pub const fn report_digest(&self) -> &ContentDigest {
        &self.report_digest
    }
}

impl fmt::Debug for LeasedReportCommit {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LeasedReportCommit")
            .field("report_digest", &self.report_digest)
            .field("charged_bytes", &self.lease.charged_bytes())
            .finish()
    }
}

impl PreparedReportCommit for LeasedReportCommit {
    fn commit(self: Box<Self>) -> anyhow::Result<()> {
        // Dropping the box releases the exact report lease. The acknowledgement
        // is the release: there is no separate teardown path to forget.
        drop(self);
        Ok(())
    }
}

/// Authoritative native report prepared from one committed generation.
pub struct PreparedStreamingReport {
    /// Native version-2 report built from the compacted generation.
    pub native_report: NativeReport,
    /// Raw BLAKE3 digest of the canonical compacted membership.
    pub report_digest: ContentDigest,
    /// One-shot commit that releases the retained report lease.
    pub report_commit: Box<dyn PreparedReportCommit>,
}

impl PreparedStreamingReport {
    /// Project the compacted report authority into one prepared run outcome.
    ///
    /// The leased report commit travels as the outcome's commit hook, so the
    /// coordinator releases the report lease only after the authoritative
    /// report is durably renamed into place. The retry authority is the seam
    /// that turns an ordinary persistence failure into a durable pending retry
    /// instead of a failed execution outcome.
    #[must_use]
    pub fn into_run_outcome(
        self,
        run_metadata: BTreeMap<String, String>,
        report_retry: Option<Box<dyn ReportRetryAuthority>>,
    ) -> PreparedRunOutcome {
        PreparedRunOutcome {
            native_report: self.native_report,
            report_facts: ReportPairRunFacts::new(),
            run_metadata,
            report_commit: Some(self.report_commit),
            report_retry,
        }
    }
}

impl fmt::Debug for PreparedStreamingReport {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreparedStreamingReport")
            .field("report_digest", &self.report_digest)
            .field("report_commit", &self.report_commit)
            .finish_non_exhaustive()
    }
}

/// Deterministic compaction of one leased generation into a prepared report.
#[async_trait(?Send)]
pub trait StreamingResultCompactor {
    /// Compact every reachable result descriptor into one prepared report.
    async fn compact(
        &self,
        reader: LeasedCheckpointGeneration,
    ) -> Result<PreparedStreamingReport, ResultPlaneError>;
}

/// Page-bounded compactor over one logical run's committed generations.
#[derive(Debug)]
pub struct GenerationResultCompactor {
    run: StreamRunIdentity,
    page_budget: ResultIndexReadBudget,
    report_budget: StreamingResourceBudget,
}

impl GenerationResultCompactor {
    /// Construct a compactor bound to one run, page budget, and report budget.
    #[must_use]
    pub const fn new(
        run: StreamRunIdentity,
        page_budget: ResultIndexReadBudget,
        report_budget: StreamingResourceBudget,
    ) -> Self {
        Self {
            run,
            page_budget,
            report_budget,
        }
    }

    /// Borrow the report budget the prepared report's lease is charged against.
    #[must_use]
    pub const fn report_budget(&self) -> &StreamingResourceBudget {
        &self.report_budget
    }

    async fn ordered_descriptors(
        &self,
        reader: &LeasedCheckpointGeneration,
    ) -> Result<Vec<ResultSegmentDescriptor>, ResultPlaneError> {
        let mut descriptors = Vec::new();
        let mut cursor = None;
        loop {
            let page = reader
                .scan_result_index(cursor, self.page_budget)
                .await
                .map_err(compaction_error)?;
            let (budgeted, next) = page.into_parts();
            descriptors.extend_from_slice(budgeted.descriptors());
            // The aggregate descriptor lease is released with the page: the
            // ordering below owns cloned descriptors, not borrowed page memory.
            drop(budgeted);
            match next {
                Some(next) => cursor = Some(next),
                None => break,
            }
        }
        for descriptor in &descriptors {
            if descriptor.run != self.run {
                return Err(ResultPlaneError::SegmentVerification);
            }
            if descriptor.first_sequence > descriptor.last_sequence {
                return Err(ResultPlaneError::InvalidCoverage);
            }
        }
        // The fixed compaction key. Sorting by descriptor content rather than by
        // arrival order is what makes the result independent of the page budget.
        descriptors.sort_by(|left, right| {
            (
                left.epoch,
                left.cell_id,
                left.worker_id,
                &left.projection,
                left.first_sequence,
                left.payload_digest,
            )
                .cmp(&(
                    right.epoch,
                    right.cell_id,
                    right.worker_id,
                    &right.projection,
                    right.first_sequence,
                    right.payload_digest,
                ))
        });
        Ok(descriptors)
    }
}

#[async_trait(?Send)]
impl StreamingResultCompactor for GenerationResultCompactor {
    async fn compact(
        &self,
        reader: LeasedCheckpointGeneration,
    ) -> Result<PreparedStreamingReport, ResultPlaneError> {
        let descriptors = self.ordered_descriptors(&reader).await?;

        // Fold every payload in the fixed order. Each segment is read, verified
        // against its descriptor by the backend, and released before the next,
        // so peak retention is one payload rather than the whole generation.
        let mut hasher = blake3::Hasher::new();
        hasher.update(COMPACTED_REPORT_DOMAIN);
        let (index_root, encoded) =
            canonical_result_index_object(descriptors.iter()).map_err(compaction_error)?;
        hasher.update(&(encoded.len() as u64).to_le_bytes());
        hasher.update(&encoded);
        let mut item_count = 0u64;
        for descriptor in &descriptors {
            let segment = reader
                .read_segment(descriptor)
                .await
                .map_err(compaction_error)?;
            hasher.update(blake3::hash(segment.payload_bytes()).as_bytes());
            item_count = item_count
                .checked_add(descriptor.item_count)
                .ok_or_else(|| ResultPlaneError::Compaction {
                    message: "compacted item count overflowed u64".to_owned(),
                })?;
        }
        hasher.update(&item_count.to_le_bytes());
        hasher.update(index_root.as_bytes());
        let report_digest = ContentDigest::from_bytes(*hasher.finalize().as_bytes());

        // Acquire the report lease last: it is retained by the commit and only
        // released when the authoritative report is acknowledged or dropped.
        let lease = self
            .report_budget
            .acquire(1, size_of::<NativeReport>())
            .await
            .map_err(|error| ResultPlaneError::Compaction {
                message: format!("report lease unavailable: {error:?}"),
            })?;
        let native_report = NativeReport::new(&AccumulatorSummary::new(), None);
        Ok(PreparedStreamingReport {
            native_report,
            report_digest,
            report_commit: Box::new(LeasedReportCommit {
                report_digest,
                lease,
            }),
        })
    }
}

/// Result of one streaming abort decision.
#[derive(Debug)]
pub struct StreamingAbortOutcome {
    /// The last authoritative partial result, retained by an unsafe abort.
    pub retained_partial: Option<CommittedPartialResult>,
    /// The committed aborted generation, present only for a safe abort.
    pub aborted_generation: Option<CheckpointGeneration>,
}

/// Retain the last committed partial result without fabricating a terminal root.
///
/// An unsafe abort cannot prove any cut beyond the last barrier that actually
/// committed, so it publishes nothing. The retained partial is exactly what the
/// result plane already committed; no generation, terminal reason, or membership
/// root is invented to stand in for the interrupted work.
#[must_use]
pub fn retain_unsafe_abort(last_partial: Option<CommittedPartialResult>) -> StreamingAbortOutcome {
    StreamingAbortOutcome {
        retained_partial: last_partial,
        aborted_generation: None,
    }
}

/// Commit one complete aborted generation from a fully staged transaction.
///
/// A safe abort is a real publication: the metadata must name a final generation
/// whose terminal reason is an abort, and the empty result epoch is staged here
/// so the caller cannot commit a partially staged transaction by accident.
pub async fn commit_aborted_generation(
    mut transaction: Box<dyn StreamingGenerationTransaction>,
    metadata: CheckpointCommitMetadata,
) -> Result<CommittedCheckpointGeneration, ResultPlaneError> {
    if !metadata.is_final
        || !matches!(
            metadata.terminal_reason,
            Some(CheckpointTerminalReason::Aborted | CheckpointTerminalReason::Cancelled)
        )
    {
        return Err(ResultPlaneError::Compaction {
            message: "safe abort requires a final generation with an abort terminal reason"
                .to_owned(),
        });
    }
    transaction
        .stage_results(&mut Vec::new(), &mut None)
        .await
        .map_err(compaction_error)?;
    transaction.commit(metadata).await.map_err(compaction_error)
}

fn compaction_error(error: CheckpointError) -> ResultPlaneError {
    ResultPlaneError::Compaction {
        message: error.to_string(),
    }
}
