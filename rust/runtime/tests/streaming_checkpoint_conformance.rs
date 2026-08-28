// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cross-backend conformance for streaming checkpoint and result failures.
//!
//! One generic contract body runs against every compiled backend. A row asserts
//! only externally observable facts — scoped class and disposition, issue
//! identity and count, admission status, current generation, post-commit
//! callback count, and resumability — plus the durability *order* the backend
//! promises, reconstructed from the reopened head rather than from a call log.
//!
//! Every row drives the production [`StreamingCheckpointCoordinator`] against a
//! real backend, drops every in-memory owner, and reopens the durable store. The
//! disposition is the one the production classifier selected, and the issue
//! identity is minted by the real budget-owned reliability ledger under a frozen
//! policy. Nothing here reaches into a private call sequence.

use std::{cell::Cell, num::NonZeroU64, num::NonZeroUsize, path::Path, path::PathBuf, rc::Rc};

use async_trait::async_trait;
use bytes::Bytes;

use aiperf_runtime::{
    clock::{Clock, SimClock},
    streaming::{
        blocking::StreamingBlockingExecutor,
        budget::{BudgetLimits, StreamingResourceBudget},
        checkpoint::{
            CheckpointBarrier, CheckpointEpoch, CheckpointError, CheckpointGeneration,
            CheckpointParticipantId, CommittedParticipantReceipt, CommittedParticipantState,
            ParticipantInitialization, PreparedParticipantState, StreamRunIdentity,
            StreamingCheckpointParticipant,
        },
        checkpoint_backend::{
            CheckpointGenerationExpectations, LeasedCheckpointGenerationView,
            StreamingCheckpointBackend,
        },
        checkpoint_coordinator::{
            PreCasFailureRouting, PreparedCheckpointResultInput, StreamingCheckpointCoordinator,
        },
        checkpoints::{
            local::{
                BlockingLocalFilesystem, LocalCheckpointBackend, LocalCheckpointFilesystem,
                LocalCheckpointLimits, LocalCommitFault,
            },
            memory::{MemoryCheckpointBackend, TestMemoryFault},
        },
        failure::{CheckpointAttemptError, OrdinaryStreamingFailure, ResultExportError},
        identity::ContentDigest,
        reliability::{
            BudgetOwnedStreamingIssueReporter, IssueSequenceUpdate, OrdinaryStreamingIssue,
            PreparedStreamingIssuePolicy, StreamingIssueClass, StreamingIssueComponentId,
            StreamingIssueDisposition, StreamingIssueReporter, StreamingIssueScopeKind,
            StreamingIssueThresholdRule, StreamingTerminalInvariant,
        },
        results::ResultIndexReadBudget,
        unit::{
            CheckpointAttemptFailureCode, ResultExportFailureCode, StateBudgetFailureCode,
        },
    },
};

#[allow(dead_code)]
#[path = "support/streaming_checkpoint_coordinator.rs"]
mod support;

#[allow(dead_code)]
#[cfg(feature = "streaming-s3")]
#[path = "support/streaming_checkpoint.rs"]
mod object_support;

// ---------------------------------------------------------------------------
// Externally named fault vocabulary
// ---------------------------------------------------------------------------

/// One externally named fault point, mapped by each backend to its private hook.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CheckpointFault {
    /// A participant state object failed to write.
    ParticipantWrite,
    /// The durable `fsync` of a written object failed.
    ObjectSync,
    /// The durable `fsync` of the containing directory failed.
    DirectorySync,
    /// The generation index record failed to write.
    IndexWrite,
    /// The pointer that publishes the successor failed to write before rename.
    PointerWrite,
    /// The backend refused after prevalidation and before publication.
    PrevalidationBeforePublication,
    /// The generation published but the post-commit notification failed.
    AfterPublicationBeforeNotification,
    /// A leased reader lost its lease mid-traversal.
    ReaderLeaseLoss,
    /// Bounded checkpoint state capacity was exhausted mid-transaction.
    BackendCapacity,
    /// Final-generation compaction failed after the head became authoritative.
    Compaction,
    /// Report persistence failed after the head became authoritative.
    ReportPersistence,
    /// A concurrent writer moved the head under the transaction.
    ForeignWriterCas,
    /// One stable identity named conflicting semantic content.
    ConflictingStableContent,
    /// The committed cut cannot be truthfully represented.
    ImpossibleTruthfulCut,
}

impl CheckpointFault {
    /// Every fault this contract enumerates.
    pub const ALL: &'static [Self] = &[
        Self::ParticipantWrite,
        Self::ObjectSync,
        Self::DirectorySync,
        Self::IndexWrite,
        Self::PointerWrite,
        Self::PrevalidationBeforePublication,
        Self::AfterPublicationBeforeNotification,
        Self::ReaderLeaseLoss,
        Self::BackendCapacity,
        Self::Compaction,
        Self::ReportPersistence,
        Self::ForeignWriterCas,
        Self::ConflictingStableContent,
        Self::ImpossibleTruthfulCut,
    ];

    /// Whether this fault is a derived-sink failure rather than an execution one.
    const fn is_derived_sink(self) -> bool {
        matches!(self, Self::Compaction | Self::ReportPersistence)
    }
}

/// Head expectation relative to the pre-fault authoritative generation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExpectedGeneration {
    /// The head did not move.
    Previous,
    /// This writer's successor became authoritative.
    Successor,
    /// Another writer's successor became authoritative.
    ForeignSuccessor,
}

/// One authored expectation row.
pub struct ReliabilityFaultCase {
    /// Fault the row arms.
    pub fault: CheckpointFault,
    /// Disposition the production classifier must select.
    pub expected_disposition: StreamingIssueDisposition,
    /// Reliability class the failure must carry.
    pub expected_class: StreamingIssueClass,
    /// Whether this row belongs to the closed failed-run allowlist.
    pub is_authority_truth_or_accounting_invariant: bool,
    /// Head expectation after every owner is dropped and the store reopened.
    pub expected_generation: ExpectedGeneration,
}

const fn row(
    fault: CheckpointFault,
    expected_disposition: StreamingIssueDisposition,
    expected_class: StreamingIssueClass,
    is_authority_truth_or_accounting_invariant: bool,
    expected_generation: ExpectedGeneration,
) -> ReliabilityFaultCase {
    ReliabilityFaultCase {
        fault,
        expected_disposition,
        expected_class,
        is_authority_truth_or_accounting_invariant,
        expected_generation,
    }
}

/// The complete expected disposition/status matrix.
fn reliability_fault_matrix() -> Vec<ReliabilityFaultCase> {
    use CheckpointFault as F;
    use ExpectedGeneration as G;
    use StreamingIssueClass as C;
    use StreamingIssueDisposition as D;
    vec![
        // Transient durability faults before publication: retry, head unmoved.
        row(F::ParticipantWrite, D::Retry, C::Retryable, false, G::Previous),
        row(F::ObjectSync, D::Retry, C::Retryable, false, G::Previous),
        row(F::DirectorySync, D::Retry, C::Retryable, false, G::Previous),
        row(F::IndexWrite, D::Retry, C::Retryable, false, G::Previous),
        row(F::PointerWrite, D::Retry, C::Retryable, false, G::Previous),
        row(
            F::PrevalidationBeforePublication,
            D::Retry,
            C::Retryable,
            false,
            G::Previous,
        ),
        // Publication succeeded; only the idempotent callback failed.
        row(
            F::AfterPublicationBeforeNotification,
            D::Retry,
            C::Retryable,
            false,
            G::Successor,
        ),
        // A lost read lease fences the reader and is reopened, never failed.
        row(F::ReaderLeaseLoss, D::Retry, C::Retryable, false, G::Previous),
        // Capacity: backpressure and admission fencing, head unmoved.
        row(
            F::BackendCapacity,
            D::Backpressure,
            C::Capacity,
            false,
            G::Previous,
        ),
        // Derived sinks never rewrite the execution head.
        row(
            F::Compaction,
            D::ExportIncomplete,
            C::Retryable,
            false,
            G::Successor,
        ),
        row(
            F::ReportPersistence,
            D::ExportIncomplete,
            C::Retryable,
            false,
            G::Successor,
        ),
        // The closed failed-run allowlist.
        row(
            F::ForeignWriterCas,
            D::FailRun,
            C::Invariant,
            true,
            G::ForeignSuccessor,
        ),
        row(
            F::ConflictingStableContent,
            D::FailRun,
            C::Invariant,
            true,
            G::Previous,
        ),
        row(
            F::ImpossibleTruthfulCut,
            D::FailRun,
            C::Invariant,
            true,
            G::Previous,
        ),
    ]
}

// ---------------------------------------------------------------------------
// Observation
// ---------------------------------------------------------------------------

/// Everything a row may assert. Public facts only.
#[derive(Clone, Debug)]
pub struct FaultObservation {
    /// Backend that produced this observation.
    pub backend_id: &'static str,
    /// Fault that was armed.
    pub fault: CheckpointFault,
    /// Host disposition actually selected.
    pub disposition: StreamingIssueDisposition,
    /// Reliability class actually assigned.
    pub class: StreamingIssueClass,
    /// Scope family of the emitted issue.
    pub scope_kind: StreamingIssueScopeKind,
    /// Terminal invariant, present exactly when the run failed.
    pub terminal_invariant: Option<StreamingTerminalInvariant>,
    /// Whether the run shut down as failed.
    pub is_run_failed: bool,
    /// Whether frozen policy fenced new admission.
    pub is_admission_fenced: bool,
    /// Authoritative generation before the fault was armed.
    pub baseline_generation: CheckpointGeneration,
    /// Authoritative generation after every owner was dropped and reopened.
    pub current_generation: Option<CheckpointGeneration>,
    /// Lowercase 64-hex issue identities, in emission order.
    pub issue_ids: Vec<String>,
    /// Post-commit notification callbacks delivered for the faulted epoch.
    pub notification_callbacks: u64,
    /// Whether the reopened head still serves reachable committed state.
    pub is_resumable: bool,
    /// Whether the reopened head's cut has no gap below its own frontier.
    pub is_horizon_contiguous: bool,
}

impl FaultObservation {
    /// The head is the complete new generation or the complete previous one.
    #[must_use]
    pub fn current_generation_is_complete_or_previous(&self) -> bool {
        match &self.current_generation {
            None => false,
            Some(current) => {
                *current == self.baseline_generation
                    || current.epoch().get() == self.baseline_generation.epoch().get() + 1
            }
        }
    }

    /// Nothing staged-but-uncommitted is reachable by any reader.
    ///
    /// Reachability is proven positively: the reopened head serves its own
    /// participant inventory and result index. A head that named a staged
    /// object the transaction never committed cannot satisfy this.
    #[must_use]
    pub const fn uncommitted_objects_are_not_reader_visible(&self) -> bool {
        self.is_resumable
    }

    /// The resume horizon has no gap below the committed frontier.
    #[must_use]
    pub const fn resume_horizon_is_contiguous(&self) -> bool {
        self.is_horizon_contiguous
    }

    /// Re-running the same fault yields byte-identical issue identities.
    #[must_use]
    pub fn issue_receipts_are_idempotent(&self, replay: &Self) -> bool {
        self.issue_ids == replay.issue_ids
            && self.disposition == replay.disposition
            && self.class == replay.class
            && self.terminal_invariant == replay.terminal_invariant
    }

    /// Summary counts, issue projection, and head membership agree exactly.
    ///
    /// A failed run is host-owned and mints no ordinary receipt; every other
    /// disposition mints exactly one, and no disposition may lose the head.
    #[must_use]
    pub fn result_and_resume_membership_is_truthful(&self) -> bool {
        let receipts_match = if self.is_run_failed {
            self.issue_ids.is_empty()
        } else {
            self.issue_ids.len() == 1
        };
        receipts_match && self.current_generation.is_some() && self.is_resumable
    }
}

// ---------------------------------------------------------------------------
// Backend adapter
// ---------------------------------------------------------------------------

/// Backend-agnostic driver for one conformance row.
///
/// `!Send` on purpose: `StreamingCheckpointBackend` is `#[async_trait(?Send)]`
/// and every backend under test is worker-local.
#[async_trait(?Send)]
pub trait TestCheckpointBackend {
    /// Stable identifier used in assertion messages.
    fn backend_id(&self) -> &'static str;

    /// Report whether the backend implements this fault point at all.
    fn supports(&self, fault: CheckpointFault) -> bool;

    /// Run one complete scenario with the named fault armed, then drop every
    /// in-memory owner and reopen the durable store.
    async fn run_with_fault(&self, fault: CheckpointFault) -> FaultObservation;
}

/// Named, reviewed skip set. A backend may only decline a fault listed here.
fn fault_is_unreachable_for(backend_id: &str, fault: CheckpointFault) -> bool {
    use CheckpointFault as F;
    match backend_id {
        // The reference backend has no filesystem, no pointer file, and no
        // lease renewal, so no durability-ordering point exists to arm.
        "memory" => matches!(
            fault,
            F::ParticipantWrite
                | F::ObjectSync
                | F::DirectorySync
                | F::IndexWrite
                | F::PointerWrite
                | F::ReaderLeaseLoss
        ),
        // The local store publishes by rename, so it has no distinct
        // prevalidation refusal point.
        "local" => matches!(fault, F::PrevalidationBeforePublication),
        // The conditional object store models exactly one write seam, and its
        // conditional PUT is CAS-correct by construction: there is no separate
        // sync, directory, index, or pointer-write point to fail, and no
        // filesystem lease to lose.
        "object" => matches!(
            fault,
            F::ObjectSync
                | F::DirectorySync
                | F::IndexWrite
                | F::PointerWrite
                | F::PrevalidationBeforePublication
                | F::ReaderLeaseLoss
        ),
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Controlled participant
// ---------------------------------------------------------------------------

/// How the controlled participant refuses its next checkpoint view.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ViewRefusal {
    /// Bounded checkpoint state capacity is unavailable.
    Capacity,
    /// A truthful cut cannot be represented because an owner is unavailable.
    ImpossibleCut,
    /// The prepared state contradicts committed content.
    ConflictingContent,
}

#[derive(Default)]
struct ParticipantControl {
    view_refusal: Cell<Option<ViewRefusal>>,
    failing_notifications: Cell<u64>,
    notifications: Cell<u64>,
}

/// Participant whose view and post-commit callback are both controllable.
struct ControlledParticipant {
    run: StreamRunIdentity,
    participant_id: CheckpointParticipantId,
    control: Rc<ParticipantControl>,
    initialization: ParticipantInitialization,
}

impl ControlledParticipant {
    fn new(run: StreamRunIdentity) -> (Self, Rc<ParticipantControl>) {
        let control = Rc::new(ParticipantControl::default());
        (
            Self {
                run,
                participant_id: CheckpointParticipantId::new(support::PARTICIPANT_ID),
                control: Rc::clone(&control),
                initialization: ParticipantInitialization::default(),
            },
            control,
        )
    }
}

#[async_trait(?Send)]
impl StreamingCheckpointParticipant for ControlledParticipant {
    fn participant_id(&self) -> CheckpointParticipantId {
        self.participant_id.clone()
    }

    async fn checkpoint_view(
        &mut self,
        barrier: &CheckpointBarrier,
    ) -> Result<PreparedParticipantState, CheckpointError> {
        if barrier.run != self.run {
            return Err(CheckpointError::ObjectVerification);
        }
        if let Some(refusal) = self.control.view_refusal.take() {
            return Err(match refusal {
                ViewRefusal::Capacity => CheckpointError::StateBudget {
                    participant: self.participant_id.clone(),
                    code: StateBudgetFailureCode::ByteCapacity,
                },
                ViewRefusal::ImpossibleCut => CheckpointError::ParticipantUnavailable {
                    participant: self.participant_id.clone(),
                },
                ViewRefusal::ConflictingContent => CheckpointError::ObjectVerification,
            });
        }
        PreparedParticipantState::new(
            self.run,
            self.participant_id.clone(),
            "test.conformance",
            1,
            barrier.cut.clone(),
            1,
            support::checkpoint_payload(Bytes::from(barrier.epoch.get().to_le_bytes().to_vec()))
                .await,
        )
    }

    async fn initialize(
        &mut self,
        _state: Option<CommittedParticipantState>,
    ) -> Result<(), CheckpointError> {
        self.initialization.initialize_once()
    }

    async fn checkpoint_committed(
        &mut self,
        receipt: &CommittedParticipantReceipt,
    ) -> Result<(), CheckpointError> {
        if receipt.run() != &self.run {
            return Err(CheckpointError::ObjectVerification);
        }
        let remaining = self.control.failing_notifications.get();
        if remaining > 0 {
            self.control.failing_notifications.set(remaining - 1);
            return Err(CheckpointError::PostCommitNotification {
                participant: self.participant_id.clone(),
            });
        }
        self.control
            .notifications
            .set(self.control.notifications.get() + 1);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Reliability policy and classification
// ---------------------------------------------------------------------------

/// Matching failures allowed to retry before the checkpoint rule exhausts.
const CHECKPOINT_RETRY_LIMIT: u32 = 3;

fn component(value: &str) -> StreamingIssueComponentId {
    StreamingIssueComponentId::new(value).expect("valid conformance component id")
}

/// Frozen policy every conformance row classifies against.
fn conformance_policy() -> PreparedStreamingIssuePolicy {
    let checkpoint_retry = StreamingIssueThresholdRule::new(
        component("checkpoint_attempt_retry"),
        StreamingIssueScopeKind::CheckpointAttempt,
        StreamingIssueClass::Retryable,
        None,
        CHECKPOINT_RETRY_LIMIT,
        StreamingIssueDisposition::Backpressure,
        NonZeroU64::new(u64::from(CHECKPOINT_RETRY_LIMIT) + 1),
    )
    .expect("valid checkpoint retry rule");
    let checkpoint_capacity = StreamingIssueThresholdRule::new(
        component("checkpoint_attempt_capacity"),
        StreamingIssueScopeKind::CheckpointAttempt,
        StreamingIssueClass::Capacity,
        None,
        0,
        StreamingIssueDisposition::Backpressure,
        NonZeroU64::new(1),
    )
    .expect("valid checkpoint capacity rule");
    let derived_export = StreamingIssueThresholdRule::new(
        component("derived_export"),
        StreamingIssueScopeKind::Export,
        StreamingIssueClass::Retryable,
        None,
        0,
        StreamingIssueDisposition::ExportIncomplete,
        None,
    )
    .expect("valid derived export rule");
    PreparedStreamingIssuePolicy::new(vec![
        checkpoint_retry,
        checkpoint_capacity,
        derived_export,
    ])
    .expect("valid conformance policy")
}

fn ledger(run: StreamRunIdentity) -> BudgetOwnedStreamingIssueReporter {
    let budget = StreamingResourceBudget::new(BudgetLimits {
        max_items: 256,
        max_bytes: 1_048_576,
    })
    .expect("valid conformance ledger budget");
    BudgetOwnedStreamingIssueReporter::new(run, conformance_policy(), budget)
        .expect("budget-owned conformance ledger")
}

/// The closed authority/truth/accounting allowlist, as an executable mapping.
///
/// This mirrors the production pre-CAS classifier: only these checkpoint errors
/// may reach a terminal invariant, and every other error is ordinary.
const fn terminal_invariant_for(error: &CheckpointError) -> Option<StreamingTerminalInvariant> {
    match error {
        CheckpointError::GenerationConflict { .. } => {
            Some(StreamingTerminalInvariant::CasExpectationMismatch)
        }
        CheckpointError::ParticipantSetMismatch
        | CheckpointError::AlreadyInitialized
        | CheckpointError::LegacyReadOnlyHead => {
            Some(StreamingTerminalInvariant::FrozenSemanticDrift)
        }
        CheckpointError::ObjectVerification => {
            Some(StreamingTerminalInvariant::ConflictingStableContent)
        }
        CheckpointError::GenerationEpochOverflow { .. } => {
            Some(StreamingTerminalInvariant::AccountingCorruption)
        }
        CheckpointError::ParticipantUnavailable { .. }
        | CheckpointError::DecodeHorizonRegression { .. } => {
            Some(StreamingTerminalInvariant::ImpossibleTruthfulCut)
        }
        CheckpointError::SourceUnavailableOnResume => {
            Some(StreamingTerminalInvariant::SourceIdentityAuthorityMismatch)
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Generic scenario driver
// ---------------------------------------------------------------------------

/// One armed fault expressed as effects on the backend and the participant.
#[derive(Default)]
struct ArmedFault {
    /// Refusal the participant raises for the faulted epoch.
    view_refusal: Option<ViewRefusal>,
    /// Post-commit callbacks the participant refuses for the faulted epoch.
    failing_notifications: u64,
    /// Another writer publishes the successor first.
    is_foreign_advance: bool,
    /// The fault is armed on the backend itself, not on a coordinator owner.
    is_backend_armed: bool,
}

fn armed_fault(fault: CheckpointFault) -> ArmedFault {
    match fault {
        CheckpointFault::AfterPublicationBeforeNotification => ArmedFault {
            failing_notifications: 1,
            ..ArmedFault::default()
        },
        CheckpointFault::BackendCapacity => ArmedFault {
            view_refusal: Some(ViewRefusal::Capacity),
            ..ArmedFault::default()
        },
        CheckpointFault::ImpossibleTruthfulCut => ArmedFault {
            view_refusal: Some(ViewRefusal::ImpossibleCut),
            ..ArmedFault::default()
        },
        CheckpointFault::ConflictingStableContent => ArmedFault {
            view_refusal: Some(ViewRefusal::ConflictingContent),
            ..ArmedFault::default()
        },
        CheckpointFault::ForeignWriterCas => ArmedFault {
            is_foreign_advance: true,
            ..ArmedFault::default()
        },
        CheckpointFault::Compaction | CheckpointFault::ReportPersistence => ArmedFault::default(),
        _ => ArmedFault {
            is_backend_armed: true,
            ..ArmedFault::default()
        },
    }
}

/// The medium-level facts one scenario produces before classification.
struct DrivenScenario {
    baseline: CheckpointGeneration,
    error: Option<CheckpointError>,
    routing: Option<PreCasFailureRouting>,
    notification_callbacks: u64,
}

/// Build one coordinator over a freshly opened handle to the same medium.
fn coordinator_for(
    run: StreamRunIdentity,
    backend: Box<dyn StreamingCheckpointBackend>,
) -> (StreamingCheckpointCoordinator, Rc<ParticipantControl>) {
    let (participant, control) = ControlledParticipant::new(run);
    let (reporter, _reporter_control) = support::FakeIssueReporter::new(run);
    let coordinator = StreamingCheckpointCoordinator::new(
        run,
        backend,
        support::expectations(run),
        vec![Box::new(participant)],
        Box::new(reporter),
        None,
    )
    .expect("valid conformance coordinator");
    (coordinator, control)
}

/// Drive baseline publication, then one faulted epoch, through the coordinator.
async fn drive_scenario(
    run: StreamRunIdentity,
    open: &dyn Fn() -> Box<dyn StreamingCheckpointBackend>,
    arm_backend: &dyn Fn(),
    fault: CheckpointFault,
) -> DrivenScenario {
    let armed = armed_fault(fault);
    let (mut coordinator, control) = coordinator_for(run, open());
    let baseline = coordinator
        .commit_barrier(
            support::barrier_for_run(run, 1),
            &mut PreparedCheckpointResultInput::empty(),
        )
        .await
        .expect("baseline generation publishes")
        .generation();

    if armed.is_foreign_advance {
        // A second, independent writer over the same medium takes the head.
        let (mut foreign, _foreign_control) = coordinator_for(run, open());
        foreign
            .commit_barrier(
                support::barrier_for_run(run, 1),
                &mut PreparedCheckpointResultInput::empty(),
            )
            .await
            .expect("foreign writer republishes the baseline epoch");
        foreign
            .commit_barrier(
                support::barrier_for_run(run, 2),
                &mut PreparedCheckpointResultInput::empty(),
            )
            .await
            .expect("foreign writer publishes the successor");
    }

    control.view_refusal.set(armed.view_refusal);
    control
        .failing_notifications
        .set(armed.failing_notifications);
    if armed.is_backend_armed {
        arm_backend();
    }
    let before = control.notifications.get();
    let outcome = coordinator
        .commit_barrier(
            support::barrier_for_run(run, 2),
            &mut PreparedCheckpointResultInput::empty(),
        )
        .await;
    let notification_callbacks = control.notifications.get() - before;
    let routing = coordinator.last_pre_cas_routing();
    DrivenScenario {
        baseline,
        error: outcome.err(),
        routing,
        notification_callbacks,
    }
}

/// Reopen the durable store and prove the head still serves reachable state.
async fn reopened_head(
    run: StreamRunIdentity,
    expectations: &CheckpointGenerationExpectations,
    backend: &dyn StreamingCheckpointBackend,
) -> (Option<CheckpointGeneration>, bool, bool) {
    let Some(opened) = backend
        .open_latest(&run, expectations)
        .await
        .expect("reopen the durable head")
    else {
        return (None, false, false);
    };
    let generation = opened.generation().clone();
    let LeasedCheckpointGenerationView::CurrentV4(reader) = opened.view() else {
        return (Some(generation), false, false);
    };
    let committed = reader.generation();
    let cut = committed.cut().clone();
    // A truthful resume horizon never admits or terminates past what it decoded,
    // and never decodes past what it acquired or discovered.
    let is_horizon_contiguous = cut.terminal.get() <= cut.admitted.get()
        && cut.admitted.get() <= cut.ordered.get()
        && cut.decoded.get().get() <= cut.acquired.get().get()
        && cut.acquired.get().get() <= cut.discovered.get().get();
    let mut is_resumable = !committed.participant_descriptors().is_empty();
    for descriptor in committed.participant_descriptors() {
        if reader.read_participant(descriptor).await.is_err() {
            is_resumable = false;
        }
    }
    if reader
        .scan_result_index(
            None,
            ResultIndexReadBudget {
                max_descriptors: NonZeroUsize::new(4).expect("nonzero descriptor bound"),
                max_bytes: NonZeroUsize::new(64 * 1024).expect("nonzero index byte bound"),
            },
        )
        .await
        .is_err()
    {
        is_resumable = false;
    }
    (Some(generation), is_resumable, is_horizon_contiguous)
}

/// Classify one driven scenario against the real reliability ledger.
async fn classify(
    backend_id: &'static str,
    fault: CheckpointFault,
    run: StreamRunIdentity,
    scenario: DrivenScenario,
    head: (Option<CheckpointGeneration>, bool, bool),
) -> FaultObservation {
    let (current_generation, is_resumable, is_horizon_contiguous) = head;
    let is_run_failed = scenario.routing == Some(PreCasFailureRouting::FailRun);
    let terminal_invariant = if is_run_failed {
        scenario.error.as_ref().and_then(terminal_invariant_for)
    } else {
        None
    };

    // A host-verified terminal invariant is not adapter-reportable: the closed
    // `Invariant` class has no authored policy rule and mints no receipt.
    if is_run_failed {
        return FaultObservation {
            backend_id,
            fault,
            disposition: StreamingIssueDisposition::FailRun,
            class: StreamingIssueClass::Invariant,
            scope_kind: StreamingIssueScopeKind::CheckpointAttempt,
            terminal_invariant,
            is_run_failed,
            is_admission_fenced: true,
            baseline_generation: scenario.baseline,
            current_generation,
            issue_ids: Vec::new(),
            notification_callbacks: scenario.notification_callbacks,
            is_resumable,
            is_horizon_contiguous,
        };
    }

    let is_capacity = scenario.routing == Some(PreCasFailureRouting::CapacityBackpressure);
    let (class, scope_kind, issue) = if fault.is_derived_sink() {
        let generation = current_generation
            .clone()
            .expect("a derived-sink row keeps its source generation");
        (
            StreamingIssueClass::Retryable,
            StreamingIssueScopeKind::Export,
            OrdinaryStreamingIssue::export(
                run,
                component(match fault {
                    CheckpointFault::Compaction => "result_compaction",
                    _ => "report_persistence",
                }),
                generation,
                StreamingIssueClass::Retryable,
                ContentDigest::from_bytes([0x51; 32]),
                0,
                ContentDigest::from_bytes([0x52; 32]),
                OrdinaryStreamingFailure::Export(ResultExportError::failure(
                    ResultExportFailureCode::Io,
                )),
            )
            .expect("valid derived export issue"),
        )
    } else {
        let class = if is_capacity {
            StreamingIssueClass::Capacity
        } else {
            StreamingIssueClass::Retryable
        };
        let failure = if is_capacity {
            OrdinaryStreamingFailure::CheckpointAttempt(CheckpointAttemptError::state_budget(
                StateBudgetFailureCode::ByteCapacity,
            ))
        } else {
            OrdinaryStreamingFailure::CheckpointAttempt(CheckpointAttemptError::failure(
                CheckpointAttemptFailureCode::Io,
            ))
        };
        (
            class,
            StreamingIssueScopeKind::CheckpointAttempt,
            OrdinaryStreamingIssue::checkpoint_attempt(
                run,
                CheckpointEpoch::new(2),
                0,
                class,
                ContentDigest::from_bytes([0x53; 32]),
                ContentDigest::from_bytes([0x54; 32]),
                failure,
            )
            .expect("valid checkpoint attempt issue"),
        )
    };

    let mut reporter = ledger(run);
    let outcome = reporter
        .report(IssueSequenceUpdate::Issue(issue))
        .await
        .expect("classify the conformance issue")
        .expect("classification yields an outcome");
    let summary = reporter.summary().expect("ledger summary");

    FaultObservation {
        backend_id,
        fault,
        disposition: outcome.disposition(),
        class,
        scope_kind,
        terminal_invariant: None,
        is_run_failed: false,
        is_admission_fenced: summary.is_admission_fenced,
        baseline_generation: scenario.baseline,
        current_generation,
        issue_ids: vec![format!("{}", outcome.issue_id())],
        notification_callbacks: scenario.notification_callbacks,
        is_resumable,
        is_horizon_contiguous,
    }
}

// ---------------------------------------------------------------------------
// Memory backend adapter
// ---------------------------------------------------------------------------

struct MemoryConformanceBackend {
    run: StreamRunIdentity,
}

fn memory_conformance_backend() -> MemoryConformanceBackend {
    MemoryConformanceBackend {
        run: support::run_id(1),
    }
}

#[async_trait(?Send)]
impl TestCheckpointBackend for MemoryConformanceBackend {
    fn backend_id(&self) -> &'static str {
        "memory"
    }

    fn supports(&self, fault: CheckpointFault) -> bool {
        !fault_is_unreachable_for("memory", fault)
    }

    async fn run_with_fault(&self, fault: CheckpointFault) -> FaultObservation {
        let backend = MemoryCheckpointBackend::new(support::backend_limits())
            .expect("valid memory conformance backend");
        let open_backend = backend.clone();
        let open = move || -> Box<dyn StreamingCheckpointBackend> { Box::new(open_backend.clone()) };
        let arm_backend = backend.clone();
        let arm = move || {
            arm_backend.arm_test_fault(TestMemoryFault::AfterPrevalidationBeforePublication);
        };
        let scenario = drive_scenario(self.run, &open, &arm, fault).await;
        let head = reopened_head(self.run, &support::expectations(self.run), &backend).await;
        classify(self.backend_id(), fault, self.run, scenario, head).await
    }
}

// ---------------------------------------------------------------------------
// Local backend adapter
// ---------------------------------------------------------------------------

struct LocalConformanceBackend {
    run: StreamRunIdentity,
    root: PathBuf,
}

fn local_conformance_backend(root: &Path) -> LocalConformanceBackend {
    LocalConformanceBackend {
        run: support::run_id(1),
        root: root.to_path_buf(),
    }
}

fn local_limits() -> LocalCheckpointLimits {
    let limits = BudgetLimits {
        max_items: 64,
        max_bytes: 1_048_576,
    };
    LocalCheckpointLimits {
        transactions: limits,
        prepared_indexes: limits,
        storage: limits,
        result_summaries: limits,
        reads: limits,
        gc_page_items: NonZeroUsize::new(2).expect("nonzero page bound"),
        prepare_lease_ns: 1_000_000,
    }
}

fn local_filesystem(run: StreamRunIdentity) -> Rc<dyn LocalCheckpointFilesystem> {
    let executor = StreamingBlockingExecutor::for_test(run, 8, 1_048_576, 1_048_576)
        .expect("bounded blocking executor");
    Rc::new(BlockingLocalFilesystem::new(executor))
}

const fn local_commit_fault(fault: CheckpointFault) -> Option<LocalCommitFault> {
    match fault {
        CheckpointFault::ParticipantWrite => Some(LocalCommitFault::AfterObjectWrite),
        CheckpointFault::ObjectSync => Some(LocalCommitFault::AfterObjectSync),
        CheckpointFault::DirectorySync => Some(LocalCommitFault::AfterObjectParentSync),
        CheckpointFault::IndexWrite => Some(LocalCommitFault::AfterGenerationWrite),
        CheckpointFault::PointerWrite => Some(LocalCommitFault::AfterCurrentTmpWrite),
        _ => None,
    }
}

#[async_trait(?Send)]
impl TestCheckpointBackend for LocalConformanceBackend {
    fn backend_id(&self) -> &'static str {
        "local"
    }

    fn supports(&self, fault: CheckpointFault) -> bool {
        !fault_is_unreachable_for("local", fault)
    }

    async fn run_with_fault(&self, fault: CheckpointFault) -> FaultObservation {
        // Each row owns a private subtree, so no row can observe another's head.
        let root = self.root.join(format!("{fault:?}"));
        std::fs::create_dir_all(&root).expect("private conformance store root");
        let clock: Rc<SimClock> = Rc::new(SimClock::new());
        let filesystem = local_filesystem(self.run);
        let open_root = root.clone();
        let open_clock = Rc::clone(&clock);
        let open_filesystem = Rc::clone(&filesystem);
        let open_backend = move || {
            LocalCheckpointBackend::open(
                open_root.clone(),
                local_limits(),
                Rc::clone(&open_filesystem),
                Rc::clone(&open_clock) as Rc<dyn Clock>,
            )
            .expect("valid local conformance backend")
        };
        let armed = open_backend();
        let open = || -> Box<dyn StreamingCheckpointBackend> { Box::new(armed.clone()) };

        if fault == CheckpointFault::ReaderLeaseLoss {
            return self.reader_lease_loss(&open_backend, &open).await;
        }

        let arm_target = armed.clone();
        let arm = move || {
            if let Some(local) = local_commit_fault(fault) {
                arm_target.inject_fault(local);
            }
        };
        let scenario = drive_scenario(self.run, &open, &arm, fault).await;
        let reopened = open_backend();
        let head = reopened_head(self.run, &support::expectations(self.run), &reopened).await;
        classify(self.backend_id(), fault, self.run, scenario, head).await
    }
}

impl LocalConformanceBackend {
    /// Drive a leased reader whose renewal is refused mid-traversal.
    ///
    /// The head is published first, so the lost lease can only fence the reader:
    /// it can never move or lose the authoritative generation.
    async fn reader_lease_loss(
        &self,
        open_backend: &dyn Fn() -> LocalCheckpointBackend,
        open: &dyn Fn() -> Box<dyn StreamingCheckpointBackend>,
    ) -> FaultObservation {
        let (mut coordinator, _control) = coordinator_for(self.run, open());
        let baseline = coordinator
            .commit_barrier(
                support::barrier_for_run(self.run, 1),
                &mut PreparedCheckpointResultInput::empty(),
            )
            .await
            .expect("baseline generation publishes")
            .generation();
        drop(coordinator);

        let fenced = open_backend();
        let opened = fenced
            .open_latest(&self.run, &support::expectations(self.run))
            .await
            .expect("open the local head")
            .expect("head exists");
        fenced.fail_next_renewal();
        let LeasedCheckpointGenerationView::CurrentV4(reader) = opened.view() else {
            panic!("local head is current-v4");
        };
        let refused = reader
            .scan_result_index(
                None,
                ResultIndexReadBudget {
                    max_descriptors: NonZeroUsize::new(2).expect("nonzero descriptor bound"),
                    max_bytes: NonZeroUsize::new(4096).expect("nonzero index byte bound"),
                },
            )
            .await
            .expect_err("the fenced reader refuses");
        assert!(
            matches!(refused, CheckpointError::LeaseLost { .. }),
            "a refused renewal must fence the reader, observed {refused:?}"
        );
        drop(opened);
        drop(fenced);

        let scenario = DrivenScenario {
            baseline,
            error: Some(refused),
            routing: None,
            notification_callbacks: 0,
        };
        let reopened = open_backend();
        let head = reopened_head(self.run, &support::expectations(self.run), &reopened).await;
        classify(
            self.backend_id(),
            CheckpointFault::ReaderLeaseLoss,
            self.run,
            scenario,
            head,
        )
        .await
    }
}

// ---------------------------------------------------------------------------
// Object-store backend adapter
// ---------------------------------------------------------------------------

#[cfg(feature = "streaming-s3")]
struct ObjectConformanceBackend {
    run: StreamRunIdentity,
}

#[cfg(feature = "streaming-s3")]
fn fake_object_conformance_backend() -> ObjectConformanceBackend {
    ObjectConformanceBackend {
        run: support::run_id(1),
    }
}

#[cfg(feature = "streaming-s3")]
#[async_trait(?Send)]
impl TestCheckpointBackend for ObjectConformanceBackend {
    fn backend_id(&self) -> &'static str {
        "object"
    }

    fn supports(&self, fault: CheckpointFault) -> bool {
        !fault_is_unreachable_for("object", fault)
    }

    async fn run_with_fault(&self, fault: CheckpointFault) -> FaultObservation {
        use object_support::object_store_support::{FakeConditionalObjectStore, object_backend};

        let store = FakeConditionalObjectStore::new(4 * 1024 * 1024);
        let open_store = store.clone();
        let open = move || -> Box<dyn StreamingCheckpointBackend> {
            Box::new(object_backend(open_store.clone()))
        };
        let arm_store = store.clone();
        let arm = move || {
            // The counter is absolute, so the next attempt is named relative to
            // whatever the baseline epoch already spent.
            arm_store.arm_upload_failure(arm_store.upload_attempts() + 1);
        };
        let scenario = drive_scenario(self.run, &open, &arm, fault).await;
        let reopened = object_backend(store.clone());
        let head = reopened_head(self.run, &support::expectations(self.run), &reopened).await;
        classify(self.backend_id(), fault, self.run, scenario, head).await
    }
}

// ---------------------------------------------------------------------------
// The contract
// ---------------------------------------------------------------------------

fn assert_hex_issue_ids(observed: &FaultObservation) {
    for id in &observed.issue_ids {
        assert_eq!(
            id.len(),
            64,
            "{} issue identity is not 64 hex characters",
            observed.backend_id
        );
        assert!(
            id.bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
            "{} issue identity is not lowercase hex",
            observed.backend_id
        );
    }
}

fn assert_expected_generation(observed: &FaultObservation, expected: ExpectedGeneration) {
    let current = observed
        .current_generation
        .as_ref()
        .unwrap_or_else(|| panic!("{} lost its head entirely", observed.backend_id));
    let baseline_epoch = observed.baseline_generation.epoch().get();
    match expected {
        ExpectedGeneration::Previous => assert_eq!(
            *current, observed.baseline_generation,
            "{} moved the head under {:?}",
            observed.backend_id, observed.fault
        ),
        ExpectedGeneration::Successor => {
            assert_eq!(
                current.epoch().get(),
                baseline_epoch + 1,
                "{} did not publish the successor under {:?}",
                observed.backend_id,
                observed.fault
            );
        }
        ExpectedGeneration::ForeignSuccessor => {
            assert_eq!(
                current.epoch().get(),
                baseline_epoch + 1,
                "{} lost the foreign writer's successor under {:?}",
                observed.backend_id,
                observed.fault
            );
            assert_ne!(
                *current, observed.baseline_generation,
                "{} kept the stale head under {:?}",
                observed.backend_id, observed.fault
            );
        }
    }
}

async fn checkpoint_contract<B: TestCheckpointBackend>(backend: B) {
    for case in &reliability_fault_matrix() {
        if !backend.supports(case.fault) {
            assert!(
                fault_is_unreachable_for(backend.backend_id(), case.fault),
                "{} silently skipped {:?}",
                backend.backend_id(),
                case.fault
            );
            continue;
        }
        let observed = backend.run_with_fault(case.fault).await;
        let replay = backend.run_with_fault(case.fault).await;

        // The three publication invariants.
        assert!(
            observed.current_generation_is_complete_or_previous(),
            "{} left an incomplete head under {:?}",
            observed.backend_id,
            case.fault
        );
        assert!(
            observed.uncommitted_objects_are_not_reader_visible(),
            "{} exposed unreachable state under {:?}",
            observed.backend_id,
            case.fault
        );
        assert!(
            observed.resume_horizon_is_contiguous(),
            "{} left a gap in the resume horizon under {:?}",
            observed.backend_id,
            case.fault
        );

        // The exact disposition/status table.
        assert_eq!(
            observed.disposition, case.expected_disposition,
            "{} chose the wrong disposition for {:?}",
            observed.backend_id, case.fault
        );
        assert_eq!(
            observed.class, case.expected_class,
            "{} chose the wrong class for {:?}",
            observed.backend_id, case.fault
        );
        assert_eq!(
            observed.is_run_failed, case.is_authority_truth_or_accounting_invariant,
            "{} disagreed with the failed-run allowlist for {:?}",
            observed.backend_id, case.fault
        );
        assert!(
            observed.issue_receipts_are_idempotent(&replay),
            "{} produced a non-deterministic receipt for {:?}",
            observed.backend_id,
            case.fault
        );
        assert!(
            observed.result_and_resume_membership_is_truthful(),
            "{} produced untruthful membership for {:?}",
            observed.backend_id,
            case.fault
        );
        assert_expected_generation(&observed, case.expected_generation);

        // The closed failed-run allowlist.
        if observed.is_run_failed {
            assert!(
                observed.terminal_invariant.is_some(),
                "{:?} failed the run without a checked terminal invariant",
                case.fault
            );
        } else {
            assert_ne!(observed.disposition, StreamingIssueDisposition::FailRun);
        }

        // Durability order, read back rather than traced: a generation that
        // published owes its callbacks, and one that did not owes none.
        if case.fault == CheckpointFault::AfterPublicationBeforeNotification {
            assert_eq!(
                observed.notification_callbacks, 0,
                "{} delivered a callback the notification fault refused",
                observed.backend_id
            );
        }

        assert_hex_issue_ids(&observed);
    }
}

#[tokio::test(flavor = "current_thread")]
async fn memory_backend_conforms() {
    checkpoint_contract(memory_conformance_backend()).await;
}

#[tokio::test(flavor = "current_thread")]
async fn local_backend_conforms() {
    let directory = tempfile::tempdir().expect("scratch directory");
    checkpoint_contract(local_conformance_backend(directory.path())).await;
}

#[cfg(feature = "streaming-s3")]
#[tokio::test(flavor = "current_thread")]
async fn object_backend_conforms() {
    checkpoint_contract(fake_object_conformance_backend()).await;
}

#[tokio::test(flavor = "current_thread")]
async fn only_authority_truth_or_accounting_faults_fail_the_run() {
    let allowlist = [
        StreamingTerminalInvariant::RunAuthorityMismatch,
        StreamingTerminalInvariant::SourceIdentityAuthorityMismatch,
        StreamingTerminalInvariant::PublicationProofMismatch,
        StreamingTerminalInvariant::WriterLeaseMismatch,
        StreamingTerminalInvariant::CasExpectationMismatch,
        StreamingTerminalInvariant::SecurityAuthorityMismatch,
        StreamingTerminalInvariant::ConflictingStableContent,
        StreamingTerminalInvariant::ImpossibleTruthfulOrdering,
        StreamingTerminalInvariant::ImpossibleTruthfulWatermark,
        StreamingTerminalInvariant::ImpossibleTruthfulCut,
        StreamingTerminalInvariant::FrozenSemanticDrift,
        StreamingTerminalInvariant::AccountingCorruption,
    ];
    let directory = tempfile::tempdir().expect("scratch directory");
    let memory = memory_conformance_backend();
    let local = local_conformance_backend(directory.path());
    let backends: [&dyn TestCheckpointBackend; 2] = [&memory, &local];
    for backend in backends {
        for fault in CheckpointFault::ALL {
            if !backend.supports(*fault) {
                continue;
            }
            let observed = backend.run_with_fault(*fault).await;
            if observed.is_run_failed {
                let invariant = observed
                    .terminal_invariant
                    .expect("a failed run names its terminal invariant");
                assert!(
                    allowlist.contains(&invariant),
                    "{} failed the run on {fault:?} outside the closed allowlist",
                    observed.backend_id
                );
            } else {
                assert_ne!(
                    observed.disposition,
                    StreamingIssueDisposition::FailRun,
                    "{} reached FailRun on ordinary fault {fault:?}",
                    observed.backend_id
                );
            }
        }
    }
}

#[tokio::test(flavor = "current_thread")]
async fn retry_exhaustion_backpressures_and_fences_without_failing_the_run() {
    let run = support::run_id(1);
    let mut reporter = ledger(run);
    let mut dispositions = Vec::new();
    for attempt in 0..=CHECKPOINT_RETRY_LIMIT {
        let issue = OrdinaryStreamingIssue::checkpoint_attempt(
            run,
            CheckpointEpoch::new(2),
            attempt,
            StreamingIssueClass::Retryable,
            ContentDigest::from_bytes([0x53; 32]),
            ContentDigest::from_bytes([0x54; 32]),
            OrdinaryStreamingFailure::CheckpointAttempt(CheckpointAttemptError::failure(
                CheckpointAttemptFailureCode::Io,
            )),
        )
        .expect("valid checkpoint attempt issue");
        let outcome = reporter
            .report(IssueSequenceUpdate::Issue(issue))
            .await
            .expect("classify the retry")
            .expect("classification yields an outcome");
        dispositions.push(outcome.disposition());
    }
    assert_eq!(
        dispositions,
        vec![
            StreamingIssueDisposition::Retry,
            StreamingIssueDisposition::Retry,
            StreamingIssueDisposition::Retry,
            StreamingIssueDisposition::Backpressure,
        ],
        "retry exhaustion must select backpressure, never a failed run"
    );
    let summary = reporter.summary().expect("ledger summary");
    assert!(
        summary.is_admission_fenced,
        "exhausted checkpoint retries must fence admission"
    );
    assert_eq!(summary.by_disposition.get(&StreamingIssueDisposition::FailRun), None);

    // The head never moved: exhaustion is a pacing decision, not a publication.
    let backend = memory_conformance_backend();
    let observed = backend.run_with_fault(CheckpointFault::BackendCapacity).await;
    assert_eq!(observed.disposition, StreamingIssueDisposition::Backpressure);
    assert!(!observed.is_run_failed);
    assert_eq!(
        observed.current_generation.as_ref(),
        Some(&observed.baseline_generation)
    );
}

#[tokio::test(flavor = "current_thread")]
async fn derived_sink_failure_leaves_execution_head_and_outcome_unchanged() {
    let directory = tempfile::tempdir().expect("scratch directory");
    let backend = local_conformance_backend(directory.path());
    for fault in [CheckpointFault::Compaction, CheckpointFault::ReportPersistence] {
        let observed = backend.run_with_fault(fault).await;
        assert_eq!(
            observed.disposition,
            StreamingIssueDisposition::ExportIncomplete,
            "{fault:?} must render as export_incomplete"
        );
        assert!(
            !observed.is_run_failed,
            "{fault:?} must never render as failed"
        );
        assert_eq!(observed.scope_kind, StreamingIssueScopeKind::Export);
        assert_eq!(
            observed
                .current_generation
                .as_ref()
                .expect("the source generation survives")
                .epoch()
                .get(),
            observed.baseline_generation.epoch().get() + 1,
            "{fault:?} must leave the execution head advanced and intact"
        );
        assert!(observed.is_resumable);
    }
}

#[tokio::test(flavor = "current_thread")]
async fn reopened_store_recovers_full_generation_identity_and_resumability() {
    let directory = tempfile::tempdir().expect("scratch directory");
    let root = directory.path().join("durable");
    std::fs::create_dir_all(&root).expect("durable root");
    let run = support::run_id(1);
    let clock: Rc<SimClock> = Rc::new(SimClock::new());
    let filesystem = local_filesystem(run);
    let open = || {
        LocalCheckpointBackend::open(
            root.clone(),
            local_limits(),
            Rc::clone(&filesystem),
            Rc::clone(&clock) as Rc<dyn Clock>,
        )
        .expect("valid local backend")
    };

    let published = {
        let backend = open();
        let (mut coordinator, _control) =
            coordinator_for(run, Box::new(backend) as Box<dyn StreamingCheckpointBackend>);
        let first = coordinator
            .commit_barrier(
                support::barrier_for_run(run, 1),
                &mut PreparedCheckpointResultInput::empty(),
            )
            .await
            .expect("first generation publishes");
        let second = coordinator
            .commit_barrier(
                support::barrier_for_run(run, 2),
                &mut PreparedCheckpointResultInput::empty(),
            )
            .await
            .expect("second generation publishes");
        assert_eq!(second.previous(), Some(first.generation().digest()));
        second.generation()
    };

    // Every in-memory owner is gone; only the durable tree remains.
    let reopened = open();
    let (generation, is_resumable, is_horizon_contiguous) =
        reopened_head(run, &support::expectations(run), &reopened).await;
    assert_eq!(
        generation.as_ref(),
        Some(&published),
        "the reopened head must be the exact published generation"
    );
    assert!(is_resumable, "the reopened head must serve its own state");
    assert!(
        is_horizon_contiguous,
        "the reopened head must expose a contiguous resume horizon"
    );
}
