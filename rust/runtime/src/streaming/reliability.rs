// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Host-owned streaming issue facts, deterministic policy, and authority vocabulary.
//!
//! This module deliberately stops before budget-owned receipt storage and
//! checkpoint integration. Ordinary owners can construct closed facts, but
//! only the host can construct a live decision or terminal failure outcome.
//!
//! An accepted move-only [`PreparedSessionQuarantineInstall`] is retained by the
//! reporter and contributes the third root of [`HandledIssueCut`]; it is bound to
//! the exact barrier and receipt root it was minted against, so the
//! acknowledgement and the detailed receipt set can only commit together.

use std::{
    cell::{Cell, RefCell},
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet, VecDeque},
    fmt,
    io::{self, Write},
    mem::size_of,
    num::NonZeroU64,
    rc::Rc,
};

use async_trait::async_trait;
use bytes::Bytes;
use serde::{Deserialize, Deserializer, Serialize, Serializer, ser::SerializeSeq};
use serde_json::value::RawValue;

use super::{
    action::{
        ActionTerminalMembershipOutcomeView, CheckedActionFailureTerminalEvidenceView,
        CheckedActionTerminalMembershipView, FrozenActionInventoryView,
    },
    budget::{BudgetError, BudgetLease, LeasedByteBuffer, StreamingResourceBudget},
    checkpoint::{
        BudgetedCheckpointBytes, CheckpointBarrier, CheckpointEpoch, CheckpointError,
        CheckpointGeneration, CheckpointParticipantId, CommittedCheckpointGeneration,
        CommittedParticipantReceipt, CommittedParticipantState, PreparedParticipantState,
        StreamRunIdentity, StreamingCheckpointParticipant,
    },
    failure::{
        OrdinaryStreamingFailure, ResultExportError, ResultExportFailureCode,
        StableStreamingFailure, StreamingFailureStage,
    },
    identity::{
        ContentDigest, GlobalSequence, ImmutableObjectIdentity, StableActionId, StableRecordId,
        StableSessionKey,
    },
    results::{BudgetedResultDescriptor, PreparedResultEpoch, ResultPartition},
    session::SessionQuarantineTombstoneView,
    unit::{SourcePosition, StateBudgetFailureCode},
};

/// Maximum checked length, in bytes, of a stable component identifier.
///
/// The checked constructor enforces this bound, so it is a proven maximum for
/// every derived reservation rather than an authored cushion.
pub const MAX_COMPONENT_ID_BYTES: usize = 128;

/// Stable identifier for a reliability rule, failure code, or host component.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(try_from = "String")]
pub struct StreamingIssueComponentId(String);

impl StreamingIssueComponentId {
    /// Construct a checked lowercase ASCII identifier.
    ///
    /// The retained allocation is shrunk toward its exact length so that budget
    /// charges computed from [`Self::retained_bytes`] track the real allocation
    /// rather than an oversized caller buffer.
    pub fn new(value: impl Into<String>) -> Result<Self, StreamingReliabilityError> {
        let mut value = value.into();
        let bytes = value.as_bytes();
        let is_valid_first = bytes.first().is_some_and(u8::is_ascii_lowercase);
        let is_valid_tail = bytes.get(1..).is_some_and(|tail| {
            tail.iter()
                .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || *byte == b'_')
        });
        if !(1..=MAX_COMPONENT_ID_BYTES).contains(&bytes.len()) || !is_valid_first || !is_valid_tail
        {
            return Err(StreamingReliabilityError::InvalidComponentId);
        }
        value.shrink_to_fit();
        Ok(Self(value))
    }

    /// Borrow the checked stable identifier.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Return the retained heap bytes behind this identifier.
    ///
    /// The charge is the retained capacity, not the length, so a short
    /// identifier held in an oversized allocation cannot bypass the
    /// fixed-memory invariant.
    #[must_use]
    pub fn retained_bytes(&self) -> usize {
        self.0.capacity()
    }
}

impl TryFrom<String> for StreamingIssueComponentId {
    type Error = StreamingReliabilityError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

/// Frozen identity of one stream and its exact immutable source authority.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StreamingInputDomainIdentity {
    stream_identity: ContentDigest,
    source_identity: ImmutableObjectIdentity,
}

impl StreamingInputDomainIdentity {
    /// Bind a stream semantic identity to its immutable source authority.
    #[must_use]
    pub const fn new(
        stream_identity: ContentDigest,
        source_identity: ImmutableObjectIdentity,
    ) -> Self {
        Self {
            stream_identity,
            source_identity,
        }
    }

    /// Borrow the frozen stream semantic identity.
    #[must_use]
    pub const fn stream_identity(&self) -> &ContentDigest {
        &self.stream_identity
    }

    /// Borrow the exact immutable source identity.
    #[must_use]
    pub const fn source_identity(&self) -> &ImmutableObjectIdentity {
        &self.source_identity
    }
}

/// Scope whose ordinary work observed an issue.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "scope", rename_all = "snake_case", deny_unknown_fields)]
pub enum StreamingIssueScope {
    /// Run-wide diagnostic work that loses no membership.
    Run,
    /// One immutable partition generation.
    Partition {
        /// Exact stream/source domain of the partition.
        input_domain: StreamingInputDomainIdentity,
        /// Immutable object generation that failed.
        object: ImmutableObjectIdentity,
    },
    /// One stable source record.
    Record {
        /// Exact stream/source domain of the record.
        input_domain: StreamingInputDomainIdentity,
        /// Stable record identity within the domain.
        record_id: StableRecordId,
    },
    /// One stable logical session.
    Session {
        /// Exact stream/source domain of the session.
        input_domain: StreamingInputDomainIdentity,
        /// Stable session identity within the domain.
        session_key: StableSessionKey,
    },
    /// One stable executable action.
    Action {
        /// Stable action identity in the host action sequence.
        action_id: StableActionId,
    },
    /// One derived exporter attempt against a complete generation identity.
    Export {
        /// Stable exporter identity.
        exporter_id: StreamingIssueComponentId,
        /// Exact immutable checkpoint generation.
        generation: CheckpointGeneration,
    },
    /// One attempt to prepare or publish a successor checkpoint epoch.
    CheckpointAttempt {
        /// Successor checkpoint epoch.
        generation: CheckpointEpoch,
        /// Dense attempt ordinal within that epoch.
        attempt_ordinal: u32,
    },
}

impl StreamingIssueScope {
    /// Return the closed scope kind used by policy matching.
    #[must_use]
    pub const fn kind(&self) -> StreamingIssueScopeKind {
        match self {
            Self::Run => StreamingIssueScopeKind::Run,
            Self::Partition { .. } => StreamingIssueScopeKind::Partition,
            Self::Record { .. } => StreamingIssueScopeKind::Record,
            Self::Session { .. } => StreamingIssueScopeKind::Session,
            Self::Action { .. } => StreamingIssueScopeKind::Action,
            Self::Export { .. } => StreamingIssueScopeKind::Export,
            Self::CheckpointAttempt { .. } => StreamingIssueScopeKind::CheckpointAttempt,
        }
    }

    /// Borrow the exact input domain for an input-scoped issue.
    #[must_use]
    pub const fn input_domain(&self) -> Option<&StreamingInputDomainIdentity> {
        match self {
            Self::Partition { input_domain, .. }
            | Self::Record { input_domain, .. }
            | Self::Session { input_domain, .. } => Some(input_domain),
            Self::Run
            | Self::Action { .. }
            | Self::Export { .. }
            | Self::CheckpointAttempt { .. } => None,
        }
    }

    /// Borrow the immutable partition object when this is partition-scoped.
    #[must_use]
    pub const fn partition_object(&self) -> Option<&ImmutableObjectIdentity> {
        match self {
            Self::Partition { object, .. } => Some(object),
            _ => None,
        }
    }

    /// Return the stable record identity when this is record-scoped.
    #[must_use]
    pub const fn record_id(&self) -> Option<StableRecordId> {
        match self {
            Self::Record { record_id, .. } => Some(*record_id),
            _ => None,
        }
    }

    /// Return the stable session identity when this is session-scoped.
    #[must_use]
    pub const fn session_key(&self) -> Option<StableSessionKey> {
        match self {
            Self::Session { session_key, .. } => Some(*session_key),
            _ => None,
        }
    }

    /// Return the stable action identity when this is action-scoped.
    #[must_use]
    pub const fn action_id(&self) -> Option<StableActionId> {
        match self {
            Self::Action { action_id } => Some(*action_id),
            _ => None,
        }
    }

    /// Borrow the exporter identity when this is export-scoped.
    #[must_use]
    pub const fn exporter_id(&self) -> Option<&StreamingIssueComponentId> {
        match self {
            Self::Export { exporter_id, .. } => Some(exporter_id),
            _ => None,
        }
    }

    /// Borrow the full generation when this is export-scoped.
    #[must_use]
    pub const fn export_generation(&self) -> Option<&CheckpointGeneration> {
        match self {
            Self::Export { generation, .. } => Some(generation),
            _ => None,
        }
    }

    /// Return the successor epoch when this is checkpoint-attempt-scoped.
    #[must_use]
    pub const fn checkpoint_epoch(&self) -> Option<CheckpointEpoch> {
        match self {
            Self::CheckpointAttempt { generation, .. } => Some(*generation),
            _ => None,
        }
    }

    /// Return the attempt ordinal when this is checkpoint-attempt-scoped.
    #[must_use]
    pub const fn checkpoint_attempt_ordinal(&self) -> Option<u32> {
        match self {
            Self::CheckpointAttempt {
                attempt_ordinal, ..
            } => Some(*attempt_ordinal),
            _ => None,
        }
    }
}

/// Reliability class asserted by one closed typed observation.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamingIssueClass {
    /// The same operation may succeed after bounded retry.
    Retryable,
    /// The named input cannot succeed under the validated plan.
    Permanent,
    /// The observation contradicts a host-verified runtime invariant.
    Invariant,
    /// Explicit bounded capacity is currently unavailable.
    Capacity,
}

/// Terminal invariant classification available only through the host classifier.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamingTerminalInvariant {
    /// Logical run authority does not match.
    RunAuthorityMismatch,
    /// Frozen immutable source authority does not match.
    SourceIdentityAuthorityMismatch,
    /// Publication proof does not match committed authority.
    PublicationProofMismatch,
    /// Writer lease authority does not match.
    WriterLeaseMismatch,
    /// A compare-and-swap expectation does not match.
    CasExpectationMismatch,
    /// Runtime security authority does not match.
    SecurityAuthorityMismatch,
    /// One stable identity names conflicting semantic content.
    ConflictingStableContent,
    /// Truthful stable ordering cannot be represented.
    ImpossibleTruthfulOrdering,
    /// A truthful watermark cannot be represented.
    ImpossibleTruthfulWatermark,
    /// A truthful checkpoint cut cannot be represented.
    ImpossibleTruthfulCut,
    /// A frozen semantic input drifted after preparation.
    FrozenSemanticDrift,
    /// Resource, membership, metric, receipt, or result accounting is corrupt.
    AccountingCorruption,
}

/// Host-selected issue disposition.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamingIssueDisposition {
    /// Retry under the frozen policy.
    Retry,
    /// Pause until bounded capacity becomes available.
    Backpressure,
    /// Exclude one invalid record or session under durable evidence.
    Quarantine,
    /// Advance past one immutable input partition under durable evidence.
    Hole,
    /// Continue a run-scoped diagnostic proven to lose no membership.
    Continue,
    /// Finalize one failed action as truthful terminal membership.
    TerminalActionReceipt,
    /// Preserve the generation while marking one derived export incomplete.
    ExportIncomplete,
    /// Abort only after the private host classifier verifies an invariant.
    FailRun,
}

/// Closed issue scope used by authored threshold matching.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamingIssueScopeKind {
    /// Run-wide diagnostic scope.
    Run,
    /// Immutable partition scope.
    Partition,
    /// Stable record scope.
    Record,
    /// Stable session scope.
    Session,
    /// Stable action scope.
    Action,
    /// Derived exporter scope.
    Export,
    /// Checkpoint-attempt scope.
    CheckpointAttempt,
}

/// Deterministic order facts attached to one issue.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StreamingIssueOrderKey {
    /// Exact input domain for partition, record, and session facts.
    pub input_domain: Option<StreamingInputDomainIdentity>,
    /// Stable source position for input-scoped facts.
    pub source_position: Option<SourcePosition>,
    /// Dense global sequence for action-scoped facts.
    pub global_sequence: Option<GlobalSequence>,
    /// Host-assigned logical retry ordinal.
    pub retry_ordinal: u32,
    /// Stable scope-specific final tie breaker.
    pub scope_tiebreaker: ContentDigest,
}

impl StreamingIssueOrderKey {
    /// Construct run-local diagnostic order.
    #[must_use]
    pub const fn run(retry_ordinal: u32, scope_tiebreaker: ContentDigest) -> Self {
        Self {
            input_domain: None,
            source_position: None,
            global_sequence: None,
            retry_ordinal,
            scope_tiebreaker,
        }
    }

    /// Construct deterministic order within one exact input domain.
    #[must_use]
    pub const fn input(
        input_domain: StreamingInputDomainIdentity,
        source_position: SourcePosition,
        retry_ordinal: u32,
        scope_tiebreaker: ContentDigest,
    ) -> Self {
        Self {
            input_domain: Some(input_domain),
            source_position: Some(source_position),
            global_sequence: None,
            retry_ordinal,
            scope_tiebreaker,
        }
    }

    /// Construct deterministic action order.
    #[must_use]
    pub const fn action(
        global_sequence: GlobalSequence,
        retry_ordinal: u32,
        scope_tiebreaker: ContentDigest,
    ) -> Self {
        Self {
            input_domain: None,
            source_position: None,
            global_sequence: Some(global_sequence),
            retry_ordinal,
            scope_tiebreaker,
        }
    }
}

/// Invalid combination of ordinary issue facts.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StreamingIssueValidationError {
    /// The chosen order key cannot order the chosen scope.
    OrderScopeMismatch,
    /// Controlled stop is host control, not an ordinary adapter issue.
    ControlledStopIsNotOrdinary,
    /// Invariant classification is reserved for the private host classifier.
    InvariantIsHostOwned,
    /// The typed failure does not belong to the selected issue scope.
    FailureScopeMismatch,
    /// A supposedly closed failure exposed an invalid stable component code.
    InvalidComponentId,
}

impl fmt::Display for StreamingIssueValidationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid streaming issue: {self:?}")
    }
}

impl std::error::Error for StreamingIssueValidationError {}

/// Move-only typed ordinary issue submitted to the host reliability owner.
///
/// Live issue authority has no Serde construction path:
///
/// ```compile_fail
/// # use aiperf_runtime::streaming::reliability::OrdinaryStreamingIssue;
/// let _: OrdinaryStreamingIssue = serde_json::from_str("{}").unwrap();
/// ```
#[derive(Debug, Eq, PartialEq)]
pub struct OrdinaryStreamingIssue {
    run: StreamRunIdentity,
    scope: StreamingIssueScope,
    class: StreamingIssueClass,
    stage: StreamingFailureStage,
    code: StreamingIssueComponentId,
    semantic_context_digest: ContentDigest,
    order: StreamingIssueOrderKey,
    failure: OrdinaryStreamingFailure,
}

impl OrdinaryStreamingIssue {
    /// Validate explicit typed issue facts without selecting a disposition.
    ///
    /// Prefer the scope-specific constructors for new call sites. This
    /// constructor remains for the Task 1D reporter-handle contract.
    pub fn new(
        run: StreamRunIdentity,
        scope: StreamingIssueScope,
        class: StreamingIssueClass,
        semantic_context_digest: ContentDigest,
        order: StreamingIssueOrderKey,
        failure: OrdinaryStreamingFailure,
    ) -> Result<Self, StreamingIssueValidationError> {
        if matches!(&failure, OrdinaryStreamingFailure::Source(error) if error.is_stopped()) {
            return Err(StreamingIssueValidationError::ControlledStopIsNotOrdinary);
        }
        if class == StreamingIssueClass::Invariant {
            return Err(StreamingIssueValidationError::InvariantIsHostOwned);
        }
        if !failure_matches_scope(&scope, &failure) {
            return Err(StreamingIssueValidationError::FailureScopeMismatch);
        }
        if !scope_order_matches(&scope, &order) {
            return Err(StreamingIssueValidationError::OrderScopeMismatch);
        }
        let stage = failure.stage();
        let code = StreamingIssueComponentId::new(failure.code())
            .map_err(|_| StreamingIssueValidationError::InvalidComponentId)?;
        Ok(Self {
            run,
            scope,
            class,
            stage,
            code,
            semantic_context_digest,
            order,
            failure,
        })
    }

    /// Construct one run-scoped diagnostic fact.
    #[allow(clippy::too_many_arguments)]
    pub fn run_diagnostic(
        run: StreamRunIdentity,
        class: StreamingIssueClass,
        semantic_context_digest: ContentDigest,
        retry_ordinal: u32,
        scope_tiebreaker: ContentDigest,
        failure: OrdinaryStreamingFailure,
    ) -> Result<Self, StreamingReliabilityError> {
        Self::new(
            run,
            StreamingIssueScope::Run,
            class,
            semantic_context_digest,
            StreamingIssueOrderKey::run(retry_ordinal, scope_tiebreaker),
            failure,
        )
        .map_err(Into::into)
    }

    /// Construct one immutable-partition-scoped fact.
    #[allow(clippy::too_many_arguments)]
    pub fn partition(
        run: StreamRunIdentity,
        input_domain: StreamingInputDomainIdentity,
        object: ImmutableObjectIdentity,
        class: StreamingIssueClass,
        semantic_context_digest: ContentDigest,
        source_position: SourcePosition,
        retry_ordinal: u32,
        scope_tiebreaker: ContentDigest,
        failure: OrdinaryStreamingFailure,
    ) -> Result<Self, StreamingReliabilityError> {
        let order = StreamingIssueOrderKey::input(
            input_domain.clone(),
            source_position,
            retry_ordinal,
            scope_tiebreaker,
        );
        Self::new(
            run,
            StreamingIssueScope::Partition {
                input_domain,
                object,
            },
            class,
            semantic_context_digest,
            order,
            failure,
        )
        .map_err(Into::into)
    }

    /// Construct one stable-record-scoped fact.
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        run: StreamRunIdentity,
        input_domain: StreamingInputDomainIdentity,
        record_id: StableRecordId,
        class: StreamingIssueClass,
        semantic_context_digest: ContentDigest,
        source_position: SourcePosition,
        retry_ordinal: u32,
        scope_tiebreaker: ContentDigest,
        failure: OrdinaryStreamingFailure,
    ) -> Result<Self, StreamingReliabilityError> {
        let order = StreamingIssueOrderKey::input(
            input_domain.clone(),
            source_position,
            retry_ordinal,
            scope_tiebreaker,
        );
        Self::new(
            run,
            StreamingIssueScope::Record {
                input_domain,
                record_id,
            },
            class,
            semantic_context_digest,
            order,
            failure,
        )
        .map_err(Into::into)
    }

    /// Construct one stable-session-scoped fact.
    #[allow(clippy::too_many_arguments)]
    pub fn session(
        run: StreamRunIdentity,
        input_domain: StreamingInputDomainIdentity,
        session_key: StableSessionKey,
        class: StreamingIssueClass,
        semantic_context_digest: ContentDigest,
        source_position: SourcePosition,
        retry_ordinal: u32,
        scope_tiebreaker: ContentDigest,
        failure: OrdinaryStreamingFailure,
    ) -> Result<Self, StreamingReliabilityError> {
        let order = StreamingIssueOrderKey::input(
            input_domain.clone(),
            source_position,
            retry_ordinal,
            scope_tiebreaker,
        );
        Self::new(
            run,
            StreamingIssueScope::Session {
                input_domain,
                session_key,
            },
            class,
            semantic_context_digest,
            order,
            failure,
        )
        .map_err(Into::into)
    }

    /// Construct one stable-action-scoped fact.
    #[allow(clippy::too_many_arguments)]
    pub fn action(
        run: StreamRunIdentity,
        action_id: StableActionId,
        class: StreamingIssueClass,
        semantic_context_digest: ContentDigest,
        global_sequence: GlobalSequence,
        retry_ordinal: u32,
        scope_tiebreaker: ContentDigest,
        failure: OrdinaryStreamingFailure,
    ) -> Result<Self, StreamingReliabilityError> {
        Self::new(
            run,
            StreamingIssueScope::Action { action_id },
            class,
            semantic_context_digest,
            StreamingIssueOrderKey::action(global_sequence, retry_ordinal, scope_tiebreaker),
            failure,
        )
        .map_err(Into::into)
    }

    /// Construct one full-generation exporter-scoped fact.
    #[allow(clippy::too_many_arguments)]
    pub fn export(
        run: StreamRunIdentity,
        exporter_id: StreamingIssueComponentId,
        generation: CheckpointGeneration,
        class: StreamingIssueClass,
        semantic_context_digest: ContentDigest,
        retry_ordinal: u32,
        scope_tiebreaker: ContentDigest,
        failure: OrdinaryStreamingFailure,
    ) -> Result<Self, StreamingReliabilityError> {
        Self::new(
            run,
            StreamingIssueScope::Export {
                exporter_id,
                generation,
            },
            class,
            semantic_context_digest,
            StreamingIssueOrderKey::run(retry_ordinal, scope_tiebreaker),
            failure,
        )
        .map_err(Into::into)
    }

    /// Construct one exact checkpoint-attempt-scoped fact.
    #[allow(clippy::too_many_arguments)]
    pub fn checkpoint_attempt(
        run: StreamRunIdentity,
        generation: CheckpointEpoch,
        attempt_ordinal: u32,
        class: StreamingIssueClass,
        semantic_context_digest: ContentDigest,
        scope_tiebreaker: ContentDigest,
        failure: OrdinaryStreamingFailure,
    ) -> Result<Self, StreamingReliabilityError> {
        Self::new(
            run,
            StreamingIssueScope::CheckpointAttempt {
                generation,
                attempt_ordinal,
            },
            class,
            semantic_context_digest,
            StreamingIssueOrderKey::run(attempt_ordinal, scope_tiebreaker),
            failure,
        )
        .map_err(Into::into)
    }

    /// Borrow the logical run identity.
    #[must_use]
    pub const fn run(&self) -> &StreamRunIdentity {
        &self.run
    }

    /// Borrow the typed issue scope.
    #[must_use]
    pub const fn scope(&self) -> &StreamingIssueScope {
        &self.scope
    }

    /// Return the submitted reliability class.
    #[must_use]
    pub const fn class(&self) -> StreamingIssueClass {
        self.class
    }

    /// Return the stable failure stage.
    #[must_use]
    pub const fn stage(&self) -> StreamingFailureStage {
        self.stage
    }

    /// Borrow the checked stable failure code.
    #[must_use]
    pub const fn code(&self) -> &StreamingIssueComponentId {
        &self.code
    }

    /// Borrow the semantic context digest.
    #[must_use]
    pub const fn semantic_context_digest(&self) -> &ContentDigest {
        &self.semantic_context_digest
    }

    /// Borrow the deterministic order key.
    #[must_use]
    pub const fn order(&self) -> &StreamingIssueOrderKey {
        &self.order
    }

    /// Borrow the original closed typed failure.
    #[must_use]
    pub const fn failure(&self) -> &OrdinaryStreamingFailure {
        &self.failure
    }

    /// Return the deterministic v2 identity of this ordinary fact.
    ///
    /// This fixed-size digest is not a disposition or terminal authority.
    #[must_use]
    pub fn issue_id(&self) -> ContentDigest {
        issue_id(self, None)
    }
}

fn failure_matches_scope(scope: &StreamingIssueScope, failure: &OrdinaryStreamingFailure) -> bool {
    match (scope, failure) {
        (StreamingIssueScope::Run, _) => true,
        (StreamingIssueScope::Partition { .. }, OrdinaryStreamingFailure::Source(_))
        | (StreamingIssueScope::Record { .. }, OrdinaryStreamingFailure::Format(_))
        | (StreamingIssueScope::Session { .. }, OrdinaryStreamingFailure::Session(_))
        | (StreamingIssueScope::Action { .. }, OrdinaryStreamingFailure::Action(_))
        | (
            StreamingIssueScope::CheckpointAttempt { .. },
            OrdinaryStreamingFailure::CheckpointAttempt(_),
        )
        | (StreamingIssueScope::Export { .. }, OrdinaryStreamingFailure::Export(_)) => true,
        (
            StreamingIssueScope::Partition { .. }
            | StreamingIssueScope::Record { .. }
            | StreamingIssueScope::Session { .. }
            | StreamingIssueScope::Action { .. }
            | StreamingIssueScope::Export { .. }
            | StreamingIssueScope::CheckpointAttempt { .. },
            OrdinaryStreamingFailure::Source(_)
            | OrdinaryStreamingFailure::Format(_)
            | OrdinaryStreamingFailure::Session(_)
            | OrdinaryStreamingFailure::Action(_)
            | OrdinaryStreamingFailure::CheckpointAttempt(_)
            | OrdinaryStreamingFailure::Export(_),
        ) => false,
    }
}

fn scope_order_matches(scope: &StreamingIssueScope, order: &StreamingIssueOrderKey) -> bool {
    match scope {
        StreamingIssueScope::Run | StreamingIssueScope::Export { .. } => {
            order.input_domain.is_none()
                && order.source_position.is_none()
                && order.global_sequence.is_none()
        }
        StreamingIssueScope::CheckpointAttempt {
            attempt_ordinal, ..
        } => {
            order.input_domain.is_none()
                && order.source_position.is_none()
                && order.global_sequence.is_none()
                && order.retry_ordinal == *attempt_ordinal
        }
        StreamingIssueScope::Partition { input_domain, .. }
        | StreamingIssueScope::Record { input_domain, .. }
        | StreamingIssueScope::Session { input_domain, .. } => {
            order.input_domain.as_ref() == Some(input_domain)
                && order.source_position.is_some()
                && order.global_sequence.is_none()
        }
        StreamingIssueScope::Action { .. } => {
            order.input_domain.is_none()
                && order.source_position.is_none()
                && order.global_sequence.is_some()
        }
    }
}

/// Frozen threshold rule checked before any source polling.
#[derive(Debug, Eq, PartialEq, Serialize)]
pub struct StreamingIssueThresholdRule {
    rule_id: StreamingIssueComponentId,
    scope: StreamingIssueScopeKind,
    class: StreamingIssueClass,
    code: Option<StreamingIssueComponentId>,
    retry_limit: u32,
    exhausted_disposition: StreamingIssueDisposition,
    admission_fence_count: Option<NonZeroU64>,
}

impl StreamingIssueThresholdRule {
    /// Construct one checked exact-code or wildcard threshold rule.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        rule_id: StreamingIssueComponentId,
        scope: StreamingIssueScopeKind,
        class: StreamingIssueClass,
        code: Option<StreamingIssueComponentId>,
        retry_limit: u32,
        exhausted_disposition: StreamingIssueDisposition,
        admission_fence_count: Option<NonZeroU64>,
    ) -> Result<Self, StreamingReliabilityError> {
        if !is_allowed_authored_disposition(scope, class, exhausted_disposition) {
            return Err(StreamingReliabilityError::IllegalDisposition);
        }
        Ok(Self {
            rule_id,
            scope,
            class,
            code,
            retry_limit,
            exhausted_disposition,
            admission_fence_count,
        })
    }

    /// Borrow the stable rule identity.
    #[must_use]
    pub const fn rule_id(&self) -> &StreamingIssueComponentId {
        &self.rule_id
    }

    /// Return the matched scope kind.
    #[must_use]
    pub const fn scope(&self) -> StreamingIssueScopeKind {
        self.scope
    }

    /// Return the matched reliability class.
    #[must_use]
    pub const fn class(&self) -> StreamingIssueClass {
        self.class
    }

    /// Borrow the exact code, or return `None` for the wildcard.
    #[must_use]
    pub const fn code(&self) -> Option<&StreamingIssueComponentId> {
        self.code.as_ref()
    }

    /// Return the number of matching failures allowed to retry.
    #[must_use]
    pub const fn retry_limit(&self) -> u32 {
        self.retry_limit
    }

    /// Return the checked disposition after retry exhaustion.
    #[must_use]
    pub const fn exhausted_disposition(&self) -> StreamingIssueDisposition {
        self.exhausted_disposition
    }

    /// Return the optional matching-count admission fence.
    #[must_use]
    pub const fn admission_fence_count(&self) -> Option<NonZeroU64> {
        self.admission_fence_count
    }
}

/// Prepared deterministic issue policy with canonical identity.
#[derive(Debug)]
pub struct PreparedStreamingIssuePolicy {
    digest: ContentDigest,
    rules: Box<[StreamingIssueThresholdRule]>,
}

impl PreparedStreamingIssuePolicy {
    /// Validate, canonicalize, and digest one authored rule set.
    pub fn new(
        rules: impl IntoIterator<Item = StreamingIssueThresholdRule>,
    ) -> Result<Self, StreamingReliabilityError> {
        let mut rules: Vec<_> = rules.into_iter().collect();
        let mut rule_ids = BTreeSet::new();
        let mut exact_keys = BTreeSet::new();
        let mut wildcard_keys = BTreeSet::new();

        for rule in &rules {
            if !rule_ids.insert(rule.rule_id.clone()) {
                return Err(StreamingReliabilityError::AmbiguousPolicyRule);
            }
            let scope_class = (rule.scope, rule.class);
            match &rule.code {
                Some(code) => {
                    if !exact_keys.insert((rule.scope, rule.class, code.clone())) {
                        return Err(StreamingReliabilityError::AmbiguousPolicyRule);
                    }
                }
                None => {
                    if !wildcard_keys.insert(scope_class) {
                        return Err(StreamingReliabilityError::AmbiguousPolicyRule);
                    }
                }
            }
        }
        if exact_keys
            .iter()
            .any(|(scope, class, _)| !wildcard_keys.contains(&(*scope, *class)))
        {
            return Err(StreamingReliabilityError::MissingPolicyRule);
        }

        rules.sort_by(canonical_rule_order);
        let digest = policy_digest(&rules);
        Ok(Self {
            digest,
            rules: rules.into_boxed_slice(),
        })
    }

    /// Borrow the canonical frozen policy digest.
    #[must_use]
    pub const fn digest(&self) -> &ContentDigest {
        &self.digest
    }

    /// Return the checked exact-code rule, falling back to the sole wildcard.
    pub fn rule_for(
        &self,
        issue: &OrdinaryStreamingIssue,
    ) -> Result<&StreamingIssueThresholdRule, StreamingReliabilityError> {
        let scope = issue.scope.kind();
        let class = issue.class;
        self.rules
            .iter()
            .find(|rule| {
                rule.scope == scope
                    && rule.class == class
                    && rule.code.as_ref() == Some(&issue.code)
            })
            .or_else(|| {
                self.rules
                    .iter()
                    .find(|rule| rule.scope == scope && rule.class == class && rule.code.is_none())
            })
            .ok_or(StreamingReliabilityError::MissingPolicyRule)
    }
}

fn canonical_rule_order(
    left: &StreamingIssueThresholdRule,
    right: &StreamingIssueThresholdRule,
) -> Ordering {
    left.scope
        .cmp(&right.scope)
        .then_with(|| left.class.cmp(&right.class))
        .then_with(|| match (left.code.is_some(), right.code.is_some()) {
            (true, false) => Ordering::Less,
            (false, true) => Ordering::Greater,
            _ => Ordering::Equal,
        })
        .then_with(|| left.code.cmp(&right.code))
        .then_with(|| left.rule_id.cmp(&right.rule_id))
}

fn is_allowed_authored_disposition(
    scope: StreamingIssueScopeKind,
    class: StreamingIssueClass,
    disposition: StreamingIssueDisposition,
) -> bool {
    match (scope, class, disposition) {
        (_, StreamingIssueClass::Invariant, _) => false,
        (StreamingIssueScopeKind::Run, _, _) => false,
        (
            StreamingIssueScopeKind::Partition,
            StreamingIssueClass::Retryable
            | StreamingIssueClass::Permanent
            | StreamingIssueClass::Capacity,
            StreamingIssueDisposition::Retry
            | StreamingIssueDisposition::Backpressure
            | StreamingIssueDisposition::Hole,
        ) => true,
        (
            StreamingIssueScopeKind::Record | StreamingIssueScopeKind::Session,
            StreamingIssueClass::Retryable
            | StreamingIssueClass::Permanent
            | StreamingIssueClass::Capacity,
            StreamingIssueDisposition::Retry
            | StreamingIssueDisposition::Backpressure
            | StreamingIssueDisposition::Quarantine,
        ) => true,
        (
            StreamingIssueScopeKind::Action,
            StreamingIssueClass::Retryable
            | StreamingIssueClass::Permanent
            | StreamingIssueClass::Capacity,
            StreamingIssueDisposition::Retry
            | StreamingIssueDisposition::Backpressure
            | StreamingIssueDisposition::TerminalActionReceipt,
        ) => true,
        (
            StreamingIssueScopeKind::Export,
            StreamingIssueClass::Retryable
            | StreamingIssueClass::Permanent
            | StreamingIssueClass::Capacity,
            StreamingIssueDisposition::Retry
            | StreamingIssueDisposition::Backpressure
            | StreamingIssueDisposition::ExportIncomplete,
        ) => true,
        (
            StreamingIssueScopeKind::CheckpointAttempt,
            StreamingIssueClass::Retryable
            | StreamingIssueClass::Permanent
            | StreamingIssueClass::Capacity,
            StreamingIssueDisposition::Retry | StreamingIssueDisposition::Backpressure,
        ) => true,
        (
            StreamingIssueScopeKind::Partition
            | StreamingIssueScopeKind::Record
            | StreamingIssueScopeKind::Session
            | StreamingIssueScopeKind::Action
            | StreamingIssueScopeKind::Export
            | StreamingIssueScopeKind::CheckpointAttempt,
            StreamingIssueClass::Retryable
            | StreamingIssueClass::Permanent
            | StreamingIssueClass::Capacity,
            StreamingIssueDisposition::Quarantine
            | StreamingIssueDisposition::Hole
            | StreamingIssueDisposition::Continue
            | StreamingIssueDisposition::TerminalActionReceipt
            | StreamingIssueDisposition::ExportIncomplete
            | StreamingIssueDisposition::FailRun,
        ) => false,
    }
}

/// Private-field live policy decision; only this module can construct it.
///
/// ```compile_fail
/// # use aiperf_runtime::streaming::reliability::{
/// #     StreamingIssueDecision, StreamingIssueDisposition,
/// # };
/// let _ = StreamingIssueDecision {
///     disposition: StreamingIssueDisposition::FailRun,
///     rule: todo!(),
///     needs_admission_fence: false,
/// };
/// ```
#[derive(Debug, Eq, PartialEq)]
pub struct StreamingIssueDecision {
    disposition: StreamingIssueDisposition,
    rule: StreamingIssueThresholdRule,
    needs_admission_fence: bool,
}

/// Deterministic threshold evidence embedded in one persisted receipt.
#[derive(Debug, Eq, PartialEq, Serialize)]
pub struct StreamingIssueThresholdReceipt {
    policy_digest: ContentDigest,
    rule_id: StreamingIssueComponentId,
    prior_matching_count: u64,
    resulting_matching_count: u64,
    retry_ordinal: u32,
    is_exhausted: bool,
}

/// Serialize-only persisted issue receipt, separate from live authority.
///
/// ```compile_fail
/// # use aiperf_runtime::streaming::reliability::PersistedStreamingIssueReceipt;
/// let _: PersistedStreamingIssueReceipt = serde_json::from_str("{}").unwrap();
/// ```
#[derive(Debug, Eq, PartialEq, Serialize)]
pub struct PersistedStreamingIssueReceipt {
    wire_version: u32,
    issue_id: ContentDigest,
    run: StreamRunIdentity,
    scope: StreamingIssueScope,
    class: StreamingIssueClass,
    stage: StreamingFailureStage,
    code: StreamingIssueComponentId,
    semantic_context_digest: ContentDigest,
    order: StreamingIssueOrderKey,
    terminal_invariant: Option<StreamingTerminalInvariant>,
    disposition: StreamingIssueDisposition,
    threshold: StreamingIssueThresholdReceipt,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PersistedStreamingIssueReceiptWire {
    wire_version: u32,
    issue_id: ContentDigest,
    run: StreamRunIdentity,
    scope: StreamingIssueScope,
    class: StreamingIssueClass,
    stage: StreamingFailureStage,
    code: StreamingIssueComponentId,
    semantic_context_digest: ContentDigest,
    order: StreamingIssueOrderKey,
    terminal_invariant: Option<StreamingTerminalInvariant>,
    disposition: StreamingIssueDisposition,
    threshold: StreamingIssueThresholdReceiptWire,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StreamingIssueThresholdReceiptWire {
    policy_digest: ContentDigest,
    rule_id: StreamingIssueComponentId,
    prior_matching_count: u64,
    resulting_matching_count: u64,
    retry_ordinal: u32,
    is_exhausted: bool,
}

impl From<StreamingIssueThresholdReceiptWire> for StreamingIssueThresholdReceipt {
    fn from(wire: StreamingIssueThresholdReceiptWire) -> Self {
        Self {
            policy_digest: wire.policy_digest,
            rule_id: wire.rule_id,
            prior_matching_count: wire.prior_matching_count,
            resulting_matching_count: wire.resulting_matching_count,
            retry_ordinal: wire.retry_ordinal,
            is_exhausted: wire.is_exhausted,
        }
    }
}

impl From<PersistedStreamingIssueReceiptWire> for PersistedStreamingIssueReceipt {
    fn from(wire: PersistedStreamingIssueReceiptWire) -> Self {
        Self {
            wire_version: wire.wire_version,
            issue_id: wire.issue_id,
            run: wire.run,
            scope: wire.scope,
            class: wire.class,
            stage: wire.stage,
            code: wire.code,
            semantic_context_digest: wire.semantic_context_digest,
            order: wire.order,
            terminal_invariant: wire.terminal_invariant,
            disposition: wire.disposition,
            threshold: wire.threshold.into(),
        }
    }
}

const ISSUE_RECEIPT_WIRE_VERSION: u32 = 2;

/// Participant-state schema identity for the host issue ledger.
const ISSUE_LEDGER_STATE_SCHEMA_ID: &str = "aiperf.streaming.issue-ledger";

/// Participant-state schema version for the host issue ledger.
const ISSUE_LEDGER_STATE_WIRE_VERSION: u32 = 1;

#[derive(Serialize)]
#[serde(deny_unknown_fields)]
struct IssueLedgerFrontierWire<'a> {
    domain: &'a StreamingInputDomainIdentity,
    through: SourcePosition,
}

#[derive(Serialize)]
#[serde(deny_unknown_fields)]
struct IssueLedgerCounterWire<'a> {
    key: &'a StreamingIssueCounterKey,
    count: u64,
}

#[derive(Serialize)]
#[serde(deny_unknown_fields)]
struct IssueLedgerStateWire<'a> {
    wire_version: u32,
    run: StreamRunIdentity,
    policy_digest: ContentDigest,
    barrier_epoch: CheckpointEpoch,
    handled_cut: &'a HandledIssueCut,
    action_frontier: Option<GlobalSequence>,
    input_frontiers: Vec<IssueLedgerFrontierWire<'a>>,
    counters: Vec<IssueLedgerCounterWire<'a>>,
    summary: &'a StreamingIssueSummary,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RestoredIssueLedgerFrontier {
    domain: StreamingInputDomainIdentity,
    through: SourcePosition,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RestoredIssueLedgerCounter {
    key: StreamingIssueCounterKey,
    count: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RestoredIssueLedgerState {
    wire_version: u32,
    run: StreamRunIdentity,
    policy_digest: ContentDigest,
    barrier_epoch: CheckpointEpoch,
    handled_cut: HandledIssueCut,
    action_frontier: Option<GlobalSequence>,
    input_frontiers: Vec<RestoredIssueLedgerFrontier>,
    counters: Vec<RestoredIssueLedgerCounter>,
    summary: StreamingIssueSummary,
}

/// The fixed-size facts a retained receipt answers without decoding.
///
/// Every field is `Copy` and inline, so the compact receipt owns exactly one
/// heap allocation: its canonical strict-v2 encoding.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CompactIssueReceiptFacts {
    issue_id: ContentDigest,
    disposition: StreamingIssueDisposition,
    scope_kind: StreamingIssueScopeKind,
    action_id: Option<StableActionId>,
    global_sequence: Option<GlobalSequence>,
}

impl CompactIssueReceiptFacts {
    fn from_receipt(receipt: &PersistedStreamingIssueReceipt) -> Self {
        Self {
            issue_id: receipt.issue_id,
            disposition: receipt.disposition,
            scope_kind: receipt.scope.kind(),
            action_id: receipt.scope.action_id(),
            global_sequence: receipt.order.global_sequence,
        }
    }
}

/// Move-only detailed receipt retaining only its canonical bytes.
///
/// The receipt retains the exact strict-v2 encoding and a fixed-size block of
/// `Copy` facts; the parsed data-transfer object is materialized on demand
/// under its own lease through [`Self::materialize_parsed`] and is never
/// retained. The encoded bytes and their exact charge are inseparable.
pub struct BudgetOwnedStreamingIssueReceipt {
    facts: CompactIssueReceiptFacts,
    encoded: BudgetedCheckpointBytes,
}

impl BudgetOwnedStreamingIssueReceipt {
    /// Borrow the exact strict-v2 wire bytes.
    #[must_use]
    pub fn encoded_bytes(&self) -> &[u8] {
        self.encoded.as_bytes()
    }

    /// Return the exact encoded allocation charge.
    ///
    /// This is the receipt's complete retained heap charge; no further parsed
    /// allocation is held.
    #[must_use]
    pub fn encoded_charge_bytes(&self) -> usize {
        self.encoded.charged_bytes()
    }

    /// Return the deterministic issue identity retained by this receipt.
    #[must_use]
    pub const fn issue_id(&self) -> ContentDigest {
        self.facts.issue_id
    }

    /// Return the checked disposition retained by this receipt.
    #[must_use]
    pub const fn disposition(&self) -> StreamingIssueDisposition {
        self.facts.disposition
    }

    /// Return the checked scope kind retained by this receipt.
    #[must_use]
    pub const fn scope_kind(&self) -> StreamingIssueScopeKind {
        self.facts.scope_kind
    }

    /// Return the retained action identity, when this receipt is action-scoped.
    #[must_use]
    pub const fn action_id(&self) -> Option<StableActionId> {
        self.facts.action_id
    }

    /// Return the retained dense action sequence, when one was ordered.
    #[must_use]
    pub const fn global_sequence(&self) -> Option<GlobalSequence> {
        self.facts.global_sequence
    }

    /// Materialize the parsed receipt on demand under its own exact lease.
    ///
    /// Capacity is acquired for the proven bound
    /// `size_of::<PersistedStreamingIssueReceipt>() + encoded.len()` — every
    /// `String` in the parsed value is a substring of the encoding, so that sum
    /// cannot be exceeded — and is then shrunk synchronously to the exact
    /// retained charge. The decoded identity is revalidated against the
    /// retained compact facts, so a materialized value cannot diverge from the
    /// bytes.
    ///
    /// # Errors
    ///
    /// Returns [`StreamingReliabilityError::StateBudget`] when capacity is
    /// unavailable and [`StreamingReliabilityError::CorruptCheckpointState`]
    /// when the retained bytes do not strictly decode to the retained identity.
    pub fn materialize_parsed(
        &self,
        budget: &StreamingResourceBudget,
    ) -> Result<BudgetOwnedParsedIssueReceipt, StreamingReliabilityError> {
        let bound_bytes = super::budget::checked_sum([
            size_of::<PersistedStreamingIssueReceipt>(),
            self.encoded.as_bytes().len(),
        ])
        .map_err(|_| StreamingReliabilityError::CounterOverflow)?;
        let mut lease = budget
            .try_acquire(1, bound_bytes)
            .map_err(state_budget_error)?;
        // The Serialize-only receipt has no `Deserialize`; its strict wire DTO
        // is the only decode path, so unknown fields are still rejected.
        let receipt: PersistedStreamingIssueReceipt =
            serde_json::from_slice::<PersistedStreamingIssueReceiptWire>(self.encoded.as_bytes())
                .map_err(|_| StreamingReliabilityError::CorruptCheckpointState)?
                .into();
        if receipt.issue_id != self.facts.issue_id
            || receipt.disposition != self.facts.disposition
            || receipt.scope.kind() != self.facts.scope_kind
        {
            return Err(StreamingReliabilityError::CorruptCheckpointState);
        }
        let exact_bytes = parsed_receipt_bytes(&receipt)
            .map_err(|_| StreamingReliabilityError::CounterOverflow)?;
        lease
            .shrink_to(1, exact_bytes)
            .map_err(|_| StreamingReliabilityError::CorruptCheckpointState)?;
        Ok(BudgetOwnedParsedIssueReceipt { receipt, lease })
    }
}

impl fmt::Debug for BudgetOwnedStreamingIssueReceipt {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BudgetOwnedStreamingIssueReceipt")
            .field("issue_id", &self.facts.issue_id)
            .field("encoded_charge_bytes", &self.encoded.charged_bytes())
            .finish_non_exhaustive()
    }
}

/// A parsed receipt materialized on demand, inseparable from its exact lease.
pub struct BudgetOwnedParsedIssueReceipt {
    receipt: PersistedStreamingIssueReceipt,
    lease: BudgetLease,
}

impl BudgetOwnedParsedIssueReceipt {
    /// Borrow the strictly decoded receipt.
    #[must_use]
    pub const fn receipt(&self) -> &PersistedStreamingIssueReceipt {
        &self.receipt
    }

    /// Return the exact parsed allocation charge.
    #[must_use]
    pub fn charged_bytes(&self) -> usize {
        self.lease.charged_bytes()
    }
}

impl fmt::Debug for BudgetOwnedParsedIssueReceipt {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BudgetOwnedParsedIssueReceipt")
            .field("issue_id", &self.receipt.issue_id)
            .field("charged_bytes", &self.lease.charged_bytes())
            .finish_non_exhaustive()
    }
}

/// Roots proving the detailed issues handled by one truthful checkpoint cut.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct HandledIssueCut {
    receipt_root: ContentDigest,
    input_frontier_root: ContentDigest,
    quarantine_tombstone_root: ContentDigest,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HandledIssueCutWire {
    receipt_root: ContentDigest,
    input_frontier_root: ContentDigest,
    quarantine_tombstone_root: ContentDigest,
}

/// Canonical hash domain for the retained session-quarantine tombstone root.
const QUARANTINE_TOMBSTONE_ROOT_DOMAIN: &[u8] = b"aiperf.streaming.quarantine-tombstone-root.v1";

/// Canonical hash domain binding one quarantine receipt to a tombstone view.
const QUARANTINE_RECEIPT_BINDING_DOMAIN: &[u8] =
    b"aiperf.streaming.quarantine-receipt-binding.v1";

impl HandledIssueCut {
    /// Construct the canonical cut containing no handled issues.
    #[must_use]
    pub fn empty() -> Self {
        Self::checked(
            empty_root(b"aiperf.streaming.issue-receipt-root.v1"),
            empty_root(b"aiperf.streaming.issue-input-frontier-root.v1"),
            empty_root(QUARANTINE_TOMBSTONE_ROOT_DOMAIN),
        )
    }

    fn checked(
        receipt_root: ContentDigest,
        input_frontier_root: ContentDigest,
        quarantine_tombstone_root: ContentDigest,
    ) -> Self {
        Self {
            receipt_root,
            input_frontier_root,
            quarantine_tombstone_root,
        }
    }

    /// Borrow the detailed receipt membership root.
    #[must_use]
    pub const fn receipt_root(&self) -> &ContentDigest {
        &self.receipt_root
    }

    /// Borrow the canonical per-input no-more-before frontier root.
    #[must_use]
    pub const fn input_frontier_root(&self) -> &ContentDigest {
        &self.input_frontier_root
    }

    /// Borrow the retained session-quarantine tombstone root.
    #[must_use]
    pub const fn quarantine_tombstone_root(&self) -> &ContentDigest {
        &self.quarantine_tombstone_root
    }
}

impl<'de> Deserialize<'de> for HandledIssueCut {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = HandledIssueCutWire::deserialize(deserializer)?;
        Ok(Self::checked(
            wire.receipt_root,
            wire.input_frontier_root,
            wire.quarantine_tombstone_root,
        ))
    }
}

/// Fold one canonical input-frontier root over a plain domain-to-position map.
///
/// Restore validates the decoded frontier set against the committed cut before
/// any budgeted entry exists, so the fold cannot read the retained map.
fn input_frontier_root_of(
    frontiers: &BTreeMap<StreamingInputDomainIdentity, SourcePosition>,
) -> ContentDigest {
    let mut hasher = blake3::Hasher::new();
    update_hash_field(
        &mut hasher,
        b"aiperf.streaming.issue-input-frontier-root.v1",
    );
    for (domain, through) in frontiers {
        update_hash_field(&mut hasher, domain.stream_identity.as_bytes());
        update_hash_field(&mut hasher, domain.source_identity.as_bytes());
        update_hash_field(&mut hasher, &through.get().to_le_bytes());
    }
    ContentDigest::from_bytes(*hasher.finalize().as_bytes())
}

fn empty_root(domain: &'static [u8]) -> ContentDigest {
    let mut hasher = blake3::Hasher::new();
    update_hash_field(&mut hasher, domain);
    ContentDigest::from_bytes(*hasher.finalize().as_bytes())
}

fn digest_fields(domain: &'static [u8], fields: &[&[u8]]) -> ContentDigest {
    let mut hasher = blake3::Hasher::new();
    update_hash_field(&mut hasher, domain);
    for field in fields {
        update_hash_field(&mut hasher, field);
    }
    ContentDigest::from_bytes(*hasher.finalize().as_bytes())
}

/// Fixed-size non-authoritative outcome returned after host classification.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StreamingIssueOutcome {
    issue_id: ContentDigest,
    disposition: StreamingIssueDisposition,
    needs_admission_fence: bool,
}

impl StreamingIssueOutcome {
    /// Return the deterministic issue identity.
    #[must_use]
    pub const fn issue_id(self) -> ContentDigest {
        self.issue_id
    }

    /// Return the host-selected disposition.
    #[must_use]
    pub const fn disposition(self) -> StreamingIssueDisposition {
        self.disposition
    }

    /// Return whether frozen policy requires an admission fence.
    #[must_use]
    pub const fn needs_admission_fence(self) -> bool {
        self.needs_admission_fence
    }
}

/// Counter domain whose matching threshold state advances deterministically.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(
    tag = "domain",
    content = "identity",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum StreamingIssueCounterDomain {
    /// Run-local diagnostic counters.
    Run,
    /// Counters local to one exact stream/source identity.
    Input(StreamingInputDomainIdentity),
    /// The dense run-local action sequence.
    Action,
    /// One exporter and complete immutable generation.
    Export {
        /// Stable exporter identity.
        exporter_id: StreamingIssueComponentId,
        /// Exact checkpoint generation.
        generation: CheckpointGeneration,
    },
    /// Run-local checkpoint-attempt counters.
    CheckpointAttempt,
}

/// Exact deterministic threshold counter key.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StreamingIssueCounterKey {
    domain: StreamingIssueCounterDomain,
    rule_id: StreamingIssueComponentId,
}

impl StreamingIssueCounterKey {
    /// Borrow the deterministic counter domain.
    #[must_use]
    pub const fn domain(&self) -> &StreamingIssueCounterDomain {
        &self.domain
    }

    /// Borrow the stable rule identity.
    #[must_use]
    pub const fn rule_id(&self) -> &StreamingIssueComponentId {
        &self.rule_id
    }
}

/// Borrowed read-only threshold counter view.
///
/// The view borrows budget-owned counter entries, so its trait implementations
/// are written by hand over the observable counts rather than derived.
pub struct StreamingIssueCounterView<'a> {
    counters: Option<&'a BTreeMap<StreamingIssueCounterKey, RetainedCounter>>,
}

impl<'a> StreamingIssueCounterView<'a> {
    /// Return the matching count for one exact key.
    #[must_use]
    pub fn get(&self, key: &StreamingIssueCounterKey) -> Option<u64> {
        self.counters?.get(key).map(|counter| counter.count)
    }

    /// Iterate over counters in canonical key order.
    pub fn iter(&self) -> impl Iterator<Item = (&'a StreamingIssueCounterKey, u64)> {
        self.counters
            .into_iter()
            .flat_map(|counters| counters.iter().map(|(key, counter)| (key, counter.count)))
    }
}

impl Clone for StreamingIssueCounterView<'_> {
    fn clone(&self) -> Self {
        *self
    }
}

impl Copy for StreamingIssueCounterView<'_> {}

impl std::fmt::Debug for StreamingIssueCounterView<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

impl PartialEq for StreamingIssueCounterView<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.iter().eq(other.iter())
    }
}

impl Eq for StreamingIssueCounterView<'_> {}

/// Fixed-size aggregate issue summary.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StreamingIssueSummary {
    /// Total retained issue observations.
    pub total: u64,
    /// Counts grouped by exact scope.
    pub by_scope: BTreeMap<StreamingIssueScopeKind, u64>,
    /// Counts grouped by reliability class.
    pub by_class: BTreeMap<StreamingIssueClass, u64>,
    /// Counts grouped by host disposition.
    pub by_disposition: BTreeMap<StreamingIssueDisposition, u64>,
    /// Whether issue policy fenced new admission.
    pub is_admission_fenced: bool,
}

impl StreamingIssueSummary {
    fn empty() -> Self {
        Self {
            total: 0,
            by_scope: BTreeMap::new(),
            by_class: BTreeMap::new(),
            by_disposition: BTreeMap::new(),
            is_admission_fenced: false,
        }
    }
}

/// Closed reliability policy and authority failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StreamingReliabilityError {
    /// A stable component identifier is malformed.
    InvalidComponentId,
    /// An issue scope and deterministic order do not match.
    InvalidScopeOrder,
    /// Restored policy identity does not match the frozen policy.
    PolicyDigestMismatch,
    /// A deterministic threshold counter would overflow.
    CounterOverflow,
    /// An authored or computed disposition is illegal for its facts.
    IllegalDisposition,
    /// Ordinary policy attempted to mint fail-run authority.
    IllegalFailRun,
    /// Ordinary facts attempted to mint invariant authority.
    IllegalTerminalInvariant,
    /// Facts name a different logical run.
    ForeignRun,
    /// Bounded reliability state is unavailable.
    StateBudget(StateBudgetFailureCode),
    /// Strict persisted state is corrupt.
    CorruptCheckpointState,
    /// Multiple rules can author the same match.
    AmbiguousPolicyRule,
    /// An exact rule lacks its deterministic wildcard fallback.
    MissingPolicyRule,
    /// Ordered issue evidence is not contiguous.
    NonContiguousIssueFrontier,
    /// Action membership does not match retained terminal evidence.
    InvalidActionTerminalMembership,
    /// Frozen action inventory cannot prove the requested dense prefix.
    IncompleteActionInventory,
    /// Detailed reporter state is not available in this pure policy slice.
    ReliabilityStateUnavailable,
    /// The requested quarantine issue is not a retained quarantine receipt.
    QuarantineReceiptUnavailable,
    /// A prepared tombstone view no longer matches the session owner's view.
    StaleQuarantineTombstoneView,
    /// Tombstone payload or view metadata could not obtain exact capacity.
    QuarantineInstallBudget(StateBudgetFailureCode),
    /// Export receipt names a different logical run.
    ExportReceiptRunMismatch,
    /// Export receipt names a different full checkpoint generation.
    ExportReceiptGenerationMismatch,
    /// Export receipt names a different derived sink.
    ExportReceiptSinkMismatch,
    /// Export receipt names a different dense attempt ordinal.
    ExportReceiptAttemptMismatch,
    /// Export receipt does not bind the frozen policy.
    ExportReceiptPolicyMismatch,
    /// Export receipt or embedded receipt digest/length does not match.
    ExportReceiptDigestLengthMismatch,
    /// Export receipt counter transition is not dense.
    NonContiguousExportCounter,
    /// Durable status does not make this receipt reachable.
    DerivedExportReceiptUnreachable,
    /// Status authority names a generation that is not a committed final generation.
    NonFinalGenerationAuthority,
    /// Export receipt names a stage/code pair no export failure can produce.
    ExportReceiptFailureUnrepresentable,
    /// Export receipt facts cannot reconstruct a legal ordinary issue.
    ExportReceiptClassCodeMismatch,
    /// Export receipt names a rule the frozen policy does not select.
    ExportReceiptRuleMismatch,
    /// Export receipt exhaustion disagrees with the recomputed retry limit.
    ExportReceiptExhaustionMismatch,
    /// Export receipt disposition disagrees with the recomputed policy decision.
    ExportReceiptDispositionMismatch,
    /// Export receipt encoded or parsed state could not obtain exact capacity.
    ExportReceiptBudget(StateBudgetFailureCode),
    /// The current-attempt index names a reporter token with no retained action.
    ///
    /// The ledger refuses rather than panicking so a corrupt or partially
    /// restored index is reported to its owner with every lease intact.
    CorruptActionAttemptIndex,
    /// An undecided retained action failure no longer owns its reserved issue.
    ///
    /// The retained entry is returned to the ledger before this is reported, so
    /// its reporter token stays addressable and no charge is released twice.
    MissingPendingActionIssue,
    /// Different issue content claims one stable ordered submission slot.
    ConflictingIssueSubmission,
    /// An input-scoped fact carries no deterministic source position.
    MissingInputSourcePosition,
    /// The reporter owner has terminally closed its adapter endpoint.
    ReporterClosed,
    /// A restored action frontier crosses sequences no retained gap-closure
    /// proof covers.
    UnprovenActionGapClosure,
    /// A retained or restored gap-closure proof disagrees with the exact
    /// coverage recomputed from retained terminal membership.
    ForgedActionGapClosure,
}

impl fmt::Display for StreamingReliabilityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "streaming reliability error: {self:?}")
    }
}

impl std::error::Error for StreamingReliabilityError {}

impl From<StreamingIssueValidationError> for StreamingReliabilityError {
    fn from(error: StreamingIssueValidationError) -> Self {
        match error {
            StreamingIssueValidationError::OrderScopeMismatch => Self::InvalidScopeOrder,
            StreamingIssueValidationError::ControlledStopIsNotOrdinary => Self::IllegalDisposition,
            StreamingIssueValidationError::InvariantIsHostOwned => Self::IllegalTerminalInvariant,
            StreamingIssueValidationError::FailureScopeMismatch => Self::IllegalDisposition,
            StreamingIssueValidationError::InvalidComponentId => Self::InvalidComponentId,
        }
    }
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CheckedActionSequenceOutcome {
    Succeeded,
    Failed { issue_id: ContentDigest },
}

#[allow(dead_code)]
#[derive(Debug)]
// The frozen inventory is borrowed only while the proof is minted, so the
// reporter cannot reconsult it after a restart. The coverage digest is the
// ledger-recomputable half of the proof: it binds the exact terminal membership
// the reporter held when the inventory accounted for the gap.
struct SealedActionGapClosureProof {
    membership_root: ContentDigest,
    coverage_digest: ContentDigest,
    lease: BudgetLease,
}

/// Opaque checked terminal action fact minted from a sealed borrowed view.
#[allow(dead_code)]
#[derive(Debug)]
pub struct CheckedActionTerminalFact {
    run: StreamRunIdentity,
    action_id: StableActionId,
    sequence: GlobalSequence,
    outcome: CheckedActionSequenceOutcome,
    membership_digest: ContentDigest,
    lease: BudgetLease,
}

/// Move-only token for a failed action retained without frontier advancement.
#[allow(dead_code)]
#[derive(Debug)]
pub struct QueuedActionFailure {
    reporter_token: u64,
}

/// Checked retry branch that carries no terminal identity.
///
/// ```compile_fail
/// # use aiperf_runtime::streaming::reliability::{
/// #     PreparedActionFailureIdentity, PreparedActionRetry,
/// # };
/// # fn cannot_terminalize(retry: PreparedActionRetry) {
/// let _: PreparedActionFailureIdentity = retry.into();
/// # }
/// ```
#[derive(Debug)]
pub struct PreparedActionRetry {
    retry_ordinal: u32,
}

impl PreparedActionRetry {
    /// Return the checked logical retry ordinal.
    #[must_use]
    pub const fn retry_ordinal(&self) -> u32 {
        self.retry_ordinal
    }
}

/// Checked backpressure branch that carries no terminal identity.
///
/// ```compile_fail
/// # use aiperf_runtime::streaming::reliability::{
/// #     PreparedActionBackpressure, PreparedActionFailureIdentity,
/// # };
/// # fn cannot_terminalize(backpressure: PreparedActionBackpressure) {
/// let _: PreparedActionFailureIdentity = backpressure.into();
/// # }
/// ```
#[derive(Debug)]
pub struct PreparedActionBackpressure {
    needs_admission_fence: bool,
}

impl PreparedActionBackpressure {
    /// Return whether frozen policy requires an admission fence.
    #[must_use]
    pub const fn needs_admission_fence(&self) -> bool {
        self.needs_admission_fence
    }
}

/// The only checked state that can authorize failed terminal action membership.
#[derive(Debug)]
pub struct PreparedActionFailureIdentity {
    run: StreamRunIdentity,
    action_id: StableActionId,
    sequence: GlobalSequence,
    issue_id: ContentDigest,
    terminal_evidence_digest: ContentDigest,
}

impl PreparedActionFailureIdentity {
    /// Borrow the logical run owning the prepared failure.
    #[must_use]
    pub const fn run(&self) -> &StreamRunIdentity {
        &self.run
    }

    /// Return the stable action identity.
    #[must_use]
    pub const fn action_id(&self) -> StableActionId {
        self.action_id
    }

    /// Return the dense action sequence.
    #[must_use]
    pub const fn sequence(&self) -> GlobalSequence {
        self.sequence
    }

    /// Return the reporter-retained issue identity.
    #[must_use]
    pub const fn issue_id(&self) -> ContentDigest {
        self.issue_id
    }

    /// Return the checked terminal attempt evidence digest.
    #[must_use]
    pub const fn terminal_evidence_digest(&self) -> ContentDigest {
        self.terminal_evidence_digest
    }
}

/// Exhaustive two-phase action failure disposition.
pub enum ActionFailureDisposition {
    /// Dense predecessor evidence is not complete; poll this same token later.
    Pending(QueuedActionFailure),
    /// Retry under a checked ordinal without terminal authority.
    Retry(PreparedActionRetry),
    /// Apply bounded backpressure without terminal authority.
    Backpressure(PreparedActionBackpressure),
    /// Consume the sole branch that carries failed terminal identity.
    TerminalActionReceipt(PreparedActionFailureIdentity),
}

impl fmt::Debug for ActionFailureDisposition {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pending(_) => formatter.write_str("Pending(..)"),
            Self::Retry(value) => formatter.debug_tuple("Retry").field(value).finish(),
            Self::Backpressure(value) => {
                formatter.debug_tuple("Backpressure").field(value).finish()
            }
            Self::TerminalActionReceipt(value) => formatter
                .debug_tuple("TerminalActionReceipt")
                .field(value)
                .finish(),
        }
    }
}

/// Opaque checked proof that no action before a dense frontier is missing.
#[allow(dead_code)]
#[derive(Debug)]
pub struct CheckedNoMoreActionsBefore {
    through: GlobalSequence,
    proof: SealedActionGapClosureProof,
}

/// Ordered facts accepted by the future budget-owned reporter ledger.
pub enum IssueSequenceUpdate {
    /// One ordinary non-action fact.
    Issue(OrdinaryStreamingIssue),
    /// Input-domain proof that no later issue can precede this position.
    NoMoreBefore {
        /// Exact input domain advanced by the producer.
        input_domain: StreamingInputDomainIdentity,
        /// Greatest source position covered by the proof.
        through: SourcePosition,
    },
    /// Reporter-minted checked action terminal membership.
    CheckedActionTerminal(CheckedActionTerminalFact),
    /// Reporter-minted dense action gap proof.
    CheckedNoMoreActionsBefore(CheckedNoMoreActionsBefore),
    /// Move-only session-quarantine acknowledgement returned for acceptance.
    ///
    /// Acceptance retains the acknowledgement, its exact payload lease, and its
    /// exact view lease, and makes its root visible in the next
    /// [`HandledIssueCut`]. A refused acknowledgement is dropped, releasing both
    /// leases and leaving previously accepted authority unchanged.
    PreparedSessionQuarantineInstall(PreparedSessionQuarantineInstall),
}

/// Fixed-size non-authoritative result of submitting one ordinary issue.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StreamingIssueReportStatus {
    /// The host retained the issue for ordered classification.
    Accepted,
    /// The bounded host input ledger has no current capacity.
    Backpressured,
}

/// Closed reporter-handle submission failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StreamingIssueReportError {
    /// The reporter has closed.
    Closed,
    /// The reporter rejected internally inconsistent typed facts.
    InvalidIssue,
}

impl fmt::Display for StreamingIssueReportError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "streaming issue reporter error: {self:?}")
    }
}

impl std::error::Error for StreamingIssueReportError {}

/// Worker-local host endpoint behind the cloneable Task 1D reporter handle.
#[async_trait(?Send)]
pub trait StreamingIssueReporterEndpoint {
    /// Submit one move-only typed issue and return bounded admission status.
    async fn report(
        &self,
        issue: OrdinaryStreamingIssue,
    ) -> Result<StreamingIssueReportStatus, StreamingIssueReportError>;
}

/// Cloneable opaque injection handle for host-owned reliability reporting.
#[derive(Clone)]
pub struct StreamingIssueReporterHandle {
    inner: Rc<dyn StreamingIssueReporterEndpoint>,
}

impl StreamingIssueReporterHandle {
    /// Erase one worker-local host reporting endpoint.
    #[must_use]
    pub fn new<T>(reporter: T) -> Self
    where
        T: StreamingIssueReporterEndpoint + 'static,
    {
        Self {
            inner: Rc::new(reporter),
        }
    }

    /// Submit one typed issue without granting adapter disposition authority.
    pub async fn report(
        &self,
        issue: OrdinaryStreamingIssue,
    ) -> Result<StreamingIssueReportStatus, StreamingIssueReportError> {
        self.inner.report(issue).await
    }
}

impl fmt::Debug for StreamingIssueReporterHandle {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("StreamingIssueReporterHandle")
            .finish_non_exhaustive()
    }
}

/// Bytes the strict-v2 receipt encoding spends on everything except its three
/// checked component identifiers.
///
/// The encoding is a fixed JSON object: constant field names and punctuation,
/// one `u32` wire version, four `[u8; 32]` digest arrays, the fixed-width order
/// block, two `u64` counters, one `u32` retry ordinal, and the closed set of
/// scope, class, stage, disposition, and terminal-invariant tags. Every
/// contributor is bounded by the type system, so this is a proven maximum
/// rather than an authored cushion, and `receipt_encoding_bound_is_sufficient`
/// fails loudly if a new fixed-width field outgrows it.
const RECEIPT_ENCODING_FIXED_BOUND_BYTES: usize = 1408;

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct PendingInputKey {
    position: SourcePosition,
    scope: StreamingIssueScopeKind,
    tiebreaker: ContentDigest,
    retry_ordinal: u32,
}

struct PendingIssue {
    issue: OrdinaryStreamingIssue,
    reservation: BudgetLease,
    retained_issue_bytes: usize,
}

#[derive(Clone, Copy)]
struct ActionDecisionSnapshot {
    disposition: StreamingIssueDisposition,
    retry_ordinal: u32,
    needs_admission_fence: bool,
    issue_id: ContentDigest,
}

struct PendingActionFailure {
    reporter_token: u64,
    run: StreamRunIdentity,
    action_id: StableActionId,
    sequence: GlobalSequence,
    terminal_evidence_digest: ContentDigest,
    retry_ordinal: u32,
    issue_id: ContentDigest,
    pending: Option<PendingIssue>,
    decision: Option<ActionDecisionSnapshot>,
}

struct RetainedActionTerminal {
    fact: CheckedActionTerminalFact,
    entry_lease: BudgetLease,
}

// The lease is never read; it is retained so an accepted gap closure keeps
// paying for itself for as long as the frontier it advanced is retained.
#[allow(dead_code)]
#[derive(Debug)]
struct RetainedActionGapClosure {
    through: GlobalSequence,
    membership_root: ContentDigest,
    coverage_digest: ContentDigest,
    lease: BudgetLease,
}

/// Strict persisted form of the sole retained action gap-closure proof.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct PersistedActionGapClosure {
    /// Greatest action sequence the frozen inventory accounted for.
    pub(crate) through: GlobalSequence,
    /// Frozen-inventory membership root that authorized the closure.
    pub(crate) membership_root: ContentDigest,
    /// Ledger-recomputable digest of the terminal membership it covered.
    pub(crate) coverage_digest: ContentDigest,
}

const fn larger_charge(left: usize, right: usize) -> usize {
    if left > right { left } else { right }
}

// One constant charge covers both the transient prepared token and the single
// retained closure, so acceptance moves the lease without shrinking, splitting,
// or reacquiring capacity.
const ACTION_GAP_CLOSURE_CHARGE_BYTES: usize = larger_charge(
    size_of::<CheckedNoMoreActionsBefore>(),
    size_of::<RetainedActionGapClosure>(),
);

struct RetainedReceipt {
    receipt: BudgetOwnedStreamingIssueReceipt,
    outcome: StreamingIssueOutcome,
    /// Exact charge for this receipt's ordered-map entry; released on removal.
    entry_lease: BudgetLease,
    /// Monotonic insertion ordinal. A committed generation retires exactly the
    /// receipts whose ordinal is below the barrier's frozen next-ordinal, which
    /// avoids retaining an unbudgeted per-generation identity list.
    ordinal: u64,
}

/// Retained between `checkpoint_view` and `checkpoint_committed`.
///
/// Holding this is what makes retirement conditional: dropping it (a cancelled
/// or superseded barrier) leaves every detailed receipt and retry identity in
/// place.
struct PendingParticipantCommit {
    epoch: CheckpointEpoch,
    represented_cut: CheckpointCut,
    descriptor_digest: ContentDigest,
    receipt_root: ContentDigest,
    handled_cut: HandledIssueCut,
    retire_through_ordinal: u64,
    result_index_root: Option<ContentDigest>,
}

/// One input frontier and the exact lease covering its ordered-map entry.
struct RetainedInputFrontier {
    through: SourcePosition,
    entry_lease: BudgetLease,
}

/// One threshold counter and the exact lease covering its ordered-map entry.
struct RetainedCounter {
    count: u64,
    entry_lease: BudgetLease,
}

/// One outer pending-input domain bucket and the exact lease covering its entry.
struct RetainedDomainPending {
    pending: BTreeMap<PendingInputKey, PendingIssue>,
    entry_lease: BudgetLease,
}

/// One in-flight action attempt token and the lease covering its entry.
struct CurrentActionAttempt {
    reporter_token: u64,
    entry_lease: BudgetLease,
}

struct QueuedHandleIssue {
    pending: PendingIssue,
    /// Requeues already spent on this exact reservation.
    ///
    /// Bounding the retry keeps a deterministic capacity failure from parking a
    /// live lease at the head of the queue forever.
    requeue_attempts: u32,
}

/// Maximum queued host submissions awaiting the next owner drain.
///
/// The ring buffer is allocated once at this exact capacity and charged once,
/// so a host that never drains cannot grow reporter residency. A full queue is
/// reported as backpressure, not as a budget failure.
const MAX_QUEUED_SUBMISSIONS: usize = 256;

/// Bounded retry budget for one requeued reserved submission.
///
/// Only capacity failures requeue, and shared-budget headroom either returns
/// within a few drains or is structurally unavailable. The bound converts that
/// deterministic case from an unbounded loop into one released reservation.
const MAX_SUBMISSION_REQUEUE_ATTEMPTS: u32 = 8;

/// Return the exact byte charge every reporter takes for its submission queue.
///
/// The ring buffer is allocated once at [`MAX_QUEUED_SUBMISSIONS`] and never
/// reallocates, so this is a constant of the build. It is public so budget
/// sizing outside this module can be written against the real charge rather
/// than a copied literal.
#[must_use]
pub fn submission_queue_charge_bytes() -> usize {
    MAX_QUEUED_SUBMISSIONS.saturating_mul(size_of::<QueuedHandleIssue>())
}

struct ReporterSubmissionEndpoint {
    run: StreamRunIdentity,
    budget: StreamingResourceBudget,
    /// Terminal liveness published across the shared `Rc` to every handle.
    ///
    /// Handles outlive the owner, and the owner is the sole drain authority, so
    /// a queued reservation that arrives after the owner is gone can never be
    /// classified. The flag makes that state observable instead of silently
    /// accepting facts forever.
    is_closed: Cell<bool>,
    queue: RefCell<VecDeque<QueuedHandleIssue>>,
    /// Charges the ring buffer allocated once at its constructed capacity.
    ///
    /// The queue never reallocates, so the charge stays exact for the whole
    /// endpoint lifetime. It is held in a `RefCell` because a terminal close
    /// runs behind the shared `&self` endpoint and must return this charge
    /// rather than park it until the last handle clone drops.
    queue_lease: RefCell<Option<BudgetLease>>,
}

/// Exact capacity one terminal reporter close returned to the shared budget.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReporterCloseAccounting {
    /// Charged items the close released.
    pub released_items: usize,
    /// Charged bytes the close released.
    pub released_bytes: usize,
}

impl ReporterSubmissionEndpoint {
    /// Terminally close the endpoint and release every endpoint-owned charge.
    ///
    /// The first call performs the transition; a later call observes an already
    /// closed endpoint, an empty queue, and a spent ring-buffer lease, so
    /// repeated close, explicit close followed by owner drop, and owner drop
    /// alone are all equivalent to one close. The caller-supplied
    /// [`StreamingResourceBudget`] is shared with other participants and is
    /// deliberately never closed here.
    fn close(&self) -> ReporterCloseAccounting {
        let was_open = !self.is_closed.replace(true);
        // The `RefMut` temporaries end with their own statements, so the leases
        // below are released with no outstanding borrow on either field.
        let drained = std::mem::take(&mut *self.queue.borrow_mut());
        let queue_lease = self.queue_lease.borrow_mut().take();
        let mut released_items = 0usize;
        let mut released_bytes = 0usize;
        for queued in &drained {
            released_items =
                released_items.saturating_add(queued.pending.reservation.charged_items());
            released_bytes =
                released_bytes.saturating_add(queued.pending.reservation.charged_bytes());
        }
        if let Some(lease) = &queue_lease {
            released_items = released_items.saturating_add(lease.charged_items());
            released_bytes = released_bytes.saturating_add(lease.charged_bytes());
        }
        // Dropping each queued reservation and the ring-buffer lease returns
        // their exact item and byte charges through `BudgetLease`'s RAII
        // release; there is no manual release path.
        drop(drained);
        drop(queue_lease);
        if was_open {
            tracing::debug!(
                released_items,
                released_bytes,
                component = "streaming_issue_ledger",
                "closed streaming issue reporter endpoint"
            );
        }
        ReporterCloseAccounting {
            released_items,
            released_bytes,
        }
    }
}

#[async_trait(?Send)]
impl StreamingIssueReporterEndpoint for ReporterSubmissionEndpoint {
    async fn report(
        &self,
        issue: OrdinaryStreamingIssue,
    ) -> Result<StreamingIssueReportStatus, StreamingIssueReportError> {
        if self.is_closed.get() {
            return Err(StreamingIssueReportError::Closed);
        }
        if issue.run != self.run || matches!(issue.scope, StreamingIssueScope::Action { .. }) {
            return Err(StreamingIssueReportError::InvalidIssue);
        }
        let pending = match reserve_pending_issue(&self.budget, issue) {
            Ok(pending) => pending,
            Err(StreamingReliabilityError::StateBudget(_)) => {
                return Ok(StreamingIssueReportStatus::Backpressured);
            }
            Err(_) => return Err(StreamingIssueReportError::InvalidIssue),
        };
        let mut queue = self.queue.borrow_mut();
        if self.is_closed.get() {
            // A close observed after the reservation must not park a charge on
            // a queue no owner can drain. Release the borrow first so the lease
            // drop runs with the queue unborrowed.
            drop(queue);
            drop(pending);
            return Err(StreamingIssueReportError::Closed);
        }
        if queue.len() == MAX_QUEUED_SUBMISSIONS {
            // Dropping `pending` releases its reservation, so a refused
            // submission costs the budget nothing.
            return Ok(StreamingIssueReportStatus::Backpressured);
        }
        queue.push_back(QueuedHandleIssue {
            pending,
            requeue_attempts: 0,
        });
        Ok(StreamingIssueReportStatus::Accepted)
    }
}

/// Budget-owned in-memory detailed issue reporter and deterministic ledger.
///
/// The owner is worker-local. Every queued fact and detailed receipt retains
/// an exact item/byte lease, while callers receive only fixed-size outcomes or
/// move-only prepared views.
pub struct BudgetOwnedStreamingIssueReporter {
    run: StreamRunIdentity,
    policy: PreparedStreamingIssuePolicy,
    budget: StreamingResourceBudget,
    submission: Rc<ReporterSubmissionEndpoint>,
    input_frontiers: BTreeMap<StreamingInputDomainIdentity, RetainedInputFrontier>,
    pending_inputs: BTreeMap<StreamingInputDomainIdentity, RetainedDomainPending>,
    pending_actions: BTreeMap<u64, PendingActionFailure>,
    current_action_attempts: BTreeMap<GlobalSequence, CurrentActionAttempt>,
    action_terminals: BTreeMap<GlobalSequence, RetainedActionTerminal>,
    action_frontier: Option<GlobalSequence>,
    action_gap_closure: Option<RetainedActionGapClosure>,
    next_reporter_token: u64,
    receipts: BTreeMap<ContentDigest, RetainedReceipt>,
    next_receipt_ordinal: u64,
    /// Root of every receipt already published to a committed result epoch and
    /// therefore no longer retained in memory. `None` until the first retirement
    /// or restore, which keeps a fresh ledger's cut byte-identical to
    /// [`HandledIssueCut::empty`].
    retired_receipt_root: Option<ContentDigest>,
    pending_commit: Option<PendingParticipantCommit>,
    accepted_quarantine: Option<PreparedSessionQuarantineInstall>,
    counters: BTreeMap<StreamingIssueCounterKey, RetainedCounter>,
    summary: StreamingIssueSummary,
    is_initialized: bool,
}

impl BudgetOwnedStreamingIssueReporter {
    /// Construct one empty reporter under a frozen run, policy, and budget.
    ///
    /// Construction acquires the exact charge for the fixed-capacity submission
    /// ring buffer, so a reporter cannot exist whose queue is unbudgeted.
    ///
    /// # Errors
    ///
    /// Returns [`StreamingReliabilityError::StateBudget`] when the budget
    /// cannot admit the submission queue.
    pub fn new(
        run: StreamRunIdentity,
        policy: PreparedStreamingIssuePolicy,
        budget: StreamingResourceBudget,
    ) -> Result<Self, StreamingReliabilityError> {
        // The deque is allocated first and charged from its own reported
        // capacity, so an over-allocating `with_capacity` cannot be undercharged.
        let queue = VecDeque::with_capacity(MAX_QUEUED_SUBMISSIONS);
        let queue_bytes = super::budget::ring_buffer_bytes::<QueuedHandleIssue>(queue.capacity())
            .map_err(|_| StreamingReliabilityError::CounterOverflow)?;
        let queue_lease = budget
            .try_acquire(1, queue_bytes)
            .map_err(state_budget_error)?;
        let submission = Rc::new(ReporterSubmissionEndpoint {
            run,
            budget: budget.clone(),
            is_closed: Cell::new(false),
            queue: RefCell::new(queue),
            queue_lease: RefCell::new(Some(queue_lease)),
        });
        Ok(Self {
            run,
            policy,
            budget,
            submission,
            input_frontiers: BTreeMap::new(),
            pending_inputs: BTreeMap::new(),
            pending_actions: BTreeMap::new(),
            current_action_attempts: BTreeMap::new(),
            action_terminals: BTreeMap::new(),
            action_frontier: None,
            action_gap_closure: None,
            next_reporter_token: 0,
            receipts: BTreeMap::new(),
            next_receipt_ordinal: 0,
            retired_receipt_root: None,
            pending_commit: None,
            accepted_quarantine: None,
            counters: BTreeMap::new(),
            summary: StreamingIssueSummary::empty(),
            is_initialized: false,
        })
    }

    /// Return the number of retained detailed receipts.
    #[must_use]
    pub fn retained_receipt_count(&self) -> usize {
        self.receipts.len()
    }

    /// Borrow one retained budget-owned receipt by deterministic identity.
    #[must_use]
    pub fn retained_receipt(
        &self,
        issue_id: &ContentDigest,
    ) -> Option<&BudgetOwnedStreamingIssueReceipt> {
        self.receipts.get(issue_id).map(|value| &value.receipt)
    }

    /// Return the retained prior disposition for one deterministic identity.
    ///
    /// The reserved submission path is deferred: a handle cannot classify, so
    /// its exact replay returns the prior state here rather than through
    /// [`StreamingIssueReportStatus`]. The value is byte-identical to the
    /// outcome the first submission of this identity produced, because a replay
    /// never re-enters classification.
    #[must_use]
    pub fn retained_outcome(&self, issue_id: &ContentDigest) -> Option<StreamingIssueOutcome> {
        self.receipts.get(issue_id).map(|value| value.outcome)
    }

    /// Project the sole retained action gap-closure proof for checkpoint wire
    /// encoding. The reporter keeps its lease; only the strict fields leave.
    pub(crate) fn persisted_action_gap_closure(&self) -> Option<PersistedActionGapClosure> {
        self.action_gap_closure
            .as_ref()
            .map(|retained| PersistedActionGapClosure {
                through: retained.through,
                membership_root: retained.membership_root,
                coverage_digest: retained.coverage_digest,
            })
    }

    /// Revalidate a restored proof against restored terminals and frontier,
    /// then install it under a fresh exact lease.
    ///
    /// Every refusal path returns before the single assignment, so a failed
    /// restore leaves the retained closure at its prior value and charges
    /// nothing. The checkpoint restore path must call this after installing
    /// `action_terminals` and `action_frontier` and before marking the
    /// participant initialized.
    pub(crate) fn install_restored_action_gap_closure(
        &mut self,
        persisted: Option<PersistedActionGapClosure>,
    ) -> Result<(), StreamingReliabilityError> {
        let restored = checked_restored_action_gap_closure(
            &self.run,
            &self.budget,
            &self.action_terminals,
            self.action_frontier,
            persisted,
        )?;
        self.action_gap_closure = restored;
        Ok(())
    }

    /// Terminally close the adapter endpoint and release its queued charges.
    ///
    /// Surviving [`StreamingIssueReporterHandle`] clones then observe
    /// [`StreamingIssueReportError::Closed`], and further owner submissions
    /// refuse with [`StreamingReliabilityError::ReporterClosed`]. Retained
    /// receipts, counters, and the summary stay readable. The operation is
    /// idempotent and is performed unconditionally by `Drop`, so a second call
    /// reports no further released capacity.
    ///
    /// This is deliberately inherent rather than a [`StreamingIssueReporter`]
    /// method: the trait must stay object safe for erased injection, and no
    /// other implementation is forced to adopt this lifecycle.
    pub fn close(&mut self) -> ReporterCloseAccounting {
        self.submission.close()
    }

    /// Return whether the adapter endpoint still admits submissions.
    #[must_use]
    pub fn is_open(&self) -> bool {
        !self.submission.is_closed.get()
    }

    fn ensure_open(&self) -> Result<(), StreamingReliabilityError> {
        if self.submission.is_closed.get() {
            return Err(StreamingReliabilityError::ReporterClosed);
        }
        Ok(())
    }

    fn drain_submission_queue(&mut self) -> Result<(), StreamingReliabilityError> {
        self.ensure_open()?;
        loop {
            // The `RefMut` temporary ends with this statement, so nothing below
            // runs while the shared queue is borrowed.
            let queued = self.submission.queue.borrow_mut().pop_front();
            let Some(queued) = queued else {
                return Ok(());
            };
            let issue_id = queued.pending.issue.issue_id();
            match self.resolve_issue_identity(&issue_id) {
                IssueIdentityResolution::RetainedOutcome(outcome) => {
                    // Exact replay of an already-classified identity. The prior
                    // state stands unchanged and is readable through
                    // `retained_outcome`; dropping the duplicate reservation
                    // returns its exact charge through `BudgetLease`.
                    drop(queued);
                    tracing::debug!(
                        issue_id = ?issue_id,
                        disposition = ?outcome.disposition,
                        component = "streaming_issue_ledger",
                        "replayed reserved submission returned the retained disposition"
                    );
                    continue;
                }
                IssueIdentityResolution::AlreadyPending => {
                    drop(queued);
                    tracing::debug!(
                        issue_id = ?issue_id,
                        component = "streaming_issue_ledger",
                        "replayed reserved submission is already retained"
                    );
                    continue;
                }
                IssueIdentityResolution::Novel => {}
            }
            let QueuedHandleIssue {
                pending,
                requeue_attempts,
            } = queued;
            let Err((error, pending)) = self.submit_reserved_issue(pending) else {
                continue;
            };
            if !is_retryable_submission_error(&error) {
                // The failure is a deterministic function of this issue, the
                // frozen policy, and monotone ledger state, so requeueing it
                // could only reproduce it. Drop the reservation, releasing its
                // exact charge, and surface the error exactly once.
                drop(pending);
                tracing::error!(
                    error = %error,
                    issue_id = ?issue_id,
                    component = "streaming_issue_ledger",
                    "dropped a reserved submission that cannot be retried"
                );
                return Err(error);
            }
            let Some(next_attempts) = requeue_attempts
                .checked_add(1)
                .filter(|attempts| *attempts <= MAX_SUBMISSION_REQUEUE_ATTEMPTS)
            else {
                drop(pending);
                tracing::error!(
                    error = %error,
                    issue_id = ?issue_id,
                    requeue_attempts,
                    component = "streaming_issue_ledger",
                    "dropped a reserved submission after exhausting its retry bound"
                );
                return Err(error);
            };
            // The move-only reservation goes back with its lease intact, so a
            // later drain retries against fresher shared-budget headroom.
            self.submission
                .queue
                .borrow_mut()
                .push_front(QueuedHandleIssue {
                    pending,
                    requeue_attempts: next_attempts,
                });
            return Err(error);
        }
    }

    fn submit_issue(
        &mut self,
        issue: OrdinaryStreamingIssue,
    ) -> Result<Option<StreamingIssueOutcome>, StreamingReliabilityError> {
        let issue_id = issue.issue_id();
        match self.resolve_issue_identity(&issue_id) {
            IssueIdentityResolution::RetainedOutcome(outcome) => return Ok(Some(outcome)),
            IssueIdentityResolution::AlreadyPending => return Ok(None),
            IssueIdentityResolution::Novel => {}
        }
        // Reserving only after the identity is proven novel keeps an exact
        // replay from taking a receipt-sized charge it can never consume.
        let pending = reserve_pending_issue(&self.budget, issue)?;
        self.submit_reserved_issue(pending)
            .map_err(|(error, _pending)| error)
    }

    // Returning the move-only pending reservation preserves retry authority
    // without an unbudgeted recovery allocation. Callers must first prove the
    // identity novel through `resolve_issue_identity`; this primitive assumes
    // novelty and mutates counters, the summary, and retained receipts.
    #[allow(clippy::result_large_err)]
    fn submit_reserved_issue(
        &mut self,
        pending: PendingIssue,
    ) -> Result<Option<StreamingIssueOutcome>, (StreamingReliabilityError, PendingIssue)> {
        if pending.issue.run != self.run {
            return Err((StreamingReliabilityError::ForeignRun, pending));
        }
        if matches!(pending.issue.scope, StreamingIssueScope::Action { .. }) {
            return Err((
                StreamingReliabilityError::InvalidActionTerminalMembership,
                pending,
            ));
        }
        let Some(input_domain) = pending.issue.scope.input_domain().cloned() else {
            return self.classify_pending(pending).map(Some);
        };
        // Input-scoped constructors set a source position, but the reserved
        // path accepts facts an adapter minted, so unwind with a typed error
        // and the intact lease rather than aborting the worker.
        let Some(position) = pending.issue.order.source_position else {
            return Err((
                StreamingReliabilityError::MissingInputSourcePosition,
                pending,
            ));
        };
        if self
            .input_frontiers
            .get(&input_domain)
            .is_some_and(|frontier| position <= frontier.through)
        {
            return Err((
                StreamingReliabilityError::NonContiguousIssueFrontier,
                pending,
            ));
        }
        let key = PendingInputKey {
            position,
            scope: pending.issue.scope.kind(),
            tiebreaker: pending.issue.order.scope_tiebreaker,
            retry_ordinal: pending.issue.order.retry_ordinal,
        };
        if let Some(domain) = self.pending_inputs.get_mut(&input_domain) {
            if domain.pending.contains_key(&key) {
                // The caller proved this exact identity novel, so an occupied
                // ordering slot means different content claims one stable slot.
                return Err((
                    StreamingReliabilityError::ConflictingIssueSubmission,
                    pending,
                ));
            }
            domain.pending.insert(key, pending);
            return Ok(None);
        }
        // The outer bucket is a new allocation; its inner entry is already
        // covered by the pending reservation minted in `reserve_pending_issue`.
        let bytes = match super::budget::ordered_map_entry_bytes::<
            StreamingInputDomainIdentity,
            RetainedDomainPending,
        >() {
            Ok(bytes) => bytes,
            Err(_) => return Err((StreamingReliabilityError::CounterOverflow, pending)),
        };
        let entry_lease = match self.budget.try_acquire(1, bytes) {
            Ok(lease) => lease,
            Err(error) => return Err((state_budget_error(error), pending)),
        };
        let mut domain = RetainedDomainPending {
            pending: BTreeMap::new(),
            entry_lease,
        };
        domain.pending.insert(key, pending);
        self.pending_inputs.insert(input_domain, domain);
        Ok(None)
    }

    fn resolve_issue_identity(&self, issue_id: &ContentDigest) -> IssueIdentityResolution {
        if let Some(retained) = self.receipts.get(issue_id) {
            return IssueIdentityResolution::RetainedOutcome(retained.outcome);
        }
        let is_pending = self.pending_inputs.values().any(|domain| {
            domain
                .pending
                .values()
                .any(|candidate| candidate.issue.issue_id() == *issue_id)
        }) || self
            .pending_actions
            .values()
            .any(|candidate| candidate.issue_id == *issue_id);
        if is_pending {
            return IssueIdentityResolution::AlreadyPending;
        }
        IssueIdentityResolution::Novel
    }

    fn enqueue_action_failure(
        &mut self,
        evidence: &dyn CheckedActionFailureTerminalEvidenceView,
        issue: OrdinaryStreamingIssue,
    ) -> Result<QueuedActionFailure, StreamingReliabilityError> {
        if evidence.run() != &self.run || issue.run != self.run {
            return Err(StreamingReliabilityError::ForeignRun);
        }
        let action_id = issue
            .scope
            .action_id()
            .ok_or(StreamingReliabilityError::InvalidActionTerminalMembership)?;
        let sequence = issue
            .order
            .global_sequence
            .ok_or(StreamingReliabilityError::InvalidActionTerminalMembership)?;
        if evidence.action_id() != action_id || evidence.sequence() != sequence {
            return Err(StreamingReliabilityError::InvalidActionTerminalMembership);
        }
        let issue_id = issue.issue_id();
        if let Some(existing) = self.pending_actions.values().find(|existing| {
            existing.action_id == action_id
                && existing.sequence == sequence
                && existing.terminal_evidence_digest == evidence.terminal_evidence_digest()
                && existing.issue_id == issue_id
        }) {
            return Ok(QueuedActionFailure {
                reporter_token: existing.reporter_token,
            });
        }
        if self
            .action_frontier
            .is_some_and(|frontier| sequence <= frontier)
        {
            return Err(StreamingReliabilityError::CorruptCheckpointState);
        }
        if self.pending_actions.values().any(|existing| {
            existing.terminal_evidence_digest == evidence.terminal_evidence_digest()
                && existing.issue_id != issue_id
        }) {
            return Err(StreamingReliabilityError::CorruptCheckpointState);
        }
        if let Some(current_token) = self
            .current_action_attempts
            .get(&sequence)
            .map(|current| current.reporter_token)
        {
            let current = self
                .pending_actions
                .get(&current_token)
                .ok_or(StreamingReliabilityError::CorruptActionAttemptIndex)?;
            if current.action_id != action_id
                || current
                    .decision
                    .is_none_or(|decision| decision.disposition != StreamingIssueDisposition::Retry)
                || current
                    .retry_ordinal
                    .checked_add(1)
                    .is_none_or(|expected| expected != issue.order.retry_ordinal)
            {
                return Err(StreamingReliabilityError::CorruptCheckpointState);
            }
        } else if self
            .pending_actions
            .values()
            .any(|existing| existing.action_id == action_id)
        {
            return Err(StreamingReliabilityError::CorruptCheckpointState);
        }
        // Charge the current-attempt index entry before any state mutation. A
        // retry reuses the existing entry and needs no new charge.
        let attempt_entry_lease = if self.current_action_attempts.contains_key(&sequence) {
            None
        } else {
            let bytes = super::budget::ordered_map_entry_bytes::<GlobalSequence, CurrentActionAttempt>()
                .map_err(|_| StreamingReliabilityError::CounterOverflow)?;
            Some(
                self.budget
                    .try_acquire(1, bytes)
                    .map_err(state_budget_error)?,
            )
        };
        let pending = reserve_pending_issue(&self.budget, issue)?;
        let reporter_token = self.next_reporter_token;
        self.next_reporter_token = self
            .next_reporter_token
            .checked_add(1)
            .ok_or(StreamingReliabilityError::CounterOverflow)?;
        self.pending_actions.insert(
            reporter_token,
            PendingActionFailure {
                reporter_token,
                run: self.run,
                action_id,
                sequence,
                terminal_evidence_digest: evidence.terminal_evidence_digest(),
                retry_ordinal: pending.issue.order.retry_ordinal,
                issue_id,
                pending: Some(pending),
                decision: None,
            },
        );
        match self.current_action_attempts.get_mut(&sequence) {
            Some(current) => current.reporter_token = reporter_token,
            None => {
                let Some(entry_lease) = attempt_entry_lease else {
                    return Err(StreamingReliabilityError::CorruptCheckpointState);
                };
                self.current_action_attempts.insert(
                    sequence,
                    CurrentActionAttempt {
                        reporter_token,
                        entry_lease,
                    },
                );
            }
        }
        Ok(QueuedActionFailure { reporter_token })
    }

    fn poll_action_failure(
        &mut self,
        queued: QueuedActionFailure,
    ) -> Result<ActionFailureDisposition, StreamingReliabilityError> {
        let mut entry = self
            .pending_actions
            .remove(&queued.reporter_token)
            .ok_or(StreamingReliabilityError::InvalidActionTerminalMembership)?;
        let sequence = entry.sequence;
        if let Some(decision) = entry.decision {
            // The entry is reinserted before returning either way: a refused
            // disposition must not strand its token in the current-attempt
            // index or silently release the charge the entry still owns.
            let disposition = action_disposition(&entry, decision);
            self.pending_actions.insert(queued.reporter_token, entry);
            return disposition;
        }
        let is_ready = sequence.get() == 0
            || self
                .action_frontier
                .is_some_and(|frontier| frontier.get() >= sequence.get() - 1);
        if !is_ready {
            self.pending_actions.insert(queued.reporter_token, entry);
            return Ok(ActionFailureDisposition::Pending(queued));
        }
        let Some(pending) = entry.pending.take() else {
            self.pending_actions.insert(queued.reporter_token, entry);
            return Err(StreamingReliabilityError::MissingPendingActionIssue);
        };
        let outcome = match self.classify_pending(pending) {
            Ok(outcome) => outcome,
            Err((error, pending)) => {
                entry.pending = Some(pending);
                self.pending_actions.insert(queued.reporter_token, entry);
                return Err(error);
            }
        };
        let decision = ActionDecisionSnapshot {
            disposition: outcome.disposition,
            retry_ordinal: entry.retry_ordinal,
            needs_admission_fence: outcome.needs_admission_fence,
            issue_id: outcome.issue_id,
        };
        entry.decision = Some(decision);
        let disposition = action_disposition(&entry, decision);
        self.pending_actions.insert(queued.reporter_token, entry);
        disposition
    }

    fn prepare_checked_action_terminal(
        &mut self,
        membership: &dyn CheckedActionTerminalMembershipView,
    ) -> Result<CheckedActionTerminalFact, StreamingReliabilityError> {
        if membership.run() != &self.run {
            return Err(StreamingReliabilityError::ForeignRun);
        }
        let outcome = match membership.outcome() {
            ActionTerminalMembershipOutcomeView::Succeeded => {
                CheckedActionSequenceOutcome::Succeeded
            }
            ActionTerminalMembershipOutcomeView::Failed { issue_id } => {
                let retained = self
                    .receipts
                    .get(&issue_id)
                    .ok_or(StreamingReliabilityError::InvalidActionTerminalMembership)?;
                if retained.outcome.disposition != StreamingIssueDisposition::TerminalActionReceipt
                    || retained.receipt.action_id() != Some(membership.action_id())
                    || retained.receipt.global_sequence() != Some(membership.sequence())
                {
                    return Err(StreamingReliabilityError::InvalidActionTerminalMembership);
                }
                CheckedActionSequenceOutcome::Failed { issue_id }
            }
        };
        let lease = self
            .budget
            .try_acquire(1, size_of::<CheckedActionTerminalFact>())
            .map_err(state_budget_error)?;
        Ok(CheckedActionTerminalFact {
            run: self.run,
            action_id: membership.action_id(),
            sequence: membership.sequence(),
            outcome,
            membership_digest: membership.membership_digest(),
            lease,
        })
    }

    fn retain_action_terminal(
        &mut self,
        fact: CheckedActionTerminalFact,
    ) -> Result<(), StreamingReliabilityError> {
        if fact.run != self.run {
            return Err(StreamingReliabilityError::ForeignRun);
        }
        if let Some(existing) = self.action_terminals.get(&fact.sequence) {
            if existing.fact.action_id != fact.action_id
                || existing.fact.outcome != fact.outcome
                || existing.fact.membership_digest != fact.membership_digest
            {
                return Err(StreamingReliabilityError::InvalidActionTerminalMembership);
            }
            return Ok(());
        }
        if self
            .action_frontier
            .is_some_and(|frontier| fact.sequence <= frontier)
        {
            return Err(StreamingReliabilityError::InvalidActionTerminalMembership);
        }
        if let CheckedActionSequenceOutcome::Failed { issue_id } = fact.outcome {
            let retained = self
                .receipts
                .get(&issue_id)
                .ok_or(StreamingReliabilityError::InvalidActionTerminalMembership)?;
            if retained.outcome.disposition != StreamingIssueDisposition::TerminalActionReceipt {
                return Err(StreamingReliabilityError::InvalidActionTerminalMembership);
            }
        }
        let sequence = fact.sequence;
        let entry_bytes = super::budget::ordered_map_entry_bytes::<
            GlobalSequence,
            RetainedActionTerminal,
        >()
        .map_err(|_| StreamingReliabilityError::CounterOverflow)?;
        let entry_lease = self
            .budget
            .try_acquire(1, entry_bytes)
            .map_err(state_budget_error)?;
        self.action_terminals
            .insert(sequence, RetainedActionTerminal { fact, entry_lease });
        let Some(mut next) = self
            .action_frontier
            .map_or(Some(0), |frontier| frontier.get().checked_add(1))
        else {
            return Ok(());
        };
        while self
            .action_terminals
            .contains_key(&GlobalSequence::new(next))
        {
            self.action_frontier = Some(GlobalSequence::new(next));
            let Some(incremented) = next.checked_add(1) else {
                break;
            };
            next = incremented;
        }
        Ok(())
    }

    fn prepare_action_gap_closure(
        &self,
        inventory: &dyn FrozenActionInventoryView,
        through: GlobalSequence,
    ) -> Result<CheckedNoMoreActionsBefore, StreamingReliabilityError> {
        if inventory.run() != &self.run {
            return Err(StreamingReliabilityError::ForeignRun);
        }
        if through > inventory.through() {
            return Err(StreamingReliabilityError::InvalidActionTerminalMembership);
        }
        for retained in self
            .action_terminals
            .values()
            .filter(|retained| retained.fact.sequence <= through)
        {
            if !inventory.contains_terminal(retained.fact.sequence, retained.fact.membership_digest)
            {
                return Err(StreamingReliabilityError::InvalidActionTerminalMembership);
            }
        }
        for (sequence, attempt) in self.current_action_attempts.range(..=through) {
            let current = self
                .pending_actions
                .get(&attempt.reporter_token)
                .ok_or(StreamingReliabilityError::CorruptActionAttemptIndex)?;
            if current.sequence != *sequence || !self.action_terminals.contains_key(sequence) {
                return Err(StreamingReliabilityError::InvalidActionTerminalMembership);
            }
        }
        let membership_root = inventory.membership_root();
        let coverage_digest =
            action_gap_coverage_digest(&self.run, &self.action_terminals, through, membership_root);
        let lease = self
            .budget
            .try_acquire(1, ACTION_GAP_CLOSURE_CHARGE_BYTES)
            .map_err(state_budget_error)?;
        Ok(CheckedNoMoreActionsBefore {
            through,
            proof: SealedActionGapClosureProof {
                membership_root,
                coverage_digest,
                lease,
            },
        })
    }

    fn retain_action_gap_closure(
        &mut self,
        closure: CheckedNoMoreActionsBefore,
    ) -> Result<(), StreamingReliabilityError> {
        // Refuse before destructuring so a rejected closure returns its capacity
        // with the dropped token instead of charging the ledger.
        if self
            .action_frontier
            .is_some_and(|frontier| closure.through < frontier)
        {
            return Err(StreamingReliabilityError::InvalidActionTerminalMembership);
        }
        let CheckedNoMoreActionsBefore { through, proof } = closure;
        let SealedActionGapClosureProof {
            membership_root,
            coverage_digest,
            lease,
        } = proof;
        if self.action_gap_closure.as_ref().is_some_and(|existing| {
            existing.through == through
                && (existing.membership_root != membership_root
                    || existing.coverage_digest != coverage_digest)
        }) {
            return Err(StreamingReliabilityError::ForgedActionGapClosure);
        }
        // Assigning replaces any superseded closure, whose drop returns its
        // exact charge, so the ledger retains one closure lease at a time.
        self.action_gap_closure = Some(RetainedActionGapClosure {
            through,
            membership_root,
            coverage_digest,
            lease,
        });
        self.action_frontier = Some(through);
        Ok(())
    }

    fn advance_input_frontier(
        &mut self,
        input_domain: StreamingInputDomainIdentity,
        through: SourcePosition,
    ) -> Result<(), StreamingReliabilityError> {
        if self
            .input_frontiers
            .get(&input_domain)
            .is_some_and(|current| through < current.through)
        {
            return Err(StreamingReliabilityError::NonContiguousIssueFrontier);
        }

        // Acquire the frontier entry before any state mutation, so a refusal
        // leaves both maps and every pending lease untouched. An existing
        // frontier already owns its entry lease and needs no new charge.
        let new_entry_lease = if self.input_frontiers.contains_key(&input_domain) {
            None
        } else {
            let bytes = input_frontier_entry_bytes()
                .map_err(|_| StreamingReliabilityError::CounterOverflow)?;
            Some(
                self.budget
                    .try_acquire(1, bytes)
                    .map_err(state_budget_error)?,
            )
        };

        loop {
            // Selection and removal share one borrow, so no second lookup can
            // observe a different map and no absent-key branch exists.
            let Some(domain) = self.pending_inputs.get_mut(&input_domain) else {
                break;
            };
            let Some(first) = domain.pending.first_entry() else {
                break;
            };
            if first.key().position > through {
                break;
            }
            let (next_key, pending) = first.remove_entry();
            if let Err((error, pending)) = self.classify_pending(pending) {
                // Reinsertion cannot allocate a new outer entry: the bucket was
                // present when the key was drawn and is only removed below.
                let Some(domain) = self.pending_inputs.get_mut(&input_domain) else {
                    return Err(StreamingReliabilityError::CorruptCheckpointState);
                };
                domain.pending.insert(next_key, pending);
                return Err(error);
            }
        }

        match self.input_frontiers.get_mut(&input_domain) {
            Some(current) => current.through = through,
            None => {
                let Some(entry_lease) = new_entry_lease else {
                    return Err(StreamingReliabilityError::CorruptCheckpointState);
                };
                self.input_frontiers.insert(
                    input_domain.clone(),
                    RetainedInputFrontier {
                        through,
                        entry_lease,
                    },
                );
            }
        }
        if self
            .pending_inputs
            .get(&input_domain)
            .is_some_and(|domain| domain.pending.is_empty())
        {
            // Removing the bucket drops its entry lease, releasing the charge.
            self.pending_inputs.remove(&input_domain);
        }
        Ok(())
    }

    // Returning the move-only pending reservation preserves retry authority
    // without an unbudgeted recovery allocation.
    #[allow(clippy::result_large_err)]
    fn classify_pending(
        &mut self,
        pending: PendingIssue,
    ) -> Result<StreamingIssueOutcome, (StreamingReliabilityError, PendingIssue)> {
        let rule = match self.policy.rule_for(&pending.issue) {
            Ok(rule) => rule,
            Err(error) => return Err((error, pending)),
        };
        // Checked before the reservation is consumed: after
        // `budget_owned_receipt_from_reservation` the pending issue can no
        // longer be reconstructed, so a late failure would have no move-only
        // retry value to return.
        let Some(next_ordinal) = self.next_receipt_ordinal.checked_add(1) else {
            return Err((StreamingReliabilityError::CounterOverflow, pending));
        };
        let key = counter_key_for_issue(&pending.issue, rule.rule_id.clone());
        let prior_matching_count = self.counters.get(&key).map_or(0, |counter| counter.count);
        // Acquire the counter entry before minting the receipt, so a refusal
        // here cannot leave a receipt retained against an uncharged counter.
        let counter_lease = if self.counters.contains_key(&key) {
            None
        } else {
            let bytes = match counter_entry_bytes(&key) {
                Ok(bytes) => bytes,
                Err(_) => return Err((StreamingReliabilityError::CounterOverflow, pending)),
            };
            match self.budget.try_acquire(1, bytes) {
                Ok(lease) => Some(lease),
                Err(error) => return Err((state_budget_error(error), pending)),
            }
        };
        let resulting_matching_count = match prior_matching_count.checked_add(1) {
            Some(count) => count,
            None => return Err((StreamingReliabilityError::CounterOverflow, pending)),
        };
        let is_exhausted = prior_matching_count >= u64::from(rule.retry_limit);
        let disposition = if is_exhausted {
            rule.exhausted_disposition
        } else {
            StreamingIssueDisposition::Retry
        };
        let needs_admission_fence = rule
            .admission_fence_count
            .is_some_and(|count| resulting_matching_count >= count.get());
        let threshold = StreamingIssueThresholdReceipt {
            policy_digest: self.policy.digest,
            rule_id: rule.rule_id.clone(),
            prior_matching_count,
            resulting_matching_count,
            retry_ordinal: pending.issue.order.retry_ordinal,
            is_exhausted,
        };
        let issue_id = pending.issue.issue_id();
        let receipt = persisted_receipt_from_issue(&pending.issue, disposition, threshold);
        let mut next_summary = self.summary.clone();
        if let Err(error) = update_summary(&mut next_summary, &receipt, needs_admission_fence) {
            return Err((error, pending));
        }
        let (owned, receipt_entry_lease, pending_lease) = match budget_owned_receipt_from_reservation(
            receipt,
            pending.reservation,
            pending.retained_issue_bytes,
        ) {
            Ok(value) => value,
            Err((error, reservation)) => {
                return Err((
                    error,
                    PendingIssue {
                        issue: pending.issue,
                        reservation,
                        retained_issue_bytes: pending.retained_issue_bytes,
                    },
                ));
            }
        };
        let outcome = StreamingIssueOutcome {
            issue_id,
            disposition,
            needs_admission_fence,
        };
        match self.counters.get_mut(&key) {
            Some(counter) => counter.count = resulting_matching_count,
            None => {
                let Some(entry_lease) = counter_lease else {
                    return Err((
                        StreamingReliabilityError::CorruptCheckpointState,
                        PendingIssue {
                            issue: pending.issue,
                            reservation: pending_lease,
                            retained_issue_bytes: pending.retained_issue_bytes,
                        },
                    ));
                };
                self.counters.insert(
                    key,
                    RetainedCounter {
                        count: resulting_matching_count,
                        entry_lease,
                    },
                );
            }
        }
        drop(pending_lease);
        self.summary = next_summary;
        self.receipts.insert(
            issue_id,
            RetainedReceipt {
                receipt: owned,
                outcome,
                entry_lease: receipt_entry_lease,
                ordinal: self.next_receipt_ordinal,
            },
        );
        self.next_receipt_ordinal = next_ordinal;
        Ok(outcome)
    }

    fn receipt_root(&self) -> ContentDigest {
        let mut hasher = blake3::Hasher::new();
        update_hash_field(&mut hasher, b"aiperf.streaming.issue-receipt-root.v1");
        // Chaining only after a retirement or restore keeps a fresh ledger's
        // root byte-identical to `HandledIssueCut::empty()`.
        if let Some(retired) = &self.retired_receipt_root {
            update_hash_field(&mut hasher, retired.as_bytes());
        }
        for (issue_id, retained) in &self.receipts {
            update_hash_field(&mut hasher, issue_id.as_bytes());
            update_hash_field(&mut hasher, retained.receipt.encoded_bytes());
        }
        ContentDigest::from_bytes(*hasher.finalize().as_bytes())
    }

    /// Compute the exact cut this ledger currently represents.
    ///
    /// One owner for all three roots, so the quarantine-tombstone term is
    /// derived in exactly one place and a refused acknowledgement refuses the
    /// whole cut rather than silently downgrading one root.
    fn handled_cut(
        &self,
        barrier: &CheckpointBarrier,
    ) -> Result<HandledIssueCut, StreamingReliabilityError> {
        let receipt_root = self.receipt_root();
        let quarantine_tombstone_root = self.quarantine_tombstone_root(&receipt_root, barrier)?;
        Ok(HandledIssueCut::checked(
            receipt_root,
            self.input_frontier_root(),
            quarantine_tombstone_root,
        ))
    }

    fn input_frontier_root(&self) -> ContentDigest {
        let mut hasher = blake3::Hasher::new();
        update_hash_field(
            &mut hasher,
            b"aiperf.streaming.issue-input-frontier-root.v1",
        );
        for (domain, frontier) in &self.input_frontiers {
            update_hash_field(&mut hasher, domain.stream_identity.as_bytes());
            update_hash_field(&mut hasher, domain.source_identity.as_bytes());
            update_hash_field(&mut hasher, &frontier.through.get().to_le_bytes());
        }
        ContentDigest::from_bytes(*hasher.finalize().as_bytes())
    }

    async fn prepare_receipt_partition_view(
        &mut self,
        barrier: &CheckpointBarrier,
    ) -> Result<PreparedIssueReceiptPartitionView, StreamingReliabilityError> {
        self.drain_submission_queue()?;
        if barrier.run != self.run {
            return Err(StreamingReliabilityError::ForeignRun);
        }
        let receipt_root = self.receipt_root();
        let handled_cut = self.handled_cut(barrier)?;
        let receipt_count = u64::try_from(self.receipts.len())
            .map_err(|_| StreamingReliabilityError::CounterOverflow)?;
        let wire = IssueReceiptPartitionWire {
            wire_version: ISSUE_RECEIPT_WIRE_VERSION,
            run: self.run,
            barrier_epoch: barrier.epoch,
            receipt_root,
            handled_cut: &handled_cut,
            receipts: RetainedReceiptSequence {
                receipts: &self.receipts,
            },
        };
        // Measure before admission: the encoder streams, so the exact payload
        // length is known without materializing a single payload byte.
        let payload_bytes_len =
            measured_json_len(&wire).map_err(|_| StreamingReliabilityError::CorruptCheckpointState)?;
        let view_charge_bytes = size_of::<PreparedIssueReceiptPartitionView>();
        let aggregate_bytes = payload_bytes_len
            .checked_add(view_charge_bytes)
            .ok_or(StreamingReliabilityError::CounterOverflow)?;
        let mut payload_lease = self
            .budget
            .acquire(2, aggregate_bytes)
            .await
            .map_err(state_budget_error)?;
        // Everything below this point is synchronous: the aggregate is split to
        // exact leases with no suspension between admission and subdivision.
        let view_lease = payload_lease
            .split_off(1, view_charge_bytes)
            .map_err(state_budget_error)?;
        let mut payload_buffer =
            LeasedByteBuffer::with_exact_capacity(payload_lease).map_err(state_budget_error)?;
        serde_json::to_writer(&mut payload_buffer, &wire)
            .map_err(|_| StreamingReliabilityError::CorruptCheckpointState)?;
        let (payload_bytes, payload_lease) =
            payload_buffer.into_full().map_err(state_budget_error)?;
        let payload = BudgetedCheckpointBytes::from_compact(payload_bytes, payload_lease)
            .map_err(|_| StreamingReliabilityError::CorruptCheckpointState)?;
        Ok(PreparedIssueReceiptPartitionView {
            run: self.run,
            barrier: barrier.clone(),
            receipt_root,
            handled_cut,
            receipt_count,
            payload,
            view_lease,
        })
    }

    /// Wire the exact ledger state one barrier represents and retain the
    /// pre-CAS retirement authority for it.
    async fn prepare_ledger_participant_state(
        &mut self,
        barrier: &CheckpointBarrier,
    ) -> Result<PreparedParticipantState, CheckpointError> {
        let participant = self.participant_id();
        self.drain_submission_queue()
            .map_err(|error| checkpoint_error_from_reliability(participant.clone(), error))?;
        if barrier.run != self.run {
            return Err(CheckpointError::ObjectVerification);
        }
        let handled_cut = self
            .handled_cut(barrier)
            .map_err(|error| checkpoint_error_from_reliability(participant.clone(), error))?;
        let receipt_root = *handled_cut.receipt_root();
        let wire = IssueLedgerStateWire {
            wire_version: ISSUE_LEDGER_STATE_WIRE_VERSION,
            run: self.run,
            policy_digest: self.policy.digest,
            barrier_epoch: barrier.epoch,
            handled_cut: &handled_cut,
            action_frontier: self.action_frontier,
            input_frontiers: self
                .input_frontiers
                .iter()
                .map(|(domain, frontier)| IssueLedgerFrontierWire {
                    domain,
                    through: frontier.through,
                })
                .collect(),
            counters: self
                .counters
                .iter()
                .map(|(key, counter)| IssueLedgerCounterWire {
                    key,
                    count: counter.count,
                })
                .collect(),
            summary: &self.summary,
        };
        // Measure before admission, then write into a buffer whose capacity is
        // its charge, so the encoding never exists outside the budget.
        let encoded_len =
            measured_json_len(&wire).map_err(|_| CheckpointError::ObjectVerification)?;
        let item_count = self
            .input_frontiers
            .len()
            .checked_add(self.counters.len())
            .and_then(|total| u64::try_from(total).ok())
            .ok_or(CheckpointError::ObjectVerification)?;
        let lease = self
            .budget
            .acquire(1, encoded_len)
            .await
            .map_err(|error| CheckpointError::StateBudget {
                participant: participant.clone(),
                code: budget_failure_code(error),
            })?;
        let mut buffer =
            LeasedByteBuffer::with_exact_capacity(lease).map_err(|error| {
                CheckpointError::StateBudget {
                    participant: participant.clone(),
                    code: budget_failure_code(error),
                }
            })?;
        serde_json::to_writer(&mut buffer, &wire)
            .map_err(|_| CheckpointError::ObjectVerification)?;
        let (bytes, lease) = buffer.into_full().map_err(|error| {
            CheckpointError::StateBudget {
                participant: participant.clone(),
                code: budget_failure_code(error),
            }
        })?;
        let payload = BudgetedCheckpointBytes::from_compact(bytes, lease)?;
        let prepared = PreparedParticipantState::new(
            self.run,
            participant,
            ISSUE_LEDGER_STATE_SCHEMA_ID,
            ISSUE_LEDGER_STATE_WIRE_VERSION,
            barrier.cut.clone(),
            item_count,
            payload,
        )?;
        // Retained last: every fallible step above has already succeeded, so a
        // refused or cancelled view never installs a retirement authority.
        self.pending_commit = Some(PendingParticipantCommit {
            epoch: barrier.epoch,
            represented_cut: barrier.cut.clone(),
            descriptor_digest: prepared.descriptor().digest()?,
            receipt_root,
            handled_cut,
            retire_through_ordinal: self.next_receipt_ordinal,
            result_index_root: None,
        });
        Ok(prepared)
    }

    /// Restore exactly one committed ledger state, mutating nothing until every
    /// borrowed check has passed.
    fn restore_ledger_state(
        &mut self,
        state: &CommittedParticipantState,
    ) -> Result<(), CheckpointError> {
        let participant = self.participant_id();
        if state.run() != &self.run {
            return Err(CheckpointError::ObjectVerification);
        }
        let descriptor = state.descriptor();
        if descriptor.participant_id != participant
            || descriptor.schema_id != ISSUE_LEDGER_STATE_SCHEMA_ID
            || descriptor.schema_version != ISSUE_LEDGER_STATE_WIRE_VERSION
        {
            return Err(CheckpointError::ObjectVerification);
        }
        if !self.receipts.is_empty()
            || !self.counters.is_empty()
            || !self.input_frontiers.is_empty()
            || self.retired_receipt_root.is_some()
        {
            return Err(CheckpointError::ObjectVerification);
        }
        let restored: RestoredIssueLedgerState = serde_json::from_slice(state.payload_bytes())
            .map_err(|_| CheckpointError::ObjectVerification)?;
        if restored.wire_version != ISSUE_LEDGER_STATE_WIRE_VERSION
            || restored.run != self.run
            || restored.policy_digest != self.policy.digest
        {
            return Err(CheckpointError::ObjectVerification);
        }
        let mut input_frontiers = BTreeMap::new();
        for frontier in restored.input_frontiers {
            if input_frontiers
                .insert(frontier.domain, frontier.through)
                .is_some()
            {
                return Err(CheckpointError::ObjectVerification);
            }
        }
        let mut counters = BTreeMap::new();
        for counter in restored.counters {
            if counters.insert(counter.key, counter.count).is_some() {
                return Err(CheckpointError::ObjectVerification);
            }
        }
        if input_frontier_root_of(&input_frontiers) != *restored.handled_cut.input_frontier_root() {
            return Err(CheckpointError::ObjectVerification);
        }
        let item_count = input_frontiers
            .len()
            .checked_add(counters.len())
            .and_then(|total| u64::try_from(total).ok())
            .ok_or(CheckpointError::ObjectVerification)?;
        if item_count != descriptor.item_count {
            return Err(CheckpointError::ObjectVerification);
        }
        // Every restored ordered-map entry carries its exact lease, acquired
        // before any of them is installed, so a refusal here leaves the ledger
        // empty rather than partially charged.
        let mut retained_frontiers = BTreeMap::new();
        for (domain, through) in input_frontiers {
            let bytes = input_frontier_entry_bytes()
                .map_err(|_| CheckpointError::ObjectVerification)?;
            let entry_lease = self.budget.try_acquire(1, bytes).map_err(|error| {
                CheckpointError::StateBudget {
                    participant: participant.clone(),
                    code: budget_failure_code(error),
                }
            })?;
            retained_frontiers.insert(
                domain,
                RetainedInputFrontier {
                    through,
                    entry_lease,
                },
            );
        }
        let mut retained_counters = BTreeMap::new();
        for (key, count) in counters {
            let bytes =
                counter_entry_bytes(&key).map_err(|_| CheckpointError::ObjectVerification)?;
            let entry_lease = self.budget.try_acquire(1, bytes).map_err(|error| {
                CheckpointError::StateBudget {
                    participant: participant.clone(),
                    code: budget_failure_code(error),
                }
            })?;
            retained_counters.insert(key, RetainedCounter { count, entry_lease });
        }
        // First mutation. Everything above is a pure check on borrowed input or
        // a fully released local.
        self.input_frontiers = retained_frontiers;
        self.counters = retained_counters;
        self.summary = restored.summary;
        self.action_frontier = restored.action_frontier;
        self.retired_receipt_root = Some(*restored.handled_cut.receipt_root());
        Ok(())
    }

    /// Retain the pre-CAS binding between the staged epoch and this ledger.
    fn bind_result_epoch(
        &mut self,
        prepared: &PreparedResultEpoch,
    ) -> Result<(), StreamingReliabilityError> {
        let Some(pending) = self.pending_commit.as_mut() else {
            return Err(StreamingReliabilityError::ReliabilityStateUnavailable);
        };
        let Some(binding) = prepared.issue_receipt_binding() else {
            return Err(StreamingReliabilityError::CorruptCheckpointState);
        };
        if binding.receipt_root() != &pending.receipt_root
            || binding.handled_cut() != &pending.handled_cut
            || binding.result_index_root() != prepared.index_root()
        {
            return Err(StreamingReliabilityError::CorruptCheckpointState);
        }
        pending.result_index_root = Some(*prepared.index_root());
        Ok(())
    }

    async fn prepare_quarantine_install(
        &mut self,
        view: &dyn SessionQuarantineTombstoneView,
        issue_id: ContentDigest,
        barrier: &CheckpointBarrier,
        budget: &StreamingResourceBudget,
    ) -> Result<PreparedSessionQuarantineInstall, StreamingReliabilityError> {
        self.drain_submission_queue()?;
        if view.run() != &self.run || barrier.run != self.run {
            return Err(StreamingReliabilityError::ForeignRun);
        }
        let retained = self
            .receipts
            .get(&issue_id)
            .ok_or(StreamingReliabilityError::QuarantineReceiptUnavailable)?;
        if retained.outcome.disposition != StreamingIssueDisposition::Quarantine
            || retained.receipt.scope_kind() != StreamingIssueScopeKind::Session
        {
            return Err(StreamingReliabilityError::QuarantineReceiptUnavailable);
        }
        let entries = view.canonical_encoded_entries();
        let payload_digest = ContentDigest::from_bytes(*blake3::hash(entries).as_bytes());
        if payload_digest != view.tombstone_root() {
            return Err(StreamingReliabilityError::StaleQuarantineTombstoneView);
        }
        let receipt_binding_root = digest_fields(
            QUARANTINE_RECEIPT_BINDING_DOMAIN,
            &[issue_id.as_bytes(), self.receipt_root().as_bytes()],
        );
        let view_charge_bytes = size_of::<PreparedSessionQuarantineInstall>();
        let aggregate_bytes = entries
            .len()
            .checked_add(view_charge_bytes)
            .ok_or(StreamingReliabilityError::CounterOverflow)?;
        let mut payload_lease = budget
            .acquire(2, aggregate_bytes)
            .await
            .map_err(quarantine_install_budget_error)?;
        // Synchronous from here: the aggregate is split to exact leases and the
        // payload is copied into a buffer that cannot exceed its charge.
        let view_lease = payload_lease
            .split_off(1, view_charge_bytes)
            .map_err(quarantine_install_budget_error)?;
        let mut payload_buffer = LeasedByteBuffer::with_exact_capacity(payload_lease)
            .map_err(quarantine_install_budget_error)?;
        payload_buffer
            .write_all(entries)
            .map_err(|_| StreamingReliabilityError::CorruptCheckpointState)?;
        let (payload_bytes, payload_lease) = payload_buffer
            .into_full()
            .map_err(quarantine_install_budget_error)?;
        let payload = BudgetedCheckpointBytes::from_compact(payload_bytes, payload_lease)
            .map_err(|_| StreamingReliabilityError::CorruptCheckpointState)?;
        Ok(PreparedSessionQuarantineInstall {
            barrier: barrier.clone(),
            issue_id,
            tombstone_root: view.tombstone_root(),
            view_revision: view.revision(),
            receipt_binding_root,
            payload_digest,
            payload,
            view_lease,
        })
    }

    fn verify_quarantine_install(
        &self,
        prepared: &PreparedSessionQuarantineInstall,
        current_view: &dyn SessionQuarantineTombstoneView,
        barrier: &CheckpointBarrier,
    ) -> Result<(), StreamingReliabilityError> {
        if current_view.run() != &self.run || barrier.run != self.run {
            return Err(StreamingReliabilityError::ForeignRun);
        }
        self.revalidate_quarantine_binding(
            &prepared.issue_id,
            &prepared.receipt_binding_root,
            &self.receipt_root(),
        )?;
        let current_payload_digest = ContentDigest::from_bytes(
            *blake3::hash(current_view.canonical_encoded_entries()).as_bytes(),
        );
        if prepared.barrier != *barrier
            || prepared.tombstone_root != current_view.tombstone_root()
            || prepared.view_revision != current_view.revision()
            || prepared.payload_digest != current_payload_digest
            || prepared.payload.as_bytes() != current_view.canonical_encoded_entries()
        {
            return Err(StreamingReliabilityError::StaleQuarantineTombstoneView);
        }
        Ok(())
    }

    /// Recheck that a retained quarantine receipt still binds this acknowledgement.
    ///
    /// The binding is only valid while the exact receipt set that produced it is
    /// the receipt set about to be cut, so the acknowledgement and the receipt
    /// root can only ever commit together.
    fn revalidate_quarantine_binding(
        &self,
        issue_id: &ContentDigest,
        receipt_binding_root: &ContentDigest,
        receipt_root: &ContentDigest,
    ) -> Result<(), StreamingReliabilityError> {
        let retained = self
            .receipts
            .get(issue_id)
            .ok_or(StreamingReliabilityError::QuarantineReceiptUnavailable)?;
        if retained.outcome.disposition != StreamingIssueDisposition::Quarantine
            || retained.receipt.scope_kind() != StreamingIssueScopeKind::Session
        {
            return Err(StreamingReliabilityError::QuarantineReceiptUnavailable);
        }
        let expected = digest_fields(
            QUARANTINE_RECEIPT_BINDING_DOMAIN,
            &[issue_id.as_bytes(), receipt_root.as_bytes()],
        );
        if expected != *receipt_binding_root {
            return Err(StreamingReliabilityError::StaleQuarantineTombstoneView);
        }
        Ok(())
    }

    /// Compute the canonical quarantine-tombstone root emitted for one barrier.
    ///
    /// With no accepted acknowledgement this is exactly the canonical empty root
    /// of [`HandledIssueCut::empty`]. With one, the acknowledgement must still be
    /// bound to `receipt_root` and to `barrier`, otherwise the cut is refused
    /// rather than silently downgraded to the empty root.
    fn quarantine_tombstone_root(
        &self,
        receipt_root: &ContentDigest,
        barrier: &CheckpointBarrier,
    ) -> Result<ContentDigest, StreamingReliabilityError> {
        let mut hasher = blake3::Hasher::new();
        update_hash_field(&mut hasher, QUARANTINE_TOMBSTONE_ROOT_DOMAIN);
        if let Some(accepted) = &self.accepted_quarantine {
            if accepted.barrier != *barrier {
                return Err(StreamingReliabilityError::StaleQuarantineTombstoneView);
            }
            self.revalidate_quarantine_binding(
                &accepted.issue_id,
                &accepted.receipt_binding_root,
                receipt_root,
            )?;
            update_hash_field(&mut hasher, accepted.tombstone_root.as_bytes());
            update_hash_field(&mut hasher, &accepted.view_revision.to_le_bytes());
            update_hash_field(&mut hasher, accepted.receipt_binding_root.as_bytes());
            update_hash_field(&mut hasher, accepted.payload_digest.as_bytes());
        }
        Ok(ContentDigest::from_bytes(*hasher.finalize().as_bytes()))
    }

    /// Accept and retain one move-only quarantine acknowledgement.
    ///
    /// Acceptance revalidates reporter-side authority and enforces monotonic
    /// barrier epoch and view revision, so a stale acknowledgement can never
    /// become valid again through a digest replay. A refusal drops the
    /// acknowledgement — releasing its exact payload and view leases — and
    /// leaves previously accepted authority, receipts, and counters unchanged.
    fn accept_quarantine_install(
        &mut self,
        prepared: PreparedSessionQuarantineInstall,
    ) -> Result<(), StreamingReliabilityError> {
        if prepared.barrier.run != self.run {
            return Err(StreamingReliabilityError::ForeignRun);
        }
        let receipt_root = self.receipt_root();
        self.revalidate_quarantine_binding(
            &prepared.issue_id,
            &prepared.receipt_binding_root,
            &receipt_root,
        )?;
        if let Some(accepted) = &self.accepted_quarantine {
            let is_regressed = prepared.barrier.epoch < accepted.barrier.epoch
                || prepared.view_revision < accepted.view_revision;
            let is_inconsistent_replay = prepared.view_revision == accepted.view_revision
                && (prepared.tombstone_root != accepted.tombstone_root
                    || prepared.payload_digest != accepted.payload_digest);
            if is_regressed || is_inconsistent_replay {
                return Err(StreamingReliabilityError::StaleQuarantineTombstoneView);
            }
        }
        self.accepted_quarantine = Some(prepared);
        Ok(())
    }

    async fn prepare_export_failure(
        &mut self,
        run: &StreamRunIdentity,
        generation: &CheckpointGeneration,
        sink_id: &StreamingIssueComponentId,
        attempt_ordinal: u32,
        outcome: ResultSinkAttemptOutcome,
        budget: &StreamingResourceBudget,
    ) -> Result<PreparedExportAttemptFailure, StreamingReliabilityError> {
        self.drain_submission_queue()?;
        if run != &self.run {
            return Err(StreamingReliabilityError::ExportReceiptRunMismatch);
        }
        let ResultSinkAttemptOutcome::Failed(issue) = outcome;
        if issue.run != self.run {
            return Err(StreamingReliabilityError::ExportReceiptRunMismatch);
        }
        let Some(issue_generation) = issue.scope.export_generation() else {
            return Err(StreamingReliabilityError::ExportReceiptGenerationMismatch);
        };
        if issue_generation != generation {
            return Err(StreamingReliabilityError::ExportReceiptGenerationMismatch);
        }
        if issue.scope.exporter_id() != Some(sink_id) {
            return Err(StreamingReliabilityError::ExportReceiptSinkMismatch);
        }
        if issue.order.retry_ordinal != attempt_ordinal {
            return Err(StreamingReliabilityError::ExportReceiptAttemptMismatch);
        }
        let rule = self.policy.rule_for(&issue)?;
        let counter_before = u64::from(attempt_ordinal);
        let counter_after = counter_before
            .checked_add(1)
            .ok_or(StreamingReliabilityError::NonContiguousExportCounter)?;
        let is_exhausted = counter_before >= u64::from(rule.retry_limit);
        let disposition = if is_exhausted {
            rule.exhausted_disposition
        } else {
            StreamingIssueDisposition::Retry
        };
        let threshold = StreamingIssueThresholdReceipt {
            policy_digest: self.policy.digest,
            rule_id: rule.rule_id.clone(),
            prior_matching_count: counter_before,
            resulting_matching_count: counter_after,
            retry_ordinal: attempt_ordinal,
            is_exhausted,
        };
        let embedded_receipt = persisted_receipt_from_issue(&issue, disposition, threshold);
        // The embedded encoding is never retained; only its length and digest
        // are. Streaming into the hasher avoids the buffer entirely.
        let (embedded_encoded_len, embedded_receipt_digest) =
            measured_json_digest(&embedded_receipt)
                .map_err(|_| StreamingReliabilityError::CorruptCheckpointState)?;
        let embedded_receipt_length = u64::try_from(embedded_encoded_len)
            .map_err(|_| StreamingReliabilityError::ExportReceiptDigestLengthMismatch)?;
        let issue_id = embedded_receipt.issue_id;
        let persisted = PersistedExportIssueReceipt {
            wire_version: EXPORT_RECEIPT_WIRE_VERSION,
            run: self.run,
            generation: generation.clone(),
            sink_id: sink_id.clone(),
            attempt_ordinal,
            issue_id,
            policy_digest: self.policy.digest,
            counter_before,
            counter_after,
            embedded_receipt_digest,
            embedded_receipt_length,
            embedded_receipt,
        };
        // Measure before admission: the encoder streams, so the exact encoded
        // length is known without materializing a single payload byte.
        let encoded_len = measured_json_len(&persisted)
            .map_err(|_| StreamingReliabilityError::CorruptCheckpointState)?;
        let receipt_length = u64::try_from(encoded_len)
            .map_err(|_| StreamingReliabilityError::ExportReceiptDigestLengthMismatch)?;
        let parsed_charge_bytes = parsed_export_receipt_bytes(&persisted)
            .map_err(|_| StreamingReliabilityError::CounterOverflow)?;
        let aggregate_bytes = encoded_len
            .checked_add(parsed_charge_bytes)
            .ok_or(StreamingReliabilityError::CounterOverflow)?;
        let mut parsed_lease = budget
            .acquire(2, aggregate_bytes)
            .await
            .map_err(export_budget_error)?;
        // Synchronous from here: the aggregate is split to exact leases and
        // written into a buffer that cannot exceed its charge, so no suspension
        // separates admission from allocation.
        let encoded_lease = parsed_lease
            .split_off(1, encoded_len)
            .map_err(export_budget_error)?;
        let mut encoded_buffer =
            LeasedByteBuffer::with_exact_capacity(encoded_lease).map_err(export_budget_error)?;
        serde_json::to_writer(&mut encoded_buffer, &persisted)
            .map_err(|_| StreamingReliabilityError::CorruptCheckpointState)?;
        let (encoded_bytes, encoded_lease) =
            encoded_buffer.into_full().map_err(export_budget_error)?;
        let receipt_digest = ContentDigest::from_bytes(*blake3::hash(&encoded_bytes).as_bytes());
        let encoded = BudgetedCheckpointBytes::from_compact(encoded_bytes, encoded_lease)
            .map_err(|_| StreamingReliabilityError::CorruptCheckpointState)?;
        Ok(PreparedExportAttemptFailure {
            decision: CheckedExportAttemptDecision {
                issue_id,
                is_exhausted,
            },
            receipt: BudgetOwnedExportIssueReceipt {
                receipt: persisted,
                encoded,
                parsed_lease,
                parsed_charge_bytes,
            },
            reference: DerivedExportReceiptReference {
                receipt_digest,
                receipt_length,
                embedded_receipt_digest,
                embedded_receipt_length,
            },
            attempt_ordinal,
            counter_before,
        })
    }
}

impl Drop for BudgetOwnedStreamingIssueReporter {
    fn drop(&mut self) {
        // Handle clones outlive the owner through the shared `Rc` endpoint, so
        // the endpoint's terminal state is published here. Unlike
        // `blocking.rs`'s `ExecutorInner`, which constructs and exclusively owns
        // its budgets, this reporter's `StreamingResourceBudget` is
        // caller-supplied and shared with other participants; closing it would
        // starve them. Only reporter-owned charges are released: the queued
        // reservations and the ring-buffer lease by `close`, and every retained
        // pending issue, action reservation, and receipt by the ordinary field
        // drop that follows.
        let _ = self.submission.close();
    }
}

/// Outcome of resolving one deterministic issue identity against the ledger.
///
/// Both the direct and the reserved submission paths resolve through this so an
/// exact replay observes the prior state and never re-enters classification.
enum IssueIdentityResolution {
    /// The identity is already classified; this is its retained prior outcome.
    RetainedOutcome(StreamingIssueOutcome),
    /// The identity is retained for ordered classification but not decided.
    AlreadyPending,
    /// The identity is unknown to the ledger.
    Novel,
}

/// Return whether one ordered-submission failure can succeed on a later drain.
///
/// Only the shared-budget capacity families are transient: their headroom is
/// released by other participants over time. Every other variant is a
/// deterministic function of the issue, the frozen policy, and monotone ledger
/// state, so requeueing it would reproduce it forever. The match is exhaustive
/// with no wildcard so a future variant must be classified deliberately.
fn is_retryable_submission_error(error: &StreamingReliabilityError) -> bool {
    match error {
        StreamingReliabilityError::StateBudget(_)
        | StreamingReliabilityError::QuarantineInstallBudget(_)
        | StreamingReliabilityError::ExportReceiptBudget(_) => true,
        StreamingReliabilityError::InvalidComponentId
        | StreamingReliabilityError::InvalidScopeOrder
        | StreamingReliabilityError::PolicyDigestMismatch
        | StreamingReliabilityError::CounterOverflow
        | StreamingReliabilityError::IllegalDisposition
        | StreamingReliabilityError::IllegalFailRun
        | StreamingReliabilityError::IllegalTerminalInvariant
        | StreamingReliabilityError::ForeignRun
        | StreamingReliabilityError::CorruptCheckpointState
        | StreamingReliabilityError::AmbiguousPolicyRule
        | StreamingReliabilityError::MissingPolicyRule
        | StreamingReliabilityError::NonContiguousIssueFrontier
        | StreamingReliabilityError::InvalidActionTerminalMembership
        | StreamingReliabilityError::IncompleteActionInventory
        | StreamingReliabilityError::ReliabilityStateUnavailable
        | StreamingReliabilityError::QuarantineReceiptUnavailable
        | StreamingReliabilityError::StaleQuarantineTombstoneView
        | StreamingReliabilityError::ExportReceiptRunMismatch
        | StreamingReliabilityError::ExportReceiptGenerationMismatch
        | StreamingReliabilityError::ExportReceiptSinkMismatch
        | StreamingReliabilityError::ExportReceiptAttemptMismatch
        | StreamingReliabilityError::ExportReceiptPolicyMismatch
        | StreamingReliabilityError::ExportReceiptDigestLengthMismatch
        | StreamingReliabilityError::NonContiguousExportCounter
        | StreamingReliabilityError::DerivedExportReceiptUnreachable
        | StreamingReliabilityError::NonFinalGenerationAuthority
        | StreamingReliabilityError::ExportReceiptFailureUnrepresentable
        | StreamingReliabilityError::ExportReceiptClassCodeMismatch
        | StreamingReliabilityError::ExportReceiptRuleMismatch
        | StreamingReliabilityError::ExportReceiptExhaustionMismatch
        | StreamingReliabilityError::ExportReceiptDispositionMismatch
        | StreamingReliabilityError::CorruptActionAttemptIndex
        | StreamingReliabilityError::MissingPendingActionIssue
        | StreamingReliabilityError::ConflictingIssueSubmission
        | StreamingReliabilityError::MissingInputSourcePosition
        | StreamingReliabilityError::ReporterClosed
        | StreamingReliabilityError::UnprovenActionGapClosure
        | StreamingReliabilityError::ForgedActionGapClosure => false,
    }
}


#[derive(Serialize)]
#[serde(deny_unknown_fields)]
struct IssueReceiptPartitionWire<'a> {
    wire_version: u32,
    run: StreamRunIdentity,
    barrier_epoch: CheckpointEpoch,
    receipt_root: ContentDigest,
    handled_cut: &'a HandledIssueCut,
    /// Each retained receipt is embedded verbatim from its canonical encoding,
    /// so the partition payload never re-serializes a materialized DTO.
    receipts: RetainedReceiptSequence<'a>,
}

/// Borrowed retained-receipt map serialized as its verbatim receipt sequence.
///
/// Encoding directly from the map produces the byte-identical JSON array a
/// materialized `Vec<Box<RawValue>>` produced, in the same `BTreeMap` key order,
/// without the intermediate vector and its per-receipt `String` copy — neither
/// of which the budget ever charged.
struct RetainedReceiptSequence<'a> {
    receipts: &'a BTreeMap<ContentDigest, RetainedReceipt>,
}

impl Serialize for RetainedReceiptSequence<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut sequence = serializer.serialize_seq(Some(self.receipts.len()))?;
        for retained in self.receipts.values() {
            // The retained bytes are the canonical encoding, so they are
            // borrowed as raw JSON rather than re-encoded from a parsed DTO.
            let raw: &RawValue = serde_json::from_slice(retained.receipt.encoded_bytes())
                .map_err(serde::ser::Error::custom)?;
            sequence.serialize_element(raw)?;
        }
        sequence.end()
    }
}

/// Non-destructive move-only detailed-receipt partition prepared at a barrier.
pub struct PreparedIssueReceiptPartitionView {
    run: StreamRunIdentity,
    barrier: CheckpointBarrier,
    receipt_root: ContentDigest,
    handled_cut: HandledIssueCut,
    receipt_count: u64,
    payload: BudgetedCheckpointBytes,
    view_lease: BudgetLease,
}

impl PreparedIssueReceiptPartitionView {
    /// Borrow the logical run owning this prepared view.
    #[must_use]
    pub const fn run(&self) -> &StreamRunIdentity {
        &self.run
    }

    /// Borrow the exact checkpoint barrier bound into the view.
    #[must_use]
    pub const fn barrier(&self) -> &CheckpointBarrier {
        &self.barrier
    }

    /// Borrow the detailed receipt membership root.
    #[must_use]
    pub const fn receipt_root(&self) -> &ContentDigest {
        &self.receipt_root
    }

    /// Borrow the handled-issue cut represented by the view.
    #[must_use]
    pub const fn handled_cut(&self) -> &HandledIssueCut {
        &self.handled_cut
    }

    /// Borrow the exact immutable partition payload.
    #[must_use]
    pub fn payload_bytes(&self) -> &[u8] {
        self.payload.as_bytes()
    }

    /// Return the exact payload byte charge.
    #[must_use]
    pub fn payload_charge_bytes(&self) -> usize {
        self.payload.charged_bytes()
    }

    /// Return the exact view-metadata byte charge.
    #[must_use]
    pub fn view_charge_bytes(&self) -> usize {
        self.view_lease.charged_bytes()
    }

    /// Consume the view into the reserved immutable result-partition handoff.
    pub fn into_result_partition(
        self,
        descriptor: BudgetedResultDescriptor,
    ) -> Result<PreparedIssueReceiptResultPartition, StreamingReliabilityError> {
        let fields = descriptor.descriptor();
        let payload_length = u64::try_from(self.payload.as_bytes().len())
            .map_err(|_| StreamingReliabilityError::CorruptCheckpointState)?;
        let payload_digest =
            ContentDigest::from_bytes(*blake3::hash(self.payload.as_bytes()).as_bytes());
        if fields.run != self.run
            || fields.epoch != self.barrier.epoch
            || fields.projection.as_str() != "streaming_issue_receipts"
            || fields.schema.get() != ISSUE_RECEIPT_WIRE_VERSION
            || fields.first_sequence != GlobalSequence::new(0)
            || fields.last_sequence != GlobalSequence::new(0)
            || fields.item_count != self.receipt_count
            || fields.byte_length != payload_length
            || fields.membership_root != self.receipt_root
            || fields.payload_digest != payload_digest
        {
            return Err(StreamingReliabilityError::CorruptCheckpointState);
        }
        let partition = ResultPartition::new(descriptor, self.payload)
            .map_err(|_| StreamingReliabilityError::CorruptCheckpointState)?;
        Ok(PreparedIssueReceiptResultPartition {
            partition,
            run: self.run,
            barrier_epoch: self.barrier.epoch,
            receipt_root: self.receipt_root,
            handled_cut: self.handled_cut,
            view_lease: self.view_lease,
        })
    }
}

/// Move-only result handoff retaining payload, descriptor, and view authority.
pub struct PreparedIssueReceiptResultPartition {
    partition: ResultPartition,
    run: StreamRunIdentity,
    barrier_epoch: CheckpointEpoch,
    receipt_root: ContentDigest,
    handled_cut: HandledIssueCut,
    view_lease: BudgetLease,
}

impl PreparedIssueReceiptResultPartition {
    /// Borrow the intact result partition.
    #[must_use]
    pub const fn partition(&self) -> &ResultPartition {
        &self.partition
    }

    /// Borrow the logical run binding.
    #[must_use]
    pub const fn run(&self) -> &StreamRunIdentity {
        &self.run
    }

    /// Return the exact barrier epoch binding.
    #[must_use]
    pub const fn barrier_epoch(&self) -> CheckpointEpoch {
        self.barrier_epoch
    }

    /// Borrow the exact receipt membership root.
    #[must_use]
    pub const fn receipt_root(&self) -> &ContentDigest {
        &self.receipt_root
    }

    /// Borrow the exact handled-issue cut.
    #[must_use]
    pub const fn handled_cut(&self) -> &HandledIssueCut {
        &self.handled_cut
    }

    /// Return the exact retained view-metadata charge.
    #[must_use]
    pub fn view_charge_bytes(&self) -> usize {
        self.view_lease.charged_bytes()
    }

    #[allow(dead_code)]
    pub(crate) fn into_parts(self) -> (ResultPartition, HandledIssueCut, BudgetLease) {
        (self.partition, self.handled_cut, self.view_lease)
    }

    /// Consume the handoff into the payload the backend stages plus the binding
    /// the prepared epoch carries back to the reporter.
    ///
    /// The view lease travels with the binding, so a dropped prepared epoch
    /// releases it exactly once.
    pub(crate) fn into_staged_parts(
        self,
        result_index_root: ContentDigest,
    ) -> (ResultPartition, PreparedIssueReceiptEpochBinding) {
        (
            self.partition,
            PreparedIssueReceiptEpochBinding {
                receipt_root: self.receipt_root,
                handled_cut: self.handled_cut,
                result_index_root,
                view_lease: self.view_lease,
            },
        )
    }
}

/// Pre-CAS binding of one staged receipt partition to its result-index root.
///
/// The binding retains the view lease, so the exact charge survives exactly as
/// long as the prepared epoch that carries it.
///
/// ```compile_fail
/// # use aiperf_runtime::streaming::reliability::PreparedIssueReceiptEpochBinding;
/// # fn cannot_separate(value: PreparedIssueReceiptEpochBinding) {
/// let _lease = value.view_lease;
/// # }
/// ```
#[derive(Debug)]
pub struct PreparedIssueReceiptEpochBinding {
    receipt_root: ContentDigest,
    handled_cut: HandledIssueCut,
    result_index_root: ContentDigest,
    view_lease: BudgetLease,
}

impl PreparedIssueReceiptEpochBinding {
    /// Borrow the exact detailed-receipt membership root.
    #[must_use]
    pub const fn receipt_root(&self) -> &ContentDigest {
        &self.receipt_root
    }

    /// Borrow the exact handled-issue cut staged into this epoch.
    #[must_use]
    pub const fn handled_cut(&self) -> &HandledIssueCut {
        &self.handled_cut
    }

    /// Borrow the exact canonical result-index root of the staged epoch.
    #[must_use]
    pub const fn result_index_root(&self) -> &ContentDigest {
        &self.result_index_root
    }

    /// Return the exact retained view-metadata charge.
    #[must_use]
    pub fn view_charge_bytes(&self) -> usize {
        self.view_lease.charged_bytes()
    }
}

/// Move-only acknowledgement of one non-destructive session tombstone view.
pub struct PreparedSessionQuarantineInstall {
    barrier: CheckpointBarrier,
    issue_id: ContentDigest,
    tombstone_root: ContentDigest,
    view_revision: u64,
    receipt_binding_root: ContentDigest,
    payload_digest: ContentDigest,
    payload: BudgetedCheckpointBytes,
    view_lease: BudgetLease,
}

impl PreparedSessionQuarantineInstall {
    /// Borrow the exact barrier bound into this acknowledgement.
    #[must_use]
    pub const fn barrier(&self) -> &CheckpointBarrier {
        &self.barrier
    }

    /// Borrow the retained tombstone-map root.
    #[must_use]
    pub const fn tombstone_root(&self) -> &ContentDigest {
        &self.tombstone_root
    }

    /// Return the monotonic session-owner view revision.
    #[must_use]
    pub const fn view_revision(&self) -> u64 {
        self.view_revision
    }

    /// Borrow the root binding the issue receipt to this tombstone view.
    #[must_use]
    pub const fn receipt_binding_root(&self) -> &ContentDigest {
        &self.receipt_binding_root
    }

    /// Borrow the exact compact tombstone payload.
    #[must_use]
    pub fn payload_bytes(&self) -> &[u8] {
        self.payload.as_bytes()
    }

    /// Return the exact payload allocation charge.
    #[must_use]
    pub fn payload_charge_bytes(&self) -> usize {
        self.payload.charged_bytes()
    }

    /// Return the exact acknowledgement-view allocation charge.
    #[must_use]
    pub fn view_charge_bytes(&self) -> usize {
        self.view_lease.charged_bytes()
    }
}

/// Closed result-sink attempt fact submitted to the reliability owner.
pub enum ResultSinkAttemptOutcome {
    /// One typed ordinary export-scoped failure.
    Failed(OrdinaryStreamingIssue),
}

/// Serialize-only durable export issue receipt.
#[derive(Debug, Eq, PartialEq, Serialize)]
pub struct PersistedExportIssueReceipt {
    wire_version: u32,
    run: StreamRunIdentity,
    generation: CheckpointGeneration,
    sink_id: StreamingIssueComponentId,
    attempt_ordinal: u32,
    issue_id: ContentDigest,
    policy_digest: ContentDigest,
    counter_before: u64,
    counter_after: u64,
    embedded_receipt_digest: ContentDigest,
    embedded_receipt_length: u64,
    embedded_receipt: PersistedStreamingIssueReceipt,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PersistedExportIssueReceiptWire {
    wire_version: u32,
    run: StreamRunIdentity,
    generation: CheckpointGeneration,
    sink_id: StreamingIssueComponentId,
    attempt_ordinal: u32,
    issue_id: ContentDigest,
    policy_digest: ContentDigest,
    counter_before: u64,
    counter_after: u64,
    embedded_receipt_digest: ContentDigest,
    embedded_receipt_length: u64,
    embedded_receipt: PersistedStreamingIssueReceiptWire,
}

const EXPORT_RECEIPT_WIRE_VERSION: u32 = 1;

/// Move-only durable export receipt retaining separate exact leases.
pub struct BudgetOwnedExportIssueReceipt {
    receipt: PersistedExportIssueReceipt,
    encoded: BudgetedCheckpointBytes,
    parsed_lease: BudgetLease,
    parsed_charge_bytes: usize,
}

impl BudgetOwnedExportIssueReceipt {
    /// Return the deterministic embedded issue identity.
    #[must_use]
    pub const fn issue_id(&self) -> ContentDigest {
        self.receipt.issue_id
    }

    /// Return the exact encoded allocation charge.
    #[must_use]
    pub fn encoded_charge_bytes(&self) -> usize {
        self.encoded.charged_bytes()
    }

    /// Return the exact parsed allocation charge.
    #[must_use]
    pub fn parsed_charge_bytes(&self) -> usize {
        debug_assert_eq!(self.parsed_charge_bytes, self.parsed_lease.charged_bytes());
        self.parsed_lease.charged_bytes()
    }
}

/// Durable status reference to an outer and embedded receipt object.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct DerivedExportReceiptReference {
    receipt_digest: ContentDigest,
    receipt_length: u64,
    embedded_receipt_digest: ContentDigest,
    embedded_receipt_length: u64,
}

impl DerivedExportReceiptReference {
    /// Rebuild the reference from the four durably persisted status fields.
    ///
    /// This is the sole reconstruction seam for a post-restart status owner. It is
    /// private to `reliability`: outside this module the only way to obtain a
    /// reference remains an in-process prepared failure.
    const fn from_status_fields(
        receipt_digest: ContentDigest,
        receipt_length: u64,
        embedded_receipt_digest: ContentDigest,
        embedded_receipt_length: u64,
    ) -> Self {
        Self {
            receipt_digest,
            receipt_length,
            embedded_receipt_digest,
            embedded_receipt_length,
        }
    }

    /// Borrow the raw BLAKE3 digest of the complete export receipt.
    #[must_use]
    pub const fn receipt_digest(&self) -> &ContentDigest {
        &self.receipt_digest
    }

    /// Return the exact complete export-receipt length.
    #[must_use]
    pub const fn receipt_length(&self) -> u64 {
        self.receipt_length
    }

    /// Borrow the raw BLAKE3 digest of the embedded detailed receipt.
    #[must_use]
    pub const fn embedded_receipt_digest(&self) -> &ContentDigest {
        &self.embedded_receipt_digest
    }

    /// Return the exact embedded detailed-receipt length.
    #[must_use]
    pub const fn embedded_receipt_length(&self) -> u64 {
        self.embedded_receipt_length
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CheckedExportAttemptDecision {
    issue_id: ContentDigest,
    is_exhausted: bool,
}

/// Move-only checked export failure and its inseparable durable receipt.
pub struct PreparedExportAttemptFailure {
    decision: CheckedExportAttemptDecision,
    receipt: BudgetOwnedExportIssueReceipt,
    reference: DerivedExportReceiptReference,
    attempt_ordinal: u32,
    counter_before: u64,
}

impl PreparedExportAttemptFailure {
    /// Borrow the exact budget-owned receipt.
    #[must_use]
    pub const fn receipt(&self) -> &BudgetOwnedExportIssueReceipt {
        &self.receipt
    }

    /// Return the deterministic issue identity.
    #[must_use]
    pub const fn issue_id(&self) -> ContentDigest {
        self.decision.issue_id
    }

    /// Return whether the frozen export retry threshold is exhausted.
    #[must_use]
    pub const fn is_exhausted(&self) -> bool {
        self.decision.is_exhausted
    }

    /// Return the exact status-owned attempt ordinal.
    #[must_use]
    pub const fn attempt_ordinal(&self) -> u32 {
        self.attempt_ordinal
    }

    /// Return the exact predecessor status counter.
    #[must_use]
    pub const fn counter_before(&self) -> u64 {
        self.counter_before
    }

    /// Borrow the durable outer and embedded receipt reference.
    #[must_use]
    pub const fn receipt_reference(&self) -> &DerivedExportReceiptReference {
        &self.reference
    }

    /// Consume the failure into the sole persistence handoff.
    #[must_use]
    pub fn into_persistence(self) -> PreparedExportReceiptPersistence {
        PreparedExportReceiptPersistence { failure: self }
    }
}

/// Allocation-free persistence handoff preserving both leases and decision.
pub struct PreparedExportReceiptPersistence {
    failure: PreparedExportAttemptFailure,
}

impl PreparedExportReceiptPersistence {
    /// Borrow the exact strict durable receipt bytes.
    #[must_use]
    pub fn encoded_bytes(&self) -> &[u8] {
        self.failure.receipt.encoded.as_bytes()
    }

    /// Borrow the exact durable receipt reference.
    #[must_use]
    pub const fn receipt_reference(&self) -> &DerivedExportReceiptReference {
        &self.failure.reference
    }

    /// Return the deterministic issue identity.
    #[must_use]
    pub const fn issue_id(&self) -> ContentDigest {
        self.failure.decision.issue_id
    }

    /// Return whether this failure exhausts frozen retries.
    #[must_use]
    pub const fn is_exhausted(&self) -> bool {
        self.failure.decision.is_exhausted
    }

    /// Return the exact attempt ordinal.
    #[must_use]
    pub const fn attempt_ordinal(&self) -> u32 {
        self.failure.attempt_ordinal
    }

    /// Return the exact predecessor counter.
    #[must_use]
    pub const fn counter_before(&self) -> u64 {
        self.failure.counter_before
    }
}

/// Verified predecessor status for one derived export sink attempt.
///
/// This type is the sole proof that a caller read its `last_attempt_ordinal`,
/// `counter_before`, and receipt reference from a durably persisted predecessor
/// status bound to a committed final checkpoint generation. It has private
/// fields, no constructor reachable outside this module, and no `Deserialize`,
/// so it cannot be forged from wire bytes or fabricated by a sibling module.
///
/// ```compile_fail
/// # use aiperf_runtime::streaming::reliability::VerifiedDerivedSinkAttemptStatus;
/// # fn cannot_fabricate(status: VerifiedDerivedSinkAttemptStatus) {
/// let _ordinal = status.last_attempt_ordinal;
/// # }
/// ```
pub struct VerifiedDerivedSinkAttemptStatus {
    run: StreamRunIdentity,
    generation: CheckpointGeneration,
    sink_id: StreamingIssueComponentId,
    last_attempt_ordinal: u32,
    counter_before: u64,
    reference: DerivedExportReceiptReference,
}

impl VerifiedDerivedSinkAttemptStatus {
    /// Bind independently persisted predecessor fields to a committed final
    /// generation.
    ///
    /// The run and generation are taken from the committed authority rather than
    /// from the caller, so a status owner cannot name a generation it does not
    /// hold. Density is enforced here, once, rather than at every reader: the
    /// forward path defines `counter_before` as `u64::from(attempt_ordinal)`, so
    /// any other pairing is not a reachable predecessor status.
    fn from_status_owner(
        final_generation: &CommittedCheckpointGeneration,
        sink_id: StreamingIssueComponentId,
        last_attempt_ordinal: u32,
        counter_before: u64,
        reference: DerivedExportReceiptReference,
    ) -> Result<Self, StreamingReliabilityError> {
        if !final_generation.is_final() {
            return Err(StreamingReliabilityError::NonFinalGenerationAuthority);
        }
        if u64::from(last_attempt_ordinal) != counter_before {
            return Err(StreamingReliabilityError::NonContiguousExportCounter);
        }
        Ok(Self {
            run: *final_generation.run(),
            generation: final_generation.generation(),
            sink_id,
            last_attempt_ordinal,
            counter_before,
            reference,
        })
    }

    /// Borrow the logical run proven by the committed generation.
    #[must_use]
    pub const fn run(&self) -> &StreamRunIdentity {
        &self.run
    }

    /// Borrow the committed final generation identity.
    #[must_use]
    pub const fn generation(&self) -> &CheckpointGeneration {
        &self.generation
    }

    /// Borrow the derived sink this status describes.
    #[must_use]
    pub const fn sink_id(&self) -> &StreamingIssueComponentId {
        &self.sink_id
    }

    /// Return the independently derived predecessor attempt ordinal.
    #[must_use]
    pub const fn last_attempt_ordinal(&self) -> u32 {
        self.last_attempt_ordinal
    }

    /// Return the independently derived predecessor counter.
    #[must_use]
    pub const fn counter_before(&self) -> u64 {
        self.counter_before
    }

    /// Borrow the durable outer and embedded receipt reference.
    #[must_use]
    pub const fn receipt_reference(&self) -> &DerivedExportReceiptReference {
        &self.reference
    }
}

/// Sealed status-authored expectations used for ledger-free receipt reopen.
///
/// The context borrows the frozen policy rather than retaining its digest.
/// Retaining only a digest would make the restore path structurally unable to
/// recompute a decision and force it to replay the one the durable document
/// carries. The borrow is deliberate: `PreparedStreamingIssuePolicy` is owned by
/// the run's frozen configuration for the whole restore, so no reference count is
/// warranted.
pub struct DurableExportReceiptValidationContext<'policy> {
    run: StreamRunIdentity,
    generation: CheckpointGeneration,
    sink_id: StreamingIssueComponentId,
    policy: &'policy PreparedStreamingIssuePolicy,
    expected_attempt_ordinal: u32,
    expected_counter_before: u64,
    expected_reference: DerivedExportReceiptReference,
}

impl<'policy> DurableExportReceiptValidationContext<'policy> {
    /// Mint validation inputs only from a committed final generation and a
    /// verified predecessor status bound to that same generation.
    ///
    /// Both authorities are required and cross-checked. The status already
    /// carries the run and generation it was minted against; passing the
    /// committed generation again proves the caller is restoring under the
    /// generation it currently holds, not under a stale status object.
    fn from_final_generation_status(
        final_generation: &CommittedCheckpointGeneration,
        policy: &'policy PreparedStreamingIssuePolicy,
        status: &VerifiedDerivedSinkAttemptStatus,
    ) -> Result<Self, StreamingReliabilityError> {
        if !final_generation.is_final() {
            return Err(StreamingReliabilityError::NonFinalGenerationAuthority);
        }
        if status.run != *final_generation.run() {
            return Err(StreamingReliabilityError::ExportReceiptRunMismatch);
        }
        if status.generation != final_generation.generation() {
            return Err(StreamingReliabilityError::ExportReceiptGenerationMismatch);
        }
        Ok(Self {
            run: status.run,
            generation: status.generation.clone(),
            sink_id: status.sink_id.clone(),
            policy,
            expected_attempt_ordinal: status.last_attempt_ordinal,
            expected_counter_before: status.counter_before,
            expected_reference: status.reference.clone(),
        })
    }
}

/// Rebuild the exact typed export failure named by a persisted stage and code.
///
/// The export failure space is closed and small, so an exhaustive search over the
/// constructors is both correct and cheaper than a parallel string table that
/// could drift from `failure.rs`. Returning `None` is the only correct answer for
/// a stage/code pair no constructor can produce: such a pair is unreachable, not
/// merely unrecognized.
fn export_failure_for_stage_and_code(
    stage: StreamingFailureStage,
    code: &StreamingIssueComponentId,
) -> Option<OrdinaryStreamingFailure> {
    const FAILURE_CODES: [ResultExportFailureCode; 3] = [
        ResultExportFailureCode::Io,
        ResultExportFailureCode::Unavailable,
        ResultExportFailureCode::Attempt,
    ];
    const BUDGET_CODES: [StateBudgetFailureCode; 4] = [
        StateBudgetFailureCode::ItemCapacity,
        StateBudgetFailureCode::ByteCapacity,
        StateBudgetFailureCode::SpillCapacity,
        StateBudgetFailureCode::ProvisionalCapacity,
    ];
    FAILURE_CODES
        .into_iter()
        .map(ResultExportError::failure)
        .chain(BUDGET_CODES.into_iter().map(ResultExportError::state_budget))
        .find(|error| error.stage() == stage && error.code() == code.as_str())
        .map(OrdinaryStreamingFailure::Export)
}

/// Strictly restore one status-reachable export receipt without a live ledger.
///
/// Every field of the returned receipt is either taken from the verified status
/// authority in `context` or recomputed here through the frozen policy. The
/// durable document supplies only comparison inputs and the three facts a status
/// record does not carry: issue class, semantic context digest, and scope
/// tiebreaker. Those three are validated transitively, because any change to them
/// alters the recomputed issue identity or the selected rule.
pub async fn restore_durable_export_issue_receipt(
    encoded: BudgetedCheckpointBytes,
    context: &DurableExportReceiptValidationContext<'_>,
    parsed_budget: &StreamingResourceBudget,
) -> Result<BudgetOwnedExportIssueReceipt, StreamingReliabilityError> {
    let expected_reference = &context.expected_reference;
    let encoded_len = encoded.as_bytes().len();
    let encoded_length = u64::try_from(encoded_len)
        .map_err(|_| StreamingReliabilityError::ExportReceiptDigestLengthMismatch)?;
    let encoded_digest = ContentDigest::from_bytes(*blake3::hash(encoded.as_bytes()).as_bytes());
    if encoded_length != expected_reference.receipt_length
        || encoded_digest != expected_reference.receipt_digest
    {
        return Err(StreamingReliabilityError::ExportReceiptDigestLengthMismatch);
    }
    // Admit the proved parse upper bound before the parse allocates anything.
    // Every refusal below drops this lease, releasing the whole reservation.
    let parsed_reservation_bytes =
        restored_export_receipt_bound_bytes().map_err(|_| StreamingReliabilityError::CounterOverflow)?;
    let mut parsed_lease = parsed_budget
        .acquire(1, parsed_reservation_bytes)
        .await
        .map_err(export_budget_error)?;
    let wire: PersistedExportIssueReceiptWire = serde_json::from_slice(encoded.as_bytes())
        .map_err(|_| StreamingReliabilityError::DerivedExportReceiptUnreachable)?;
    if wire.wire_version != EXPORT_RECEIPT_WIRE_VERSION {
        return Err(StreamingReliabilityError::DerivedExportReceiptUnreachable);
    }
    if wire.embedded_receipt.wire_version != ISSUE_RECEIPT_WIRE_VERSION {
        return Err(StreamingReliabilityError::DerivedExportReceiptUnreachable);
    }
    if wire.run != context.run || wire.embedded_receipt.run != context.run {
        return Err(StreamingReliabilityError::ExportReceiptRunMismatch);
    }
    if wire.generation != context.generation {
        return Err(StreamingReliabilityError::ExportReceiptGenerationMismatch);
    }
    if wire.sink_id != context.sink_id {
        return Err(StreamingReliabilityError::ExportReceiptSinkMismatch);
    }
    let policy_digest = *context.policy.digest();
    if wire.policy_digest != policy_digest
        || wire.embedded_receipt.threshold.policy_digest != policy_digest
    {
        return Err(StreamingReliabilityError::ExportReceiptPolicyMismatch);
    }

    // Convert once, then read only the converted receipt. The wire DTO is not
    // consulted again for any embedded fact, so no later check can accidentally
    // trust a field that the total equality comparison below does not cover.
    let embedded_receipt_digest = wire.embedded_receipt_digest;
    let embedded_receipt_length = wire.embedded_receipt_length;
    let outer_issue_id = wire.issue_id;
    let outer_attempt_ordinal = wire.attempt_ordinal;
    let outer_counter_before = wire.counter_before;
    let outer_counter_after = wire.counter_after;
    let stored_receipt = persisted_receipt_from_wire(wire.embedded_receipt);

    if stored_receipt.terminal_invariant.is_some() {
        return Err(StreamingReliabilityError::DerivedExportReceiptUnreachable);
    }
    match &stored_receipt.scope {
        StreamingIssueScope::Export {
            exporter_id,
            generation,
        } if exporter_id == &context.sink_id && generation == &context.generation => {}
        StreamingIssueScope::Export { exporter_id, .. } if exporter_id != &context.sink_id => {
            return Err(StreamingReliabilityError::ExportReceiptSinkMismatch);
        }
        StreamingIssueScope::Export { .. } => {
            return Err(StreamingReliabilityError::ExportReceiptGenerationMismatch);
        }
        _ => return Err(StreamingReliabilityError::ExportReceiptGenerationMismatch),
    }

    // The status authority owns the ordinal. Everything ordinal-derived below is
    // computed from the status value, never from the document.
    let attempt_ordinal = context.expected_attempt_ordinal;
    if outer_attempt_ordinal != attempt_ordinal
        || stored_receipt.order.retry_ordinal != attempt_ordinal
        || stored_receipt.threshold.retry_ordinal != attempt_ordinal
    {
        return Err(StreamingReliabilityError::ExportReceiptAttemptMismatch);
    }
    let counter_before = u64::from(attempt_ordinal);
    if counter_before != context.expected_counter_before {
        return Err(StreamingReliabilityError::NonContiguousExportCounter);
    }
    let counter_after = counter_before
        .checked_add(1)
        .ok_or(StreamingReliabilityError::NonContiguousExportCounter)?;
    if outer_counter_before != counter_before
        || outer_counter_after != counter_after
        || stored_receipt.threshold.prior_matching_count != counter_before
        || stored_receipt.threshold.resulting_matching_count != counter_after
    {
        return Err(StreamingReliabilityError::NonContiguousExportCounter);
    }

    // Re-derive the typed failure, then rebuild the issue through the same checked
    // constructor the forward path uses. The constructor derives stage and code
    // from the failure, so a tampered stage or code cannot survive: it either
    // names no constructible failure, or it re-derives to different bytes and the
    // equality comparison below rejects it.
    let failure = export_failure_for_stage_and_code(stored_receipt.stage, &stored_receipt.code)
        .ok_or(StreamingReliabilityError::ExportReceiptFailureUnrepresentable)?;
    let issue = OrdinaryStreamingIssue::export(
        context.run,
        context.sink_id.clone(),
        context.generation.clone(),
        stored_receipt.class,
        stored_receipt.semantic_context_digest,
        attempt_ordinal,
        stored_receipt.order.scope_tiebreaker,
        failure,
    )
    .map_err(|_| StreamingReliabilityError::ExportReceiptClassCodeMismatch)?;

    // Recompute the exact legal policy decision through the merged policy engine.
    let rule = context.policy.rule_for(&issue)?;
    let is_exhausted = counter_before >= u64::from(rule.retry_limit);
    let disposition = if is_exhausted {
        rule.exhausted_disposition
    } else {
        StreamingIssueDisposition::Retry
    };
    if !is_allowed_authored_disposition(StreamingIssueScopeKind::Export, issue.class, disposition) {
        return Err(StreamingReliabilityError::IllegalDisposition);
    }
    if stored_receipt.threshold.rule_id != rule.rule_id {
        return Err(StreamingReliabilityError::ExportReceiptRuleMismatch);
    }
    if stored_receipt.threshold.is_exhausted != is_exhausted {
        return Err(StreamingReliabilityError::ExportReceiptExhaustionMismatch);
    }
    if stored_receipt.disposition != disposition {
        return Err(StreamingReliabilityError::ExportReceiptDispositionMismatch);
    }
    let threshold = StreamingIssueThresholdReceipt {
        policy_digest,
        rule_id: rule.rule_id.clone(),
        prior_matching_count: counter_before,
        resulting_matching_count: counter_after,
        retry_ordinal: attempt_ordinal,
        is_exhausted,
    };
    let embedded_receipt = persisted_receipt_from_issue(&issue, disposition, threshold);
    if embedded_receipt != stored_receipt {
        return Err(StreamingReliabilityError::DerivedExportReceiptUnreachable);
    }
    let issue_id = embedded_receipt.issue_id;
    if outer_issue_id != issue_id {
        return Err(StreamingReliabilityError::DerivedExportReceiptUnreachable);
    }

    // Serialize the recomputed receipt, not the document, so the retained bytes
    // are provably the ones the policy authorizes. The encoding is never kept —
    // only its length and digest — so it streams into the hasher.
    let (embedded_encoded_len, embedded_digest) = measured_json_digest(&embedded_receipt)
        .map_err(|_| StreamingReliabilityError::DerivedExportReceiptUnreachable)?;
    let embedded_length = u64::try_from(embedded_encoded_len)
        .map_err(|_| StreamingReliabilityError::ExportReceiptDigestLengthMismatch)?;
    if embedded_receipt_length != embedded_length
        || embedded_receipt_digest != embedded_digest
        || expected_reference.embedded_receipt_length != embedded_length
        || expected_reference.embedded_receipt_digest != embedded_digest
    {
        return Err(StreamingReliabilityError::ExportReceiptDigestLengthMismatch);
    }

    let receipt = PersistedExportIssueReceipt {
        wire_version: EXPORT_RECEIPT_WIRE_VERSION,
        run: context.run,
        generation: context.generation.clone(),
        sink_id: context.sink_id.clone(),
        attempt_ordinal,
        issue_id,
        policy_digest,
        counter_before,
        counter_after,
        embedded_receipt_digest: embedded_digest,
        embedded_receipt_length: embedded_length,
        embedded_receipt,
    };
    // Synchronous exact-down adjustment on the success path only.
    let parsed_charge_bytes = parsed_export_receipt_bytes(&receipt)
        .map_err(|_| StreamingReliabilityError::CounterOverflow)?;
    parsed_lease
        .shrink_to(1, parsed_charge_bytes)
        .map_err(export_budget_error)?;
    Ok(BudgetOwnedExportIssueReceipt {
        receipt,
        encoded,
        parsed_lease,
        parsed_charge_bytes,
    })
}

/// Return a proved upper bound on one restored export receipt's parse charge.
///
/// [`parsed_export_receipt_bytes`] is four fixed structural terms plus the
/// retained bytes of at most four checked component identifiers, each bounded by
/// [`MAX_COMPONENT_ID_BYTES`]. Acquiring this bound before the parse and
/// shrinking to the exact charge afterwards keeps the parsed document inside the
/// budget for its whole life, rather than admitting it after the fact.
fn restored_export_receipt_bound_bytes() -> Result<usize, BudgetError> {
    super::budget::checked_sum([
        size_of::<PersistedExportIssueReceipt>(),
        size_of::<PersistedStreamingIssueReceipt>(),
        size_of::<PreparedExportAttemptFailure>(),
        size_of::<CheckedExportAttemptDecision>(),
        MAX_COMPONENT_ID_BYTES,
        MAX_COMPONENT_ID_BYTES,
        MAX_COMPONENT_ID_BYTES,
        MAX_COMPONENT_ID_BYTES,
    ])
}

fn persisted_receipt_from_wire(
    wire: PersistedStreamingIssueReceiptWire,
) -> PersistedStreamingIssueReceipt {
    PersistedStreamingIssueReceipt {
        wire_version: wire.wire_version,
        issue_id: wire.issue_id,
        run: wire.run,
        scope: wire.scope,
        class: wire.class,
        stage: wire.stage,
        code: wire.code,
        semantic_context_digest: wire.semantic_context_digest,
        order: wire.order,
        terminal_invariant: wire.terminal_invariant,
        disposition: wire.disposition,
        threshold: StreamingIssueThresholdReceipt {
            policy_digest: wire.threshold.policy_digest,
            rule_id: wire.threshold.rule_id,
            prior_matching_count: wire.threshold.prior_matching_count,
            resulting_matching_count: wire.threshold.resulting_matching_count,
            retry_ordinal: wire.threshold.retry_ordinal,
            is_exhausted: wire.threshold.is_exhausted,
        },
    }
}

/// Return a proven upper bound on the strict-v2 encoding of the receipt one
/// issue produces.
///
/// The bound is the fixed structural maximum plus the three checked component
/// identifiers at their validated maximum length. Both contributors are proven,
/// so acquiring this bound before serialization and shrinking to the exact
/// encoded length afterwards keeps the transient encoding inside the budget.
fn receipt_encoding_bound_bytes() -> Result<usize, BudgetError> {
    super::budget::checked_sum([
        RECEIPT_ENCODING_FIXED_BOUND_BYTES,
        MAX_COMPONENT_ID_BYTES,
        MAX_COMPONENT_ID_BYTES,
        MAX_COMPONENT_ID_BYTES,
    ])
}

/// Return a proven upper bound on the parsed receipt allocation.
fn parsed_receipt_bound_bytes() -> Result<usize, BudgetError> {
    super::budget::checked_sum([
        size_of::<PersistedStreamingIssueReceipt>(),
        MAX_COMPONENT_ID_BYTES,
        MAX_COMPONENT_ID_BYTES,
        MAX_COMPONENT_ID_BYTES,
    ])
}

/// Return the exact structural charge of one pending-input ordered-map entry.
fn pending_input_entry_bytes() -> Result<usize, BudgetError> {
    super::budget::ordered_map_entry_bytes::<PendingInputKey, PendingIssue>()
}

/// Return the exact structural charge of one retained-receipt ordered-map entry.
fn receipt_entry_bytes() -> Result<usize, BudgetError> {
    super::budget::ordered_map_entry_bytes::<ContentDigest, RetainedReceipt>()
}

/// Return the exact structural charge of one input-frontier ordered-map entry.
fn input_frontier_entry_bytes() -> Result<usize, BudgetError> {
    super::budget::ordered_map_entry_bytes::<StreamingInputDomainIdentity, RetainedInputFrontier>()
}

/// Return the exact structural charge of one counter ordered-map entry,
/// including the checked identifiers owned by its key.
fn counter_entry_bytes(key: &StreamingIssueCounterKey) -> Result<usize, BudgetError> {
    super::budget::checked_sum([
        super::budget::ordered_map_entry_bytes::<StreamingIssueCounterKey, RetainedCounter>()?,
        key.rule_id.retained_bytes(),
        match &key.domain {
            StreamingIssueCounterDomain::Export { exporter_id, .. } => exporter_id.retained_bytes(),
            StreamingIssueCounterDomain::Run
            | StreamingIssueCounterDomain::Input(_)
            | StreamingIssueCounterDomain::Action
            | StreamingIssueCounterDomain::CheckpointAttempt => 0,
        },
    ])
}

fn reserve_pending_issue(
    budget: &StreamingResourceBudget,
    issue: OrdinaryStreamingIssue,
) -> Result<PendingIssue, StreamingReliabilityError> {
    let retained_issue_bytes =
        retained_issue_bytes(&issue).map_err(|_| StreamingReliabilityError::CounterOverflow)?;
    // Three charged items: the pending issue itself, its ordered-map entry
    // while it waits behind an input frontier, and the encoded receipt that
    // classification will retain. Every byte contributor is derived, and
    // `budget_owned_receipt_from_reservation` shrinks the reservation to its
    // exact retained size on success.
    let total_bytes = super::budget::checked_sum([
        retained_issue_bytes,
        pending_input_entry_bytes().map_err(|_| StreamingReliabilityError::CounterOverflow)?,
        receipt_encoding_bound_bytes().map_err(|_| StreamingReliabilityError::CounterOverflow)?,
        parsed_receipt_bound_bytes().map_err(|_| StreamingReliabilityError::CounterOverflow)?,
        receipt_entry_bytes().map_err(|_| StreamingReliabilityError::CounterOverflow)?,
    ])
    .map_err(|_| StreamingReliabilityError::CounterOverflow)?;
    let reservation = budget
        .try_acquire(3, total_bytes)
        .map_err(state_budget_error)?;
    Ok(PendingIssue {
        issue,
        reservation,
        retained_issue_bytes,
    })
}

fn action_disposition(
    entry: &PendingActionFailure,
    decision: ActionDecisionSnapshot,
) -> Result<ActionFailureDisposition, StreamingReliabilityError> {
    match decision.disposition {
        StreamingIssueDisposition::Retry => {
            Ok(ActionFailureDisposition::Retry(PreparedActionRetry {
                retry_ordinal: decision.retry_ordinal,
            }))
        }
        StreamingIssueDisposition::Backpressure => Ok(ActionFailureDisposition::Backpressure(
            PreparedActionBackpressure {
                needs_admission_fence: decision.needs_admission_fence,
            },
        )),
        StreamingIssueDisposition::TerminalActionReceipt => Ok(
            ActionFailureDisposition::TerminalActionReceipt(PreparedActionFailureIdentity {
                run: entry.run,
                action_id: entry.action_id,
                sequence: entry.sequence,
                issue_id: decision.issue_id,
                terminal_evidence_digest: entry.terminal_evidence_digest,
            }),
        ),
        StreamingIssueDisposition::Quarantine
        | StreamingIssueDisposition::Hole
        | StreamingIssueDisposition::Continue
        | StreamingIssueDisposition::ExportIncomplete
        | StreamingIssueDisposition::FailRun => Err(StreamingReliabilityError::IllegalDisposition),
    }
}

/// Return the exact retained bytes of one ordinary issue.
///
/// Component identifiers charge their retained capacity, not their length, so a
/// short identifier held in an oversized allocation cannot bypass the
/// fixed-memory invariant.
fn retained_issue_bytes(issue: &OrdinaryStreamingIssue) -> Result<usize, BudgetError> {
    super::budget::checked_sum([
        size_of::<OrdinaryStreamingIssue>(),
        issue.code.retained_bytes(),
        issue
            .scope
            .exporter_id()
            .map_or(0, StreamingIssueComponentId::retained_bytes),
    ])
}

/// Return the exact retained bytes of one parsed persisted receipt.
fn parsed_receipt_bytes(receipt: &PersistedStreamingIssueReceipt) -> Result<usize, BudgetError> {
    super::budget::checked_sum([
        size_of::<PersistedStreamingIssueReceipt>(),
        receipt.code.retained_bytes(),
        receipt.threshold.rule_id.retained_bytes(),
        receipt
            .scope
            .exporter_id()
            .map_or(0, StreamingIssueComponentId::retained_bytes),
    ])
}

/// Return the exact retained bytes of one parsed persisted export receipt.
fn parsed_export_receipt_bytes(
    receipt: &PersistedExportIssueReceipt,
) -> Result<usize, BudgetError> {
    super::budget::checked_sum([
        size_of::<PersistedExportIssueReceipt>(),
        receipt.sink_id.retained_bytes(),
        parsed_receipt_bytes(&receipt.embedded_receipt)?,
        size_of::<PreparedExportAttemptFailure>(),
        size_of::<CheckedExportAttemptDecision>(),
    ])
}

fn persisted_receipt_from_issue(
    issue: &OrdinaryStreamingIssue,
    disposition: StreamingIssueDisposition,
    threshold: StreamingIssueThresholdReceipt,
) -> PersistedStreamingIssueReceipt {
    let issue_id = issue.issue_id();
    PersistedStreamingIssueReceipt {
        wire_version: ISSUE_RECEIPT_WIRE_VERSION,
        issue_id,
        run: issue.run,
        scope: issue.scope.clone(),
        class: issue.class,
        stage: issue.stage,
        code: issue.code.clone(),
        semantic_context_digest: issue.semantic_context_digest,
        order: issue.order.clone(),
        terminal_invariant: None,
        disposition,
        threshold,
    }
}

fn budget_owned_receipt_from_reservation(
    receipt: PersistedStreamingIssueReceipt,
    mut reservation: BudgetLease,
    retained_issue_bytes: usize,
) -> Result<
    (BudgetOwnedStreamingIssueReceipt, BudgetLease, BudgetLease),
    (StreamingReliabilityError, BudgetLease),
> {
    // The reservation already covers the proven encoding bound, so this
    // allocation is inside the budget before it exists.
    let encoded = match serde_json::to_vec(&receipt) {
        Ok(encoded) => encoded,
        Err(_) => {
            return Err((
                StreamingReliabilityError::CorruptCheckpointState,
                reservation,
            ));
        }
    };
    let entry_bytes = match receipt_entry_bytes() {
        Ok(bytes) => bytes,
        Err(_) => return Err((StreamingReliabilityError::CounterOverflow, reservation)),
    };
    let exact_bytes =
        match super::budget::checked_sum([retained_issue_bytes, encoded.len(), entry_bytes]) {
            Ok(bytes) => bytes,
            Err(_) => return Err((StreamingReliabilityError::CounterOverflow, reservation)),
        };
    // Shrinking here is the synchronous settlement of the proven bound to the
    // exact retained charge; it can only ever release capacity.
    if reservation.shrink_to(3, exact_bytes).is_err() {
        return Err((
            StreamingReliabilityError::StateBudget(StateBudgetFailureCode::ByteCapacity),
            reservation,
        ));
    }
    let encoded_lease = match reservation.split_off(1, encoded.len()) {
        Ok(lease) => lease,
        Err(_) => {
            return Err((
                StreamingReliabilityError::CorruptCheckpointState,
                reservation,
            ));
        }
    };
    let entry_lease = match reservation.split_off(1, entry_bytes) {
        Ok(lease) => lease,
        Err(_) => {
            return Err((
                StreamingReliabilityError::CorruptCheckpointState,
                reservation,
            ));
        }
    };
    let facts = CompactIssueReceiptFacts::from_receipt(&receipt);
    let encoded = match BudgetedCheckpointBytes::new(Bytes::from(encoded), encoded_lease) {
        Ok(encoded) => encoded,
        Err(_) => {
            return Err((
                StreamingReliabilityError::CorruptCheckpointState,
                reservation,
            ));
        }
    };
    // `receipt` is dropped here: only the compact facts and canonical bytes are
    // retained. The returned leases are, in order, the retained receipt, its
    // ordered-map entry authority, and the pending-issue remainder the caller
    // releases after classification.
    Ok((
        BudgetOwnedStreamingIssueReceipt { facts, encoded },
        entry_lease,
        reservation,
    ))
}

// Canonical, restart-stable digest over the terminal membership one gap closure
// crossed. The reporter can recompute this without the frozen inventory, which
// no longer exists after a restart.
fn action_gap_coverage_digest(
    run: &StreamRunIdentity,
    action_terminals: &BTreeMap<GlobalSequence, RetainedActionTerminal>,
    through: GlobalSequence,
    membership_root: ContentDigest,
) -> ContentDigest {
    let mut hasher = blake3::Hasher::new();
    update_hash_field(
        &mut hasher,
        b"aiperf.streaming.action-gap-closure-coverage.v1",
    );
    update_hash_field(&mut hasher, run.logical_replay_run().as_bytes());
    update_hash_field(&mut hasher, &through.get().to_le_bytes());
    update_hash_field(&mut hasher, membership_root.as_bytes());
    let mut covered: u64 = 0;
    for (sequence, retained) in action_terminals.range(..=through) {
        update_hash_field(&mut hasher, &sequence.get().to_le_bytes());
        update_hash_field(&mut hasher, retained.fact.membership_digest.as_bytes());
        covered = covered.saturating_add(1);
    }
    update_hash_field(&mut hasher, &covered.to_le_bytes());
    ContentDigest::from_bytes(*hasher.finalize().as_bytes())
}

// A restored frontier is provable when every sequence it crosses is either a
// retained terminal or covered by the retained closure. The walk stops at the
// first absent sequence, so it is bounded by the retained terminal count.
fn is_action_frontier_proven(
    action_terminals: &BTreeMap<GlobalSequence, RetainedActionTerminal>,
    frontier: GlobalSequence,
    closure_through: Option<GlobalSequence>,
) -> bool {
    let mut next = match closure_through {
        Some(through) if through >= frontier => return true,
        Some(through) => match through.get().checked_add(1) {
            Some(next) => next,
            None => return true,
        },
        None => 0,
    };
    while next <= frontier.get() {
        if !action_terminals.contains_key(&GlobalSequence::new(next)) {
            return false;
        }
        let Some(incremented) = next.checked_add(1) else {
            return true;
        };
        next = incremented;
    }
    true
}

// Restore-side revalidation. This takes borrowed restored state rather than
// `&mut self` so the caller validates before installing any reporter field.
fn checked_restored_action_gap_closure(
    run: &StreamRunIdentity,
    budget: &StreamingResourceBudget,
    action_terminals: &BTreeMap<GlobalSequence, RetainedActionTerminal>,
    action_frontier: Option<GlobalSequence>,
    persisted: Option<PersistedActionGapClosure>,
) -> Result<Option<RetainedActionGapClosure>, StreamingReliabilityError> {
    let Some(persisted) = persisted else {
        let Some(frontier) = action_frontier else {
            return Ok(None);
        };
        if is_action_frontier_proven(action_terminals, frontier, None) {
            return Ok(None);
        }
        return Err(StreamingReliabilityError::UnprovenActionGapClosure);
    };
    let Some(frontier) = action_frontier else {
        return Err(StreamingReliabilityError::UnprovenActionGapClosure);
    };
    if persisted.through > frontier {
        return Err(StreamingReliabilityError::UnprovenActionGapClosure);
    }
    if !is_action_frontier_proven(action_terminals, frontier, Some(persisted.through)) {
        return Err(StreamingReliabilityError::UnprovenActionGapClosure);
    }
    let recomputed = action_gap_coverage_digest(
        run,
        action_terminals,
        persisted.through,
        persisted.membership_root,
    );
    if recomputed != persisted.coverage_digest {
        return Err(StreamingReliabilityError::ForgedActionGapClosure);
    }
    let lease = budget
        .try_acquire(1, ACTION_GAP_CLOSURE_CHARGE_BYTES)
        .map_err(state_budget_error)?;
    Ok(Some(RetainedActionGapClosure {
        through: persisted.through,
        membership_root: persisted.membership_root,
        coverage_digest: persisted.coverage_digest,
        lease,
    }))
}

fn counter_key_for_issue(
    issue: &OrdinaryStreamingIssue,
    rule_id: StreamingIssueComponentId,
) -> StreamingIssueCounterKey {
    let domain = match &issue.scope {
        StreamingIssueScope::Run => StreamingIssueCounterDomain::Run,
        StreamingIssueScope::Partition { input_domain, .. }
        | StreamingIssueScope::Record { input_domain, .. }
        | StreamingIssueScope::Session { input_domain, .. } => {
            StreamingIssueCounterDomain::Input(input_domain.clone())
        }
        StreamingIssueScope::Action { .. } => StreamingIssueCounterDomain::Action,
        StreamingIssueScope::Export {
            exporter_id,
            generation,
        } => StreamingIssueCounterDomain::Export {
            exporter_id: exporter_id.clone(),
            generation: generation.clone(),
        },
        StreamingIssueScope::CheckpointAttempt { .. } => {
            StreamingIssueCounterDomain::CheckpointAttempt
        }
    };
    StreamingIssueCounterKey { domain, rule_id }
}

fn update_summary(
    summary: &mut StreamingIssueSummary,
    receipt: &PersistedStreamingIssueReceipt,
    needs_admission_fence: bool,
) -> Result<(), StreamingReliabilityError> {
    summary.total = summary
        .total
        .checked_add(1)
        .ok_or(StreamingReliabilityError::CounterOverflow)?;
    increment_summary_counter(&mut summary.by_scope, receipt.scope.kind())?;
    increment_summary_counter(&mut summary.by_class, receipt.class)?;
    increment_summary_counter(&mut summary.by_disposition, receipt.disposition)?;
    summary.is_admission_fenced |= needs_admission_fence;
    Ok(())
}

fn increment_summary_counter<K: Ord + Copy>(
    counters: &mut BTreeMap<K, u64>,
    key: K,
) -> Result<(), StreamingReliabilityError> {
    let next = counters
        .get(&key)
        .copied()
        .unwrap_or(0)
        .checked_add(1)
        .ok_or(StreamingReliabilityError::CounterOverflow)?;
    counters.insert(key, next);
    Ok(())
}

// The classification is exposed as the bare code so scope-specific budget
// errors can wrap it directly instead of narrowing a wider error back down.
/// Counting adapter that measures encoded length without retaining the encoding.
///
/// `serde_json::to_writer` streams; wrapping a sink in this adapter yields the
/// exact encoded length with no output-proportional allocation, which is what
/// makes a pre-serialization aggregate acquisition exact rather than guessed.
struct CountingWriter<W: Write> {
    inner: W,
    count: usize,
}

impl<W: Write> CountingWriter<W> {
    fn new(inner: W) -> Self {
        Self { inner, count: 0 }
    }
}

impl<W: Write> Write for CountingWriter<W> {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        self.inner.write_all(data)?;
        self.count = self
            .count
            .checked_add(data.len())
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "encoded length overflow"))?;
        Ok(data.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

/// Measure the exact JSON encoding length of `value` without allocating it.
fn measured_json_len<T>(value: &T) -> Result<usize, serde_json::Error>
where
    T: Serialize + ?Sized,
{
    let mut writer = CountingWriter::new(io::sink());
    serde_json::to_writer(&mut writer, value)?;
    Ok(writer.count)
}

/// Measure the exact JSON encoding length and digest without allocating it.
///
/// `blake3::Hasher` implements [`Write`], so the encoder streams directly into
/// the hasher. This is the allocation-free replacement for buffering an
/// encoding solely to read its length and content digest.
fn measured_json_digest<T>(value: &T) -> Result<(usize, ContentDigest), serde_json::Error>
where
    T: Serialize + ?Sized,
{
    let mut writer = CountingWriter::new(blake3::Hasher::new());
    serde_json::to_writer(&mut writer, value)?;
    let digest = ContentDigest::from_bytes(*writer.inner.finalize().as_bytes());
    Ok((writer.count, digest))
}

const fn budget_failure_code(error: BudgetError) -> StateBudgetFailureCode {
    match error {
        BudgetError::CapacityUnavailable | BudgetError::RequestExceedsCapacity => {
            StateBudgetFailureCode::ByteCapacity
        }
        BudgetError::ZeroCapacity
        | BudgetError::PermitCountTooLarge
        | BudgetError::Closed
        | BudgetError::CannotGrowLease
        | BudgetError::InvalidFragmentItemCharge { .. }
        | BudgetError::ActionPayloadUndercharged { .. }
        | BudgetError::PartialLeasedBuffer { .. }
        | BudgetError::AccountingOverflow => StateBudgetFailureCode::ItemCapacity,
    }
}

fn state_budget_error(error: BudgetError) -> StreamingReliabilityError {
    StreamingReliabilityError::StateBudget(budget_failure_code(error))
}

fn export_budget_error(error: BudgetError) -> StreamingReliabilityError {
    StreamingReliabilityError::ExportReceiptBudget(budget_failure_code(error))
}

fn quarantine_install_budget_error(error: BudgetError) -> StreamingReliabilityError {
    StreamingReliabilityError::QuarantineInstallBudget(budget_failure_code(error))
}

/// Participant callbacks return [`CheckpointError`]; reliability failures keep
/// their exact budget classification instead of collapsing into storage.
fn checkpoint_error_from_reliability(
    participant: CheckpointParticipantId,
    error: StreamingReliabilityError,
) -> CheckpointError {
    match error {
        StreamingReliabilityError::StateBudget(code)
        | StreamingReliabilityError::ExportReceiptBudget(code)
        | StreamingReliabilityError::QuarantineInstallBudget(code) => {
            CheckpointError::StateBudget { participant, code }
        }
        _ => CheckpointError::ObjectVerification,
    }
}

/// Sole mutable host reliability owner and checkpoint participant.
///
/// The synchronous action methods establish the required no-borrow-across-await
/// boundary. Their defaults refuse because the parallel budget slice has not
/// yet supplied a precharged detailed-state owner.
#[async_trait(?Send)]
pub trait StreamingIssueReporter: StreamingCheckpointParticipant {
    /// Return the cloneable adapter injection handle.
    fn handle(&self) -> StreamingIssueReporterHandle;

    /// Retain checked failed-action evidence without advancing the action frontier.
    fn enqueue_failed_action(
        &mut self,
        _evidence: &dyn CheckedActionFailureTerminalEvidenceView,
        _issue: OrdinaryStreamingIssue,
    ) -> Result<QueuedActionFailure, StreamingReliabilityError> {
        Err(StreamingReliabilityError::ReliabilityStateUnavailable)
    }

    /// Poll one retained failed action at an explicit synchronous event boundary.
    fn poll_failed_action(
        &mut self,
        _queued: QueuedActionFailure,
    ) -> Result<ActionFailureDisposition, StreamingReliabilityError> {
        Err(StreamingReliabilityError::ReliabilityStateUnavailable)
    }

    /// Mint one checked action terminal fact from sealed membership.
    fn prepare_action_terminal(
        &mut self,
        _membership: &dyn CheckedActionTerminalMembershipView,
    ) -> Result<CheckedActionTerminalFact, StreamingReliabilityError> {
        Err(StreamingReliabilityError::ReliabilityStateUnavailable)
    }

    /// Mint one dense action gap proof from the immutable action inventory.
    fn prepare_no_more_actions_before(
        &mut self,
        _inventory: &dyn FrozenActionInventoryView,
        _through: GlobalSequence,
    ) -> Result<CheckedNoMoreActionsBefore, StreamingReliabilityError> {
        Err(StreamingReliabilityError::ReliabilityStateUnavailable)
    }

    /// Prepare a budget-owned acknowledgement of a retained tombstone view.
    async fn prepare_session_quarantine_install(
        &mut self,
        _view: &dyn SessionQuarantineTombstoneView,
        _issue_id: ContentDigest,
        _barrier: &CheckpointBarrier,
        _budget: &StreamingResourceBudget,
    ) -> Result<PreparedSessionQuarantineInstall, StreamingReliabilityError> {
        Err(StreamingReliabilityError::ReliabilityStateUnavailable)
    }

    /// Recheck a prepared acknowledgement against a fresh sealed view.
    fn verify_session_quarantine_install(
        &self,
        _prepared: &PreparedSessionQuarantineInstall,
        _current_view: &dyn SessionQuarantineTombstoneView,
        _barrier: &CheckpointBarrier,
    ) -> Result<(), StreamingReliabilityError> {
        Err(StreamingReliabilityError::ReliabilityStateUnavailable)
    }

    /// Borrow the accepted move-only quarantine acknowledgement, if one is retained.
    fn accepted_quarantine_install(&self) -> Option<&PreparedSessionQuarantineInstall> {
        None
    }

    /// Prepare one strict budget-owned derived-sink failure receipt.
    async fn prepare_export_attempt_failure(
        &mut self,
        _run: &StreamRunIdentity,
        _generation: &CheckpointGeneration,
        _sink_id: &StreamingIssueComponentId,
        _attempt_ordinal: u32,
        _outcome: ResultSinkAttemptOutcome,
        _budget: &StreamingResourceBudget,
    ) -> Result<PreparedExportAttemptFailure, StreamingReliabilityError> {
        Err(StreamingReliabilityError::ReliabilityStateUnavailable)
    }

    /// Submit an ordered update to the future budget-owned ledger.
    async fn report(
        &mut self,
        _update: IssueSequenceUpdate,
    ) -> Result<Option<StreamingIssueOutcome>, StreamingReliabilityError> {
        Err(StreamingReliabilityError::ReliabilityStateUnavailable)
    }

    /// Prepare a non-destructive detailed-receipt partition at one barrier.
    async fn receipt_partition_view(
        &mut self,
        _barrier: &CheckpointBarrier,
    ) -> Result<PreparedIssueReceiptPartitionView, StreamingReliabilityError> {
        Err(StreamingReliabilityError::ReliabilityStateUnavailable)
    }

    /// Retain the pre-CAS binding between one staged result epoch and the
    /// detailed receipts that epoch publishes.
    fn bind_prepared_result_epoch(
        &mut self,
        _prepared: &PreparedResultEpoch,
    ) -> Result<(), StreamingReliabilityError> {
        Err(StreamingReliabilityError::ReliabilityStateUnavailable)
    }

    /// Borrow deterministic matching counters.
    fn counters(&self) -> StreamingIssueCounterView<'_> {
        StreamingIssueCounterView { counters: None }
    }

    /// Return the current fixed-size summary.
    fn summary(&self) -> Result<StreamingIssueSummary, StreamingReliabilityError> {
        Ok(StreamingIssueSummary::empty())
    }
}

#[async_trait(?Send)]
impl StreamingCheckpointParticipant for BudgetOwnedStreamingIssueReporter {
    fn participant_id(&self) -> CheckpointParticipantId {
        CheckpointParticipantId::new("streaming_issue_ledger")
    }

    async fn checkpoint_view(
        &mut self,
        barrier: &CheckpointBarrier,
    ) -> Result<PreparedParticipantState, CheckpointError> {
        self.prepare_ledger_participant_state(barrier).await
    }

    async fn initialize(
        &mut self,
        state: Option<CommittedParticipantState>,
    ) -> Result<(), CheckpointError> {
        if self.is_initialized {
            return Err(CheckpointError::AlreadyInitialized);
        }
        if let Some(state) = state {
            self.restore_ledger_state(&state)?;
        }
        // Only a successful fresh start or restore consumes the one-shot guard,
        // so a refused restore stays retryable.
        self.is_initialized = true;
        Ok(())
    }

    async fn checkpoint_committed(
        &mut self,
        receipt: &CommittedParticipantReceipt,
    ) -> Result<(), CheckpointError> {
        let participant = self.participant_id();
        if receipt.run() != &self.run || receipt.participant_id() != &participant {
            return Err(CheckpointError::PostCommitNotification { participant });
        }
        let Some(pending) = self.pending_commit.as_ref() else {
            // Idempotent re-delivery after retirement, or a barrier this owner
            // never prepared. Neither retires anything.
            return Ok(());
        };
        let Some(result_index_root) = pending.result_index_root else {
            return Err(CheckpointError::PostCommitNotification { participant });
        };
        if receipt.generation().epoch() != pending.epoch
            || receipt.descriptor_digest() != &pending.descriptor_digest
            || receipt.represented_cut() != &pending.represented_cut
            || receipt.result_index_root() != &result_index_root
        {
            return Err(CheckpointError::PostCommitNotification { participant });
        }
        let retire_through_ordinal = pending.retire_through_ordinal;
        let receipt_root = pending.receipt_root;
        // First mutation, after every comparison has passed.
        self.pending_commit = None;
        self.receipts
            .retain(|_, retained| retained.ordinal >= retire_through_ordinal);
        self.retired_receipt_root = Some(receipt_root);
        Ok(())
    }
}

#[async_trait(?Send)]
impl StreamingIssueReporter for BudgetOwnedStreamingIssueReporter {
    fn handle(&self) -> StreamingIssueReporterHandle {
        StreamingIssueReporterHandle {
            inner: self.submission.clone(),
        }
    }

    fn enqueue_failed_action(
        &mut self,
        evidence: &dyn CheckedActionFailureTerminalEvidenceView,
        issue: OrdinaryStreamingIssue,
    ) -> Result<QueuedActionFailure, StreamingReliabilityError> {
        self.drain_submission_queue()?;
        self.enqueue_action_failure(evidence, issue)
    }

    fn poll_failed_action(
        &mut self,
        queued: QueuedActionFailure,
    ) -> Result<ActionFailureDisposition, StreamingReliabilityError> {
        self.drain_submission_queue()?;
        self.poll_action_failure(queued)
    }

    fn prepare_action_terminal(
        &mut self,
        membership: &dyn CheckedActionTerminalMembershipView,
    ) -> Result<CheckedActionTerminalFact, StreamingReliabilityError> {
        self.drain_submission_queue()?;
        self.prepare_checked_action_terminal(membership)
    }

    fn prepare_no_more_actions_before(
        &mut self,
        inventory: &dyn FrozenActionInventoryView,
        through: GlobalSequence,
    ) -> Result<CheckedNoMoreActionsBefore, StreamingReliabilityError> {
        self.drain_submission_queue()?;
        self.prepare_action_gap_closure(inventory, through)
    }

    async fn prepare_session_quarantine_install(
        &mut self,
        view: &dyn SessionQuarantineTombstoneView,
        issue_id: ContentDigest,
        barrier: &CheckpointBarrier,
        budget: &StreamingResourceBudget,
    ) -> Result<PreparedSessionQuarantineInstall, StreamingReliabilityError> {
        self.prepare_quarantine_install(view, issue_id, barrier, budget)
            .await
    }

    fn verify_session_quarantine_install(
        &self,
        prepared: &PreparedSessionQuarantineInstall,
        current_view: &dyn SessionQuarantineTombstoneView,
        barrier: &CheckpointBarrier,
    ) -> Result<(), StreamingReliabilityError> {
        self.verify_quarantine_install(prepared, current_view, barrier)
    }

    fn accepted_quarantine_install(&self) -> Option<&PreparedSessionQuarantineInstall> {
        self.accepted_quarantine.as_ref()
    }

    async fn prepare_export_attempt_failure(
        &mut self,
        run: &StreamRunIdentity,
        generation: &CheckpointGeneration,
        sink_id: &StreamingIssueComponentId,
        attempt_ordinal: u32,
        outcome: ResultSinkAttemptOutcome,
        budget: &StreamingResourceBudget,
    ) -> Result<PreparedExportAttemptFailure, StreamingReliabilityError> {
        self.prepare_export_failure(run, generation, sink_id, attempt_ordinal, outcome, budget)
            .await
    }

    async fn report(
        &mut self,
        update: IssueSequenceUpdate,
    ) -> Result<Option<StreamingIssueOutcome>, StreamingReliabilityError> {
        self.drain_submission_queue()?;
        match update {
            IssueSequenceUpdate::Issue(issue) => self.submit_issue(issue),
            IssueSequenceUpdate::NoMoreBefore {
                input_domain,
                through,
            } => {
                self.advance_input_frontier(input_domain, through)?;
                Ok(None)
            }
            IssueSequenceUpdate::CheckedActionTerminal(fact) => {
                self.retain_action_terminal(fact)?;
                Ok(None)
            }
            IssueSequenceUpdate::CheckedNoMoreActionsBefore(closure) => {
                self.retain_action_gap_closure(closure)?;
                Ok(None)
            }
            IssueSequenceUpdate::PreparedSessionQuarantineInstall(prepared) => {
                self.accept_quarantine_install(prepared)?;
                Ok(None)
            }
        }
    }

    async fn receipt_partition_view(
        &mut self,
        barrier: &CheckpointBarrier,
    ) -> Result<PreparedIssueReceiptPartitionView, StreamingReliabilityError> {
        self.prepare_receipt_partition_view(barrier).await
    }

    fn bind_prepared_result_epoch(
        &mut self,
        prepared: &PreparedResultEpoch,
    ) -> Result<(), StreamingReliabilityError> {
        self.bind_result_epoch(prepared)
    }

    fn counters(&self) -> StreamingIssueCounterView<'_> {
        StreamingIssueCounterView {
            counters: Some(&self.counters),
        }
    }

    fn summary(&self) -> Result<StreamingIssueSummary, StreamingReliabilityError> {
        Ok(self.summary.clone())
    }
}

#[allow(dead_code)]
enum HostFailure {
    RunAuthorityMismatch,
    SourceIdentityAuthorityMismatch,
    PublicationProofMismatch,
    WriterLeaseMismatch,
    CasExpectationMismatch,
    SecurityAuthorityMismatch,
    ConflictingStableContent,
    ImpossibleTruthfulOrdering,
    ImpossibleTruthfulWatermark,
    ImpossibleTruthfulCut,
    FrozenSemanticDrift,
    AccountingCorruption,
}

#[allow(dead_code)]
struct VerifiedHostIssue {
    invariant: StreamingTerminalInvariant,
}

#[allow(dead_code)]
fn classify_host_failure(failure: HostFailure) -> VerifiedHostIssue {
    let invariant = match failure {
        HostFailure::RunAuthorityMismatch => StreamingTerminalInvariant::RunAuthorityMismatch,
        HostFailure::SourceIdentityAuthorityMismatch => {
            StreamingTerminalInvariant::SourceIdentityAuthorityMismatch
        }
        HostFailure::PublicationProofMismatch => {
            StreamingTerminalInvariant::PublicationProofMismatch
        }
        HostFailure::WriterLeaseMismatch => StreamingTerminalInvariant::WriterLeaseMismatch,
        HostFailure::CasExpectationMismatch => StreamingTerminalInvariant::CasExpectationMismatch,
        HostFailure::SecurityAuthorityMismatch => {
            StreamingTerminalInvariant::SecurityAuthorityMismatch
        }
        HostFailure::ConflictingStableContent => {
            StreamingTerminalInvariant::ConflictingStableContent
        }
        HostFailure::ImpossibleTruthfulOrdering => {
            StreamingTerminalInvariant::ImpossibleTruthfulOrdering
        }
        HostFailure::ImpossibleTruthfulWatermark => {
            StreamingTerminalInvariant::ImpossibleTruthfulWatermark
        }
        HostFailure::ImpossibleTruthfulCut => StreamingTerminalInvariant::ImpossibleTruthfulCut,
        HostFailure::FrozenSemanticDrift => StreamingTerminalInvariant::FrozenSemanticDrift,
        HostFailure::AccountingCorruption => StreamingTerminalInvariant::AccountingCorruption,
    };
    VerifiedHostIssue { invariant }
}

#[allow(dead_code)]
fn fail_run_decision(verified: &VerifiedHostIssue) -> StreamingIssueDecision {
    StreamingIssueDecision {
        disposition: StreamingIssueDisposition::FailRun,
        rule: StreamingIssueThresholdRule {
            rule_id: StreamingIssueComponentId("host_terminal_invariant".to_owned()),
            scope: StreamingIssueScopeKind::Run,
            class: StreamingIssueClass::Invariant,
            code: Some(StreamingIssueComponentId(
                terminal_invariant_code(verified.invariant).to_owned(),
            )),
            retry_limit: 0,
            exhausted_disposition: StreamingIssueDisposition::FailRun,
            admission_fence_count: None,
        },
        needs_admission_fence: true,
    }
}

#[allow(dead_code)]
struct VerifiedNoMembershipLoss;

fn policy_digest(rules: &[StreamingIssueThresholdRule]) -> ContentDigest {
    let mut hasher = blake3::Hasher::new();
    update_hash_field(&mut hasher, b"aiperf.streaming.issue-policy.v1");
    for rule in rules {
        update_hash_field(&mut hasher, rule.rule_id.as_str().as_bytes());
        update_hash_field(&mut hasher, scope_kind_tag(rule.scope));
        update_hash_field(&mut hasher, issue_class_tag(rule.class));
        update_hash_field(&mut hasher, &[u8::from(rule.code.is_some())]);
        if let Some(code) = &rule.code {
            update_hash_field(&mut hasher, code.as_str().as_bytes());
        }
        update_hash_field(&mut hasher, &rule.retry_limit.to_le_bytes());
        update_hash_field(&mut hasher, disposition_tag(rule.exhausted_disposition));
        update_hash_field(
            &mut hasher,
            &[u8::from(rule.admission_fence_count.is_some())],
        );
        if let Some(count) = rule.admission_fence_count {
            update_hash_field(&mut hasher, &count.get().to_le_bytes());
        }
    }
    ContentDigest::from_bytes(*hasher.finalize().as_bytes())
}

fn issue_id(
    issue: &OrdinaryStreamingIssue,
    terminal_invariant: Option<StreamingTerminalInvariant>,
) -> ContentDigest {
    let mut hasher = blake3::Hasher::new();
    update_hash_field(&mut hasher, b"aiperf.streaming.issue-receipt.v2");
    update_hash_field(&mut hasher, issue.run.logical_replay_run().as_bytes());
    hash_scope(&mut hasher, &issue.scope);
    update_hash_field(&mut hasher, issue_class_tag(issue.class));
    update_hash_field(&mut hasher, failure_stage_tag(issue.stage));
    update_hash_field(&mut hasher, issue.code.as_str().as_bytes());
    update_hash_field(&mut hasher, issue.semantic_context_digest.as_bytes());
    update_hash_field(&mut hasher, &[u8::from(issue.order.input_domain.is_some())]);
    if let Some(input_domain) = &issue.order.input_domain {
        update_hash_field(&mut hasher, input_domain.stream_identity.as_bytes());
        update_hash_field(&mut hasher, input_domain.source_identity.as_bytes());
    }
    update_hash_field(
        &mut hasher,
        &[u8::from(issue.order.source_position.is_some())],
    );
    if let Some(source_position) = issue.order.source_position {
        update_hash_field(&mut hasher, &source_position.get().to_le_bytes());
    }
    update_hash_field(
        &mut hasher,
        &[u8::from(issue.order.global_sequence.is_some())],
    );
    if let Some(global_sequence) = issue.order.global_sequence {
        update_hash_field(&mut hasher, &global_sequence.get().to_le_bytes());
    }
    update_hash_field(&mut hasher, &issue.order.retry_ordinal.to_le_bytes());
    update_hash_field(&mut hasher, issue.order.scope_tiebreaker.as_bytes());
    update_hash_field(&mut hasher, &[u8::from(terminal_invariant.is_some())]);
    if let Some(invariant) = terminal_invariant {
        update_hash_field(&mut hasher, terminal_invariant_tag(invariant));
    }
    ContentDigest::from_bytes(*hasher.finalize().as_bytes())
}

fn hash_scope(hasher: &mut blake3::Hasher, scope: &StreamingIssueScope) {
    update_hash_field(hasher, scope_kind_tag(scope.kind()));
    match scope {
        StreamingIssueScope::Run => {}
        StreamingIssueScope::Partition {
            input_domain,
            object,
        } => {
            update_hash_field(hasher, input_domain.stream_identity.as_bytes());
            update_hash_field(hasher, input_domain.source_identity.as_bytes());
            update_hash_field(hasher, object.as_bytes());
        }
        StreamingIssueScope::Record {
            input_domain,
            record_id,
        } => {
            update_hash_field(hasher, input_domain.stream_identity.as_bytes());
            update_hash_field(hasher, input_domain.source_identity.as_bytes());
            update_hash_field(hasher, record_id.as_bytes());
        }
        StreamingIssueScope::Session {
            input_domain,
            session_key,
        } => {
            update_hash_field(hasher, input_domain.stream_identity.as_bytes());
            update_hash_field(hasher, input_domain.source_identity.as_bytes());
            update_hash_field(hasher, session_key.as_bytes());
        }
        StreamingIssueScope::Action { action_id } => {
            update_hash_field(hasher, action_id.as_bytes());
        }
        StreamingIssueScope::Export {
            exporter_id,
            generation,
        } => {
            update_hash_field(hasher, exporter_id.as_str().as_bytes());
            update_hash_field(hasher, &generation.epoch().get().to_le_bytes());
            update_hash_field(hasher, generation.digest().as_bytes());
        }
        StreamingIssueScope::CheckpointAttempt {
            generation,
            attempt_ordinal,
        } => {
            update_hash_field(hasher, &generation.get().to_le_bytes());
            update_hash_field(hasher, &attempt_ordinal.to_le_bytes());
        }
    }
}

fn update_hash_field(hasher: &mut blake3::Hasher, field: &[u8]) {
    hasher.update(&(field.len() as u64).to_le_bytes());
    hasher.update(field);
}

const fn scope_kind_tag(scope: StreamingIssueScopeKind) -> &'static [u8] {
    match scope {
        StreamingIssueScopeKind::Run => b"run",
        StreamingIssueScopeKind::Partition => b"partition",
        StreamingIssueScopeKind::Record => b"record",
        StreamingIssueScopeKind::Session => b"session",
        StreamingIssueScopeKind::Action => b"action",
        StreamingIssueScopeKind::Export => b"export",
        StreamingIssueScopeKind::CheckpointAttempt => b"checkpoint_attempt",
    }
}

const fn issue_class_tag(class: StreamingIssueClass) -> &'static [u8] {
    match class {
        StreamingIssueClass::Retryable => b"retryable",
        StreamingIssueClass::Permanent => b"permanent",
        StreamingIssueClass::Invariant => b"invariant",
        StreamingIssueClass::Capacity => b"capacity",
    }
}

const fn disposition_tag(disposition: StreamingIssueDisposition) -> &'static [u8] {
    match disposition {
        StreamingIssueDisposition::Retry => b"retry",
        StreamingIssueDisposition::Backpressure => b"backpressure",
        StreamingIssueDisposition::Quarantine => b"quarantine",
        StreamingIssueDisposition::Hole => b"hole",
        StreamingIssueDisposition::Continue => b"continue",
        StreamingIssueDisposition::TerminalActionReceipt => b"terminal_action_receipt",
        StreamingIssueDisposition::ExportIncomplete => b"export_incomplete",
        StreamingIssueDisposition::FailRun => b"fail_run",
    }
}

const fn failure_stage_tag(stage: StreamingFailureStage) -> &'static [u8] {
    match stage {
        StreamingFailureStage::Source => b"source",
        StreamingFailureStage::Acquisition => b"acquisition",
        StreamingFailureStage::Decode => b"decode",
        StreamingFailureStage::Ordering => b"ordering",
        StreamingFailureStage::StateBudget => b"state_budget",
        StreamingFailureStage::Session => b"session",
        StreamingFailureStage::Placement => b"placement",
        StreamingFailureStage::Dispatch => b"dispatch",
        StreamingFailureStage::Checkpoint => b"checkpoint",
        StreamingFailureStage::Result => b"result",
    }
}

const fn terminal_invariant_tag(invariant: StreamingTerminalInvariant) -> &'static [u8] {
    terminal_invariant_code(invariant).as_bytes()
}

const fn terminal_invariant_code(invariant: StreamingTerminalInvariant) -> &'static str {
    match invariant {
        StreamingTerminalInvariant::RunAuthorityMismatch => "run_authority_mismatch",
        StreamingTerminalInvariant::SourceIdentityAuthorityMismatch => {
            "source_identity_authority_mismatch"
        }
        StreamingTerminalInvariant::PublicationProofMismatch => "publication_proof_mismatch",
        StreamingTerminalInvariant::WriterLeaseMismatch => "writer_lease_mismatch",
        StreamingTerminalInvariant::CasExpectationMismatch => "cas_expectation_mismatch",
        StreamingTerminalInvariant::SecurityAuthorityMismatch => "security_authority_mismatch",
        StreamingTerminalInvariant::ConflictingStableContent => "conflicting_stable_content",
        StreamingTerminalInvariant::ImpossibleTruthfulOrdering => "impossible_truthful_ordering",
        StreamingTerminalInvariant::ImpossibleTruthfulWatermark => "impossible_truthful_watermark",
        StreamingTerminalInvariant::ImpossibleTruthfulCut => "impossible_truthful_cut",
        StreamingTerminalInvariant::FrozenSemanticDrift => "frozen_semantic_drift",
        StreamingTerminalInvariant::AccountingCorruption => "accounting_corruption",
    }
}

#[cfg(test)]
mod tests {
    /// Exact bytes every reporter charges for its bounded submission ring
    /// buffer; test budgets are written relative to it so a change in
    /// `QueuedHandleIssue`'s layout cannot silently starve them.
    const QUEUE_CHARGE_BYTES: usize = MAX_QUEUED_SUBMISSIONS * size_of::<QueuedHandleIssue>();

    use super::*;
    use crate::streaming::{
        action::{
            ActionTerminalMembershipOutcomeView, CheckedActionFailureTerminalEvidence,
            CheckedActionTerminalMembership, FrozenActionInventory,
        },
        checkpoint::{
            AcquisitionHorizon, AdmissionHorizon, CheckpointCut, CheckpointGenerationCandidate,
            CheckpointGenerationPublicationProof, CheckpointParticipantPlan,
            CheckpointTerminalReason, DecodeHorizon, DiscoveryHorizon, EventTimeWatermark,
            OrderedActionHorizon, TerminalActionHorizon,
        },
        failure::{
            ActionExecutionError, ActionFailureCode, DecodeFailureCode, ResultExportError,
            ResultExportFailureCode, SessionCoordinatorError, SessionFailureCode,
            StreamFormatError,
        },
        identity::{LogicalReplayRunId, SessionCausalFrontier},
        session::CheckedSessionQuarantineTombstoneView,
        unit::EventTimeUtc,
    };

    fn component(value: &str) -> StreamingIssueComponentId {
        StreamingIssueComponentId::new(value)
            .unwrap_or_else(|error| panic!("valid component ID: {error}"))
    }

    fn record_issue() -> OrdinaryStreamingIssue {
        OrdinaryStreamingIssue::record(
            StreamRunIdentity::new(LogicalReplayRunId::from_bytes([0x11; 32])),
            StreamingInputDomainIdentity::new(
                ContentDigest::from_bytes([0x21; 32]),
                ImmutableObjectIdentity::from_bytes([0x20; 32]),
            ),
            StableRecordId::from_bytes([0x22; 32]),
            StreamingIssueClass::Permanent,
            ContentDigest::from_bytes([0x33; 32]),
            SourcePosition::new(7),
            0,
            ContentDigest::from_bytes([0x44; 32]),
            OrdinaryStreamingFailure::Format(StreamFormatError::decode(DecodeFailureCode::Syntax)),
        )
        .unwrap_or_else(|error| panic!("valid record issue: {error}"))
    }

    fn rule(id: &str, code: Option<&str>, retry_limit: u32) -> StreamingIssueThresholdRule {
        StreamingIssueThresholdRule::new(
            component(id),
            StreamingIssueScopeKind::Record,
            StreamingIssueClass::Permanent,
            code.map(component),
            retry_limit,
            StreamingIssueDisposition::Quarantine,
            None,
        )
        .unwrap_or_else(|error| panic!("valid rule: {error}"))
    }

    fn action_rule(
        id: &str,
        retry_limit: u32,
        exhausted: StreamingIssueDisposition,
    ) -> StreamingIssueThresholdRule {
        StreamingIssueThresholdRule::new(
            component(id),
            StreamingIssueScopeKind::Action,
            StreamingIssueClass::Permanent,
            None,
            retry_limit,
            exhausted,
            None,
        )
        .unwrap_or_else(|error| panic!("valid action rule: {error}"))
    }

    fn action_issue(sequence: u64, retry_ordinal: u32) -> OrdinaryStreamingIssue {
        OrdinaryStreamingIssue::action(
            StreamRunIdentity::new(LogicalReplayRunId::from_bytes([0x81; 32])),
            StableActionId::from_bytes([sequence as u8 + 1; 32]),
            StreamingIssueClass::Permanent,
            ContentDigest::from_bytes([0x82; 32]),
            GlobalSequence::new(sequence),
            retry_ordinal,
            ContentDigest::from_bytes([sequence as u8 + 3; 32]),
            OrdinaryStreamingFailure::Action(ActionExecutionError::action(
                ActionFailureCode::Endpoint,
            )),
        )
        .unwrap_or_else(|error| panic!("valid action issue: {error}"))
    }

    fn test_barrier(run: StreamRunIdentity, epoch: u64) -> CheckpointBarrier {
        let event_time =
            EventTimeUtc::new(10).unwrap_or_else(|error| panic!("valid event time: {error}"));
        CheckpointBarrier {
            run,
            epoch: CheckpointEpoch::new(epoch),
            cut: CheckpointCut {
                discovered: DiscoveryHorizon::new(SourcePosition::new(10)),
                acquired: AcquisitionHorizon::new(SourcePosition::new(10)),
                decoded: DecodeHorizon::new(SourcePosition::new(10)),
                ordered: OrderedActionHorizon::new(GlobalSequence::new(10)),
                admitted: AdmissionHorizon::new(GlobalSequence::new(10)),
                terminal: TerminalActionHorizon::new(GlobalSequence::new(10)),
                event_watermark: EventTimeWatermark::Hard {
                    through: event_time,
                },
                causal_frontier: SessionCausalFrontier {
                    through_sequence: GlobalSequence::new(10),
                    event_time: Some(event_time),
                    digest: ContentDigest::from_bytes([0xa1; 32]),
                },
                handled_issues: HandledIssueCut::empty(),
            },
            plan_digest: ContentDigest::from_bytes([0xa2; 32]),
        }
    }

    fn copy_rule(rule: &StreamingIssueThresholdRule) -> StreamingIssueThresholdRule {
        StreamingIssueThresholdRule {
            rule_id: rule.rule_id.clone(),
            scope: rule.scope,
            class: rule.class,
            code: rule.code.clone(),
            retry_limit: rule.retry_limit,
            exhausted_disposition: rule.exhausted_disposition,
            admission_fence_count: rule.admission_fence_count,
        }
    }

    fn ordinary_decision(
        policy: &PreparedStreamingIssuePolicy,
        issue: &OrdinaryStreamingIssue,
        prior_matching_count: u64,
    ) -> Result<(StreamingIssueDecision, StreamingIssueThresholdReceipt), StreamingReliabilityError>
    {
        let rule = policy.rule_for(issue)?;
        let resulting_matching_count = prior_matching_count
            .checked_add(1)
            .ok_or(StreamingReliabilityError::CounterOverflow)?;
        let is_exhausted = prior_matching_count >= u64::from(rule.retry_limit);
        let disposition = if is_exhausted {
            rule.exhausted_disposition
        } else {
            StreamingIssueDisposition::Retry
        };
        let needs_admission_fence = rule
            .admission_fence_count
            .is_some_and(|count| resulting_matching_count >= count.get());
        Ok((
            StreamingIssueDecision {
                disposition,
                rule: copy_rule(rule),
                needs_admission_fence,
            },
            StreamingIssueThresholdReceipt {
                policy_digest: policy.digest,
                rule_id: rule.rule_id.clone(),
                prior_matching_count,
                resulting_matching_count,
                retry_ordinal: issue.order.retry_ordinal,
                is_exhausted,
            },
        ))
    }

    fn counter_key(
        issue: &OrdinaryStreamingIssue,
        rule_id: StreamingIssueComponentId,
    ) -> StreamingIssueCounterKey {
        let domain = match &issue.scope {
            StreamingIssueScope::Run => StreamingIssueCounterDomain::Run,
            StreamingIssueScope::Partition { input_domain, .. }
            | StreamingIssueScope::Record { input_domain, .. }
            | StreamingIssueScope::Session { input_domain, .. } => {
                StreamingIssueCounterDomain::Input(input_domain.clone())
            }
            StreamingIssueScope::Action { .. } => StreamingIssueCounterDomain::Action,
            StreamingIssueScope::Export {
                exporter_id,
                generation,
            } => StreamingIssueCounterDomain::Export {
                exporter_id: exporter_id.clone(),
                generation: generation.clone(),
            },
            StreamingIssueScope::CheckpointAttempt { .. } => {
                StreamingIssueCounterDomain::CheckpointAttempt
            }
        };
        StreamingIssueCounterKey { domain, rule_id }
    }

    #[test]
    fn exhaustive_classifier_maps_only_the_terminal_boundary() {
        let cases = [
            (
                HostFailure::RunAuthorityMismatch,
                StreamingTerminalInvariant::RunAuthorityMismatch,
            ),
            (
                HostFailure::SourceIdentityAuthorityMismatch,
                StreamingTerminalInvariant::SourceIdentityAuthorityMismatch,
            ),
            (
                HostFailure::PublicationProofMismatch,
                StreamingTerminalInvariant::PublicationProofMismatch,
            ),
            (
                HostFailure::WriterLeaseMismatch,
                StreamingTerminalInvariant::WriterLeaseMismatch,
            ),
            (
                HostFailure::CasExpectationMismatch,
                StreamingTerminalInvariant::CasExpectationMismatch,
            ),
            (
                HostFailure::SecurityAuthorityMismatch,
                StreamingTerminalInvariant::SecurityAuthorityMismatch,
            ),
            (
                HostFailure::ConflictingStableContent,
                StreamingTerminalInvariant::ConflictingStableContent,
            ),
            (
                HostFailure::ImpossibleTruthfulOrdering,
                StreamingTerminalInvariant::ImpossibleTruthfulOrdering,
            ),
            (
                HostFailure::ImpossibleTruthfulWatermark,
                StreamingTerminalInvariant::ImpossibleTruthfulWatermark,
            ),
            (
                HostFailure::ImpossibleTruthfulCut,
                StreamingTerminalInvariant::ImpossibleTruthfulCut,
            ),
            (
                HostFailure::FrozenSemanticDrift,
                StreamingTerminalInvariant::FrozenSemanticDrift,
            ),
            (
                HostFailure::AccountingCorruption,
                StreamingTerminalInvariant::AccountingCorruption,
            ),
        ];
        for (failure, expected) in cases {
            let verified = classify_host_failure(failure);
            assert_eq!(verified.invariant, expected);
            assert_eq!(
                fail_run_decision(&verified).disposition,
                StreamingIssueDisposition::FailRun
            );
        }
    }

    #[test]
    fn complete_scope_class_disposition_product_is_closed() {
        let scopes = [
            StreamingIssueScopeKind::Run,
            StreamingIssueScopeKind::Partition,
            StreamingIssueScopeKind::Record,
            StreamingIssueScopeKind::Session,
            StreamingIssueScopeKind::Action,
            StreamingIssueScopeKind::Export,
            StreamingIssueScopeKind::CheckpointAttempt,
        ];
        let classes = [
            StreamingIssueClass::Retryable,
            StreamingIssueClass::Permanent,
            StreamingIssueClass::Invariant,
            StreamingIssueClass::Capacity,
        ];
        let dispositions = [
            StreamingIssueDisposition::Retry,
            StreamingIssueDisposition::Backpressure,
            StreamingIssueDisposition::Quarantine,
            StreamingIssueDisposition::Hole,
            StreamingIssueDisposition::Continue,
            StreamingIssueDisposition::TerminalActionReceipt,
            StreamingIssueDisposition::ExportIncomplete,
            StreamingIssueDisposition::FailRun,
        ];

        for scope in scopes {
            for class in classes {
                for disposition in dispositions {
                    let expected = if class == StreamingIssueClass::Invariant
                        || scope == StreamingIssueScopeKind::Run
                    {
                        false
                    } else {
                        match scope {
                            StreamingIssueScopeKind::Partition => matches!(
                                disposition,
                                StreamingIssueDisposition::Retry
                                    | StreamingIssueDisposition::Backpressure
                                    | StreamingIssueDisposition::Hole
                            ),
                            StreamingIssueScopeKind::Record | StreamingIssueScopeKind::Session => {
                                matches!(
                                    disposition,
                                    StreamingIssueDisposition::Retry
                                        | StreamingIssueDisposition::Backpressure
                                        | StreamingIssueDisposition::Quarantine
                                )
                            }
                            StreamingIssueScopeKind::Action => matches!(
                                disposition,
                                StreamingIssueDisposition::Retry
                                    | StreamingIssueDisposition::Backpressure
                                    | StreamingIssueDisposition::TerminalActionReceipt
                            ),
                            StreamingIssueScopeKind::Export => matches!(
                                disposition,
                                StreamingIssueDisposition::Retry
                                    | StreamingIssueDisposition::Backpressure
                                    | StreamingIssueDisposition::ExportIncomplete
                            ),
                            StreamingIssueScopeKind::CheckpointAttempt => matches!(
                                disposition,
                                StreamingIssueDisposition::Retry
                                    | StreamingIssueDisposition::Backpressure
                            ),
                            StreamingIssueScopeKind::Run => false,
                        }
                    };
                    assert_eq!(
                        is_allowed_authored_disposition(scope, class, disposition),
                        expected,
                        "scope={scope:?} class={class:?} disposition={disposition:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn checkpoint_attempt_order_requires_attempt_ordinal_equality() {
        let scope = StreamingIssueScope::CheckpointAttempt {
            generation: CheckpointEpoch::new(5),
            attempt_ordinal: 7,
        };
        let mismatched = StreamingIssueOrderKey::run(8, ContentDigest::from_bytes([0x55; 32]));

        assert!(!scope_order_matches(&scope, &mismatched));
    }

    #[test]
    fn ordinary_policy_never_constructs_fail_run_or_invariant() {
        assert!(
            StreamingIssueThresholdRule::new(
                component("terminal"),
                StreamingIssueScopeKind::Action,
                StreamingIssueClass::Permanent,
                None,
                0,
                StreamingIssueDisposition::FailRun,
                None,
            )
            .is_err()
        );
        let mut issue = record_issue();
        issue.class = StreamingIssueClass::Invariant;
        assert_eq!(
            OrdinaryStreamingIssue::new(
                issue.run,
                issue.scope,
                issue.class,
                issue.semantic_context_digest,
                issue.order,
                issue.failure,
            ),
            Err(StreamingIssueValidationError::InvariantIsHostOwned)
        );
    }

    #[test]
    fn threshold_counts_are_deterministic_and_domain_local() {
        let policy = PreparedStreamingIssuePolicy::new(vec![
            rule("syntax", Some("syntax"), 1),
            rule("record_default", None, 0),
        ])
        .unwrap_or_else(|error| panic!("valid policy: {error}"));
        let issue = record_issue();
        let (first, first_receipt) = ordinary_decision(&policy, &issue, 0)
            .unwrap_or_else(|error| panic!("first decision: {error}"));
        let (second, second_receipt) = ordinary_decision(&policy, &issue, 1)
            .unwrap_or_else(|error| panic!("second decision: {error}"));
        assert_eq!(first.disposition, StreamingIssueDisposition::Retry);
        assert!(!first_receipt.is_exhausted);
        assert_eq!(second.disposition, StreamingIssueDisposition::Quarantine);
        assert!(second_receipt.is_exhausted);

        let key = counter_key(&issue, component("syntax"));
        assert_eq!(
            key.domain(),
            &StreamingIssueCounterDomain::Input(
                issue
                    .scope
                    .input_domain()
                    .unwrap_or_else(|| panic!("record domain"))
                    .clone()
            )
        );

        let action = OrdinaryStreamingIssue::action(
            *issue.run(),
            StableActionId::from_bytes([0x66; 32]),
            StreamingIssueClass::Permanent,
            ContentDigest::from_bytes([0x67; 32]),
            GlobalSequence::new(9),
            0,
            ContentDigest::from_bytes([0x68; 32]),
            OrdinaryStreamingFailure::Action(ActionExecutionError::action(
                ActionFailureCode::Endpoint,
            )),
        )
        .unwrap_or_else(|error| panic!("valid action issue: {error}"));
        assert_eq!(
            counter_key(&action, component("action_default")).domain(),
            &StreamingIssueCounterDomain::Action
        );
    }

    #[test]
    fn strict_persisted_wire_rejects_unknown_fields() {
        let policy = PreparedStreamingIssuePolicy::new(vec![
            rule("syntax", Some("syntax"), 1),
            rule("record_default", None, 0),
        ])
        .unwrap_or_else(|error| panic!("valid policy: {error}"));
        let issue = record_issue();
        let (decision, threshold) = ordinary_decision(&policy, &issue, 0)
            .unwrap_or_else(|error| panic!("valid decision: {error}"));
        let receipt = PersistedStreamingIssueReceipt {
            wire_version: ISSUE_RECEIPT_WIRE_VERSION,
            issue_id: issue.issue_id(),
            run: *issue.run(),
            scope: issue.scope.clone(),
            class: issue.class,
            stage: issue.stage,
            code: issue.code.clone(),
            semantic_context_digest: issue.semantic_context_digest,
            order: issue.order.clone(),
            terminal_invariant: None,
            disposition: decision.disposition,
            threshold,
        };
        let value = serde_json::to_value(receipt)
            .unwrap_or_else(|error| panic!("serialize receipt: {error}"));

        let mut root_tamper = value.clone();
        root_tamper
            .as_object_mut()
            .unwrap_or_else(|| panic!("receipt object"))
            .insert("unexpected".to_owned(), serde_json::Value::Bool(true));
        assert!(serde_json::from_value::<PersistedStreamingIssueReceiptWire>(root_tamper).is_err());

        let mut threshold_tamper = value;
        threshold_tamper["threshold"]
            .as_object_mut()
            .unwrap_or_else(|| panic!("threshold object"))
            .insert("unexpected".to_owned(), serde_json::Value::Bool(true));
        assert!(
            serde_json::from_value::<PersistedStreamingIssueReceiptWire>(threshold_tamper).is_err()
        );
    }

    #[test]
    fn terminal_action_receipt_is_the_only_branch_with_failure_identity() {
        let retry = ActionFailureDisposition::Retry(PreparedActionRetry { retry_ordinal: 2 });
        let backpressure = ActionFailureDisposition::Backpressure(PreparedActionBackpressure {
            needs_admission_fence: true,
        });
        let terminal =
            ActionFailureDisposition::TerminalActionReceipt(PreparedActionFailureIdentity {
                run: StreamRunIdentity::new(LogicalReplayRunId::from_bytes([1; 32])),
                action_id: StableActionId::from_bytes([2; 32]),
                sequence: GlobalSequence::new(3),
                issue_id: ContentDigest::from_bytes([4; 32]),
                terminal_evidence_digest: ContentDigest::from_bytes([5; 32]),
            });

        assert!(matches!(retry, ActionFailureDisposition::Retry(_)));
        assert!(matches!(
            backpressure,
            ActionFailureDisposition::Backpressure(_)
        ));
        assert!(matches!(
            terminal,
            ActionFailureDisposition::TerminalActionReceipt(_)
        ));
    }

    #[test]
    fn checked_action_view_outcome_maps_without_open_terminal_construction() {
        let succeeded = ActionTerminalMembershipOutcomeView::Succeeded;
        let failed = ActionTerminalMembershipOutcomeView::Failed {
            issue_id: ContentDigest::from_bytes([7; 32]),
        };
        assert!(matches!(
            succeeded,
            ActionTerminalMembershipOutcomeView::Succeeded
        ));
        assert!(matches!(
            failed,
            ActionTerminalMembershipOutcomeView::Failed { .. }
        ));
    }

    #[test]
    fn failed_action_poll_is_ordered_idempotent_and_type_state_exact() {
        let action_budget = StreamingResourceBudget::new(super::super::budget::BudgetLimits {
            max_items: 65,
            max_bytes: QUEUE_CHARGE_BYTES + 64 * 1024,
        })
        .unwrap_or_else(|error| panic!("valid action budget: {error}"));
        let run = StreamRunIdentity::new(LogicalReplayRunId::from_bytes([0x81; 32]));
        let policy = PreparedStreamingIssuePolicy::new([action_rule(
            "action_default",
            0,
            StreamingIssueDisposition::TerminalActionReceipt,
        )])
        .unwrap_or_else(|error| panic!("valid action policy: {error}"));
        let mut reporter = BudgetOwnedStreamingIssueReporter::new(run, policy, action_budget)
            .unwrap_or_else(|error| panic!("budget-owned reporter: {error}"));

        let issue = action_issue(1, 0);
        let evidence = CheckedActionFailureTerminalEvidence::for_test(
            run,
            issue
                .scope()
                .action_id()
                .unwrap_or_else(|| panic!("action ID")),
            GlobalSequence::new(1),
            ContentDigest::from_bytes([0x91; 32]),
        );
        let queued = reporter
            .enqueue_failed_action(&evidence, issue)
            .unwrap_or_else(|error| panic!("queue action failure: {error}"));
        let queued = match reporter
            .poll_failed_action(queued)
            .unwrap_or_else(|error| panic!("poll blocked action: {error}"))
        {
            ActionFailureDisposition::Pending(queued) => queued,
            other => panic!("expected pending, got {other:?}"),
        };

        let success = CheckedActionTerminalMembership::for_test(
            run,
            StableActionId::from_bytes([1; 32]),
            GlobalSequence::new(0),
            ActionTerminalMembershipOutcomeView::Succeeded,
            ContentDigest::from_bytes([0x92; 32]),
        );
        let terminal = reporter
            .prepare_action_terminal(&success)
            .unwrap_or_else(|error| panic!("prepare success terminal: {error}"));
        futures::executor::block_on(
            reporter.report(IssueSequenceUpdate::CheckedActionTerminal(terminal)),
        )
        .unwrap_or_else(|error| panic!("record success terminal: {error}"));

        let failure = match reporter
            .poll_failed_action(queued)
            .unwrap_or_else(|error| panic!("poll ready action: {error}"))
        {
            ActionFailureDisposition::TerminalActionReceipt(failure) => failure,
            other => panic!("expected terminal failure, got {other:?}"),
        };
        assert_eq!(failure.sequence(), GlobalSequence::new(1));

        let replay_issue = action_issue(1, 0);
        let replay = reporter
            .enqueue_failed_action(&evidence, replay_issue)
            .unwrap_or_else(|error| panic!("requeue identical failure: {error}"));
        let replay_failure = match reporter
            .poll_failed_action(replay)
            .unwrap_or_else(|error| panic!("replay terminal failure: {error}"))
        {
            ActionFailureDisposition::TerminalActionReceipt(failure) => failure,
            other => panic!("expected replay terminal failure, got {other:?}"),
        };
        assert_eq!(replay_failure.issue_id(), failure.issue_id());
        assert_eq!(reporter.summary().unwrap().total, 1);
    }

    #[test]
    fn failed_action_retry_replay_then_terminal_attempt_counts_each_identity_once() {
        let action_budget = StreamingResourceBudget::new(super::super::budget::BudgetLimits {
            max_items: 97,
            max_bytes: QUEUE_CHARGE_BYTES + 96 * 1024,
        })
        .unwrap_or_else(|error| panic!("valid action budget: {error}"));
        let run = StreamRunIdentity::new(LogicalReplayRunId::from_bytes([0x81; 32]));
        let policy = PreparedStreamingIssuePolicy::new([action_rule(
            "action_default",
            1,
            StreamingIssueDisposition::TerminalActionReceipt,
        )])
        .unwrap_or_else(|error| panic!("valid action policy: {error}"));
        let mut reporter = BudgetOwnedStreamingIssueReporter::new(run, policy, action_budget)
            .unwrap_or_else(|error| panic!("budget-owned reporter: {error}"));

        let first_issue = action_issue(0, 0);
        let first_evidence = CheckedActionFailureTerminalEvidence::for_test(
            run,
            first_issue
                .scope()
                .action_id()
                .unwrap_or_else(|| panic!("action ID")),
            GlobalSequence::new(0),
            ContentDigest::from_bytes([0xa0; 32]),
        );
        let first = reporter
            .enqueue_failed_action(&first_evidence, first_issue)
            .unwrap_or_else(|error| panic!("queue first attempt: {error}"));
        let retry = match reporter
            .poll_failed_action(first)
            .unwrap_or_else(|error| panic!("poll first attempt: {error}"))
        {
            ActionFailureDisposition::Retry(retry) => retry,
            other => panic!("expected retry, got {other:?}"),
        };
        assert_eq!(retry.retry_ordinal(), 0);

        let replay = reporter
            .enqueue_failed_action(&first_evidence, action_issue(0, 0))
            .unwrap_or_else(|error| panic!("requeue first attempt: {error}"));
        assert!(matches!(
            reporter
                .poll_failed_action(replay)
                .unwrap_or_else(|error| panic!("replay first decision: {error}")),
            ActionFailureDisposition::Retry(_)
        ));

        let second_issue = action_issue(0, 1);
        let second_evidence = CheckedActionFailureTerminalEvidence::for_test(
            run,
            second_issue
                .scope()
                .action_id()
                .unwrap_or_else(|| panic!("action ID")),
            GlobalSequence::new(0),
            ContentDigest::from_bytes([0xa1; 32]),
        );
        let second = reporter
            .enqueue_failed_action(&second_evidence, second_issue)
            .unwrap_or_else(|error| panic!("queue second attempt: {error}"));
        let terminal = match reporter
            .poll_failed_action(second)
            .unwrap_or_else(|error| panic!("poll second attempt: {error}"))
        {
            ActionFailureDisposition::TerminalActionReceipt(terminal) => terminal,
            other => panic!("expected terminal receipt, got {other:?}"),
        };
        assert_eq!(terminal.sequence(), GlobalSequence::new(0));
        assert_eq!(reporter.summary().unwrap().total, 2);

        let old_retry = reporter
            .enqueue_failed_action(&first_evidence, action_issue(0, 0))
            .unwrap_or_else(|error| panic!("requeue old attempt: {error}"));
        assert!(matches!(
            reporter
                .poll_failed_action(old_retry)
                .unwrap_or_else(|error| panic!("replay old retry: {error}")),
            ActionFailureDisposition::Retry(_)
        ));
        assert_eq!(reporter.summary().unwrap().total, 2);
    }

    #[test]
    fn action_backpressure_and_gap_closure_never_mint_terminal_identity() {
        let action_budget = StreamingResourceBudget::new(super::super::budget::BudgetLimits {
            max_items: 65,
            max_bytes: QUEUE_CHARGE_BYTES + 64 * 1024,
        })
        .unwrap_or_else(|error| panic!("valid action budget: {error}"));
        let run = StreamRunIdentity::new(LogicalReplayRunId::from_bytes([0x81; 32]));
        let rule = StreamingIssueThresholdRule::new(
            component("action_default"),
            StreamingIssueScopeKind::Action,
            StreamingIssueClass::Permanent,
            None,
            0,
            StreamingIssueDisposition::Backpressure,
            NonZeroU64::new(1),
        )
        .unwrap_or_else(|error| panic!("valid action rule: {error}"));
        let policy = PreparedStreamingIssuePolicy::new([rule])
            .unwrap_or_else(|error| panic!("valid action policy: {error}"));
        let mut reporter = BudgetOwnedStreamingIssueReporter::new(run, policy, action_budget)
            .unwrap_or_else(|error| panic!("budget-owned reporter: {error}"));

        let issue = action_issue(2, 0);
        let evidence = CheckedActionFailureTerminalEvidence::for_test(
            run,
            issue
                .scope()
                .action_id()
                .unwrap_or_else(|| panic!("action ID")),
            GlobalSequence::new(2),
            ContentDigest::from_bytes([0xa2; 32]),
        );
        let queued = reporter
            .enqueue_failed_action(&evidence, issue)
            .unwrap_or_else(|error| panic!("queue action failure: {error}"));
        let queued = match reporter
            .poll_failed_action(queued)
            .unwrap_or_else(|error| panic!("poll blocked action: {error}"))
        {
            ActionFailureDisposition::Pending(queued) => queued,
            other => panic!("expected pending, got {other:?}"),
        };

        let inventory = FrozenActionInventory::for_test(
            run,
            GlobalSequence::new(1),
            ContentDigest::from_bytes([0xa3; 32]),
            BTreeMap::new(),
        );
        let closure = reporter
            .prepare_no_more_actions_before(&inventory, GlobalSequence::new(1))
            .unwrap_or_else(|error| panic!("prepare action gap closure: {error}"));
        futures::executor::block_on(
            reporter.report(IssueSequenceUpdate::CheckedNoMoreActionsBefore(closure)),
        )
        .unwrap_or_else(|error| panic!("record action gap closure: {error}"));

        let backpressure = match reporter
            .poll_failed_action(queued)
            .unwrap_or_else(|error| panic!("poll ready action: {error}"))
        {
            ActionFailureDisposition::Backpressure(backpressure) => backpressure,
            other => panic!("expected backpressure, got {other:?}"),
        };
        assert!(backpressure.needs_admission_fence());
        assert_eq!(reporter.summary().unwrap().total, 1);
    }

    #[test]
    fn synchronous_action_enqueue_refuses_immediately_without_advancing_state() {
        let action_budget = StreamingResourceBudget::new(super::super::budget::BudgetLimits {
            max_items: 5,
            max_bytes: QUEUE_CHARGE_BYTES + 64 * 1024,
        })
        .unwrap_or_else(|error| panic!("valid action budget: {error}"));
        let held = action_budget
            .try_acquire(2, 0)
            .unwrap_or_else(|error| panic!("hold reporter capacity: {error}"));
        let run = StreamRunIdentity::new(LogicalReplayRunId::from_bytes([0x81; 32]));
        let policy = PreparedStreamingIssuePolicy::new([action_rule(
            "action_default",
            0,
            StreamingIssueDisposition::TerminalActionReceipt,
        )])
        .unwrap_or_else(|error| panic!("valid action policy: {error}"));
        let mut reporter =
            BudgetOwnedStreamingIssueReporter::new(run, policy, action_budget.clone())
            .unwrap_or_else(|error| panic!("budget-owned reporter: {error}"));
        let issue = action_issue(0, 0);
        let evidence = CheckedActionFailureTerminalEvidence::for_test(
            run,
            issue
                .scope()
                .action_id()
                .unwrap_or_else(|| panic!("action ID")),
            GlobalSequence::new(0),
            ContentDigest::from_bytes([0xa4; 32]),
        );

        assert!(matches!(
            reporter.enqueue_failed_action(&evidence, issue),
            Err(StreamingReliabilityError::StateBudget(_))
        ));
        assert_eq!(reporter.next_reporter_token, 0);
        assert!(reporter.pending_actions.is_empty());
        assert!(reporter.current_action_attempts.is_empty());
        assert_eq!(reporter.summary().unwrap().total, 0);

        drop(held);
        let queued = reporter
            .enqueue_failed_action(&evidence, action_issue(0, 0))
            .unwrap_or_else(|error| panic!("queue after capacity returns: {error}"));
        assert_eq!(queued.reporter_token, 0);
    }

    #[test]
    fn quarantine_prepare_is_non_destructive_and_stale_revision_never_revalidates() {
        let run = StreamRunIdentity::new(LogicalReplayRunId::from_bytes([0xb1; 32]));
        let input_domain = StreamingInputDomainIdentity::new(
            ContentDigest::from_bytes([0xb2; 32]),
            ImmutableObjectIdentity::from_bytes([0xb3; 32]),
        );
        let policy = PreparedStreamingIssuePolicy::new([StreamingIssueThresholdRule::new(
            component("session_default"),
            StreamingIssueScopeKind::Session,
            StreamingIssueClass::Permanent,
            None,
            0,
            StreamingIssueDisposition::Quarantine,
            None,
        )
        .unwrap_or_else(|error| panic!("valid session rule: {error}"))])
        .unwrap_or_else(|error| panic!("valid session policy: {error}"));
        let reporter_budget = StreamingResourceBudget::new(super::super::budget::BudgetLimits {
            max_items: 65,
            max_bytes: QUEUE_CHARGE_BYTES + 64 * 1024,
        })
        .unwrap_or_else(|error| panic!("valid reporter budget: {error}"));
        let install_budget = StreamingResourceBudget::new(super::super::budget::BudgetLimits {
            max_items: 9,
            max_bytes: QUEUE_CHARGE_BYTES + 8 * 1024,
        })
        .unwrap_or_else(|error| panic!("valid install budget: {error}"));
        let mut reporter = BudgetOwnedStreamingIssueReporter::new(run, policy, reporter_budget)
            .unwrap_or_else(|error| panic!("budget-owned reporter: {error}"));
        let issue = OrdinaryStreamingIssue::session(
            run,
            input_domain.clone(),
            StableSessionKey::from_bytes([0xb4; 32]),
            StreamingIssueClass::Permanent,
            ContentDigest::from_bytes([0xb5; 32]),
            SourcePosition::new(7),
            0,
            ContentDigest::from_bytes([0xb6; 32]),
            OrdinaryStreamingFailure::Session(SessionCoordinatorError::session(
                SessionFailureCode::MissingPredecessor,
            )),
        )
        .unwrap_or_else(|error| panic!("valid session issue: {error}"));
        let issue_id = issue.issue_id();
        futures::executor::block_on(reporter.report(IssueSequenceUpdate::Issue(issue)))
            .unwrap_or_else(|error| panic!("retain session issue: {error}"));
        futures::executor::block_on(reporter.report(IssueSequenceUpdate::NoMoreBefore {
            input_domain,
            through: SourcePosition::new(7),
        }))
        .unwrap_or_else(|error| panic!("advance session issue: {error}"));

        let entries = b"canonical-tombstones";
        let root = ContentDigest::from_bytes(*blake3::hash(entries).as_bytes());
        let current = CheckedSessionQuarantineTombstoneView::for_test(run, root, 4, entries);
        let barrier = test_barrier(run, 3);
        let prepared = futures::executor::block_on(reporter.prepare_session_quarantine_install(
            &current,
            issue_id,
            &barrier,
            &install_budget,
        ))
        .unwrap_or_else(|error| panic!("prepare quarantine install: {error}"));
        reporter
            .verify_session_quarantine_install(&prepared, &current, &barrier)
            .unwrap_or_else(|error| panic!("verify current install: {error}"));
        assert_eq!(prepared.tombstone_root(), &root);
        assert_eq!(prepared.view_revision(), 4);
        assert_eq!(prepared.payload_bytes(), entries);

        let replayed_digest = CheckedSessionQuarantineTombstoneView::for_test(run, root, 5, entries);
        assert_eq!(
            reporter.verify_session_quarantine_install(&prepared, &replayed_digest, &barrier,),
            Err(StreamingReliabilityError::StaleQuarantineTombstoneView)
        );
        drop(prepared);
        assert_eq!(install_budget.snapshot().used_items, 0);
        assert_eq!(install_budget.snapshot().used_bytes, 0);
    }

    fn session_quarantine_policy() -> PreparedStreamingIssuePolicy {
        PreparedStreamingIssuePolicy::new([StreamingIssueThresholdRule::new(
            component("session_default"),
            StreamingIssueScopeKind::Session,
            StreamingIssueClass::Permanent,
            None,
            0,
            StreamingIssueDisposition::Quarantine,
            None,
        )
        .unwrap_or_else(|error| panic!("valid session rule: {error}"))])
        .unwrap_or_else(|error| panic!("valid session policy: {error}"))
    }

    fn session_quarantine_issue(
        run: StreamRunIdentity,
        input_domain: StreamingInputDomainIdentity,
        session: u8,
        position: u64,
    ) -> OrdinaryStreamingIssue {
        OrdinaryStreamingIssue::session(
            run,
            input_domain,
            StableSessionKey::from_bytes([session; 32]),
            StreamingIssueClass::Permanent,
            ContentDigest::from_bytes([0xb5; 32]),
            SourcePosition::new(position),
            0,
            ContentDigest::from_bytes([0xb6; 32]),
            OrdinaryStreamingFailure::Session(SessionCoordinatorError::session(
                SessionFailureCode::MissingPredecessor,
            )),
        )
        .unwrap_or_else(|error| panic!("valid session issue: {error}"))
    }

    fn quarantine_reporter_budget() -> StreamingResourceBudget {
        StreamingResourceBudget::new(super::super::budget::BudgetLimits {
            max_items: 65,
            max_bytes: QUEUE_CHARGE_BYTES + 64 * 1024,
        })
        .unwrap_or_else(|error| panic!("valid reporter budget: {error}"))
    }

    fn quarantine_install_budget() -> StreamingResourceBudget {
        StreamingResourceBudget::new(super::super::budget::BudgetLimits {
            max_items: 16,
            max_bytes: 16 * 1024,
        })
        .unwrap_or_else(|error| panic!("valid install budget: {error}"))
    }

    /// Seed one retained session-quarantine receipt and close its input domain.
    fn quarantine_seeded_reporter(
        run: StreamRunIdentity,
        input_domain: &StreamingInputDomainIdentity,
        session: u8,
        position: u64,
    ) -> (BudgetOwnedStreamingIssueReporter, ContentDigest) {
        let mut reporter =
            BudgetOwnedStreamingIssueReporter::new(run, session_quarantine_policy(), quarantine_reporter_budget())
                .unwrap_or_else(|error| panic!("budget-owned reporter: {error}"));
        let issue = session_quarantine_issue(run, input_domain.clone(), session, position);
        let issue_id = issue.issue_id();
        futures::executor::block_on(reporter.report(IssueSequenceUpdate::Issue(issue)))
            .unwrap_or_else(|error| panic!("retain session issue: {error}"));
        futures::executor::block_on(reporter.report(IssueSequenceUpdate::NoMoreBefore {
            input_domain: input_domain.clone(),
            through: SourcePosition::new(position),
        }))
        .unwrap_or_else(|error| panic!("advance session issue: {error}"));
        (reporter, issue_id)
    }

    #[test]
    fn fresh_ledger_cut_is_byte_identical_to_empty() {
        let run = StreamRunIdentity::new(LogicalReplayRunId::from_bytes([0xda; 32]));
        let mut reporter = BudgetOwnedStreamingIssueReporter::new(
            run,
            session_quarantine_policy(),
            quarantine_reporter_budget(),
        )
        .unwrap_or_else(|error| panic!("budget-owned reporter: {error}"));
        let barrier = test_barrier(run, 1);
        let view = futures::executor::block_on(reporter.receipt_partition_view(&barrier))
            .unwrap_or_else(|error| panic!("prepare partition view: {error}"));
        assert_eq!(view.handled_cut(), &HandledIssueCut::empty());
    }

    #[test]
    fn revalidated_acknowledgement_emits_accepted_tombstone_root() {
        let run = StreamRunIdentity::new(LogicalReplayRunId::from_bytes([0xd1; 32]));
        let input_domain = StreamingInputDomainIdentity::new(
            ContentDigest::from_bytes([0xd2; 32]),
            ImmutableObjectIdentity::from_bytes([0xd3; 32]),
        );
        let install_budget = quarantine_install_budget();
        let (mut reporter, issue_id) = quarantine_seeded_reporter(run, &input_domain, 0xb4, 7);

        let entries = b"canonical-tombstones";
        let root = ContentDigest::from_bytes(*blake3::hash(entries).as_bytes());
        let view = CheckedSessionQuarantineTombstoneView::for_test(run, root, 4, entries);
        let barrier = test_barrier(run, 3);
        let prepared = futures::executor::block_on(reporter.prepare_session_quarantine_install(
            &view,
            issue_id,
            &barrier,
            &install_budget,
        ))
        .unwrap_or_else(|error| panic!("prepare quarantine install: {error}"));
        let binding_root = *prepared.receipt_binding_root();
        assert_eq!(
            futures::executor::block_on(
                reporter.report(IssueSequenceUpdate::PreparedSessionQuarantineInstall(
                    prepared
                ))
            ),
            Ok(None)
        );
        assert!(reporter.accepted_quarantine_install().is_some());

        let partition = futures::executor::block_on(reporter.receipt_partition_view(&barrier))
            .unwrap_or_else(|error| panic!("prepare partition view: {error}"));

        let mut hasher = blake3::Hasher::new();
        update_hash_field(&mut hasher, QUARANTINE_TOMBSTONE_ROOT_DOMAIN);
        update_hash_field(&mut hasher, root.as_bytes());
        update_hash_field(&mut hasher, &4_u64.to_le_bytes());
        update_hash_field(&mut hasher, binding_root.as_bytes());
        update_hash_field(&mut hasher, root.as_bytes());
        let expected = ContentDigest::from_bytes(*hasher.finalize().as_bytes());

        assert_eq!(
            partition.handled_cut().quarantine_tombstone_root(),
            &expected
        );
        // The checkpoint wave binds its candidate equality to this root, so the
        // ledger proves here that the value is real rather than a second empty.
        assert_ne!(
            partition.handled_cut().quarantine_tombstone_root(),
            HandledIssueCut::empty().quarantine_tombstone_root()
        );
        assert_eq!(partition.handled_cut().receipt_root(), partition.receipt_root());
    }

    #[test]
    fn receipt_root_drift_invalidates_the_accepted_quarantine_acknowledgement() {
        let run = StreamRunIdentity::new(LogicalReplayRunId::from_bytes([0xd4; 32]));
        let input_domain = StreamingInputDomainIdentity::new(
            ContentDigest::from_bytes([0xd5; 32]),
            ImmutableObjectIdentity::from_bytes([0xd6; 32]),
        );
        let install_budget = quarantine_install_budget();
        let (mut reporter, issue_id) = quarantine_seeded_reporter(run, &input_domain, 0xb4, 7);

        let entries = b"canonical-tombstones";
        let root = ContentDigest::from_bytes(*blake3::hash(entries).as_bytes());
        let view = CheckedSessionQuarantineTombstoneView::for_test(run, root, 4, entries);
        let barrier = test_barrier(run, 3);
        let prepared = futures::executor::block_on(reporter.prepare_session_quarantine_install(
            &view,
            issue_id,
            &barrier,
            &install_budget,
        ))
        .unwrap_or_else(|error| panic!("prepare quarantine install: {error}"));
        futures::executor::block_on(
            reporter.report(IssueSequenceUpdate::PreparedSessionQuarantineInstall(
                prepared,
            )),
        )
        .unwrap_or_else(|error| panic!("accept quarantine install: {error}"));

        // A later detailed receipt moves the receipt root, so the acknowledgement
        // and the receipt set can no longer commit together.
        futures::executor::block_on(reporter.report(IssueSequenceUpdate::Issue(
            session_quarantine_issue(run, input_domain.clone(), 0xb7, 9),
        )))
        .unwrap_or_else(|error| panic!("retain second session issue: {error}"));
        futures::executor::block_on(reporter.report(IssueSequenceUpdate::NoMoreBefore {
            input_domain,
            through: SourcePosition::new(9),
        }))
        .unwrap_or_else(|error| panic!("advance second session issue: {error}"));

        let before = install_budget.snapshot();
        assert_eq!(
            futures::executor::block_on(reporter.receipt_partition_view(&barrier))
                .err()
                .unwrap_or_else(|| panic!("stale acknowledgement must refuse the cut")),
            StreamingReliabilityError::StaleQuarantineTombstoneView
        );
        assert_eq!(reporter.retained_receipt_count(), 2);
        assert!(reporter.accepted_quarantine_install().is_some());
        assert_eq!(install_budget.snapshot().used_items, before.used_items);
        assert_eq!(install_budget.snapshot().used_bytes, before.used_bytes);
    }

    #[test]
    fn quarantine_acknowledgement_is_move_only() {
        let run = StreamRunIdentity::new(LogicalReplayRunId::from_bytes([0xd7; 32]));
        let input_domain = StreamingInputDomainIdentity::new(
            ContentDigest::from_bytes([0xd8; 32]),
            ImmutableObjectIdentity::from_bytes([0xd9; 32]),
        );
        let install_budget = quarantine_install_budget();
        let (mut reporter, issue_id) = quarantine_seeded_reporter(run, &input_domain, 0xb4, 7);

        let entries = b"canonical-tombstones";
        let root = ContentDigest::from_bytes(*blake3::hash(entries).as_bytes());
        let barrier = test_barrier(run, 3);
        let current = CheckedSessionQuarantineTombstoneView::for_test(run, root, 5, entries);
        let accepted = futures::executor::block_on(reporter.prepare_session_quarantine_install(
            &current,
            issue_id,
            &barrier,
            &install_budget,
        ))
        .unwrap_or_else(|error| panic!("prepare current install: {error}"));
        futures::executor::block_on(
            reporter.report(IssueSequenceUpdate::PreparedSessionQuarantineInstall(
                accepted,
            )),
        )
        .unwrap_or_else(|error| panic!("accept current install: {error}"));
        let accepted_root =
            *futures::executor::block_on(reporter.receipt_partition_view(&barrier))
                .unwrap_or_else(|error| panic!("prepare partition view: {error}"))
                .handled_cut()
                .quarantine_tombstone_root();
        let retained = install_budget.snapshot();

        let regressed_view = CheckedSessionQuarantineTombstoneView::for_test(run, root, 4, entries);
        let regressed = futures::executor::block_on(reporter.prepare_session_quarantine_install(
            &regressed_view,
            issue_id,
            &barrier,
            &install_budget,
        ))
        .unwrap_or_else(|error| panic!("prepare regressed install: {error}"));
        assert_eq!(
            futures::executor::block_on(reporter.report(
                IssueSequenceUpdate::PreparedSessionQuarantineInstall(regressed)
            )),
            Err(StreamingReliabilityError::StaleQuarantineTombstoneView)
        );

        // The refused acknowledgement was moved into `report` and dropped there,
        // releasing exactly its own payload and view leases.
        assert_eq!(install_budget.snapshot().used_items, retained.used_items);
        assert_eq!(install_budget.snapshot().used_bytes, retained.used_bytes);

        let after = futures::executor::block_on(reporter.receipt_partition_view(&barrier))
            .unwrap_or_else(|error| panic!("prepare partition view after refusal: {error}"));
        assert_eq!(
            after.handled_cut().quarantine_tombstone_root(),
            &accepted_root
        );
    }

    #[test]
    fn receipt_partition_view_acquires_before_serializing() {
        let run = StreamRunIdentity::new(LogicalReplayRunId::from_bytes([0xe5; 32]));
        let budget = quarantine_reporter_budget();
        let mut reporter = BudgetOwnedStreamingIssueReporter::new(
            run,
            session_quarantine_policy(),
            budget.clone(),
        )
        .unwrap_or_else(|error| panic!("budget-owned reporter: {error}"));
        let barrier = test_barrier(run, 2);
        let view = futures::executor::block_on(reporter.receipt_partition_view(&barrier))
            .unwrap_or_else(|error| panic!("prepare partition view: {error}"));

        // The reporter's only prior charge is its exact submission-queue lease,
        // so the peak equals the retained total exactly when — and only when —
        // one aggregate acquisition precedes serialization and is split down to
        // the realized payload and view leases with nothing over-acquired.
        let snapshot = budget.snapshot();
        assert_eq!(
            snapshot.used_bytes,
            QUEUE_CHARGE_BYTES + view.payload_charge_bytes() + view.view_charge_bytes()
        );
        assert_eq!(snapshot.high_water_bytes, snapshot.used_bytes);
        assert_eq!(snapshot.high_water_items, snapshot.used_items);
        assert_eq!(view.payload_charge_bytes(), view.payload_bytes().len());
    }

    #[test]
    fn export_failure_acquires_aggregate_before_allocation() {
        let run = StreamRunIdentity::new(LogicalReplayRunId::from_bytes([0xe6; 32]));
        let generation = CheckpointGeneration::new(
            CheckpointEpoch::new(2),
            ContentDigest::from_bytes([0xe7; 32]),
        );
        let sink_id = component("native_report");
        let reporter_budget = StreamingResourceBudget::new(super::super::budget::BudgetLimits {
            max_items: 17,
            max_bytes: QUEUE_CHARGE_BYTES + 32 * 1024,
        })
        .unwrap_or_else(|error| panic!("valid reporter budget: {error}"));
        let export_budget = StreamingResourceBudget::new(super::super::budget::BudgetLimits {
            max_items: 9,
            max_bytes: 32 * 1024,
        })
        .unwrap_or_else(|error| panic!("valid export budget: {error}"));
        let mut reporter = BudgetOwnedStreamingIssueReporter::new(
            run,
            export_recomputation_policy(),
            reporter_budget,
        )
        .unwrap_or_else(|error| panic!("budget-owned reporter: {error}"));
        let issue = OrdinaryStreamingIssue::export(
            run,
            sink_id.clone(),
            generation.clone(),
            StreamingIssueClass::Permanent,
            ContentDigest::from_bytes([0xc3; 32]),
            0,
            ContentDigest::from_bytes([0xc4; 32]),
            OrdinaryStreamingFailure::Export(ResultExportError::failure(
                ResultExportFailureCode::Attempt,
            )),
        )
        .unwrap_or_else(|error| panic!("valid export issue: {error}"));
        let prepared = futures::executor::block_on(reporter.prepare_export_attempt_failure(
            &run,
            &generation,
            &sink_id,
            0,
            ResultSinkAttemptOutcome::Failed(issue),
            &export_budget,
        ))
        .unwrap_or_else(|error| panic!("prepare export failure: {error}"));

        // The export budget is touched by this preparation alone, so its peak is
        // the aggregate acquisition and its retained total is the exact split.
        let expected = prepared.receipt.encoded.charged_bytes()
            + prepared.receipt.parsed_charge_bytes();
        let snapshot = export_budget.snapshot();
        assert_eq!(snapshot.used_items, 2);
        assert_eq!(snapshot.used_bytes, expected);
        assert_eq!(snapshot.high_water_bytes, expected);
        assert_eq!(
            prepared.receipt.encoded.charged_bytes(),
            prepared.receipt.encoded.as_bytes().len()
        );
        drop(prepared);
        assert_eq!(export_budget.snapshot().used_bytes, 0);
    }

    #[test]
    fn quarantine_install_acquires_before_payload_copy() {
        let run = StreamRunIdentity::new(LogicalReplayRunId::from_bytes([0xe8; 32]));
        let input_domain = StreamingInputDomainIdentity::new(
            ContentDigest::from_bytes([0xe9; 32]),
            ImmutableObjectIdentity::from_bytes([0xea; 32]),
        );
        let (mut reporter, issue_id) = quarantine_seeded_reporter(run, &input_domain, 0xb4, 7);
        let entries = b"canonical-tombstones";
        let root = ContentDigest::from_bytes(*blake3::hash(entries).as_bytes());
        let view = CheckedSessionQuarantineTombstoneView::for_test(run, root, 4, entries);
        let barrier = test_barrier(run, 3);

        // A budget that cannot admit the aggregate refuses with a typed
        // install-budget failure rather than panicking, and retains nothing.
        let starved = StreamingResourceBudget::new(super::super::budget::BudgetLimits {
            max_items: 2,
            max_bytes: entries.len(),
        })
        .unwrap_or_else(|error| panic!("valid starved budget: {error}"));
        assert!(matches!(
            futures::executor::block_on(reporter.prepare_session_quarantine_install(
                &view,
                issue_id,
                &barrier,
                &starved,
            )),
            Err(StreamingReliabilityError::QuarantineInstallBudget(_))
        ));
        assert_eq!(starved.snapshot().used_items, 0);
        assert_eq!(starved.snapshot().used_bytes, 0);

        let install_budget = quarantine_install_budget();
        let prepared = futures::executor::block_on(reporter.prepare_session_quarantine_install(
            &view,
            issue_id,
            &barrier,
            &install_budget,
        ))
        .unwrap_or_else(|error| panic!("prepare quarantine install: {error}"));
        let expected = prepared.payload_charge_bytes() + prepared.view_charge_bytes();
        let snapshot = install_budget.snapshot();
        assert_eq!(snapshot.used_items, 2);
        assert_eq!(snapshot.used_bytes, expected);
        assert_eq!(snapshot.high_water_bytes, expected);
        assert_eq!(prepared.payload_charge_bytes(), entries.len());
        drop(prepared);
        assert_eq!(install_budget.snapshot().used_bytes, 0);
    }

    /// Mint one committed, final, participant-free checkpoint generation.
    fn committed_final_generation(
        run: StreamRunIdentity,
        epoch: u64,
    ) -> CommittedCheckpointGeneration {
        committed_generation_with_finality(run, epoch, true)
    }

    fn committed_generation_with_finality(
        run: StreamRunIdentity,
        epoch: u64,
        is_final: bool,
    ) -> CommittedCheckpointGeneration {
        let event_time =
            EventTimeUtc::new(1).unwrap_or_else(|error| panic!("valid event time: {error}"));
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
        let plan = CheckpointParticipantPlan::new([])
            .unwrap_or_else(|error| panic!("valid empty participant plan: {error}"));
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
            is_final,
            is_final.then_some(CheckpointTerminalReason::Completed),
        )
        .unwrap_or_else(|error| panic!("valid generation candidate: {error}"));
        let proof = CheckpointGenerationPublicationProof::for_generation(candidate.generation());
        candidate
            .promote(
                &run,
                &plan,
                &ContentDigest::from_bytes([0xe2; 32]),
                &ContentDigest::from_bytes([0xe3; 32]),
                proof,
            )
            .unwrap_or_else(|error| panic!("promote committed generation: {error}"))
    }

    /// Build the two-class export policy used by the recomputation tests.
    ///
    /// `Permanent` exhausts on the first attempt; `Retryable` allows three
    /// retries. A tampered class therefore selects a different rule, which is the
    /// property the class-tampering test relies on.
    fn export_recomputation_policy() -> PreparedStreamingIssuePolicy {
        PreparedStreamingIssuePolicy::new([
            StreamingIssueThresholdRule::new(
                component("export_permanent"),
                StreamingIssueScopeKind::Export,
                StreamingIssueClass::Permanent,
                None,
                0,
                StreamingIssueDisposition::ExportIncomplete,
                None,
            )
            .unwrap_or_else(|error| panic!("valid permanent export rule: {error}")),
            StreamingIssueThresholdRule::new(
                component("export_retryable"),
                StreamingIssueScopeKind::Export,
                StreamingIssueClass::Retryable,
                None,
                3,
                StreamingIssueDisposition::ExportIncomplete,
                None,
            )
            .unwrap_or_else(|error| panic!("valid retryable export rule: {error}")),
        ])
        .unwrap_or_else(|error| panic!("valid export policy: {error}"))
    }

    /// Charge and wrap durable bytes for one restore attempt.
    fn charged_bytes(budget: &StreamingResourceBudget, bytes: &[u8]) -> BudgetedCheckpointBytes {
        let lease = futures::executor::block_on(budget.acquire(1, bytes.len()))
            .unwrap_or_else(|error| panic!("charge durable receipt: {error}"));
        BudgetedCheckpointBytes::new(Bytes::copy_from_slice(bytes), lease)
            .unwrap_or_else(|error| panic!("valid durable receipt bytes: {error}"))
    }

    fn export_restore_budget() -> StreamingResourceBudget {
        StreamingResourceBudget::new(super::super::budget::BudgetLimits {
            max_items: 4,
            max_bytes: 32 * 1024,
        })
        .unwrap_or_else(|error| panic!("valid restore budget: {error}"))
    }

    /// Re-encode a mutated wire document and derive its matching reference.
    fn reencode_tampered(
        wire: PersistedExportIssueReceiptWire,
    ) -> (Vec<u8>, DerivedExportReceiptReference) {
        let embedded_receipt = persisted_receipt_from_wire(wire.embedded_receipt);
        let embedded_encoded = serde_json::to_vec(&embedded_receipt)
            .unwrap_or_else(|error| panic!("encode tampered embedded receipt: {error}"));
        let embedded_receipt_digest =
            ContentDigest::from_bytes(*blake3::hash(&embedded_encoded).as_bytes());
        let embedded_receipt_length = embedded_encoded.len() as u64;
        let tampered_receipt = PersistedExportIssueReceipt {
            wire_version: wire.wire_version,
            run: wire.run,
            generation: wire.generation,
            sink_id: wire.sink_id,
            attempt_ordinal: wire.attempt_ordinal,
            issue_id: wire.issue_id,
            policy_digest: wire.policy_digest,
            counter_before: wire.counter_before,
            counter_after: wire.counter_after,
            embedded_receipt_digest,
            embedded_receipt_length,
            embedded_receipt,
        };
        let encoded = serde_json::to_vec(&tampered_receipt)
            .unwrap_or_else(|error| panic!("encode tampered export receipt: {error}"));
        let reference = DerivedExportReceiptReference::from_status_fields(
            ContentDigest::from_bytes(*blake3::hash(&encoded).as_bytes()),
            encoded.len() as u64,
            embedded_receipt_digest,
            embedded_receipt_length,
        );
        (encoded, reference)
    }

    /// Prepare one export failure through the forward path and return its
    /// durable bytes, status reference, and deterministic identity.
    fn prepared_export_bytes(
        run: StreamRunIdentity,
        generation: &CheckpointGeneration,
        sink_id: &StreamingIssueComponentId,
        class: StreamingIssueClass,
        attempt_ordinal: u32,
    ) -> (Vec<u8>, DerivedExportReceiptReference, ContentDigest) {
        let reporter_budget = StreamingResourceBudget::new(super::super::budget::BudgetLimits {
            max_items: 17,
            max_bytes: QUEUE_CHARGE_BYTES + 32 * 1024,
        })
        .unwrap_or_else(|error| panic!("valid reporter budget: {error}"));
        let export_budget = StreamingResourceBudget::new(super::super::budget::BudgetLimits {
            max_items: 9,
            max_bytes: 32 * 1024,
        })
        .unwrap_or_else(|error| panic!("valid export budget: {error}"));
        let mut reporter = BudgetOwnedStreamingIssueReporter::new(
            run,
            export_recomputation_policy(),
            reporter_budget,
        )
        .unwrap_or_else(|error| panic!("budget-owned reporter: {error}"));
        let issue = OrdinaryStreamingIssue::export(
            run,
            sink_id.clone(),
            generation.clone(),
            class,
            ContentDigest::from_bytes([0xc3; 32]),
            attempt_ordinal,
            ContentDigest::from_bytes([0xc4; 32]),
            OrdinaryStreamingFailure::Export(ResultExportError::failure(
                ResultExportFailureCode::Attempt,
            )),
        )
        .unwrap_or_else(|error| panic!("valid export issue: {error}"));
        let prepared = futures::executor::block_on(reporter.prepare_export_attempt_failure(
            &run,
            generation,
            sink_id,
            attempt_ordinal,
            ResultSinkAttemptOutcome::Failed(issue),
            &export_budget,
        ))
        .unwrap_or_else(|error| panic!("prepare export failure: {error}"));
        let issue_id = prepared.issue_id();
        let reference = prepared.receipt_reference().clone();
        let persistence = prepared.into_persistence();
        let bytes = persistence.encoded_bytes().to_vec();
        drop(persistence);
        assert_eq!(export_budget.snapshot().used_items, 0);
        (bytes, reference, issue_id)
    }

    /// Mint the verified status and validation context for one restore attempt.
    fn export_restore_context<'policy>(
        committed: &CommittedCheckpointGeneration,
        policy: &'policy PreparedStreamingIssuePolicy,
        sink_id: StreamingIssueComponentId,
        attempt_ordinal: u32,
        reference: DerivedExportReceiptReference,
    ) -> DurableExportReceiptValidationContext<'policy> {
        let status = VerifiedDerivedSinkAttemptStatus::from_status_owner(
            committed,
            sink_id,
            attempt_ordinal,
            u64::from(attempt_ordinal),
            reference,
        )
        .unwrap_or_else(|error| panic!("verified status: {error}"));
        DurableExportReceiptValidationContext::from_final_generation_status(
            committed, policy, &status,
        )
        .unwrap_or_else(|error| panic!("mint validation context: {error}"))
    }

    #[test]
    fn export_restore_recomputes_policy_decision() {
        let run = StreamRunIdentity::new(LogicalReplayRunId::from_bytes([0xc1; 32]));
        let committed = committed_final_generation(run, 9);
        let generation = committed.generation();
        let sink_id = component("native_report");
        let (bytes, reference, issue_id) =
            prepared_export_bytes(run, &generation, &sink_id, StreamingIssueClass::Permanent, 0);

        let policy = export_recomputation_policy();
        let context =
            export_restore_context(&committed, &policy, sink_id, 0, reference.clone());
        let budget = export_restore_budget();
        let restored = futures::executor::block_on(restore_durable_export_issue_receipt(
            charged_bytes(&budget, &bytes),
            &context,
            &budget,
        ))
        .unwrap_or_else(|error| panic!("restore durable export receipt: {error}"));

        assert_eq!(restored.issue_id(), issue_id);
        assert_eq!(
            restored.encoded_charge_bytes(),
            reference.receipt_length() as usize
        );
        let embedded = &restored.receipt.embedded_receipt;
        assert_eq!(embedded.threshold.rule_id, component("export_permanent"));
        assert!(embedded.threshold.is_exhausted);
        assert_eq!(
            embedded.disposition,
            StreamingIssueDisposition::ExportIncomplete
        );
        assert_eq!(embedded.threshold.prior_matching_count, 0);
        assert_eq!(embedded.threshold.resulting_matching_count, 1);
        assert_eq!(budget.snapshot().used_items, 2);
        drop(restored);
        assert_eq!(budget.snapshot().used_items, 0);
    }

    #[test]
    fn export_restore_rejects_status_fields_that_fail_verification() {
        let run = StreamRunIdentity::new(LogicalReplayRunId::from_bytes([0xc5; 32]));
        let committed = committed_final_generation(run, 4);
        let generation = committed.generation();
        let sink_id = component("native_report");
        let (bytes, _, _) =
            prepared_export_bytes(run, &generation, &sink_id, StreamingIssueClass::Permanent, 0);
        let policy = export_recomputation_policy();
        let budget = export_restore_budget();

        let decode = || -> PersistedExportIssueReceiptWire {
            serde_json::from_slice(&bytes)
                .unwrap_or_else(|error| panic!("decode test export receipt: {error}"))
        };

        // A rule the frozen policy never selects for these facts.
        let mut wire = decode();
        wire.embedded_receipt.threshold.rule_id = component("export_retryable");
        let (tampered, tampered_reference) = reencode_tampered(wire);
        let context =
            export_restore_context(&committed, &policy, sink_id.clone(), 0, tampered_reference);
        assert_eq!(
            futures::executor::block_on(restore_durable_export_issue_receipt(
                charged_bytes(&budget, &tampered),
                &context,
                &budget,
            ))
            .err(),
            Some(StreamingReliabilityError::ExportReceiptRuleMismatch)
        );
        assert_eq!(budget.snapshot().used_items, 0);

        // A class that selects a different rule, and so a different rule identity.
        let mut wire = decode();
        wire.embedded_receipt.class = StreamingIssueClass::Retryable;
        let (tampered, tampered_reference) = reencode_tampered(wire);
        let context =
            export_restore_context(&committed, &policy, sink_id.clone(), 0, tampered_reference);
        assert_eq!(
            futures::executor::block_on(restore_durable_export_issue_receipt(
                charged_bytes(&budget, &tampered),
                &context,
                &budget,
            ))
            .err(),
            Some(StreamingReliabilityError::ExportReceiptRuleMismatch)
        );
        assert_eq!(budget.snapshot().used_items, 0);

        // A host-owned class no ordinary issue can carry.
        let mut wire = decode();
        wire.embedded_receipt.class = StreamingIssueClass::Invariant;
        let (tampered, tampered_reference) = reencode_tampered(wire);
        let context =
            export_restore_context(&committed, &policy, sink_id.clone(), 0, tampered_reference);
        assert_eq!(
            futures::executor::block_on(restore_durable_export_issue_receipt(
                charged_bytes(&budget, &tampered),
                &context,
                &budget,
            ))
            .err(),
            Some(StreamingReliabilityError::ExportReceiptClassCodeMismatch)
        );
        assert_eq!(budget.snapshot().used_items, 0);

        // A code no export failure constructor can produce.
        let mut wire = decode();
        wire.embedded_receipt.code = component("result_export_forged");
        let (tampered, tampered_reference) = reencode_tampered(wire);
        let context = export_restore_context(&committed, &policy, sink_id, 0, tampered_reference);
        assert_eq!(
            futures::executor::block_on(restore_durable_export_issue_receipt(
                charged_bytes(&budget, &tampered),
                &context,
                &budget,
            ))
            .err(),
            Some(StreamingReliabilityError::ExportReceiptFailureUnrepresentable)
        );
        assert_eq!(budget.snapshot().used_items, 0);
    }

    #[test]
    fn export_restore_recomputes_retry_limit_and_exhaustion_exactly() {
        let run = StreamRunIdentity::new(LogicalReplayRunId::from_bytes([0xd1; 32]));
        let committed = committed_final_generation(run, 7);
        let generation = committed.generation();
        let sink_id = component("native_report");
        let policy = export_recomputation_policy();
        let budget = export_restore_budget();

        // Retryable rule, limit three: ordinal one is below the limit, so the
        // recomputed decision is a non-exhausted retry.
        let (bytes, reference, issue_id) =
            prepared_export_bytes(run, &generation, &sink_id, StreamingIssueClass::Retryable, 1);
        let context = export_restore_context(&committed, &policy, sink_id.clone(), 1, reference);
        let restored = futures::executor::block_on(restore_durable_export_issue_receipt(
            charged_bytes(&budget, &bytes),
            &context,
            &budget,
        ))
        .unwrap_or_else(|error| panic!("restore retryable export receipt: {error}"));
        assert_eq!(restored.issue_id(), issue_id);
        {
            let embedded = &restored.receipt.embedded_receipt;
            assert!(!embedded.threshold.is_exhausted);
            assert_eq!(embedded.disposition, StreamingIssueDisposition::Retry);
            assert_eq!(embedded.threshold.rule_id, component("export_retryable"));
        }
        drop(restored);
        assert_eq!(budget.snapshot().used_items, 0);

        // Flipping exhaustion alone is rejected against the recomputed limit.
        let mut wire: PersistedExportIssueReceiptWire = serde_json::from_slice(&bytes)
            .unwrap_or_else(|error| panic!("decode retryable receipt: {error}"));
        wire.embedded_receipt.threshold.is_exhausted = true;
        let (tampered, tampered_reference) = reencode_tampered(wire);
        let context = export_restore_context(&committed, &policy, sink_id, 1, tampered_reference);
        assert_eq!(
            futures::executor::block_on(restore_durable_export_issue_receipt(
                charged_bytes(&budget, &tampered),
                &context,
                &budget,
            ))
            .err(),
            Some(StreamingReliabilityError::ExportReceiptExhaustionMismatch)
        );
        assert_eq!(budget.snapshot().used_items, 0);
    }

    #[test]
    fn export_restore_rejects_tampered_disposition() {
        let run = StreamRunIdentity::new(LogicalReplayRunId::from_bytes([0xd6; 32]));
        let committed = committed_final_generation(run, 3);
        let generation = committed.generation();
        let sink_id = component("native_report");
        let (bytes, _, _) =
            prepared_export_bytes(run, &generation, &sink_id, StreamingIssueClass::Permanent, 0);
        let policy = export_recomputation_policy();
        let budget = export_restore_budget();

        // Exhausted plus an illegal export disposition: the prior reachability gate
        // accepted this document because it only constrained the non-exhausted case.
        let mut wire: PersistedExportIssueReceiptWire = serde_json::from_slice(&bytes)
            .unwrap_or_else(|error| panic!("decode permanent receipt: {error}"));
        wire.embedded_receipt.disposition = StreamingIssueDisposition::Quarantine;
        let (tampered, tampered_reference) = reencode_tampered(wire);
        let context = export_restore_context(&committed, &policy, sink_id, 0, tampered_reference);
        assert_eq!(
            futures::executor::block_on(restore_durable_export_issue_receipt(
                charged_bytes(&budget, &tampered),
                &context,
                &budget,
            ))
            .err(),
            Some(StreamingReliabilityError::ExportReceiptDispositionMismatch)
        );
        assert_eq!(budget.snapshot().used_items, 0);
    }

    #[test]
    fn derived_export_reference_cannot_be_built_from_unverified_fields() {
        let run = StreamRunIdentity::new(LogicalReplayRunId::from_bytes([0xd9; 32]));
        let committed = committed_final_generation(run, 11);
        let generation = committed.generation();
        let sink_id = component("native_report");
        let (bytes, reference, issue_id) =
            prepared_export_bytes(run, &generation, &sink_id, StreamingIssueClass::Permanent, 1);

        // A non-dense predecessor pair is not a reachable status.
        assert_eq!(
            VerifiedDerivedSinkAttemptStatus::from_status_owner(
                &committed,
                sink_id.clone(),
                1,
                0,
                reference.clone(),
            )
            .err(),
            Some(StreamingReliabilityError::NonContiguousExportCounter)
        );

        // A generation that was never made final cannot author export status.
        let non_final = committed_generation_with_finality(run, 12, false);
        assert_eq!(
            VerifiedDerivedSinkAttemptStatus::from_status_owner(
                &non_final,
                sink_id.clone(),
                1,
                1,
                reference.clone(),
            )
            .err(),
            Some(StreamingReliabilityError::NonFinalGenerationAuthority)
        );

        // A status minted against one generation cannot restore under another.
        let policy = export_recomputation_policy();
        let status = VerifiedDerivedSinkAttemptStatus::from_status_owner(
            &committed,
            sink_id,
            1,
            1,
            reference,
        )
        .unwrap_or_else(|error| panic!("verified status: {error}"));
        let other_final = committed_final_generation(run, 13);
        assert_eq!(
            DurableExportReceiptValidationContext::from_final_generation_status(
                &other_final,
                &policy,
                &status,
            )
            .err(),
            Some(StreamingReliabilityError::ExportReceiptGenerationMismatch)
        );

        // The matching authority restores, and the decision is the recomputed one.
        let context = DurableExportReceiptValidationContext::from_final_generation_status(
            &committed, &policy, &status,
        )
        .unwrap_or_else(|error| panic!("mint validation context: {error}"));
        let budget = export_restore_budget();
        let restored = futures::executor::block_on(restore_durable_export_issue_receipt(
            charged_bytes(&budget, &bytes),
            &context,
            &budget,
        ))
        .unwrap_or_else(|error| panic!("restore later durable export receipt: {error}"));
        assert_eq!(restored.issue_id(), issue_id);
        assert_eq!(restored.receipt.counter_before, 1);
        assert_eq!(restored.receipt.counter_after, 2);
        drop(restored);
        assert_eq!(budget.snapshot().used_items, 0);
    }

    fn typed_error_action_reporter(
        budget: StreamingResourceBudget,
        exhausted: StreamingIssueDisposition,
    ) -> BudgetOwnedStreamingIssueReporter {
        let run = StreamRunIdentity::new(LogicalReplayRunId::from_bytes([0x81; 32]));
        let policy = PreparedStreamingIssuePolicy::new([action_rule(
            "action_default",
            0,
            exhausted,
        )])
        .unwrap_or_else(|error| panic!("valid action policy: {error}"));
        BudgetOwnedStreamingIssueReporter::new(run, policy, budget)
            .unwrap_or_else(|error| panic!("budget-owned reporter: {error}"))
    }

    fn typed_error_action_evidence(
        issue: &OrdinaryStreamingIssue,
    ) -> CheckedActionFailureTerminalEvidence {
        CheckedActionFailureTerminalEvidence::for_test(
            issue.run,
            issue
                .scope()
                .action_id()
                .unwrap_or_else(|| panic!("action ID")),
            issue
                .order
                .global_sequence
                .unwrap_or_else(|| panic!("action sequence")),
            ContentDigest::from_bytes([0x91; 32]),
        )
    }

    #[test]
    fn missing_retained_action_attempt_returns_typed_error() {
        let budget = StreamingResourceBudget::new(super::super::budget::BudgetLimits {
            max_items: 65,
            max_bytes: QUEUE_CHARGE_BYTES + 64 * 1024,
        })
        .unwrap_or_else(|error| panic!("valid action budget: {error}"));
        let mut reporter = typed_error_action_reporter(
            budget.clone(),
            StreamingIssueDisposition::TerminalActionReceipt,
        );

        // Exactly the corruption the removed panic asserted away: an index
        // entry whose reporter token no longer resolves to a retained action.
        let issue = action_issue(1, 0);
        let evidence = typed_error_action_evidence(&issue);
        let orphan_lease = reporter
            .budget
            .try_acquire(1, 0)
            .unwrap_or_else(|error| panic!("orphan attempt entry: {error}"));
        reporter.current_action_attempts.insert(
            GlobalSequence::new(1),
            CurrentActionAttempt {
                reporter_token: 4242,
                entry_lease: orphan_lease,
            },
        );
        assert!(!reporter.pending_actions.contains_key(&4242));

        let before = budget.snapshot();
        let error = reporter
            .enqueue_failed_action(&evidence, issue)
            .expect_err("stale current-attempt index must refuse");
        assert_eq!(error, StreamingReliabilityError::CorruptActionAttemptIndex);

        let after = budget.snapshot();
        assert_eq!(after.used_items, before.used_items);
        assert_eq!(after.used_bytes, before.used_bytes);
    }

    #[test]
    fn absent_current_attempt_returns_typed_error_and_keeps_lease() {
        let budget = StreamingResourceBudget::new(super::super::budget::BudgetLimits {
            max_items: 65,
            max_bytes: QUEUE_CHARGE_BYTES + 64 * 1024,
        })
        .unwrap_or_else(|error| panic!("valid action budget: {error}"));
        let mut reporter = {
            let mut reporter = typed_error_action_reporter(
                budget.clone(),
                StreamingIssueDisposition::TerminalActionReceipt,
            );
            let orphan_lease = reporter
                .budget
                .try_acquire(1, 0)
                .unwrap_or_else(|error| panic!("orphan attempt entry: {error}"));
            reporter.current_action_attempts.insert(
                GlobalSequence::new(1),
                CurrentActionAttempt {
                    reporter_token: 4242,
                    entry_lease: orphan_lease,
                },
            );
            reporter
        };
        let inventory = FrozenActionInventory::for_test(
            StreamRunIdentity::new(LogicalReplayRunId::from_bytes([0x81; 32])),
            GlobalSequence::new(1),
            ContentDigest::from_bytes([0xa3; 32]),
            BTreeMap::new(),
        );

        let before = budget.snapshot();
        let error = reporter
            .prepare_no_more_actions_before(&inventory, GlobalSequence::new(1))
            .expect_err("stale current-attempt index must refuse gap closure");
        assert_eq!(error, StreamingReliabilityError::CorruptActionAttemptIndex);

        // The refusal precedes the proof acquisition, so no lease is minted and
        // none can be leaked.
        let after = budget.snapshot();
        assert_eq!(after.used_items, before.used_items);
        assert_eq!(after.used_bytes, before.used_bytes);
    }

    #[test]
    fn undecided_action_without_pending_issue_returns_typed_error() {
        let budget = StreamingResourceBudget::new(super::super::budget::BudgetLimits {
            max_items: 65,
            max_bytes: QUEUE_CHARGE_BYTES + 64 * 1024,
        })
        .unwrap_or_else(|error| panic!("valid action budget: {error}"));
        let mut reporter = typed_error_action_reporter(
            budget.clone(),
            StreamingIssueDisposition::TerminalActionReceipt,
        );

        let issue = action_issue(0, 0);
        let evidence = typed_error_action_evidence(&issue);
        let queued = reporter
            .enqueue_failed_action(&evidence, issue)
            .unwrap_or_else(|error| panic!("queue action failure: {error}"));
        let token = queued.reporter_token;

        // Release the reservation exactly once, leaving an undecided entry with
        // no pending issue. The charge is already returned by the lease drop,
        // so a second release would be a double free.
        let entry = reporter
            .pending_actions
            .get_mut(&token)
            .unwrap_or_else(|| panic!("entry is retained"));
        assert!(entry.decision.is_none());
        drop(entry.pending.take());
        let after_release = budget.snapshot();

        let error = reporter
            .poll_failed_action(queued)
            .expect_err("missing reservation must refuse");
        assert_eq!(error, StreamingReliabilityError::MissingPendingActionIssue);

        // The entry is returned to the ledger, so its token stays addressable
        // and the current-attempt index is still consistent.
        assert!(reporter.pending_actions.contains_key(&token));
        let settled = budget.snapshot();
        assert_eq!(settled.used_items, after_release.used_items);
        assert_eq!(settled.used_bytes, after_release.used_bytes);
    }

    #[test]
    fn refused_action_disposition_retains_its_pending_entry() {
        let budget = StreamingResourceBudget::new(super::super::budget::BudgetLimits {
            max_items: 65,
            max_bytes: QUEUE_CHARGE_BYTES + 64 * 1024,
        })
        .unwrap_or_else(|error| panic!("valid action budget: {error}"));
        // `Quarantine` is illegal for the Action scope, so `action_disposition`
        // refuses after the decision is recorded. `StreamingIssueThresholdRule::new`
        // rejects the pair, so the rule is built by struct literal.
        let rule = StreamingIssueThresholdRule {
            exhausted_disposition: StreamingIssueDisposition::Quarantine,
            ..copy_rule(&action_rule(
                "action_default",
                0,
                StreamingIssueDisposition::TerminalActionReceipt,
            ))
        };
        let run = StreamRunIdentity::new(LogicalReplayRunId::from_bytes([0x81; 32]));
        let policy = PreparedStreamingIssuePolicy::new([rule])
            .unwrap_or_else(|error| panic!("valid action policy: {error}"));
        let mut reporter = BudgetOwnedStreamingIssueReporter::new(run, policy, budget.clone())
            .unwrap_or_else(|error| panic!("budget-owned reporter: {error}"));

        let issue = action_issue(0, 0);
        let evidence = typed_error_action_evidence(&issue);
        let queued = reporter
            .enqueue_failed_action(&evidence, issue)
            .unwrap_or_else(|error| panic!("queue action failure: {error}"));
        let token = queued.reporter_token;

        let error = reporter
            .poll_failed_action(queued)
            .expect_err("illegal action disposition must refuse");
        assert_eq!(error, StreamingReliabilityError::IllegalDisposition);

        // Before the fix the entry was dropped here and the current-attempt
        // index named a token no lookup could resolve.
        assert!(reporter.pending_actions.contains_key(&token));
        assert_eq!(
            reporter
                .current_action_attempts
                .get(&GlobalSequence::new(0))
                .map(|attempt| attempt.reporter_token),
            Some(token)
        );
        // The refusal is stable: polling again reports the same typed error
        // from the retained decision rather than panicking.
        let repeat = reporter
            .poll_failed_action(QueuedActionFailure {
                reporter_token: token,
            })
            .expect_err("retained refusal repeats");
        assert_eq!(repeat, StreamingReliabilityError::IllegalDisposition);
    }

    #[test]
    fn budget_failure_code_is_total_over_budget_error() {
        use super::super::budget::BudgetError;

        let cases = [
            (
                BudgetError::ZeroCapacity,
                StateBudgetFailureCode::ItemCapacity,
            ),
            (
                BudgetError::PermitCountTooLarge,
                StateBudgetFailureCode::ItemCapacity,
            ),
            (
                BudgetError::RequestExceedsCapacity,
                StateBudgetFailureCode::ByteCapacity,
            ),
            (
                BudgetError::CapacityUnavailable,
                StateBudgetFailureCode::ByteCapacity,
            ),
            (BudgetError::Closed, StateBudgetFailureCode::ItemCapacity),
            (
                BudgetError::CannotGrowLease,
                StateBudgetFailureCode::ItemCapacity,
            ),
            (
                BudgetError::InvalidFragmentItemCharge { charged_items: 3 },
                StateBudgetFailureCode::ItemCapacity,
            ),
            (
                BudgetError::ActionPayloadUndercharged {
                    required_bytes: 4,
                    retained_bytes: 1,
                },
                StateBudgetFailureCode::ItemCapacity,
            ),
            (
                BudgetError::AccountingOverflow,
                StateBudgetFailureCode::ItemCapacity,
            ),
        ];
        for (error, expected) in cases {
            assert_eq!(budget_failure_code(error), expected);
            assert_eq!(
                state_budget_error(error),
                StreamingReliabilityError::StateBudget(expected)
            );
            assert_eq!(
                export_budget_error(error),
                StreamingReliabilityError::ExportReceiptBudget(expected)
            );
        }
    }

    #[test]
    fn measured_json_len_matches_serialized_length() {
        let value = serde_json::json!({
            "component": "aiperf.streaming.reliability",
            "counts": [1, 2, 3],
            "nested": {"unicode": "\u{00e9}\u{4e2d}"},
        });
        let encoded = serde_json::to_vec(&value)
            .unwrap_or_else(|error| panic!("encodable value: {error}"));
        let measured =
            measured_json_len(&value).unwrap_or_else(|error| panic!("measurable: {error}"));
        assert_eq!(measured, encoded.len());

        let (digest_len, digest) =
            measured_json_digest(&value).unwrap_or_else(|error| panic!("measurable: {error}"));
        assert_eq!(digest_len, encoded.len());
        assert_eq!(
            digest,
            ContentDigest::from_bytes(*blake3::hash(&encoded).as_bytes())
        );
    }

    #[test]
    fn queue_charge_matches_ring_buffer_capacity() {
        // Empirical check of the one std guarantee the charge rests on:
        // `VecDeque::with_capacity` promises *at least* the requested
        // capacity, so the reporter charges `queue.capacity()` after
        // construction rather than the requested constant. This asserts the
        // charge tracks the realized capacity whether or not std over-allocates.
        let queue: VecDeque<QueuedHandleIssue> = VecDeque::with_capacity(MAX_QUEUED_SUBMISSIONS);
        assert!(queue.capacity() >= MAX_QUEUED_SUBMISSIONS);

        let realized = super::super::budget::ring_buffer_bytes::<QueuedHandleIssue>(queue.capacity())
            .unwrap_or_else(|error| panic!("representable ring charge: {error}"));
        assert!(realized >= submission_queue_charge_bytes());

        let budget = StreamingResourceBudget::new(super::super::budget::BudgetLimits {
            max_items: 8,
            max_bytes: QUEUE_CHARGE_BYTES + 4 * 1024,
        })
        .unwrap_or_else(|error| panic!("valid budget: {error}"));
        let reporter = BudgetOwnedStreamingIssueReporter::new(
            StreamRunIdentity::new(LogicalReplayRunId::from_bytes([0x31; 32])),
            PreparedStreamingIssuePolicy::new([rule("record_default", None, 0)])
                .unwrap_or_else(|error| panic!("valid policy: {error}")),
            budget.clone(),
        )
        .unwrap_or_else(|error| panic!("budget-owned reporter: {error}"));

        // The queue is the reporter's only construction-time charge.
        assert_eq!(budget.snapshot().used_bytes, realized);
        assert_eq!(budget.snapshot().used_items, 1);
        drop(reporter);
        assert_eq!(budget.snapshot().used_bytes, 0);
        assert_eq!(budget.snapshot().used_items, 0);
    }

    #[test]
    fn oversized_component_id_allocation_cannot_bypass_the_charge() {
        // `String::shrink_to_fit` is documented as best-effort, so this asserts
        // the safety property directly — the charge is the retained capacity,
        // which always covers the length — rather than an exact shrink that the
        // allocator is free not to perform.
        let mut oversized = String::with_capacity(4096);
        oversized.push_str("component_id");
        let id = StreamingIssueComponentId::new(oversized)
            .unwrap_or_else(|error| panic!("valid component ID: {error}"));

        assert_eq!(id.as_str(), "component_id");
        assert!(id.retained_bytes() >= id.as_str().len());
        assert!(id.retained_bytes() <= 4096);
    }

    #[test]
    fn reporter_new_reports_budget_exhaustion() {
        let budget = StreamingResourceBudget::new(super::super::budget::BudgetLimits {
            max_items: 8,
            max_bytes: 64,
        })
        .unwrap_or_else(|error| panic!("valid budget: {error}"));

        assert!(matches!(
            BudgetOwnedStreamingIssueReporter::new(
                StreamRunIdentity::new(LogicalReplayRunId::from_bytes([0x32; 32])),
                PreparedStreamingIssuePolicy::new([rule("record_default", None, 0)])
                    .unwrap_or_else(|error| panic!("valid policy: {error}")),
                budget.clone(),
            ),
            Err(StreamingReliabilityError::StateBudget(_))
        ));
        assert_eq!(budget.snapshot().used_bytes, 0);
        assert_eq!(budget.snapshot().used_items, 0);
    }

    #[test]
    fn ordered_map_entry_charge_matches_retained_entries() {
        let receipt_entry = receipt_entry_bytes()
            .unwrap_or_else(|error| panic!("representable receipt entry: {error}"));
        assert_eq!(
            receipt_entry,
            super::super::budget::ordered_map_entry_bytes::<ContentDigest, RetainedReceipt>()
                .unwrap_or_else(|error| panic!("representable entry: {error}"))
        );

        let frontier_entry = input_frontier_entry_bytes()
            .unwrap_or_else(|error| panic!("representable frontier entry: {error}"));
        assert_eq!(
            frontier_entry,
            super::super::budget::ordered_map_entry_bytes::<
                StreamingInputDomainIdentity,
                RetainedInputFrontier,
            >()
            .unwrap_or_else(|error| panic!("representable entry: {error}"))
        );
    }

    /// Build one reporter over a budget sized for the whole reserved-submission
    /// inventory these tests exercise.
    fn submission_test_reporter(
        max_items: usize,
        max_bytes: usize,
    ) -> (
        StreamingResourceBudget,
        BudgetOwnedStreamingIssueReporter,
        StreamingInputDomainIdentity,
    ) {
        let budget = StreamingResourceBudget::new(super::super::budget::BudgetLimits {
            max_items,
            max_bytes,
        })
        .unwrap_or_else(|error| panic!("valid budget: {error}"));
        let policy = PreparedStreamingIssuePolicy::new([rule("record_default", None, 0)])
            .unwrap_or_else(|error| panic!("valid policy: {error}"));
        let reporter = BudgetOwnedStreamingIssueReporter::new(
            StreamRunIdentity::new(LogicalReplayRunId::from_bytes([0x11; 32])),
            policy,
            budget.clone(),
        )
        .unwrap_or_else(|error| panic!("budget-owned reporter: {error}"));
        let input_domain = StreamingInputDomainIdentity::new(
            ContentDigest::from_bytes([0x21; 32]),
            ImmutableObjectIdentity::from_bytes([0x20; 32]),
        );
        (budget, reporter, input_domain)
    }

    #[test]
    fn duplicate_reserved_submission_is_idempotent() {
        let (budget, mut reporter, input_domain) =
            submission_test_reporter(64, QUEUE_CHARGE_BYTES + 256 * 1024);
        let issue_id = record_issue().issue_id();
        futures::executor::block_on(reporter.report(IssueSequenceUpdate::Issue(record_issue())))
            .unwrap_or_else(|error| panic!("retain record issue: {error}"));
        futures::executor::block_on(reporter.report(IssueSequenceUpdate::NoMoreBefore {
            input_domain,
            through: SourcePosition::new(7),
        }))
        .unwrap_or_else(|error| panic!("classify record issue: {error}"));
        let classified = reporter
            .retained_outcome(&issue_id)
            .unwrap_or_else(|| panic!("classified receipt"));
        let settled_bytes = budget.snapshot().used_bytes;
        let settled_items = budget.snapshot().used_items;

        let handle = reporter.handle();
        assert!(matches!(
            futures::executor::block_on(handle.report(record_issue())),
            Ok(StreamingIssueReportStatus::Accepted)
        ));
        assert!(budget.snapshot().used_bytes > settled_bytes);

        reporter
            .drain_submission_queue()
            .unwrap_or_else(|error| panic!("drain replayed submission: {error}"));

        // The replay neither re-enters classification nor retains a second
        // receipt, and dropping its reservation returns the exact charge.
        assert_eq!(reporter.retained_outcome(&issue_id), Some(classified));
        assert_eq!(reporter.receipts.len(), 1);
        assert!(reporter.submission.queue.borrow().is_empty());
        assert_eq!(budget.snapshot().used_bytes, settled_bytes);
        assert_eq!(budget.snapshot().used_items, settled_items);
    }

    #[test]
    fn requeue_preserves_entry_lease_and_bounds_attempts() {
        let max_items = 64;
        let (budget, mut reporter, _input_domain) =
            submission_test_reporter(max_items, QUEUE_CHARGE_BYTES + 256 * 1024);
        let pending = reserve_pending_issue(&budget, record_issue())
            .unwrap_or_else(|error| panic!("reserve pending issue: {error}"));
        reporter
            .submission
            .queue
            .borrow_mut()
            .push_back(QueuedHandleIssue {
                pending,
                requeue_attempts: 0,
            });
        let reserved_bytes = budget.snapshot().used_bytes;
        let held = budget
            .try_acquire(max_items - budget.snapshot().used_items, 0)
            .unwrap_or_else(|error| panic!("hold remaining items: {error}"));

        assert!(matches!(
            reporter.drain_submission_queue(),
            Err(StreamingReliabilityError::StateBudget(_))
        ));
        {
            let queue = reporter.submission.queue.borrow();
            assert_eq!(queue.len(), 1);
            assert_eq!(
                queue
                    .front()
                    .unwrap_or_else(|| panic!("requeued submission"))
                    .requeue_attempts,
                1
            );
        }
        // The reservation went back with its lease intact, so a later drain can
        // retry against fresher headroom without a second charge.
        assert_eq!(budget.snapshot().used_bytes, reserved_bytes);

        reporter
            .submission
            .queue
            .borrow_mut()
            .front_mut()
            .unwrap_or_else(|| panic!("requeued submission"))
            .requeue_attempts = MAX_SUBMISSION_REQUEUE_ATTEMPTS;
        assert!(matches!(
            reporter.drain_submission_queue(),
            Err(StreamingReliabilityError::StateBudget(_))
        ));
        assert!(reporter.submission.queue.borrow().is_empty());
        assert!(budget.snapshot().used_bytes < reserved_bytes);
        drop(held);
    }

    #[test]
    fn submit_reserved_issue_without_source_position_returns_typed_error() {
        let (budget, mut reporter, _input_domain) =
            submission_test_reporter(64, QUEUE_CHARGE_BYTES + 256 * 1024);
        let mut pending = reserve_pending_issue(&budget, record_issue())
            .unwrap_or_else(|error| panic!("reserve pending issue: {error}"));
        // An adapter-minted fact can carry an input domain without the position
        // the scope constructors always set.
        pending.issue.order.source_position = None;
        let reserved_bytes = budget.snapshot().used_bytes;

        let Err((error, pending)) = reporter.submit_reserved_issue(pending) else {
            panic!("a positionless input-scoped fact must not classify");
        };

        assert!(matches!(
            error,
            StreamingReliabilityError::MissingInputSourcePosition
        ));
        assert!(reporter.pending_inputs.is_empty());
        // The unwind returned the intact lease rather than aborting the worker.
        assert_eq!(budget.snapshot().used_bytes, reserved_bytes);
        drop(pending);
        assert!(budget.snapshot().used_bytes < reserved_bytes);
    }

    #[test]
    fn queue_at_capacity_rejects_without_allocating() {
        let (budget, reporter, _input_domain) =
            submission_test_reporter(4 * MAX_QUEUED_SUBMISSIONS, QUEUE_CHARGE_BYTES + 8 * 1024 * 1024);
        let handle = reporter.handle();
        for _ in 0..MAX_QUEUED_SUBMISSIONS {
            assert!(matches!(
                futures::executor::block_on(handle.report(record_issue())),
                Ok(StreamingIssueReportStatus::Accepted)
            ));
        }
        assert_eq!(
            reporter.submission.queue.borrow().len(),
            MAX_QUEUED_SUBMISSIONS
        );
        let full_bytes = budget.snapshot().used_bytes;
        let full_items = budget.snapshot().used_items;

        assert!(matches!(
            futures::executor::block_on(handle.report(record_issue())),
            Ok(StreamingIssueReportStatus::Backpressured)
        ));

        // The refused submission released its reservation, so backpressure
        // costs the shared budget nothing and never grows the ring buffer.
        assert_eq!(
            reporter.submission.queue.borrow().len(),
            MAX_QUEUED_SUBMISSIONS
        );
        assert_eq!(budget.snapshot().used_bytes, full_bytes);
        assert_eq!(budget.snapshot().used_items, full_items);
    }

    #[test]
    fn owner_drop_closes_endpoint_and_releases_queue_lease() {
        let (budget, reporter, _input_domain) =
            submission_test_reporter(64, QUEUE_CHARGE_BYTES + 256 * 1024);
        let handle = reporter.handle();
        assert!(matches!(
            futures::executor::block_on(handle.report(record_issue())),
            Ok(StreamingIssueReportStatus::Accepted)
        ));
        let queue_lease_bytes = reporter
            .submission
            .queue_lease
            .borrow()
            .as_ref()
            .map(BudgetLease::charged_bytes)
            .unwrap_or_else(|| panic!("live ring-buffer lease"));
        assert!(queue_lease_bytes >= submission_queue_charge_bytes());
        assert_ne!(budget.snapshot().used_bytes, 0);

        drop(reporter);

        // The queued reservation and the endpoint-parked ring-buffer lease both
        // return to the caller-supplied budget, which is shared with other
        // participants and must never be starved by a dead reporter.
        assert_eq!(budget.snapshot().used_items, 0);
        assert_eq!(budget.snapshot().used_bytes, 0);
        assert!(matches!(
            futures::executor::block_on(handle.report(record_issue())),
            Err(StreamingIssueReportError::Closed)
        ));
        assert_eq!(budget.snapshot().used_bytes, 0);
    }

    #[test]
    fn close_is_idempotent_and_reports_released_bytes() {
        let run = StreamRunIdentity::new(LogicalReplayRunId::from_bytes([0x81; 32]));
        let budget = StreamingResourceBudget::new(super::super::budget::BudgetLimits {
            max_items: 64,
            max_bytes: QUEUE_CHARGE_BYTES + 256 * 1024,
        })
        .unwrap_or_else(|error| panic!("valid budget: {error}"));
        let policy = PreparedStreamingIssuePolicy::new([action_rule(
            "action_default",
            4,
            StreamingIssueDisposition::TerminalActionReceipt,
        )])
        .unwrap_or_else(|error| panic!("valid action policy: {error}"));
        let mut reporter = BudgetOwnedStreamingIssueReporter::new(run, policy, budget.clone())
            .unwrap_or_else(|error| panic!("budget-owned reporter: {error}"));
        let queue_lease_bytes = reporter
            .submission
            .queue_lease
            .borrow()
            .as_ref()
            .map(BudgetLease::charged_bytes)
            .unwrap_or_else(|| panic!("live ring-buffer lease"));

        // An undecided retained action failure is exactly the owner-side state
        // a typed error path preserves rather than destroys. It is charged
        // against the same budget and must not be counted by the endpoint close.
        let issue = action_issue(0, 0);
        let evidence = CheckedActionFailureTerminalEvidence::for_test(
            run,
            issue
                .scope()
                .action_id()
                .unwrap_or_else(|| panic!("action ID")),
            GlobalSequence::new(0),
            ContentDigest::from_bytes([0xa4; 32]),
        );
        reporter
            .enqueue_failed_action(&evidence, issue)
            .unwrap_or_else(|error| panic!("retain action failure: {error}"));
        assert_eq!(reporter.pending_actions.len(), 1);

        let pending = reserve_pending_issue(&budget, record_issue())
            .unwrap_or_else(|error| panic!("reserve pending issue: {error}"));
        let queued_items = pending.reservation.charged_items();
        let queued_bytes = pending.reservation.charged_bytes();
        reporter
            .submission
            .queue
            .borrow_mut()
            .push_back(QueuedHandleIssue {
                pending,
                requeue_attempts: 0,
            });

        let accounting = reporter.close();

        assert_eq!(
            accounting,
            ReporterCloseAccounting {
                released_items: queued_items + 1,
                released_bytes: queued_bytes + queue_lease_bytes,
            }
        );
        assert!(!reporter.is_open());
        // The retained action failure is still charged: close is an endpoint
        // transition, not a ledger teardown, and its accounting says so.
        assert_ne!(budget.snapshot().used_bytes, 0);
        assert_eq!(reporter.pending_actions.len(), 1);
        assert_eq!(
            reporter
                .summary()
                .unwrap_or_else(|error| panic!("readable summary: {error}"))
                .total,
            0
        );

        assert_eq!(
            reporter.close(),
            ReporterCloseAccounting {
                released_items: 0,
                released_bytes: 0,
            }
        );

        drop(reporter);

        // Ordinary field drop releases the owner-retained charge the close
        // deliberately left alone, so the two paths together leak nothing.
        assert_eq!(budget.snapshot().used_items, 0);
        assert_eq!(budget.snapshot().used_bytes, 0);
    }

    #[test]
    fn submit_after_close_reports_closed_endpoint() {
        let (budget, mut reporter, _input_domain) =
            submission_test_reporter(64, QUEUE_CHARGE_BYTES + 256 * 1024);
        let handle = reporter.handle();
        assert!(reporter.is_open());

        reporter.close();

        assert!(!reporter.is_open());
        assert!(matches!(
            futures::executor::block_on(handle.report(record_issue())),
            Err(StreamingIssueReportError::Closed)
        ));
        assert!(matches!(
            futures::executor::block_on(
                reporter.report(IssueSequenceUpdate::Issue(record_issue()))
            ),
            Err(StreamingReliabilityError::ReporterClosed)
        ));
        // A refused submission on either side advances no ledger state and
        // charges nothing.
        assert_eq!(
            reporter
                .summary()
                .unwrap_or_else(|error| panic!("readable summary: {error}"))
                .total,
            0
        );
        assert!(reporter.receipts.is_empty());
        assert_eq!(budget.snapshot().used_items, 0);
        assert_eq!(budget.snapshot().used_bytes, 0);
    }
}
