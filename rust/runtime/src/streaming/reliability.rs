// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Host-owned streaming issue facts, deterministic policy, and authority vocabulary.
//!
//! This module deliberately stops before budget-owned receipt storage and
//! checkpoint integration. Ordinary owners can construct closed facts, but
//! only the host can construct a live decision or terminal failure outcome.

use std::{
    cell::RefCell,
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet, VecDeque},
    fmt,
    mem::size_of,
    num::NonZeroU64,
    rc::Rc,
};

use async_trait::async_trait;
use bytes::Bytes;
use serde::{Deserialize, Deserializer, Serialize};

use super::{
    action::{
        ActionTerminalMembershipOutcomeView, CheckedActionFailureTerminalEvidenceView,
        CheckedActionTerminalMembershipView, FrozenActionInventoryView,
    },
    budget::{BudgetError, BudgetLease, StreamingResourceBudget},
    checkpoint::{
        BudgetedCheckpointBytes, CheckpointBarrier, CheckpointEpoch, CheckpointError,
        CheckpointGeneration, CheckpointParticipantId, CommittedParticipantReceipt,
        CommittedParticipantState, PreparedParticipantState, StreamRunIdentity,
        StreamingCheckpointParticipant,
    },
    failure::{OrdinaryStreamingFailure, StreamingFailureStage},
    identity::{
        ContentDigest, GlobalSequence, ImmutableObjectIdentity, StableActionId, StableRecordId,
        StableSessionKey,
    },
    results::{BudgetedResultDescriptor, ResultPartition},
    session::SessionQuarantineTombstoneView,
    unit::{SourcePosition, StateBudgetFailureCode},
};

/// Stable identifier for a reliability rule, failure code, or host component.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(try_from = "String")]
pub struct StreamingIssueComponentId(String);

impl StreamingIssueComponentId {
    /// Construct a checked lowercase ASCII identifier.
    pub fn new(value: impl Into<String>) -> Result<Self, StreamingReliabilityError> {
        let value = value.into();
        let bytes = value.as_bytes();
        let is_valid_first = bytes.first().is_some_and(u8::is_ascii_lowercase);
        let is_valid_tail = bytes.get(1..).is_some_and(|tail| {
            tail.iter()
                .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || *byte == b'_')
        });
        if !(1..=128).contains(&bytes.len()) || !is_valid_first || !is_valid_tail {
            return Err(StreamingReliabilityError::InvalidComponentId);
        }
        Ok(Self(value))
    }

    /// Borrow the checked stable identifier.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
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

#[allow(dead_code)]
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

#[allow(dead_code)]
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

const ISSUE_RECEIPT_WIRE_VERSION: u32 = 2;

/// Move-only detailed receipt with exact encoded and parsed-state charges.
///
/// The encoded bytes, verified parsed facts, and both leases are private and
/// cannot be detached from one another.
pub struct BudgetOwnedStreamingIssueReceipt {
    receipt: PersistedStreamingIssueReceipt,
    encoded: BudgetedCheckpointBytes,
    parsed_lease: BudgetLease,
    parsed_charge_bytes: usize,
}

impl BudgetOwnedStreamingIssueReceipt {
    /// Borrow the exact strict-v2 wire bytes.
    #[must_use]
    pub fn encoded_bytes(&self) -> &[u8] {
        self.encoded.as_bytes()
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

    /// Return the deterministic issue identity retained by this receipt.
    #[must_use]
    pub const fn issue_id(&self) -> ContentDigest {
        self.receipt.issue_id
    }

    /// Return the checked disposition retained by this receipt.
    #[must_use]
    pub const fn disposition(&self) -> StreamingIssueDisposition {
        self.receipt.disposition
    }
}

impl fmt::Debug for BudgetOwnedStreamingIssueReceipt {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BudgetOwnedStreamingIssueReceipt")
            .field("issue_id", &self.receipt.issue_id)
            .field("encoded_charge_bytes", &self.encoded.charged_bytes())
            .field("parsed_charge_bytes", &self.parsed_lease.charged_bytes())
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

impl HandledIssueCut {
    /// Construct the canonical cut containing no handled issues.
    #[must_use]
    pub fn empty() -> Self {
        Self::checked(
            empty_root(b"aiperf.streaming.issue-receipt-root.v1"),
            empty_root(b"aiperf.streaming.issue-input-frontier-root.v1"),
            empty_root(b"aiperf.streaming.quarantine-tombstone-root.v1"),
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
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StreamingIssueCounterView<'a> {
    counters: &'a BTreeMap<StreamingIssueCounterKey, u64>,
}

impl<'a> StreamingIssueCounterView<'a> {
    /// Return the matching count for one exact key.
    #[must_use]
    pub fn get(&self, key: &StreamingIssueCounterKey) -> Option<u64> {
        self.counters.get(key).copied()
    }

    /// Iterate over counters in canonical key order.
    pub fn iter(&self) -> impl Iterator<Item = (&StreamingIssueCounterKey, &u64)> {
        self.counters.iter()
    }
}

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
struct SealedActionGapClosureProof {
    membership_root: ContentDigest,
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

static EMPTY_COUNTERS: BTreeMap<StreamingIssueCounterKey, u64> = BTreeMap::new();

const RECEIPT_ENCODED_RESERVATION_BYTES: usize = 4096;

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
}

struct RetainedReceipt {
    receipt: BudgetOwnedStreamingIssueReceipt,
    outcome: StreamingIssueOutcome,
}

struct QueuedHandleIssue {
    pending: PendingIssue,
}

struct ReporterSubmissionEndpoint {
    run: StreamRunIdentity,
    budget: StreamingResourceBudget,
    queue: RefCell<VecDeque<QueuedHandleIssue>>,
}

#[async_trait(?Send)]
impl StreamingIssueReporterEndpoint for ReporterSubmissionEndpoint {
    async fn report(
        &self,
        issue: OrdinaryStreamingIssue,
    ) -> Result<StreamingIssueReportStatus, StreamingIssueReportError> {
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
        self.queue
            .borrow_mut()
            .push_back(QueuedHandleIssue { pending });
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
    input_frontiers: BTreeMap<StreamingInputDomainIdentity, SourcePosition>,
    pending_inputs: BTreeMap<StreamingInputDomainIdentity, BTreeMap<PendingInputKey, PendingIssue>>,
    pending_actions: BTreeMap<u64, PendingActionFailure>,
    current_action_attempts: BTreeMap<GlobalSequence, u64>,
    action_terminals: BTreeMap<GlobalSequence, RetainedActionTerminal>,
    action_frontier: Option<GlobalSequence>,
    next_reporter_token: u64,
    receipts: BTreeMap<ContentDigest, RetainedReceipt>,
    counters: BTreeMap<StreamingIssueCounterKey, u64>,
    summary: StreamingIssueSummary,
    is_initialized: bool,
}

impl BudgetOwnedStreamingIssueReporter {
    /// Construct one empty reporter under a frozen run, policy, and budget.
    #[must_use]
    pub fn new(
        run: StreamRunIdentity,
        policy: PreparedStreamingIssuePolicy,
        budget: StreamingResourceBudget,
    ) -> Self {
        let submission = Rc::new(ReporterSubmissionEndpoint {
            run,
            budget: budget.clone(),
            queue: RefCell::new(VecDeque::new()),
        });
        Self {
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
            next_reporter_token: 0,
            receipts: BTreeMap::new(),
            counters: BTreeMap::new(),
            summary: StreamingIssueSummary::empty(),
            is_initialized: false,
        }
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

    fn drain_submission_queue(&mut self) -> Result<(), StreamingReliabilityError> {
        loop {
            let queued = self.submission.queue.borrow_mut().pop_front();
            let Some(queued) = queued else {
                return Ok(());
            };
            if let Err((error, pending)) = self.submit_reserved_issue(queued.pending) {
                self.submission
                    .queue
                    .borrow_mut()
                    .push_front(QueuedHandleIssue { pending });
                return Err(error);
            }
        }
    }

    fn submit_issue(
        &mut self,
        issue: OrdinaryStreamingIssue,
    ) -> Result<Option<StreamingIssueOutcome>, StreamingReliabilityError> {
        let issue_id = issue.issue_id();
        if let Some(retained) = self.receipts.get(&issue_id) {
            return Ok(Some(retained.outcome));
        }
        if self.pending_issue_exists(&issue_id) {
            return Ok(None);
        }
        let pending = reserve_pending_issue(&self.budget, issue)?;
        self.submit_reserved_issue(pending)
            .map_err(|(error, _pending)| error)
    }

    // Returning the move-only pending reservation preserves retry authority
    // without an unbudgeted recovery allocation.
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
        let position = pending
            .issue
            .order
            .source_position
            .unwrap_or_else(|| unreachable!("input-scoped constructors set a source position"));
        if self
            .input_frontiers
            .get(&input_domain)
            .is_some_and(|through| position <= *through)
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
        let domain_pending = self.pending_inputs.entry(input_domain).or_default();
        if domain_pending.contains_key(&key) {
            return Err((StreamingReliabilityError::CorruptCheckpointState, pending));
        }
        domain_pending.insert(key, pending);
        Ok(None)
    }

    fn pending_issue_exists(&self, issue_id: &ContentDigest) -> bool {
        self.pending_inputs.values().any(|pending| {
            pending
                .values()
                .any(|candidate| candidate.issue.issue_id() == *issue_id)
        }) || self
            .pending_actions
            .values()
            .any(|candidate| candidate.issue_id == *issue_id)
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
        if let Some(current_token) = self.current_action_attempts.get(&sequence).copied() {
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
        self.current_action_attempts
            .insert(sequence, reporter_token);
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
                    || retained.receipt.receipt.scope.action_id() != Some(membership.action_id())
                    || retained.receipt.receipt.order.global_sequence != Some(membership.sequence())
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
        self.action_terminals
            .insert(sequence, RetainedActionTerminal { fact });
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
        for (sequence, token) in self.current_action_attempts.range(..=through) {
            let current = self
                .pending_actions
                .get(token)
                .ok_or(StreamingReliabilityError::CorruptActionAttemptIndex)?;
            if current.sequence != *sequence || !self.action_terminals.contains_key(sequence) {
                return Err(StreamingReliabilityError::InvalidActionTerminalMembership);
            }
        }
        let lease = self
            .budget
            .try_acquire(1, size_of::<CheckedNoMoreActionsBefore>())
            .map_err(state_budget_error)?;
        Ok(CheckedNoMoreActionsBefore {
            through,
            proof: SealedActionGapClosureProof {
                membership_root: inventory.membership_root(),
                lease,
            },
        })
    }

    fn retain_action_gap_closure(
        &mut self,
        closure: CheckedNoMoreActionsBefore,
    ) -> Result<(), StreamingReliabilityError> {
        let _membership_root = closure.proof.membership_root;
        let _lease = closure.proof.lease;
        if self
            .action_frontier
            .is_some_and(|frontier| closure.through < frontier)
        {
            return Err(StreamingReliabilityError::InvalidActionTerminalMembership);
        }
        self.action_frontier = Some(closure.through);
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
            .is_some_and(|current| through < *current)
        {
            return Err(StreamingReliabilityError::NonContiguousIssueFrontier);
        }

        loop {
            // Selection and removal share one borrow, so no second lookup can
            // observe a different map and no absent-key branch exists.
            let Some(domain_pending) = self.pending_inputs.get_mut(&input_domain) else {
                break;
            };
            let Some(first) = domain_pending.first_entry() else {
                break;
            };
            if first.key().position > through {
                break;
            }
            let (next_key, pending) = first.remove_entry();
            if let Err((error, pending)) = self.classify_pending(pending) {
                self.pending_inputs
                    .entry(input_domain.clone())
                    .or_default()
                    .insert(next_key, pending);
                return Err(error);
            }
        }
        self.input_frontiers.insert(input_domain.clone(), through);
        if self
            .pending_inputs
            .get(&input_domain)
            .is_some_and(BTreeMap::is_empty)
        {
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
        let key = counter_key_for_issue(&pending.issue, rule.rule_id.clone());
        let prior_matching_count = self.counters.get(&key).copied().unwrap_or(0);
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
        let (owned, pending_lease) = match budget_owned_receipt_from_reservation(
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
        drop(pending_lease);
        let outcome = StreamingIssueOutcome {
            issue_id,
            disposition,
            needs_admission_fence,
        };
        self.counters.insert(key, resulting_matching_count);
        self.summary = next_summary;
        self.receipts.insert(
            issue_id,
            RetainedReceipt {
                receipt: owned,
                outcome,
            },
        );
        Ok(outcome)
    }

    fn receipt_root(&self) -> ContentDigest {
        let mut hasher = blake3::Hasher::new();
        update_hash_field(&mut hasher, b"aiperf.streaming.issue-receipt-root.v1");
        for (issue_id, retained) in &self.receipts {
            update_hash_field(&mut hasher, issue_id.as_bytes());
            update_hash_field(&mut hasher, retained.receipt.encoded_bytes());
        }
        ContentDigest::from_bytes(*hasher.finalize().as_bytes())
    }

    fn input_frontier_root(&self) -> ContentDigest {
        let mut hasher = blake3::Hasher::new();
        update_hash_field(
            &mut hasher,
            b"aiperf.streaming.issue-input-frontier-root.v1",
        );
        for (domain, through) in &self.input_frontiers {
            update_hash_field(&mut hasher, domain.stream_identity.as_bytes());
            update_hash_field(&mut hasher, domain.source_identity.as_bytes());
            update_hash_field(&mut hasher, &through.get().to_le_bytes());
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
        let handled_cut = HandledIssueCut::checked(
            receipt_root,
            self.input_frontier_root(),
            HandledIssueCut::empty().quarantine_tombstone_root,
        );
        let wire = IssueReceiptPartitionWire {
            wire_version: ISSUE_RECEIPT_WIRE_VERSION,
            run: self.run,
            barrier_epoch: barrier.epoch,
            receipt_root,
            handled_cut: &handled_cut,
            receipts: self
                .receipts
                .values()
                .map(|value| &value.receipt.receipt)
                .collect(),
        };
        let encoded = serde_json::to_vec(&wire)
            .map_err(|_| StreamingReliabilityError::CorruptCheckpointState)?;
        let view_charge_bytes = size_of::<PreparedIssueReceiptPartitionView>();
        let (payload_lease, view_lease) = self
            .budget
            .acquire_pair(
                super::budget::BudgetCharge {
                    items: 1,
                    bytes: encoded.len(),
                },
                super::budget::BudgetCharge {
                    items: 1,
                    bytes: view_charge_bytes,
                },
            )
            .await
            .map_err(state_budget_error)?;
        let payload = BudgetedCheckpointBytes::new(Bytes::from(encoded), payload_lease)
            .map_err(|_| StreamingReliabilityError::CorruptCheckpointState)?;
        Ok(PreparedIssueReceiptPartitionView {
            run: self.run,
            barrier: barrier.clone(),
            receipt_root,
            handled_cut,
            receipt_count: u64::try_from(self.receipts.len())
                .map_err(|_| StreamingReliabilityError::CounterOverflow)?,
            payload,
            view_lease,
        })
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
            || !matches!(
                retained.receipt.receipt.scope,
                StreamingIssueScope::Session { .. }
            )
        {
            return Err(StreamingReliabilityError::QuarantineReceiptUnavailable);
        }
        let entries = view.canonical_encoded_entries();
        let payload_digest = ContentDigest::from_bytes(*blake3::hash(entries).as_bytes());
        if payload_digest != view.tombstone_root() {
            return Err(StreamingReliabilityError::StaleQuarantineTombstoneView);
        }
        let receipt_binding_root = digest_fields(
            b"aiperf.streaming.quarantine-receipt-binding.v1",
            &[issue_id.as_bytes(), self.receipt_root().as_bytes()],
        );
        let view_charge_bytes = size_of::<PreparedSessionQuarantineInstall>();
        let (payload_lease, view_lease) = budget
            .acquire_pair(
                super::budget::BudgetCharge {
                    items: 1,
                    bytes: entries.len(),
                },
                super::budget::BudgetCharge {
                    items: 1,
                    bytes: view_charge_bytes,
                },
            )
            .await
            .map_err(|error| {
                StreamingReliabilityError::QuarantineInstallBudget(budget_failure_code(error))
            })?;
        let payload = BudgetedCheckpointBytes::new(Bytes::copy_from_slice(entries), payload_lease)
            .map_err(|_| StreamingReliabilityError::CorruptCheckpointState)?;
        Ok(PreparedSessionQuarantineInstall {
            barrier: barrier.clone(),
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
        let embedded_encoded = serde_json::to_vec(&embedded_receipt)
            .map_err(|_| StreamingReliabilityError::CorruptCheckpointState)?;
        let embedded_receipt_length = u64::try_from(embedded_encoded.len())
            .map_err(|_| StreamingReliabilityError::ExportReceiptDigestLengthMismatch)?;
        let embedded_receipt_digest =
            ContentDigest::from_bytes(*blake3::hash(&embedded_encoded).as_bytes());
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
        let encoded = serde_json::to_vec(&persisted)
            .map_err(|_| StreamingReliabilityError::CorruptCheckpointState)?;
        let parsed_charge_bytes = parsed_export_receipt_bytes(&persisted);
        let (encoded_lease, parsed_lease) = budget
            .acquire_pair(
                super::budget::BudgetCharge {
                    items: 1,
                    bytes: encoded.len(),
                },
                super::budget::BudgetCharge {
                    items: 1,
                    bytes: parsed_charge_bytes,
                },
            )
            .await
            .map_err(export_budget_error)?;
        let receipt_length = u64::try_from(encoded.len())
            .map_err(|_| StreamingReliabilityError::ExportReceiptDigestLengthMismatch)?;
        let receipt_digest = ContentDigest::from_bytes(*blake3::hash(&encoded).as_bytes());
        let encoded = BudgetedCheckpointBytes::new(Bytes::from(encoded), encoded_lease)
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

#[derive(Serialize)]
#[serde(deny_unknown_fields)]
struct IssueReceiptPartitionWire<'a> {
    wire_version: u32,
    run: StreamRunIdentity,
    barrier_epoch: CheckpointEpoch,
    receipt_root: ContentDigest,
    handled_cut: &'a HandledIssueCut,
    receipts: Vec<&'a PersistedStreamingIssueReceipt>,
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
}

/// Move-only acknowledgement of one non-destructive session tombstone view.
pub struct PreparedSessionQuarantineInstall {
    barrier: CheckpointBarrier,
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

/// Sealed status-authored expectations used for ledger-free receipt reopen.
pub struct DurableExportReceiptValidationContext {
    run: StreamRunIdentity,
    generation: CheckpointGeneration,
    sink_id: StreamingIssueComponentId,
    policy_digest: ContentDigest,
    expected_attempt_ordinal: u32,
    expected_counter_before: u64,
}

impl DurableExportReceiptValidationContext {
    /// Construct validation inputs only after the status owner has verified its
    /// independently persisted predecessor fields.
    #[allow(dead_code)]
    pub(crate) fn from_status_authority(
        run: StreamRunIdentity,
        generation: CheckpointGeneration,
        sink_id: StreamingIssueComponentId,
        policy_digest: ContentDigest,
        expected_attempt_ordinal: u32,
        expected_counter_before: u64,
    ) -> Self {
        Self {
            run,
            generation,
            sink_id,
            policy_digest,
            expected_attempt_ordinal,
            expected_counter_before,
        }
    }
}

/// Strictly restore one status-reachable export receipt without a live ledger.
pub async fn restore_durable_export_issue_receipt(
    encoded: BudgetedCheckpointBytes,
    expected_reference: &DerivedExportReceiptReference,
    context: &DurableExportReceiptValidationContext,
    parsed_budget: &StreamingResourceBudget,
) -> Result<BudgetOwnedExportIssueReceipt, StreamingReliabilityError> {
    let encoded_length = u64::try_from(encoded.as_bytes().len())
        .map_err(|_| StreamingReliabilityError::ExportReceiptDigestLengthMismatch)?;
    let encoded_digest = ContentDigest::from_bytes(*blake3::hash(encoded.as_bytes()).as_bytes());
    if encoded_length != expected_reference.receipt_length
        || encoded_digest != expected_reference.receipt_digest
    {
        return Err(StreamingReliabilityError::ExportReceiptDigestLengthMismatch);
    }
    let wire: PersistedExportIssueReceiptWire = serde_json::from_slice(encoded.as_bytes())
        .map_err(|_| StreamingReliabilityError::DerivedExportReceiptUnreachable)?;
    if wire.wire_version != EXPORT_RECEIPT_WIRE_VERSION {
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
    if wire.attempt_ordinal != context.expected_attempt_ordinal
        || wire.embedded_receipt.order.retry_ordinal != context.expected_attempt_ordinal
        || wire.embedded_receipt.threshold.retry_ordinal != context.expected_attempt_ordinal
    {
        return Err(StreamingReliabilityError::ExportReceiptAttemptMismatch);
    }
    if wire.policy_digest != context.policy_digest
        || wire.embedded_receipt.threshold.policy_digest != context.policy_digest
    {
        return Err(StreamingReliabilityError::ExportReceiptPolicyMismatch);
    }
    if wire.counter_before != context.expected_counter_before
        || wire.embedded_receipt.threshold.prior_matching_count != context.expected_counter_before
        || wire.counter_after
            != wire
                .counter_before
                .checked_add(1)
                .ok_or(StreamingReliabilityError::NonContiguousExportCounter)?
        || wire.embedded_receipt.threshold.resulting_matching_count != wire.counter_after
    {
        return Err(StreamingReliabilityError::NonContiguousExportCounter);
    }
    if wire.embedded_receipt.wire_version != ISSUE_RECEIPT_WIRE_VERSION
        || wire.embedded_receipt.terminal_invariant.is_some()
        || wire.embedded_receipt.class == StreamingIssueClass::Invariant
        || wire.embedded_receipt.disposition == StreamingIssueDisposition::FailRun
        || !scope_order_matches(&wire.embedded_receipt.scope, &wire.embedded_receipt.order)
    {
        return Err(StreamingReliabilityError::DerivedExportReceiptUnreachable);
    }
    if !wire.embedded_receipt.threshold.is_exhausted
        && wire.embedded_receipt.disposition != StreamingIssueDisposition::Retry
    {
        return Err(StreamingReliabilityError::DerivedExportReceiptUnreachable);
    }
    match &wire.embedded_receipt.scope {
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
    let recomputed_issue_id = issue_id_from_wire(&wire.embedded_receipt);
    if wire.issue_id != recomputed_issue_id || wire.embedded_receipt.issue_id != recomputed_issue_id
    {
        return Err(StreamingReliabilityError::DerivedExportReceiptUnreachable);
    }
    let embedded_receipt = persisted_receipt_from_wire(wire.embedded_receipt);
    let embedded_encoded = serde_json::to_vec(&embedded_receipt)
        .map_err(|_| StreamingReliabilityError::DerivedExportReceiptUnreachable)?;
    let embedded_length = u64::try_from(embedded_encoded.len())
        .map_err(|_| StreamingReliabilityError::ExportReceiptDigestLengthMismatch)?;
    let embedded_digest = ContentDigest::from_bytes(*blake3::hash(&embedded_encoded).as_bytes());
    if wire.embedded_receipt_length != embedded_length
        || wire.embedded_receipt_digest != embedded_digest
        || expected_reference.embedded_receipt_length != embedded_length
        || expected_reference.embedded_receipt_digest != embedded_digest
    {
        return Err(StreamingReliabilityError::ExportReceiptDigestLengthMismatch);
    }
    let receipt = PersistedExportIssueReceipt {
        wire_version: wire.wire_version,
        run: wire.run,
        generation: wire.generation,
        sink_id: wire.sink_id,
        attempt_ordinal: wire.attempt_ordinal,
        issue_id: wire.issue_id,
        policy_digest: wire.policy_digest,
        counter_before: wire.counter_before,
        counter_after: wire.counter_after,
        embedded_receipt_digest: wire.embedded_receipt_digest,
        embedded_receipt_length: wire.embedded_receipt_length,
        embedded_receipt,
    };
    let parsed_charge_bytes = parsed_export_receipt_bytes(&receipt);
    let parsed_lease = parsed_budget
        .acquire(1, parsed_charge_bytes)
        .await
        .map_err(export_budget_error)?;
    Ok(BudgetOwnedExportIssueReceipt {
        receipt,
        encoded,
        parsed_lease,
        parsed_charge_bytes,
    })
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

fn issue_id_from_wire(wire: &PersistedStreamingIssueReceiptWire) -> ContentDigest {
    let mut hasher = blake3::Hasher::new();
    update_hash_field(&mut hasher, b"aiperf.streaming.issue-receipt.v2");
    update_hash_field(&mut hasher, wire.run.logical_replay_run().as_bytes());
    hash_scope(&mut hasher, &wire.scope);
    update_hash_field(&mut hasher, issue_class_tag(wire.class));
    update_hash_field(&mut hasher, failure_stage_tag(wire.stage));
    update_hash_field(&mut hasher, wire.code.as_str().as_bytes());
    update_hash_field(&mut hasher, wire.semantic_context_digest.as_bytes());
    update_hash_field(&mut hasher, &[u8::from(wire.order.input_domain.is_some())]);
    if let Some(input_domain) = &wire.order.input_domain {
        update_hash_field(&mut hasher, input_domain.stream_identity.as_bytes());
        update_hash_field(&mut hasher, input_domain.source_identity.as_bytes());
    }
    update_hash_field(
        &mut hasher,
        &[u8::from(wire.order.source_position.is_some())],
    );
    if let Some(source_position) = wire.order.source_position {
        update_hash_field(&mut hasher, &source_position.get().to_le_bytes());
    }
    update_hash_field(
        &mut hasher,
        &[u8::from(wire.order.global_sequence.is_some())],
    );
    if let Some(global_sequence) = wire.order.global_sequence {
        update_hash_field(&mut hasher, &global_sequence.get().to_le_bytes());
    }
    update_hash_field(&mut hasher, &wire.order.retry_ordinal.to_le_bytes());
    update_hash_field(&mut hasher, wire.order.scope_tiebreaker.as_bytes());
    update_hash_field(&mut hasher, &[u8::from(wire.terminal_invariant.is_some())]);
    if let Some(invariant) = wire.terminal_invariant {
        update_hash_field(&mut hasher, terminal_invariant_tag(invariant));
    }
    ContentDigest::from_bytes(*hasher.finalize().as_bytes())
}

fn reserve_pending_issue(
    budget: &StreamingResourceBudget,
    issue: OrdinaryStreamingIssue,
) -> Result<PendingIssue, StreamingReliabilityError> {
    let retained_issue_bytes = retained_issue_bytes(&issue);
    let parsed_reservation = retained_issue_bytes
        .checked_add(size_of::<PersistedStreamingIssueReceipt>())
        .and_then(|bytes| bytes.checked_add(512))
        .ok_or(StreamingReliabilityError::CounterOverflow)?;
    let total_bytes = retained_issue_bytes
        .checked_add(RECEIPT_ENCODED_RESERVATION_BYTES)
        .and_then(|bytes| bytes.checked_add(parsed_reservation))
        .ok_or(StreamingReliabilityError::CounterOverflow)?;
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

fn retained_issue_bytes(issue: &OrdinaryStreamingIssue) -> usize {
    size_of::<OrdinaryStreamingIssue>()
        + issue.code.as_str().len()
        + issue
            .scope
            .exporter_id()
            .map_or(0, |value| value.as_str().len())
}

fn parsed_receipt_bytes(receipt: &PersistedStreamingIssueReceipt) -> usize {
    size_of::<PersistedStreamingIssueReceipt>()
        + receipt.code.as_str().len()
        + receipt.threshold.rule_id.as_str().len()
        + receipt
            .scope
            .exporter_id()
            .map_or(0, |value| value.as_str().len())
}

fn parsed_export_receipt_bytes(receipt: &PersistedExportIssueReceipt) -> usize {
    size_of::<PersistedExportIssueReceipt>()
        + receipt.sink_id.as_str().len()
        + parsed_receipt_bytes(&receipt.embedded_receipt)
        + size_of::<PreparedExportAttemptFailure>()
        + size_of::<CheckedExportAttemptDecision>()
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
) -> Result<(BudgetOwnedStreamingIssueReceipt, BudgetLease), (StreamingReliabilityError, BudgetLease)>
{
    let encoded = match serde_json::to_vec(&receipt) {
        Ok(encoded) => encoded,
        Err(_) => {
            return Err((
                StreamingReliabilityError::CorruptCheckpointState,
                reservation,
            ));
        }
    };
    let parsed_charge_bytes = parsed_receipt_bytes(&receipt);
    let exact_bytes = match retained_issue_bytes
        .checked_add(encoded.len())
        .and_then(|bytes| bytes.checked_add(parsed_charge_bytes))
    {
        Some(bytes) => bytes,
        None => return Err((StreamingReliabilityError::CounterOverflow, reservation)),
    };
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
    let parsed_lease = match reservation.split_off(1, parsed_charge_bytes) {
        Ok(lease) => lease,
        Err(_) => {
            return Err((
                StreamingReliabilityError::CorruptCheckpointState,
                reservation,
            ));
        }
    };
    let encoded = match BudgetedCheckpointBytes::new(Bytes::from(encoded), encoded_lease) {
        Ok(encoded) => encoded,
        Err(_) => {
            return Err((
                StreamingReliabilityError::CorruptCheckpointState,
                reservation,
            ));
        }
    };
    Ok((
        BudgetOwnedStreamingIssueReceipt {
            receipt,
            encoded,
            parsed_lease,
            parsed_charge_bytes,
        },
        reservation,
    ))
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
        | BudgetError::AccountingOverflow => StateBudgetFailureCode::ItemCapacity,
    }
}

fn state_budget_error(error: BudgetError) -> StreamingReliabilityError {
    StreamingReliabilityError::StateBudget(budget_failure_code(error))
}

fn export_budget_error(error: BudgetError) -> StreamingReliabilityError {
    StreamingReliabilityError::ExportReceiptBudget(budget_failure_code(error))
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

    /// Borrow deterministic matching counters.
    fn counters(&self) -> StreamingIssueCounterView<'_> {
        StreamingIssueCounterView {
            counters: &EMPTY_COUNTERS,
        }
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
        _barrier: &CheckpointBarrier,
    ) -> Result<PreparedParticipantState, CheckpointError> {
        Err(CheckpointError::ParticipantUnavailable {
            participant: self.participant_id(),
        })
    }

    async fn initialize(
        &mut self,
        state: Option<CommittedParticipantState>,
    ) -> Result<(), CheckpointError> {
        if self.is_initialized {
            return Err(CheckpointError::AlreadyInitialized);
        }
        self.is_initialized = true;
        if state.is_some() {
            return Err(CheckpointError::ObjectVerification);
        }
        Ok(())
    }

    async fn checkpoint_committed(
        &mut self,
        receipt: &CommittedParticipantReceipt,
    ) -> Result<(), CheckpointError> {
        if receipt.run() != &self.run || receipt.participant_id() != &self.participant_id() {
            return Err(CheckpointError::PostCommitNotification {
                participant: self.participant_id(),
            });
        }
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
        }
    }

    async fn receipt_partition_view(
        &mut self,
        barrier: &CheckpointBarrier,
    ) -> Result<PreparedIssueReceiptPartitionView, StreamingReliabilityError> {
        self.prepare_receipt_partition_view(barrier).await
    }

    fn counters(&self) -> StreamingIssueCounterView<'_> {
        StreamingIssueCounterView {
            counters: &self.counters,
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
    use super::*;
    use crate::streaming::{
        action::{
            ActionTerminalMembershipOutcomeView, CheckedActionFailureTerminalEvidence,
            CheckedActionTerminalMembership, FrozenActionInventory,
        },
        checkpoint::{
            AcquisitionHorizon, AdmissionHorizon, CheckpointCut, DecodeHorizon, DiscoveryHorizon,
            EventTimeWatermark, OrderedActionHorizon, TerminalActionHorizon,
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
            max_items: 64,
            max_bytes: 64 * 1024,
        })
        .unwrap_or_else(|error| panic!("valid action budget: {error}"));
        let run = StreamRunIdentity::new(LogicalReplayRunId::from_bytes([0x81; 32]));
        let policy = PreparedStreamingIssuePolicy::new([action_rule(
            "action_default",
            0,
            StreamingIssueDisposition::TerminalActionReceipt,
        )])
        .unwrap_or_else(|error| panic!("valid action policy: {error}"));
        let mut reporter = BudgetOwnedStreamingIssueReporter::new(run, policy, action_budget);

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
            max_items: 96,
            max_bytes: 96 * 1024,
        })
        .unwrap_or_else(|error| panic!("valid action budget: {error}"));
        let run = StreamRunIdentity::new(LogicalReplayRunId::from_bytes([0x81; 32]));
        let policy = PreparedStreamingIssuePolicy::new([action_rule(
            "action_default",
            1,
            StreamingIssueDisposition::TerminalActionReceipt,
        )])
        .unwrap_or_else(|error| panic!("valid action policy: {error}"));
        let mut reporter = BudgetOwnedStreamingIssueReporter::new(run, policy, action_budget);

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
            max_items: 64,
            max_bytes: 64 * 1024,
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
        let mut reporter = BudgetOwnedStreamingIssueReporter::new(run, policy, action_budget);

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
            max_items: 4,
            max_bytes: 64 * 1024,
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
            BudgetOwnedStreamingIssueReporter::new(run, policy, action_budget.clone());
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
            max_items: 64,
            max_bytes: 64 * 1024,
        })
        .unwrap_or_else(|error| panic!("valid reporter budget: {error}"));
        let install_budget = StreamingResourceBudget::new(super::super::budget::BudgetLimits {
            max_items: 8,
            max_bytes: 8 * 1024,
        })
        .unwrap_or_else(|error| panic!("valid install budget: {error}"));
        let mut reporter = BudgetOwnedStreamingIssueReporter::new(run, policy, reporter_budget);
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

    #[test]
    fn export_failure_persistence_restores_from_status_authority_without_ledger() {
        let run = StreamRunIdentity::new(LogicalReplayRunId::from_bytes([0xc1; 32]));
        let generation = CheckpointGeneration::new(
            CheckpointEpoch::new(9),
            ContentDigest::from_bytes([0xc2; 32]),
        );
        let sink_id = component("native_report");
        let policy = PreparedStreamingIssuePolicy::new([StreamingIssueThresholdRule::new(
            component("export_default"),
            StreamingIssueScopeKind::Export,
            StreamingIssueClass::Permanent,
            None,
            0,
            StreamingIssueDisposition::ExportIncomplete,
            None,
        )
        .unwrap_or_else(|error| panic!("valid export rule: {error}"))])
        .unwrap_or_else(|error| panic!("valid export policy: {error}"));
        let policy_digest = *policy.digest();
        let reporter_budget = StreamingResourceBudget::new(super::super::budget::BudgetLimits {
            max_items: 16,
            max_bytes: 32 * 1024,
        })
        .unwrap_or_else(|error| panic!("valid reporter budget: {error}"));
        let export_budget = StreamingResourceBudget::new(super::super::budget::BudgetLimits {
            max_items: 8,
            max_bytes: 32 * 1024,
        })
        .unwrap_or_else(|error| panic!("valid export budget: {error}"));
        let mut reporter = BudgetOwnedStreamingIssueReporter::new(run, policy, reporter_budget);
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
        assert!(prepared.is_exhausted());
        assert_eq!(prepared.attempt_ordinal(), 0);
        assert_eq!(prepared.counter_before(), 0);
        let issue_id = prepared.issue_id();
        let reference = prepared.receipt_reference().clone();
        let persistence = prepared.into_persistence();
        assert_eq!(persistence.issue_id(), issue_id);
        assert_eq!(export_budget.snapshot().used_items, 2);

        let stored = persistence.encoded_bytes().to_vec();
        let tamper_source = stored.clone();
        drop(persistence);
        assert_eq!(export_budget.snapshot().used_items, 0);
        let stored_budget = StreamingResourceBudget::new(super::super::budget::BudgetLimits {
            max_items: 4,
            max_bytes: 32 * 1024,
        })
        .unwrap_or_else(|error| panic!("valid stored budget: {error}"));
        let stored_lease = futures::executor::block_on(stored_budget.acquire(1, stored.len()))
            .unwrap_or_else(|error| panic!("charge stored receipt: {error}"));
        let stored = BudgetedCheckpointBytes::new(Bytes::from(stored), stored_lease)
            .unwrap_or_else(|error| panic!("valid stored receipt bytes: {error}"));
        let context = DurableExportReceiptValidationContext::from_status_authority(
            run,
            generation,
            sink_id,
            policy_digest,
            0,
            0,
        );
        let restored = futures::executor::block_on(restore_durable_export_issue_receipt(
            stored,
            &reference,
            &context,
            &stored_budget,
        ))
        .unwrap_or_else(|error| panic!("restore durable export receipt: {error}"));
        assert_eq!(restored.issue_id(), issue_id);
        assert_eq!(
            restored.encoded_charge_bytes(),
            reference.receipt_length() as usize
        );
        assert_eq!(stored_budget.snapshot().used_items, 2);
        drop(restored);
        assert_eq!(stored_budget.snapshot().used_items, 0);

        let mut wire: PersistedExportIssueReceiptWire = serde_json::from_slice(&tamper_source)
            .unwrap_or_else(|error| panic!("decode test export receipt: {error}"));
        wire.embedded_receipt.threshold.is_exhausted = false;
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
        let tampered = serde_json::to_vec(&tampered_receipt)
            .unwrap_or_else(|error| panic!("encode tampered export receipt: {error}"));
        let tampered_reference = DerivedExportReceiptReference {
            receipt_digest: ContentDigest::from_bytes(*blake3::hash(&tampered).as_bytes()),
            receipt_length: tampered.len() as u64,
            embedded_receipt_digest,
            embedded_receipt_length,
        };
        let tampered_lease = futures::executor::block_on(stored_budget.acquire(1, tampered.len()))
            .unwrap_or_else(|error| panic!("charge tampered receipt: {error}"));
        let tampered = BudgetedCheckpointBytes::new(Bytes::from(tampered), tampered_lease)
            .unwrap_or_else(|error| panic!("valid tampered receipt bytes: {error}"));
        assert!(matches!(
            futures::executor::block_on(restore_durable_export_issue_receipt(
                tampered,
                &tampered_reference,
                &context,
                &stored_budget,
            )),
            Err(StreamingReliabilityError::DerivedExportReceiptUnreachable)
        ));
        assert_eq!(stored_budget.snapshot().used_items, 0);
    }

    #[test]
    fn later_export_exhaustion_restores_only_under_exact_status_counter() {
        let run = StreamRunIdentity::new(LogicalReplayRunId::from_bytes([0xd1; 32]));
        let generation = CheckpointGeneration::new(
            CheckpointEpoch::new(12),
            ContentDigest::from_bytes([0xd2; 32]),
        );
        let sink_id = component("native_report");
        let policy = PreparedStreamingIssuePolicy::new([StreamingIssueThresholdRule::new(
            component("export_default"),
            StreamingIssueScopeKind::Export,
            StreamingIssueClass::Permanent,
            None,
            1,
            StreamingIssueDisposition::ExportIncomplete,
            None,
        )
        .unwrap_or_else(|error| panic!("valid export rule: {error}"))])
        .unwrap_or_else(|error| panic!("valid export policy: {error}"));
        let policy_digest = *policy.digest();
        let reporter_budget = StreamingResourceBudget::new(super::super::budget::BudgetLimits {
            max_items: 16,
            max_bytes: 32 * 1024,
        })
        .unwrap_or_else(|error| panic!("valid reporter budget: {error}"));
        let export_budget = StreamingResourceBudget::new(super::super::budget::BudgetLimits {
            max_items: 8,
            max_bytes: 32 * 1024,
        })
        .unwrap_or_else(|error| panic!("valid export budget: {error}"));
        let mut reporter = BudgetOwnedStreamingIssueReporter::new(run, policy, reporter_budget);

        let first_issue = OrdinaryStreamingIssue::export(
            run,
            sink_id.clone(),
            generation.clone(),
            StreamingIssueClass::Permanent,
            ContentDigest::from_bytes([0xd3; 32]),
            0,
            ContentDigest::from_bytes([0xd4; 32]),
            OrdinaryStreamingFailure::Export(ResultExportError::failure(
                ResultExportFailureCode::Attempt,
            )),
        )
        .unwrap_or_else(|error| panic!("valid first export issue: {error}"));
        let first = futures::executor::block_on(reporter.prepare_export_attempt_failure(
            &run,
            &generation,
            &sink_id,
            0,
            ResultSinkAttemptOutcome::Failed(first_issue),
            &export_budget,
        ))
        .unwrap_or_else(|error| panic!("prepare first export failure: {error}"));
        assert!(!first.is_exhausted());
        assert_eq!(first.counter_before(), 0);
        drop(first);

        let second_issue = OrdinaryStreamingIssue::export(
            run,
            sink_id.clone(),
            generation.clone(),
            StreamingIssueClass::Permanent,
            ContentDigest::from_bytes([0xd3; 32]),
            1,
            ContentDigest::from_bytes([0xd5; 32]),
            OrdinaryStreamingFailure::Export(ResultExportError::failure(
                ResultExportFailureCode::Attempt,
            )),
        )
        .unwrap_or_else(|error| panic!("valid second export issue: {error}"));
        let second = futures::executor::block_on(reporter.prepare_export_attempt_failure(
            &run,
            &generation,
            &sink_id,
            1,
            ResultSinkAttemptOutcome::Failed(second_issue),
            &export_budget,
        ))
        .unwrap_or_else(|error| panic!("prepare second export failure: {error}"));
        assert!(second.is_exhausted());
        assert_eq!(second.counter_before(), 1);
        let issue_id = second.issue_id();
        let reference = second.receipt_reference().clone();
        let persistence = second.into_persistence();
        let stored_bytes = persistence.encoded_bytes().to_vec();
        drop(persistence);

        let stored_budget = StreamingResourceBudget::new(super::super::budget::BudgetLimits {
            max_items: 4,
            max_bytes: 32 * 1024,
        })
        .unwrap_or_else(|error| panic!("valid stored budget: {error}"));
        let wrong_lease = futures::executor::block_on(stored_budget.acquire(1, stored_bytes.len()))
            .unwrap_or_else(|error| panic!("charge stored receipt: {error}"));
        let wrong_bytes =
            BudgetedCheckpointBytes::new(Bytes::copy_from_slice(&stored_bytes), wrong_lease)
                .unwrap_or_else(|error| panic!("valid stored receipt bytes: {error}"));
        let wrong_context = DurableExportReceiptValidationContext::from_status_authority(
            run,
            generation.clone(),
            sink_id.clone(),
            policy_digest,
            1,
            0,
        );
        assert!(matches!(
            futures::executor::block_on(restore_durable_export_issue_receipt(
                wrong_bytes,
                &reference,
                &wrong_context,
                &stored_budget,
            )),
            Err(StreamingReliabilityError::NonContiguousExportCounter)
        ));
        assert_eq!(stored_budget.snapshot().used_items, 0);

        let stored_lease =
            futures::executor::block_on(stored_budget.acquire(1, stored_bytes.len()))
                .unwrap_or_else(|error| panic!("recharge stored receipt: {error}"));
        let stored = BudgetedCheckpointBytes::new(Bytes::from(stored_bytes), stored_lease)
            .unwrap_or_else(|error| panic!("valid stored receipt bytes: {error}"));
        let context = DurableExportReceiptValidationContext::from_status_authority(
            run,
            generation,
            sink_id,
            policy_digest,
            1,
            1,
        );
        let restored = futures::executor::block_on(restore_durable_export_issue_receipt(
            stored,
            &reference,
            &context,
            &stored_budget,
        ))
        .unwrap_or_else(|error| panic!("restore later durable export receipt: {error}"));
        assert_eq!(restored.issue_id(), issue_id);
        drop(restored);
        assert_eq!(stored_budget.snapshot().used_items, 0);
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
            max_items: 64,
            max_bytes: 64 * 1024,
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
        reporter
            .current_action_attempts
            .insert(GlobalSequence::new(1), 4242);
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
            max_items: 64,
            max_bytes: 64 * 1024,
        })
        .unwrap_or_else(|error| panic!("valid action budget: {error}"));
        let mut reporter = {
            let mut reporter = typed_error_action_reporter(
                budget.clone(),
                StreamingIssueDisposition::TerminalActionReceipt,
            );
            reporter
                .current_action_attempts
                .insert(GlobalSequence::new(1), 4242);
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
            max_items: 64,
            max_bytes: 64 * 1024,
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
            max_items: 64,
            max_bytes: 64 * 1024,
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
        let mut reporter = BudgetOwnedStreamingIssueReporter::new(run, policy, budget.clone());

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
                .copied(),
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
}
