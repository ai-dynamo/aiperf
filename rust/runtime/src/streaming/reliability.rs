// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Host-owned streaming issue facts, deterministic policy, and authority vocabulary.
//!
//! This module deliberately stops before budget-owned receipt storage and
//! checkpoint integration. Ordinary owners can construct closed facts, but
//! only the host can construct a live decision or terminal failure outcome.

use std::{
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet},
    fmt,
    num::NonZeroU64,
    rc::Rc,
};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use super::{
    action::{
        CheckedActionFailureTerminalEvidenceView, CheckedActionTerminalMembershipView,
        FrozenActionInventoryView,
    },
    checkpoint::{
        CheckpointEpoch, CheckpointGeneration, StreamRunIdentity, StreamingCheckpointParticipant,
    },
    failure::{OrdinaryStreamingFailure, StreamingFailureStage},
    identity::{
        ContentDigest, GlobalSequence, ImmutableObjectIdentity, StableActionId, StableRecordId,
        StableSessionKey,
    },
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

fn scope_order_matches(scope: &StreamingIssueScope, order: &StreamingIssueOrderKey) -> bool {
    match scope {
        StreamingIssueScope::Run
        | StreamingIssueScope::Export { .. }
        | StreamingIssueScope::CheckpointAttempt { .. } => {
            order.input_domain.is_none()
                && order.source_position.is_none()
                && order.global_sequence.is_none()
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

    /// Submit an ordered update to the future budget-owned ledger.
    async fn report(
        &mut self,
        _update: IssueSequenceUpdate,
    ) -> Result<Option<StreamingIssueOutcome>, StreamingReliabilityError> {
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
        action::ActionTerminalMembershipOutcomeView,
        failure::{DecodeFailureCode, StreamFormatError},
        identity::LogicalReplayRunId,
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
            OrdinaryStreamingFailure::Format(StreamFormatError::decode(DecodeFailureCode::Syntax)),
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
}
