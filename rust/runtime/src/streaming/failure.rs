// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Stable failure stages and codes shared by streaming extension contracts.

use std::{fmt, rc::Rc};

use serde::{Deserialize, Serialize};

use async_trait::async_trait;

use super::{
    checkpoint::{CheckpointError, StreamRunIdentity, StreamingCheckpointParticipant},
    identity::{
        ContentDigest, GlobalSequence, ImmutableObjectIdentity, StableActionId, StableRecordId,
        StableSessionKey,
    },
    unit::{SourcePosition, StateBudgetFailureCode},
};

/// Stable stage at which a streaming failure occurred.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamingFailureStage {
    /// Source discovery or snapshot construction.
    Source,
    /// Immutable partition acquisition.
    Acquisition,
    /// Format decoding.
    Decode,
    /// Watermark or stable ordering.
    Ordering,
    /// Bounded state admission.
    StateBudget,
    /// Cross-record session coordination.
    Session,
    /// Cellular or worker placement.
    Placement,
    /// Action submission or execution.
    Dispatch,
    /// Checkpoint state or publication.
    Checkpoint,
    /// Checkpoint-native result processing.
    Result,
}

/// An error with a stable stage and machine-readable code.
pub trait StableStreamingFailure: std::error::Error {
    /// Return the stage that retains authority for this failure.
    fn stage(&self) -> StreamingFailureStage;

    /// Return the stable lowercase failure code.
    fn code(&self) -> &'static str;
}

macro_rules! failure_codes {
    (
        $(#[$enum_meta:meta])*
        pub enum $name:ident {
            $($(#[$variant_meta:meta])* $variant:ident => $code:literal),+ $(,)?
        }
    ) => {
        $(#[$enum_meta])*
        #[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
        #[serde(rename_all = "snake_case")]
        pub enum $name {
            $($(#[$variant_meta])* $variant),+
        }

        impl $name {
            pub const fn code(self) -> &'static str {
                match self {
                    $(Self::$variant => $code),+
                }
            }
        }
    };
}

failure_codes! {
    /// Stable source-discovery failure classification.
    pub enum SourceFailureCode {
        /// Inventory discovery failed.
        Discovery => "discovery",
        /// A stable source snapshot could not be formed.
        Snapshot => "snapshot",
        /// Previously identified source content mutated.
        MutatedObject => "mutated_object",
        /// The selected source is unavailable.
        SourceUnavailable => "source_unavailable",
    }
}

failure_codes! {
    /// Stable immutable-acquisition failure classification.
    pub enum AcquisitionFailureCode {
        /// Content authority could not be opened.
        Open => "open",
        /// Content bytes could not be read.
        Read => "read",
        /// Acquired bytes did not match immutable identity.
        IdentityMismatch => "identity_mismatch",
        /// An authored object limit was exceeded.
        ObjectLimitExceeded => "object_limit_exceeded",
        /// Retained acquired bytes and their capacity lease disagree.
        BudgetInvariant => "budget_invariant",
        /// A reader returned an empty, oversized, or discontinuous sequential chunk.
        InvalidChunk => "invalid_chunk",
    }
}

failure_codes! {
    /// Stable format-decoding failure classification.
    pub enum DecodeFailureCode {
        /// Input syntax is invalid.
        Syntax => "syntax",
        /// Input does not satisfy the selected schema.
        Schema => "schema",
        /// One record exceeds the validated bound.
        OversizedRecord => "oversized_record",
        /// A decoder cursor is invalid for the immutable input.
        InvalidCursor => "invalid_cursor",
        /// Required replay metadata is absent.
        MissingReplayMetadata => "missing_replay_metadata",
        /// Recorded replay geometry is impossible.
        InvalidReplayGeometry => "invalid_replay_geometry",
        /// Recorded synthesis authority does not match validation.
        SynthesisAuthorityMismatch => "synthesis_authority_mismatch",
        /// The immutable synthesis profile cannot be prepared.
        SynthesisProfileUnavailable => "synthesis_profile_unavailable",
        /// Retained resume bytes and their capacity lease disagree.
        BudgetInvariant => "budget_invariant",
    }
}

failure_codes! {
    /// Stable ordering and watermark failure classification.
    pub enum OrderingFailureCode {
        /// An event violated the selected late-data policy.
        LateData => "late_data",
        /// An event contradicted an asserted watermark.
        WatermarkViolation => "watermark_violation",
        /// A stable order coordinate cannot be represented.
        CoordinateOverflow => "coordinate_overflow",
    }
}

failure_codes! {
    /// Stable session-program failure classification.
    pub enum SessionFailureCode {
        /// A declared predecessor cannot be resolved.
        MissingPredecessor => "missing_predecessor",
        /// One logical mutation identity has conflicting content.
        ConflictingMutation => "conflicting_mutation",
        /// Causal state cannot be retained within a proven bound.
        UnboundedCausalityState => "unbounded_causality_state",
    }
}

failure_codes! {
    /// Stable placement failure classification.
    pub enum PlacementFailureCode {
        /// No valid execution route is available.
        RouteUnavailable => "route_unavailable",
        /// An event names a fenced ownership epoch.
        StaleOwnershipEpoch => "stale_ownership_epoch",
        /// Peers disagree about immutable semantic content.
        DigestMismatch => "digest_mismatch",
        /// The target coordinate cannot be represented.
        TargetOverflow => "target_overflow",
        /// Placement was cancelled.
        Cancelled => "cancelled",
    }
}

failure_codes! {
    /// Stable action-binding and dispatch failure classification.
    pub enum ActionFailureCode {
        /// No prepared binding accepts the action schema.
        MissingBinding => "missing_binding",
        /// The action could not be dispatched.
        Dispatch => "dispatch",
        /// The endpoint returned a terminal failure.
        Endpoint => "endpoint",
        /// Action execution was cancelled.
        Cancelled => "cancelled",
        /// Retained payload bytes and their capacity lease disagree.
        BudgetInvariant => "budget_invariant",
    }
}

const fn budget_code(code: StateBudgetFailureCode) -> &'static str {
    match code {
        StateBudgetFailureCode::ItemCapacity => "item_capacity",
        StateBudgetFailureCode::ByteCapacity => "byte_capacity",
        StateBudgetFailureCode::SpillCapacity => "spill_capacity",
        StateBudgetFailureCode::ProvisionalCapacity => "provisional_capacity",
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum StreamSourceErrorKind {
    Source(SourceFailureCode),
    Acquisition(AcquisitionFailureCode),
    Ordering(OrderingFailureCode),
    ControlledStop,
}

/// Source discovery and immutable-acquisition error.
///
/// Controlled stop cannot be constructed by an adapter:
///
/// ```compile_fail
/// use aiperf_runtime::streaming::failure::StreamSourceError;
/// let _ = StreamSourceError::controlled_stop();
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StreamSourceError {
    kind: StreamSourceErrorKind,
}

impl StreamSourceError {
    /// Construct a source-discovery failure from a closed classification.
    pub const fn source(code: SourceFailureCode) -> Self {
        Self {
            kind: StreamSourceErrorKind::Source(code),
        }
    }

    /// Construct an immutable-acquisition failure from a closed classification.
    pub const fn acquisition(code: AcquisitionFailureCode) -> Self {
        Self {
            kind: StreamSourceErrorKind::Acquisition(code),
        }
    }

    /// Construct a source-ordering failure from a closed classification.
    pub const fn ordering(code: OrderingFailureCode) -> Self {
        Self {
            kind: StreamSourceErrorKind::Ordering(code),
        }
    }

    pub(crate) const fn controlled_stop() -> Self {
        Self {
            kind: StreamSourceErrorKind::ControlledStop,
        }
    }

    /// Return whether this error is the opaque host-controlled stop outcome.
    #[must_use]
    pub const fn is_stopped(&self) -> bool {
        matches!(self.kind, StreamSourceErrorKind::ControlledStop)
    }

    const fn failure_stage(&self) -> StreamingFailureStage {
        match self.kind {
            StreamSourceErrorKind::Source(_) | StreamSourceErrorKind::ControlledStop => {
                StreamingFailureStage::Source
            }
            StreamSourceErrorKind::Acquisition(_) => StreamingFailureStage::Acquisition,
            StreamSourceErrorKind::Ordering(_) => StreamingFailureStage::Ordering,
        }
    }

    const fn failure_code(&self) -> &'static str {
        match self.kind {
            StreamSourceErrorKind::Source(code) => code.code(),
            StreamSourceErrorKind::Acquisition(code) => code.code(),
            StreamSourceErrorKind::Ordering(code) => code.code(),
            StreamSourceErrorKind::ControlledStop => "stopped",
        }
    }
}

/// Format decode, ordering, or state-budget error.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StreamFormatError {
    /// Decoder-owned format failure.
    Decode(DecodeFailureCode),
    /// Format-owned ordering or frontier failure.
    Ordering(OrderingFailureCode),
    /// Decoder state exceeded an explicit capacity.
    StateBudget(StateBudgetFailureCode),
}

impl StreamFormatError {
    /// Construct a decode failure.
    pub const fn decode(code: DecodeFailureCode) -> Self {
        Self::Decode(code)
    }
    /// Construct an ordering failure.
    pub const fn ordering(code: OrderingFailureCode) -> Self {
        Self::Ordering(code)
    }
    /// Construct a state-budget failure.
    pub const fn state_budget(code: StateBudgetFailureCode) -> Self {
        Self::StateBudget(code)
    }
    const fn failure_stage(&self) -> StreamingFailureStage {
        match self {
            Self::Decode(_) => StreamingFailureStage::Decode,
            Self::Ordering(_) => StreamingFailureStage::Ordering,
            Self::StateBudget(_) => StreamingFailureStage::StateBudget,
        }
    }
    const fn failure_code(&self) -> &'static str {
        match self {
            Self::Decode(code) => code.code(),
            Self::Ordering(code) => code.code(),
            Self::StateBudget(code) => budget_code(*code),
        }
    }
}

/// Session coordination or retained-state error.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SessionCoordinatorError {
    /// Session-program semantic failure.
    Session(SessionFailureCode),
    /// Session state exceeded an explicit capacity.
    StateBudget(StateBudgetFailureCode),
}

impl SessionCoordinatorError {
    /// Construct a session semantic failure.
    pub const fn session(code: SessionFailureCode) -> Self {
        Self::Session(code)
    }
    /// Construct a session state-budget failure.
    pub const fn state_budget(code: StateBudgetFailureCode) -> Self {
        Self::StateBudget(code)
    }
    const fn failure_stage(&self) -> StreamingFailureStage {
        match self {
            Self::Session(_) => StreamingFailureStage::Session,
            Self::StateBudget(_) => StreamingFailureStage::StateBudget,
        }
    }
    const fn failure_code(&self) -> &'static str {
        match self {
            Self::Session(code) => code.code(),
            Self::StateBudget(code) => budget_code(*code),
        }
    }
}

/// Action binding, placement, dispatch, or retained-state error.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ActionExecutionError {
    /// Placement authority refused the action.
    Placement(PlacementFailureCode),
    /// The selected binding or dispatch path failed.
    Action(ActionFailureCode),
    /// Action state exceeded an explicit capacity.
    StateBudget(StateBudgetFailureCode),
}

impl ActionExecutionError {
    /// Construct a placement failure.
    pub const fn placement(code: PlacementFailureCode) -> Self {
        Self::Placement(code)
    }
    /// Construct an action failure.
    pub const fn action(code: ActionFailureCode) -> Self {
        Self::Action(code)
    }
    /// Construct an action state-budget failure.
    pub const fn state_budget(code: StateBudgetFailureCode) -> Self {
        Self::StateBudget(code)
    }
    const fn failure_stage(&self) -> StreamingFailureStage {
        match self {
            Self::Placement(_) => StreamingFailureStage::Placement,
            Self::Action(_) => StreamingFailureStage::Dispatch,
            Self::StateBudget(_) => StreamingFailureStage::StateBudget,
        }
    }
    const fn failure_code(&self) -> &'static str {
        match self {
            Self::Placement(code) => code.code(),
            Self::Action(code) => code.code(),
            Self::StateBudget(code) => budget_code(*code),
        }
    }
}

macro_rules! impl_failure {
    ($error:ty) => {
        impl fmt::Display for $error {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str(self.failure_code())
            }
        }
        impl std::error::Error for $error {}
        impl StableStreamingFailure for $error {
            fn stage(&self) -> StreamingFailureStage {
                self.failure_stage()
            }
            fn code(&self) -> &'static str {
                self.failure_code()
            }
        }
    };
}

impl_failure!(StreamSourceError);
impl_failure!(StreamFormatError);
impl_failure!(SessionCoordinatorError);
impl_failure!(ActionExecutionError);

impl StableStreamingFailure for CheckpointError {
    fn stage(&self) -> StreamingFailureStage {
        match self {
            Self::StateBudget { .. } => StreamingFailureStage::StateBudget,
            Self::ResultIndexReadBudgetTooSmall { .. } => StreamingFailureStage::Result,
            _ => StreamingFailureStage::Checkpoint,
        }
    }

    fn code(&self) -> &'static str {
        match self {
            Self::AlreadyInitialized => "already_initialized",
            Self::GenerationConflict { .. } => "generation_conflict",
            Self::ParticipantSetMismatch => "participant_set_mismatch",
            Self::CutBlockedByInflight { .. } => "cut_blocked_by_inflight",
            Self::StateBudget { code, .. } => budget_code(*code),
            Self::BackendBudget { code, .. } => match code {
                super::checkpoint::CheckpointBackendBudgetFailureCode::ItemCapacity => {
                    "backend_item_capacity"
                }
                super::checkpoint::CheckpointBackendBudgetFailureCode::ByteCapacity => {
                    "backend_byte_capacity"
                }
                super::checkpoint::CheckpointBackendBudgetFailureCode::Closed => "backend_closed",
                super::checkpoint::CheckpointBackendBudgetFailureCode::Unrepresentable => {
                    "backend_unrepresentable"
                }
            },
            Self::ResultIndexReadBudgetTooSmall { .. } => "result_index_read_budget_too_small",
            Self::GenerationEpochOverflow { .. } => "generation_epoch_overflow",
            Self::DecodeHorizonRegression { .. } => "decode_horizon_regression",
            Self::ParticipantUnavailable { .. } => "participant_unavailable",
            Self::ObjectVerification => "object_verification",
            Self::LeaseLost { .. } => "lease_lost",
            Self::PostCommitNotification { .. } => "post_commit_notification",
            Self::SourceUnavailableOnResume => "source_unavailable_on_resume",
            Self::Storage { .. } => "storage",
        }
    }
}

/// Frozen identity of one stream and its exact immutable source authority.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
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

    /// Borrow the stream semantic identity.
    #[must_use]
    pub const fn stream_identity(&self) -> &ContentDigest {
        &self.stream_identity
    }

    /// Borrow the immutable source authority identity.
    #[must_use]
    pub const fn source_identity(&self) -> &ImmutableObjectIdentity {
        &self.source_identity
    }
}

/// Scope whose ordinary work observed an issue.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StreamingIssueScope {
    /// Run-wide source or policy work.
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
    /// One checkpoint attempt identified by its successor epoch.
    CheckpointAttempt { epoch: u64 },
}

/// Reliability class asserted by a closed typed ordinary failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StreamingIssueClass {
    /// The same operation may succeed after bounded retry.
    Retryable,
    /// The named input cannot succeed under the validated plan.
    Permanent,
    /// The observation contradicts a runtime invariant.
    Invariant,
    /// Explicit bounded capacity is currently unavailable.
    Capacity,
}

/// Stable order domain for deterministic issue handling.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum StreamingIssueOrderKey {
    /// Run-scoped diagnostic order.
    Run {
        /// Host-assigned retry ordinal.
        retry_ordinal: u32,
        /// Stable scope-specific tie breaker.
        scope_tiebreaker: ContentDigest,
    },
    /// Partition, record, or session input-domain order.
    Input {
        /// Exact input domain repeated from the issue scope.
        input_domain: StreamingInputDomainIdentity,
        /// Stable source-domain position.
        source_position: SourcePosition,
        /// Host-assigned retry ordinal.
        retry_ordinal: u32,
        /// Stable scope-specific tie breaker.
        scope_tiebreaker: ContentDigest,
    },
    /// Host-assigned action sequence order.
    Action {
        /// Dense global action sequence.
        global_sequence: GlobalSequence,
        /// Host-assigned retry ordinal.
        retry_ordinal: u32,
        /// Stable scope-specific tie breaker.
        scope_tiebreaker: ContentDigest,
    },
    /// Checkpoint-attempt order.
    Checkpoint {
        /// Successor checkpoint epoch.
        epoch: u64,
        /// Host-assigned retry ordinal.
        retry_ordinal: u32,
        /// Stable scope-specific tie breaker.
        scope_tiebreaker: ContentDigest,
    },
}

impl StreamingIssueOrderKey {
    /// Construct deterministic order within one exact input domain.
    #[must_use]
    pub const fn input(
        input_domain: StreamingInputDomainIdentity,
        source_position: SourcePosition,
        retry_ordinal: u32,
        scope_tiebreaker: ContentDigest,
    ) -> Self {
        Self::Input {
            input_domain,
            source_position,
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
        Self::Action {
            global_sequence,
            retry_ordinal,
            scope_tiebreaker,
        }
    }
}

/// Closed ordinary failures accepted from source, format, session, and action adapters.
#[derive(Debug, Eq, PartialEq)]
pub enum OrdinaryStreamingFailure {
    /// Source-owned failure.
    Source(StreamSourceError),
    /// Format-owned failure.
    Format(StreamFormatError),
    /// Session-program failure.
    Session(SessionCoordinatorError),
    /// Action-binding failure.
    Action(ActionExecutionError),
}

impl OrdinaryStreamingFailure {
    /// Stable stage retaining authority for this typed failure.
    #[must_use]
    pub fn stage(&self) -> StreamingFailureStage {
        match self {
            Self::Source(error) => error.stage(),
            Self::Format(error) => error.stage(),
            Self::Session(error) => error.stage(),
            Self::Action(error) => error.stage(),
        }
    }

    /// Stable closed failure code.
    #[must_use]
    pub fn code(&self) -> &'static str {
        match self {
            Self::Source(error) => error.code(),
            Self::Format(error) => error.code(),
            Self::Session(error) => error.code(),
            Self::Action(error) => error.code(),
        }
    }
}

/// Invalid combination of an issue scope and deterministic order domain.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StreamingIssueValidationError {
    /// The chosen order key cannot order the chosen scope.
    OrderScopeMismatch,
    /// Controlled stop is host control, not an ordinary adapter issue.
    ControlledStopIsNotOrdinary,
    /// Invariant classification is reserved for the private host classifier.
    InvariantIsHostOwned,
}

impl fmt::Display for StreamingIssueValidationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid streaming issue: {self:?}")
    }
}

impl std::error::Error for StreamingIssueValidationError {}

/// Move-only typed ordinary issue submitted to the host reliability owner.
#[derive(Debug, Eq, PartialEq)]
pub struct OrdinaryStreamingIssue {
    run: StreamRunIdentity,
    scope: StreamingIssueScope,
    class: StreamingIssueClass,
    semantic_context_digest: ContentDigest,
    order: StreamingIssueOrderKey,
    failure: OrdinaryStreamingFailure,
}

impl OrdinaryStreamingIssue {
    /// Validate and bind typed issue facts without selecting a host disposition.
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
        let has_compatible_order = match (&scope, &order) {
            (StreamingIssueScope::Run, StreamingIssueOrderKey::Run { .. })
            | (StreamingIssueScope::Action { .. }, StreamingIssueOrderKey::Action { .. }) => true,
            (
                StreamingIssueScope::Partition {
                    input_domain: scope_domain,
                    ..
                }
                | StreamingIssueScope::Record {
                    input_domain: scope_domain,
                    ..
                }
                | StreamingIssueScope::Session {
                    input_domain: scope_domain,
                    ..
                },
                StreamingIssueOrderKey::Input {
                    input_domain: order_domain,
                    ..
                },
            ) => scope_domain == order_domain,
            (
                StreamingIssueScope::CheckpointAttempt { epoch: scope_epoch },
                StreamingIssueOrderKey::Checkpoint {
                    epoch: order_epoch, ..
                },
            ) => scope_epoch == order_epoch,
            _ => false,
        };
        if !has_compatible_order {
            return Err(StreamingIssueValidationError::OrderScopeMismatch);
        }
        Ok(Self {
            run,
            scope,
            class,
            semantic_context_digest,
            order,
            failure,
        })
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
    /// Borrow the semantic context digest.
    #[must_use]
    pub const fn semantic_context_digest(&self) -> &ContentDigest {
        &self.semantic_context_digest
    }
    /// Return the deterministic order key.
    #[must_use]
    pub const fn order(&self) -> &StreamingIssueOrderKey {
        &self.order
    }
    /// Borrow the closed typed failure.
    #[must_use]
    pub const fn failure(&self) -> &OrdinaryStreamingFailure {
        &self.failure
    }
}

/// Fixed-size non-authoritative result of submitting one ordinary issue.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StreamingIssueReportStatus {
    /// The host retained the issue for ordered classification.
    Accepted,
    /// The bounded host input ledger has no current capacity.
    Backpressured,
}

/// Closed reporter submission failure.
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

/// Worker-local host endpoint behind the cloneable reporter handle.
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

/// Sole mutable host reliability owner and checkpoint participant.
pub trait StreamingIssueReporter: StreamingCheckpointParticipant {
    /// Return the cloneable adapter injection handle.
    fn handle(&self) -> StreamingIssueReporterHandle;
}
