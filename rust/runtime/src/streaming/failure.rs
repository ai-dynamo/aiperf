// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Stable failure stages and codes shared by streaming extension contracts.

use std::fmt;

use serde::{Deserialize, Serialize};

use super::{checkpoint::CheckpointError, unit::StateBudgetFailureCode};

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
        /// A sequential reader ended before its immutable advertised length.
        TruncatedObject => "truncated_object",
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

pub use super::reliability::{
    OrdinaryStreamingIssue, StreamingInputDomainIdentity, StreamingIssueClass,
    StreamingIssueOrderKey, StreamingIssueReportError, StreamingIssueReportStatus,
    StreamingIssueReporter, StreamingIssueReporterEndpoint, StreamingIssueReporterHandle,
    StreamingIssueScope, StreamingIssueValidationError,
};
