// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Stable failure stages and codes shared by streaming extension contracts.

use std::{fmt, rc::Rc};

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
            const fn code(self) -> &'static str {
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
        /// Host control requested source shutdown.
        Stopped => "stopped",
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

/// Source discovery and immutable-acquisition error.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StreamSourceError {
    /// Source-owned discovery or snapshot failure.
    Source {
        /// Stable failure classification.
        code: SourceFailureCode,
        /// Bounded human-readable context.
        message: String,
    },
    /// Immutable partition acquisition failure.
    Acquisition {
        /// Stable failure classification.
        code: AcquisitionFailureCode,
        /// Bounded human-readable context.
        message: String,
    },
    /// Source-owned ordering or frontier failure.
    Ordering {
        /// Stable failure classification.
        code: OrderingFailureCode,
        /// Bounded human-readable context.
        message: String,
    },
    /// Host control stopped a pending source.
    Stopped {
        /// Bounded human-readable context.
        message: String,
    },
}

impl StreamSourceError {
    /// Construct a source-discovery failure.
    pub fn source(code: SourceFailureCode, message: impl Into<String>) -> Self {
        Self::Source {
            code,
            message: message.into(),
        }
    }

    /// Construct an immutable-acquisition failure.
    pub fn acquisition(code: AcquisitionFailureCode, message: impl Into<String>) -> Self {
        Self::Acquisition {
            code,
            message: message.into(),
        }
    }

    /// Construct a source-ordering failure.
    pub fn ordering(code: OrderingFailureCode, message: impl Into<String>) -> Self {
        Self::Ordering {
            code,
            message: message.into(),
        }
    }

    /// Construct a host-requested stop outcome.
    pub fn stopped(message: impl Into<String>) -> Self {
        Self::Stopped {
            message: message.into(),
        }
    }

    fn failure_stage(&self) -> StreamingFailureStage {
        match self {
            Self::Source { .. } | Self::Stopped { .. } => StreamingFailureStage::Source,
            Self::Acquisition { .. } => StreamingFailureStage::Acquisition,
            Self::Ordering { .. } => StreamingFailureStage::Ordering,
        }
    }

    fn failure_code(&self) -> &'static str {
        match self {
            Self::Source { code, .. } => code.code(),
            Self::Acquisition { code, .. } => code.code(),
            Self::Ordering { code, .. } => code.code(),
            Self::Stopped { .. } => SourceFailureCode::Stopped.code(),
        }
    }

    fn message(&self) -> &str {
        match self {
            Self::Source { message, .. }
            | Self::Acquisition { message, .. }
            | Self::Ordering { message, .. }
            | Self::Stopped { message } => message,
        }
    }
}

/// Format decode, ordering, or state-budget error.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StreamFormatError {
    /// Decoder-owned format failure.
    Decode {
        /// Stable failure classification.
        code: DecodeFailureCode,
        /// Bounded human-readable context.
        message: String,
    },
    /// Format-owned ordering or frontier failure.
    Ordering {
        /// Stable failure classification.
        code: OrderingFailureCode,
        /// Bounded human-readable context.
        message: String,
    },
    /// Decoder state exceeded an explicit capacity.
    StateBudget {
        /// Stable failure classification.
        code: StateBudgetFailureCode,
        /// Bounded human-readable context.
        message: String,
    },
}

impl StreamFormatError {
    /// Construct a decode failure.
    pub fn decode(code: DecodeFailureCode, message: impl Into<String>) -> Self {
        Self::Decode {
            code,
            message: message.into(),
        }
    }

    /// Construct an ordering failure.
    pub fn ordering(code: OrderingFailureCode, message: impl Into<String>) -> Self {
        Self::Ordering {
            code,
            message: message.into(),
        }
    }

    /// Construct a state-budget failure.
    pub fn state_budget(code: StateBudgetFailureCode, message: impl Into<String>) -> Self {
        Self::StateBudget {
            code,
            message: message.into(),
        }
    }

    fn failure_stage(&self) -> StreamingFailureStage {
        match self {
            Self::Decode { .. } => StreamingFailureStage::Decode,
            Self::Ordering { .. } => StreamingFailureStage::Ordering,
            Self::StateBudget { .. } => StreamingFailureStage::StateBudget,
        }
    }

    fn failure_code(&self) -> &'static str {
        match self {
            Self::Decode { code, .. } => code.code(),
            Self::Ordering { code, .. } => code.code(),
            Self::StateBudget { code, .. } => budget_code(*code),
        }
    }

    fn message(&self) -> &str {
        match self {
            Self::Decode { message, .. }
            | Self::Ordering { message, .. }
            | Self::StateBudget { message, .. } => message,
        }
    }
}

/// Session coordination or retained-state error.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SessionCoordinatorError {
    /// Session-program semantic failure.
    Session {
        /// Stable failure classification.
        code: SessionFailureCode,
        /// Bounded human-readable context.
        message: String,
    },
    /// Session state exceeded an explicit capacity.
    StateBudget {
        /// Stable failure classification.
        code: StateBudgetFailureCode,
        /// Bounded human-readable context.
        message: String,
    },
}

impl SessionCoordinatorError {
    /// Construct a session semantic failure.
    pub fn session(code: SessionFailureCode, message: impl Into<String>) -> Self {
        Self::Session {
            code,
            message: message.into(),
        }
    }

    /// Construct a session state-budget failure.
    pub fn state_budget(code: StateBudgetFailureCode, message: impl Into<String>) -> Self {
        Self::StateBudget {
            code,
            message: message.into(),
        }
    }

    fn failure_stage(&self) -> StreamingFailureStage {
        match self {
            Self::Session { .. } => StreamingFailureStage::Session,
            Self::StateBudget { .. } => StreamingFailureStage::StateBudget,
        }
    }

    fn failure_code(&self) -> &'static str {
        match self {
            Self::Session { code, .. } => code.code(),
            Self::StateBudget { code, .. } => budget_code(*code),
        }
    }

    fn message(&self) -> &str {
        match self {
            Self::Session { message, .. } | Self::StateBudget { message, .. } => message,
        }
    }
}

/// Action binding, placement, dispatch, or retained-state error.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ActionExecutionError {
    /// Placement authority refused the action.
    Placement {
        /// Stable failure classification.
        code: PlacementFailureCode,
        /// Bounded human-readable context.
        message: String,
    },
    /// The selected binding or dispatch path failed.
    Action {
        /// Stable failure classification.
        code: ActionFailureCode,
        /// Bounded human-readable context.
        message: String,
    },
    /// Action state exceeded an explicit capacity.
    StateBudget {
        /// Stable failure classification.
        code: StateBudgetFailureCode,
        /// Bounded human-readable context.
        message: String,
    },
}

impl ActionExecutionError {
    /// Construct a placement failure.
    pub fn placement(code: PlacementFailureCode, message: impl Into<String>) -> Self {
        Self::Placement {
            code,
            message: message.into(),
        }
    }

    /// Construct an action failure.
    pub fn action(code: ActionFailureCode, message: impl Into<String>) -> Self {
        Self::Action {
            code,
            message: message.into(),
        }
    }

    /// Construct an action state-budget failure.
    pub fn state_budget(code: StateBudgetFailureCode, message: impl Into<String>) -> Self {
        Self::StateBudget {
            code,
            message: message.into(),
        }
    }

    fn failure_stage(&self) -> StreamingFailureStage {
        match self {
            Self::Placement { .. } => StreamingFailureStage::Placement,
            Self::Action { .. } => StreamingFailureStage::Dispatch,
            Self::StateBudget { .. } => StreamingFailureStage::StateBudget,
        }
    }

    fn failure_code(&self) -> &'static str {
        match self {
            Self::Placement { code, .. } => code.code(),
            Self::Action { code, .. } => code.code(),
            Self::StateBudget { code, .. } => budget_code(*code),
        }
    }

    fn message(&self) -> &str {
        match self {
            Self::Placement { message, .. }
            | Self::Action { message, .. }
            | Self::StateBudget { message, .. } => message,
        }
    }
}

macro_rules! impl_failure {
    ($error:ty) => {
        impl fmt::Display for $error {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(formatter, "{}: {}", self.failure_code(), self.message())
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
            Self::BackendBudget { .. } => "backend_budget",
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

/// Immutable issue record accepted by the host reliability boundary.
#[derive(Debug, Eq, PartialEq)]
pub struct StreamingIssue {
    /// Stable stage retaining failure authority.
    pub stage: StreamingFailureStage,
    /// Stable machine-readable code.
    pub code: &'static str,
    /// Bounded human-readable context.
    pub message: String,
}

impl StreamingIssue {
    /// Capture the stable identity and display context of one failure.
    #[must_use]
    pub fn from_failure(failure: &dyn StableStreamingFailure) -> Self {
        Self {
            stage: failure.stage(),
            code: failure.code(),
            message: failure.to_string(),
        }
    }
}

/// Host-owned sink behind the source/format issue reporting handle.
pub trait StreamingIssueReporterOps {
    /// Accept one typed issue without selecting retry or shutdown policy.
    fn report(&self, issue: StreamingIssue);
}

/// Cloneable opaque injection handle for host-owned reliability reporting.
#[derive(Clone)]
pub struct StreamingIssueReporter {
    inner: Rc<dyn StreamingIssueReporterOps>,
}

impl StreamingIssueReporter {
    /// Erase one worker-local host reporting implementation.
    #[must_use]
    pub fn new<T>(reporter: T) -> Self
    where
        T: StreamingIssueReporterOps + 'static,
    {
        Self {
            inner: Rc::new(reporter),
        }
    }

    /// Forward one typed issue without applying reliability policy locally.
    pub fn report(&self, issue: StreamingIssue) {
        self.inner.report(issue);
    }
}

impl fmt::Debug for StreamingIssueReporter {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("StreamingIssueReporter")
            .finish_non_exhaustive()
    }
}
