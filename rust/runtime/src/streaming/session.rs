// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Run-scoped session coordination and bounded action-output contracts.

use std::any::Any;

use async_trait::async_trait;
use serde_json::value::RawValue;

use super::{
    action::ActionExecutionEvent,
    checkpoint::StreamingCheckpointParticipant,
    format::{SessionWatermark, StreamingFormatDescriptor},
    identity::{ContentDigest, SessionCausalFrontier},
    source::SourceSeal,
    unit::{ExecutableDatasetAction, StreamingSessionFragment},
};

pub use super::failure::{SessionCoordinatorError, SessionFailureCode};

/// Immutable registry metadata for one session-program implementation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StreamingSessionProgramDescriptor {
    /// Stable registry identifier.
    pub id: &'static str,
    /// Human-readable implementation description.
    pub description: &'static str,
    /// Action schemas the program can emit.
    pub action_schemas: &'static [&'static str],
}

/// Type-erased, strictly validated session-program configuration.
pub trait ValidatedStreamingSessionProgramConfig: std::fmt::Debug + Send + Sync {
    /// Borrow the concrete startup-only value.
    fn as_any(&self) -> &dyn Any;

    /// Consume the concrete startup-only value.
    fn into_any(self: Box<Self>) -> Box<dyn Any + Send + Sync>;
}

impl<T> ValidatedStreamingSessionProgramConfig for T
where
    T: Any + std::fmt::Debug + Send + Sync,
{
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn into_any(self: Box<Self>) -> Box<dyn Any + Send + Sync> {
        self
    }
}

/// Host-owned session-program preparation context.
#[derive(Clone, Debug)]
pub struct StreamingSessionPrepareContext {
    /// Semantic digest of the complete validated session program.
    pub program_semantic_digest: ContentDigest,
}

/// Startup session-program validation and preparation contract.
pub trait StreamingSessionProgramFactory: std::fmt::Debug + Send + Sync {
    /// Describe the exact compiled session program.
    fn descriptor(&self) -> &'static StreamingSessionProgramDescriptor;

    /// Strictly decode and validate session-owned configuration.
    fn validate(
        &self,
        authored: &RawValue,
        format: &StreamingFormatDescriptor,
        workload: &crate::engine::registry::WorkloadDescriptor,
    ) -> Result<Box<dyn ValidatedStreamingSessionProgramConfig>, SessionCoordinatorError>;

    /// Prepare one run-scoped session coordinator.
    fn prepare(
        &self,
        config: Box<dyn ValidatedStreamingSessionProgramConfig>,
        context: &StreamingSessionPrepareContext,
    ) -> Result<Box<dyn StreamingSessionCoordinator>, SessionCoordinatorError>;
}

/// Sole run-scoped owner of canonical cross-partition session state.
#[async_trait(?Send)]
pub trait StreamingSessionCoordinator: StreamingCheckpointParticipant {
    /// Incorporate one canonical fragment and emit newly causal-ready actions.
    async fn ingest(
        &mut self,
        fragment: StreamingSessionFragment,
        output: &mut dyn DatasetActionSink,
    ) -> Result<(), SessionCoordinatorError>;

    /// Advance format-proven session completeness.
    async fn advance_watermark(
        &mut self,
        watermark: SessionWatermark,
        output: &mut dyn DatasetActionSink,
    ) -> Result<(), SessionCoordinatorError>;

    /// Incorporate one execution event into canonical session state.
    async fn observe_execution(
        &mut self,
        event: ActionExecutionEvent,
        output: &mut dyn DatasetActionSink,
    ) -> Result<(), SessionCoordinatorError>;

    /// Validate source exhaustion and seal session state.
    async fn seal(
        &mut self,
        seal: SourceSeal,
        output: &mut dyn DatasetActionSink,
    ) -> Result<SessionSealReceipt, SessionCoordinatorError>;
}

/// Bounded asynchronous output owned by the host pipeline.
#[async_trait(?Send)]
pub trait DatasetActionSink {
    /// Send one causally ready move-only action.
    async fn send_action(
        &mut self,
        action: ExecutableDatasetAction,
    ) -> Result<(), SessionCoordinatorError>;

    /// Advance the monotonic causal completeness proof.
    async fn advance_causal_frontier(
        &mut self,
        frontier: SessionCausalFrontier,
    ) -> Result<(), SessionCoordinatorError>;
}

/// Receipt returned after the session coordinator accepts a source seal.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SessionSealReceipt {
    /// Digest binding final canonical session state.
    pub digest: ContentDigest,
    /// Final causal frontier.
    pub causal_frontier: SessionCausalFrontier,
}
