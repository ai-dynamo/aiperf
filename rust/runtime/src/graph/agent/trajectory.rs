// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded worker-local trajectory facts retained by agent drivers.

use std::cell::RefCell;
use std::error::Error;
use std::fmt::{self, Display};

use bytes::Bytes;

use crate::graph::tools::ToolDispatchResult;

use super::{AgentResponseHandle, AgentResponseSource};

/// Completed trace facts retained without command text or endpoint secrets.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AgentTrajectory {
    /// Run correlation identity; distinct from the document and invocation ids.
    pub run_id: String,
    /// Stable trajectory document identity.
    pub trajectory_id: String,
    /// Stable root invocation identity.
    pub invocation_id: String,
    /// Selected response wires in dispatch order.
    pub dispatched_response_wires: Vec<Bytes>,
    /// Turn indexes that reused earlier response bytes as copied context.
    pub copied_context_turns: Vec<usize>,
    /// Correlated terminal tool outcomes in dispatch order.
    pub tool_results: Vec<ToolDispatchResult>,
    /// Formatted tool observations supplied to the subsequent request build.
    pub observations: Vec<Bytes>,
    /// Typed response ownership retained beside every selected wire.
    pub responses: Vec<AgentTrajectoryResponse>,
}

/// One selected response retained in the trajectory without reserializing it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AgentTrajectoryResponse {
    /// Content-addressed response reference.
    pub handle: AgentResponseHandle,
    /// Selection provenance.
    pub source: AgentResponseSource,
    /// Logical turn that selected this response.
    pub logical_turn: usize,
    /// Whether this dispatch consumed copied context rather than a new generation.
    pub is_copied_context: bool,
}

/// Trajectory persistence failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AgentTrajectoryError(String);

impl Display for AgentTrajectoryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for AgentTrajectoryError {}

/// Worker-local sink for one trace trajectory.
pub trait AgentTrajectorySink {
    /// Append a selected response wire.
    fn append_response(
        &mut self,
        response: AgentTrajectoryResponse,
        wire: Bytes,
    ) -> Result<(), AgentTrajectoryError>;
    /// Append one correlated terminal tool result.
    fn append_tool_result(
        &mut self,
        result: ToolDispatchResult,
    ) -> Result<(), AgentTrajectoryError>;
    /// Append ordered observation bytes before the next request is materialized.
    fn append_observations(&mut self, observations: Vec<Bytes>)
    -> Result<(), AgentTrajectoryError>;
    /// Snapshot the accumulated bounded trajectory facts.
    fn snapshot(&self) -> AgentTrajectory;
}

/// In-memory trajectory sink for deterministic driver tests.
#[derive(Default)]
pub struct InMemoryAgentTrajectorySink {
    trajectory: RefCell<AgentTrajectory>,
}

impl AgentTrajectorySink for InMemoryAgentTrajectorySink {
    fn append_response(
        &mut self,
        response: AgentTrajectoryResponse,
        wire: Bytes,
    ) -> Result<(), AgentTrajectoryError> {
        let mut trajectory = self.trajectory.borrow_mut();
        trajectory.dispatched_response_wires.push(wire);
        if response.is_copied_context {
            let turn = trajectory.dispatched_response_wires.len().saturating_sub(1);
            trajectory.copied_context_turns.push(turn);
        }
        trajectory.responses.push(response);
        Ok(())
    }

    fn append_tool_result(
        &mut self,
        result: ToolDispatchResult,
    ) -> Result<(), AgentTrajectoryError> {
        self.trajectory.borrow_mut().tool_results.push(result);
        Ok(())
    }

    fn append_observations(
        &mut self,
        observations: Vec<Bytes>,
    ) -> Result<(), AgentTrajectoryError> {
        self.trajectory
            .borrow_mut()
            .observations
            .extend(observations);
        Ok(())
    }

    fn snapshot(&self) -> AgentTrajectory {
        self.trajectory.borrow().clone()
    }
}

/// Factory creating one trajectory sink per admitted trace.
pub trait AgentTrajectorySinkFactory: Send + Sync {
    /// Create a fresh worker-local sink for the trace identities.
    fn create(
        &self,
        run_id: &str,
        trajectory_id: &str,
        invocation_id: &str,
    ) -> Result<Box<dyn AgentTrajectorySink>, AgentTrajectoryError>;
}

/// Stock in-memory trajectory-sink factory.
#[derive(Clone, Copy, Debug, Default)]
pub struct InMemoryAgentTrajectorySinkFactory;

impl AgentTrajectorySinkFactory for InMemoryAgentTrajectorySinkFactory {
    fn create(
        &self,
        _run_id: &str,
        _trajectory_id: &str,
        _invocation_id: &str,
    ) -> Result<Box<dyn AgentTrajectorySink>, AgentTrajectoryError> {
        Ok(Box::new(InMemoryAgentTrajectorySink::default()))
    }
}

/// Decoder for an authored trajectory selected by a continuation.
pub trait AgentTrajectoryCodec: Send + Sync {
    /// Decode a registered trajectory format into its normalized representation.
    fn decode(&self, bytes: Bytes) -> Result<AgentTrajectory, AgentTrajectoryError>;
}

/// Deterministic codec used to prove continuation registration without parsing sessions.
#[derive(Clone, Copy, Debug, Default)]
pub struct InMemoryAgentTrajectoryCodec;

impl AgentTrajectoryCodec for InMemoryAgentTrajectoryCodec {
    fn decode(&self, bytes: Bytes) -> Result<AgentTrajectory, AgentTrajectoryError> {
        if bytes.is_empty() {
            return Err(AgentTrajectoryError(
                "agent trajectory input is empty".into(),
            ));
        }
        Ok(AgentTrajectory::default())
    }
}
