// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Deterministic agent-turn coordination over the worker-local seams.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::{self, Display};

use async_trait::async_trait;
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::graph::tools::{
    AgentObservationFormatter, AgentToolCallDecoder, ToolDispatchError, ToolDispatcher,
};

use super::{
    AgentInvocationIdentity, AgentResponseHandle, AgentResponseSource, AgentResponseStore,
    AgentResponseStoreError, AgentTrajectory, AgentTrajectoryError, AgentTrajectoryResponse,
    AgentTrajectorySink, InvocationLeaseFactory,
};

/// Explicit selection of the response bytes used by one turn.
#[derive(Clone, Debug)]
pub enum ResponseSelection {
    /// Bytes selected immediately and interned before trajectory append.
    Inline {
        /// Origin of selected bytes.
        source: AgentResponseSource,
        /// Immutable wire bytes.
        wire: Bytes,
    },
    /// Reuse response bytes selected by an earlier turn.
    Reuse(AgentResponseHandle),
}

/// One deterministic turn decision for the frozen replay seam.
#[derive(Clone, Debug)]
pub struct AgentTurn {
    selection: ResponseSelection,
    is_copied_context: bool,
}

impl AgentTurn {
    /// Construct one selected response and its sequential tool calls.
    pub fn new(selection: ResponseSelection, is_copied_context: bool) -> Self {
        Self {
            selection,
            is_copied_context,
        }
    }
}

/// Driver-loop failure with explicit source boundary context.
#[derive(Debug)]
pub struct AgentLoopError(String);

impl AgentLoopError {
    /// Build an explicit agent-loop boundary failure.
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl Display for AgentLoopError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for AgentLoopError {}

impl From<AgentResponseStoreError> for AgentLoopError {
    fn from(error: AgentResponseStoreError) -> Self {
        Self(error.to_string())
    }
}

impl From<AgentTrajectoryError> for AgentLoopError {
    fn from(error: AgentTrajectoryError) -> Self {
        Self(error.to_string())
    }
}

impl From<ToolDispatchError> for AgentLoopError {
    fn from(error: ToolDispatchError) -> Self {
        Self(error.to_string())
    }
}

/// Coordinates predetermined response selections and sequential tool calls.
#[async_trait(?Send)]
pub trait AgentTurnCoordinator {
    /// Execute the trace-local agent turns through injected worker-local seams.
    async fn run(
        &mut self,
        response_store: &mut dyn AgentResponseStore,
        trajectory: &mut dyn AgentTrajectorySink,
        leases: &dyn InvocationLeaseFactory,
        dispatcher: &dyn ToolDispatcher,
        decoder: &dyn AgentToolCallDecoder,
        formatter: &dyn AgentObservationFormatter,
    ) -> Result<AgentTrajectory, AgentLoopError>;
}

/// Strict serializable selector for one registered turn coordinator.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct AgentTurnCoordinatorSpec {
    /// Registered coordinator identifier.
    pub kind: String,
    /// Coordinator-specific, prevalidated data.
    #[serde(default)]
    pub data: BTreeMap<String, Value>,
}

/// Frozen factory creating one coordinator for every root or child invocation.
pub trait AgentTurnCoordinatorFactory: Send + Sync {
    /// Create one fresh, non-shared coordinator.
    fn create(
        &self,
        invocation: &AgentInvocationIdentity,
        spec: &AgentTurnCoordinatorSpec,
    ) -> Result<Box<dyn AgentTurnCoordinator>, AgentLoopError>;
}

/// Deterministic coordinator used by recorded replay and contract tests.
pub struct StaticAgentTurnCoordinator {
    turns: Vec<AgentTurn>,
}

/// In-memory factory retaining deterministic turns solely for seam tests.
#[derive(Clone, Debug)]
pub struct StaticAgentTurnCoordinatorFactory {
    turns: Vec<AgentTurn>,
}

impl Default for StaticAgentTurnCoordinatorFactory {
    fn default() -> Self {
        Self::new([])
    }
}

impl StaticAgentTurnCoordinatorFactory {
    /// Construct a factory whose each invocation receives an independent turn cursor.
    pub fn new(turns: impl IntoIterator<Item = AgentTurn>) -> Self {
        Self {
            turns: turns.into_iter().collect(),
        }
    }
}

impl AgentTurnCoordinatorFactory for StaticAgentTurnCoordinatorFactory {
    fn create(
        &self,
        _invocation: &AgentInvocationIdentity,
        spec: &AgentTurnCoordinatorSpec,
    ) -> Result<Box<dyn AgentTurnCoordinator>, AgentLoopError> {
        if spec.kind != "static" || !spec.data.is_empty() {
            return Err(AgentLoopError::new(format!(
                "static coordinator cannot create {:?}",
                spec.kind
            )));
        }
        Ok(Box::new(StaticAgentTurnCoordinator::new(
            self.turns.clone(),
        )))
    }
}

impl StaticAgentTurnCoordinator {
    /// Construct a finite authored turn sequence.
    pub fn new(turns: impl IntoIterator<Item = AgentTurn>) -> Self {
        Self {
            turns: turns.into_iter().collect(),
        }
    }
}

#[async_trait(?Send)]
impl AgentTurnCoordinator for StaticAgentTurnCoordinator {
    async fn run(
        &mut self,
        response_store: &mut dyn AgentResponseStore,
        trajectory: &mut dyn AgentTrajectorySink,
        leases: &dyn InvocationLeaseFactory,
        dispatcher: &dyn ToolDispatcher,
        decoder: &dyn AgentToolCallDecoder,
        formatter: &dyn AgentObservationFormatter,
    ) -> Result<AgentTrajectory, AgentLoopError> {
        let _lease = leases.acquire();
        let mut selected = Vec::<AgentResponseHandle>::with_capacity(self.turns.len());
        for (turn_index, turn) in self.turns.iter().enumerate() {
            let (handle, source, is_reused) = match &turn.selection {
                ResponseSelection::Inline { source, wire } => (
                    response_store.intern(source.clone(), wire.clone())?,
                    source.clone(),
                    false,
                ),
                ResponseSelection::Reuse(handle) => {
                    if !selected.contains(handle) {
                        return Err(AgentLoopError(format!(
                            "turn {turn_index} reuses a response not selected by this invocation"
                        )));
                    }
                    (
                        handle.clone(),
                        AgentResponseSource::Reused {
                            original_turn: selected
                                .iter()
                                .position(|selected_handle| selected_handle == handle)
                                .unwrap_or_default(),
                        },
                        true,
                    )
                }
            };
            let wire = response_store.get(&handle)?;
            trajectory.append_response(
                AgentTrajectoryResponse {
                    handle: handle.clone(),
                    source,
                    logical_turn: turn_index,
                    is_copied_context: is_reused || turn.is_copied_context,
                },
                wire.clone(),
            )?;
            selected.push(handle);
            let calls = decoder.decode(&wire)?;
            let mut results = Vec::with_capacity(calls.len());
            for call in &calls {
                let result = dispatcher.dispatch(call.dispatch_request()).await?;
                trajectory.append_tool_result(result.clone())?;
                results.push(result);
            }
            let observations = formatter.format(&calls, &results)?;
            trajectory.append_observations(observations.clone())?;
            if turn_index + 1 < self.turns.len() {
                trajectory.append_subsequent_dispatch_prompt(build_subsequent_dispatch_prompt(
                    &wire,
                    &observations,
                )?)?;
            }
        }
        Ok(trajectory.snapshot())
    }
}

fn build_subsequent_dispatch_prompt(
    selected_response: &Bytes,
    observations: &[Bytes],
) -> Result<Bytes, AgentLoopError> {
    let observation_bytes = observations.iter().try_fold(0usize, |total, observation| {
        total
            .checked_add(observation.len())
            .ok_or_else(|| AgentLoopError::new("agent observation prompt length overflow"))
    })?;
    let separators = observations.len();
    let capacity = selected_response
        .len()
        .checked_add(observation_bytes)
        .and_then(|total| total.checked_add(separators))
        .ok_or_else(|| AgentLoopError::new("agent dispatch prompt length overflow"))?;
    let mut prompt = Vec::with_capacity(capacity);
    prompt.extend_from_slice(selected_response);
    for observation in observations {
        prompt.push(b'\n');
        prompt.extend_from_slice(observation);
    }
    Ok(Bytes::from(prompt))
}
