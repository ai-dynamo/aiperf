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

use crate::dataset::Handle;
use crate::graph::tools::{
    AgentObservationFormatter, AgentToolCallDecoder, ToolDispatchContext, ToolDispatchError,
};

use super::{
    AgentInvocationIdentity, AgentInvocationLease, AgentResponseHandle, AgentResponseSource,
    AgentResponseStore, AgentResponseStoreError, AgentTrajectory, AgentTrajectoryError,
    AgentTrajectoryResponse, AgentTrajectorySink, InvocationLeaseFactory,
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

/// One bounded live action selected by an agent turn coordinator.
#[derive(Clone, Debug)]
pub enum LiveAgentTurnDirective {
    /// Dispatch a declared model binding through Rust-owned model authority.
    DispatchModel {
        /// Validated model binding identifier.
        binding: String,
        /// Opaque prompt segment handle.
        prompt: Handle,
    },
    /// Invoke one declared supervised tool adapter.
    InvokeTool {
        /// Validated adapter identifier.
        adapter: String,
        /// Opaque tool-input segment handle.
        input: Handle,
    },
    /// Advance one declared supervised environment adapter.
    StepEnvironment {
        /// Validated adapter identifier.
        adapter: String,
        /// Opaque action segment handle.
        action: Handle,
    },
    /// Select one previously validated conditional edge.
    SelectBranch {
        /// Validated graph edge identifier.
        edge: String,
    },
    /// Complete the live turn progression with declared output handles.
    Complete {
        /// Opaque output segment handles.
        outputs: Vec<Handle>,
    },
}

/// One sealed policy-model request for a live agent turn.
///
/// Only the NativeGraph rollout coordinator constructs this request from its
/// imported prompt snapshot and a Rust-owned frozen environment observation.
pub struct LiveAgentPolicyDecisionRequest {
    prompt: Bytes,
    observation: Bytes,
    max_decision_bytes: usize,
}

impl fmt::Debug for LiveAgentPolicyDecisionRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LiveAgentPolicyDecisionRequest")
            .field("prompt_bytes", &self.prompt.len())
            .field("observation_bytes", &self.observation.len())
            .field("max_decision_bytes", &self.max_decision_bytes)
            .finish()
    }
}

impl LiveAgentPolicyDecisionRequest {
    /// Constructs a bounded model-decision request from Rust-owned rollout facts.
    pub(crate) fn new(
        prompt: Bytes,
        observation: Bytes,
        max_decision_bytes: usize,
    ) -> Result<Self, AgentLoopError> {
        if max_decision_bytes == 0 {
            return Err(AgentLoopError::new(
                "live policy decision requires a positive byte limit",
            ));
        }
        Ok(Self {
            prompt,
            observation,
            max_decision_bytes,
        })
    }

    /// Returns the imported immutable policy-prompt bytes.
    pub fn prompt(&self) -> &[u8] {
        &self.prompt
    }

    /// Returns the Rust-read frozen environment observation bytes.
    pub fn observation(&self) -> &[u8] {
        &self.observation
    }

    /// Returns the exact maximum raw decision bytes the model runtime may collect.
    pub const fn max_decision_bytes(&self) -> usize {
        self.max_decision_bytes
    }
}

/// Host-owned bounded source for raw bytes of one live policy decision.
///
/// The coordinator supplies the destination buffer, so a model runtime cannot return an owned
/// response frame through this boundary. Implementations must write at most the supplied capacity.
#[async_trait(?Send)]
pub trait LiveAgentPolicyDecisionReader {
    /// Reads at most `destination.len()` raw decision bytes, returning zero only at end of input.
    async fn read(&mut self, destination: &mut [u8]) -> Result<usize, AgentLoopError>;
}

/// Bounded Rust-owned collector for one streaming live policy decision.
///
/// The collector allocates only after a reader has written into a host-owned bounded buffer.
pub struct LiveAgentPolicyDecisionCollector {
    max_decision_bytes: usize,
    bytes: Vec<u8>,
}

impl fmt::Debug for LiveAgentPolicyDecisionCollector {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LiveAgentPolicyDecisionCollector")
            .field("collected_bytes", &self.bytes.len())
            .field("max_decision_bytes", &self.max_decision_bytes)
            .finish()
    }
}

impl LiveAgentPolicyDecisionCollector {
    /// Creates the collector for one positive package-selected decision cap.
    pub(crate) fn new(max_decision_bytes: usize) -> Result<Self, AgentLoopError> {
        if max_decision_bytes == 0 {
            return Err(AgentLoopError::new(
                "live policy decision requires a positive byte limit",
            ));
        }
        Ok(Self {
            max_decision_bytes,
            bytes: Vec::new(),
        })
    }

    fn append(&mut self, chunk: &[u8]) -> Result<(), AgentLoopError> {
        let required = self
            .bytes
            .len()
            .checked_add(chunk.len())
            .ok_or_else(|| AgentLoopError::new("live policy decision byte length overflow"))?;
        if required > self.max_decision_bytes {
            return Err(AgentLoopError::new(
                "live policy decision exceeds the selected byte limit",
            ));
        }
        self.bytes
            .try_reserve(chunk.len())
            .map_err(|_| AgentLoopError::new("live policy decision allocation failed"))?;
        self.bytes.extend_from_slice(chunk);
        Ok(())
    }

    /// Reads a complete decision through host-owned bounded buffers before retaining any bytes.
    pub async fn collect_from(
        &mut self,
        reader: &mut dyn LiveAgentPolicyDecisionReader,
    ) -> Result<(), AgentLoopError> {
        const READ_CHUNK_BYTES: usize = 4_096;
        let mut scratch = [0_u8; READ_CHUNK_BYTES];
        loop {
            let remaining = self
                .max_decision_bytes
                .checked_sub(self.bytes.len())
                .ok_or_else(|| AgentLoopError::new("live policy decision byte length overflow"))?;
            if remaining == 0 {
                let mut probe = [0_u8; 1];
                let observed = reader.read(&mut probe).await?;
                if observed > probe.len() {
                    return Err(AgentLoopError::new(
                        "live policy decision reader exceeded the host-owned buffer",
                    ));
                }
                return if observed == 0 {
                    Ok(())
                } else {
                    Err(AgentLoopError::new(
                        "live policy decision exceeds the selected byte limit",
                    ))
                };
            }
            let capacity = remaining.min(scratch.len());
            let observed = reader.read(&mut scratch[..capacity]).await?;
            if observed > capacity {
                return Err(AgentLoopError::new(
                    "live policy decision reader exceeded the host-owned buffer",
                ));
            }
            if observed == 0 {
                return Ok(());
            }
            self.append(&scratch[..observed])?;
        }
    }

    /// Returns the currently retained model-output length without exposing its bytes.
    pub const fn len(&self) -> usize {
        self.bytes.len()
    }

    pub(crate) fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}

impl AgentTurn {
    /// Construct one selected response and its copied-context marking.
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
    /// Select one live agent action after observing prior bounded results.
    ///
    /// Existing recorded coordinators remain single-plan implementations and
    /// return this typed refusal until a live coordinator is explicitly selected.
    async fn next_live_turn(&mut self) -> Result<LiveAgentTurnDirective, AgentLoopError> {
        Err(AgentLoopError::new(
            "agent turn coordinator does not support live progression",
        ))
    }

    /// Requests one bounded raw policy decision through an existing live coordinator.
    ///
    /// Recorded coordinators retain the default refusal. NativeGraph rollout
    /// composition installs its package-bound model-runtime coordinator instead.
    async fn next_live_policy_decision(
        &mut self,
        _: &LiveAgentPolicyDecisionRequest,
    ) -> Result<Box<dyn LiveAgentPolicyDecisionReader>, AgentLoopError> {
        Err(AgentLoopError::new(
            "agent turn coordinator does not support live policy decisions",
        ))
    }

    /// Execute the trace-local agent turns through injected worker-local seams.
    async fn run(
        &mut self,
        response_store: &mut dyn AgentResponseStore,
        trajectory: &mut dyn AgentTrajectorySink,
        leases: &dyn InvocationLeaseFactory,
        invocation_lease: &dyn AgentInvocationLease,
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

/// In-memory factory retaining deterministic turns for recorded replay and seam tests.
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
        invocation_lease: &dyn AgentInvocationLease,
        decoder: &dyn AgentToolCallDecoder,
        formatter: &dyn AgentObservationFormatter,
    ) -> Result<AgentTrajectory, AgentLoopError> {
        let _lease = leases.acquire();
        let dispatcher = invocation_lease.dispatcher();
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
                let result = dispatcher
                    .dispatch(call.dispatch_request(), &ToolDispatchContext::default())
                    .await?;
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

#[cfg(test)]
mod tests {
    use bytes::Bytes;

    use super::{LiveAgentPolicyDecisionCollector, LiveAgentPolicyDecisionRequest};

    #[test]
    fn policy_decision_collector_refuses_an_oversized_chunk_before_retaining_it() {
        let mut collector = LiveAgentPolicyDecisionCollector::new(3)
            .expect("a positive selected decision cap is valid");

        let error = collector
            .append(&[b'x'; 4])
            .expect_err("an oversized provider chunk must not be retained");

        assert!(
            error
                .to_string()
                .contains("exceeds the selected byte limit")
        );
        assert_eq!(collector.len(), 0);
    }

    #[test]
    fn policy_decision_request_debug_redacts_raw_prompt_and_observation_bytes() {
        let request = LiveAgentPolicyDecisionRequest::new(
            Bytes::from_static(b"prompt-secret"),
            Bytes::from_static(b"observation-secret"),
            32,
        )
        .expect("fixture request is valid");

        let debug = format!("{request:?}");
        assert!(!debug.contains("prompt-secret"));
        assert!(!debug.contains("observation-secret"));
        assert!(debug.contains("prompt_bytes"));
    }
}
