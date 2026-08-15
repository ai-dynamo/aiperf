// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Worker-local tool-dispatch boundary and deterministic in-memory fake.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::error::Error;
use std::fmt::{self, Display};

use async_trait::async_trait;
use bytes::Bytes;

/// A recorded tool call ready for execution.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolDispatchRequest {
    /// Provider-assigned tool-call correlation identifier.
    pub call_id: String,
    /// Authored command, retained only by the dispatcher boundary.
    pub command: String,
}

impl ToolDispatchRequest {
    /// Construct a correlated command request.
    pub fn new(call_id: impl Into<String>, command: impl Into<String>) -> Self {
        Self {
            call_id: call_id.into(),
            command: command.into(),
        }
    }
}

/// Terminal result for one attempted tool command.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolDispatchResult {
    /// Provider-assigned tool-call correlation identifier.
    pub call_id: String,
    /// Process-style result code when a command reached execution.
    pub exit_code: i32,
    /// Combined bounded command output.
    pub output: Bytes,
    /// Whether execution reached its deadline.
    pub is_timed_out: bool,
}

impl ToolDispatchResult {
    /// Construct a successful deterministic result.
    pub fn completed(call_id: impl Into<String>, exit_code: i32, output: Bytes) -> Self {
        Self {
            call_id: call_id.into(),
            exit_code,
            output,
            is_timed_out: false,
        }
    }
}

/// Tool-dispatch failure that prevented a terminal command result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolDispatchError(String);

impl ToolDispatchError {
    /// Construct an explicit dispatch-boundary failure.
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl Display for ToolDispatchError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for ToolDispatchError {}

/// Worker-local dispatcher for correlated recorded tool commands.
#[async_trait(?Send)]
pub trait ToolDispatcher {
    /// Execute one command and return its terminal result.
    async fn dispatch(
        &self,
        request: ToolDispatchRequest,
    ) -> Result<ToolDispatchResult, ToolDispatchError>;
}

/// Provider-neutral decoded tool call retained before command dispatch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AgentToolCall {
    /// Provider-assigned correlation identifier.
    pub call_id: String,
    /// Dispatcher-owned command payload.
    pub command: String,
}

impl AgentToolCall {
    /// Convert this decoded call into the minimal execution request.
    pub fn dispatch_request(&self) -> ToolDispatchRequest {
        ToolDispatchRequest::new(self.call_id.clone(), self.command.clone())
    }
}

/// Decodes provider response bytes into provider-neutral tool calls.
pub trait AgentToolCallDecoder {
    /// Decode every tool call selected by this assistant response.
    fn decode(&self, response_wire: &Bytes) -> Result<Vec<AgentToolCall>, ToolDispatchError>;
}

/// Formats correlated command results for the subsequent request materialization.
pub trait AgentObservationFormatter {
    /// Format observations in call order after correlation validation.
    fn format(
        &self,
        calls: &[AgentToolCall],
        results: &[ToolDispatchResult],
    ) -> Result<Vec<Bytes>, ToolDispatchError>;
}

/// Deterministic decoder fake that yields authored call batches in turn order.
#[derive(Default)]
pub struct InMemoryAgentToolCallDecoder {
    calls: RefCell<VecDeque<Vec<AgentToolCall>>>,
}

impl InMemoryAgentToolCallDecoder {
    /// Construct a fake that yields one batch for each decoded response.
    pub fn from_call_batches(batches: impl IntoIterator<Item = Vec<AgentToolCall>>) -> Self {
        Self {
            calls: RefCell::new(batches.into_iter().collect()),
        }
    }
}

impl AgentToolCallDecoder for InMemoryAgentToolCallDecoder {
    fn decode(&self, _response_wire: &Bytes) -> Result<Vec<AgentToolCall>, ToolDispatchError> {
        self.calls.borrow_mut().pop_front().ok_or_else(|| {
            ToolDispatchError::new("no deterministic decoded tool-call batch remains")
        })
    }
}

/// Deterministic formatter that preserves provider call correlation in its wire.
#[derive(Clone, Copy, Debug, Default)]
pub struct InMemoryAgentObservationFormatter;

impl AgentObservationFormatter for InMemoryAgentObservationFormatter {
    fn format(
        &self,
        calls: &[AgentToolCall],
        results: &[ToolDispatchResult],
    ) -> Result<Vec<Bytes>, ToolDispatchError> {
        if calls.len() != results.len() {
            return Err(ToolDispatchError::new(
                "tool observation count does not match decoded calls",
            ));
        }
        calls
            .iter()
            .zip(results)
            .map(|(call, result)| {
                if call.call_id != result.call_id {
                    return Err(ToolDispatchError::new(format!(
                        "tool result call {:?} does not match decoded call {:?}",
                        result.call_id, call.call_id
                    )));
                }
                let mut observation =
                    Vec::with_capacity(call.call_id.len() + result.output.len() + 1);
                observation.extend_from_slice(call.call_id.as_bytes());
                observation.push(b':');
                observation.extend_from_slice(&result.output);
                Ok(Bytes::from(observation))
            })
            .collect()
    }
}

/// Deterministic worker-local dispatcher used by driver tests.
#[derive(Default)]
pub struct InMemoryToolDispatcher {
    results: RefCell<VecDeque<ToolDispatchResult>>,
}

impl InMemoryToolDispatcher {
    /// Construct a fake that yields results in authored command order.
    pub fn from_results(results: impl IntoIterator<Item = ToolDispatchResult>) -> Self {
        Self {
            results: RefCell::new(results.into_iter().collect()),
        }
    }
}

#[async_trait(?Send)]
impl ToolDispatcher for InMemoryToolDispatcher {
    async fn dispatch(
        &self,
        request: ToolDispatchRequest,
    ) -> Result<ToolDispatchResult, ToolDispatchError> {
        let result = self.results.borrow_mut().pop_front().ok_or_else(|| {
            ToolDispatchError::new(format!(
                "no deterministic tool result remains for call {:?}",
                request.call_id
            ))
        })?;
        if result.call_id != request.call_id {
            return Err(ToolDispatchError::new(format!(
                "deterministic tool result call {:?} does not match request {:?}",
                result.call_id, request.call_id
            )));
        }
        Ok(result)
    }
}
