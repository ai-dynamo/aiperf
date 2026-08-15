// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Content-addressed response ownership for worker-local agent trajectories.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::{self, Display};

use bytes::Bytes;

/// Origin of a response selected by an agent driver.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AgentResponseSource {
    /// Bytes captured from the recorded source trajectory.
    Recorded,
    /// Bytes produced by the benchmark endpoint.
    Live,
    /// Bytes normalized from an explicitly loaded trajectory.
    Loaded,
    /// Provenance marker for a response copied from an earlier logical turn.
    Reused {
        /// Original logical turn that first selected the bytes.
        original_turn: usize,
    },
}

/// BLAKE3-addressed immutable response bytes.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct AgentResponseHandle(String);

impl AgentResponseHandle {
    /// Return the stable content address used by trajectory references.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Response-store failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AgentResponseStoreError(String);

impl Display for AgentResponseStoreError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for AgentResponseStoreError {}

/// Worker-local immutable response store.
pub trait AgentResponseStore {
    /// Intern selected bytes and return their stable reference.
    fn intern(
        &mut self,
        source: AgentResponseSource,
        wire: Bytes,
    ) -> Result<AgentResponseHandle, AgentResponseStoreError>;
    /// Read previously interned bytes.
    fn get(&self, handle: &AgentResponseHandle) -> Result<Bytes, AgentResponseStoreError>;
}

/// In-memory response store for deterministic trace drivers and unit tests.
#[derive(Default)]
pub struct InMemoryAgentResponseStore {
    entries: RefCell<BTreeMap<AgentResponseHandle, Bytes>>,
}

impl AgentResponseStore for InMemoryAgentResponseStore {
    fn intern(
        &mut self,
        source: AgentResponseSource,
        wire: Bytes,
    ) -> Result<AgentResponseHandle, AgentResponseStoreError> {
        let source_tag: &[u8] = match source {
            AgentResponseSource::Recorded => b"recorded",
            AgentResponseSource::Live => b"live",
            AgentResponseSource::Loaded => b"loaded",
            AgentResponseSource::Reused { .. } => b"reused",
        };
        let mut hasher = blake3::Hasher::new();
        hasher.update(source_tag);
        hasher.update(&wire);
        let handle = AgentResponseHandle(hasher.finalize().to_hex().to_string());
        self.entries
            .borrow_mut()
            .entry(handle.clone())
            .or_insert(wire);
        Ok(handle)
    }

    fn get(&self, handle: &AgentResponseHandle) -> Result<Bytes, AgentResponseStoreError> {
        self.entries.borrow().get(handle).cloned().ok_or_else(|| {
            AgentResponseStoreError(format!(
                "unknown agent response handle {:?}",
                handle.as_str()
            ))
        })
    }
}

/// Factory creating one response store per admitted trace.
pub trait AgentResponseStoreFactory: Send + Sync {
    /// Create a fresh worker-local store for `trace_id`.
    fn create(
        &self,
        trace_id: &str,
    ) -> Result<Box<dyn AgentResponseStore>, AgentResponseStoreError>;
}

/// Stock in-memory response-store factory used by deterministic driver fakes.
#[derive(Clone, Copy, Debug, Default)]
pub struct InMemoryAgentResponseStoreFactory;

impl AgentResponseStoreFactory for InMemoryAgentResponseStoreFactory {
    fn create(
        &self,
        _trace_id: &str,
    ) -> Result<Box<dyn AgentResponseStore>, AgentResponseStoreError> {
        Ok(Box::new(InMemoryAgentResponseStore::default()))
    }
}
