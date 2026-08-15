// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Invocation leases for deterministic trace-local agent ownership.

use std::cell::Cell;
use std::rc::Rc;

use async_trait::async_trait;

use crate::graph::tools::{InMemoryToolDispatcher, ToolDispatcher};
/// One trace-local invocation lease.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InvocationLease {
    /// Deterministic per-factory ordinal.
    pub ordinal: u64,
}

/// Factory minting trace-local invocation ownership.
pub trait InvocationLeaseFactory {
    /// Acquire one lease before running an invocation.
    fn acquire(&self) -> InvocationLease;
}

/// Frozen factory creating one worker-local lease factory per trace.
pub trait InvocationLeaseFactoryFactory: Send + Sync {
    /// Create a fresh lease factory for one trace-owned agent loop.
    fn create(
        &self,
        trace_id: &str,
    ) -> Result<Box<dyn InvocationLeaseFactory>, crate::graph::agent::AgentLoopError>;
}

/// Worker-local deterministic lease factory.
#[derive(Default)]
pub struct InMemoryInvocationLeaseFactory {
    next: Cell<u64>,
}

/// Stock factory for trace-local deterministic invocation leases.
#[derive(Clone, Copy, Debug, Default)]
pub struct InMemoryInvocationLeaseFactoryFactory;

impl InvocationLeaseFactoryFactory for InMemoryInvocationLeaseFactoryFactory {
    fn create(
        &self,
        _trace_id: &str,
    ) -> Result<Box<dyn InvocationLeaseFactory>, crate::graph::agent::AgentLoopError> {
        Ok(Box::new(InMemoryInvocationLeaseFactory::default()))
    }
}

impl InvocationLeaseFactory for InMemoryInvocationLeaseFactory {
    fn acquire(&self) -> InvocationLease {
        let ordinal = self.next.get();
        self.next.set(ordinal.saturating_add(1));
        InvocationLease { ordinal }
    }
}

/// Identity of one root or delegated agent invocation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AgentInvocationIdentity {
    /// Run-wide correlation identity.
    pub run_id: String,
    /// Distinct trajectory-document identity.
    pub trajectory_id: String,
    /// Distinct invocation identity.
    pub invocation_id: String,
    /// Optional parent invocation for a delegated child.
    pub parent_invocation_id: Option<String>,
}

/// Environment ownership selected for a delegated invocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AgentInvocationEnvironment {
    /// Borrow the parent dispatcher and later sandbox lease.
    Shared,
    /// Ask composition to provision an isolated child environment.
    Isolated,
}

/// Request passed to lifecycle-owned invocation leasing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AgentInvocationRequest {
    /// Child or root invocation identity.
    pub identity: AgentInvocationIdentity,
    /// Requested parent sharing discipline.
    pub environment: AgentInvocationEnvironment,
}

/// Terminal child fact retained until the parent joins in authored order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DelegatedInvocationTerminal {
    /// Child identity, distinct from its parent and trajectory document.
    pub identity: AgentInvocationIdentity,
    /// Authored deterministic join ordinal.
    pub join_ordinal: usize,
}

/// Sort delegated terminal facts by their authored join order.
pub fn deterministic_delegated_join_order(
    mut terminals: Vec<DelegatedInvocationTerminal>,
) -> Result<Vec<DelegatedInvocationTerminal>, crate::graph::agent::AgentLoopError> {
    terminals.sort_by_key(|terminal| terminal.join_ordinal);
    if terminals
        .windows(2)
        .any(|pair| pair[0].join_ordinal == pair[1].join_ordinal)
    {
        return Err(crate::graph::agent::AgentLoopError::new(
            "delegated invocations have duplicate join ordinals",
        ));
    }
    Ok(terminals)
}

/// Worker-local lease for one agent invocation.
#[async_trait(?Send)]
pub trait AgentInvocationLease {
    /// Borrow the sole dispatcher authorized for this invocation.
    fn dispatcher(&self) -> Rc<dyn ToolDispatcher>;
    /// Close child resources before the parent or trace ends.
    async fn close(&mut self) -> Result<(), crate::graph::agent::AgentLoopError>;
}

/// Lifecycle-owned factory for root and delegated invocation leases.
#[async_trait(?Send)]
pub trait AgentInvocationLeaseFactory {
    /// Open one fresh root or child lease without exposing provisioning to drivers.
    async fn open(
        &self,
        request: &AgentInvocationRequest,
        parent: Option<&dyn AgentInvocationLease>,
    ) -> Result<Box<dyn AgentInvocationLease>, crate::graph::agent::AgentLoopError>;
}

/// Deterministic lease that simply retains its injected fake dispatcher.
pub struct InMemoryAgentInvocationLease {
    dispatcher: Rc<dyn ToolDispatcher>,
    is_closed: Cell<bool>,
}

#[async_trait(?Send)]
impl AgentInvocationLease for InMemoryAgentInvocationLease {
    fn dispatcher(&self) -> Rc<dyn ToolDispatcher> {
        self.dispatcher.clone()
    }

    async fn close(&mut self) -> Result<(), crate::graph::agent::AgentLoopError> {
        self.is_closed.set(true);
        Ok(())
    }
}

/// Deterministic lifecycle fake used by the future live-driver contract tests.
pub struct InMemoryAgentInvocationLeaseFactory {
    next_dispatcher: Cell<u64>,
}

impl InMemoryAgentInvocationLeaseFactory {
    /// Construct a fake that gives every isolated lease its own dispatcher.
    pub fn new() -> Self {
        Self {
            next_dispatcher: Cell::new(0),
        }
    }
}

impl Default for InMemoryAgentInvocationLeaseFactory {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait(?Send)]
impl AgentInvocationLeaseFactory for InMemoryAgentInvocationLeaseFactory {
    async fn open(
        &self,
        request: &AgentInvocationRequest,
        parent: Option<&dyn AgentInvocationLease>,
    ) -> Result<Box<dyn AgentInvocationLease>, crate::graph::agent::AgentLoopError> {
        if request.environment == AgentInvocationEnvironment::Shared && parent.is_none() {
            return Err(crate::graph::agent::AgentLoopError::new(
                "shared delegated invocation requires a parent lease",
            ));
        }
        let dispatcher = if request.environment == AgentInvocationEnvironment::Shared {
            parent
                .ok_or_else(|| {
                    crate::graph::agent::AgentLoopError::new(
                        "shared delegated invocation requires a parent lease",
                    )
                })?
                .dispatcher()
        } else {
            let next = self.next_dispatcher.get();
            self.next_dispatcher.set(next.saturating_add(1));
            Rc::new(InMemoryToolDispatcher::default())
        };
        Ok(Box::new(InMemoryAgentInvocationLease {
            dispatcher,
            is_closed: Cell::new(false),
        }))
    }
}
