// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Invocation leases for deterministic trace-local agent ownership.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use async_trait::async_trait;

use crate::eval::ArtifactDigest;
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

/// Immutable workspace authority selected for one root or branch invocation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AgentInvocationWorkspace {
    /// The canonical task workspace, owned by the root or borrowed by a
    /// shared delegated invocation.
    Root,
    /// One branch candidate receives an isolated overlay owned by the factory.
    ///
    /// The request deliberately contains no candidate content digest or host
    /// path. The child lease mints that digest only after its workspace reaches
    /// its completion boundary.
    IsolatedBranch {
        /// Declared branch identifier from the immutable control contract.
        branch_id: String,
        /// Deterministic candidate identifier scoped to `branch_id`.
        candidate_id: String,
        /// Parent invocation which owns the source workspace snapshot.
        parent_invocation_id: String,
        /// Immutable source snapshot identity selected before child provisioning.
        parent_snapshot_digest: String,
    },
}

/// Immutable branch-workspace candidate returned only by a completed child lease.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AgentInvocationWorkspaceCandidate {
    /// Declared candidate identifier selected by the parent merge contract.
    id: String,
    /// Factory-minted immutable content identity for the completed overlay.
    digest: ArtifactDigest,
}

impl AgentInvocationWorkspaceCandidate {
    /// Creates a candidate from a factory-minted validated artifact identity.
    pub fn new(id: String, digest: ArtifactDigest) -> Self {
        Self { id, digest }
    }

    /// Validates a factory-provided textual artifact identity before it may enter
    /// a workspace candidate or a control receipt.
    pub fn parse(
        id: String,
        digest: impl Into<String>,
    ) -> Result<Self, crate::graph::agent::AgentLoopError> {
        let digest = ArtifactDigest::parse(digest).map_err(|_| {
            crate::graph::agent::AgentLoopError::new(
                "workspace candidate digest is not a valid artifact identity",
            )
        })?;
        Ok(Self::new(id, digest))
    }

    /// Borrows the contract-selected candidate identifier.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Borrows the validated immutable workspace content identity.
    pub fn digest(&self) -> &ArtifactDigest {
        &self.digest
    }
}

/// Request passed to lifecycle-owned invocation leasing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AgentInvocationRequest {
    /// Child or root invocation identity.
    pub identity: AgentInvocationIdentity,
    /// Requested parent sharing discipline.
    pub environment: AgentInvocationEnvironment,
    /// Workspace authority requested from the lifecycle-owning factory.
    pub workspace: AgentInvocationWorkspace,
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
    /// Synchronously fence resources when an owning task is cancelled or dropped.
    fn close_on_drop(&mut self);
    /// Freeze this child workspace into a merge candidate after successful work.
    ///
    /// Root and shared invocations return `None`; an isolated branch may return
    /// one immutable candidate only while its lease remains open.
    async fn complete_workspace(
        &mut self,
    ) -> Result<Option<AgentInvocationWorkspaceCandidate>, crate::graph::agent::AgentLoopError>
    {
        Ok(None)
    }
    /// Close child resources before the parent or trace ends.
    async fn close(&mut self) -> Result<(), crate::graph::agent::AgentLoopError>;
}

/// Cancellation-safe opening operation that owns provisioned resources until lease transfer.
#[async_trait(?Send)]
pub trait AgentInvocationLeaseOpening {
    /// Finish provisioning and transfer the opened lease to the caller.
    async fn open(
        &mut self,
    ) -> Result<Box<dyn AgentInvocationLease>, crate::graph::agent::AgentLoopError>;
    /// Synchronously fence provisioned resources if opening is cancelled or dropped.
    fn cancel_on_drop(&mut self);
}

/// Lifecycle-owned factory for root and delegated invocation leases.
pub trait AgentInvocationLeaseFactory {
    /// Begin one fresh root or child lease with cancellation-safe resource ownership.
    fn begin_open(
        &self,
        request: &AgentInvocationRequest,
        parent: Option<&dyn AgentInvocationLease>,
    ) -> Result<Box<dyn AgentInvocationLeaseOpening>, crate::graph::agent::AgentLoopError>;
}

struct InMemoryAgentInvocationLeaseOpening {
    lease: Option<Box<dyn AgentInvocationLease>>,
}

#[async_trait(?Send)]
impl AgentInvocationLeaseOpening for InMemoryAgentInvocationLeaseOpening {
    async fn open(
        &mut self,
    ) -> Result<Box<dyn AgentInvocationLease>, crate::graph::agent::AgentLoopError> {
        self.lease.take().ok_or_else(|| {
            crate::graph::agent::AgentLoopError::new(
                "invocation lease opening was already consumed",
            )
        })
    }

    fn cancel_on_drop(&mut self) {
        if let Some(mut lease) = self.lease.take() {
            lease.close_on_drop();
        }
    }
}

/// Frozen factory creating one lifecycle owner for every trace-owned agent loop.
pub trait AgentInvocationLeaseFactoryFactory: Send + Sync {
    /// Create one lifecycle owner and transfer its root dispatcher ownership.
    fn create(
        &self,
        trace_id: &str,
        root_dispatcher: Rc<dyn ToolDispatcher>,
    ) -> Result<Box<dyn AgentInvocationLeaseFactory>, crate::graph::agent::AgentLoopError>;
}

/// Stock factory for deterministic trace-local lifecycle owners.
#[derive(Clone, Copy, Debug, Default)]
pub struct InMemoryAgentInvocationLeaseFactoryFactory;

impl AgentInvocationLeaseFactoryFactory for InMemoryAgentInvocationLeaseFactoryFactory {
    fn create(
        &self,
        _trace_id: &str,
        root_dispatcher: Rc<dyn ToolDispatcher>,
    ) -> Result<Box<dyn AgentInvocationLeaseFactory>, crate::graph::agent::AgentLoopError> {
        Ok(Box::new(
            InMemoryAgentInvocationLeaseFactory::with_root_dispatcher(root_dispatcher),
        ))
    }
}

/// Deterministic lease that simply retains its injected fake dispatcher.
pub struct InMemoryAgentInvocationLease {
    dispatcher: Rc<dyn ToolDispatcher>,
    workspace: AgentInvocationWorkspace,
    is_closed: Cell<bool>,
}

#[async_trait(?Send)]
impl AgentInvocationLease for InMemoryAgentInvocationLease {
    fn dispatcher(&self) -> Rc<dyn ToolDispatcher> {
        self.dispatcher.clone()
    }

    fn close_on_drop(&mut self) {
        self.is_closed.set(true);
    }

    async fn complete_workspace(
        &mut self,
    ) -> Result<Option<AgentInvocationWorkspaceCandidate>, crate::graph::agent::AgentLoopError>
    {
        if self.is_closed.get() {
            return Err(crate::graph::agent::AgentLoopError::new(
                "cannot complete a closed invocation workspace",
            ));
        }
        let AgentInvocationWorkspace::IsolatedBranch {
            branch_id,
            candidate_id,
            parent_invocation_id,
            parent_snapshot_digest,
        } = &self.workspace
        else {
            return Ok(None);
        };
        let material = format!(
            "native-graph-workspace-candidate-v1\x1f{branch_id}\x1f{candidate_id}\x1f{parent_invocation_id}\x1f{parent_snapshot_digest}"
        );
        Ok(Some(AgentInvocationWorkspaceCandidate::new(
            candidate_id.clone(),
            ArtifactDigest::from_bytes(material.as_bytes()),
        )))
    }

    async fn close(&mut self) -> Result<(), crate::graph::agent::AgentLoopError> {
        self.close_on_drop();
        Ok(())
    }
}

/// Deterministic lifecycle fake used by the future live-driver contract tests.
pub struct InMemoryAgentInvocationLeaseFactory {
    next_dispatcher: Cell<u64>,
    root_dispatcher: RefCell<Option<Rc<dyn ToolDispatcher>>>,
}

impl InMemoryAgentInvocationLeaseFactory {
    /// Construct a fake that gives every isolated lease its own dispatcher.
    pub fn new() -> Self {
        Self {
            next_dispatcher: Cell::new(0),
            root_dispatcher: RefCell::new(None),
        }
    }

    /// Construct a per-trace lifecycle owner for an already-created root dispatcher.
    pub fn with_root_dispatcher(root_dispatcher: Rc<dyn ToolDispatcher>) -> Self {
        Self {
            next_dispatcher: Cell::new(0),
            root_dispatcher: RefCell::new(Some(root_dispatcher)),
        }
    }
}

impl Default for InMemoryAgentInvocationLeaseFactory {
    fn default() -> Self {
        Self::new()
    }
}

impl AgentInvocationLeaseFactory for InMemoryAgentInvocationLeaseFactory {
    fn begin_open(
        &self,
        request: &AgentInvocationRequest,
        parent: Option<&dyn AgentInvocationLease>,
    ) -> Result<Box<dyn AgentInvocationLeaseOpening>, crate::graph::agent::AgentLoopError> {
        validate_workspace_request(request)?;
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
            self.root_dispatcher.borrow_mut().take().unwrap_or_else(|| {
                let next = self.next_dispatcher.get();
                self.next_dispatcher.set(next.saturating_add(1));
                Rc::new(InMemoryToolDispatcher::default())
            })
        };
        Ok(Box::new(InMemoryAgentInvocationLeaseOpening {
            lease: Some(Box::new(InMemoryAgentInvocationLease {
                dispatcher,
                workspace: request.workspace.clone(),
                is_closed: Cell::new(false),
            })),
        }))
    }
}

fn validate_workspace_request(
    request: &AgentInvocationRequest,
) -> Result<(), crate::graph::agent::AgentLoopError> {
    match &request.workspace {
        AgentInvocationWorkspace::Root
            if request.identity.parent_invocation_id.is_none()
                && request.environment == AgentInvocationEnvironment::Isolated =>
        {
            Ok(())
        }
        AgentInvocationWorkspace::Root
            if request.identity.parent_invocation_id.is_some()
                && request.environment == AgentInvocationEnvironment::Shared =>
        {
            Ok(())
        }
        AgentInvocationWorkspace::Root => Err(crate::graph::agent::AgentLoopError::new(
            "root workspace must be owned by an isolated root or borrowed by a shared child",
        )),
        AgentInvocationWorkspace::IsolatedBranch {
            branch_id,
            candidate_id,
            parent_invocation_id,
            parent_snapshot_digest,
        } => {
            if request.environment != AgentInvocationEnvironment::Isolated
                || branch_id.is_empty()
                || candidate_id.is_empty()
                || parent_invocation_id.is_empty()
                || parent_snapshot_digest.is_empty()
                || request.identity.parent_invocation_id.as_deref() != Some(parent_invocation_id)
            {
                return Err(crate::graph::agent::AgentLoopError::new(
                    "isolated branch workspace request does not match its parent invocation authority",
                ));
            }
            Ok(())
        }
    }
}
