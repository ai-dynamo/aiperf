// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded staged trace driver for source-lowered NativeGraph programs.

use std::collections::{BTreeMap, BTreeSet};
use std::num::NonZeroU32;
use std::rc::Rc;
use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::Value;

use crate::dataset::Handle;
use crate::eval::ArtifactDigest;
use crate::graph::driver::{
    LifecycleLeaseGuard, LifecycleOpeningGuard, TraceDriverCapabilities, TraceDriverContext,
    TraceDriverError, TraceDriverProvenance, TraceDriverSpec, TraceEnvironmentSpec, TraceIdentity,
    TraceProgramDriver, TraceProgramDriverFactory, TraceStageDirective, TraceStageResult,
    WorkerIdentity,
};
use crate::graph::model::{
    GraphRecord, GraphTracePlan, GraphTraceProgram, START_NODE_ID, StaticEdge,
};
use crate::graph::sink::GraphReplyStatus;
use crate::graph::supplement::{
    DeclaredDynamicControlName, DynamicControlCounters, DynamicControlOperation,
    DynamicControlReceipt, TraceTerminalSupplement,
};
use crate::graph::{agent, tools};

use super::lowering::{
    NativeGraphControlContract, canonical_control_digest, canonical_static_projection_digest,
    initial_dynamic_stage, validate_control_flow_contract, validate_dynamic_native_graph_source,
    validate_native_graph_stage, validate_native_graph_trace_plan,
};

pub(crate) const NATIVE_GRAPH_LIVE_DRIVER_KIND: &str = "native_graph_live";

/// Factory for the Rust-owned staged NativeGraph driver family.
#[derive(Clone)]
pub struct NativeGraphLiveTraceProgramDriverFactory {
    agent_loop: Arc<NativeGraphLiveAgentLoopFactories>,
}

impl Default for NativeGraphLiveTraceProgramDriverFactory {
    fn default() -> Self {
        Self {
            agent_loop: Arc::new(NativeGraphLiveAgentLoopFactories::default()),
        }
    }
}

/// Frozen lifecycle and tool seams for one live NativeGraph trace.
pub struct NativeGraphLiveAgentLoopFactories {
    lifecycle_lease: Arc<dyn agent::AgentInvocationLeaseFactoryFactory>,
    tool_dispatcher: Arc<dyn tools::ToolDispatcherFactory>,
}

impl Default for NativeGraphLiveAgentLoopFactories {
    fn default() -> Self {
        Self {
            lifecycle_lease: Arc::new(agent::InMemoryAgentInvocationLeaseFactoryFactory),
            tool_dispatcher: Arc::new(tools::NativeToolDispatcherFactory::default()),
        }
    }
}

impl NativeGraphLiveAgentLoopFactories {
    /// Freeze the trace-local lifecycle and tool-dispatch composition.
    pub fn new(
        lifecycle_lease: Arc<dyn agent::AgentInvocationLeaseFactoryFactory>,
        tool_dispatcher: Arc<dyn tools::ToolDispatcherFactory>,
    ) -> Self {
        Self {
            lifecycle_lease,
            tool_dispatcher,
        }
    }
}

impl NativeGraphLiveTraceProgramDriverFactory {
    /// Replace the live graph's frozen lifecycle and tool-dispatch seams.
    pub fn with_agent_loop_factories(
        mut self,
        agent_loop: NativeGraphLiveAgentLoopFactories,
    ) -> Self {
        self.agent_loop = Arc::new(agent_loop);
        self
    }
}

impl TraceProgramDriverFactory for NativeGraphLiveTraceProgramDriverFactory {
    fn capabilities(
        &self,
        spec: &TraceDriverSpec,
    ) -> Result<TraceDriverCapabilities, TraceDriverError> {
        let _ = validate_live_driver_spec(spec)?;
        let capabilities = TraceDriverCapabilities {
            has_live_turns: true,
            ..TraceDriverCapabilities::default()
        };
        capabilities.validate(spec, &spec.kind)?;
        Ok(capabilities)
    }

    fn create(
        &self,
        worker: WorkerIdentity,
        trace: &TraceIdentity,
        spec: &TraceDriverSpec,
    ) -> Result<Box<dyn TraceProgramDriver>, TraceDriverError> {
        let (data, provenance) = validate_live_driver_spec(spec)?;
        let control_digest = ArtifactDigest::parse(provenance.control_digest().to_owned())
            .map_err(|error| TraceDriverError::new(error.to_string()))?;
        Ok(Box::new(NativeGraphLiveTraceProgramDriver {
            worker,
            trace: trace.clone(),
            stage_bound: data.control_flow.stage_bound,
            terminal_outputs: data.control_flow.terminal_outputs.clone(),
            control_flow: data.control_flow,
            control_digest,
            provenance,
            agent_loop: self.agent_loop.clone(),
            opening_session: None,
            session: None,
            state: LiveDriverState::Unopened,
        }))
    }
}

/// Worker-local cursor over one source-lowered NativeGraph program.
struct NativeGraphLiveTraceProgramDriver {
    worker: WorkerIdentity,
    trace: TraceIdentity,
    stage_bound: NonZeroU32,
    terminal_outputs: Vec<String>,
    control_flow: NativeGraphControlContract,
    control_digest: ArtifactDigest,
    provenance: TraceDriverProvenance,
    agent_loop: Arc<NativeGraphLiveAgentLoopFactories>,
    opening_session: Option<NativeGraphLiveTraceOpeningSession>,
    session: Option<NativeGraphLiveTraceSession>,
    state: LiveDriverState,
}

/// External resources owned for one root live graph invocation.
struct NativeGraphLiveTraceSession {
    dispatcher: Rc<dyn tools::ToolDispatcher>,
    lifecycle_factory: Box<dyn agent::AgentInvocationLeaseFactory>,
    root_lease: LifecycleLeaseGuard,
    root_invocation_id: String,
    branch_leases: BTreeMap<BranchWorkspaceKey, LifecycleLeaseGuard>,
}

/// Resources installed before dispatcher open so cancellation can roll them back.
struct NativeGraphLiveTraceOpeningSession {
    dispatcher: Rc<dyn tools::ToolDispatcher>,
    lifecycle_factory: Box<dyn agent::AgentInvocationLeaseFactory>,
    root_lease: LifecycleLeaseGuard,
    root_invocation_id: String,
}

impl NativeGraphLiveTraceOpeningSession {
    fn into_session(self) -> NativeGraphLiveTraceSession {
        NativeGraphLiveTraceSession {
            dispatcher: self.dispatcher,
            lifecycle_factory: self.lifecycle_factory,
            root_lease: self.root_lease,
            root_invocation_id: self.root_invocation_id,
            branch_leases: BTreeMap::new(),
        }
    }
}

/// One declared branch candidate's isolated child workspace ownership.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct BranchWorkspaceKey {
    branch_id: String,
    candidate_id: String,
}

impl NativeGraphLiveTraceSession {
    fn selected_branch_dispatcher(
        &self,
        cursor: &DynamicCursor,
        nodes: &BTreeSet<String>,
        control_flow: &NativeGraphControlContract,
    ) -> Option<Rc<dyn tools::ToolDispatcher>> {
        let key = selected_branch_key(cursor, nodes, control_flow)?;
        self.branch_leases
            .get(&key)
            .map(|lease| lease.lease().dispatcher())
    }
}

fn selected_branch_key(
    cursor: &DynamicCursor,
    nodes: &BTreeSet<String>,
    control_flow: &NativeGraphControlContract,
) -> Option<BranchWorkspaceKey> {
    cursor
        .branch_choices
        .iter()
        .find_map(|(branch_id, candidate_id)| {
            let branch = control_flow
                .branches
                .iter()
                .find(|branch| &branch.id == branch_id)?;
            let candidate = branch
                .candidates
                .iter()
                .find(|candidate| &candidate.id == candidate_id)?;
            candidate
                .nodes
                .iter()
                .any(|node| nodes.contains(node))
                .then(|| BranchWorkspaceKey {
                    branch_id: branch_id.clone(),
                    candidate_id: candidate_id.clone(),
                })
        })
}

enum LiveDriverState {
    Unopened,
    Ready(GraphTracePlan),
    AwaitingObservation {
        plan_identity: String,
    },
    DynamicReady(DynamicCursor),
    AwaitingDynamicObservation {
        plan_identity: String,
        cursor: DynamicCursor,
        executed_nodes: BTreeSet<String>,
    },
    ReadyToComplete {
        outputs: BTreeMap<String, Handle>,
        receipts: Vec<DynamicControlReceipt>,
    },
    Finished,
}

/// Rust-owned progress for one source-declared dynamic graph path.
struct DynamicCursor {
    source: GraphTracePlan,
    completed: BTreeSet<String>,
    disabled: BTreeSet<String>,
    channels: BTreeMap<String, Value>,
    branch_choices: BTreeMap<String, String>,
    workspace_candidates: BTreeMap<BranchWorkspaceKey, agent::AgentInvocationWorkspaceCandidate>,
    requires_workspace_candidates: bool,
    loops: BTreeMap<String, LoopProgress>,
    control_digest: ArtifactDigest,
    receipts: Vec<DynamicControlReceipt>,
    next_stage: u32,
}

/// Rust-owned bounded progress for one declared feedback edge.
#[derive(Default)]
struct LoopProgress {
    iterations: u32,
    retries: u32,
    awaits_backedge: bool,
}

#[async_trait(?Send)]
impl TraceProgramDriver for NativeGraphLiveTraceProgramDriver {
    async fn open(
        &mut self,
        program: &GraphTraceProgram,
        context: &TraceDriverContext<'_>,
    ) -> Result<(), TraceDriverError> {
        if !matches!(self.state, LiveDriverState::Unopened) {
            return Err(TraceDriverError::new(
                "native graph live driver is already open",
            ));
        }
        if program.driver.kind != NATIVE_GRAPH_LIVE_DRIVER_KIND || self.trace != *context.trace {
            return Err(TraceDriverError::new(
                "native graph live driver received another program or trace",
            ));
        }
        // Preserve the selected driver's finite-bound error before immutable
        // provenance validation so a substituted budget has one precise cause.
        if parse_live_driver_data(&program.driver)?
            .control_flow
            .stage_bound
            != self.stage_bound
        {
            return Err(TraceDriverError::new(
                "native graph live driver stage bound does not match its selected program",
            ));
        }
        let (program_data, program_provenance) = validate_live_driver_spec(&program.driver)?;
        if program_data.control_flow != self.control_flow {
            return Err(TraceDriverError::new(
                "native graph live driver control-flow contract does not match its selected program",
            ));
        }
        if program_provenance != self.provenance {
            return Err(TraceDriverError::new(
                "native graph live driver source provenance does not match its selected program",
            ));
        }
        if !self.terminal_outputs.is_empty() {
            return Err(TraceDriverError::new(
                "native graph live driver requires frozen terminal handles before stage execution",
            ));
        }
        let requires_workspace_candidates =
            has_dynamic_control(&self.control_flow) && context.execution.is_some();
        if has_dynamic_control(&self.control_flow) {
            validate_dynamic_native_graph_source(&program.profiling, &self.control_flow)
                .map_err(|error| TraceDriverError::new(error.to_string()))?;
            let source_digest = canonical_static_projection_digest(&program.profiling)
                .map_err(|error| TraceDriverError::new(error.to_string()))?;
            if source_digest != self.control_flow.source_graph_digest {
                return Err(TraceDriverError::new(
                    "native graph live driver source graph does not match its immutable control contract",
                ));
            }
            let initial = initial_dynamic_stage(&program.profiling)
                .map_err(|error| TraceDriverError::new(error.to_string()))?;
            let initial_digest = canonical_static_projection_digest(&initial)
                .map_err(|error| TraceDriverError::new(error.to_string()))?;
            if initial_digest != self.control_flow.static_projection_digest {
                return Err(TraceDriverError::new(
                    "native graph live driver initial stage does not match its immutable projection",
                ));
            }
            if requires_workspace_candidates {
                self.open_session(program, context).await?;
            }
            self.state = LiveDriverState::DynamicReady(DynamicCursor {
                channels: program.profiling.trace.initial_state.clone(),
                source: program.profiling.clone(),
                completed: BTreeSet::new(),
                disabled: BTreeSet::new(),
                branch_choices: BTreeMap::new(),
                workspace_candidates: BTreeMap::new(),
                requires_workspace_candidates,
                loops: BTreeMap::new(),
                control_digest: self.control_digest.clone(),
                receipts: Vec::new(),
                next_stage: 0,
            });
        } else {
            validate_native_graph_stage(&program.profiling, &self.control_flow)
                .map_err(|error| TraceDriverError::new(error.to_string()))?;
            self.state = LiveDriverState::Ready(program.profiling.clone());
        }
        Ok(())
    }

    fn stage_bound(&self) -> Option<NonZeroU32> {
        Some(self.stage_bound)
    }

    fn tool_dispatcher(&self) -> Option<Rc<dyn tools::ToolDispatcher>> {
        let session = self.session.as_ref()?;
        match &self.state {
            LiveDriverState::AwaitingDynamicObservation {
                cursor,
                executed_nodes,
                ..
            } => Some(
                session
                    .selected_branch_dispatcher(cursor, executed_nodes, &self.control_flow)
                    .unwrap_or_else(|| session.dispatcher.clone()),
            ),
            _ => Some(session.dispatcher.clone()),
        }
    }

    async fn next_stage(
        &mut self,
        _context: &TraceDriverContext<'_>,
    ) -> Result<Option<TraceStageDirective>, TraceDriverError> {
        match std::mem::replace(&mut self.state, LiveDriverState::Finished) {
            LiveDriverState::Ready(plan) => {
                validate_native_graph_stage(&plan, &self.control_flow)
                    .map_err(|error| TraceDriverError::new(error.to_string()))?;
                let plan_identity = format!("{}::stage-0", self.trace.trace_id);
                self.state = LiveDriverState::AwaitingObservation { plan_identity };
                Ok(Some(TraceStageDirective::Execute(plan)))
            }
            LiveDriverState::DynamicReady(cursor) => {
                let ready = dynamic_ready_nodes(&cursor, &self.control_flow)?;
                if ready.is_empty() {
                    self.state = LiveDriverState::Finished;
                    return Ok(Some(TraceStageDirective::Complete(
                        self.terminal_supplement(BTreeMap::new(), cursor.receipts),
                    )));
                }
                let plan = dynamic_stage_projection(&cursor, &ready)?;
                validate_dynamic_stage_projection(&plan, &cursor.source, &self.control_flow)?;
                self.ensure_stage_dispatcher(&cursor, &ready)?;
                let plan_identity = format!("{}::stage-{}", self.trace.trace_id, cursor.next_stage);
                self.state = LiveDriverState::AwaitingDynamicObservation {
                    plan_identity,
                    cursor,
                    executed_nodes: ready,
                };
                Ok(Some(TraceStageDirective::Execute(plan)))
            }
            LiveDriverState::ReadyToComplete { outputs, receipts } => {
                self.state = LiveDriverState::Finished;
                Ok(Some(TraceStageDirective::Complete(
                    self.terminal_supplement(outputs, receipts),
                )))
            }
            LiveDriverState::AwaitingObservation { plan_identity } => {
                self.state = LiveDriverState::AwaitingObservation { plan_identity };
                Err(TraceDriverError::new(
                    "native graph live driver requested a stage before observing the prior stage",
                ))
            }
            LiveDriverState::AwaitingDynamicObservation {
                plan_identity,
                cursor,
                executed_nodes,
            } => {
                self.state = LiveDriverState::AwaitingDynamicObservation {
                    plan_identity,
                    cursor,
                    executed_nodes,
                };
                Err(TraceDriverError::new(
                    "native graph live driver requested a stage before observing the prior stage",
                ))
            }
            LiveDriverState::Unopened => {
                self.state = LiveDriverState::Unopened;
                Err(TraceDriverError::new(
                    "native graph live driver was not opened before stage selection",
                ))
            }
            LiveDriverState::Finished => Ok(None),
        }
    }

    async fn observe_stage(&mut self, result: TraceStageResult) -> Result<(), TraceDriverError> {
        match std::mem::replace(&mut self.state, LiveDriverState::Finished) {
            LiveDriverState::AwaitingObservation { plan_identity } => {
                if result.plan_identity != plan_identity {
                    self.state = LiveDriverState::AwaitingObservation { plan_identity };
                    return Err(TraceDriverError::new(format!(
                        "native graph live driver observed {:?}, expected {:?}",
                        result.plan_identity, self.trace.trace_id
                    )));
                }
                ensure_completed_stage(&result)?;
                let outputs = self
                    .terminal_outputs
                    .iter()
                    .map(|channel| {
                        result.output_handles.get(channel).cloned().map_or_else(
                            || Err(TraceDriverError::new(format!("native graph live driver did not receive declared terminal output {channel:?}"))),
                            |handle| Ok((channel.clone(), handle)),
                        )
                    })
                    .collect::<Result<BTreeMap<_, _>, _>>()?;
                self.state = LiveDriverState::ReadyToComplete {
                    outputs,
                    receipts: Vec::new(),
                };
                Ok(())
            }
            LiveDriverState::AwaitingDynamicObservation {
                plan_identity,
                mut cursor,
                executed_nodes,
            } => {
                if result.plan_identity != plan_identity {
                    self.state = LiveDriverState::AwaitingDynamicObservation {
                        plan_identity,
                        cursor,
                        executed_nodes,
                    };
                    return Err(TraceDriverError::new(
                        "native graph live driver observed another dynamic stage",
                    ));
                }
                ensure_completed_stage(&result)?;
                cursor.channels.extend(result.channels);
                cursor.completed.extend(executed_nodes.iter().cloned());
                apply_model_decisions(&mut cursor, &executed_nodes, &self.control_flow)?;
                self.open_selected_branch_leases(&cursor).await?;
                apply_loop_decisions(&mut cursor, &executed_nodes, &self.control_flow)?;
                advance_loop_backedges(&mut cursor, &self.control_flow);
                self.complete_selected_branch_workspaces(&mut cursor)
                    .await?;
                apply_selected_joins(&mut cursor, &self.control_flow)?;
                cursor.next_stage = cursor.next_stage.checked_add(1).ok_or_else(|| {
                    TraceDriverError::new("native graph dynamic stage counter overflow")
                })?;
                if dynamic_ready_nodes(&cursor, &self.control_flow)?.is_empty() {
                    self.state = LiveDriverState::ReadyToComplete {
                        outputs: BTreeMap::new(),
                        receipts: cursor.receipts,
                    };
                } else {
                    self.state = LiveDriverState::DynamicReady(cursor);
                }
                Ok(())
            }
            state => {
                self.state = state;
                Err(TraceDriverError::new(
                    "native graph live driver received an unexpected stage observation",
                ))
            }
        }
    }

    async fn abort_open(&mut self) -> Result<(), TraceDriverError> {
        self.state = LiveDriverState::Finished;
        self.close_opening_session().await
    }

    async fn close(&mut self) -> Result<(), TraceDriverError> {
        self.state = LiveDriverState::Finished;
        let opening = self.close_opening_session().await;
        let session = self.close_session().await;
        match (opening, session) {
            (Err(error), _) => Err(error),
            (Ok(()), result) => result,
        }
    }

    async fn run(
        &mut self,
        _program: &GraphTraceProgram,
        _context: &TraceDriverContext<'_>,
    ) -> Result<TraceTerminalSupplement, TraceDriverError> {
        Err(TraceDriverError::new(
            "native graph live driver requires bounded staged graph execution",
        ))
    }
}

impl NativeGraphLiveTraceProgramDriver {
    async fn open_session(
        &mut self,
        program: &GraphTraceProgram,
        context: &TraceDriverContext<'_>,
    ) -> Result<(), TraceDriverError> {
        if self.opening_session.is_some() || self.session.is_some() {
            return Err(TraceDriverError::new(
                "native graph live session is already opening or open",
            ));
        }
        let execution = context.execution.as_ref().ok_or_else(|| {
            TraceDriverError::new("native graph live session requires worker execution context")
        })?;
        if execution.invocation.run_id() != self.trace.run_id
            || execution.invocation.trajectory_id() != self.trace.trajectory_id
            || execution.invocation.task_snapshot_digest()
                != Some(self.control_flow.source_snapshot_digest.as_str())
        {
            return Err(TraceDriverError::new(
                "native graph invocation context does not match immutable trace and task provenance",
            ));
        }
        let dispatcher = self
            .agent_loop
            .tool_dispatcher
            .create(&self.trace.trace_id)
            .map_err(|error| TraceDriverError::new(error.to_string()))?;
        let lifecycle_factory = self
            .agent_loop
            .lifecycle_lease
            .create(&self.trace.trace_id, dispatcher.clone())
            .map_err(|error| TraceDriverError::new(error.to_string()))?;
        let request = agent::AgentInvocationRequest {
            identity: agent::AgentInvocationIdentity {
                run_id: self.trace.run_id.clone(),
                trajectory_id: self.trace.trajectory_id.clone(),
                invocation_id: execution.invocation.root_invocation_id().to_owned(),
                parent_invocation_id: None,
            },
            environment: agent::AgentInvocationEnvironment::Isolated,
            workspace: agent::AgentInvocationWorkspace::Root,
        };
        let opening = lifecycle_factory
            .begin_open(&request, None)
            .map_err(|error| TraceDriverError::new(error.to_string()))?;
        let mut opening = LifecycleOpeningGuard::new(opening);
        let lease = opening
            .open()
            .await
            .map_err(|error| TraceDriverError::new(error.to_string()))?;
        let root_lease = LifecycleLeaseGuard::new(lease);
        let environment = program
            .environment
            .as_ref()
            .map(TraceEnvironmentSpec::resolve)
            .transpose()
            .map_err(|error| TraceDriverError::new(error.to_string()))?;
        self.opening_session = Some(NativeGraphLiveTraceOpeningSession {
            dispatcher,
            lifecycle_factory,
            root_lease,
            root_invocation_id: execution.invocation.root_invocation_id().to_owned(),
        });
        let opened = {
            let session = self.opening_session.as_ref().ok_or_else(|| {
                TraceDriverError::new("native graph live session lost its opening ownership")
            })?;
            session
                .dispatcher
                .open_trace(tools::TraceOpenContext {
                    trace: &self.trace,
                    environment: environment.as_ref(),
                    workspace: environment
                        .as_ref()
                        .map(|environment| &environment.workspace),
                    clock: execution.clock,
                    segments: execution.segments,
                    invocation: execution.invocation,
                })
                .await
        };
        if let Err(error) = opened {
            let _ = self.close_opening_session().await;
            return Err(TraceDriverError::new(error.to_string()));
        }
        let session = self.opening_session.take().ok_or_else(|| {
            TraceDriverError::new("native graph live session lost its opening ownership")
        })?;
        self.session = Some(session.into_session());
        Ok(())
    }

    fn ensure_stage_dispatcher(
        &self,
        cursor: &DynamicCursor,
        nodes: &BTreeSet<String>,
    ) -> Result<(), TraceDriverError> {
        if !cursor.requires_workspace_candidates {
            return Ok(());
        }
        let session = self.session.as_ref().ok_or_else(|| {
            TraceDriverError::new("dynamic NativeGraph stage lost its invocation session")
        })?;
        if selected_branch_key(cursor, nodes, &self.control_flow).is_some()
            && session
                .selected_branch_dispatcher(cursor, nodes, &self.control_flow)
                .is_none()
        {
            return Err(TraceDriverError::new(
                "dynamic NativeGraph selected branch has no isolated invocation lease",
            ));
        }
        Ok(())
    }

    async fn open_selected_branch_leases(
        &mut self,
        cursor: &DynamicCursor,
    ) -> Result<(), TraceDriverError> {
        if !cursor.requires_workspace_candidates {
            return Ok(());
        }
        let session = self.session.as_mut().ok_or_else(|| {
            TraceDriverError::new(
                "dynamic NativeGraph branch selection lost its invocation session",
            )
        })?;
        for (branch_id, candidate_id) in &cursor.branch_choices {
            let key = BranchWorkspaceKey {
                branch_id: branch_id.clone(),
                candidate_id: candidate_id.clone(),
            };
            if session.branch_leases.contains_key(&key) {
                continue;
            }
            let request = agent::AgentInvocationRequest {
                identity: agent::AgentInvocationIdentity {
                    run_id: self.trace.run_id.clone(),
                    trajectory_id: self.trace.trajectory_id.clone(),
                    invocation_id: format!(
                        "{}::branch::{branch_id}::{candidate_id}",
                        self.trace.trace_id
                    ),
                    parent_invocation_id: Some(session.root_invocation_id.clone()),
                },
                environment: agent::AgentInvocationEnvironment::Isolated,
                workspace: agent::AgentInvocationWorkspace::IsolatedBranch {
                    branch_id: branch_id.clone(),
                    candidate_id: candidate_id.clone(),
                    parent_invocation_id: session.root_invocation_id.clone(),
                    parent_snapshot_digest: self.control_flow.source_snapshot_digest.clone(),
                },
            };
            let opening = session
                .lifecycle_factory
                .begin_open(&request, Some(session.root_lease.lease()))
                .map_err(|error| TraceDriverError::new(error.to_string()))?;
            let mut opening = LifecycleOpeningGuard::new(opening);
            let lease = opening
                .open()
                .await
                .map_err(|error| TraceDriverError::new(error.to_string()))?;
            session
                .branch_leases
                .insert(key, LifecycleLeaseGuard::new(lease));
        }
        Ok(())
    }

    async fn complete_selected_branch_workspaces(
        &mut self,
        cursor: &mut DynamicCursor,
    ) -> Result<(), TraceDriverError> {
        if !cursor.requires_workspace_candidates {
            return Ok(());
        }
        let session = self.session.as_mut().ok_or_else(|| {
            TraceDriverError::new(
                "dynamic NativeGraph branch completion lost its invocation session",
            )
        })?;
        let completed = cursor.completed.clone();
        let keys = cursor
            .branch_choices
            .iter()
            .filter_map(|(branch_id, candidate_id)| {
                let branch = self
                    .control_flow
                    .branches
                    .iter()
                    .find(|branch| &branch.id == branch_id)?;
                let candidate = branch
                    .candidates
                    .iter()
                    .find(|candidate| &candidate.id == candidate_id)?;
                candidate
                    .nodes
                    .iter()
                    .all(|node| completed.contains(node))
                    .then(|| BranchWorkspaceKey {
                        branch_id: branch_id.clone(),
                        candidate_id: candidate_id.clone(),
                    })
            })
            .filter(|key| !cursor.workspace_candidates.contains_key(key))
            .collect::<Vec<_>>();
        for key in keys {
            let mut lease = session.branch_leases.remove(&key).ok_or_else(|| {
                TraceDriverError::new("selected NativeGraph branch has no lease to complete")
            })?;
            let candidate = lease
                .lease_mut()
                .complete_workspace()
                .await
                .map_err(|error| TraceDriverError::new(error.to_string()))?
                .ok_or_else(|| {
                    TraceDriverError::new(
                        "selected NativeGraph branch lease did not return a workspace candidate",
                    )
                })?;
            lease
                .close()
                .await
                .map_err(|error| TraceDriverError::new(error.to_string()))?;
            if candidate.id() != key.candidate_id.as_str() {
                return Err(TraceDriverError::new(
                    "selected NativeGraph branch lease returned another candidate identity",
                ));
            }
            cursor.workspace_candidates.insert(key, candidate);
        }
        Ok(())
    }

    async fn close_opening_session(&mut self) -> Result<(), TraceDriverError> {
        let Some(mut session) = self.opening_session.take() else {
            return Ok(());
        };
        let mut failure = None;
        if let Err(error) = session.dispatcher.close_trace(&self.trace).await {
            failure.get_or_insert_with(|| TraceDriverError::new(error.to_string()));
        }
        if let Err(error) = session.root_lease.close().await {
            failure.get_or_insert_with(|| TraceDriverError::new(error.to_string()));
        }
        failure.map_or(Ok(()), Err)
    }

    async fn close_session(&mut self) -> Result<(), TraceDriverError> {
        let Some(mut session) = self.session.take() else {
            return Ok(());
        };
        let mut failure = None;
        for (_, mut lease) in session.branch_leases {
            if let Err(error) = lease.close().await {
                failure.get_or_insert_with(|| TraceDriverError::new(error.to_string()));
            }
        }
        if let Err(error) = session.dispatcher.close_trace(&self.trace).await {
            failure.get_or_insert_with(|| TraceDriverError::new(error.to_string()));
        }
        if let Err(error) = session.root_lease.close().await {
            failure.get_or_insert_with(|| TraceDriverError::new(error.to_string()));
        }
        failure.map_or(Ok(()), Err)
    }

    fn terminal_supplement(
        &self,
        outputs: BTreeMap<String, Handle>,
        receipts: Vec<DynamicControlReceipt>,
    ) -> TraceTerminalSupplement {
        TraceTerminalSupplement::new(
            self.trace.run_id.clone(),
            self.trace.trajectory_id.clone(),
            self.trace.trace_id.clone(),
            self.worker.worker_id,
            NATIVE_GRAPH_LIVE_DRIVER_KIND,
        )
        .with_terminal_outputs(outputs)
        .with_dynamic_control_receipts(receipts)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LiveDriverData {
    control_flow: NativeGraphControlContract,
}

fn validate_live_driver_spec(
    spec: &TraceDriverSpec,
) -> Result<(LiveDriverData, TraceDriverProvenance), TraceDriverError> {
    let data = parse_live_driver_data(spec)?;
    validate_control_flow_contract(&data.control_flow)
        .map_err(|error| TraceDriverError::new(error.to_string()))?;
    let provenance = spec.source_provenance().cloned().ok_or_else(|| {
        TraceDriverError::new("native graph live driver is missing immutable source provenance")
    })?;
    if !provenance.matches_source_digest(&data.control_flow.source_snapshot_digest) {
        return Err(TraceDriverError::new(
            "native graph live driver source digest does not match immutable source provenance",
        ));
    }
    if !provenance.matches_static_projection_digest(&data.control_flow.static_projection_digest) {
        return Err(TraceDriverError::new(
            "native graph live driver static projection does not match immutable static projection provenance",
        ));
    }
    let control_digest = canonical_control_digest(&data.control_flow)
        .map_err(|error| TraceDriverError::new(error.to_string()))?;
    if !provenance.matches_control_digest(&control_digest) {
        return Err(TraceDriverError::new(
            "native graph live driver control contract does not match immutable lowering provenance",
        ));
    }
    Ok((data, provenance))
}

fn has_dynamic_control(control_flow: &NativeGraphControlContract) -> bool {
    !control_flow.branches.is_empty()
        || !control_flow.joins.is_empty()
        || !control_flow.loops.is_empty()
}

fn ensure_completed_stage(result: &TraceStageResult) -> Result<(), TraceDriverError> {
    if result.terminal_status == GraphReplyStatus::Completed {
        Ok(())
    } else {
        Err(TraceDriverError::new(
            "native graph live driver received a non-completed graph stage",
        ))
    }
}

fn dynamic_ready_nodes(
    cursor: &DynamicCursor,
    control_flow: &NativeGraphControlContract,
) -> Result<BTreeSet<String>, TraceDriverError> {
    let mut ready = BTreeSet::new();
    for node_id in cursor.source.graph.nodes.keys() {
        if cursor.completed.contains(node_id) || cursor.disabled.contains(node_id) {
            continue;
        }
        let predecessors = cursor
            .source
            .graph
            .edges
            .iter()
            .filter(|edge| edge.target == *node_id)
            .filter(|edge| !is_loop_backedge(edge, control_flow))
            .filter(|edge| !cursor.disabled.contains(&edge.source))
            .collect::<Vec<_>>();
        if predecessors.is_empty() {
            continue;
        }
        if predecessors
            .iter()
            .all(|edge| edge.source == START_NODE_ID || cursor.completed.contains(&edge.source))
        {
            ready.insert(node_id.clone());
        }
    }
    Ok(ready)
}

fn is_loop_backedge(edge: &StaticEdge, control_flow: &NativeGraphControlContract) -> bool {
    control_flow.loops.iter().any(|loop_spec| {
        edge.source == loop_spec.backedge.source && edge.target == loop_spec.backedge.target
    })
}

fn dynamic_stage_projection(
    cursor: &DynamicCursor,
    nodes: &BTreeSet<String>,
) -> Result<GraphTracePlan, TraceDriverError> {
    let graph = GraphRecord {
        version: cursor.source.graph.version.clone(),
        system: cursor.source.graph.system.clone(),
        state: cursor.source.graph.state.clone(),
        nodes: nodes
            .iter()
            .map(|node_id| {
                cursor
                    .source
                    .graph
                    .nodes
                    .get(node_id)
                    .cloned()
                    .map(|node| (node_id.clone(), node))
                    .ok_or_else(|| {
                        TraceDriverError::new(format!(
                            "dynamic stage references undeclared source node {node_id:?}"
                        ))
                    })
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?,
        edges: nodes
            .iter()
            .flat_map(|node_id| {
                [
                    StaticEdge {
                        source: START_NODE_ID.into(),
                        target: node_id.clone(),
                        delay_after_predecessor_us: None,
                        min_start_delay_us: None,
                        delay_after_predecessor_start_us: None,
                        delay_after_predecessor_first_token_us: None,
                    },
                    StaticEdge {
                        source: node_id.clone(),
                        target: "END".into(),
                        delay_after_predecessor_us: None,
                        min_start_delay_us: None,
                        delay_after_predecessor_start_us: None,
                        delay_after_predecessor_first_token_us: None,
                    },
                ]
            })
            .collect(),
    };
    Ok(GraphTracePlan {
        graph,
        trace: crate::graph::model::TraceRecord {
            id: cursor.source.trace.id.clone(),
            graph_ref: None,
            initial_state: cursor.channels.clone(),
        },
        arrival_offset_ns: cursor.source.arrival_offset_ns,
    })
}

fn validate_dynamic_stage_projection(
    plan: &GraphTracePlan,
    source: &GraphTracePlan,
    control_flow: &NativeGraphControlContract,
) -> Result<(), TraceDriverError> {
    validate_native_graph_trace_plan(plan)
        .map_err(|error| TraceDriverError::new(error.to_string()))?;
    let channels = plan.graph.state.keys().cloned().collect::<Vec<_>>();
    if channels != control_flow.stage_channel_ids
        || plan.graph.nodes.iter().any(|(node_id, node)| {
            source
                .graph
                .nodes
                .get(node_id)
                .map(|source| serde_json::to_value(source).ok())
                != Some(serde_json::to_value(node).ok())
        })
    {
        return Err(TraceDriverError::new(
            "native graph dynamic stage is not a declared immutable source projection",
        ));
    }
    Ok(())
}

fn apply_model_decisions(
    cursor: &mut DynamicCursor,
    executed_nodes: &BTreeSet<String>,
    control_flow: &NativeGraphControlContract,
) -> Result<(), TraceDriverError> {
    for branch in &control_flow.branches {
        if !executed_nodes.contains(&branch.selector_node) {
            continue;
        }
        let selection = selector_value(cursor.channels.get(&branch.selector_channel).ok_or_else(
            || {
                TraceDriverError::new(format!(
                    "dynamic branch {:?} did not receive model selector channel {:?}",
                    branch.id, branch.selector_channel
                ))
            },
        )?)?;
        let candidate = branch
            .candidates
            .iter()
            .find(|candidate| candidate.match_value == selection)
            .ok_or_else(|| {
                TraceDriverError::new(format!(
                    "dynamic branch {:?} rejected undeclared model result {:?}",
                    branch.id, selection
                ))
            })?;
        if cursor
            .branch_choices
            .insert(branch.id.clone(), candidate.id.clone())
            .is_some()
        {
            return Err(TraceDriverError::new(format!(
                "dynamic branch {:?} selected more than once without a declared loop receipt",
                branch.id
            )));
        }
        append_control_receipt(
            cursor,
            DynamicControlOperation::Branch,
            &branch.id,
            &candidate.id,
            candidate,
            None,
            0,
            0,
        )?;
        for alternative in &branch.candidates {
            if alternative.id != candidate.id {
                cursor.disabled.extend(alternative.nodes.iter().cloned());
            }
        }
    }
    Ok(())
}

fn apply_selected_joins(
    cursor: &mut DynamicCursor,
    control_flow: &NativeGraphControlContract,
) -> Result<(), TraceDriverError> {
    for join in &control_flow.joins {
        if cursor.channels.contains_key(&join.output_channel) {
            continue;
        }
        let Some(selected) = cursor.branch_choices.get(&join.selector).cloned() else {
            continue;
        };
        if !join.candidates.contains(&selected) {
            return Err(TraceDriverError::new(format!(
                "dynamic join {:?} does not admit selected candidate {:?}",
                join.id, selected
            )));
        }
        let branch = control_flow
            .branches
            .iter()
            .find(|branch| branch.id == join.selector)
            .ok_or_else(|| TraceDriverError::new("dynamic join lost its declared branch"))?;
        let candidate = branch
            .candidates
            .iter()
            .find(|candidate| candidate.id == selected)
            .ok_or_else(|| TraceDriverError::new("dynamic join lost its selected candidate"))?;
        if candidate.channels.len() != 1 {
            return Err(TraceDriverError::new(format!(
                "dynamic join {:?} requires one selected candidate channel",
                join.id
            )));
        }
        let channel = &candidate.channels[0];
        let Some(value) = cursor.channels.get(channel).cloned() else {
            continue;
        };
        let workspace_digest = if cursor.requires_workspace_candidates {
            let key = BranchWorkspaceKey {
                branch_id: branch.id.clone(),
                candidate_id: selected.clone(),
            };
            Some(
                cursor
                    .workspace_candidates
                    .get(&key)
                    .ok_or_else(|| {
                        TraceDriverError::new(
                            "selected NativeGraph branch did not complete an immutable workspace candidate before merge",
                        )
                    })?
                    .digest()
                    .clone(),
            )
        } else {
            None
        };
        cursor.channels.insert(join.output_channel.clone(), value);
        append_control_receipt(
            cursor,
            DynamicControlOperation::Merge,
            &join.id,
            &selected,
            candidate,
            workspace_digest.as_ref(),
            0,
            0,
        )?;
    }
    Ok(())
}

fn apply_loop_decisions(
    cursor: &mut DynamicCursor,
    executed_nodes: &BTreeSet<String>,
    control_flow: &NativeGraphControlContract,
) -> Result<(), TraceDriverError> {
    for loop_spec in &control_flow.loops {
        if !executed_nodes.contains(&loop_spec.selector_node) {
            continue;
        }
        let selection = selector_value(
            cursor
                .channels
                .get(&loop_spec.selector_channel)
                .ok_or_else(|| {
                    TraceDriverError::new(format!(
                        "dynamic loop {:?} did not receive model selector channel {:?}",
                        loop_spec.id, loop_spec.selector_channel
                    ))
                })?,
        )?;
        let progress = cursor.loops.entry(loop_spec.id.clone()).or_default();
        let operation = if loop_spec.retry_match.as_deref() == Some(selection.as_str()) {
            if progress.retries >= loop_spec.max_retries {
                return Err(TraceDriverError::new(format!(
                    "dynamic loop {:?} exceeded its {}-retry budget",
                    loop_spec.id, loop_spec.max_retries
                )));
            }
            progress.retries = progress
                .retries
                .checked_add(1)
                .ok_or_else(|| TraceDriverError::new("native graph retry counter overflow"))?;
            DynamicControlOperation::Retry
        } else if selection == loop_spec.continue_match {
            DynamicControlOperation::Loop
        } else {
            progress.awaits_backedge = false;
            continue;
        };
        if progress.iterations >= loop_spec.max_iterations.get() {
            return Err(TraceDriverError::new(format!(
                "dynamic loop {:?} exceeded its {}-iteration horizon",
                loop_spec.id, loop_spec.max_iterations
            )));
        }
        progress.iterations = progress
            .iterations
            .checked_add(1)
            .ok_or_else(|| TraceDriverError::new("native graph iteration counter overflow"))?;
        progress.awaits_backedge = true;
        let iterations = progress.iterations;
        let retries = progress.retries;
        for member in &loop_spec.members {
            if member != &loop_spec.selector_node {
                cursor.completed.remove(member);
            }
        }
        let selected_candidate = match operation {
            DynamicControlOperation::Loop => "continue",
            DynamicControlOperation::Retry => "retry",
            DynamicControlOperation::Branch | DynamicControlOperation::Merge => {
                return Err(TraceDriverError::new(
                    "dynamic loop selected an invalid control operation",
                ));
            }
        };
        append_control_receipt(
            cursor,
            operation,
            &loop_spec.id,
            selected_candidate,
            loop_spec,
            None,
            iterations,
            retries,
        )?;
    }
    Ok(())
}

fn advance_loop_backedges(cursor: &mut DynamicCursor, control_flow: &NativeGraphControlContract) {
    for loop_spec in &control_flow.loops {
        let Some(progress) = cursor.loops.get_mut(&loop_spec.id) else {
            continue;
        };
        if progress.awaits_backedge && cursor.completed.contains(&loop_spec.backedge.source) {
            cursor.completed.remove(&loop_spec.backedge.target);
            progress.awaits_backedge = false;
        }
    }
}

fn append_control_receipt<T: serde::Serialize>(
    cursor: &mut DynamicCursor,
    operation: DynamicControlOperation,
    control_id: &str,
    selected_candidate: &str,
    declaration: &T,
    selected_candidate_digest: Option<&ArtifactDigest>,
    loop_iterations: u32,
    retries: u32,
) -> Result<(), TraceDriverError> {
    let sequence = u32::try_from(cursor.receipts.len())
        .map_err(|_| TraceDriverError::new("native graph dynamic receipt sequence overflow"))?;
    let completed_stages = cursor
        .next_stage
        .checked_add(1)
        .ok_or_else(|| TraceDriverError::new("native graph dynamic stage counter overflow"))?;
    let declaration = serde_json::to_vec(declaration).map_err(|error| {
        TraceDriverError::new(format!("serializing declared dynamic candidate: {error}"))
    })?;
    let control_id =
        DeclaredDynamicControlName::parse(control_id.to_owned()).map_err(TraceDriverError::new)?;
    let selected_candidate = DeclaredDynamicControlName::parse(selected_candidate.to_owned())
        .map_err(TraceDriverError::new)?;
    cursor.receipts.push(DynamicControlReceipt::new(
        cursor.control_digest.clone(),
        sequence,
        operation,
        control_id,
        selected_candidate,
        selected_candidate_digest
            .cloned()
            .unwrap_or_else(|| ArtifactDigest::from_bytes(&declaration)),
        DynamicControlCounters {
            completed_stages,
            loop_iterations,
            retries,
        },
    ));
    Ok(())
}

fn selector_value(value: &Value) -> Result<String, TraceDriverError> {
    match value {
        Value::String(value) => Ok(value.clone()),
        Value::Array(values) => values
            .last()
            .map(selector_value)
            .transpose()?
            .ok_or_else(|| TraceDriverError::new("model selector channel is empty")),
        Value::Object(object) => object
            .get("content")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .ok_or_else(|| TraceDriverError::new("model selector channel has no text content")),
        _ => Err(TraceDriverError::new(
            "model selector channel is not a supported text result",
        )),
    }
}

fn parse_live_driver_data(spec: &TraceDriverSpec) -> Result<LiveDriverData, TraceDriverError> {
    if spec.kind != NATIVE_GRAPH_LIVE_DRIVER_KIND {
        return Err(TraceDriverError::new(format!(
            "native graph live factory cannot create trace driver {:?}",
            spec.kind
        )));
    }
    let fields = serde_json::Map::from_iter(spec.data.clone());
    serde_json::from_value(Value::Object(fields)).map_err(|error| {
        TraceDriverError::new(format!("invalid native graph live driver data: {error}"))
    })
}
