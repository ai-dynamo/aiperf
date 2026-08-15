// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Serializable trace-program driver specifications and worker-local contracts.
//!
//! These DTOs travel with a complete graph program at placement boundaries. They
//! deliberately describe a driver or environment recipe without carrying an
//! opened process, host path, or trait object.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::{self, Display};
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::graph::model::GraphTraceProgram;
use crate::graph::supplement::TraceTerminalSupplement;
use crate::graph::{agent, tools};

/// Stable replay task identity shared by input discovery and graph programs.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(deny_unknown_fields)]
pub struct ReplayTaskIdentity {
    /// Supported source adapter.
    pub adapter: String,
    /// Task-family identifier.
    pub family: String,
    /// Upstream task identifier.
    pub task_id: String,
    /// Optional descriptive workload role.
    #[serde(default)]
    pub primary_role: Option<String>,
}

/// One validated environment recipe selected before placement preflight.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TraceEnvironmentSpec {
    /// Registered environment-recipe identifier.
    pub kind: String,
    /// Recipe-specific, transportable configuration validated by that recipe.
    #[serde(default)]
    pub data: BTreeMap<String, Value>,
}

/// Source facts retained for recorded replay without credentials.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ReplayTraceMetadata {
    /// Zero-based ordinal of this task in its source manifest.
    pub manifest_ordinal: usize,
    /// Stable identity of the replay task.
    pub identity: ReplayTaskIdentity,
    /// BLAKE3 digest of the decompressed source recording.
    pub source_digest: String,
    /// Optional digest of normalization targets derived from the source.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub normalization_target_digest: Option<String>,
    /// Per-call target output lengths in source order.
    #[serde(default)]
    pub target_output_tokens: Vec<u64>,
    /// Expected model-call count from the source recording.
    pub expected_llm_node_count: u64,
    /// Expected completed tool-command count from the source recording.
    pub expected_tool_node_count: u64,
    /// Resolved request-profile identity used to lower recorded calls.
    pub request_profile_identity: String,
    /// Stable comparability labels retained in result provenance.
    #[serde(default)]
    pub comparability_annotations: BTreeMap<String, Value>,
}

/// Serializable selector for one registered trace-program driver.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TraceDriverSpec {
    /// Registered driver identifier.
    pub kind: String,
    /// Driver-specific, validated configuration.
    #[serde(default)]
    pub data: BTreeMap<String, Value>,
    /// Authored continuation mode, validated before any environment is provisioned.
    #[serde(default)]
    pub continuation: AgentContinuationSpec,
    /// Whether this trace asks the driver to delegate an invocation.
    #[serde(default)]
    pub delegation: TraceDelegationSpec,
}

impl TraceDriverSpec {
    /// Build the built-in static graph driver specification.
    pub fn static_graph() -> Self {
        Self {
            kind: "static_graph".into(),
            data: BTreeMap::new(),
            continuation: AgentContinuationSpec::Fresh,
            delegation: TraceDelegationSpec::None,
        }
    }

    /// Build the stock strict recorded-replay driver specification.
    pub fn recorded_replay() -> Self {
        Self {
            kind: "recorded_replay".into(),
            data: BTreeMap::new(),
            continuation: AgentContinuationSpec::Fresh,
            delegation: TraceDelegationSpec::None,
        }
    }

    /// Replace the authored continuation choice.
    pub fn with_continuation(mut self, continuation: AgentContinuationSpec) -> Self {
        self.continuation = continuation;
        self
    }

    /// Request delegated invocation support from this driver.
    pub fn with_delegation(mut self) -> Self {
        self.delegation = TraceDelegationSpec::Requested;
        self
    }

    /// Whether this is the built-in static graph driver with no extra settings.
    pub fn is_static_graph(&self) -> bool {
        self.kind == "static_graph"
            && self.data.is_empty()
            && self.continuation == AgentContinuationSpec::Fresh
            && self.delegation == TraceDelegationSpec::None
    }
}

/// Continuation input selected by an agent-oriented trace driver.
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum AgentContinuationSpec {
    /// Start a new invocation with no prior agent-owned state.
    #[default]
    Fresh,
    /// Normalize and load an authored trajectory artifact.
    Load {
        /// Artifact reference resolved by a registered trajectory codec.
        trajectory: String,
    },
    /// Resume from a driver-owned checkpoint artifact.
    Resume {
        /// Checkpoint reference resolved by the selected driver.
        checkpoint: String,
    },
}

/// Delegation request retained in a strict trace-driver specification.
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TraceDelegationSpec {
    /// The trace has no descendant invocation.
    #[default]
    None,
    /// The trace requires a driver capable of delegated invocation.
    Requested,
}

/// Stable identity of the worker which owns all trace-local agent state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorkerIdentity {
    /// Zero-based worker ordinal.
    pub worker_id: usize,
}

/// Stable identity of one trace invocation within a run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TraceIdentity {
    /// Run-wide correlation identity.
    pub run_id: String,
    /// Trace-local document identity.
    pub trajectory_id: String,
    /// Graph trace identity.
    pub trace_id: String,
}

/// Borrowed driver inputs owned by normal placement composition.
///
/// The context deliberately carries no HTTP client, timer, process, workspace,
/// or metric sink. Those resources remain owned by placement and later task
/// factories; this seam can only return bounded trace facts.
pub struct TraceDriverContext<'a> {
    /// Trace identity allocated by the placement owner.
    pub trace: &'a TraceIdentity,
}

/// Failure at the trace-driver composition boundary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TraceDriverError(String);

impl TraceDriverError {
    /// Build an explicit driver-boundary failure.
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl Display for TraceDriverError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for TraceDriverError {}

/// Declared abilities of a selected trace-program driver.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TraceDriverCapabilities {
    /// The driver may choose subsequent turns from live responses.
    pub has_live_turns: bool,
    /// The driver accepts a normalized authored trajectory.
    pub has_load: bool,
    /// The driver accepts a driver checkpoint.
    pub has_resume: bool,
    /// The driver supports explicit response reuse or branching.
    pub has_response_reuse: bool,
    /// The driver supports branch selection.
    pub has_branching: bool,
    /// The driver supports delegated child invocations.
    pub has_delegation: bool,
}

impl TraceDriverCapabilities {
    /// Refuse an authored mode the selected driver cannot honor.
    pub fn validate(
        self,
        spec: &TraceDriverSpec,
        driver_kind: &str,
    ) -> Result<(), TraceDriverError> {
        match &spec.continuation {
            AgentContinuationSpec::Fresh => {}
            AgentContinuationSpec::Load { .. } if self.has_load => {}
            AgentContinuationSpec::Load { .. } => {
                return Err(TraceDriverError::new(format!(
                    "trace driver {driver_kind:?} does not support loading an agent trajectory"
                )));
            }
            AgentContinuationSpec::Resume { .. } if self.has_resume => {}
            AgentContinuationSpec::Resume { .. } => {
                return Err(TraceDriverError::new(format!(
                    "trace driver {driver_kind:?} does not support resuming an agent checkpoint"
                )));
            }
        }
        if spec.delegation == TraceDelegationSpec::Requested && !self.has_delegation {
            return Err(TraceDriverError::new(format!(
                "trace driver {driver_kind:?} does not support delegated invocations"
            )));
        }
        Ok(())
    }
}

/// Trace-local program driver selected through [`TraceDriverSpec`].
#[async_trait(?Send)]
pub trait TraceProgramDriver {
    /// Run one program and return bounded terminal facts to placement.
    async fn run(
        &mut self,
        program: &GraphTraceProgram,
        context: &TraceDriverContext<'_>,
    ) -> Result<TraceTerminalSupplement, TraceDriverError>;
}

/// Frozen factory for worker-local trace-program drivers.
pub trait TraceProgramDriverFactory: Send + Sync {
    /// Validate the selected mode before environment or worker provisioning.
    fn capabilities(
        &self,
        spec: &TraceDriverSpec,
    ) -> Result<TraceDriverCapabilities, TraceDriverError>;

    /// Create one fresh, non-shared driver for an admitted trace.
    fn create(
        &self,
        worker: WorkerIdentity,
        trace: &TraceIdentity,
        spec: &TraceDriverSpec,
    ) -> Result<Box<dyn TraceProgramDriver>, TraceDriverError>;
}

/// Stock factory for strict recorded replay.
#[derive(Clone)]
pub struct RecordedReplayTraceProgramDriverFactory {
    agent_loop: Arc<RecordedReplayAgentLoopFactories>,
}

impl Default for RecordedReplayTraceProgramDriverFactory {
    fn default() -> Self {
        Self {
            agent_loop: Arc::new(RecordedReplayAgentLoopFactories::default()),
        }
    }
}

/// Frozen factories required to compose one recorded agent loop per trace.
pub struct RecordedReplayAgentLoopFactories {
    coordinator: Arc<dyn agent::AgentTurnCoordinatorFactory>,
    response_store: Arc<dyn agent::AgentResponseStoreFactory>,
    trajectory_sink: Arc<dyn agent::AgentTrajectorySinkFactory>,
    invocation_lease: Arc<dyn agent::InvocationLeaseFactoryFactory>,
    tool_dispatcher: Arc<dyn tools::ToolDispatcherFactory>,
    tool_decoder: Arc<dyn tools::AgentToolCallDecoderFactory>,
    observation_formatter: Arc<dyn tools::AgentObservationFormatterFactory>,
}

impl Default for RecordedReplayAgentLoopFactories {
    fn default() -> Self {
        Self {
            coordinator: Arc::new(agent::StaticAgentTurnCoordinatorFactory::default()),
            response_store: Arc::new(agent::InMemoryAgentResponseStoreFactory),
            trajectory_sink: Arc::new(agent::InMemoryAgentTrajectorySinkFactory),
            invocation_lease: Arc::new(agent::InMemoryInvocationLeaseFactoryFactory),
            tool_dispatcher: Arc::new(tools::InMemoryToolDispatcherFactory),
            tool_decoder: Arc::new(tools::InMemoryAgentToolCallDecoderFactory),
            observation_formatter: Arc::new(tools::InMemoryAgentObservationFormatterFactory),
        }
    }
}

impl RecordedReplayAgentLoopFactories {
    /// Freeze every worker-local agent-loop factory used by recorded replay.
    pub fn new(
        coordinator: Arc<dyn agent::AgentTurnCoordinatorFactory>,
        response_store: Arc<dyn agent::AgentResponseStoreFactory>,
        trajectory_sink: Arc<dyn agent::AgentTrajectorySinkFactory>,
        invocation_lease: Arc<dyn agent::InvocationLeaseFactoryFactory>,
        tool_dispatcher: Arc<dyn tools::ToolDispatcherFactory>,
        tool_decoder: Arc<dyn tools::AgentToolCallDecoderFactory>,
        observation_formatter: Arc<dyn tools::AgentObservationFormatterFactory>,
    ) -> Self {
        Self {
            coordinator,
            response_store,
            trajectory_sink,
            invocation_lease,
            tool_dispatcher,
            tool_decoder,
            observation_formatter,
        }
    }
}

impl RecordedReplayTraceProgramDriverFactory {
    /// Replace the frozen worker-local agent-loop composition before application setup.
    pub fn with_agent_loop_factories(
        mut self,
        agent_loop: RecordedReplayAgentLoopFactories,
    ) -> Self {
        self.agent_loop = Arc::new(agent_loop);
        self
    }
}

/// Stock registry over the built-in static and recorded-replay driver families.
#[derive(Clone)]
pub struct NativeTraceProgramDriverFactory {
    recorded_replay: RecordedReplayTraceProgramDriverFactory,
}

impl Default for NativeTraceProgramDriverFactory {
    fn default() -> Self {
        Self {
            recorded_replay: RecordedReplayTraceProgramDriverFactory::default(),
        }
    }
}

impl TraceProgramDriverFactory for NativeTraceProgramDriverFactory {
    fn capabilities(
        &self,
        spec: &TraceDriverSpec,
    ) -> Result<TraceDriverCapabilities, TraceDriverError> {
        match spec.kind.as_str() {
            "static_graph" => {
                let capabilities = TraceDriverCapabilities::default();
                capabilities.validate(spec, &spec.kind)?;
                Ok(capabilities)
            }
            "recorded_replay" => self.recorded_replay.capabilities(spec),
            _ => Err(TraceDriverError::new(format!(
                "no linked trace program driver for {:?}",
                spec.kind
            ))),
        }
    }

    fn create(
        &self,
        worker: WorkerIdentity,
        trace: &TraceIdentity,
        spec: &TraceDriverSpec,
    ) -> Result<Box<dyn TraceProgramDriver>, TraceDriverError> {
        match spec.kind.as_str() {
            "static_graph" => {
                self.capabilities(spec)?;
                Ok(Box::new(StaticGraphTraceProgramDriver { worker }))
            }
            "recorded_replay" => self.recorded_replay.create(worker, trace, spec),
            _ => Err(TraceDriverError::new(format!(
                "no linked trace program driver for {:?}",
                spec.kind
            ))),
        }
    }
}

/// Bounded terminal driver for the legacy static graph family.
struct StaticGraphTraceProgramDriver {
    worker: WorkerIdentity,
}

#[async_trait(?Send)]
impl TraceProgramDriver for StaticGraphTraceProgramDriver {
    async fn run(
        &mut self,
        program: &GraphTraceProgram,
        context: &TraceDriverContext<'_>,
    ) -> Result<TraceTerminalSupplement, TraceDriverError> {
        if !program.driver.is_static_graph() {
            return Err(TraceDriverError::new(
                "static graph driver received another program",
            ));
        }
        Ok(TraceTerminalSupplement::new(
            context.trace.run_id.clone(),
            context.trace.trajectory_id.clone(),
            context.trace.trace_id.clone(),
            self.worker.worker_id,
            "static_graph",
        ))
    }
}

impl TraceProgramDriverFactory for RecordedReplayTraceProgramDriverFactory {
    fn capabilities(
        &self,
        spec: &TraceDriverSpec,
    ) -> Result<TraceDriverCapabilities, TraceDriverError> {
        if spec.kind != "recorded_replay" {
            return Err(TraceDriverError::new(format!(
                "recorded replay factory cannot create trace driver {:?}",
                spec.kind
            )));
        }
        if !spec.data.is_empty() {
            return Err(TraceDriverError::new(
                "trace driver \"recorded_replay\" does not accept driver-specific data",
            ));
        }
        let capabilities = TraceDriverCapabilities {
            has_response_reuse: true,
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
        self.capabilities(spec)?;
        Ok(Box::new(RecordedReplayTraceProgramDriver {
            worker,
            trace: trace.clone(),
            agent_loop: self.agent_loop.clone(),
        }))
    }
}

/// Bounded recorded-replay terminal driver used until placement owns execution.
struct RecordedReplayTraceProgramDriver {
    worker: WorkerIdentity,
    trace: TraceIdentity,
    agent_loop: Arc<RecordedReplayAgentLoopFactories>,
}

#[async_trait(?Send)]
impl TraceProgramDriver for RecordedReplayTraceProgramDriver {
    async fn run(
        &mut self,
        program: &GraphTraceProgram,
        context: &TraceDriverContext<'_>,
    ) -> Result<TraceTerminalSupplement, TraceDriverError> {
        if program.driver.kind != "recorded_replay" {
            return Err(TraceDriverError::new(
                "recorded replay driver received another program",
            ));
        }
        if self.trace != *context.trace {
            return Err(TraceDriverError::new(
                "recorded replay driver cannot run a different trace invocation",
            ));
        }
        let invocation = agent::AgentInvocationIdentity {
            run_id: self.trace.run_id.clone(),
            trajectory_id: self.trace.trajectory_id.clone(),
            invocation_id: format!("{}::root", self.trace.trace_id),
            parent_invocation_id: None,
        };
        let coordinator_spec = agent::AgentTurnCoordinatorSpec {
            kind: "static".into(),
            data: BTreeMap::new(),
        };
        let mut coordinator = self
            .agent_loop
            .coordinator
            .create(&invocation, &coordinator_spec)
            .map_err(|error| TraceDriverError::new(error.to_string()))?;
        let mut response_store = self
            .agent_loop
            .response_store
            .create(&self.trace.trace_id)
            .map_err(|error| TraceDriverError::new(error.to_string()))?;
        let mut trajectory_sink = self
            .agent_loop
            .trajectory_sink
            .create(
                &self.trace.run_id,
                &self.trace.trajectory_id,
                &invocation.invocation_id,
            )
            .map_err(|error| TraceDriverError::new(error.to_string()))?;
        let invocation_lease = self
            .agent_loop
            .invocation_lease
            .create(&self.trace.trace_id)
            .map_err(|error| TraceDriverError::new(error.to_string()))?;
        let tool_dispatcher = self
            .agent_loop
            .tool_dispatcher
            .create(&self.trace.trace_id)
            .map_err(|error| TraceDriverError::new(error.to_string()))?;
        let tool_decoder = self
            .agent_loop
            .tool_decoder
            .create(&self.trace.trace_id)
            .map_err(|error| TraceDriverError::new(error.to_string()))?;
        let observation_formatter = self
            .agent_loop
            .observation_formatter
            .create(&self.trace.trace_id)
            .map_err(|error| TraceDriverError::new(error.to_string()))?;
        let trajectory = coordinator
            .run(
                response_store.as_mut(),
                trajectory_sink.as_mut(),
                invocation_lease.as_ref(),
                tool_dispatcher.as_ref(),
                tool_decoder.as_ref(),
                observation_formatter.as_ref(),
            )
            .await
            .map_err(|error| TraceDriverError::new(error.to_string()))?;
        if trajectory.run_id != self.trace.run_id
            || trajectory.trajectory_id != self.trace.trajectory_id
            || trajectory.invocation_id != invocation.invocation_id
        {
            return Err(TraceDriverError::new(
                "recorded replay trajectory sink returned mismatched identities",
            ));
        }
        Ok(TraceTerminalSupplement::new(
            context.trace.run_id.clone(),
            context.trace.trajectory_id.clone(),
            context.trace.trace_id.clone(),
            self.worker.worker_id,
            "recorded_replay",
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    struct CountingCoordinatorFactory {
        calls: Arc<AtomicUsize>,
        identities: Arc<Mutex<Vec<String>>>,
    }

    impl agent::AgentTurnCoordinatorFactory for CountingCoordinatorFactory {
        fn create(
            &self,
            invocation: &agent::AgentInvocationIdentity,
            spec: &agent::AgentTurnCoordinatorSpec,
        ) -> Result<Box<dyn agent::AgentTurnCoordinator>, agent::AgentLoopError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.identities
                .lock()
                .expect("test factory identity log is available")
                .push(invocation.invocation_id.clone());
            agent::StaticAgentTurnCoordinatorFactory::default().create(invocation, spec)
        }
    }

    struct CountingResponseStoreFactory {
        calls: Arc<AtomicUsize>,
        identities: Arc<Mutex<Vec<String>>>,
    }

    impl agent::AgentResponseStoreFactory for CountingResponseStoreFactory {
        fn create(
            &self,
            trace_id: &str,
        ) -> Result<Box<dyn agent::AgentResponseStore>, agent::AgentResponseStoreError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.identities
                .lock()
                .expect("test factory identity log is available")
                .push(trace_id.into());
            agent::InMemoryAgentResponseStoreFactory.create(trace_id)
        }
    }

    struct CountingTrajectorySinkFactory {
        calls: Arc<AtomicUsize>,
        identities: Arc<Mutex<Vec<String>>>,
    }

    impl agent::AgentTrajectorySinkFactory for CountingTrajectorySinkFactory {
        fn create(
            &self,
            run_id: &str,
            trajectory_id: &str,
            invocation_id: &str,
        ) -> Result<Box<dyn agent::AgentTrajectorySink>, agent::AgentTrajectoryError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.identities
                .lock()
                .expect("test factory identity log is available")
                .push(format!("{run_id}/{trajectory_id}/{invocation_id}"));
            agent::InMemoryAgentTrajectorySinkFactory.create(run_id, trajectory_id, invocation_id)
        }
    }

    struct CountingLeaseFactoryFactory {
        calls: Arc<AtomicUsize>,
        identities: Arc<Mutex<Vec<String>>>,
    }

    impl agent::InvocationLeaseFactoryFactory for CountingLeaseFactoryFactory {
        fn create(
            &self,
            trace_id: &str,
        ) -> Result<Box<dyn agent::InvocationLeaseFactory>, agent::AgentLoopError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.identities
                .lock()
                .expect("test factory identity log is available")
                .push(trace_id.into());
            agent::InMemoryInvocationLeaseFactoryFactory.create(trace_id)
        }
    }

    struct CountingToolDispatcherFactory {
        calls: Arc<AtomicUsize>,
        identities: Arc<Mutex<Vec<String>>>,
    }

    impl tools::ToolDispatcherFactory for CountingToolDispatcherFactory {
        fn create(
            &self,
            trace_id: &str,
        ) -> Result<Rc<dyn tools::ToolDispatcher>, tools::ToolDispatchError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.identities
                .lock()
                .expect("test factory identity log is available")
                .push(trace_id.into());
            tools::InMemoryToolDispatcherFactory.create(trace_id)
        }
    }

    struct CountingToolDecoderFactory {
        calls: Arc<AtomicUsize>,
        identities: Arc<Mutex<Vec<String>>>,
    }

    impl tools::AgentToolCallDecoderFactory for CountingToolDecoderFactory {
        fn create(
            &self,
            trace_id: &str,
        ) -> Result<Box<dyn tools::AgentToolCallDecoder>, tools::ToolDispatchError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.identities
                .lock()
                .expect("test factory identity log is available")
                .push(trace_id.into());
            tools::InMemoryAgentToolCallDecoderFactory.create(trace_id)
        }
    }

    struct CountingObservationFormatterFactory {
        calls: Arc<AtomicUsize>,
        identities: Arc<Mutex<Vec<String>>>,
    }

    impl tools::AgentObservationFormatterFactory for CountingObservationFormatterFactory {
        fn create(
            &self,
            trace_id: &str,
        ) -> Result<Box<dyn tools::AgentObservationFormatter>, tools::ToolDispatchError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.identities
                .lock()
                .expect("test factory identity log is available")
                .push(trace_id.into());
            tools::InMemoryAgentObservationFormatterFactory.create(trace_id)
        }
    }

    fn recorded_program(trace_id: &str) -> GraphTraceProgram {
        let mut program = GraphTraceProgram::static_graph(crate::graph::model::GraphTracePlan {
            graph: crate::graph::model::GraphRecord::default(),
            trace: crate::graph::model::TraceRecord {
                id: trace_id.into(),
                graph_ref: None,
                initial_state: BTreeMap::new(),
            },
            arrival_offset_ns: None,
        });
        program.driver = TraceDriverSpec::recorded_replay();
        program
    }

    #[tokio::test(flavor = "current_thread")]
    async fn recorded_replay_driver_composes_fresh_agent_loop_factories_per_trace() {
        let coordinator = Arc::new(AtomicUsize::new(0));
        let response_store = Arc::new(AtomicUsize::new(0));
        let trajectory_sink = Arc::new(AtomicUsize::new(0));
        let invocation_lease = Arc::new(AtomicUsize::new(0));
        let tool_dispatcher = Arc::new(AtomicUsize::new(0));
        let tool_decoder = Arc::new(AtomicUsize::new(0));
        let observation_formatter = Arc::new(AtomicUsize::new(0));
        let coordinator_identities = Arc::new(Mutex::new(Vec::new()));
        let response_store_identities = Arc::new(Mutex::new(Vec::new()));
        let trajectory_sink_identities = Arc::new(Mutex::new(Vec::new()));
        let invocation_lease_identities = Arc::new(Mutex::new(Vec::new()));
        let tool_dispatcher_identities = Arc::new(Mutex::new(Vec::new()));
        let tool_decoder_identities = Arc::new(Mutex::new(Vec::new()));
        let observation_formatter_identities = Arc::new(Mutex::new(Vec::new()));
        let factory = RecordedReplayTraceProgramDriverFactory::default().with_agent_loop_factories(
            RecordedReplayAgentLoopFactories::new(
                Arc::new(CountingCoordinatorFactory {
                    calls: coordinator.clone(),
                    identities: coordinator_identities.clone(),
                }),
                Arc::new(CountingResponseStoreFactory {
                    calls: response_store.clone(),
                    identities: response_store_identities.clone(),
                }),
                Arc::new(CountingTrajectorySinkFactory {
                    calls: trajectory_sink.clone(),
                    identities: trajectory_sink_identities.clone(),
                }),
                Arc::new(CountingLeaseFactoryFactory {
                    calls: invocation_lease.clone(),
                    identities: invocation_lease_identities.clone(),
                }),
                Arc::new(CountingToolDispatcherFactory {
                    calls: tool_dispatcher.clone(),
                    identities: tool_dispatcher_identities.clone(),
                }),
                Arc::new(CountingToolDecoderFactory {
                    calls: tool_decoder.clone(),
                    identities: tool_decoder_identities.clone(),
                }),
                Arc::new(CountingObservationFormatterFactory {
                    calls: observation_formatter.clone(),
                    identities: observation_formatter_identities.clone(),
                }),
            ),
        );
        let traces = [
            TraceIdentity {
                run_id: "run-a".into(),
                trajectory_id: "trajectory-a".into(),
                trace_id: "trace-a".into(),
            },
            TraceIdentity {
                run_id: "run-b".into(),
                trajectory_id: "trajectory-b".into(),
                trace_id: "trace-b".into(),
            },
        ];
        let mut supplements = Vec::new();
        for trace in &traces {
            let program = recorded_program(&trace.trace_id);
            let mut driver = factory
                .create(WorkerIdentity { worker_id: 0 }, trace, &program.driver)
                .expect("recorded replay driver is created");
            supplements.push(
                driver
                    .run(&program, &TraceDriverContext { trace })
                    .await
                    .expect("recorded replay runs its frozen agent loop"),
            );
        }

        for calls in [
            &coordinator,
            &response_store,
            &trajectory_sink,
            &invocation_lease,
            &tool_dispatcher,
            &tool_decoder,
            &observation_formatter,
        ] {
            assert_eq!(calls.load(Ordering::SeqCst), 2);
        }
        for identities in [
            &coordinator_identities,
            &response_store_identities,
            &trajectory_sink_identities,
            &invocation_lease_identities,
            &tool_dispatcher_identities,
            &tool_decoder_identities,
            &observation_formatter_identities,
        ] {
            let identities = identities
                .lock()
                .expect("test factory identity log is available");
            assert_eq!(identities.len(), 2);
            assert_ne!(identities[0], identities[1]);
        }
        assert_ne!(supplements[0].trace_id, supplements[1].trace_id);
        assert_ne!(supplements[0].trajectory_id, supplements[1].trajectory_id);
    }
}
