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

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::graph::model::GraphTraceProgram;
use crate::graph::supplement::TraceTerminalSupplement;

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
#[derive(Clone, Copy, Debug, Default)]
pub struct RecordedReplayTraceProgramDriverFactory;

/// Stock registry over the built-in static and recorded-replay driver families.
#[derive(Clone, Copy, Debug, Default)]
pub struct NativeTraceProgramDriverFactory;

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
            "recorded_replay" => RecordedReplayTraceProgramDriverFactory.capabilities(spec),
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
            "recorded_replay" => {
                RecordedReplayTraceProgramDriverFactory.create(worker, trace, spec)
            }
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
        }))
    }
}

/// Bounded recorded-replay terminal driver used until placement owns execution.
struct RecordedReplayTraceProgramDriver {
    worker: WorkerIdentity,
    trace: TraceIdentity,
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
        Ok(TraceTerminalSupplement::new(
            context.trace.run_id.clone(),
            context.trace.trajectory_id.clone(),
            context.trace.trace_id.clone(),
            self.worker.worker_id,
            "recorded_replay",
        ))
    }
}
