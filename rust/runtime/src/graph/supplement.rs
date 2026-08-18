// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded trace-terminal facts returned by pluggable graph drivers.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::dataset::Handle;
use crate::eval::ArtifactDigest;
use crate::graph::replay::{ReplayCallMeasurement, ToolCallMeasurement};

/// Versioned terminal facts that placement can fold without reading agent state.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TraceTerminalSupplement {
    /// Stable schema version for future cellular folding.
    pub schema_version: u32,
    /// Run-wide correlation identity.
    pub run_id: String,
    /// Trace-local trajectory document identity.
    pub trajectory_id: String,
    /// Graph trace identity.
    pub trace_id: String,
    /// Controller-authored placement identity, when this trace ran in a cell.
    ///
    /// Unlike [`Self::run_id`], this exists before dispatch and survives an
    /// aggregator fold without adopting the aggregator's cell id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub planned_identity: Option<PlannedReplayTraceInstance>,
    /// Owning worker ordinal.
    pub worker_id: usize,
    /// Registered driver that emitted this bounded result.
    pub driver_kind: String,
    /// Whether the full profiling graph executor completed successfully.
    pub completed: bool,
    /// Injected-clock profiling wall duration in milliseconds.
    pub trace_wall_ms: f64,
    /// Ordered completed LLM measurements; warmup calls are never retained here.
    pub calls: Vec<ReplayCallMeasurement>,
    /// Ordered attempted tool measurements; command/output bytes are never retained.
    pub tools: Vec<ToolCallMeasurement>,
    /// Named opaque outputs selected by the authored terminal declaration.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub terminal_outputs: BTreeMap<String, Handle>,
    /// Ordered opaque decisions made by a declared dynamic NativeGraph control.
    ///
    /// This records only contract-bound selection facts. Raw model replies, tool
    /// payloads, credentials, and workspace paths remain outside the terminal
    /// supplement.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub dynamic_control_receipts: Vec<DynamicControlReceipt>,
}

/// One immutable operation selected from a declared NativeGraph control contract.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DynamicControlOperation {
    /// A model result selected one declared branch candidate.
    Branch,
    /// A declared join reduced one selected branch candidate.
    Merge,
    /// A model result continued one declared bounded loop iteration.
    Loop,
    /// A declared retry budget admitted one retry.
    Retry,
}

/// Rust-owned bounded counters accompanying one dynamic control receipt.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct DynamicControlCounters {
    /// Number of completed acyclic stages when the receipt was appended.
    pub completed_stages: u32,
    /// Iterations consumed for the selected loop, or zero for non-loop receipts.
    pub loop_iterations: u32,
    /// Retries consumed for the selected loop, or zero for non-retry receipts.
    pub retries: u32,
}

/// Validated source-declared name retained in dynamic-control evidence.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub(crate) struct DeclaredDynamicControlName(String);

impl DeclaredDynamicControlName {
    pub(crate) fn parse(value: impl Into<String>) -> Result<Self, &'static str> {
        let value = value.into();
        if value.is_empty()
            || value.len() > 128
            || !value
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
        {
            return Err("invalid declared dynamic control name");
        }
        Ok(Self(value))
    }

    fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for DeclaredDynamicControlName {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Self::parse(String::deserialize(deserializer)?).map_err(serde::de::Error::custom)
    }
}

/// Compact append-only proof of one declared dynamic control decision.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct DynamicControlReceipt {
    /// Private lowering provenance digest for the immutable control contract.
    control_digest: ArtifactDigest,
    /// Monotonic receipt ordinal within this trace.
    sequence: u32,
    /// Declared operation selected at this point in the trace.
    operation: DynamicControlOperation,
    /// Declared branch, join, loop, or retry identifier.
    control_id: DeclaredDynamicControlName,
    /// Declared candidate or merge identity selected by the operation.
    selected_candidate: DeclaredDynamicControlName,
    /// Digest of the selected declared candidate, not its model or tool payload.
    selected_candidate_digest: ArtifactDigest,
    /// Rust-owned bounded accounting at the selection point.
    counters: DynamicControlCounters,
}

impl DynamicControlReceipt {
    pub(crate) fn new(
        control_digest: ArtifactDigest,
        sequence: u32,
        operation: DynamicControlOperation,
        control_id: DeclaredDynamicControlName,
        selected_candidate: DeclaredDynamicControlName,
        selected_candidate_digest: ArtifactDigest,
        counters: DynamicControlCounters,
    ) -> Self {
        Self {
            control_digest,
            sequence,
            operation,
            control_id,
            selected_candidate,
            selected_candidate_digest,
            counters,
        }
    }

    /// Borrows the trusted lowering provenance digest.
    pub fn control_digest(&self) -> &ArtifactDigest {
        &self.control_digest
    }

    /// Returns this receipt's append-only ordinal.
    pub const fn sequence(&self) -> u32 {
        self.sequence
    }

    /// Returns the declared operation selected by this receipt.
    pub const fn operation(&self) -> DynamicControlOperation {
        self.operation
    }

    /// Borrows the validated declared control identifier.
    pub fn control_id(&self) -> &str {
        self.control_id.as_str()
    }

    /// Borrows the validated declared selected candidate name.
    pub fn selected_candidate(&self) -> &str {
        self.selected_candidate.as_str()
    }

    /// Borrows the selected candidate artifact digest.
    pub fn selected_candidate_digest(&self) -> &ArtifactDigest {
        &self.selected_candidate_digest
    }

    /// Borrows Rust-owned counter facts recorded at selection.
    pub const fn counters(&self) -> &DynamicControlCounters {
        &self.counters
    }
}

/// Runtime replay-trace identity retained for measurements and artifacts.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(deny_unknown_fields)]
pub struct ReplayTraceInstance {
    /// Run-wide correlation identity.
    pub run_id: String,
    /// Trace-local trajectory document identity.
    pub trajectory_id: String,
    /// Graph trace identity.
    pub trace_id: String,
}

impl From<&TraceTerminalSupplement> for ReplayTraceInstance {
    fn from(trace: &TraceTerminalSupplement) -> Self {
        Self {
            run_id: trace.run_id.clone(),
            trajectory_id: trace.trajectory_id.clone(),
            trace_id: trace.trace_id.clone(),
        }
    }
}

/// Controller-authored identity of one assigned recorded-replay trace.
///
/// This deliberately excludes the runtime `run_id`, which is minted only when a
/// worker starts execution. `trace_id` is the resolved plan instance id (including
/// its deterministic `::instance-N` ordinal), so it is unique within a cell's
/// planned assignment before any request is dispatched.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(deny_unknown_fields)]
pub struct PlannedReplayTraceInstance {
    /// Cell assigned by the controller before START.
    pub cell_id: u32,
    /// Template graph trace identity chosen by the controller for this instance.
    #[serde(default)]
    pub template_trace_id: Option<String>,
    /// Planned trace-local trajectory identity.
    pub trajectory_id: String,
    /// Planned graph trace instance identity.
    pub trace_id: String,
}

impl PlannedReplayTraceInstance {
    /// Construct one controller-owned replay assignment identity.
    pub fn new(
        cell_id: u32,
        trajectory_id: impl Into<String>,
        trace_id: impl Into<String>,
    ) -> Self {
        Self {
            cell_id,
            template_trace_id: None,
            trajectory_id: trajectory_id.into(),
            trace_id: trace_id.into(),
        }
    }

    /// Attach the template graph trace identity chosen for this instance.
    #[must_use]
    pub fn with_template_trace_id(mut self, template_trace_id: impl Into<String>) -> Self {
        self.template_trace_id = Some(template_trace_id.into());
        self
    }

    fn from_terminal(cell_id: u32, trace: &TraceTerminalSupplement) -> Self {
        Self::new(cell_id, &trace.trajectory_id, &trace.trace_id)
    }
}

/// Validated concrete backend identity transported with a cellular replay fold.
///
/// The transparent wire representation deliberately remains a string: an older or
/// untrusted cell can deserialize it, but the controller validates it before any
/// final artifact is written.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(transparent)]
pub struct ReplayBackendIdentity(String);

impl ReplayBackendIdentity {
    /// Build an identity from a wire label. Validation remains controller-owned.
    pub fn from_wire(identity: impl Into<String>) -> Self {
        Self(identity.into())
    }

    fn validate(&self) -> Result<(), GraphSupplementError> {
        crate::graph::tools::ToolBackendIdentity::parse(&self.0)
            .map(|_| ())
            .map_err(|_| GraphSupplementError::UnknownBackend {
                backend: self.0.clone(),
            })
    }
}

/// Versioned terminal replay facts shipped by exactly one cellular partition.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct GraphCellSupplement {
    /// Stable schema version for cross-process folding.
    pub schema_version: u32,
    /// Cell owning this terminal partition.
    pub cell_id: u32,
    /// Terminal trace facts in the cell's worker/completion order.
    pub traces: Vec<TraceTerminalSupplement>,
    /// Trace instances assigned to this cell before dispatch begins.
    #[serde(default)]
    pub expected_traces: BTreeSet<PlannedReplayTraceInstance>,
    /// Concrete tool backends selected by this cell.
    pub backend_identities: BTreeSet<ReplayBackendIdentity>,
}

impl GraphCellSupplement {
    /// Build a stock version-one cell supplement.
    pub fn new(
        cell_id: u32,
        traces: Vec<TraceTerminalSupplement>,
        backend_identities: BTreeSet<ReplayBackendIdentity>,
    ) -> Self {
        Self {
            schema_version: 1,
            cell_id,
            expected_traces: BTreeSet::new(),
            traces,
            backend_identities,
        }
    }

    /// Convert the worker-local phase fold into a cell-owned wire supplement.
    pub fn from_phase(cell_id: u32, phase: GraphPhaseSupplement) -> Self {
        let backend_identities = phase
            .traces
            .iter()
            .flat_map(|trace| trace.tools.iter())
            .map(|tool| ReplayBackendIdentity::from_wire(tool.backend.clone()))
            .collect();
        Self::new(cell_id, phase.traces, backend_identities)
    }

    /// Replace terminal-derived expectations with the controller-authored assignment.
    pub fn with_expected_traces(
        mut self,
        expected_traces: BTreeSet<PlannedReplayTraceInstance>,
    ) -> Self {
        self.expected_traces = expected_traces;
        self
    }
}

/// Pre-start collection barrier for cellular replay capability checks.
///
/// Cells report the selected driver's resolved sandbox/image result after they receive
/// their envelope and before the controller releases START. The barrier itself is
/// transport-neutral so HTTP and Velo registrations share the same refusal semantics.
pub struct GraphCellPreflightBarrier {
    cell_count: u32,
    reports: std::sync::Mutex<BTreeMap<u32, Result<(), String>>>,
    changed: tokio::sync::Notify,
}

impl GraphCellPreflightBarrier {
    /// Require one successful report from every cell before warmup may begin.
    pub fn new(cell_count: u32) -> Self {
        Self {
            cell_count,
            reports: std::sync::Mutex::new(BTreeMap::new()),
            changed: tokio::sync::Notify::new(),
        }
    }

    /// Report a completed local capability preflight.
    pub fn report(&self, cell_id: u32, result: Result<(), String>) {
        if let Ok(mut reports) = self.reports.lock() {
            reports.insert(cell_id, result);
        }
        self.changed.notify_waiters();
    }

    /// Wait until every cell has passed its preflight, failing at the first failure.
    pub async fn await_all(&self) -> Result<(), GraphSupplementError> {
        loop {
            let notified = self.changed.notified();
            {
                let reports = self.reports.lock().map_err(|_| {
                    GraphSupplementError::Message("cell preflight barrier lock was poisoned".into())
                })?;
                if let Some((&cell_id, Err(error))) =
                    reports.iter().find(|(_, result)| result.is_err())
                {
                    return Err(GraphSupplementError::FailedPreflight {
                        cell_id,
                        error: error.clone(),
                    });
                }
                if reports.len() == self.cell_count as usize {
                    return Ok(());
                }
            }
            notified.await;
        }
    }
}

impl TraceTerminalSupplement {
    /// Construct the stock version-one terminal supplement.
    pub fn new(
        run_id: String,
        trajectory_id: String,
        trace_id: String,
        worker_id: usize,
        driver_kind: impl Into<String>,
    ) -> Self {
        Self {
            schema_version: 1,
            run_id,
            trajectory_id,
            trace_id,
            planned_identity: None,
            worker_id,
            driver_kind: driver_kind.into(),
            completed: true,
            trace_wall_ms: 0.0,
            calls: Vec::new(),
            tools: Vec::new(),
            terminal_outputs: BTreeMap::new(),
            dynamic_control_receipts: Vec::new(),
        }
    }

    /// Attach the controller-authored assignment without replacing runtime facts.
    pub fn with_planned_identity(mut self, identity: PlannedReplayTraceInstance) -> Self {
        self.planned_identity = Some(identity);
        self
    }

    /// Attach the declared terminal outputs without inferring an ordering.
    pub fn with_terminal_outputs(mut self, outputs: BTreeMap<String, Handle>) -> Self {
        self.terminal_outputs = outputs;
        self
    }

    /// Attach the compact append-only dynamic-control evidence emitted by a driver.
    pub(crate) fn with_dynamic_control_receipts(
        mut self,
        receipts: Vec<DynamicControlReceipt>,
    ) -> Self {
        self.dynamic_control_receipts = receipts;
        self
    }

    /// Attach bounded profiling facts after the graph executor reaches success.
    pub fn with_profiling_measurements(
        mut self,
        trace_wall_ms: f64,
        calls: Vec<ReplayCallMeasurement>,
        tools: Vec<ToolCallMeasurement>,
    ) -> Self {
        self.trace_wall_ms = trace_wall_ms;
        self.calls = calls;
        self.tools = tools;
        self
    }
}

/// Controller-owned fold of one graph phase's terminal replay facts.
///
/// Workers append only their bounded terminal supplement through the phase event
/// stream. The controller owns ordering and every final artifact write.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct GraphPhaseSupplement {
    /// Stable schema version for phase/cell compatibility checks.
    pub schema_version: u32,
    /// Terminal facts in the phase completion order.
    pub traces: Vec<TraceTerminalSupplement>,
}

impl GraphPhaseSupplement {
    /// Construct an empty stock version-one phase fold.
    pub fn new() -> Self {
        Self {
            schema_version: 1,
            traces: Vec::new(),
        }
    }

    /// Append one compatible worker terminal fact in controller-selected order.
    pub fn push(
        &mut self,
        supplement: TraceTerminalSupplement,
    ) -> Result<(), GraphSupplementError> {
        if supplement.schema_version != self.schema_version {
            return Err(GraphSupplementError::new(format!(
                "cannot fold trace supplement schema {} into phase schema {}",
                supplement.schema_version, self.schema_version
            )));
        }
        self.traces.push(supplement);
        Ok(())
    }

    /// Fold terminal facts from another completed partition, retaining their
    /// controller-selected terminal facts for the final deterministic artifact sort.
    pub fn extend(&mut self, other: GraphPhaseSupplement) -> Result<(), GraphSupplementError> {
        if other.schema_version != self.schema_version {
            return Err(GraphSupplementError::new(format!(
                "cannot fold phase supplement schema {} into phase schema {}",
                other.schema_version, self.schema_version
            )));
        }
        self.traces.extend(other.traces);
        Ok(())
    }
}

/// Fold graph replay supplements from terminal partitions. Empty cells contribute
/// no facts; all non-empty inputs must use the stock compatible schema.
pub fn merge_graph_phase_supplements<I>(
    supplements: I,
) -> Result<GraphPhaseSupplement, GraphSupplementError>
where
    I: IntoIterator<Item = GraphPhaseSupplement>,
{
    let mut merged = GraphPhaseSupplement::new();
    for supplement in supplements {
        merged.extend(supplement)?;
    }
    Ok(merged)
}

/// Validate and merge exact expected replay trace instances from terminal cells.
///
/// The output keeps stable cell-id, worker-id, then cell-local completion order. A
/// controller must supply the expected instance set from the resolved trace programs;
/// it is never reconstructed from controller-local replay paths.
pub fn merge_graph_cell_supplements<I>(
    expected: &BTreeSet<PlannedReplayTraceInstance>,
    supplements: I,
) -> Result<GraphPhaseSupplement, GraphSupplementError>
where
    I: IntoIterator<Item = GraphCellSupplement>,
{
    let mut cells = supplements.into_iter().collect::<Vec<_>>();
    cells.sort_by_key(|cell| cell.cell_id);
    let mut seen_cells = BTreeSet::new();
    let mut seen = BTreeSet::new();
    let mut merged = GraphPhaseSupplement::new();
    for cell in cells {
        if cell.schema_version != merged.schema_version {
            return Err(GraphSupplementError::SchemaMismatch {
                actual: cell.schema_version,
                expected: merged.schema_version,
            });
        }
        if !seen_cells.insert(cell.cell_id) {
            return Err(GraphSupplementError::DuplicateCell {
                cell_id: cell.cell_id,
            });
        }
        for backend in &cell.backend_identities {
            backend.validate()?;
        }
        let observed_backends = cell
            .traces
            .iter()
            .flat_map(|trace| trace.tools.iter())
            .map(|tool| ReplayBackendIdentity::from_wire(tool.backend.clone()))
            .collect::<BTreeSet<_>>();
        for backend in &observed_backends {
            backend.validate()?;
        }
        if cell.backend_identities != observed_backends {
            return Err(GraphSupplementError::BackendAllowlistMismatch {
                cell_id: cell.cell_id,
            });
        }
        let mut traces = cell.traces.into_iter().enumerate().collect::<Vec<_>>();
        traces.sort_by_key(|(completion, trace)| (trace.worker_id, *completion));
        for (_, trace) in traces {
            if trace.schema_version != merged.schema_version {
                return Err(GraphSupplementError::SchemaMismatch {
                    actual: trace.schema_version,
                    expected: merged.schema_version,
                });
            }
            if !trace.trace_wall_ms.is_finite()
                || trace.tools.iter().any(|tool| !tool.duration_s.is_finite())
                || trace.calls.iter().any(|call| {
                    !call.raw_end_to_end_ms.is_finite()
                        || !call.raw_inference_ms.is_finite()
                        || !call.raw_generation_ms.is_finite()
                        || call.ttft_ms.is_some_and(|value| !value.is_finite())
                        || call.stream_total_ms.is_some_and(|value| !value.is_finite())
                })
            {
                return Err(GraphSupplementError::NonFiniteDuration {
                    trace_id: trace.trace_id,
                });
            }
            let identity = trace
                .planned_identity
                .clone()
                .unwrap_or_else(|| PlannedReplayTraceInstance::from_terminal(cell.cell_id, &trace));
            if !expected.contains(&identity) {
                return Err(GraphSupplementError::UnknownTrace { identity });
            }
            if !seen.insert(identity.clone()) {
                return Err(GraphSupplementError::DuplicateTrace { identity });
            }
            merged.push(trace)?;
        }
    }
    if let Some(identity) = expected.iter().find(|identity| !seen.contains(*identity)) {
        return Err(GraphSupplementError::MissingTrace {
            identity: identity.clone(),
        });
    }
    Ok(merged)
}

impl Default for GraphPhaseSupplement {
    fn default() -> Self {
        Self::new()
    }
}

/// Failure while validating or folding terminal graph supplements.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GraphSupplementError {
    /// Generic supplement boundary failure.
    Message(String),
    /// A cell or trace used a version the controller cannot safely fold.
    SchemaMismatch { actual: u32, expected: u32 },
    /// Two terminal supplements claimed the same replay trace instance.
    DuplicateTrace {
        identity: PlannedReplayTraceInstance,
    },
    /// A controller-expected trace never reached a terminal cell supplement.
    MissingTrace {
        identity: PlannedReplayTraceInstance,
    },
    /// A cell supplied a trace the resolved program did not own.
    UnknownTrace {
        identity: PlannedReplayTraceInstance,
    },
    /// The cell claimed a non-concrete or unsupported backend identity.
    UnknownBackend { backend: String },
    /// A cell declared a backend that its trace-terminal facts did not use.
    BackendAllowlistMismatch { cell_id: u32 },
    /// A terminal duration cannot safely reach an artifact boundary.
    NonFiniteDuration { trace_id: String },
    /// Two terminal supplements claimed the same cell partition.
    DuplicateCell { cell_id: u32 },
    /// A cell failed capability preflight before the START barrier.
    FailedPreflight { cell_id: u32, error: String },
}

impl GraphSupplementError {
    /// Build an explicit supplement-fold failure.
    pub fn new(message: impl Into<String>) -> Self {
        Self::Message(message.into())
    }
}

impl std::fmt::Display for GraphSupplementError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Message(message) => formatter.write_str(message),
            Self::SchemaMismatch { actual, expected } => write!(
                formatter,
                "cell supplement schema {actual} is incompatible with schema {expected}"
            ),
            Self::DuplicateTrace { identity } => write!(
                formatter,
                "duplicate cellular replay trace {:?}",
                identity.trace_id
            ),
            Self::MissingTrace { identity } => write!(
                formatter,
                "missing cellular replay trace {:?}",
                identity.trace_id
            ),
            Self::UnknownTrace { identity } => write!(
                formatter,
                "unknown cellular replay trace {:?}",
                identity.trace_id
            ),
            Self::UnknownBackend { backend } => {
                write!(formatter, "unknown cellular replay backend {backend:?}")
            }
            Self::BackendAllowlistMismatch { cell_id } => write!(
                formatter,
                "cell {cell_id} replay backend allowlist does not match trace facts"
            ),
            Self::NonFiniteDuration { trace_id } => write!(
                formatter,
                "cellular replay trace {trace_id:?} has a non-finite duration"
            ),
            Self::DuplicateCell { cell_id } => {
                write!(formatter, "duplicate cellular replay cell {cell_id}")
            }
            Self::FailedPreflight { cell_id, error } => {
                write!(formatter, "cell {cell_id} replay preflight failed: {error}")
            }
        }
    }
}

impl std::error::Error for GraphSupplementError {}
