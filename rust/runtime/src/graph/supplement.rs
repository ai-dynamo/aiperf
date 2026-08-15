// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded trace-terminal facts returned by pluggable graph drivers.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

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
}

/// Stable replay-trace identity used to verify a cellular terminal fold.
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
            worker_id,
            driver_kind: driver_kind.into(),
            completed: true,
            trace_wall_ms: 0.0,
            calls: Vec::new(),
            tools: Vec::new(),
        }
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
    expected: &BTreeSet<ReplayTraceInstance>,
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
            {
                return Err(GraphSupplementError::NonFiniteDuration {
                    trace_id: trace.trace_id,
                });
            }
            let identity = ReplayTraceInstance::from(&trace);
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
    DuplicateTrace { identity: ReplayTraceInstance },
    /// A controller-expected trace never reached a terminal cell supplement.
    MissingTrace { identity: ReplayTraceInstance },
    /// A cell supplied a trace the resolved program did not own.
    UnknownTrace { identity: ReplayTraceInstance },
    /// The cell claimed a non-concrete or unsupported backend identity.
    UnknownBackend { backend: String },
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
